//! Moxin-7B VLM API server with OpenAI-compatible endpoint.
//!
//! Usage:
//!   cargo run --release -p moxin-vlm-mlx --example server -- \
//!     --model ./models/Moxin-7B-VLM-hf --quantize 8
//!
//! API:
//!   POST /v1/chat/completions
//!   {
//!     "messages": [
//!       { "role": "user", "content": [
//!           { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,..." } },
//!           { "type": "text", "text": "What is in this image?" }
//!       ]}
//!     ],
//!     "max_tokens": 256,
//!     "temperature": 0.0
//!   }
//!
//!   POST /v1/describe  (simple endpoint)
//!   {
//!     "image": "<base64 jpeg/png>",
//!     "prompt": "What is in this image?",
//!     "max_tokens": 256
//!   }

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use base64::Engine;
use clap::Parser;
use http_body_util::Full;
use hyper::body::{Bytes, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use image::imageops::FilterType;
use mlx_rs::Array;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};

use moxin_vlm_mlx::{
    load_model, load_tokenizer, normalize_dino, normalize_siglip, Generate, KVCache, MoxinVLM,
};

#[derive(Parser)]
#[command(name = "moxin-vlm-server")]
struct Args {
    /// Path to model directory
    #[arg(long)]
    model: String,

    /// Server port
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Quantize to N bits (0 = no quantization)
    #[arg(long, default_value = "0")]
    quantize: i32,
}

// ============================================================================
// Request/Response types
// ============================================================================

#[derive(Deserialize)]
struct SimpleRequest {
    image: String, // base64
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: f32,
}

fn default_prompt() -> String {
    "Describe this image.".to_string()
}
fn default_max_tokens() -> usize {
    256
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: f32,
    #[serde(default = "default_model_name")]
    model: String,
}

fn default_model_name() -> String {
    "moxin-7b-vlm".to_string()
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: ChatContent,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ChatContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Deserialize)]
struct ImageUrl {
    url: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    index: usize,
    message: ResponseMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct ResponseMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

// ============================================================================
// Image processing
// ============================================================================

fn decode_base64_image(b64: &str) -> Result<(Array, Array)> {
    // Strip data URI prefix if present
    let data = if let Some(pos) = b64.find(",") {
        &b64[pos + 1..]
    } else {
        b64
    };

    let bytes = base64::engine::general_purpose::STANDARD.decode(data)?;
    let img = image::load_from_memory(&bytes)?;
    let img = img.resize_exact(224, 224, FilterType::CatmullRom);
    let rgb = img.to_rgb8();

    let pixels: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| p.0.iter().map(|&v| v as f32 / 255.0))
        .collect();
    let tensor = Array::from_slice(&pixels, &[1, 224, 224, 3]);

    let dino_img = normalize_dino(&tensor)?;
    let siglip_img = normalize_siglip(&tensor)?;

    Ok((dino_img, siglip_img))
}

// ============================================================================
// Inference
// ============================================================================

struct InferenceRequest {
    dino_img: Array,
    siglip_img: Array,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    response_tx: oneshot::Sender<Result<InferenceResult, String>>,
}

struct InferenceResult {
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    prefill_ms: f64,
    decode_tps: f64,
}

fn inference_worker(
    mut vlm: MoxinVLM,
    tokenizer: tokenizers::Tokenizer,
    mut rx: mpsc::Receiver<InferenceRequest>,
) {
    while let Some(req) = rx.blocking_recv() {
        let result = run_inference(
            &mut vlm,
            &tokenizer,
            req.dino_img,
            req.siglip_img,
            &req.prompt,
            req.max_tokens,
            req.temperature,
        );
        let _ = req.response_tx.send(result);
    }
}

fn run_inference(
    vlm: &mut MoxinVLM,
    tokenizer: &tokenizers::Tokenizer,
    dino_img: Array,
    siglip_img: Array,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> Result<InferenceResult, String> {
    let prompt_text = format!("In: {}\nOut:", prompt);

    let encoding = tokenizer
        .encode(prompt_text.as_str(), true)
        .map_err(|e| format!("Tokenizer error: {}", e))?;
    let input_ids = Array::from_iter(
        encoding.get_ids().iter().map(|&id| id as i32),
        &[1, encoding.get_ids().len() as i32],
    );

    let prompt_tokens = encoding.get_ids().len() + 256; // text + visual tokens

    let mut cache: Vec<KVCache> = Vec::new();
    let generator = Generate::new(
        vlm,
        &mut cache,
        temperature,
        dino_img,
        siglip_img,
        input_ids,
    );

    let eos_token_id = 2u32;
    let mut generated = Vec::new();
    let mut prefill_time = None;
    let t0 = Instant::now();

    for token_result in generator.take(max_tokens) {
        let token = token_result.map_err(|e| format!("Generation error: {:?}", e))?;

        if prefill_time.is_none() {
            prefill_time = Some(t0.elapsed());
        }

        let token_id = token.item::<u32>();
        if token_id == eos_token_id {
            break;
        }
        generated.push(token_id);
    }

    let total_time = t0.elapsed();
    let prefill_ms = prefill_time
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let decode_ms = total_time.as_secs_f64() * 1000.0 - prefill_ms;
    let decode_tps = if generated.len() > 1 {
        (generated.len() - 1) as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };

    let text = tokenizer
        .decode(&generated, true)
        .map_err(|e| format!("Decode error: {}", e))?;

    Ok(InferenceResult {
        text,
        prompt_tokens,
        completion_tokens: generated.len(),
        prefill_ms,
        decode_tps,
    })
}

// ============================================================================
// HTTP Handlers
// ============================================================================

async fn handle_request(
    req: Request<Incoming>,
    state: Arc<ServerState>,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    eprintln!("[{}] {} {}", chrono_now(), method, path);

    match (method, path.as_str()) {
        // OpenAI-compatible chat completions
        (Method::POST, "/v1/chat/completions") => {
            let body = collect_body(req).await;
            let chat_req: ChatCompletionRequest = match serde_json::from_str(&body) {
                Ok(r) => r,
                Err(e) => return Ok(json_response(400, json!({"error": format!("Invalid JSON: {}", e)}))),
            };

            // Extract image and text from messages
            let (image_b64, text) = match extract_image_and_text(&chat_req.messages) {
                Ok(v) => v,
                Err(e) => return Ok(json_response(400, json!({"error": e}))),
            };

            let image_b64 = match image_b64 {
                Some(b) => b,
                None => return Ok(json_response(400, json!({"error": "No image provided in messages"}))),
            };

            // Decode image
            let (dino_img, siglip_img) = match decode_base64_image(&image_b64) {
                Ok(v) => v,
                Err(e) => return Ok(json_response(400, json!({"error": format!("Image decode failed: {}", e)}))),
            };

            // Run inference
            let (resp_tx, resp_rx) = oneshot::channel();
            if state
                .inference_tx
                .send(InferenceRequest {
                    dino_img,
                    siglip_img,
                    prompt: text,
                    max_tokens: chat_req.max_tokens,
                    temperature: chat_req.temperature,
                    response_tx: resp_tx,
                })
                .await
                .is_err()
            {
                return Ok(json_response(500, json!({"error": "Inference worker unavailable"})));
            }

            match resp_rx.await {
                Ok(Ok(result)) => {
                    eprintln!(
                        "[{}] Generated {} tokens ({:.0}ms prefill, {:.1} tok/s)",
                        chrono_now(), result.completion_tokens, result.prefill_ms, result.decode_tps
                    );

                    let response = ChatCompletionResponse {
                        id: format!("chatcmpl-{}", uuid_simple()),
                        object: "chat.completion".to_string(),
                        model: chat_req.model,
                        choices: vec![Choice {
                            index: 0,
                            message: ResponseMessage {
                                role: "assistant".to_string(),
                                content: result.text,
                            },
                            finish_reason: "stop".to_string(),
                        }],
                        usage: Usage {
                            prompt_tokens: result.prompt_tokens,
                            completion_tokens: result.completion_tokens,
                            total_tokens: result.prompt_tokens + result.completion_tokens,
                        },
                    };
                    Ok(json_response(200, serde_json::to_value(response).unwrap()))
                }
                Ok(Err(e)) => Ok(json_response(500, json!({"error": e}))),
                Err(_) => Ok(json_response(500, json!({"error": "Inference channel closed"}))),
            }
        }

        // Simple describe endpoint
        (Method::POST, "/v1/describe") => {
            let body = collect_body(req).await;
            let simple_req: SimpleRequest = match serde_json::from_str(&body) {
                Ok(r) => r,
                Err(e) => return Ok(json_response(400, json!({"error": format!("Invalid JSON: {}", e)}))),
            };

            let (dino_img, siglip_img) = match decode_base64_image(&simple_req.image) {
                Ok(v) => v,
                Err(e) => return Ok(json_response(400, json!({"error": format!("Image decode failed: {}", e)}))),
            };

            let (resp_tx, resp_rx) = oneshot::channel();
            if state
                .inference_tx
                .send(InferenceRequest {
                    dino_img,
                    siglip_img,
                    prompt: simple_req.prompt,
                    max_tokens: simple_req.max_tokens,
                    temperature: simple_req.temperature,
                    response_tx: resp_tx,
                })
                .await
                .is_err()
            {
                return Ok(json_response(500, json!({"error": "Inference worker unavailable"})));
            }

            match resp_rx.await {
                Ok(Ok(result)) => {
                    eprintln!(
                        "[{}] Generated {} tokens ({:.0}ms prefill, {:.1} tok/s)",
                        chrono_now(), result.completion_tokens, result.prefill_ms, result.decode_tps
                    );
                    Ok(json_response(
                        200,
                        json!({
                            "description": result.text,
                            "tokens": result.completion_tokens,
                            "prefill_ms": result.prefill_ms,
                            "tokens_per_second": result.decode_tps,
                        }),
                    ))
                }
                Ok(Err(e)) => Ok(json_response(500, json!({"error": e}))),
                Err(_) => Ok(json_response(500, json!({"error": "Inference channel closed"}))),
            }
        }

        // Health / status
        (Method::GET, "/v1/models") => Ok(json_response(
            200,
            json!({
                "object": "list",
                "data": [{
                    "id": "moxin-7b-vlm",
                    "object": "model",
                    "owned_by": "moxin-org",
                }]
            }),
        )),

        (Method::GET, "/health") => Ok(json_response(200, json!({"status": "ok"}))),

        // CORS preflight
        (Method::OPTIONS, _) => Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            .header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            .body(Full::new(Bytes::new()))
            .unwrap()),

        _ => Ok(json_response(404, json!({"error": "Not found"}))),
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn extract_image_and_text(messages: &[ChatMessage]) -> Result<(Option<String>, String), String> {
    let mut image_b64 = None;
    let mut text = String::new();

    for msg in messages {
        if msg.role != "user" {
            continue;
        }
        match &msg.content {
            ChatContent::Text(t) => text = t.clone(),
            ChatContent::Parts(parts) => {
                for part in parts {
                    match part {
                        ContentPart::Text { text: t } => text = t.clone(),
                        ContentPart::ImageUrl { image_url } => {
                            image_b64 = Some(image_url.url.clone());
                        }
                    }
                }
            }
        }
    }

    if text.is_empty() {
        text = "Describe this image.".to_string();
    }

    Ok((image_b64, text))
}

async fn collect_body(req: Request<Incoming>) -> String {
    let bytes = http_body_util::BodyExt::collect(req.into_body())
        .await
        .map(|b| b.to_bytes())
        .unwrap_or_default();
    String::from_utf8_lossy(&bytes).to_string()
}

fn json_response(status: u16, body: Value) -> Response<Full<Bytes>> {
    Response::builder()
        .status(StatusCode::from_u16(status).unwrap())
        .header("Content-Type", "application/json")
        .header("Access-Control-Allow-Origin", "*")
        .body(Full::new(Bytes::from(body.to_string())))
        .unwrap()
}

fn chrono_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("{}", now)
}

fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:x}", t)
}

struct ServerState {
    inference_tx: mpsc::Sender<InferenceRequest>,
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Load model
    eprintln!("Loading model from: {}", args.model);
    let t0 = Instant::now();
    let vlm = load_model(&args.model)?;
    let vlm = if args.quantize > 0 {
        eprintln!("Quantizing to {} bits...", args.quantize);
        vlm.quantize(64, args.quantize)?
    } else {
        vlm
    };
    let tokenizer = load_tokenizer(&args.model)?;
    eprintln!("Model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // Inference channel
    let (inference_tx, inference_rx) = mpsc::channel::<InferenceRequest>(1);

    // Inference worker on dedicated thread
    std::thread::spawn(move || {
        inference_worker(vlm, tokenizer, inference_rx);
    });

    let state = Arc::new(ServerState { inference_tx });

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    let listener = TcpListener::bind(addr).await?;

    println!();
    println!("  Moxin-7B VLM API Server");
    println!("  http://127.0.0.1:{}", args.port);
    println!();
    println!("  Endpoints:");
    println!("    POST /v1/chat/completions  - OpenAI-compatible (image+text)");
    println!("    POST /v1/describe          - Simple image description");
    println!("    GET  /v1/models            - List models");
    println!("    GET  /health               - Health check");
    println!();
    println!("  Example (simple):");
    println!("    IMAGE=$(base64 < photo.jpg)");
    println!(
        "    curl -s http://127.0.0.1:{}/v1/describe \\",
        args.port
    );
    println!("      -H 'Content-Type: application/json' \\");
    println!("      -d \"{{\\\"image\\\": \\\"$IMAGE\\\", \\\"prompt\\\": \\\"What is this?\\\"}}\"");
    println!();

    loop {
        let (stream, _addr) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let state = state.clone();

        tokio::task::spawn(async move {
            if let Err(e) = http1::Builder::new()
                .serve_connection(io, service_fn(move |req| handle_request(req, state.clone())))
                .await
            {
                eprintln!("Connection error: {:?}", e);
            }
        });
    }
}
