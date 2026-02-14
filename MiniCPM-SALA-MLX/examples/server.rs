//! OpenAI-compatible HTTP API server for MiniCPM-SALA.
//!
//! Usage:
//!   cargo run --release -p minicpm-sala-mlx --example server -- \
//!     --model ./models/MiniCPM-SALA-8bit --port 8080 --no-think
//!
//! API:
//!   POST /v1/chat/completions  — OpenAI-compatible chat
//!   GET  /v1/models            — List models
//!   GET  /health               — Health check

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use http_body_util::Full;
use hyper::body::{Bytes, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::eval;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};

use minicpm_sala_mlx::{
    create_layer_caches, format_chat_prompt_multi, get_model_args, is_stop_token, load_model,
    load_tokenizer, sample, strip_thinking,
};

#[derive(Parser)]
#[command(name = "minicpm-sala-server", about = "OpenAI-compatible API server")]
struct Args {
    /// Path to model directory
    #[arg(long)]
    model: String,

    /// Server port
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Default sampling temperature (overridable per request)
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Default max tokens (overridable per request)
    #[arg(long, default_value_t = 2048)]
    max_tokens: usize,

    /// Strip <think>...</think> from responses
    #[arg(long)]
    no_think: bool,
}

// ============================================================================
// Request/Response types (OpenAI-compatible)
// ============================================================================

#[derive(Deserialize)]
struct ChatCompletionRequest {
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default = "default_model_name")]
    model: String,
}

fn default_model_name() -> String {
    "minicpm-sala-9b".to_string()
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
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
// Inference
// ============================================================================

struct InferenceRequest {
    messages: Vec<(String, String)>, // (role, content)
    max_tokens: usize,
    temperature: f32,
    no_think: bool,
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
    mut model: minicpm_sala_mlx::Model,
    tokenizer: tokenizers::Tokenizer,
    mut rx: mpsc::Receiver<InferenceRequest>,
) {
    while let Some(req) = rx.blocking_recv() {
        let result = run_inference(&mut model, &tokenizer, &req);
        let _ = req.response_tx.send(result);
    }
}

fn run_inference(
    model: &mut minicpm_sala_mlx::Model,
    tokenizer: &tokenizers::Tokenizer,
    req: &InferenceRequest,
) -> Result<InferenceResult, String> {
    // Build ChatML prompt from messages
    let system_msg = req
        .messages
        .iter()
        .find(|(r, _)| r == "system")
        .map(|(_, c)| c.as_str())
        .unwrap_or("You are a helpful assistant.");

    let turns: Vec<(&str, &str)> = req
        .messages
        .iter()
        .filter(|(r, _)| r != "system")
        .map(|(r, c)| (r.as_str(), c.as_str()))
        .collect();

    let prompt = format_chat_prompt_multi(system_msg, &turns);

    let encoding = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| format!("Tokenizer error: {}", e))?;
    let prompt_tokens = encoding.get_ids();
    let prompt_len = prompt_tokens.len();

    let input = mlx_rs::Array::from_slice(
        &prompt_tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
        &[1, prompt_len as i32],
    );

    // Fresh caches per request
    let mut caches = create_layer_caches(&model.args);

    // Prefill
    let t0 = Instant::now();
    let logits = model
        .forward(&input, &mut caches)
        .map_err(|e| format!("Forward error: {:?}", e))?;
    let last_logits = logits.index((.., -1, ..));
    let mut token = sample(&last_logits, req.temperature)
        .map_err(|e| format!("Sample error: {:?}", e))?;
    eval([&token]).map_err(|e| format!("Eval error: {:?}", e))?;
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Decode
    let mut generated_ids: Vec<u32> = Vec::new();
    let decode_start = Instant::now();

    for _ in 0..req.max_tokens {
        let token_id = token.item::<u32>();
        if is_stop_token(token_id) {
            break;
        }
        generated_ids.push(token_id);

        let input = token
            .reshape(&[1, 1])
            .map_err(|e| format!("Reshape error: {:?}", e))?;
        let logits = model
            .forward(&input, &mut caches)
            .map_err(|e| format!("Forward error: {:?}", e))?;
        let last_logits = logits.index((.., -1, ..));
        token = sample(&last_logits, req.temperature)
            .map_err(|e| format!("Sample error: {:?}", e))?;
    }
    eval([&token]).map_err(|e| format!("Eval error: {:?}", e))?;

    let decode_time = decode_start.elapsed().as_secs_f64();
    let decode_tps = if generated_ids.is_empty() {
        0.0
    } else {
        generated_ids.len() as f64 / decode_time
    };

    let mut text = tokenizer
        .decode(&generated_ids, true)
        .map_err(|e| format!("Decode error: {}", e))?;

    if req.no_think {
        text = strip_thinking(&text).to_string();
    }

    Ok(InferenceResult {
        text,
        prompt_tokens: prompt_len,
        completion_tokens: generated_ids.len(),
        prefill_ms,
        decode_tps,
    })
}

// ============================================================================
// HTTP Handlers
// ============================================================================

struct ServerState {
    inference_tx: mpsc::Sender<InferenceRequest>,
    default_temperature: f32,
    default_max_tokens: usize,
    no_think: bool,
}

async fn handle_request(
    req: Request<Incoming>,
    state: Arc<ServerState>,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    eprintln!("[{}] {} {}", timestamp(), method, path);

    match (method, path.as_str()) {
        (Method::POST, "/v1/chat/completions") => {
            let body = collect_body(req).await;
            let chat_req: ChatCompletionRequest = match serde_json::from_str(&body) {
                Ok(r) => r,
                Err(e) => {
                    return Ok(json_response(
                        400,
                        json!({"error": {"message": format!("Invalid JSON: {}", e), "type": "invalid_request_error"}}),
                    ))
                }
            };

            if chat_req.messages.is_empty() {
                return Ok(json_response(
                    400,
                    json!({"error": {"message": "messages array is empty", "type": "invalid_request_error"}}),
                ));
            }

            let messages: Vec<(String, String)> = chat_req
                .messages
                .iter()
                .map(|m| (m.role.clone(), m.content.clone()))
                .collect();

            let max_tokens = chat_req.max_tokens.unwrap_or(state.default_max_tokens);
            let temperature = chat_req.temperature.unwrap_or(state.default_temperature);

            let (resp_tx, resp_rx) = oneshot::channel();
            if state
                .inference_tx
                .send(InferenceRequest {
                    messages,
                    max_tokens,
                    temperature,
                    no_think: state.no_think,
                    response_tx: resp_tx,
                })
                .await
                .is_err()
            {
                return Ok(json_response(
                    500,
                    json!({"error": {"message": "Inference worker unavailable", "type": "server_error"}}),
                ));
            }

            match resp_rx.await {
                Ok(Ok(result)) => {
                    eprintln!(
                        "[{}] Generated {} tokens ({:.0}ms prefill, {:.1} tok/s)",
                        timestamp(),
                        result.completion_tokens,
                        result.prefill_ms,
                        result.decode_tps
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
                    Ok(json_response(
                        200,
                        serde_json::to_value(response).unwrap(),
                    ))
                }
                Ok(Err(e)) => Ok(json_response(
                    500,
                    json!({"error": {"message": e, "type": "server_error"}}),
                )),
                Err(_) => Ok(json_response(
                    500,
                    json!({"error": {"message": "Inference channel closed", "type": "server_error"}}),
                )),
            }
        }

        (Method::GET, "/v1/models") => Ok(json_response(
            200,
            json!({
                "object": "list",
                "data": [{
                    "id": "minicpm-sala-9b",
                    "object": "model",
                    "owned_by": "moxin-org",
                }]
            }),
        )),

        (Method::GET, "/health") => Ok(json_response(200, json!({"status": "ok"}))),

        (Method::OPTIONS, _) => Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            .header(
                "Access-Control-Allow-Headers",
                "Content-Type, Authorization",
            )
            .body(Full::new(Bytes::new()))
            .unwrap()),

        _ => Ok(json_response(
            404,
            json!({"error": {"message": "Not found", "type": "invalid_request_error"}}),
        )),
    }
}

// ============================================================================
// Helpers
// ============================================================================

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

fn timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn uuid_simple() -> String {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:x}", t)
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
    let model_args = get_model_args(&args.model)?;
    eprintln!(
        "  {} layers ({} sparse, {} lightning)",
        model_args.num_hidden_layers,
        model_args
            .mixer_types
            .iter()
            .filter(|t| *t == "minicpm4")
            .count(),
        model_args
            .mixer_types
            .iter()
            .filter(|t| *t == "lightning-attn")
            .count(),
    );
    if let Some(q) = &model_args.quantization {
        eprintln!("  Quantized: {} bits, group_size={}", q.bits, q.group_size);
    }

    let model = load_model(&args.model)?;
    let tokenizer = load_tokenizer(&args.model)?;
    eprintln!("Model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // Inference channel
    let (inference_tx, inference_rx) = mpsc::channel::<InferenceRequest>(1);

    // Inference worker on dedicated thread
    std::thread::spawn(move || {
        inference_worker(model, tokenizer, inference_rx);
    });

    let state = Arc::new(ServerState {
        inference_tx,
        default_temperature: args.temperature,
        default_max_tokens: args.max_tokens,
        no_think: args.no_think,
    });

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    let listener = TcpListener::bind(addr).await?;

    println!();
    println!("  MiniCPM-SALA API Server");
    println!("  http://0.0.0.0:{}", args.port);
    println!();
    println!("  Endpoints:");
    println!("    POST /v1/chat/completions  - OpenAI-compatible chat");
    println!("    GET  /v1/models            - List models");
    println!("    GET  /health               - Health check");
    println!();
    println!("  Defaults: temperature={}, max_tokens={}, no_think={}", args.temperature, args.max_tokens, args.no_think);
    println!();

    loop {
        let (stream, addr) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let state = state.clone();

        tokio::task::spawn(async move {
            if let Err(e) = http1::Builder::new()
                .serve_connection(io, service_fn(move |req| handle_request(req, state.clone())))
                .await
            {
                eprintln!("[{}] Connection error from {}: {:?}", timestamp(), addr, e);
            }
        });
    }
}
