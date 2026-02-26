//! OpenAI-compatible HTTP API server for MiniCPM-SALA with model management.
//!
//! Usage:
//!   cargo run --release -p minicpm-sala-mlx --example server -- \
//!     --model ./models/MiniCPM-SALA-8bit --port 8080 --no-think
//!
//! API:
//!   POST   /v1/chat/completions   — OpenAI-compatible chat
//!   GET    /v1/models             — List models (with metadata)
//!   POST   /v1/models/download    — Download model from HuggingFace
//!   DELETE /v1/models/{id}        — Delete a downloaded model
//!   GET    /health                — Health check

use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
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
use tokio::sync::{mpsc, oneshot, RwLock};

use minicpm_sala_mlx::{
    create_layer_caches, format_chat_prompt_multi, get_model_args, is_stop_token, load_model,
    load_tokenizer, sample, strip_thinking, ModelArgs,
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

    /// Models directory (default: ~/.ominix/models)
    #[arg(long)]
    models_dir: Option<String>,
}

// ============================================================================
// Config types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OminixConfig {
    models_dir: String,
    #[serde(default)]
    models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelEntry {
    id: String,
    #[serde(default)]
    repo_id: String,
    path: String,
    #[serde(default)]
    quantization: Option<QuantInfo>,
    #[serde(default)]
    size_bytes: Option<u64>,
    #[serde(default)]
    downloaded_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantInfo {
    bits: i32,
    group_size: i32,
}

// ============================================================================
// Config helpers
// ============================================================================

fn config_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home).join(".ominix")
}

fn config_path() -> PathBuf {
    config_dir().join("config.json")
}

fn default_models_dir() -> String {
    config_dir().join("models").to_string_lossy().to_string()
}

fn load_config(models_dir_override: Option<&str>) -> OminixConfig {
    let path = config_path();
    let mut config = if path.exists() {
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(OminixConfig {
                models_dir: default_models_dir(),
                models: vec![],
            })
    } else {
        OminixConfig {
            models_dir: default_models_dir(),
            models: vec![],
        }
    };

    if let Some(dir) = models_dir_override {
        config.models_dir = dir.to_string();
    }

    config
}

fn save_config(config: &OminixConfig) -> std::io::Result<()> {
    let dir = config_dir();
    if !dir.exists() {
        std::fs::create_dir_all(&dir)?;
    }
    let json = serde_json::to_string_pretty(config).unwrap();
    std::fs::write(config_path(), json)
}

fn calculate_model_size(model_dir: &Path) -> u64 {
    std::fs::read_dir(model_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|x| x == "safetensors")
                .unwrap_or(false)
        })
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

fn scan_models_dir(config: &mut OminixConfig) {
    let models_dir = Path::new(&config.models_dir);
    if !models_dir.exists() {
        let _ = std::fs::create_dir_all(models_dir);
        return;
    }

    // Remove entries whose paths no longer exist
    config
        .models
        .retain(|entry| Path::new(&entry.path).join("config.json").exists());

    // Scan for new model subdirectories
    let known_paths: HashSet<String> = config.models.iter().map(|m| m.path.clone()).collect();

    let entries = match std::fs::read_dir(models_dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let sub_path = entry.path();
        if !sub_path.is_dir() {
            continue;
        }

        let config_json = sub_path.join("config.json");
        if !config_json.exists() {
            continue;
        }

        let path_str = sub_path.to_string_lossy().to_string();
        if known_paths.contains(&path_str) {
            continue;
        }

        // Parse config.json to extract quantization info
        let model_args: Option<ModelArgs> = std::fs::File::open(&config_json)
            .ok()
            .and_then(|f| serde_json::from_reader(f).ok());

        let quant = model_args
            .as_ref()
            .and_then(|a| a.quantization.as_ref())
            .map(|q| QuantInfo {
                bits: q.bits,
                group_size: q.group_size,
            });

        let size_bytes = calculate_model_size(&sub_path);

        let id = sub_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        config.models.push(ModelEntry {
            id,
            repo_id: String::new(),
            path: path_str,
            quantization: quant,
            size_bytes: Some(size_bytes),
            downloaded_at: None,
        });
    }
}

// ============================================================================
// Model download
// ============================================================================

fn download_model_blocking(repo_id: &str, models_dir: &Path) -> std::result::Result<ModelEntry, String> {
    // Resolve HF token
    let token = std::env::var("HF_TOKEN").ok().or_else(|| {
        let home = std::env::var("HOME").ok()?;
        let token_path = PathBuf::from(home).join(".cache/huggingface/token");
        std::fs::read_to_string(token_path)
            .ok()
            .map(|s| s.trim().to_string())
    });

    let api = if let Some(ref token) = token {
        hf_hub::api::sync::ApiBuilder::new()
            .with_token(Some(token.clone()))
            .build()
            .map_err(|e| format!("HF API error: {}", e))?
    } else {
        hf_hub::api::sync::ApiBuilder::new()
            .build()
            .map_err(|e| format!("HF API error: {}", e))?
    };

    let repo = api.model(repo_id.to_string());

    let model_id = repo_id.split('/').last().unwrap_or(repo_id);
    let dest_dir = models_dir.join(model_id);

    if dest_dir.exists() {
        return Err(format!(
            "Model directory already exists: {}",
            dest_dir.display()
        ));
    }
    std::fs::create_dir_all(&dest_dir)
        .map_err(|e| format!("Cannot create dir: {}", e))?;

    // Download essential files
    let files_to_get = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ];
    for filename in &files_to_get {
        eprintln!("[download] Fetching {}...", filename);
        let cached = repo
            .get(filename)
            .map_err(|e| format!("Failed to download {}: {}", filename, e))?;
        std::fs::copy(&cached, dest_dir.join(filename))
            .map_err(|e| format!("Failed to copy {}: {}", filename, e))?;
    }

    // Download weights (sharded or single)
    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        std::fs::copy(&index_path, dest_dir.join("model.safetensors.index.json"))
            .map_err(|e| format!("Copy index error: {}", e))?;

        let index_content = std::fs::read_to_string(&index_path)
            .map_err(|e| format!("Read index error: {}", e))?;
        let index: Value =
            serde_json::from_str(&index_content).map_err(|e| format!("Parse index error: {}", e))?;

        if let Some(weight_map) = index["weight_map"].as_object() {
            let weight_files: HashSet<&str> =
                weight_map.values().filter_map(|v| v.as_str()).collect();
            for weight_file in &weight_files {
                eprintln!("[download] Fetching {}...", weight_file);
                let cached = repo
                    .get(weight_file)
                    .map_err(|e| format!("Download {} failed: {}", weight_file, e))?;
                std::fs::copy(&cached, dest_dir.join(weight_file))
                    .map_err(|e| format!("Copy {} failed: {}", weight_file, e))?;
            }
        }
    } else {
        eprintln!("[download] Fetching model.safetensors...");
        let cached = repo
            .get("model.safetensors")
            .map_err(|e| format!("Download weights failed: {}", e))?;
        std::fs::copy(&cached, dest_dir.join("model.safetensors"))
            .map_err(|e| format!("Copy weights failed: {}", e))?;
    }

    // Extract metadata
    let model_args: Option<ModelArgs> = std::fs::File::open(dest_dir.join("config.json"))
        .ok()
        .and_then(|f| serde_json::from_reader(f).ok());

    let quant = model_args
        .as_ref()
        .and_then(|a| a.quantization.as_ref())
        .map(|q| QuantInfo {
            bits: q.bits,
            group_size: q.group_size,
        });

    let size_bytes = calculate_model_size(&dest_dir);

    Ok(ModelEntry {
        id: model_id.to_string(),
        repo_id: repo_id.to_string(),
        path: dest_dir.to_string_lossy().to_string(),
        quantization: quant,
        size_bytes: Some(size_bytes),
        downloaded_at: Some(format!("{}", timestamp())),
    })
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

#[derive(Deserialize)]
struct DownloadRequest {
    repo_id: String,
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
    response_tx: oneshot::Sender<std::result::Result<InferenceResult, String>>,
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
) -> std::result::Result<InferenceResult, String> {
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
    config: RwLock<OminixConfig>,
    loaded_model_path: String,
}

async fn handle_request(
    req: Request<Incoming>,
    state: Arc<ServerState>,
) -> std::result::Result<Response<Full<Bytes>>, hyper::Error> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    eprintln!("[{}] {} {}", timestamp(), method, path);

    match (method.clone(), path.as_str()) {
        (Method::POST, "/v1/chat/completions") => {
            let body = collect_body(req).await;
            handle_chat_completion(&body, &state).await
        }

        (Method::GET, "/v1/models") => handle_list_models(&state).await,

        (Method::POST, "/v1/models/download") => {
            let body = collect_body(req).await;
            handle_download_model(&body, &state).await
        }

        (Method::GET, "/health") => Ok(json_response(200, json!({"status": "ok"}))),

        (Method::OPTIONS, _) => Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            .header(
                "Access-Control-Allow-Headers",
                "Content-Type, Authorization",
            )
            .body(Full::new(Bytes::new()))
            .unwrap()),

        _ if method == Method::DELETE && path.starts_with("/v1/models/") => {
            let model_id = &path["/v1/models/".len()..];
            handle_delete_model(model_id, &state).await
        }

        _ => Ok(json_response(
            404,
            json!({"error": {"message": "Not found", "type": "invalid_request_error"}}),
        )),
    }
}

async fn handle_chat_completion(
    body: &str,
    state: &Arc<ServerState>,
) -> std::result::Result<Response<Full<Bytes>>, hyper::Error> {
    let chat_req: ChatCompletionRequest = match serde_json::from_str(body) {
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

async fn handle_list_models(
    state: &Arc<ServerState>,
) -> std::result::Result<Response<Full<Bytes>>, hyper::Error> {
    let mut config = state.config.write().await;
    scan_models_dir(&mut config);

    let loaded_canonical = std::fs::canonicalize(&state.loaded_model_path).ok();

    let data: Vec<Value> = config
        .models
        .iter()
        .map(|m| {
            let entry_canonical = std::fs::canonicalize(&m.path).ok();
            let loaded =
                loaded_canonical.is_some() && loaded_canonical == entry_canonical;

            let mut obj = json!({
                "id": m.id,
                "object": "model",
                "owned_by": if m.repo_id.is_empty() {
                    "local".to_string()
                } else {
                    m.repo_id.split('/').next().unwrap_or("local").to_string()
                },
                "path": m.path,
                "loaded": loaded,
            });
            if !m.repo_id.is_empty() {
                obj["repo_id"] = json!(m.repo_id);
            }
            if let Some(ref q) = m.quantization {
                obj["quantization"] = json!({"bits": q.bits, "group_size": q.group_size});
            }
            if let Some(s) = m.size_bytes {
                obj["size_bytes"] = json!(s);
            }
            if let Some(ref d) = m.downloaded_at {
                obj["downloaded_at"] = json!(d);
            }
            obj
        })
        .collect();

    Ok(json_response(
        200,
        json!({"object": "list", "data": data}),
    ))
}

async fn handle_download_model(
    body: &str,
    state: &Arc<ServerState>,
) -> std::result::Result<Response<Full<Bytes>>, hyper::Error> {
    let req: DownloadRequest = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => {
            return Ok(json_response(
                400,
                json!({"error": {"message": format!("Invalid JSON: {}", e), "type": "invalid_request_error"}}),
            ))
        }
    };

    if req.repo_id.is_empty() {
        return Ok(json_response(
            400,
            json!({"error": {"message": "repo_id is required", "type": "invalid_request_error"}}),
        ));
    }

    let model_id = req
        .repo_id
        .split('/')
        .last()
        .unwrap_or(&req.repo_id)
        .to_string();

    // Check if already exists
    {
        let config = state.config.read().await;
        if config.models.iter().any(|m| m.id == model_id) {
            return Ok(json_response(
                409,
                json!({"error": {"message": format!("Model '{}' already exists", model_id), "type": "conflict"}}),
            ));
        }
    }

    let models_dir = PathBuf::from(&state.config.read().await.models_dir);
    let repo_id = req.repo_id.clone();
    let state_clone = state.clone();

    tokio::task::spawn_blocking(move || {
        eprintln!("[download] Starting download: {}", repo_id);
        match download_model_blocking(&repo_id, &models_dir) {
            Ok(entry) => {
                let mut config = state_clone.config.blocking_write();
                config.models.push(entry);
                let _ = save_config(&config);
                eprintln!("[download] Complete: {}", repo_id);
            }
            Err(e) => {
                eprintln!("[download] Failed: {}: {}", repo_id, e);
                // Clean up partial directory
                let dest = models_dir.join(repo_id.split('/').last().unwrap_or(&repo_id));
                let _ = std::fs::remove_dir_all(&dest);
            }
        }
    });

    Ok(json_response(
        202,
        json!({
            "status": "downloading",
            "id": model_id,
            "repo_id": req.repo_id,
        }),
    ))
}

async fn handle_delete_model(
    model_id: &str,
    state: &Arc<ServerState>,
) -> std::result::Result<Response<Full<Bytes>>, hyper::Error> {
    let mut config = state.config.write().await;

    let idx = match config.models.iter().position(|m| m.id == model_id) {
        Some(i) => i,
        None => {
            return Ok(json_response(
                404,
                json!({"error": {"message": format!("Model not found: {}", model_id), "type": "not_found"}}),
            ))
        }
    };

    // Prevent deleting the currently loaded model
    let loaded_canonical = std::fs::canonicalize(&state.loaded_model_path).ok();
    let entry_canonical = std::fs::canonicalize(&config.models[idx].path).ok();
    if loaded_canonical.is_some() && loaded_canonical == entry_canonical {
        return Ok(json_response(
            409,
            json!({"error": {"message": "Cannot delete the currently loaded model", "type": "conflict"}}),
        ));
    }

    // Remove from disk
    let path = PathBuf::from(&config.models[idx].path);
    if path.exists() {
        if let Err(e) = std::fs::remove_dir_all(&path) {
            return Ok(json_response(
                500,
                json!({"error": {"message": format!("Failed to remove directory: {}", e), "type": "server_error"}}),
            ));
        }
    }

    // Remove from config and save
    config.models.remove(idx);
    let _ = save_config(&config);

    Ok(json_response(
        200,
        json!({"id": model_id, "deleted": true}),
    ))
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

    // Load or initialize config
    let mut config = load_config(args.models_dir.as_deref());
    scan_models_dir(&mut config);
    let _ = save_config(&config);

    let loaded_model_path = std::fs::canonicalize(&args.model)
        .unwrap_or_else(|_| PathBuf::from(&args.model))
        .to_string_lossy()
        .to_string();

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

    // Register loaded model in config if not already present
    let model_path = Path::new(&args.model);
    let loaded_canonical = std::fs::canonicalize(model_path).ok();
    let already_registered = config.models.iter().any(|m| {
        std::fs::canonicalize(&m.path).ok() == loaded_canonical
    });
    if !already_registered {
        let id = model_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let quant = model_args
            .quantization
            .as_ref()
            .map(|q| QuantInfo {
                bits: q.bits,
                group_size: q.group_size,
            });
        let size_bytes = calculate_model_size(model_path);
        config.models.push(ModelEntry {
            id,
            repo_id: String::new(),
            path: loaded_model_path.clone(),
            quantization: quant,
            size_bytes: Some(size_bytes),
            downloaded_at: None,
        });
        let _ = save_config(&config);
    }

    eprintln!(
        "Models dir: {}",
        config.models_dir
    );
    eprintln!(
        "Config: {} ({} models registered)",
        config_path().display(),
        config.models.len()
    );

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
        config: RwLock::new(config),
        loaded_model_path,
    });

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    let listener = TcpListener::bind(addr).await?;

    println!();
    println!("  MiniCPM-SALA API Server");
    println!("  http://0.0.0.0:{}", args.port);
    println!();
    println!("  Endpoints:");
    println!("    POST   /v1/chat/completions   - OpenAI-compatible chat");
    println!("    GET    /v1/models             - List models (with metadata)");
    println!("    POST   /v1/models/download    - Download from HuggingFace");
    println!("    DELETE /v1/models/{{id}}        - Delete a model");
    println!("    GET    /health                - Health check");
    println!();
    println!(
        "  Defaults: temperature={}, max_tokens={}, no_think={}",
        args.temperature, args.max_tokens, args.no_think
    );
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
