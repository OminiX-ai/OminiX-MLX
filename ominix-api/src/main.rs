//! Unified OpenAI-compatible API server for OminiX-MLX models.
//!
//! Supports multiple model types behind a single HTTP endpoint:
//!   - ASR: POST /v1/audio/transcriptions (OpenAI Whisper-compatible)
//!   - TTS: POST /v1/audio/speech (OpenAI TTS-compatible)
//!   - LLM: POST /v1/chat/completions
//!
//! Usage:
//!   cargo run --release -p ominix-api -- --asr-model ~/.OminiX/models/qwen3-asr-1.7b
//!   cargo run --release -p ominix-api -- --tts-model ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
//!   cargo run --release -p ominix-api -- --port 9090 --language English
//!
//! Endpoints:
//!   POST   /v1/audio/transcriptions  — Transcribe audio (multipart or JSON)
//!   POST   /v1/audio/speech          — Synthesize speech from text
//!   GET    /v1/models                — List available models
//!   POST   /v1/models/download       — Download model from HuggingFace
//!   DELETE /v1/models/{id}           — Delete a downloaded model
//!   GET    /health                   — Health check

use std::collections::HashSet;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use http_body_util::{Full, StreamBody};
use hyper::body::{Bytes, Frame, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot, RwLock};
use tokio_stream::wrappers::ReceiverStream;

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser)]
#[command(name = "ominix-api", about = "Unified OpenAI-compatible API server for OminiX-MLX")]
struct Args {
    /// Port to listen on
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// ASR model directory (omit to disable ASR)
    #[arg(long)]
    asr_model: Option<String>,

    /// Default language for ASR transcription
    #[arg(long, default_value = "Chinese")]
    language: String,

    /// TTS model directory (omit to disable TTS)
    #[arg(long)]
    tts_model: Option<String>,

    /// Default speaker for TTS (default: "vivian")
    #[arg(long, default_value = "vivian")]
    tts_speaker: String,

    /// Default language for TTS (default: "english")
    #[arg(long, default_value = "english")]
    tts_language: String,

    /// LLM model directory (omit to disable LLM)
    #[arg(long)]
    llm_model: Option<String>,

    /// Models directory for management (default: ~/.ominix/models)
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
    model_type: String,
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
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
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
    std::fs::write(config_path(), serde_json::to_string_pretty(config).unwrap())
}

fn calculate_model_size(dir: &Path) -> u64 {
    let mut total: u64 = 0;
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // Recurse one level (e.g., speech_tokenizer/)
            total += calculate_model_size(&path);
        } else if path.extension().map(|x| x == "safetensors").unwrap_or(false) {
            if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

fn detect_quantization(model_dir: &Path) -> Option<QuantInfo> {
    let cfg: Value = std::fs::File::open(model_dir.join("config.json"))
        .ok()
        .and_then(|f| serde_json::from_reader(f).ok())?;
    let q = cfg.get("quantization")?;
    Some(QuantInfo {
        bits: q.get("bits")?.as_i64()? as i32,
        group_size: q.get("group_size")?.as_i64()? as i32,
    })
}

fn scan_models_dir(config: &mut OminixConfig) {
    let models_dir = Path::new(&config.models_dir);
    if !models_dir.exists() {
        let _ = std::fs::create_dir_all(models_dir);
        return;
    }

    config
        .models
        .retain(|e| Path::new(&e.path).join("config.json").exists());

    let known: HashSet<String> = config.models.iter().map(|m| m.path.clone()).collect();

    let entries = match std::fs::read_dir(models_dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let sub = entry.path();
        if !sub.is_dir() || !sub.join("config.json").exists() {
            continue;
        }

        let path_str = sub.to_string_lossy().to_string();
        if known.contains(&path_str) {
            continue;
        }

        let cfg: Option<Value> = std::fs::File::open(sub.join("config.json"))
            .ok()
            .and_then(|f| serde_json::from_reader(f).ok());

        let quant = cfg
            .as_ref()
            .and_then(|v| v.get("quantization"))
            .and_then(|q| {
                Some(QuantInfo {
                    bits: q.get("bits")?.as_i64()? as i32,
                    group_size: q.get("group_size")?.as_i64()? as i32,
                })
            });

        // Detect model type from config keys
        let model_type = if cfg.as_ref().map_or(false, |v| {
            v.get("talker_config").is_some()
        }) {
            "tts"
        } else if cfg.as_ref().map_or(false, |v| {
            v.get("audio_config").is_some() || v.get("thinker_config").is_some()
        }) {
            "asr"
        } else {
            "llm"
        };

        config.models.push(ModelEntry {
            id: sub
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            repo_id: String::new(),
            path: path_str,
            model_type: model_type.to_string(),
            quantization: quant,
            size_bytes: Some(calculate_model_size(&sub)),
            downloaded_at: None,
        });
    }
}

// ============================================================================
// Model download
// ============================================================================

fn download_model_blocking(
    repo_id: &str,
    models_dir: &Path,
) -> std::result::Result<ModelEntry, String> {
    let token = std::env::var("HF_TOKEN").ok().or_else(|| {
        let home = std::env::var("HOME").ok()?;
        std::fs::read_to_string(PathBuf::from(home).join(".cache/huggingface/token"))
            .ok()
            .map(|s| s.trim().to_string())
    });

    let api = if let Some(ref token) = token {
        hf_hub::api::sync::ApiBuilder::new()
            .with_token(Some(token.clone()))
            .build()
    } else {
        hf_hub::api::sync::ApiBuilder::new().build()
    }
    .map_err(|e| format!("HF API error: {}", e))?;

    let repo = api.model(repo_id.to_string());
    let model_id = repo_id.split('/').last().unwrap_or(repo_id);
    let dest = models_dir.join(model_id);

    if dest.exists() {
        return Err(format!("Already exists: {}", dest.display()));
    }
    std::fs::create_dir_all(&dest).map_err(|e| format!("mkdir: {}", e))?;

    // Required files
    for f in &["config.json", "tokenizer_config.json"] {
        eprintln!("[download] {}", f);
        let cached = repo.get(f).map_err(|e| format!("{}: {}", f, e))?;
        std::fs::copy(&cached, dest.join(f)).map_err(|e| format!("copy {}: {}", f, e))?;
    }

    // Optional files (tokenizer, vocab, merges, generation config, preprocessor config)
    for f in &[
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
        "preprocessor_config.json",
    ] {
        if let Ok(cached) = repo.get(f) {
            eprintln!("[download] {}", f);
            let _ = std::fs::copy(&cached, dest.join(f));
        }
    }

    // TTS speech tokenizer subdirectory (if present)
    let speech_tok_files = [
        "speech_tokenizer/config.json",
        "speech_tokenizer/configuration.json",
        "speech_tokenizer/model.safetensors",
        "speech_tokenizer/preprocessor_config.json",
    ];
    let has_speech_tok = repo.get("speech_tokenizer/config.json").is_ok();
    if has_speech_tok {
        let st_dir = dest.join("speech_tokenizer");
        std::fs::create_dir_all(&st_dir).map_err(|e| format!("mkdir speech_tokenizer: {}", e))?;
        for f in &speech_tok_files {
            if let Ok(cached) = repo.get(f) {
                let fname = f.strip_prefix("speech_tokenizer/").unwrap_or(f);
                eprintln!("[download] {}", f);
                std::fs::copy(&cached, st_dir.join(fname))
                    .map_err(|e| format!("copy {}: {}", f, e))?;
            }
        }
    }

    // Weights (sharded or single)
    if let Ok(idx_path) = repo.get("model.safetensors.index.json") {
        std::fs::copy(&idx_path, dest.join("model.safetensors.index.json"))
            .map_err(|e| format!("copy index: {}", e))?;
        let idx: Value = serde_json::from_str(
            &std::fs::read_to_string(&idx_path).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if let Some(wm) = idx["weight_map"].as_object() {
            let files: HashSet<&str> = wm.values().filter_map(|v| v.as_str()).collect();
            for wf in &files {
                eprintln!("[download] {}", wf);
                let cached = repo.get(wf).map_err(|e| format!("{}: {}", wf, e))?;
                std::fs::copy(&cached, dest.join(wf))
                    .map_err(|e| format!("copy {}: {}", wf, e))?;
            }
        }
    } else {
        eprintln!("[download] model.safetensors");
        let cached = repo
            .get("model.safetensors")
            .map_err(|e| format!("weights: {}", e))?;
        std::fs::copy(&cached, dest.join("model.safetensors"))
            .map_err(|e| format!("copy: {}", e))?;
    }

    let cfg: Option<Value> = std::fs::File::open(dest.join("config.json"))
        .ok()
        .and_then(|f| serde_json::from_reader(f).ok());

    let quant = cfg
        .as_ref()
        .and_then(|v| v.get("quantization"))
        .and_then(|q| {
            Some(QuantInfo {
                bits: q.get("bits")?.as_i64()? as i32,
                group_size: q.get("group_size")?.as_i64()? as i32,
            })
        });

    let model_type = if cfg.as_ref().map_or(false, |v| {
        v.get("talker_config").is_some()
    }) {
        "tts"
    } else if cfg.as_ref().map_or(false, |v| {
        v.get("audio_config").is_some() || v.get("thinker_config").is_some()
    }) {
        "asr"
    } else {
        "llm"
    };

    Ok(ModelEntry {
        id: model_id.to_string(),
        repo_id: repo_id.to_string(),
        path: dest.to_string_lossy().to_string(),
        model_type: model_type.to_string(),
        quantization: quant,
        size_bytes: Some(calculate_model_size(&dest)),
        downloaded_at: Some(format!("{}", timestamp())),
    })
}

// ============================================================================
// Request types
// ============================================================================

#[derive(Deserialize)]
struct TranscriptionRequest {
    /// Base64-encoded audio file
    #[serde(default)]
    file: Option<String>,
    /// Local file path (alternative to base64)
    #[serde(default)]
    file_path: Option<String>,
    /// Language (e.g., "Chinese", "English")
    #[serde(default)]
    language: Option<String>,
    /// Response format: "json" (default) or "verbose_json"
    #[serde(default)]
    response_format: Option<String>,
}

#[derive(Deserialize)]
struct DownloadRequest {
    repo_id: String,
}

// ============================================================================
// Chat Completions types (OpenAI-compatible)
// ============================================================================

#[derive(Deserialize)]
struct ChatCompletionRequest {
    messages: Vec<ChatMessage>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: ChatUsage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: u32,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct ChatUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
}

#[derive(Serialize)]
struct ChunkChoice {
    index: u32,
    delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(&msg.role);
        prompt.push('\n');
        prompt.push_str(&msg.content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

// ============================================================================
// LLM inference
// ============================================================================

#[cfg(feature = "llm")]
enum LlmResponseChannel {
    Full(oneshot::Sender<std::result::Result<LlmResult, String>>),
    Stream(mpsc::Sender<LlmStreamEvent>),
}

#[cfg(feature = "llm")]
struct LlmResult {
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[cfg(feature = "llm")]
enum LlmStreamEvent {
    Token(String),
    Done {
        prompt_tokens: usize,
        completion_tokens: usize,
    },
}

#[cfg(feature = "llm")]
struct LlmRequest {
    prompt: String,
    temperature: f32,
    max_tokens: u32,
    response: LlmResponseChannel,
}

// ============================================================================
// ASR inference
// ============================================================================

#[cfg(feature = "asr")]
struct AsrRequest {
    samples: Vec<f32>,
    language: String,
    audio_duration_secs: f32,
    response_tx: oneshot::Sender<std::result::Result<AsrResult, String>>,
}

#[cfg(feature = "asr")]
struct AsrResult {
    text: String,
    processing_secs: f32,
    audio_duration_secs: f32,
}

#[cfg(feature = "asr")]
fn asr_inference_worker(
    mut model: qwen3_asr_mlx::Qwen3ASR,
    mut rx: mpsc::Receiver<AsrRequest>,
) {
    while let Some(req) = rx.blocking_recv() {
        let start = Instant::now();
        let result = model
            .transcribe_samples(&req.samples, &req.language)
            .map(|text| AsrResult {
                text,
                processing_secs: start.elapsed().as_secs_f32(),
                audio_duration_secs: req.audio_duration_secs,
            })
            .map_err(|e| format!("{}", e));
        let _ = req.response_tx.send(result);
    }
}

// ============================================================================
// TTS inference
// ============================================================================

#[derive(Deserialize)]
struct SpeechRequest {
    /// Text to synthesize (required)
    input: String,
    /// Speaker name (e.g., "vivian")
    #[serde(default)]
    voice: Option<String>,
    /// Language (e.g., "english", "chinese")
    #[serde(default)]
    language: Option<String>,
    /// Response format (only "wav" supported)
    #[serde(default)]
    response_format: Option<String>,
    /// Sampling temperature
    #[serde(default)]
    temperature: Option<f32>,
    /// Top-k sampling
    #[serde(default)]
    top_k: Option<i32>,
    /// Top-p (nucleus) sampling
    #[serde(default)]
    top_p: Option<f32>,
    /// Random seed
    #[serde(default)]
    seed: Option<u64>,
    /// Base64-encoded WAV for voice cloning
    #[serde(default)]
    reference_audio: Option<String>,
    /// Transcript of reference audio (for ICL mode)
    #[serde(default)]
    reference_text: Option<String>,
    /// Voice design instruction
    #[serde(default)]
    instruct: Option<String>,
}

#[cfg(feature = "tts")]
struct TtsRequest {
    speech_req: SpeechRequest,
    default_speaker: String,
    default_language: String,
    response_tx: oneshot::Sender<std::result::Result<TtsResult, String>>,
}

#[cfg(feature = "tts")]
struct TtsResult {
    wav_bytes: Vec<u8>,
    duration_secs: f32,
    processing_secs: f32,
}

#[cfg(feature = "tts")]
fn encode_wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_samples = samples.len() as u32;
    let data_size = num_samples * 2; // 16-bit PCM = 2 bytes per sample
    let file_size = 36 + data_size;

    let mut buf = Vec::with_capacity(file_size as usize + 8);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let val = (clamped * 32767.0) as i16;
        buf.extend_from_slice(&val.to_le_bytes());
    }

    buf
}

#[cfg(feature = "tts")]
fn tts_inference_worker(
    mut synth: qwen3_tts_mlx::Synthesizer,
    mut rx: mpsc::Receiver<TtsRequest>,
) {
    while let Some(req) = rx.blocking_recv() {
        let start = Instant::now();
        let sr = &req.speech_req;

        let speaker = sr.voice.as_deref().unwrap_or(&req.default_speaker);
        let language = sr.language.as_deref().unwrap_or(&req.default_language);

        let opts = qwen3_tts_mlx::SynthesizeOptions {
            speaker,
            language,
            temperature: sr.temperature,
            top_k: sr.top_k,
            top_p: sr.top_p,
            max_new_tokens: None,
            seed: sr.seed,
        };

        let result = if let Some(ref instruct) = sr.instruct {
            // Voice design mode
            synth
                .synthesize_voice_design_with_timing(&sr.input, instruct, language, &opts)
        } else if let Some(ref ref_audio_b64) = sr.reference_audio {
            // Voice cloning mode
            use base64::Engine;
            let decoded = match base64::prelude::BASE64_STANDARD.decode(ref_audio_b64) {
                Ok(d) => d,
                Err(e) => {
                    let _ = req.response_tx.send(Err(format!("Invalid base64 reference_audio: {}", e)));
                    continue;
                }
            };

            // Decode WAV bytes to f32 samples
            let ref_samples = match decode_wav_to_f32(&decoded) {
                Ok(s) => s,
                Err(e) => {
                    let _ = req.response_tx.send(Err(format!("Invalid reference audio WAV: {}", e)));
                    continue;
                }
            };

            if let Some(ref ref_text) = sr.reference_text {
                // ICL voice cloning (note: may not work well on Apple Silicon)
                synth.synthesize_voice_clone_icl_with_timing(
                    &sr.input, &ref_samples, ref_text, language, &opts,
                )
            } else {
                // x_vector_only voice cloning
                synth.synthesize_voice_clone_with_timing(
                    &sr.input, &ref_samples, language, &opts,
                )
            }
        } else {
            // Standard preset speaker mode
            synth.synthesize_with_timing(&sr.input, &opts)
        };

        let result = match result {
            Ok((samples, timing)) => {
                let normalized = qwen3_tts_mlx::normalize_audio(&samples, 0.95);
                let duration_secs = samples.len() as f32 / synth.sample_rate as f32;
                let wav_bytes = encode_wav_bytes(&normalized, synth.sample_rate);
                eprintln!(
                    "[{}] TTS: {:.1}s audio in {:.0}ms (prefill={:.0}ms gen={:.0}ms decode={:.0}ms)",
                    timestamp(),
                    duration_secs,
                    timing.total_ms,
                    timing.prefill_ms,
                    timing.generation_ms,
                    timing.decode_ms,
                );
                Ok(TtsResult {
                    wav_bytes,
                    duration_secs,
                    processing_secs: start.elapsed().as_secs_f32(),
                })
            }
            Err(e) => Err(format!("{}", e)),
        };

        let _ = req.response_tx.send(result);
    }
}

/// Decode raw WAV bytes (16-bit PCM or 32-bit float) into f32 samples.
#[cfg(feature = "tts")]
fn decode_wav_to_f32(data: &[u8]) -> std::result::Result<Vec<f32>, String> {
    if data.len() < 44 {
        return Err("WAV data too short".into());
    }
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("Not a WAV file".into());
    }

    // Parse fmt chunk
    let audio_format = u16::from_le_bytes([data[20], data[21]]);
    let num_channels = u16::from_le_bytes([data[22], data[23]]) as usize;
    let bits_per_sample = u16::from_le_bytes([data[34], data[35]]);

    // Find data chunk
    let mut pos = 12;
    while pos + 8 < data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]) as usize;

        if chunk_id == b"data" {
            let start = pos + 8;
            let end = (start + chunk_size).min(data.len());
            let raw = &data[start..end];

            let samples: Vec<f32> = if audio_format == 1 && bits_per_sample == 16 {
                // PCM 16-bit
                raw.chunks_exact(2 * num_channels)
                    .map(|frame| {
                        let val = i16::from_le_bytes([frame[0], frame[1]]);
                        val as f32 / 32768.0
                    })
                    .collect()
            } else if audio_format == 3 && bits_per_sample == 32 {
                // IEEE float 32-bit
                raw.chunks_exact(4 * num_channels)
                    .map(|frame| f32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]))
                    .collect()
            } else {
                return Err(format!(
                    "Unsupported WAV format: audio_format={}, bits={}",
                    audio_format, bits_per_sample
                ));
            };

            return Ok(samples);
        }

        pos += 8 + chunk_size;
        // Align to 2 bytes
        if chunk_size % 2 != 0 {
            pos += 1;
        }
    }

    Err("No data chunk found in WAV".into())
}

// ============================================================================
// Audio loading helpers
// ============================================================================

#[cfg(feature = "asr")]
fn load_audio_from_bytes(data: &[u8]) -> std::result::Result<(Vec<f32>, u32), String> {
    use qwen3_asr_mlx::audio;

    let id = uuid_simple();
    let tmp = std::env::temp_dir().join(format!("ominix_asr_{}", id));
    std::fs::write(&tmp, data).map_err(|e| format!("Write temp: {}", e))?;

    // Try WAV first
    if let Ok(result) = audio::load_wav(&tmp) {
        let _ = std::fs::remove_file(&tmp);
        return Ok(result);
    }

    // Fall back to ffmpeg conversion
    let wav_tmp = std::env::temp_dir().join(format!("ominix_asr_{}.wav", id));
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-i",
            &tmp.to_string_lossy(),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-y",
            &wav_tmp.to_string_lossy(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    let _ = std::fs::remove_file(&tmp);

    match status {
        Ok(s) if s.success() => {
            let r = audio::load_wav(&wav_tmp).map_err(|e| format!("{}", e))?;
            let _ = std::fs::remove_file(&wav_tmp);
            Ok(r)
        }
        _ => Err("Unsupported audio format (not WAV and ffmpeg unavailable)".into()),
    }
}

// ============================================================================
// Multipart parsing (for OpenAI Whisper API compatibility)
// ============================================================================

struct MultipartField {
    name: String,
    data: Vec<u8>,
}

fn extract_boundary(content_type: &str) -> Option<String> {
    content_type
        .split(';')
        .map(|s| s.trim())
        .find(|s| s.starts_with("boundary="))
        .map(|s| s["boundary=".len()..].trim_matches('"').to_string())
}

fn parse_multipart(body: &[u8], boundary: &str) -> Vec<MultipartField> {
    let delim = format!("--{}", boundary);
    let delim_bytes = delim.as_bytes();
    let mut fields = Vec::new();
    let mut pos = 0;

    loop {
        let bp = match body[pos..]
            .windows(delim_bytes.len())
            .position(|w| w == delim_bytes)
        {
            Some(p) => pos + p,
            None => break,
        };
        pos = bp + delim_bytes.len();

        // End boundary --
        if pos + 2 <= body.len() && body[pos] == b'-' && body[pos + 1] == b'-' {
            break;
        }
        // Skip CRLF
        if pos + 2 <= body.len() && body[pos] == b'\r' && body[pos + 1] == b'\n' {
            pos += 2;
        }

        // Find header/body separator
        let hdr_end = match body[pos..].windows(4).position(|w| w == b"\r\n\r\n") {
            Some(p) => pos + p,
            None => break,
        };
        let headers = String::from_utf8_lossy(&body[pos..hdr_end]);
        let data_start = hdr_end + 4;

        // Find next boundary to get data end
        let data_end = match body[data_start..]
            .windows(delim_bytes.len())
            .position(|w| w == delim_bytes)
        {
            Some(p) => {
                let end = data_start + p;
                // Remove trailing CRLF
                if end >= 2 && body[end - 2] == b'\r' && body[end - 1] == b'\n' {
                    end - 2
                } else {
                    end
                }
            }
            None => body.len(),
        };

        // Extract field name from Content-Disposition
        if let Some(name) = extract_field_name(&headers) {
            fields.push(MultipartField {
                name,
                data: body[data_start..data_end].to_vec(),
            });
        }

        pos = data_start;
    }

    fields
}

fn extract_field_name(headers: &str) -> Option<String> {
    let search = "name=\"";
    let start = headers.find(search)? + search.len();
    let end = headers[start..].find('"')?;
    Some(headers[start..start + end].to_string())
}

// ============================================================================
// Body type for SSE streaming
// ============================================================================

type SseStream = StreamBody<ReceiverStream<std::result::Result<Frame<Bytes>, Infallible>>>;

enum ApiBody {
    Full(Full<Bytes>),
    Stream(SseStream),
}

impl http_body::Body for ApiBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<std::result::Result<Frame<Self::Data>, Self::Error>>> {
        match self.get_mut() {
            ApiBody::Full(body) => Pin::new(body).poll_frame(cx),
            ApiBody::Stream(body) => Pin::new(body).poll_frame(cx),
        }
    }
}

fn wrap_full(resp: Response<Full<Bytes>>) -> Response<ApiBody> {
    let (parts, body) = resp.into_parts();
    Response::from_parts(parts, ApiBody::Full(body))
}

// ============================================================================
// Server state
// ============================================================================

struct ServerState {
    #[cfg(feature = "asr")]
    asr_tx: Option<mpsc::Sender<AsrRequest>>,

    #[cfg(feature = "tts")]
    tts_tx: Option<mpsc::Sender<TtsRequest>>,
    #[cfg(feature = "tts")]
    default_tts_speaker: String,
    #[cfg(feature = "tts")]
    default_tts_language: String,

    #[cfg(feature = "llm")]
    llm_tx: Option<mpsc::Sender<LlmRequest>>,
    #[cfg(feature = "llm")]
    llm_model_name: String,

    default_language: String,
    config: RwLock<OminixConfig>,
    loaded_models: Vec<String>,
}

// ============================================================================
// HTTP handlers
// ============================================================================

async fn handle_request(
    req: Request<Incoming>,
    state: Arc<ServerState>,
) -> std::result::Result<Response<ApiBody>, hyper::Error> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    eprintln!("[{}] {} {}", timestamp(), method, path);

    match (method.clone(), path.as_str()) {
        (Method::POST, "/v1/audio/transcriptions") => {
            let ct = req
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();
            let body = collect_body_bytes(req).await;
            handle_transcription(&body, &ct, &state).await.map(wrap_full)
        }

        (Method::POST, "/v1/audio/speech") => {
            let body = collect_body(req).await;
            handle_speech(&body, &state).await.map(wrap_full)
        }

        (Method::POST, "/v1/chat/completions") => {
            let body = collect_body(req).await;
            handle_chat_completions(&body, &state).await
        }

        (Method::GET, "/v1/models") => handle_list_models(&state).await.map(wrap_full),

        (Method::POST, "/v1/models/download") => {
            let body = collect_body(req).await;
            handle_download_model(&body, &state).await.map(wrap_full)
        }

        (Method::GET, "/health") => Ok(wrap_full(json_response(200, json!({"status": "ok"})))),

        (Method::OPTIONS, _) => Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            .header(
                "Access-Control-Allow-Headers",
                "Content-Type, Authorization",
            )
            .body(ApiBody::Full(Full::new(Bytes::new())))
            .unwrap()),

        _ if method == Method::DELETE && path.starts_with("/v1/models/") => {
            let model_id = &path["/v1/models/".len()..];
            handle_delete_model(model_id, &state).await.map(wrap_full)
        }

        _ => Ok(wrap_full(json_response(
            404,
            json!({"error": {"message": "Not found", "type": "invalid_request_error"}}),
        ))),
    }
}

async fn handle_transcription(
    body: &[u8],
    content_type: &str,
    state: &Arc<ServerState>,
) -> std::result::Result<Response<Full<Bytes>>, hyper::Error> {
    #[cfg(not(feature = "asr"))]
    {
        let _ = (body, content_type, state);
        return Ok(json_response(
            501,
            json!({"error": {"message": "ASR not enabled. Build with --features asr", "type": "not_implemented"}}),
        ));
    }

    #[cfg(feature = "asr")]
    {
        use base64::prelude::*;
        use qwen3_asr_mlx::audio;

        let asr_tx = match &state.asr_tx {
            Some(tx) => tx,
            None => {
                return Ok(json_response(
                    503,
                    json!({"error": {"message": "No ASR model loaded. Start with --asr-model <path>", "type": "service_unavailable"}}),
                ));
            }
        };

        // Parse request — multipart or JSON
        let (audio_data, language, response_format) =
            if content_type.contains("multipart/form-data") {
                let boundary = match extract_boundary(content_type) {
                    Some(b) => b,
                    None => {
                        return Ok(json_response(
                            400,
                            json!({"error": {"message": "Missing boundary in multipart", "type": "invalid_request_error"}}),
                        ))
                    }
                };
                let fields = parse_multipart(body, &boundary);
                let file_data = match fields.iter().find(|f| f.name == "file") {
                    Some(f) => f.data.clone(),
                    None => {
                        return Ok(json_response(
                            400,
                            json!({"error": {"message": "Missing 'file' field", "type": "invalid_request_error"}}),
                        ))
                    }
                };
                let lang = fields
                    .iter()
                    .find(|f| f.name == "language")
                    .map(|f| String::from_utf8_lossy(&f.data).trim().to_string());
                let fmt = fields
                    .iter()
                    .find(|f| f.name == "response_format")
                    .map(|f| String::from_utf8_lossy(&f.data).trim().to_string());
                (file_data, lang, fmt)
            } else {
                let body_str = String::from_utf8_lossy(body);
                let req: TranscriptionRequest = match serde_json::from_str(&body_str) {
                    Ok(r) => r,
                    Err(e) => {
                        return Ok(json_response(
                            400,
                            json!({"error": {"message": format!("Invalid JSON: {}", e), "type": "invalid_request_error"}}),
                        ))
                    }
                };

                let audio_data = if let Some(ref b64) = req.file {
                    match BASE64_STANDARD.decode(b64) {
                        Ok(d) => d,
                        Err(e) => {
                            return Ok(json_response(
                                400,
                                json!({"error": {"message": format!("Invalid base64: {}", e), "type": "invalid_request_error"}}),
                            ))
                        }
                    }
                } else if let Some(ref path) = req.file_path {
                    match std::fs::read(path) {
                        Ok(d) => d,
                        Err(e) => {
                            return Ok(json_response(
                                400,
                                json!({"error": {"message": format!("Cannot read file: {}", e), "type": "invalid_request_error"}}),
                            ))
                        }
                    }
                } else {
                    return Ok(json_response(
                        400,
                        json!({"error": {"message": "Either 'file' (base64) or 'file_path' is required", "type": "invalid_request_error"}}),
                    ));
                };

                (audio_data, req.language, req.response_format)
            };

        let language = language.unwrap_or_else(|| state.default_language.clone());
        let verbose = response_format.as_deref() == Some("verbose_json");

        // Load and resample audio
        let (samples, sample_rate) = match load_audio_from_bytes(&audio_data) {
            Ok(r) => r,
            Err(e) => {
                return Ok(json_response(
                    400,
                    json!({"error": {"message": e, "type": "invalid_request_error"}}),
                ))
            }
        };

        let samples = match audio::resample(&samples, sample_rate, 16000) {
            Ok(s) => s,
            Err(e) => {
                return Ok(json_response(
                    500,
                    json!({"error": {"message": format!("Resample error: {}", e), "type": "server_error"}}),
                ))
            }
        };

        let audio_duration = samples.len() as f32 / 16000.0;
        eprintln!(
            "[{}] Audio: {:.1}s, {} samples, language={}",
            timestamp(),
            audio_duration,
            samples.len(),
            language
        );

        // Send to ASR inference worker
        let (resp_tx, resp_rx) = oneshot::channel();
        if asr_tx
            .send(AsrRequest {
                samples,
                language: language.clone(),
                audio_duration_secs: audio_duration,
                response_tx: resp_tx,
            })
            .await
            .is_err()
        {
            return Ok(json_response(
                500,
                json!({"error": {"message": "ASR inference worker unavailable", "type": "server_error"}}),
            ));
        }

        match resp_rx.await {
            Ok(Ok(result)) => {
                let rtf = result.audio_duration_secs / result.processing_secs;
                eprintln!(
                    "[{}] Transcribed {:.1}s in {:.2}s ({:.1}x RT)",
                    timestamp(),
                    result.audio_duration_secs,
                    result.processing_secs,
                    rtf
                );

                if verbose {
                    Ok(json_response(
                        200,
                        json!({
                            "text": result.text,
                            "language": language,
                            "duration": result.audio_duration_secs,
                            "processing_time": result.processing_secs,
                            "realtime_factor": rtf,
                        }),
                    ))
                } else {
                    Ok(json_response(200, json!({"text": result.text})))
                }
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
}

async fn handle_speech(
    body: &str,
    state: &Arc<ServerState>,
) -> std::result::Result<Response<Full<Bytes>>, hyper::Error> {
    #[cfg(not(feature = "tts"))]
    {
        let _ = (body, state);
        return Ok(json_response(
            501,
            json!({"error": {"message": "TTS not enabled. Build with --features tts", "type": "not_implemented"}}),
        ));
    }

    #[cfg(feature = "tts")]
    {
        let tts_tx = match &state.tts_tx {
            Some(tx) => tx,
            None => {
                return Ok(json_response(
                    503,
                    json!({"error": {"message": "No TTS model loaded. Start with --tts-model <path>", "type": "service_unavailable"}}),
                ));
            }
        };

        let speech_req: SpeechRequest = match serde_json::from_str(body) {
            Ok(r) => r,
            Err(e) => {
                return Ok(json_response(
                    400,
                    json!({"error": {"message": format!("Invalid JSON: {}", e), "type": "invalid_request_error"}}),
                ));
            }
        };

        if speech_req.input.is_empty() {
            return Ok(json_response(
                400,
                json!({"error": {"message": "'input' text is required", "type": "invalid_request_error"}}),
            ));
        }

        // Validate response_format
        if let Some(ref fmt) = speech_req.response_format {
            if fmt != "wav" {
                return Ok(json_response(
                    400,
                    json!({"error": {"message": format!("Unsupported response_format '{}'. Only 'wav' is supported.", fmt), "type": "invalid_request_error"}}),
                ));
            }
        }

        let voice = speech_req.voice.as_deref().unwrap_or(&state.default_tts_speaker);
        let lang = speech_req.language.as_deref().unwrap_or(&state.default_tts_language);
        eprintln!(
            "[{}] TTS request: voice={}, language={}, text=\"{}\"",
            timestamp(),
            voice,
            lang,
            if speech_req.input.len() > 80 {
                format!("{}...", &speech_req.input[..80])
            } else {
                speech_req.input.clone()
            }
        );

        let (resp_tx, resp_rx) = oneshot::channel();
        if tts_tx
            .send(TtsRequest {
                speech_req,
                default_speaker: state.default_tts_speaker.clone(),
                default_language: state.default_tts_language.clone(),
                response_tx: resp_tx,
            })
            .await
            .is_err()
        {
            return Ok(json_response(
                500,
                json!({"error": {"message": "TTS inference worker unavailable", "type": "server_error"}}),
            ));
        }

        match resp_rx.await {
            Ok(Ok(result)) => {
                eprintln!(
                    "[{}] TTS complete: {:.1}s audio in {:.2}s",
                    timestamp(),
                    result.duration_secs,
                    result.processing_secs,
                );

                Ok(Response::builder()
                    .status(StatusCode::OK)
                    .header("Content-Type", "audio/wav")
                    .header("Access-Control-Allow-Origin", "*")
                    .header("X-Audio-Duration", format!("{:.2}", result.duration_secs))
                    .header(
                        "X-Processing-Time",
                        format!("{:.2}", result.processing_secs),
                    )
                    .body(Full::new(Bytes::from(result.wav_bytes)))
                    .unwrap())
            }
            Ok(Err(e)) => Ok(json_response(
                500,
                json!({"error": {"message": e, "type": "server_error"}}),
            )),
            Err(_) => Ok(json_response(
                500,
                json!({"error": {"message": "TTS inference channel closed", "type": "server_error"}}),
            )),
        }
    }
}

async fn handle_chat_completions(
    body: &str,
    state: &Arc<ServerState>,
) -> std::result::Result<Response<ApiBody>, hyper::Error> {
    #[cfg(not(feature = "llm"))]
    {
        let _ = (body, state);
        return Ok(wrap_full(json_response(
            501,
            json!({"error": {"message": "LLM not enabled. Build with --features llm", "type": "not_implemented"}}),
        )));
    }

    #[cfg(feature = "llm")]
    {
        let llm_tx = match &state.llm_tx {
            Some(tx) => tx,
            None => {
                return Ok(wrap_full(json_response(
                    503,
                    json!({"error": {"message": "No LLM model loaded. Start with --llm-model <path>", "type": "service_unavailable"}}),
                )));
            }
        };

        let req: ChatCompletionRequest = match serde_json::from_str(body) {
            Ok(r) => r,
            Err(e) => {
                return Ok(wrap_full(json_response(
                    400,
                    json!({"error": {"message": format!("Invalid JSON: {}", e), "type": "invalid_request_error"}}),
                )));
            }
        };

        if req.messages.is_empty() {
            return Ok(wrap_full(json_response(
                400,
                json!({"error": {"message": "'messages' must not be empty", "type": "invalid_request_error"}}),
            )));
        }

        let prompt = format_chat_prompt(&req.messages);
        let temperature = req.temperature.unwrap_or(0.7);
        let max_tokens = req.max_tokens.unwrap_or(2048);
        let stream = req.stream.unwrap_or(false);
        let model_name = state.llm_model_name.clone();
        let request_id = format!("chatcmpl-{}", uuid_simple());

        eprintln!(
            "[{}] Chat completions: {} messages, temp={}, max_tokens={}, stream={}",
            timestamp(),
            req.messages.len(),
            temperature,
            max_tokens,
            stream,
        );

        if stream {
            // Streaming SSE mode
            let (event_tx, event_rx) = mpsc::channel::<LlmStreamEvent>(32);
            let llm_request = LlmRequest {
                prompt,
                temperature,
                max_tokens,
                response: LlmResponseChannel::Stream(event_tx),
            };

            if llm_tx.send(llm_request).await.is_err() {
                return Ok(wrap_full(json_response(
                    500,
                    json!({"error": {"message": "LLM inference worker unavailable", "type": "server_error"}}),
                )));
            }

            let (frame_tx, frame_rx) =
                mpsc::channel::<std::result::Result<Frame<Bytes>, Infallible>>(32);
            let rid = request_id.clone();
            let mn = model_name.clone();

            tokio::spawn(async move {
                let created = timestamp();

                // Send role delta
                let chunk = ChatCompletionChunk {
                    id: rid.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: mn.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: Some("assistant".to_string()),
                            content: None,
                        },
                        finish_reason: None,
                    }],
                };
                let data = format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap());
                let _ = frame_tx.send(Ok(Frame::data(Bytes::from(data)))).await;

                let mut event_rx = event_rx;
                while let Some(event) = event_rx.recv().await {
                    match event {
                        LlmStreamEvent::Token(text) => {
                            let chunk = ChatCompletionChunk {
                                id: rid.clone(),
                                object: "chat.completion.chunk",
                                created,
                                model: mn.clone(),
                                choices: vec![ChunkChoice {
                                    index: 0,
                                    delta: ChunkDelta {
                                        role: None,
                                        content: Some(text),
                                    },
                                    finish_reason: None,
                                }],
                            };
                            let data =
                                format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap());
                            if frame_tx
                                .send(Ok(Frame::data(Bytes::from(data))))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        LlmStreamEvent::Done { .. } => {
                            let chunk = ChatCompletionChunk {
                                id: rid.clone(),
                                object: "chat.completion.chunk",
                                created,
                                model: mn.clone(),
                                choices: vec![ChunkChoice {
                                    index: 0,
                                    delta: ChunkDelta {
                                        role: None,
                                        content: None,
                                    },
                                    finish_reason: Some("stop".to_string()),
                                }],
                            };
                            let data =
                                format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap());
                            let _ = frame_tx.send(Ok(Frame::data(Bytes::from(data)))).await;
                            let _ = frame_tx
                                .send(Ok(Frame::data(Bytes::from("data: [DONE]\n\n"))))
                                .await;
                            break;
                        }
                    }
                }
            });

            let stream_body = StreamBody::new(ReceiverStream::new(frame_rx));
            let resp = Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "text/event-stream")
                .header("Cache-Control", "no-cache")
                .header("Connection", "keep-alive")
                .header("Access-Control-Allow-Origin", "*")
                .body(ApiBody::Stream(stream_body))
                .unwrap();
            Ok(resp)
        } else {
            // Non-streaming mode
            let (resp_tx, resp_rx) = oneshot::channel();
            let llm_request = LlmRequest {
                prompt,
                temperature,
                max_tokens,
                response: LlmResponseChannel::Full(resp_tx),
            };

            if llm_tx.send(llm_request).await.is_err() {
                return Ok(wrap_full(json_response(
                    500,
                    json!({"error": {"message": "LLM inference worker unavailable", "type": "server_error"}}),
                )));
            }

            match resp_rx.await {
                Ok(Ok(result)) => {
                    eprintln!(
                        "[{}] Chat completions: {} prompt + {} completion tokens",
                        timestamp(),
                        result.prompt_tokens,
                        result.completion_tokens,
                    );
                    let resp = ChatCompletionResponse {
                        id: request_id,
                        object: "chat.completion",
                        created: timestamp(),
                        model: model_name,
                        choices: vec![ChatChoice {
                            index: 0,
                            message: ChatMessage {
                                role: "assistant".to_string(),
                                content: result.text,
                            },
                            finish_reason: "stop".to_string(),
                        }],
                        usage: ChatUsage {
                            prompt_tokens: result.prompt_tokens,
                            completion_tokens: result.completion_tokens,
                            total_tokens: result.prompt_tokens + result.completion_tokens,
                        },
                    };
                    Ok(wrap_full(json_response(
                        200,
                        serde_json::to_value(&resp).unwrap(),
                    )))
                }
                Ok(Err(e)) => Ok(wrap_full(json_response(
                    500,
                    json!({"error": {"message": e, "type": "server_error"}}),
                ))),
                Err(_) => Ok(wrap_full(json_response(
                    500,
                    json!({"error": {"message": "LLM inference channel closed", "type": "server_error"}}),
                ))),
            }
        }
    }
}

async fn handle_list_models(
    state: &Arc<ServerState>,
) -> std::result::Result<Response<Full<Bytes>>, hyper::Error> {
    let mut config = state.config.write().await;
    scan_models_dir(&mut config);

    let data: Vec<Value> = config
        .models
        .iter()
        .map(|m| {
            let loaded = state.loaded_models.iter().any(|lp| {
                std::fs::canonicalize(lp).ok() == std::fs::canonicalize(&m.path).ok()
            });

            let mut obj = json!({
                "id": m.id,
                "object": "model",
                "model_type": m.model_type,
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
        eprintln!("[download] Starting: {}", repo_id);
        match download_model_blocking(&repo_id, &models_dir) {
            Ok(entry) => {
                let mut config = state_clone.config.blocking_write();
                config.models.push(entry);
                let _ = save_config(&config);
                eprintln!("[download] Complete: {}", repo_id);
            }
            Err(e) => {
                eprintln!("[download] Failed: {}: {}", repo_id, e);
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

    // Prevent deleting loaded models
    let entry_canonical = std::fs::canonicalize(&config.models[idx].path).ok();
    let is_loaded = state.loaded_models.iter().any(|lp| {
        std::fs::canonicalize(lp).ok() == entry_canonical
    });
    if is_loaded {
        return Ok(json_response(
            409,
            json!({"error": {"message": "Cannot delete a currently loaded model", "type": "conflict"}}),
        ));
    }

    let path = PathBuf::from(&config.models[idx].path);
    if path.exists() {
        if let Err(e) = std::fs::remove_dir_all(&path) {
            return Ok(json_response(
                500,
                json!({"error": {"message": format!("Failed to remove: {}", e), "type": "server_error"}}),
            ));
        }
    }

    config.models.remove(idx);
    let _ = save_config(&config);

    Ok(json_response(200, json!({"id": model_id, "deleted": true})))
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

async fn collect_body_bytes(req: Request<Incoming>) -> Vec<u8> {
    http_body_util::BodyExt::collect(req.into_body())
        .await
        .map(|b| b.to_bytes().to_vec())
        .unwrap_or_default()
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

    // Load config
    let mut config = load_config(args.models_dir.as_deref());
    scan_models_dir(&mut config);
    let _ = save_config(&config);

    let mut loaded_models: Vec<String> = Vec::new();

    // --- ASR model ---
    #[cfg(feature = "asr")]
    let asr_tx = if let Some(ref asr_path) = args.asr_model {
        let model_path = PathBuf::from(asr_path);
        eprintln!("Loading ASR model from: {}", model_path.display());
        let t0 = Instant::now();
        let model = qwen3_asr_mlx::Qwen3ASR::load(&model_path)?;
        eprintln!("ASR model loaded in {:.1}s", t0.elapsed().as_secs_f64());

        // Register in config
        let canonical = std::fs::canonicalize(&model_path)
            .unwrap_or(model_path.clone())
            .to_string_lossy()
            .to_string();
        loaded_models.push(canonical.clone());

        if !config.models.iter().any(|m| {
            std::fs::canonicalize(&m.path).ok()
                == std::fs::canonicalize(&model_path).ok()
        }) {
            let id = model_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            config.models.push(ModelEntry {
                id,
                repo_id: String::new(),
                path: canonical,
                model_type: "asr".to_string(),
                quantization: detect_quantization(&model_path),
                size_bytes: Some(calculate_model_size(&model_path)),
                downloaded_at: None,
            });
            let _ = save_config(&config);
        }

        let (tx, rx) = mpsc::channel::<AsrRequest>(1);
        std::thread::spawn(move || asr_inference_worker(model, rx));
        Some(tx)
    } else {
        // Try default path
        let default_path = qwen3_asr_mlx::default_model_path();
        if default_path.join("config.json").exists() {
            eprintln!("Loading ASR model from default: {}", default_path.display());
            let t0 = Instant::now();
            match qwen3_asr_mlx::Qwen3ASR::load(&default_path) {
                Ok(model) => {
                    eprintln!("ASR model loaded in {:.1}s", t0.elapsed().as_secs_f64());
                    let canonical = std::fs::canonicalize(&default_path)
                        .unwrap_or(default_path.clone())
                        .to_string_lossy()
                        .to_string();
                    loaded_models.push(canonical);

                    let (tx, rx) = mpsc::channel::<AsrRequest>(1);
                    std::thread::spawn(move || asr_inference_worker(model, rx));
                    Some(tx)
                }
                Err(e) => {
                    eprintln!("Warning: failed to load default ASR model: {}", e);
                    None
                }
            }
        } else {
            None
        }
    };

    #[cfg(not(feature = "asr"))]
    eprintln!("ASR feature not enabled");

    // --- TTS model ---
    #[cfg(feature = "tts")]
    let tts_tx = if let Some(ref tts_path) = args.tts_model {
        let model_path = PathBuf::from(tts_path);
        eprintln!("Loading TTS model from: {}", model_path.display());
        let t0 = Instant::now();
        let synth = qwen3_tts_mlx::Synthesizer::load(&model_path)?;
        eprintln!(
            "TTS model loaded in {:.1}s (type={:?}, speakers={:?})",
            t0.elapsed().as_secs_f64(),
            synth.model_type(),
            synth.speakers(),
        );

        let canonical = std::fs::canonicalize(&model_path)
            .unwrap_or(model_path.clone())
            .to_string_lossy()
            .to_string();
        loaded_models.push(canonical.clone());

        if !config.models.iter().any(|m| {
            std::fs::canonicalize(&m.path).ok()
                == std::fs::canonicalize(&model_path).ok()
        }) {
            let id = model_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            config.models.push(ModelEntry {
                id,
                repo_id: String::new(),
                path: canonical,
                model_type: "tts".to_string(),
                quantization: detect_quantization(&model_path),
                size_bytes: Some(calculate_model_size(&model_path)),
                downloaded_at: None,
            });
            let _ = save_config(&config);
        }

        let (tx, rx) = mpsc::channel::<TtsRequest>(1);
        std::thread::spawn(move || tts_inference_worker(synth, rx));
        Some(tx)
    } else {
        // Auto-discover TTS model from models_dir
        let tts_auto = config
            .models
            .iter()
            .find(|m| m.model_type == "tts")
            .map(|m| PathBuf::from(&m.path));

        if let Some(ref auto_path) = tts_auto {
            if auto_path.join("config.json").exists() {
                eprintln!(
                    "Loading TTS model from models dir: {}",
                    auto_path.display()
                );
                let t0 = Instant::now();
                match qwen3_tts_mlx::Synthesizer::load(auto_path) {
                    Ok(synth) => {
                        eprintln!(
                            "TTS model loaded in {:.1}s (type={:?}, speakers={:?})",
                            t0.elapsed().as_secs_f64(),
                            synth.model_type(),
                            synth.speakers(),
                        );
                        let canonical = std::fs::canonicalize(auto_path)
                            .unwrap_or(auto_path.clone())
                            .to_string_lossy()
                            .to_string();
                        loaded_models.push(canonical);

                        let (tx, rx) = mpsc::channel::<TtsRequest>(1);
                        std::thread::spawn(move || tts_inference_worker(synth, rx));
                        Some(tx)
                    }
                    Err(e) => {
                        eprintln!("Warning: failed to load TTS model: {}", e);
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        }
    };

    #[cfg(not(feature = "tts"))]
    eprintln!("TTS feature not enabled");

    // --- LLM model ---
    #[cfg(feature = "llm")]
    let (llm_tx, llm_model_name) = if let Some(ref llm_path) = args.llm_model {
        let model_path = PathBuf::from(llm_path);
        eprintln!("Loading LLM model from: {}", model_path.display());
        let t0 = Instant::now();

        let tokenizer = qwen3_5_35b_mlx::load_tokenizer(&model_path)?;
        let model = qwen3_5_35b_mlx::load_model(&model_path)?;
        eprintln!("LLM model loaded in {:.1}s", t0.elapsed().as_secs_f64());

        // Load EOS tokens from config.json
        let config_json: Value = serde_json::from_str(
            &std::fs::read_to_string(model_path.join("config.json"))?,
        )?;
        let eos_tokens: HashSet<u32> = match &config_json["eos_token_id"] {
            Value::Array(ids) => ids
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect(),
            Value::Number(n) => {
                let mut s = HashSet::new();
                s.insert(n.as_u64().unwrap_or(248044) as u32);
                s
            }
            _ => {
                let mut s = HashSet::new();
                s.insert(248044u32);
                s
            }
        };
        eprintln!("LLM EOS tokens: {:?}", eos_tokens);

        let model_name = model_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // Register in config
        let canonical = std::fs::canonicalize(&model_path)
            .unwrap_or(model_path.clone())
            .to_string_lossy()
            .to_string();
        loaded_models.push(canonical.clone());

        if !config.models.iter().any(|m| {
            std::fs::canonicalize(&m.path).ok() == std::fs::canonicalize(&model_path).ok()
        }) {
            config.models.push(ModelEntry {
                id: model_name.clone(),
                repo_id: String::new(),
                path: canonical,
                model_type: "llm".to_string(),
                quantization: detect_quantization(&model_path),
                size_bytes: Some(calculate_model_size(&model_path)),
                downloaded_at: None,
            });
            let _ = save_config(&config);
        }

        let (tx, rx) = mpsc::channel::<LlmRequest>(1);

        // Spawn worker thread — closure captures model, tokenizer, eos_tokens
        std::thread::spawn(move || {
            use mlx_rs::ops::indexing::{IndexOp, NewAxis};

            let mut model = model;
            let mut rx = rx;

            while let Some(req) = rx.blocking_recv() {
                let encoding = match tokenizer.encode(req.prompt.as_str(), false) {
                    Ok(e) => e,
                    Err(e) => {
                        match req.response {
                            LlmResponseChannel::Full(tx) => {
                                let _ = tx.send(Err(format!("Tokenization error: {}", e)));
                            }
                            LlmResponseChannel::Stream(tx) => {
                                let _ = tx.blocking_send(LlmStreamEvent::Done {
                                    prompt_tokens: 0,
                                    completion_tokens: 0,
                                });
                            }
                        }
                        continue;
                    }
                };

                let prompt_tokens = encoding.get_ids();
                let prompt_len = prompt_tokens.len();

                let prompt_array = mlx_rs::Array::from(prompt_tokens);
                let prompt_array = prompt_array.index(NewAxis);

                let gen = qwen3_5_35b_mlx::Generate::new(
                    &mut model,
                    req.temperature,
                    &prompt_array,
                );

                match req.response {
                    LlmResponseChannel::Full(tx) => {
                        let mut tokens: Vec<u32> = Vec::new();
                        let mut tx = Some(tx);
                        for token_result in gen.take(req.max_tokens as usize) {
                            match token_result {
                                Ok(token) => {
                                    let token_id = token.item::<u32>();
                                    if eos_tokens.contains(&token_id) {
                                        break;
                                    }
                                    tokens.push(token_id);
                                }
                                Err(e) => {
                                    if let Some(tx) = tx.take() {
                                        let _ = tx.send(Err(format!(
                                            "Generation error: {}",
                                            e
                                        )));
                                    }
                                    break;
                                }
                            }
                        }
                        if let Some(tx) = tx {
                            let text =
                                tokenizer.decode(&tokens, true).unwrap_or_default();
                            let _ = tx.send(Ok(LlmResult {
                                text,
                                prompt_tokens: prompt_len,
                                completion_tokens: tokens.len(),
                            }));
                        }
                    }
                    LlmResponseChannel::Stream(tx) => {
                        let mut completion_tokens = 0usize;
                        for token_result in gen.take(req.max_tokens as usize) {
                            match token_result {
                                Ok(token) => {
                                    let token_id = token.item::<u32>();
                                    if eos_tokens.contains(&token_id) {
                                        break;
                                    }
                                    completion_tokens += 1;
                                    let text = tokenizer
                                        .decode(&[token_id], true)
                                        .unwrap_or_default();
                                    if tx
                                        .blocking_send(LlmStreamEvent::Token(text))
                                        .is_err()
                                    {
                                        break; // Client disconnected
                                    }
                                }
                                Err(e) => {
                                    eprintln!(
                                        "[{}] LLM generation error: {}",
                                        timestamp(),
                                        e
                                    );
                                    break;
                                }
                            }
                        }
                        let _ = tx.blocking_send(LlmStreamEvent::Done {
                            prompt_tokens: prompt_len,
                            completion_tokens,
                        });
                    }
                }
            }
        });

        (Some(tx), model_name)
    } else {
        (None, String::new())
    };

    #[cfg(not(feature = "llm"))]
    eprintln!("LLM feature not enabled");

    eprintln!("Models dir: {}", config.models_dir);
    eprintln!(
        "Config: {} ({} models registered)",
        config_path().display(),
        config.models.len()
    );

    let state = Arc::new(ServerState {
        #[cfg(feature = "asr")]
        asr_tx,
        #[cfg(feature = "tts")]
        tts_tx,
        #[cfg(feature = "tts")]
        default_tts_speaker: args.tts_speaker.clone(),
        #[cfg(feature = "tts")]
        default_tts_language: args.tts_language.clone(),
        #[cfg(feature = "llm")]
        llm_tx,
        #[cfg(feature = "llm")]
        llm_model_name,
        default_language: args.language.clone(),
        config: RwLock::new(config),
        loaded_models,
    });

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    let listener = TcpListener::bind(addr).await?;

    println!();
    println!("  OminiX API Server");
    println!("  http://0.0.0.0:{}", args.port);
    println!();
    println!("  Endpoints:");
    #[cfg(feature = "asr")]
    {
        let asr_status = if state.asr_tx.is_some() {
            "ready"
        } else {
            "no model"
        };
        println!(
            "    POST   /v1/audio/transcriptions  - ASR [{}]",
            asr_status
        );
    }
    #[cfg(feature = "tts")]
    {
        let tts_status = if state.tts_tx.is_some() {
            "ready"
        } else {
            "no model"
        };
        println!(
            "    POST   /v1/audio/speech           - TTS [{}]",
            tts_status
        );
    }
    #[cfg(feature = "llm")]
    {
        let llm_status = if state.llm_tx.is_some() {
            "ready"
        } else {
            "no model"
        };
        println!(
            "    POST   /v1/chat/completions      - LLM Chat [{}]",
            llm_status
        );
    }
    println!("    GET    /v1/models                - List models");
    println!("    POST   /v1/models/download       - Download from HuggingFace");
    println!("    DELETE /v1/models/{{id}}           - Delete a model");
    println!("    GET    /health                   - Health check");
    println!();
    println!("  Default ASR language: {}", args.language);
    #[cfg(feature = "tts")]
    if state.tts_tx.is_some() {
        println!(
            "  Default TTS speaker: {}, language: {}",
            args.tts_speaker, args.tts_language
        );
    }
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
