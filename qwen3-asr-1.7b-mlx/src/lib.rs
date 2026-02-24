//! # qwen3-asr-mlx
//!
//! Qwen3-ASR speech recognition on Apple Silicon using MLX.
//!
//! Supports all Qwen3-ASR model sizes (0.6B, 1.7B) — architecture is fully
//! config-driven. Models are loaded from `config.json` and safetensors weights.
//!
//! ## Architecture
//!
//! - **Audio Encoder (AuT)**: Conv2d frontend + Transformer with windowed attention
//! - **Projector**: Linear projection from encoder dim to decoder dim
//! - **Text Decoder**: Qwen3 LLM with GQA and Q/K RMSNorm
//!
//! ## Example
//!
//! ```rust,ignore
//! use qwen3_asr_mlx::{Qwen3ASR, default_model_path};
//!
//! let mut model = Qwen3ASR::load(default_model_path())?;
//! let text = model.transcribe("audio.wav")?;
//! println!("{}", text);
//! ```

pub mod audio;
pub mod encoder;
pub mod error;
pub mod model;
pub mod qwen;

pub use error::Error;
pub use model::{Qwen3ASR, Qwen3ASRConfig, SamplingConfig};
pub use audio::{AudioConfig, MelFrontend};
pub use mlx_rs_core::{KVCache, ConcatKeyValueCache};

/// Environment variable name for model path.
pub const MODEL_PATH_ENV: &str = "QWEN3_ASR_MODEL_PATH";

/// Default model directory.
pub const DEFAULT_MODEL_DIR: &str = "qwen3-asr-1.7b";

/// Get the default model path.
///
/// Resolution order:
/// 1. `QWEN3_ASR_MODEL_PATH` environment variable
/// 2. `~/.OminiX/models/qwen3-asr-1.7b`
pub fn default_model_path() -> std::path::PathBuf {
    if let Ok(path) = std::env::var(MODEL_PATH_ENV) {
        return std::path::PathBuf::from(path);
    }

    if let Some(home) = dirs::home_dir() {
        return home.join(".OminiX").join("models").join(DEFAULT_MODEL_DIR);
    }

    std::path::PathBuf::from(".")
}

/// Load a Qwen3-ASR model from a directory.
pub fn load_model(model_dir: impl AsRef<std::path::Path>) -> Result<Qwen3ASR, Error> {
    Qwen3ASR::load(model_dir)
}
