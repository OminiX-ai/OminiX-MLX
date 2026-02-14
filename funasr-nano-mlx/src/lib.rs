//! # funasr-nano-mlx
//!
//! Fun-ASR-Nano speech recognition on Apple Silicon using MLX.
//!
//! ## Architecture
//!
//! Fun-ASR-Nano-2512 combines:
//! - **SenseVoice Encoder**: Extracts audio features using SAN-M attention (70 layers)
//! - **Audio Adaptor**: 2-layer transformer projecting encoder output to LLM dimension
//! - **Qwen3-0.6B LLM**: Generates text autoregressively (28 layers)
//!
//! Total: ~985M parameters
//!
//! ## Model Path
//!
//! The model path can be configured via:
//! 1. Environment variable `FUNASR_NANO_MODEL_PATH`
//! 2. Default location: `~/.OminiX/models/funasr-nano`
//!
//! ## Example
//!
//! ```rust,ignore
//! use funasr_nano_mlx::{FunASRNano, load_model, default_model_path};
//!
//! // Load from default path or FUNASR_NANO_MODEL_PATH env var
//! let mut model = load_model(default_model_path())?;
//! let text = model.transcribe("audio.wav")?;
//! ```

pub mod audio;
pub mod error;
pub mod sensevoice_encoder;
pub mod adaptor;
pub mod qwen;
pub mod model;

// Keep whisper_encoder for reference but deprecated
#[deprecated(note = "Use sensevoice_encoder instead")]
pub mod whisper_encoder;

// Re-exports
pub use error::Error;
pub use model::{FunASRNano, FunASRNanoConfig, SamplingConfig, StreamingContext, TaskPrompt};
pub use audio::{AudioConfig, MelFrontend};

// Re-export from mlx-rs-core
pub use mlx_rs_core::{KVCache, ConcatKeyValueCache};

/// Environment variable name for model path
pub const MODEL_PATH_ENV: &str = "FUNASR_NANO_MODEL_PATH";

/// Default model directory under ~/.OminiX/models
pub const DEFAULT_MODEL_DIR: &str = "funasr-nano";

/// Get the default model path.
///
/// Resolution order:
/// 1. `FUNASR_NANO_MODEL_PATH` environment variable
/// 2. `~/.OminiX/models/funasr-nano`
///
/// # Example
///
/// ```rust,ignore
/// use funasr_nano_mlx::{load_model, default_model_path};
///
/// let model = load_model(default_model_path())?;
/// ```
pub fn default_model_path() -> std::path::PathBuf {
    // Check environment variable first
    if let Ok(path) = std::env::var(MODEL_PATH_ENV) {
        return std::path::PathBuf::from(path);
    }

    // Fall back to ~/.OminiX/models/funasr-nano
    if let Some(home) = dirs::home_dir() {
        return home.join(".OminiX").join("models").join(DEFAULT_MODEL_DIR);
    }

    // Last resort: current directory
    std::path::PathBuf::from(".")
}

/// Load a Fun-ASR-Nano model from a directory.
///
/// The directory should contain:
/// - `config.yaml` - Model configuration
/// - `model.pt` - Model weights (PyTorch format)
/// - `Qwen3-0.6B/tokenizer.json` - Tokenizer
pub fn load_model(model_dir: impl AsRef<std::path::Path>) -> Result<FunASRNano, Error> {
    FunASRNano::load(model_dir)
}

/// Load tokenizer from model directory.
pub fn load_tokenizer(
    model_dir: impl AsRef<std::path::Path>,
) -> Result<tokenizers::Tokenizer, Error> {
    // Try Qwen3-0.6B subdirectory first
    let qwen_path = model_dir.as_ref().join("Qwen3-0.6B/tokenizer.json");
    if qwen_path.exists() {
        return tokenizers::Tokenizer::from_file(qwen_path)
            .map_err(|e| Error::Tokenizer(e.to_string()));
    }

    // Fall back to root tokenizer.json
    let path = model_dir.as_ref().join("tokenizer.json");
    tokenizers::Tokenizer::from_file(path).map_err(|e| Error::Tokenizer(e.to_string()))
}
