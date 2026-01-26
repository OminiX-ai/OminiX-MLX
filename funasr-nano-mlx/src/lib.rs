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
//! ## Example
//!
//! ```rust,ignore
//! use funasr_nano_mlx::{FunASRNano, load_model};
//!
//! let mut model = load_model("path/to/model")?;
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
pub use model::{FunASRNano, FunASRNanoConfig, SamplingConfig, StreamingContext};
pub use audio::{AudioConfig, MelFrontend};

// Re-export from mlx-rs-core
pub use mlx_rs_core::{KVCache, ConcatKeyValueCache};

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
