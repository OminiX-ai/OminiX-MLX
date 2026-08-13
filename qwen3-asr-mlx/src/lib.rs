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

/// Crate version (from Cargo.toml).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

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

/// Whether the linked MLX build can decode a batch correctly.
///
/// MLX before 0.32.0 mis-handles RoPE when the sequence length is 1 and the
/// batch is larger than 1: rows after the first come back wrong, frequently all
/// zeros. Verified broken on 0.30.1 (the version this workspace pins), 0.30.3,
/// 0.30.6, 0.31.0, 0.31.1 and 0.31.2; correct on 0.32.0.
///
/// That is precisely the shape every batched decode step uses, so on an affected
/// build [`Qwen3ASR::transcribe_batch`] would return fluent-looking but wrong
/// transcripts for all but the first item — a silent corruption rather than an
/// error. Callers should probe once at startup and fall back to single-request
/// decoding when this returns `false`.
///
/// The probe runs RoPE over two identical rows at sequence length 1 and checks
/// they come back identical, which is the exact defect rather than a version
/// string comparison — so a patched or vendored build is judged on behaviour.
pub fn batched_decode_supported() -> bool {
    fn probe() -> std::result::Result<bool, mlx_rs::error::Exception> {
        use mlx_rs::builder::Builder;
        use mlx_rs::module::Module;
        use mlx_rs::ops::indexing::IndexOp;

        const DIMS: i32 = 8;
        let mut rope = mlx_rs_core::initialize_rope(DIMS, 10000.0, false, &None, 4096)?;

        // Two identical rows, one position: [batch 2, heads 1, seq 1, dims].
        let ones = [1.0f32; (2 * DIMS) as usize];
        let x = mlx_rs::Array::from_slice(&ones, &[2, 1, 1, DIMS]);

        let out = rope.forward(
            mlx_rs::nn::RopeInputBuilder::new(&x).offset(3).build()?,
        )?;

        let diff = out.index(0).subtract(out.index(1))?.abs()?.max(None)?;
        mlx_rs::transforms::eval([&diff])?;
        Ok(diff.item::<f32>() == 0.0)
    }

    probe().unwrap_or(false)
}
