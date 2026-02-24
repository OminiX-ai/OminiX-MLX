//! Error types for qwen3-asr-mlx.

use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Audio file not found: {path}")]
    AudioFileNotFound { path: PathBuf },

    #[error("Audio too short: {duration_ms}ms (minimum: {min_ms}ms)")]
    AudioTooShort { duration_ms: u64, min_ms: u64 },

    #[error("Invalid audio format: {message}")]
    AudioFormat { message: String },

    #[error("Audio error: {0}")]
    Audio(String),

    #[error("Model file not found: {path}")]
    ModelFileNotFound { path: PathBuf },

    #[error("Model loading error: {0}")]
    ModelLoad(String),

    #[error("Dimension mismatch in {component}: expected {expected}, got {actual}")]
    DimensionMismatch {
        component: &'static str,
        expected: i32,
        actual: i32,
    },

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Weight error: {0}")]
    Weight(String),

    #[error("Inference error: {0}")]
    Inference(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn audio_not_found(path: impl Into<PathBuf>) -> Self {
        Self::AudioFileNotFound { path: path.into() }
    }

    pub fn audio_too_short(duration_ms: u64, min_ms: u64) -> Self {
        Self::AudioTooShort { duration_ms, min_ms }
    }
}
