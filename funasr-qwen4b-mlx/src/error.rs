//! Error types for funasr-qwen4b-mlx.

use std::path::PathBuf;
use thiserror::Error;

/// Error type for FunASR-Qwen4B operations.
#[derive(Debug, Error)]
pub enum Error {
    /// MLX operation error
    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Tokenizer error
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// Audio file not found
    #[error("Audio file not found: {path}")]
    AudioFileNotFound { path: PathBuf },

    /// Audio too short for processing
    #[error("Audio too short: {duration_ms}ms (minimum: {min_ms}ms)")]
    AudioTooShort { duration_ms: u64, min_ms: u64 },

    /// Audio format error
    #[error("Invalid audio format: {message}")]
    AudioFormat { message: String },

    /// Audio processing error (generic)
    #[error("Audio error: {0}")]
    Audio(String),

    /// Model file not found
    #[error("Model file not found: {path}")]
    ModelFileNotFound { path: PathBuf },

    /// Model loading error
    #[error("Model loading error: {0}")]
    ModelLoad(String),

    /// Configuration dimension mismatch
    #[error("Dimension mismatch in {component}: expected {expected}, got {actual}")]
    DimensionMismatch {
        component: &'static str,
        expected: i32,
        actual: i32,
    },

    /// Configuration validation error
    #[error("Invalid configuration for {field}: {message}")]
    ConfigValidation {
        field: &'static str,
        message: String,
    },

    /// Configuration error (generic)
    #[error("Configuration error: {0}")]
    Config(String),

    /// Weight loading error
    #[error("Weight error: {0}")]
    Weight(String),

    /// Missing weight key
    #[error("Missing weight key: {key}")]
    MissingWeight { key: String },

    /// Shape mismatch in weights
    #[error("Weight shape mismatch for {key}: expected {expected:?}, got {actual:?}")]
    WeightShapeMismatch {
        key: String,
        expected: Vec<i32>,
        actual: Vec<i32>,
    },

    /// Inference error
    #[error("Inference error: {0}")]
    Inference(String),

    /// Streaming error
    #[error("Streaming error: {0}")]
    Streaming(String),
}

/// Result type alias for FunASR-Qwen4B operations.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Create an audio file not found error.
    pub fn audio_not_found(path: impl Into<PathBuf>) -> Self {
        Self::AudioFileNotFound { path: path.into() }
    }

    /// Create an audio too short error.
    pub fn audio_too_short(duration_ms: u64, min_ms: u64) -> Self {
        Self::AudioTooShort { duration_ms, min_ms }
    }

    /// Create a dimension mismatch error.
    pub fn dimension_mismatch(component: &'static str, expected: i32, actual: i32) -> Self {
        Self::DimensionMismatch {
            component,
            expected,
            actual,
        }
    }

    /// Create a config validation error.
    pub fn config_validation(field: &'static str, message: impl Into<String>) -> Self {
        Self::ConfigValidation {
            field,
            message: message.into(),
        }
    }
}
