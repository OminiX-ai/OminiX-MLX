//! Error types for Step-Audio 2

use thiserror::Error;

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;

/// Step-Audio 2 error types
#[derive(Error, Debug)]
pub enum Error {
    /// Model loading error
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    /// Weight loading error
    #[error("Failed to load weights: {0}")]
    WeightLoad(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Audio processing error
    #[error("Audio processing error: {0}")]
    Audio(String),

    /// Tokenizer error
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// Inference error
    #[error("Inference error: {0}")]
    Inference(String),

    /// TTS decoder error
    #[error("TTS decoder error: {0}")]
    TTS(String),

    /// Tool calling error
    #[error("Tool error: {0}")]
    Tool(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// MLX error
    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    /// MLX IO error (from safetensors loading)
    #[error("MLX IO error: {0}")]
    MlxIo(#[from] mlx_rs::error::IoError),

    /// JSON error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// YAML error
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),
}
