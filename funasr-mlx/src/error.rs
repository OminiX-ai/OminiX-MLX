//! Error types for funasr-mlx

use mlx_rs::error::Exception;
use thiserror::Error;

/// Error type for FunASR operations
#[derive(Debug, Error)]
pub enum Error {
    /// MLX operation failed
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),

    /// IO operation failed
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Model loading failed
    #[error("Model error: {0}")]
    Model(String),

    /// Audio processing failed
    #[error("Audio error: {0}")]
    Audio(String),

    /// Configuration error
    #[error("Config error: {0}")]
    Config(String),
}

/// Result type for FunASR operations
pub type Result<T> = std::result::Result<T, Error>;
