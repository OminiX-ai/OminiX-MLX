//! Error types for Qwen-Image

use mlx_rs::error::Exception;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum QwenImageError {
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),

    #[error("Weight loading error: {0}")]
    WeightLoadError(String),

    #[error("Invalid shape: expected {expected}, got {got}")]
    InvalidShape { expected: String, got: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
}

pub type Result<T> = std::result::Result<T, QwenImageError>;
