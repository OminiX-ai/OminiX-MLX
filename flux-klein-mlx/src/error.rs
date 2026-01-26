//! Error types for mlx-flux-klein

use thiserror::Error;

#[derive(Error, Debug)]
pub enum FluxError {
    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Invalid shape: expected {expected}, got {got}")]
    InvalidShape { expected: String, got: String },

    #[error("Weight loading error: {0}")]
    WeightLoading(String),
}

pub type Result<T> = std::result::Result<T, FluxError>;
