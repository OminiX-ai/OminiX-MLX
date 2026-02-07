use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    #[error("MLX IO error: {0}")]
    MlxIo(#[from] mlx_rs::error::IoError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Weight not found: {0}")]
    WeightNotFound(String),

    #[error("Invalid config: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, Error>;
