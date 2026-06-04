use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("mlx: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
    #[error("mlx io: {0}")]
    MlxIo(#[from] mlx_rs::error::IoError),
    #[error("weight not found: {0}")]
    WeightNotFound(String),
    #[error("config: {0}")]
    Config(String),
    #[error("model: {0}")]
    Model(String),
}

pub type Result<T> = std::result::Result<T, Error>;
