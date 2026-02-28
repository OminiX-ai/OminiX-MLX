use mlx_rs::error::Exception;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Mlx(#[from] Exception),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error("weight load error: {0}")]
    WeightLoad(#[from] mlx_rs::error::IoError),

    #[error("weight not found: {0}")]
    WeightNotFound(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("config error: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, Error>;
