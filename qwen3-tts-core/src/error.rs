use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error("weight not found: {0}")]
    WeightNotFound(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error("sampling error: {0}")]
    Sampling(String),
}

pub type Result<T> = std::result::Result<T, Error>;
