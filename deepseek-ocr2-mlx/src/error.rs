use mlx_rs::error::Exception;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("IO error: {0}")]
    MlxIo(#[from] mlx_rs::error::IoError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Model error: {0}")]
    Model(String),
}

impl From<tokenizers::Error> for Error {
    fn from(e: tokenizers::Error) -> Self {
        Error::Tokenizer(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
