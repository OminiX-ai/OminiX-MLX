use std::path::PathBuf;
use std::time::Duration;

use mlx_rs::error::Exception;

/// Error types for mlx-rs-lm operations
#[derive(Debug, thiserror::Error)]
pub enum Error {
    // =================== MLX Errors ===================
    /// MLX framework exception (computation errors, shape mismatches, etc.)
    #[error(transparent)]
    Exception(#[from] Exception),

    // =================== IO Errors ===================
    /// Standard IO error
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// JSON deserialization error (config files, etc.)
    #[error(transparent)]
    Deserialize(#[from] serde_json::Error),

    /// Weight loading error from safetensors
    #[error(transparent)]
    LoadWeights(#[from] mlx_rs::error::IoError),

    // =================== Model Loading Errors ===================
    /// Model configuration file not found or invalid
    #[error("model config error: {0}")]
    ModelConfig(String),

    /// Model file not found at specified path
    #[error("model not found: {}", path.display())]
    ModelNotFound { path: PathBuf },

    /// Required weight tensor not found in model file
    #[error("weight not found: {name}")]
    WeightNotFound { name: String },

    /// Tensor shape mismatch during model loading or inference
    #[error("shape mismatch for {name}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<i32>,
        actual: Vec<i32>,
    },

    // =================== Tokenization Errors ===================
    /// Tokenizer loading or encoding error
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    // =================== Generation Errors ===================
    /// Invalid generation configuration
    #[error("generation config error: {0}")]
    GenerationConfig(String),

    /// Generation stopped due to max tokens or other limit
    #[error("generation stopped: {reason}")]
    GenerationStopped { reason: String },

    // =================== Input Validation Errors ===================
    /// Reference audio not set before synthesis
    #[error("reference audio not set - call set_reference_audio() first")]
    ReferenceNotSet,

    /// Invalid input provided
    #[error("invalid input: {reason}")]
    InvalidInput { reason: String },

    /// Text input is empty
    #[error("text input cannot be empty")]
    EmptyInput,

    /// Text input exceeds maximum length
    #[error("text too long: {len} chars (max {max})")]
    TextTooLong { len: usize, max: usize },

    // =================== Audio/TTS Errors ===================
    /// Audio processing error (WAV I/O, resampling, etc.)
    #[error("audio error: {0}")]
    Audio(String),

    /// Invalid audio format
    #[error("invalid audio format: {format} (expected {expected})")]
    InvalidAudioFormat { format: String, expected: String },

    /// Voice cloning specific error
    #[error("voice cloning error: {0}")]
    VoiceClone(String),

    // =================== File Format Errors ===================
    /// File is corrupted or has invalid format
    #[error("file corrupted: {path} - {reason}")]
    FileCorrupted { path: PathBuf, reason: String },

    // =================== Operation Errors ===================
    /// Operation timed out
    #[error("operation timed out after {elapsed:?}")]
    Timeout { elapsed: Duration },

    /// Operation was cancelled
    #[error("operation cancelled")]
    Cancelled,

    // =================== Generic Errors ===================
    /// Boxed error for interop with other error types
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),

    /// Simple message error (prefer more specific variants when possible)
    #[error("{0}")]
    Message(String),
}

impl Error {
    /// Create a model config error
    pub fn model_config(msg: impl Into<String>) -> Self {
        Self::ModelConfig(msg.into())
    }

    /// Create a weight not found error
    pub fn weight_not_found(name: impl Into<String>) -> Self {
        Self::WeightNotFound { name: name.into() }
    }

    /// Create a shape mismatch error
    pub fn shape_mismatch(name: impl Into<String>, expected: Vec<i32>, actual: Vec<i32>) -> Self {
        Self::ShapeMismatch {
            name: name.into(),
            expected,
            actual,
        }
    }

    /// Create a tokenizer error
    pub fn tokenizer(msg: impl Into<String>) -> Self {
        Self::Tokenizer(msg.into())
    }

    /// Create a generation config error
    pub fn generation_config(msg: impl Into<String>) -> Self {
        Self::GenerationConfig(msg.into())
    }

    /// Create an audio error
    pub fn audio(msg: impl Into<String>) -> Self {
        Self::Audio(msg.into())
    }

    /// Create a voice cloning error
    pub fn voice_clone(msg: impl Into<String>) -> Self {
        Self::VoiceClone(msg.into())
    }

    /// Create a model not found error
    pub fn model_not_found(path: impl Into<PathBuf>) -> Self {
        Self::ModelNotFound { path: path.into() }
    }

    /// Create an invalid input error
    pub fn invalid_input(reason: impl Into<String>) -> Self {
        Self::InvalidInput { reason: reason.into() }
    }

    /// Create a text too long error
    pub fn text_too_long(len: usize, max: usize) -> Self {
        Self::TextTooLong { len, max }
    }

    /// Create an invalid audio format error
    pub fn invalid_audio_format(format: impl Into<String>, expected: impl Into<String>) -> Self {
        Self::InvalidAudioFormat {
            format: format.into(),
            expected: expected.into()
        }
    }

    /// Create a file corrupted error
    pub fn file_corrupted(path: impl Into<PathBuf>, reason: impl Into<String>) -> Self {
        Self::FileCorrupted {
            path: path.into(),
            reason: reason.into()
        }
    }

    /// Create a timeout error
    pub fn timeout(elapsed: Duration) -> Self {
        Self::Timeout { elapsed }
    }
}
