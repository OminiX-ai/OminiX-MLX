//! Backend-agnostic core logic for Qwen3-TTS.
//!
//! This crate provides shared types, configuration, generation logic,
//! and sampling that work with any compute backend (MLX, GGML/Ascend, etc.).
//!
//! Backend-specific code (transformer forward pass, speech decoder, speaker encoder)
//! is defined via traits in [`backend`] and implemented by backend crates.

pub mod backend;
pub mod codec_prefix;
pub mod config;
pub mod error;
pub mod generate;
pub mod sampling;
pub mod text;

pub use config::{
    CodePredictorConfig, DecoderConfig, GenerationConfig, ModelType, Qwen3TtsConfig,
    QuantizationConfig, RopeScalingConfig, SpeakerEncoderJsonConfig, SpeechTokenizerConfig,
    TalkerConfig,
};
pub use error::{Error, Result};
