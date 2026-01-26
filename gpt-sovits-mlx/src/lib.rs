//! GPT-SoVITS Voice Cloning
//!
//! Pure Rust implementation of GPT-SoVITS with MLX acceleration for Apple Silicon.
//!
//! # Features
//!
//! - **Few-shot voice cloning**: Clone any voice with just a few seconds of reference audio
//! - **Mixed Chinese-English**: Natural handling of mixed language text
//! - **High performance**: 4x realtime synthesis on Apple Silicon
//! - **Pure Rust**: No Python dependencies at runtime
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use gpt_sovits_mlx::VoiceCloner;
//!
//! let mut cloner = VoiceCloner::with_defaults()?;
//! cloner.set_reference_audio("/path/to/reference.wav")?;
//! let audio = cloner.synthesize("你好，世界！")?;
//! cloner.play(&audio)?;
//! ```
//!
//! # Performance
//!
//! On Apple Silicon (M-series):
//! - Model loading: ~50ms
//! - Synthesis: ~4x realtime (generates 20s audio in 5s)
//! - Memory: ~2GB for all models

pub mod audio;
pub mod cache;
pub mod error;
pub mod inference;
pub mod models;
pub mod text;
pub mod voice_clone;

// Re-export main types
pub use voice_clone::{VoiceCloner, VoiceClonerConfig, AudioOutput};
pub use text::{Language, preprocess_text};
pub use error::Error;
