//! FunASR-Qwen4B-MLX: LLM-based ASR with SenseVoice encoder + Qwen3-4B
//!
//! This crate provides speech recognition using:
//! - SenseVoice encoder (70 layers, 512-dim output)
//! - 4-layer transformer adaptor (512 → 2560)
//! - Qwen3-4B LLM (2560-dim, 36 layers)
//!
//! Trained on AISHELL-1 dataset with contrastive + cross-entropy loss.

pub mod audio;
pub mod error;
pub mod sensevoice_encoder;
pub mod adaptor;
pub mod model;

pub use error::{Error, Result};
pub use model::FunASRQwen4B;

// Re-export audio utilities
pub use audio::{load_wav, resample, AudioConfig, MelFrontend, MelFrontendMLX, compute_mel_spectrogram_mlx};

// Re-export qwen3-mlx types for direct access
pub use qwen3_mlx::{
    Model as Qwen3Model,
    load_model as load_qwen3,
    load_tokenizer as load_qwen3_tokenizer,
    Generate, KVCache,
};
