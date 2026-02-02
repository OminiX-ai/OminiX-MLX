//! # mlx-rs-core
//!
//! Shared inference infrastructure for MLX Rust crates.
//!
//! This crate provides common infrastructure used by all model-specific
//! crates (qwen3-mlx, glm4-mlx, gpt-sovits-mlx, zimage-mlx, etc.):
//!
//! - **Cache**: KV cache implementations for fast autoregressive decoding
//! - **Utils**: RoPE, attention masks, scaled dot-product attention
//! - **Metal Kernels**: Custom Metal kernels (fused_swiglu for MoE, fused_modulate for DiT)
//! - **Error**: Common error types and conversions
//! - **Sampler**: Token sampling strategies
//! - **Generate**: Generic token generation infrastructure
//! - **Audio**: Audio processing utilities (mel spectrograms, etc.)
//! - **Speculative**: Speculative decoding support
//! - **Convert**: Model conversion utilities (optional, requires `convert` feature)

pub mod audio;
#[cfg(feature = "convert")]
pub mod convert;
pub mod cache;
pub mod error;
pub mod generate;
pub mod metal_kernels;
pub mod sampler;
pub mod speculative;
pub mod utils;

pub use cache::{ConcatKeyValueCache, KVCache, KeyValueCache};
pub use error::{Error, Result};
pub use metal_kernels::{fused_swiglu, fused_modulate};
pub use sampler::{DefaultSampler, Sampler};
pub use utils::{
    create_attention_mask, initialize_rope, scaled_dot_product_attention,
    AttentionMask, FloatOrString, SdpaMask,
};

// The try_unwrap macro is exported via #[macro_export] in utils.rs

// Re-export tokenizers for convenience
pub use tokenizers::Tokenizer;

use mlx_rs::Array;

// ============================================================================
// Model Input/Output Traits for Generic Generation
// ============================================================================

/// Builder struct for creating model inputs
pub struct ModelInputBuilder<'a, C, T> {
    pub y: &'a Array,
    pub cache: &'a mut Vec<Option<C>>,
    pub state: &'a mut T,
}

/// Trait for model input types that can be constructed from a builder
pub trait ModelInput<'a, C, T> {
    fn from_model_input_builder(builder: ModelInputBuilder<'a, C, T>) -> Self;
}

/// Trait for model output types that provide logits
pub trait ModelOutput {
    fn logits(&self) -> &Array;
}

impl ModelOutput for Array {
    fn logits(&self) -> &Array {
        self
    }
}

/// Load a tokenizer from a model directory.
pub fn load_tokenizer(model_dir: impl AsRef<std::path::Path>) -> Result<Tokenizer> {
    let tokenizer_path = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(&tokenizer_path).map_err(|e| Error::Tokenizer(e.to_string()))
}
