//! # mixtral-mlx
//!
//! Mixtral MoE (Mixture of Experts) LLM inference on Apple Silicon with MLX.
//!
//! ## Features
//!
//! - 8 experts with top-2 routing per token
//! - Custom fused SwiGLU Metal kernel (10-12x faster)
//! - Optimized gather_qmm for expert dispatch
//! - Support for 4-bit quantized models
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use mixtral_mlx::{load_model, load_tokenizer, Generate, KVCache};
//! use mlx_rs::ops::indexing::NewAxis;
//!
//! let mut model = load_model("path/to/Mixtral-8x7B-4bit")?;
//! let tokenizer = load_tokenizer("path/to/Mixtral-8x7B-4bit")?;
//!
//! let encoding = tokenizer.encode("Hello, ", true)?;
//! let prompt = mlx_rs::Array::from(encoding.get_ids()).index(NewAxis);
//! let mut cache = Vec::new();
//!
//! let generator = Generate::<KVCache>::new(&mut model, &mut cache, 0.7, &prompt);
//!
//! for token in generator.take(50) {
//!     let token = token?;
//!     print!("{}", tokenizer.decode(&[token.item::<u32>()], true)?);
//! }
//! ```

pub mod model;

// Re-export shared components from mlx-rs-core
pub use mlx_rs_core::{
    cache::{ConcatKeyValueCache, KVCache, KeyValueCache},
    error::{Error, Result},
    fused_swiglu,  // Custom Metal kernel from shared crate
    utils::{create_attention_mask, scaled_dot_product_attention,
            AttentionMask, SdpaMask},
    load_tokenizer,
};

pub use model::{
    load_model, get_model_args,
    Generate, GenerateState, Model, ModelArgs, ModelInput,
    Attention, AttentionInput, MixtralSparseMoeBlock, SwitchGLU, DecoderLayer, MixtralModel,
    sample,
};
