//! # qwen3-mlx
//!
//! Qwen model family inference on Apple Silicon with MLX.
//!
//! Supported models:
//! - Qwen2 (dense)
//! - Qwen3 (dense)
//! - Qwen3-MoE (Mixture of Experts)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use qwen3_mlx::{load_model, load_tokenizer, Generate, KVCache};
//! use mlx_rs::ops::indexing::NewAxis;
//!
//! let model_dir = "path/to/Qwen3-4B";
//! let tokenizer = load_tokenizer(model_dir)?;
//! let mut model = load_model(model_dir)?;
//!
//! let encoding = tokenizer.encode("Hello, ", true)?;
//! let prompt = mlx_rs::Array::from(encoding.get_ids()).index(NewAxis);
//! let mut cache = Vec::new();
//!
//! let generator = Generate::<KVCache>::new(&mut model, &mut cache, 0.7, &prompt);
//!
//! for token in generator.take(50) {
//!     let token = token?;
//!     let text = tokenizer.decode(&[token.item::<u32>()], true)?;
//!     print!("{}", text);
//! }
//! ```

pub mod model;
pub mod qwen2;
pub mod qwen3_moe;

// Re-export shared components from mlx-rs-core
pub use mlx_rs_core::{
    cache::{ConcatKeyValueCache, KVCache, KeyValueCache},
    error::{Error, Result},
    utils::{create_attention_mask, initialize_rope, scaled_dot_product_attention,
            AttentionMask, FloatOrString, SdpaMask},
    load_tokenizer,
};

pub use model::{
    load_model, get_model_args,
    Generate, GenerateState, Model, ModelArgs, ModelInput,
    Attention, AttentionInput, Mlp, TransformerBlock, Qwen3Model,
    sample,
};
