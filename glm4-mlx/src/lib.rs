//! # glm4-mlx
//!
//! GLM-4 LLM inference on Apple Silicon with MLX.
//!
//! ## Features
//!
//! - Partial RoPE (rotary position embedding on half of head dimensions)
//! - Fused gate_up_proj in MLP for efficiency
//! - Extra LayerNorms (post_self_attn, post_mlp)
//! - Support for quantized (4-bit) models
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use glm4_mlx::{load_model, load_tokenizer, Generate, KVCache};
//! use mlx_rs::ops::indexing::NewAxis;
//!
//! let mut model = load_model("path/to/GLM-4-9B")?;
//! let tokenizer = load_tokenizer("path/to/GLM-4-9B")?;
//!
//! let encoding = tokenizer.encode("你好", true)?;
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
    utils::{create_attention_mask, scaled_dot_product_attention,
            AttentionMask, SdpaMask},
    load_tokenizer,
};

pub use model::{
    load_model, get_model_args,
    Generate, GenerateState, Model, ModelArgs, ModelInput,
    Glm4Attention, AttentionInput, Glm4Mlp, Glm4DecoderLayer, Glm4Model,
    sample,
};
