//! # glm47-flash-mlx
//!
//! GLM-4.7-Flash (MoE + MLA) LLM inference on Apple Silicon with MLX.
//!
//! ## Features
//!
//! - Multi-head Latent Attention (MLA) with absorbed KV projections
//! - Compressed KV cache (576 floats/token/layer, ~18x smaller than standard MHA)
//! - Mixture of Experts with top-k routing (shared + routed experts)
//! - Custom fused SwiGLU Metal kernel
//! - 6-bit quantization support
//!
//! ## Architecture
//!
//! GLM-4.7-Flash uses DeepSeek-V2 style MLA where `kv_b_proj` is absorbed into:
//! - `embed_q`: per-head projection on query side (QuantizedMultiLinear)
//! - `unembed_out`: per-head projection on output side (QuantizedMultiLinear)
//!
//! This eliminates the need to store full key/value projections, requiring only
//! a compressed latent (512 dims) plus RoPE keys (64 dims) = 576 dims per token.

pub mod model;

// Re-export shared components from mlx-rs-core
pub use mlx_rs_core::{
    cache::{ConcatKeyValueCache, KVCache, KeyValueCache},
    error::{Error, Result},
    fused_swiglu,
    utils::{create_attention_mask, scaled_dot_product_attention,
            AttentionMask, SdpaMask},
};

pub use model::{
    load_model, load_tokenizer, get_model_args, init_cache,
    Generate, GenerateState, Model, ModelArgs, ModelInput,
    MLAttention, AttentionInput, MLP, MoE, MoEGate, SwitchGLU, DecoderLayer, LanguageModel,
    QuantizedMultiLinear, sample,
};
