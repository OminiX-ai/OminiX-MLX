//! Common constants for Qwen-Image model
//!
//! These constants are used across the transformer, VAE, and pipeline components.

/// Timestep embedding dimension
pub const TIMESTEP_EMBED_DIM: i32 = 256;

/// Timestep scale factor (multiply sigma by this before sinusoidal encoding)
pub const TIMESTEP_SCALE: f32 = 1000.0;

/// RoPE theta parameter (base frequency)
pub const ROPE_THETA: f32 = 10000.0;

/// LayerNorm/RMSNorm epsilon for numerical stability
pub const LAYER_NORM_EPS: f32 = 1e-6;

/// Qwen VL template prefix token count
/// The template "<|im_start|>system\nDescribe the image..." adds 34 tokens
pub const QWEN_TEMPLATE_PREFIX_TOKENS: usize = 34;

/// Max text tokens after dropping template prefix
pub const MAX_TEXT_OUTPUT_TOKENS: usize = 77;

/// Max text input tokens (output + prefix)
pub const MAX_TEXT_INPUT_TOKENS: usize = MAX_TEXT_OUTPUT_TOKENS + QWEN_TEMPLATE_PREFIX_TOKENS;

/// RoPE lookup table maximum index
pub const ROPE_MAX_INDEX: i32 = 4096;

/// Patch embedding dimension (latent channels per patch)
pub const PATCH_EMBEDDING_DIM: i32 = 64;

/// RoPE axes dimensions [frames, height, width]
pub const ROPE_AXES_DIM: [i32; 3] = [16, 56, 56];

/// VAE spatial downsample factor (image pixels to latent)
pub const VAE_SPATIAL_DOWNSAMPLE: i32 = 16;

/// Minimum image dimension (in pixels)
pub const MIN_IMAGE_SIZE: i32 = 256;

/// Maximum image dimension (in pixels)
pub const MAX_IMAGE_SIZE: i32 = 2048;

/// Default CFG guidance scale
pub const DEFAULT_CFG_SCALE: f32 = 4.0;

/// Default number of diffusion steps
pub const DEFAULT_NUM_STEPS: i32 = 20;
