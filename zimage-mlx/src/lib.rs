//! Z-Image-Turbo Image Generation for MLX
//!
//! This crate provides a Rust implementation of the Z-Image-Turbo image generation model
//! using the mlx-rs bindings to Apple's MLX framework.
//!
//! # Features
//!
//! - **Z-Image-Turbo transformer**: 6B parameter Single-Stream DiT (S3-DiT)
//! - **9-step Turbo inference**: Distilled for fast generation
//! - **4-bit quantization**: Memory-efficient inference (~3GB vs ~12GB)
//! - **3-axis RoPE**: Optimized position encoding
//!
//! # Architecture
//!
//! Z-Image-Turbo differs from FLUX.2-klein:
//! - Uses Noise Refiner + Context Refiner + Joint blocks (vs Double + Single)
//! - 3-axis RoPE [32, 48, 48] with theta=256 (vs 4-axis)
//! - Per-block AdaLN with tanh gates
//! - Qwen3-4B layer 34 extraction (vs concat layers 8, 17, 26)
//!
//! # Quantization
//!
//! Two loading modes are supported:
//!
//! | Mode | Memory | Speed |
//! |------|--------|-------|
//! | Dequantized (f32) | ~12GB | ~1.87s/step |
//! | Quantized (4-bit) | ~3GB | ~2.08s/step |
//!
//! # Example
//!
//! ```rust,ignore
//! use zimage_mlx::{ZImageTransformer, ZImageConfig, load_quantized_zimage_transformer};
//! use flux_klein_mlx::{Qwen3TextEncoder, Decoder};
//!
//! // Load models
//! let text_encoder = Qwen3TextEncoder::new(config)?;
//! let transformer = ZImageTransformer::new(ZImageConfig::default())?;
//! let vae = Decoder::new(vae_config)?;
//!
//! // Or use quantized transformer for lower memory
//! let transformer = load_quantized_zimage_transformer(weights, config)?;
//! ```

mod zimage_model;
mod zimage_model_quantized;
mod qwen3_quantized;

pub use zimage_model::{
    ZImageConfig,
    ZImageTransformer,
    ZImageTransformerBlock,
    create_coordinate_grid,
    compute_rope_3axis,
    apply_rope_3axis,
    sanitize_mlx_weights,
    sanitize_zimage_weights,
};

pub use zimage_model_quantized::{
    ZImageTransformerQuantized,
    load_quantized_zimage_transformer,
};

pub use qwen3_quantized::{
    QuantizedQwen3TextEncoder,
    QuantizedQwen3Attention,
    QuantizedQwen3Mlp,
    QuantizedQwen3Block,
    sanitize_quantized_qwen3_weights,
    load_quantized_qwen3_encoder,
};

// Re-export shared components from mlx-flux-klein
pub use flux_klein_mlx::{
    Qwen3Config, Qwen3TextEncoder, sanitize_qwen3_weights,
    Decoder, AutoEncoderConfig,
    load_safetensors, sanitize_vae_weights,
    FluxSampler, FluxSamplerConfig,
};
