//! Qwen-Image-2512 implementation in Rust using MLX
//!
//! This crate provides a Rust implementation of the Qwen-Image text-to-image model.
//!
//! ## License
//!
//! Apache 2.0 - All code derived from:
//! - HuggingFace Diffusers (Apache 2.0)
//! - QwenLM/Qwen-Image (Apache 2.0)

pub mod error;
pub mod vae;
pub mod transformer;
pub mod pipeline;
pub mod weights;
pub mod qwen_quantized;
pub mod qwen_full_precision;
pub mod text_encoder;

pub use error::QwenImageError;
pub use vae::{QwenVAE, load_vae_from_dir};
pub use transformer::{QwenTransformer, QwenTransformerConfig};
pub use pipeline::{QwenImagePipeline, FlowMatchEulerScheduler, pack_latents, unpack_latents, encode_reference_latent, ref_shape_from_latent};
pub use qwen_quantized::{QwenQuantizedTransformer, QwenConfig, load_transformer_weights, build_edit_rope};
pub use text_encoder::{QwenTextEncoder, TextEncoderConfig, load_text_encoder};
