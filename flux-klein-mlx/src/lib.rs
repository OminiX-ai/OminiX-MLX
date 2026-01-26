//! FLUX.2-klein Image Generation for MLX
//!
//! This crate provides a Rust implementation of the FLUX.2-klein image generation model
//! using the mlx-rs bindings to Apple's MLX framework.
//!
//! # Features
//!
//! - **FLUX.2-klein transformer**: 4B parameter model optimized for Apple Silicon
//! - **Qwen3-4B text encoder**: Shared with Z-Image-Turbo
//! - **VAE decoder**: AutoencoderKL for latent-to-image decoding
//! - **4-bit quantization**: Memory-efficient inference (~3GB vs ~8GB)
//!
//! # Example
//!
//! ```rust,ignore
//! use flux_klein_mlx::{FluxKlein, Qwen3TextEncoder, Decoder};
//!
//! // Load models
//! let text_encoder = Qwen3TextEncoder::new(config)?;
//! let transformer = FluxKlein::new(params)?;
//! let vae = Decoder::new(vae_config)?;
//!
//! // Generate image
//! let text_embed = text_encoder.forward(&input_ids)?;
//! let latents = transformer.forward(&latents, &t, &text_embed)?;
//! let image = vae.forward(&latents)?;
//! ```

pub mod autoencoder;
pub mod error;
pub mod klein_model;
pub mod klein_quantized;
pub mod layers;
pub mod qwen3_encoder;
pub mod sampler;
pub mod weights;

pub use autoencoder::{AutoEncoderConfig, Decoder, Encoder};
pub use error::{FluxError, Result};
pub use klein_model::{FluxKlein, FluxKleinParams};
pub use klein_quantized::QuantizedFluxKlein;
pub use qwen3_encoder::{Qwen3Config, Qwen3TextEncoder, sanitize_qwen3_weights};
pub use sampler::{FluxSampler, FluxSamplerConfig};
pub use weights::{load_safetensors, sanitize_flux2_klein_weights, sanitize_klein_model_weights, sanitize_vae_weights, sanitize_vae_encoder_weights};
