//! VAE 3D components for Qwen-Image
//!
//! Reference: diffusers/models/autoencoders/autoencoder_kl_qwenimage.py

mod norm;
mod conv3d;
mod resample;
mod attention;
mod blocks;
mod vae;
mod weights;

pub use norm::QwenImageRMSNorm;
pub use conv3d::QwenImageCausalConv3D;
pub use resample::{QwenImageResample3D, ResampleMode};
pub use attention::QwenImageAttentionBlock3D;
pub use blocks::{QwenImageResBlock3D, QwenImageMidBlock3D, QwenImageUpBlock3D, QwenImageDownBlock3D};
pub use vae::{QwenVAE, QwenImageEncoder3D, QwenImageDecoder3D, LATENTS_MEAN, LATENTS_STD};
pub use weights::{load_vae_weights, load_vae_from_dir};
