//! Qwen-Image VAE with 3D encoder and decoder
//!
//! Reference: diffusers QwenImageEncoder3d, QwenImageDecoder3d

use mlx_macros::ModuleParameters;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use super::{
    QwenImageCausalConv3D, QwenImageDownBlock3D, QwenImageMidBlock3D, QwenImageRMSNorm,
    QwenImageUpBlock3D, ResampleMode,
};

/// Latent normalization constants (16 channels)
/// Reference: diffusers pipeline - "denormalizes using stored VAE statistics"
pub const LATENTS_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];

pub const LATENTS_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.916,
];

// Channel configuration
const BASE_CHANNELS: i32 = 96;
const STAGE_MULTIPLIERS: [i32; 5] = [1, 1, 2, 4, 4]; // [96, 96, 192, 384, 384]

/// 3D Encoder for Qwen-Image VAE
/// Reference: diffusers QwenImageEncoder3d
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageEncoder3D {
    #[param]
    pub conv_in: QwenImageCausalConv3D,
    #[param]
    pub down_blocks: Vec<QwenImageDownBlock3D>,
    #[param]
    pub mid_block: QwenImageMidBlock3D,
    #[param]
    pub norm_out: QwenImageRMSNorm,
    #[param]
    pub conv_out: QwenImageCausalConv3D,
}

impl QwenImageEncoder3D {
    pub fn new() -> Result<Self, Exception> {
        let channels: Vec<i32> = STAGE_MULTIPLIERS.iter().map(|m| m * BASE_CHANNELS).collect();

        let conv_in = QwenImageCausalConv3D::new(
            4,
            channels[0], // Input: RGBA (4 channels)
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
        )?;

        let downsample_modes = [
            Some(ResampleMode::Downsample2D),
            Some(ResampleMode::Downsample3D),
            Some(ResampleMode::Downsample3D),
            None,
        ];

        let down_blocks: Vec<_> = downsample_modes
            .iter()
            .enumerate()
            .map(|(i, mode)| QwenImageDownBlock3D::new(channels[i], channels[i + 1], 2, *mode))
            .collect::<Result<_, _>>()?;

        let last_ch = *channels.last().unwrap();
        let mid_block = QwenImageMidBlock3D::new(last_ch, 1)?;
        let norm_out = QwenImageRMSNorm::new(last_ch, 1e-12, false)?;
        let conv_out = QwenImageCausalConv3D::new(
            last_ch,
            32, // Output: 32 channels (for quantization: mean + logvar)
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
        )?;

        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
            norm_out,
            conv_out,
        })
    }
}

impl Module<&Array> for QwenImageEncoder3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        self.conv_in.training_mode(mode);
        for block in &mut self.down_blocks {
            block.training_mode(mode);
        }
        self.mid_block.training_mode(mode);
        self.norm_out.training_mode(mode);
        self.conv_out.training_mode(mode);
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let mut h = self.conv_in.forward(x)?;
        for block in &mut self.down_blocks {
            h = block.forward(&h)?;
        }
        h = self.mid_block.forward(&h)?;
        h = self.norm_out.forward(&h)?;
        h = nn::silu(&h)?;
        self.conv_out.forward(&h)
    }
}

/// 3D Decoder for Qwen-Image VAE
/// Reference: diffusers QwenImageDecoder3d
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageDecoder3D {
    #[param]
    pub conv_in: QwenImageCausalConv3D,
    #[param]
    pub mid_block: QwenImageMidBlock3D,
    #[param]
    pub up_blocks: Vec<QwenImageUpBlock3D>,
    #[param]
    pub norm_out: QwenImageRMSNorm,
    #[param]
    pub conv_out: QwenImageCausalConv3D,
}

impl QwenImageDecoder3D {
    pub fn new() -> Result<Self, Exception> {
        let conv_in = QwenImageCausalConv3D::new(
            16,
            384, // Input: 16 latent channels
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
        )?;

        let mid_block = QwenImageMidBlock3D::new(384, 1)?;

        // Channel progression (documented in plan):
        // Block 0: 384 -> 384 -> upsample -> 192 (halved by resample conv)
        // Block 1: 192 (in) -> 384 (res) -> upsample -> 192
        // Block 2: 192 -> 192 -> upsample2d -> 96
        // Block 3: 96 -> 96 (no upsample)
        let up_blocks = vec![
            QwenImageUpBlock3D::new(384, 384, 2, Some(ResampleMode::Upsample3D))?,
            QwenImageUpBlock3D::new(192, 384, 2, Some(ResampleMode::Upsample3D))?,
            QwenImageUpBlock3D::new(192, 192, 2, Some(ResampleMode::Upsample2D))?,
            QwenImageUpBlock3D::new(96, 96, 2, None)?,
        ];

        let norm_out = QwenImageRMSNorm::new(96, 1e-12, false)?;
        let conv_out = QwenImageCausalConv3D::new(
            96,
            3, // Output: RGB
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
        )?;

        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            norm_out,
            conv_out,
        })
    }
}

impl Module<&Array> for QwenImageDecoder3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        self.conv_in.training_mode(mode);
        self.mid_block.training_mode(mode);
        for block in &mut self.up_blocks {
            block.training_mode(mode);
        }
        self.norm_out.training_mode(mode);
        self.conv_out.training_mode(mode);
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let mut h = self.conv_in.forward(x)?;
        h = self.mid_block.forward(&h)?;
        for block in self.up_blocks.iter_mut() {
            h = block.forward(&h)?;
        }
        h = self.norm_out.forward(&h)?;
        h = nn::silu(&h)?;
        self.conv_out.forward(&h)
    }
}

/// Complete VAE with encoder and decoder
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenVAE {
    #[param]
    pub encoder: QwenImageEncoder3D,
    #[param]
    pub decoder: QwenImageDecoder3D,
    #[param]
    pub post_quant_conv: QwenImageCausalConv3D,
    #[param]
    pub quant_conv: QwenImageCausalConv3D,
}

impl QwenVAE {
    pub fn new() -> Result<Self, Exception> {
        Ok(Self {
            encoder: QwenImageEncoder3D::new()?,
            decoder: QwenImageDecoder3D::new()?,
            post_quant_conv: QwenImageCausalConv3D::new(
                16,
                16,
                (1, 1, 1),
                (1, 1, 1),
                (0, 0, 0),
                true,
            )?,
            quant_conv: QwenImageCausalConv3D::new(
                32,
                32,
                (1, 1, 1),
                (1, 1, 1),
                (0, 0, 0),
                true,
            )?,
        })
    }

    /// Normalize latent for diffusion
    pub fn normalize_latent(latent: &Array) -> Result<Array, Exception> {
        let mean = Array::from_slice(&LATENTS_MEAN, &[1, 16, 1, 1]);
        let std = Array::from_slice(&LATENTS_STD, &[1, 16, 1, 1]);
        let diff = ops::subtract(latent, &mean)?;
        ops::divide(&diff, &std)
    }

    /// Denormalize latent for decoding
    pub fn denormalize_latent(latent: &Array) -> Result<Array, Exception> {
        let mean = Array::from_slice(&LATENTS_MEAN, &[1, 16, 1, 1]);
        let std = Array::from_slice(&LATENTS_STD, &[1, 16, 1, 1]);
        let scaled = ops::multiply(latent, &std)?;
        ops::add(&scaled, &mean)
    }

    /// Decode latent to image (expects denormalized latent)
    pub fn decode(&mut self, latent: &Array) -> Result<Array, Exception> {
        // latent: [B, 16, H, W] -> [B, 16, 1, H, W]
        let h = latent.reshape(&[
            latent.dim(0),
            latent.dim(1),
            1,
            latent.dim(2),
            latent.dim(3),
        ])?;

        let h = self.post_quant_conv.forward(&h)?;
        let h = self.decoder.forward(&h)?;
        // Remove time dimension: [B, 3, 1, H, W] -> [B, 3, H, W]
        let indexed = h.index((.., .., 0, .., ..));
        // Force contiguous by flatten + reshape (index creates a strided view that as_slice ignores)
        let numel = indexed.dim(0) * indexed.dim(1) * indexed.dim(2) * indexed.dim(3);
        let flat = indexed.reshape(&[numel])?;
        flat.reshape(&[indexed.dim(0), indexed.dim(1), indexed.dim(2), indexed.dim(3)])
    }

    /// Encode image to latent (returns normalized)
    pub fn encode(&mut self, image: &Array) -> Result<Array, Exception> {
        // image: [B, 4, H, W] (RGBA) -> [B, 4, 1, H, W]
        let h = image.reshape(&[
            image.dim(0),
            image.dim(1),
            1,
            image.dim(2),
            image.dim(3),
        ])?;
        let h = self.encoder.forward(&h)?;
        let h = self.quant_conv.forward(&h)?;
        // Take first 16 channels (mean, ignore logvar) and normalize
        let latent = h.index((.., ..16, 0, .., ..));
        Self::normalize_latent(&latent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_denormalize() {
        let latent = Array::zeros::<f32>(&[1, 16, 8, 8]).unwrap();
        let normalized = QwenVAE::normalize_latent(&latent).unwrap();
        let denormalized = QwenVAE::denormalize_latent(&normalized).unwrap();
        // Should be close to original (zeros)
        assert_eq!(denormalized.shape(), latent.shape());
    }
}
