//! HiFT (HiFi-GAN with source modeling) Vocoder for Step-Audio 2
//!
//! Weight-based implementation. Architecture:
//! - conv_pre: 80 → 512 (kernel=7)
//! - 3 upsample stages: 512→256 (8x), 256→128 (4x), 128→64 (8x) = 256x total
//! - 9 resblocks: 3 per level, 3 layers each, with Snake activation
//! - conv_post: 64 → 18 (kernel=7), then sum to mono

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{Array, error::Exception, ops};

use crate::error::{Error, Result};

/// HiFi-GAN configuration
#[derive(Debug, Clone)]
pub struct HiFiGANConfig {
    pub num_mels: i32,
    pub initial_channel: i32,
    pub upsample_rates: Vec<i32>,
    pub resblock_kernel_sizes: Vec<i32>,
    pub num_resblocks_per_level: i32,
    pub num_layers_per_resblock: i32,
    pub num_output_channels: i32,
    pub sample_rate: i32,
}

impl Default for HiFiGANConfig {
    fn default() -> Self {
        Self {
            num_mels: 80,
            initial_channel: 512,
            upsample_rates: vec![8, 4, 8], // Total: 256x
            resblock_kernel_sizes: vec![3, 3, 3, 3, 3, 3, 3, 3, 3], // All use kernel 3
            num_resblocks_per_level: 3,
            num_layers_per_resblock: 3,
            num_output_channels: 18,
            sample_rate: 24000,
        }
    }
}

// =========================================================================
// Helper functions
// =========================================================================

/// Conv1d with same-padding, transposing PyTorch weights [out, in, K] -> MLX [out, K, in]
fn conv1d_same(x: &Array, w: &Array, b: Option<&Array>) -> std::result::Result<Array, Exception> {
    let w = w.transpose_axes(&[0, 2, 1])?;
    let kernel_size = w.shape()[1];
    let padding = kernel_size / 2;
    let out = ops::conv1d_device(x, &w, None, Some(padding), None, None, mlx_rs::StreamOrDevice::default())?;
    match b {
        Some(bias) => out.add(bias),
        None => Ok(out),
    }
}

/// Transposed conv1d: PyTorch ConvTranspose1d weight [in, out, K] -> MLX [out, K, in]
fn conv_transpose1d(x: &Array, w: &Array, b: Option<&Array>, stride: i32) -> std::result::Result<Array, Exception> {
    // MLX conv_transpose1d expects weight [C_out, K, C_in]
    // PyTorch ConvTranspose1d stores [C_in, C_out, K]
    let w = w.transpose_axes(&[1, 2, 0])?;
    let kernel_size = w.shape()[1] as i32;
    let padding = (kernel_size - stride) / 2;
    let out = ops::conv_transpose1d_device(x, &w, Some(stride), Some(padding), None, None, None, mlx_rs::StreamOrDevice::default())?;
    match b {
        Some(bias) => out.add(bias),
        None => Ok(out),
    }
}

/// Snake activation: x + (1/alpha) * sin^2(alpha * x)
fn snake(x: &Array, alpha: &Array) -> std::result::Result<Array, Exception> {
    let ax = x.multiply(alpha)?;
    let sin_ax = ops::sin(&ax)?;
    let sin2 = sin_ax.square()?;
    let inv_alpha = ops::divide(&Array::from_slice(&[1.0f32], &[]), alpha)?;
    let term = sin2.multiply(&inv_alpha)?;
    x.add(&term)
}

// =========================================================================
// HiFi-GAN
// =========================================================================

pub struct HiFiGAN {
    pub weights: HashMap<String, Array>,
    pub config: HiFiGANConfig,
    pub weights_loaded: bool,
}

impl HiFiGAN {
    pub fn new(config: HiFiGANConfig) -> Result<Self> {
        Ok(Self {
            weights: HashMap::new(),
            config,
            weights_loaded: false,
        })
    }

    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let weights_path = model_dir.join("tts_mlx").join("hifigan.safetensors");

        if !weights_path.exists() {
            return Err(Error::ModelLoad(format!(
                "HiFiGAN weights not found at {:?}", weights_path
            )));
        }

        let config = HiFiGANConfig::default();
        let weights = Array::load_safetensors(&weights_path)?;
        println!("  Loaded {} hifigan weights", weights.len());

        let model = Self { weights, config, weights_loaded: true };
        model.validate_weights()?;
        Ok(model)
    }

    fn w(&self, key: &str) -> &Array {
        self.weights.get(key).unwrap_or_else(|| panic!("Missing HiFiGAN weight: {}", key))
    }

    /// Validate critical weights exist at load time so w() won't panic during inference
    fn validate_weights(&self) -> Result<()> {
        let mut missing = Vec::new();
        for key in ["hifigan.conv_pre.weight", "hifigan.conv_pre.bias",
                     "hifigan.conv_post.weight", "hifigan.conv_post.bias"] {
            if !self.weights.contains_key(key) {
                missing.push(key.to_string());
            }
        }
        for i in 0..self.config.upsample_rates.len() {
            for suffix in ["weight", "bias"] {
                let key = format!("hifigan.ups.{}.{}", i, suffix);
                if !self.weights.contains_key(&key) {
                    missing.push(key);
                }
            }
        }
        let total_resblocks = self.config.upsample_rates.len() * self.config.num_resblocks_per_level as usize;
        for i in 0..total_resblocks {
            let key = format!("hifigan.resblocks.{}.convs1.0.weight", i);
            if !self.weights.contains_key(&key) {
                missing.push(key);
            }
        }
        if !missing.is_empty() {
            return Err(Error::ModelLoad(format!(
                "HiFiGAN missing {} critical weights: {}",
                missing.len(), missing[..missing.len().min(5)].join(", ")
            )));
        }
        Ok(())
    }

    /// Resblock: 3 layers of (snake→conv1→snake→conv2→residual)
    fn resblock(&self, x: &Array, prefix: &str) -> std::result::Result<Array, Exception> {
        let mut out = x.clone();
        let num_layers = self.config.num_layers_per_resblock as usize;

        for i in 0..num_layers {
            let residual = out.clone();

            // Snake activation 1
            let alpha1 = self.w(&format!("{}.activations1.{}.alpha", prefix, i));
            out = snake(&out, alpha1)?;

            // Conv1
            out = conv1d_same(
                &out,
                self.w(&format!("{}.convs1.{}.weight", prefix, i)),
                Some(self.w(&format!("{}.convs1.{}.bias", prefix, i))),
            )?;

            // Snake activation 2
            let alpha2 = self.w(&format!("{}.activations2.{}.alpha", prefix, i));
            out = snake(&out, alpha2)?;

            // Conv2
            out = conv1d_same(
                &out,
                self.w(&format!("{}.convs2.{}.weight", prefix, i)),
                Some(self.w(&format!("{}.convs2.{}.bias", prefix, i))),
            )?;

            out = residual.add(&out)?;
        }

        Ok(out)
    }

    /// Synthesize audio from mel spectrogram
    pub fn synthesize(&self, mel: &Array) -> Result<Array> {
        // mel: [B, mel_dim, T] from flow decoder
        // Transpose to [B, T, mel_dim] for MLX conv1d
        let mel = mel.transpose_axes(&[0, 2, 1])
            .map_err(|e| Error::Inference(format!("Transpose mel: {}", e)))?;

        // conv_pre: [B, T, 80] -> [B, T, 512]
        let mut x = conv1d_same(
            &mel,
            self.w("hifigan.conv_pre.weight"),
            Some(self.w("hifigan.conv_pre.bias")),
        ).map_err(|e| Error::Inference(format!("conv_pre: {}", e)))?;
        // 3 upsample stages
        let resblocks_per_level = self.config.num_resblocks_per_level as usize;

        for (level, &up_rate) in self.config.upsample_rates.iter().enumerate() {
            // LeakyReLU before upsample
            x = mlx_rs::nn::leaky_relu(&x, 0.1)
                .map_err(|e| Error::Inference(format!("leaky_relu: {}", e)))?;

            // Transposed convolution (upsample)
            x = conv_transpose1d(
                &x,
                self.w(&format!("hifigan.ups.{}.weight", level)),
                Some(self.w(&format!("hifigan.ups.{}.bias", level))),
                up_rate,
            ).map_err(|e| Error::Inference(format!("ups.{}: {}", level, e)))?;
            // Multi-Receptive Field Fusion: sum resblock outputs and average
            let resblock_start = level * resblocks_per_level;
            let mut xs: Option<Array> = None;
            for j in 0..resblocks_per_level {
                let rb_idx = resblock_start + j;
                let rb_out = self.resblock(&x, &format!("hifigan.resblocks.{}", rb_idx))
                    .map_err(|e| Error::Inference(format!("resblock.{}: {}", rb_idx, e)))?;
                xs = Some(match xs {
                    Some(acc) => acc.add(&rb_out)?,
                    None => rb_out,
                });
            }
            let num_rb = Array::from_slice(&[resblocks_per_level as f32], &[]);
            x = xs.unwrap().divide(&num_rb)
                .map_err(|e| Error::Inference(format!("resblock avg {}: {}", level, e)))?;
        }

        // conv_post: [B, T, 64] -> [B, T, 18]
        x = mlx_rs::nn::leaky_relu(&x, 0.1)
            .map_err(|e| Error::Inference(format!("final leaky_relu: {}", e)))?;
        x = conv1d_same(
            &x,
            self.w("hifigan.conv_post.weight"),
            Some(self.w("hifigan.conv_post.bias")),
        ).map_err(|e| Error::Inference(format!("conv_post: {}", e)))?;

        // Tanh
        x = ops::tanh(&x)
            .map_err(|e| Error::Inference(format!("tanh: {}", e)))?;

        // Sum across output channels: [B, T, 18] -> [B, T]
        let x = x.sum_axis(2, false)
            .map_err(|e| Error::Inference(format!("sum channels: {}", e)))?;

        // Normalize
        let norm = Array::from_slice(&[self.config.num_output_channels as f32], &[]);
        x.divide(&norm)
            .map_err(|e| Error::Inference(format!("normalize: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hifigan_config() {
        let config = HiFiGANConfig::default();
        assert_eq!(config.num_mels, 80);
        assert_eq!(config.initial_channel, 512);
    }
}
