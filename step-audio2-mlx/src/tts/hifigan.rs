//! HiFi-GAN Vocoder for Step-Audio 2
//!
//! Converts mel spectrograms to waveforms using a GAN-based approach.
//! Architecture follows the CosyVoice2 HiFi-GAN implementation:
//!
//! - Upsampling rates: [8, 8, 2, 2] = 256x total
//! - Upsampling kernels: [16, 16, 4, 4]
//! - ResBlock kernels: [3, 7, 11]
//! - Input: 80-dim mel spectrogram
//! - Output: 24kHz waveform

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    Array,
};

use crate::error::Result;

/// HiFi-GAN configuration
#[derive(Debug, Clone)]
pub struct HiFiGANConfig {
    /// Number of mel channels (input)
    pub num_mels: i32,
    /// Initial channel size after conv_pre
    pub initial_channel: i32,
    /// Upsampling rates (multiply together for total upsample factor)
    pub upsample_rates: Vec<i32>,
    /// Kernel sizes for upsampling convolutions
    pub upsample_kernel_sizes: Vec<i32>,
    /// ResBlock kernel sizes
    pub resblock_kernel_sizes: Vec<i32>,
    /// ResBlock dilation sizes for each kernel
    pub resblock_dilation_sizes: Vec<Vec<i32>>,
    /// Leaky ReLU negative slope
    pub leaky_relu_slope: f32,
}

impl Default for HiFiGANConfig {
    fn default() -> Self {
        Self {
            num_mels: 80,
            initial_channel: 512,
            upsample_rates: vec![8, 8, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![
                vec![1, 3, 5],
                vec![1, 3, 5],
                vec![1, 3, 5],
            ],
            leaky_relu_slope: 0.1,
        }
    }
}

impl HiFiGANConfig {
    /// Get total upsampling factor
    pub fn total_upsample_factor(&self) -> i32 {
        self.upsample_rates.iter().product()
    }
}

/// Residual block with dilated convolutions
///
/// Each ResBlock has multiple dilated convolutions with residual connections.
#[derive(Debug, Clone, ModuleParameters)]
pub struct ResBlock {
    /// Dilated convolutions for this block
    #[param]
    pub convs1: Vec<nn::Conv1d>,
    /// Second set of dilated convolutions
    #[param]
    pub convs2: Vec<nn::Conv1d>,
    /// Leaky ReLU slope
    pub leaky_slope: f32,
}

impl ResBlock {
    /// Create a new ResBlock
    pub fn new(
        channels: i32,
        kernel_size: i32,
        dilations: &[i32],
        leaky_slope: f32,
    ) -> Result<Self> {
        let padding = kernel_size / 2;

        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();

        for &dilation in dilations {
            // First conv: dilated
            let dilation_padding = (kernel_size - 1) * dilation / 2;
            convs1.push(
                nn::Conv1dBuilder::new(channels, channels, kernel_size)
                    .padding(dilation_padding)
                    .dilation(dilation)
                    .build()?,
            );

            // Second conv: no dilation
            convs2.push(
                nn::Conv1dBuilder::new(channels, channels, kernel_size)
                    .padding(padding)
                    .build()?,
            );
        }

        Ok(Self {
            convs1,
            convs2,
            leaky_slope,
        })
    }
}

impl Module<&Array> for ResBlock {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        let mut out = x.clone();

        for (conv1, conv2) in self.convs1.iter_mut().zip(self.convs2.iter_mut()) {
            let residual = out.clone();

            // LeakyReLU -> Conv1 -> LeakyReLU -> Conv2 -> Residual
            let h = nn::leaky_relu(&out, self.leaky_slope)?;
            let h = conv1.forward(&h)?;
            let h = nn::leaky_relu(&h, self.leaky_slope)?;
            let h = conv2.forward(&h)?;

            out = residual.add(&h)?;
        }

        Ok(out)
    }
}

/// Multi-Receptive Field Fusion (MRF) module
///
/// Combines outputs from multiple ResBlocks with different kernel sizes.
#[derive(Debug, Clone, ModuleParameters)]
pub struct MRF {
    /// ResBlocks with different kernel sizes
    #[param]
    pub resblocks: Vec<ResBlock>,
}

impl MRF {
    /// Create a new MRF module
    pub fn new(
        channels: i32,
        kernel_sizes: &[i32],
        dilation_sizes: &[Vec<i32>],
        leaky_slope: f32,
    ) -> Result<Self> {
        let mut resblocks = Vec::new();

        for (ks, dilations) in kernel_sizes.iter().zip(dilation_sizes.iter()) {
            resblocks.push(ResBlock::new(channels, *ks, dilations, leaky_slope)?);
        }

        Ok(Self { resblocks })
    }
}

impl Module<&Array> for MRF {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        // Sum outputs from all resblocks
        let mut out: Option<Array> = None;

        for resblock in &mut self.resblocks {
            let h = resblock.forward(x)?;
            out = Some(match out {
                None => h,
                Some(acc) => acc.add(&h)?,
            });
        }

        // Average
        let out = out.unwrap();
        let n = self.resblocks.len() as f32;
        out.divide(&Array::from_slice(&[n], &[]))
    }
}

/// Transposed convolution for upsampling
///
/// MLX doesn't have ConvTranspose1d, so we implement it using
/// Conv1d with appropriate reshaping or unfold operations.
#[derive(Debug, Clone, ModuleParameters)]
pub struct ConvTranspose1d {
    /// Weight tensor [out_channels, in_channels, kernel_size]
    #[param]
    pub weight: Param<Array>,
    /// Bias tensor [out_channels]
    #[param]
    pub bias: Param<Array>,
    /// Stride (upsampling factor)
    pub stride: i32,
    /// Padding
    pub padding: i32,
    /// Output padding
    pub output_padding: i32,
    /// Output channels
    pub out_channels: i32,
}

impl ConvTranspose1d {
    /// Create a new transposed convolution
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
    ) -> Result<Self> {
        // Initialize weights with small random values
        // Weight shape: [out_channels, in_channels, kernel_size]
        let scale = (2.0 / (in_channels * kernel_size) as f32).sqrt();
        let weight_data: Vec<f32> = (0..(out_channels * in_channels * kernel_size))
            .map(|i| (i as f32 * 0.1234).sin() * scale)
            .collect();

        let weight = Array::from_slice(
            &weight_data,
            &[out_channels, in_channels, kernel_size],
        );

        // Bias initialized to zeros
        let bias_data = vec![0.0f32; out_channels as usize];
        let bias = Array::from_slice(&bias_data, &[out_channels]);

        Ok(Self {
            weight: Param::new(weight),
            bias: Param::new(bias),
            stride,
            padding: (kernel_size - stride) / 2,
            output_padding: 0,
            out_channels,
        })
    }

    /// Create with explicit padding
    pub fn with_padding(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        padding: i32,
    ) -> Result<Self> {
        let mut layer = Self::new(in_channels, out_channels, kernel_size, stride)?;
        layer.padding = padding;
        Ok(layer)
    }
}

impl Module<&Array> for ConvTranspose1d {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        // x: [B, C_in, T]
        // Simplified upsampling: use repeat + conv to approximate transposed conv
        // This is a common approximation that works well for upsampling

        let shape = x.shape();
        let batch = shape[0];
        let in_channels = shape[1];
        let in_length = shape[2];

        // Upsample by repeating each element `stride` times
        // [B, C, T] -> [B, C, T*stride]
        let upsampled_len = in_length * self.stride;

        // Use reshape + tile to upsample
        // [B, C, T] -> [B, C, T, 1] -> [B, C, T, stride] -> [B, C, T*stride]
        let x_expanded = x.reshape(&[batch, in_channels, in_length, 1])?;

        // Create upsample pattern using tile
        let ones = Array::ones::<f32>(&[1, 1, 1, self.stride])?;
        let x_tiled = x_expanded.multiply(&ones)?;
        let x_upsampled = x_tiled.reshape(&[batch, in_channels, upsampled_len])?;

        // Apply convolution with weight
        let weight = self.weight.as_ref();
        let kernel_size = weight.shape()[2];

        // Calculate padding to maintain output size
        let out_padding = (kernel_size - 1) / 2;

        let out = mlx_rs::ops::conv1d(
            &x_upsampled,
            weight,
            1, // stride 1
            out_padding,
            1, // dilation
            1, // groups
        )?;

        // Add bias: [out_channels] -> [1, out_channels, 1]
        let bias_expanded = self.bias.as_ref().reshape(&[1, self.out_channels, 1])?;
        out.add(&bias_expanded)
    }
}

/// HiFi-GAN Generator
///
/// Converts mel spectrograms to audio waveforms.
#[derive(Debug, Clone, ModuleParameters)]
pub struct HiFiGAN {
    /// Pre-processing convolution
    #[param]
    pub conv_pre: nn::Conv1d,
    /// Upsampling layers
    #[param]
    pub ups: Vec<ConvTranspose1d>,
    /// Multi-Receptive Field Fusion modules
    #[param]
    pub mrfs: Vec<MRF>,
    /// Post-processing convolution
    #[param]
    pub conv_post: nn::Conv1d,
    /// Configuration
    pub config: HiFiGANConfig,
}

impl HiFiGAN {
    /// Create a new HiFi-GAN vocoder
    pub fn new(config: HiFiGANConfig) -> Result<Self> {
        let num_ups = config.upsample_rates.len();
        let leaky_slope = config.leaky_relu_slope;

        // Pre-conv: mel_channels -> initial_channel
        let conv_pre = nn::Conv1dBuilder::new(config.num_mels, config.initial_channel, 7)
            .padding(3)
            .build()?;

        // Upsampling layers and MRF modules
        let mut ups = Vec::new();
        let mut mrfs = Vec::new();

        let mut channels = config.initial_channel;
        for i in 0..num_ups {
            let upsample_rate = config.upsample_rates[i];
            let kernel_size = config.upsample_kernel_sizes[i];
            let out_channels = channels / 2;

            ups.push(ConvTranspose1d::with_padding(
                channels,
                out_channels,
                kernel_size,
                upsample_rate,
                (kernel_size - upsample_rate) / 2,
            )?);

            mrfs.push(MRF::new(
                out_channels,
                &config.resblock_kernel_sizes,
                &config.resblock_dilation_sizes,
                leaky_slope,
            )?);

            channels = out_channels;
        }

        // Post-conv: channels -> 1 (mono audio)
        let conv_post = nn::Conv1dBuilder::new(channels, 1, 7)
            .padding(3)
            .build()?;

        Ok(Self {
            conv_pre,
            ups,
            mrfs,
            conv_post,
            config,
        })
    }

    /// Load HiFi-GAN from model directory
    pub fn load(model_dir: impl AsRef<std::path::Path>) -> Result<Self> {
        // TODO: Implement weight loading from safetensors
        // For now, create with default config
        Self::new(HiFiGANConfig::default())
    }
}

impl Module<&Array> for HiFiGAN {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, mel: &Array) -> std::result::Result<Array, Self::Error> {
        // mel: [B, num_mels, T]
        let leaky_slope = self.config.leaky_relu_slope;

        // Pre-conv
        let mut x = self.conv_pre.forward(mel)?;

        // Upsampling blocks
        for (up, mrf) in self.ups.iter_mut().zip(self.mrfs.iter_mut()) {
            x = nn::leaky_relu(&x, leaky_slope)?;
            x = up.forward(&x)?;
            x = mrf.forward(&x)?;
        }

        // Post-conv
        x = nn::leaky_relu(&x, leaky_slope)?;
        x = self.conv_post.forward(&x)?;

        // Tanh activation for waveform [-1, 1]
        mlx_rs::ops::tanh(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hifigan_config() {
        let config = HiFiGANConfig::default();
        assert_eq!(config.num_mels, 80);
        assert_eq!(config.total_upsample_factor(), 256);
    }

    #[test]
    fn test_resblock_creation() {
        let resblock = ResBlock::new(64, 3, &[1, 3, 5], 0.1);
        assert!(resblock.is_ok());
    }

    #[test]
    fn test_mrf_creation() {
        let mrf = MRF::new(
            64,
            &[3, 7, 11],
            &[vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            0.1,
        );
        assert!(mrf.is_ok());
    }

    #[test]
    fn test_hifigan_creation() {
        let hifigan = HiFiGAN::new(HiFiGANConfig::default());
        assert!(hifigan.is_ok());
    }
}
