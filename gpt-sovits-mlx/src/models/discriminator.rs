//! Multi-Period Discriminator for VITS GAN Training
//!
//! This module implements the discriminator architecture used in HiFi-GAN/VITS
//! for adversarial training of the vocoder.
//!
//! The discriminator consists of:
//! - DiscriminatorS: Scale discriminator using 1D convolutions
//! - DiscriminatorP: Period discriminators that reshape audio to 2D
//! - MultiPeriodDiscriminator: Combines all discriminators

use mlx_rs::{
    Array,
    builder::Builder,
    macros::ModuleParameters,
    module::Module,
    nn,
    ops,
};

use crate::error::Error;

/// Leaky ReLU slope (same as Python implementation)
const LRELU_SLOPE: f32 = 0.1;

/// Calculate padding for convolution to maintain size
fn get_padding(kernel_size: i32, dilation: i32) -> i32 {
    (kernel_size * dilation - dilation) / 2
}

/// Scale Discriminator (DiscriminatorS)
///
/// Uses 1D convolutions at different scales to analyze audio
#[derive(Debug, Clone, ModuleParameters)]
pub struct DiscriminatorS {
    #[param]
    conv1: nn::Conv1d,
    #[param]
    conv2: nn::Conv1d,
    #[param]
    conv3: nn::Conv1d,
    #[param]
    conv4: nn::Conv1d,
    #[param]
    conv5: nn::Conv1d,
    #[param]
    conv6: nn::Conv1d,
    #[param]
    conv_post: nn::Conv1d,
}

impl DiscriminatorS {
    pub fn new() -> Result<Self, Error> {
        // Channel progression: 1 -> 16 -> 64 -> 256 -> 1024 -> 1024 -> 1024
        let conv1 = nn::Conv1dBuilder::new(1, 16, 15)
            .stride(1)
            .padding(7)
            .build()
            .map_err(|e| Error::Message(e.to_string()))?;

        // Note: groups disabled temporarily due to weight initialization issue
        // TODO: Fix grouped convolution weight shapes for proper HiFi-GAN discriminator
        let conv2 = nn::Conv1dBuilder::new(16, 64, 41)
            .stride(4)
            .padding(20)
            // .groups(4)  // Disabled: weight init doesn't handle groups correctly
            .build()
            .map_err(|e| Error::Message(e.to_string()))?;

        let conv3 = nn::Conv1dBuilder::new(64, 256, 41)
            .stride(4)
            .padding(20)
            // .groups(16)
            .build()
            .map_err(|e| Error::Message(e.to_string()))?;

        let conv4 = nn::Conv1dBuilder::new(256, 1024, 41)
            .stride(4)
            .padding(20)
            // .groups(64)
            .build()
            .map_err(|e| Error::Message(e.to_string()))?;

        let conv5 = nn::Conv1dBuilder::new(1024, 1024, 41)
            .stride(4)
            .padding(20)
            // .groups(256)
            .build()
            .map_err(|e| Error::Message(e.to_string()))?;

        let conv6 = nn::Conv1dBuilder::new(1024, 1024, 5)
            .stride(1)
            .padding(2)
            .build()
            .map_err(|e| Error::Message(e.to_string()))?;

        let conv_post = nn::Conv1dBuilder::new(1024, 1, 3)
            .stride(1)
            .padding(1)
            .build()
            .map_err(|e| Error::Message(e.to_string()))?;

        Ok(Self {
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            conv6,
            conv_post,
        })
    }

    /// Forward pass returning (output, feature_maps)
    /// Input: x in NCL format [batch, channels, length]
    pub fn forward(&mut self, x: &Array) -> Result<(Array, Vec<Array>), Error> {
        use mlx_rs::ops::swap_axes;
        let mut fmap = Vec::new();

        // Convert NCL -> NLC for Conv1d
        let mut x = swap_axes(x, 1, 2).map_err(|e| Error::Message(e.to_string()))?;
        x = self.conv1.forward(&x).map_err(|e| Error::Message(e.to_string()))?;
        x = nn::leaky_relu(&x, LRELU_SLOPE).map_err(|e| Error::Message(e.to_string()))?;
        fmap.push(x.clone());

        x = self.conv2.forward(&x).map_err(|e| Error::Message(e.to_string()))?;
        x = nn::leaky_relu(&x, LRELU_SLOPE).map_err(|e| Error::Message(e.to_string()))?;
        fmap.push(x.clone());

        x = self.conv3.forward(&x).map_err(|e| Error::Message(e.to_string()))?;
        x = nn::leaky_relu(&x, LRELU_SLOPE).map_err(|e| Error::Message(e.to_string()))?;
        fmap.push(x.clone());

        x = self.conv4.forward(&x).map_err(|e| Error::Message(e.to_string()))?;
        x = nn::leaky_relu(&x, LRELU_SLOPE).map_err(|e| Error::Message(e.to_string()))?;
        fmap.push(x.clone());

        x = self.conv5.forward(&x).map_err(|e| Error::Message(e.to_string()))?;
        x = nn::leaky_relu(&x, LRELU_SLOPE).map_err(|e| Error::Message(e.to_string()))?;
        fmap.push(x.clone());

        x = self.conv6.forward(&x).map_err(|e| Error::Message(e.to_string()))?;
        x = nn::leaky_relu(&x, LRELU_SLOPE).map_err(|e| Error::Message(e.to_string()))?;
        fmap.push(x.clone());

        x = self.conv_post.forward(&x).map_err(|e| Error::Message(e.to_string()))?;
        fmap.push(x.clone());

        // Flatten: [B, C, T] -> [B, C*T]
        let shape = x.shape();
        let batch = shape[0];
        x = x.reshape(&[batch, -1]).map_err(|e| Error::Message(e.to_string()))?;

        Ok((x, fmap))
    }

    /// Forward pass returning Exception (for use with value_and_grad)
    /// Input: x in NCL format [batch, channels, length]
    pub fn forward_ex(&mut self, x: &Array) -> Result<(Array, Vec<Array>), mlx_rs::error::Exception> {
        use mlx_rs::ops::swap_axes;
        let mut fmap = Vec::new();

        // Convert NCL -> NLC for Conv1d
        let mut x = swap_axes(x, 1, 2)?;
        x = self.conv1.forward(&x)?;
        x = nn::leaky_relu(&x, LRELU_SLOPE)?;
        fmap.push(x.clone());

        x = self.conv2.forward(&x)?;
        x = nn::leaky_relu(&x, LRELU_SLOPE)?;
        fmap.push(x.clone());

        x = self.conv3.forward(&x)?;
        x = nn::leaky_relu(&x, LRELU_SLOPE)?;
        fmap.push(x.clone());

        x = self.conv4.forward(&x)?;
        x = nn::leaky_relu(&x, LRELU_SLOPE)?;
        fmap.push(x.clone());

        x = self.conv5.forward(&x)?;
        x = nn::leaky_relu(&x, LRELU_SLOPE)?;
        fmap.push(x.clone());

        x = self.conv6.forward(&x)?;
        x = nn::leaky_relu(&x, LRELU_SLOPE)?;
        fmap.push(x.clone());

        x = self.conv_post.forward(&x)?;
        fmap.push(x.clone());

        // Flatten: [B, C, T] -> [B, C*T]
        let shape = x.shape();
        let batch = shape[0];
        x = x.reshape(&[batch, -1])?;

        Ok((x, fmap))
    }
}

/// Period Discriminator (DiscriminatorP)
///
/// Reshapes 1D audio to 2D based on period and applies 2D convolutions
#[derive(Debug, Clone, ModuleParameters)]
pub struct DiscriminatorP {
    period: i32,
    #[param]
    conv1: nn::Conv2d,
    #[param]
    conv2: nn::Conv2d,
    #[param]
    conv3: nn::Conv2d,
    #[param]
    conv4: nn::Conv2d,
    #[param]
    conv5: nn::Conv2d,
    #[param]
    conv_post: nn::Conv2d,
}

impl DiscriminatorP {
    pub fn new(period: i32) -> Result<Self, Error> {
        let kernel_size = 5;
        let stride = 3;

        // Channel progression: 1 -> 32 -> 128 -> 512 -> 1024 -> 1024
        let conv1 = nn::Conv2dBuilder::new(1, 32, (kernel_size, 1))
            .stride((stride, 1))
            .padding((get_padding(kernel_size, 1), 0))
            .build()
            .map_err(|e| Error::Message(e.to_string()))?;

        let conv2 = nn::Conv2dBuilder::new(32, 128, (kernel_size, 1))
            .stride((stride, 1))
            .padding((get_padding(kernel_size, 1), 0))
            .build()
            .map_err(|e| Error::Message(e.to_string()))?;

        let conv3 = nn::Conv2dBuilder::new(128, 512, (kernel_size, 1))
            .stride((stride, 1))
            .padding((get_padding(kernel_size, 1), 0))
            .build()
            .map_err(|e| Error::Message(e.to_string()))?;

        let conv4 = nn::Conv2dBuilder::new(512, 1024, (kernel_size, 1))
            .stride((stride, 1))
            .padding((get_padding(kernel_size, 1), 0))
            .build()
            .map_err(|e| Error::Message(e.to_string()))?;

        let conv5 = nn::Conv2dBuilder::new(1024, 1024, (kernel_size, 1))
            .stride((1, 1))  // No stride on last conv
            .padding((get_padding(kernel_size, 1), 0))
            .build()
            .map_err(|e| Error::Message(e.to_string()))?;

        let conv_post = nn::Conv2dBuilder::new(1024, 1, (3, 1))
            .stride((1, 1))
            .padding((1, 0))
            .build()
            .map_err(|e| Error::Message(e.to_string()))?;

        Ok(Self {
            period,
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            conv_post,
        })
    }

    /// Forward pass returning (output, feature_maps)
    pub fn forward(&mut self, x: &Array) -> Result<(Array, Vec<Array>), Error> {
        let mut fmap = Vec::new();

        // Get shape: [B, C, T]
        let shape = x.shape();
        let b = shape[0];
        let c = shape[1];
        let mut t = shape[2];

        let mut x = x.clone();

        // Pad if needed so T is divisible by period
        if t % self.period != 0 {
            let n_pad = self.period - (t % self.period);
            // Pad on the last dimension (time)
            let widths: &[(i32, i32)] = &[(0, 0), (0, 0), (0, n_pad)];
            x = ops::pad(&x, widths, None, None)
                .map_err(|e| Error::Message(e.to_string()))?;
            t = t + n_pad;
        }

        // Reshape 1D to 2D: [B, C, T] -> [B, C, T//period, period]
        x = x.reshape(&[b, c, t / self.period, self.period])
            .map_err(|e| Error::Message(e.to_string()))?;

        // Convert NCHW -> NHWC for MLX Conv2d
        // [B, C, H, W] -> [B, H, W, C]
        x = x.transpose_axes(&[0, 2, 3, 1])
            .map_err(|e| Error::Message(e.to_string()))?;

        // Apply convolutions
        x = self.conv1.forward(&x).map_err(|e| Error::Message(e.to_string()))?;
        x = nn::leaky_relu(&x, LRELU_SLOPE).map_err(|e| Error::Message(e.to_string()))?;
        fmap.push(x.clone());

        x = self.conv2.forward(&x).map_err(|e| Error::Message(e.to_string()))?;
        x = nn::leaky_relu(&x, LRELU_SLOPE).map_err(|e| Error::Message(e.to_string()))?;
        fmap.push(x.clone());

        x = self.conv3.forward(&x).map_err(|e| Error::Message(e.to_string()))?;
        x = nn::leaky_relu(&x, LRELU_SLOPE).map_err(|e| Error::Message(e.to_string()))?;
        fmap.push(x.clone());

        x = self.conv4.forward(&x).map_err(|e| Error::Message(e.to_string()))?;
        x = nn::leaky_relu(&x, LRELU_SLOPE).map_err(|e| Error::Message(e.to_string()))?;
        fmap.push(x.clone());

        x = self.conv5.forward(&x).map_err(|e| Error::Message(e.to_string()))?;
        x = nn::leaky_relu(&x, LRELU_SLOPE).map_err(|e| Error::Message(e.to_string()))?;
        fmap.push(x.clone());

        x = self.conv_post.forward(&x).map_err(|e| Error::Message(e.to_string()))?;
        fmap.push(x.clone());

        // Flatten: [B, C, H, W] -> [B, C*H*W]
        let shape = x.shape();
        let batch = shape[0];
        x = x.reshape(&[batch, -1]).map_err(|e| Error::Message(e.to_string()))?;

        Ok((x, fmap))
    }

    /// Forward pass returning Exception (for use with value_and_grad)
    pub fn forward_ex(&mut self, x: &Array) -> Result<(Array, Vec<Array>), mlx_rs::error::Exception> {
        let mut fmap = Vec::new();

        // Get shape: [B, C, T]
        let shape = x.shape();
        let b = shape[0];
        let c = shape[1];
        let mut t = shape[2];

        let mut x = x.clone();

        // Pad if needed so T is divisible by period
        if t % self.period != 0 {
            let n_pad = self.period - (t % self.period);
            let widths: &[(i32, i32)] = &[(0, 0), (0, 0), (0, n_pad)];
            x = ops::pad(&x, widths, None, None)?;
            t = t + n_pad;
        }

        // Reshape 1D to 2D: [B, C, T] -> [B, C, T//period, period]
        x = x.reshape(&[b, c, t / self.period, self.period])?;

        // Convert NCHW -> NHWC for MLX Conv2d
        // [B, C, H, W] -> [B, H, W, C]
        x = x.transpose_axes(&[0, 2, 3, 1])?;

        // Apply convolutions
        x = self.conv1.forward(&x)?;
        x = nn::leaky_relu(&x, LRELU_SLOPE)?;
        fmap.push(x.clone());

        x = self.conv2.forward(&x)?;
        x = nn::leaky_relu(&x, LRELU_SLOPE)?;
        fmap.push(x.clone());

        x = self.conv3.forward(&x)?;
        x = nn::leaky_relu(&x, LRELU_SLOPE)?;
        fmap.push(x.clone());

        x = self.conv4.forward(&x)?;
        x = nn::leaky_relu(&x, LRELU_SLOPE)?;
        fmap.push(x.clone());

        x = self.conv5.forward(&x)?;
        x = nn::leaky_relu(&x, LRELU_SLOPE)?;
        fmap.push(x.clone());

        x = self.conv_post.forward(&x)?;
        fmap.push(x.clone());

        // Flatten: [B, C, H, W] -> [B, C*H*W]
        let shape = x.shape();
        let batch = shape[0];
        x = x.reshape(&[batch, -1])?;

        Ok((x, fmap))
    }
}

/// Multi-Period Discriminator
///
/// Combines DiscriminatorS with multiple DiscriminatorP instances
/// to analyze audio at different temporal resolutions.
#[derive(Debug, Clone, ModuleParameters)]
pub struct MultiPeriodDiscriminator {
    /// Scale discriminator
    #[param]
    disc_s: DiscriminatorS,
    /// Period discriminator for period 2
    #[param]
    disc_p2: DiscriminatorP,
    /// Period discriminator for period 3
    #[param]
    disc_p3: DiscriminatorP,
    /// Period discriminator for period 5
    #[param]
    disc_p5: DiscriminatorP,
    /// Period discriminator for period 7
    #[param]
    disc_p7: DiscriminatorP,
    /// Period discriminator for period 11
    #[param]
    disc_p11: DiscriminatorP,
}

/// Configuration for MultiPeriodDiscriminator
#[derive(Debug, Clone)]
pub struct MPDConfig {
    /// Whether to use spectral norm (currently not implemented)
    pub use_spectral_norm: bool,
}

impl Default for MPDConfig {
    fn default() -> Self {
        Self {
            use_spectral_norm: false,
        }
    }
}

impl MultiPeriodDiscriminator {
    /// Create a new MultiPeriodDiscriminator with standard periods [2, 3, 5, 7, 11]
    pub fn new(_config: MPDConfig) -> Result<Self, Error> {
        Ok(Self {
            disc_s: DiscriminatorS::new()?,
            disc_p2: DiscriminatorP::new(2)?,
            disc_p3: DiscriminatorP::new(3)?,
            disc_p5: DiscriminatorP::new(5)?,
            disc_p7: DiscriminatorP::new(7)?,
            disc_p11: DiscriminatorP::new(11)?,
        })
    }

    /// Forward pass for real and generated audio
    ///
    /// Returns (real_outputs, fake_outputs, real_fmaps, fake_fmaps)
    /// where outputs are discriminator scores and fmaps are intermediate feature maps
    pub fn forward(
        &mut self,
        y_real: &Array,
        y_fake: &Array,
    ) -> Result<(Vec<Array>, Vec<Array>, Vec<Vec<Array>>, Vec<Vec<Array>>), Error> {
        let mut y_d_rs = Vec::new();
        let mut y_d_gs = Vec::new();
        let mut fmap_rs = Vec::new();
        let mut fmap_gs = Vec::new();

        // Scale discriminator
        let (y_d_r, fmap_r) = self.disc_s.forward(y_real)?;
        let (y_d_g, fmap_g) = self.disc_s.forward(y_fake)?;
        y_d_rs.push(y_d_r);
        y_d_gs.push(y_d_g);
        fmap_rs.push(fmap_r);
        fmap_gs.push(fmap_g);

        // Period discriminators - need separate calls due to mutable borrow
        let (y_d_r, fmap_r) = self.disc_p2.forward(y_real)?;
        let (y_d_g, fmap_g) = self.disc_p2.forward(y_fake)?;
        y_d_rs.push(y_d_r);
        y_d_gs.push(y_d_g);
        fmap_rs.push(fmap_r);
        fmap_gs.push(fmap_g);

        let (y_d_r, fmap_r) = self.disc_p3.forward(y_real)?;
        let (y_d_g, fmap_g) = self.disc_p3.forward(y_fake)?;
        y_d_rs.push(y_d_r);
        y_d_gs.push(y_d_g);
        fmap_rs.push(fmap_r);
        fmap_gs.push(fmap_g);

        let (y_d_r, fmap_r) = self.disc_p5.forward(y_real)?;
        let (y_d_g, fmap_g) = self.disc_p5.forward(y_fake)?;
        y_d_rs.push(y_d_r);
        y_d_gs.push(y_d_g);
        fmap_rs.push(fmap_r);
        fmap_gs.push(fmap_g);

        let (y_d_r, fmap_r) = self.disc_p7.forward(y_real)?;
        let (y_d_g, fmap_g) = self.disc_p7.forward(y_fake)?;
        y_d_rs.push(y_d_r);
        y_d_gs.push(y_d_g);
        fmap_rs.push(fmap_r);
        fmap_gs.push(fmap_g);

        let (y_d_r, fmap_r) = self.disc_p11.forward(y_real)?;
        let (y_d_g, fmap_g) = self.disc_p11.forward(y_fake)?;
        y_d_rs.push(y_d_r);
        y_d_gs.push(y_d_g);
        fmap_rs.push(fmap_r);
        fmap_gs.push(fmap_g);

        Ok((y_d_rs, y_d_gs, fmap_rs, fmap_gs))
    }

    /// Get discriminator outputs for single input (for inference/evaluation)
    pub fn forward_single(&mut self, x: &Array) -> Result<(Vec<Array>, Vec<Vec<Array>>), Error> {
        let mut outputs = Vec::new();
        let mut fmaps = Vec::new();

        // Scale discriminator
        let (out, fmap) = self.disc_s.forward(x)?;
        outputs.push(out);
        fmaps.push(fmap);

        // Period discriminators
        let (out, fmap) = self.disc_p2.forward(x)?;
        outputs.push(out);
        fmaps.push(fmap);

        let (out, fmap) = self.disc_p3.forward(x)?;
        outputs.push(out);
        fmaps.push(fmap);

        let (out, fmap) = self.disc_p5.forward(x)?;
        outputs.push(out);
        fmaps.push(fmap);

        let (out, fmap) = self.disc_p7.forward(x)?;
        outputs.push(out);
        fmaps.push(fmap);

        let (out, fmap) = self.disc_p11.forward(x)?;
        outputs.push(out);
        fmaps.push(fmap);

        Ok((outputs, fmaps))
    }

    /// Forward pass for real and generated audio returning Exception (for use with value_and_grad)
    ///
    /// Returns (real_outputs, fake_outputs, real_fmaps, fake_fmaps)
    pub fn forward_ex(
        &mut self,
        y_real: &Array,
        y_fake: &Array,
    ) -> Result<(Vec<Array>, Vec<Array>, Vec<Vec<Array>>, Vec<Vec<Array>>), mlx_rs::error::Exception> {
        let mut y_d_rs = Vec::new();
        let mut y_d_gs = Vec::new();
        let mut fmap_rs = Vec::new();
        let mut fmap_gs = Vec::new();

        // Scale discriminator
        let (y_d_r, fmap_r) = self.disc_s.forward_ex(y_real)?;
        let (y_d_g, fmap_g) = self.disc_s.forward_ex(y_fake)?;
        y_d_rs.push(y_d_r);
        y_d_gs.push(y_d_g);
        fmap_rs.push(fmap_r);
        fmap_gs.push(fmap_g);

        // Period discriminators
        let (y_d_r, fmap_r) = self.disc_p2.forward_ex(y_real)?;
        let (y_d_g, fmap_g) = self.disc_p2.forward_ex(y_fake)?;
        y_d_rs.push(y_d_r);
        y_d_gs.push(y_d_g);
        fmap_rs.push(fmap_r);
        fmap_gs.push(fmap_g);

        let (y_d_r, fmap_r) = self.disc_p3.forward_ex(y_real)?;
        let (y_d_g, fmap_g) = self.disc_p3.forward_ex(y_fake)?;
        y_d_rs.push(y_d_r);
        y_d_gs.push(y_d_g);
        fmap_rs.push(fmap_r);
        fmap_gs.push(fmap_g);

        let (y_d_r, fmap_r) = self.disc_p5.forward_ex(y_real)?;
        let (y_d_g, fmap_g) = self.disc_p5.forward_ex(y_fake)?;
        y_d_rs.push(y_d_r);
        y_d_gs.push(y_d_g);
        fmap_rs.push(fmap_r);
        fmap_gs.push(fmap_g);

        let (y_d_r, fmap_r) = self.disc_p7.forward_ex(y_real)?;
        let (y_d_g, fmap_g) = self.disc_p7.forward_ex(y_fake)?;
        y_d_rs.push(y_d_r);
        y_d_gs.push(y_d_g);
        fmap_rs.push(fmap_r);
        fmap_gs.push(fmap_g);

        let (y_d_r, fmap_r) = self.disc_p11.forward_ex(y_real)?;
        let (y_d_g, fmap_g) = self.disc_p11.forward_ex(y_fake)?;
        y_d_rs.push(y_d_r);
        y_d_gs.push(y_d_g);
        fmap_rs.push(fmap_r);
        fmap_gs.push(fmap_g);

        Ok((y_d_rs, y_d_gs, fmap_rs, fmap_gs))
    }
}

/// GAN Loss Functions for VITS training
pub mod losses {
    use mlx_rs::Array;
    use mlx_rs::ops::{mean, square, abs as array_abs, ones_like};

    use crate::error::Error;

    /// Generator adversarial loss (least squares GAN)
    ///
    /// Loss = sum(mean((disc_output - 1)^2))
    pub fn generator_loss(disc_outputs: &[Array]) -> Result<Array, Error> {
        let mut total_loss = Array::from_f32(0.0);

        for output in disc_outputs {
            // (output - 1)^2
            let ones = ones_like(output)
                .map_err(|e| Error::Message(e.to_string()))?;
            let diff = output.subtract(&ones)
                .map_err(|e| Error::Message(e.to_string()))?;
            let squared = square(&diff)
                .map_err(|e| Error::Message(e.to_string()))?;
            let loss = mean(&squared, false)
                .map_err(|e| Error::Message(e.to_string()))?;

            total_loss = total_loss.add(&loss)
                .map_err(|e| Error::Message(e.to_string()))?;
        }

        Ok(total_loss)
    }

    /// Discriminator adversarial loss (least squares GAN)
    ///
    /// Loss = sum(mean((real_output - 1)^2) + mean(fake_output^2))
    pub fn discriminator_loss(
        real_outputs: &[Array],
        fake_outputs: &[Array],
    ) -> Result<Array, Error> {
        let mut total_loss = Array::from_f32(0.0);

        for (real, fake) in real_outputs.iter().zip(fake_outputs.iter()) {
            // (real - 1)^2
            let ones = ones_like(real)
                .map_err(|e| Error::Message(e.to_string()))?;
            let diff_real = real.subtract(&ones)
                .map_err(|e| Error::Message(e.to_string()))?;
            let loss_real = mean(&square(&diff_real).map_err(|e| Error::Message(e.to_string()))?, false)
                .map_err(|e| Error::Message(e.to_string()))?;

            // fake^2
            let loss_fake = mean(&square(fake).map_err(|e| Error::Message(e.to_string()))?, false)
                .map_err(|e| Error::Message(e.to_string()))?;

            let loss = loss_real.add(&loss_fake)
                .map_err(|e| Error::Message(e.to_string()))?;
            total_loss = total_loss.add(&loss)
                .map_err(|e| Error::Message(e.to_string()))?;
        }

        Ok(total_loss)
    }

    /// Feature matching loss
    ///
    /// Loss = sum(mean(|real_fmap - fake_fmap|))
    pub fn feature_matching_loss(
        real_fmaps: &[Vec<Array>],
        fake_fmaps: &[Vec<Array>],
    ) -> Result<Array, Error> {
        let mut total_loss = Array::from_f32(0.0);

        for (real_fmap, fake_fmap) in real_fmaps.iter().zip(fake_fmaps.iter()) {
            for (real, fake) in real_fmap.iter().zip(fake_fmap.iter()) {
                let diff = real.subtract(fake)
                    .map_err(|e| Error::Message(e.to_string()))?;
                let abs_diff = array_abs(&diff)
                    .map_err(|e| Error::Message(e.to_string()))?;
                let loss = mean(&abs_diff, false)
                    .map_err(|e| Error::Message(e.to_string()))?;
                total_loss = total_loss.add(&loss)
                    .map_err(|e| Error::Message(e.to_string()))?;
            }
        }

        Ok(total_loss)
    }

    // ==================== Exception-returning versions for value_and_grad ====================

    /// Generator adversarial loss (Exception version for value_and_grad)
    pub fn generator_loss_ex(disc_outputs: &[Array]) -> Result<Array, mlx_rs::error::Exception> {
        let mut total_loss = Array::from_f32(0.0);

        for output in disc_outputs {
            let ones = ones_like(output)?;
            let diff = output.subtract(&ones)?;
            let squared = square(&diff)?;
            let loss = mean(&squared, false)?;
            total_loss = total_loss.add(&loss)?;
        }

        Ok(total_loss)
    }

    /// Discriminator adversarial loss (Exception version for value_and_grad)
    pub fn discriminator_loss_ex(
        real_outputs: &[Array],
        fake_outputs: &[Array],
    ) -> Result<Array, mlx_rs::error::Exception> {
        let mut total_loss = Array::from_f32(0.0);

        for (real, fake) in real_outputs.iter().zip(fake_outputs.iter()) {
            let ones = ones_like(real)?;
            let diff_real = real.subtract(&ones)?;
            let loss_real = mean(&square(&diff_real)?, false)?;
            let loss_fake = mean(&square(fake)?, false)?;
            let loss = loss_real.add(&loss_fake)?;
            total_loss = total_loss.add(&loss)?;
        }

        Ok(total_loss)
    }

    /// Feature matching loss (Exception version for value_and_grad)
    pub fn feature_matching_loss_ex(
        real_fmaps: &[Vec<Array>],
        fake_fmaps: &[Vec<Array>],
    ) -> Result<Array, mlx_rs::error::Exception> {
        let mut total_loss = Array::from_f32(0.0);

        for (real_fmap, fake_fmap) in real_fmaps.iter().zip(fake_fmaps.iter()) {
            for (real, fake) in real_fmap.iter().zip(fake_fmap.iter()) {
                let diff = real.subtract(fake)?;
                let abs_diff = array_abs(&diff)?;
                let loss = mean(&abs_diff, false)?;
                total_loss = total_loss.add(&loss)?;
            }
        }

        Ok(total_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_padding() {
        assert_eq!(get_padding(5, 1), 2);
        assert_eq!(get_padding(3, 1), 1);
        assert_eq!(get_padding(7, 1), 3);
    }

    #[test]
    fn test_mpd_config_default() {
        let config = MPDConfig::default();
        assert!(!config.use_spectral_norm);
    }
}
