//! Weight Normalization for Convolution Layers
//!
//! Implements weight normalization as described in "Weight Normalization: A Simple
//! Reparameterization to Accelerate Training of Deep Neural Networks" (Salimans & Kingma, 2016).
//!
//! Weight normalization separates the magnitude (g) and direction (v) of weights:
//! `weight = g * v / ||v||`
//!
//! This provides:
//! 1. Better optimization landscape (magnitude and direction trained separately)
//! 2. Implicit regularization (bounded weight magnitude)
//! 3. Training stability (prevents unbounded weight drift)
//!
//! ## Why This Matters for VITS Training
//!
//! The pretrained VITS/HiFiGAN model uses weight normalization. When finetuning
//! without weight normalization, weights can drift freely, causing:
//! - Amplitude issues (7-10x lower audio amplitude)
//! - Training instability
//! - Loss of voice quality
//!
//! With weight normalization, the optimizer updates g and v separately, constraining
//! how the weight magnitude can change.
//!
//! ## Important: Gradient Projection for weight_v
//!
//! In PyTorch, `nn.utils.weight_norm` applies a **gradient hook** that projects
//! the weight_v gradient to maintain `||v|| = constant`. The proper gradient is:
//!
//! ```text
//! dL/dv = (g/||v||) * (I - v*v^T/||v||²) * dL/dw
//! ```
//!
//! MLX's autodiff computes gradients through `weight = g * v / ||v||` without this
//! projection, causing ||v|| to grow during training. This can lead to:
//! - weight_v norm increasing by 50-100%+ after just 2 epochs
//! - Computed weights drifting significantly from pretrained values
//! - Audio quality degradation
//!
//! **Solution**: Call `normalize_v()` after each optimizer update to re-project
//! weight_v back to its original norm. This is "constrained optimization" -
//! we apply the gradient update, then project back onto the constraint manifold.
//!
//! ```rust,ignore
//! // After optimizer.update():
//! for layer in &mut model.weight_norm_layers {
//!     layer.normalize_v()?;
//! }
//! ```

use mlx_rs::{
    error::Exception,
    macros::ModuleParameters,
    module::Param,
    ops::{conv1d, conv_transpose1d, sqrt, sum_axes},
    Array,
};

/// Weight-normalized Conv1d layer
///
/// Stores separate weight_g (magnitude) and weight_v (direction), computing
/// the actual weight as: `weight = g * v / ||v||`
///
/// This matches PyTorch's `torch.nn.utils.weight_norm(Conv1d(...))`.
#[derive(Debug, Clone, ModuleParameters)]
pub struct WeightNormConv1d {
    /// Magnitude parameter: [out_channels, 1, 1]
    #[param]
    pub weight_g: Param<Array>,
    /// Direction parameter: [out_channels, kernel_size, in_channels] (MLX format)
    #[param]
    pub weight_v: Param<Array>,
    /// Optional bias: [out_channels]
    #[param]
    pub bias: Param<Option<Array>>,
    /// Input channels
    pub in_channels: i32,
    /// Output channels
    pub out_channels: i32,
    /// Kernel size
    pub kernel_size: i32,
    /// Stride
    pub stride: i32,
    /// Padding
    pub padding: i32,
    /// Dilation
    pub dilation: i32,
}

impl WeightNormConv1d {
    /// Create a new weight-normalized Conv1d layer
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        padding: i32,
        dilation: i32,
        bias: bool,
    ) -> Result<Self, Exception> {
        // Initialize weight_v with same initialization as Conv1d
        // MLX Conv1d format: [out_channels, kernel_size, in_channels]
        let scale = (1.0 / (in_channels * kernel_size) as f32).sqrt();
        let weight_v = mlx_rs::random::uniform::<f32, f32>(
            -scale,
            scale,
            &[out_channels, kernel_size, in_channels],
            None,
        )?;

        // Compute initial weight_g as L2 norm of weight_v
        // Norm computed over kernel_size and in_channels dimensions (axes 1 and 2)
        let v_squared = weight_v.square()?;
        let norm_sq = sum_axes(&v_squared, &[1, 2], true)?;
        let weight_g = sqrt(&norm_sq.add(mlx_rs::array!(1e-12f32))?)?;

        let bias_val = if bias {
            Some(Array::zeros::<f32>(&[out_channels])?)
        } else {
            None
        };

        Ok(Self {
            weight_g: Param::new(weight_g),
            weight_v: Param::new(weight_v),
            bias: Param::new(bias_val),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
        })
    }

    /// Initialize from existing weight_g and weight_v tensors
    ///
    /// Used when loading pretrained weights that have weight_g/weight_v already.
    pub fn from_weights(
        weight_g: Array,
        weight_v: Array,
        bias: Option<Array>,
        stride: i32,
        padding: i32,
        dilation: i32,
    ) -> Result<Self, Exception> {
        let shape = weight_v.shape();
        let out_channels = shape[0] as i32;
        let kernel_size = shape[1] as i32;
        let in_channels = shape[2] as i32;

        Ok(Self {
            weight_g: Param::new(weight_g),
            weight_v: Param::new(weight_v),
            bias: Param::new(bias),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
        })
    }

    /// Compute the actual weight from weight_g and weight_v
    ///
    /// `weight = g * v / ||v||`
    pub fn weight(&self) -> Result<Array, Exception> {
        // Compute L2 norm of weight_v over kernel and in_channel dims
        let v_squared = self.weight_v.as_ref().square()?;
        let norm_sq = sum_axes(&v_squared, &[1, 2], true)?;
        let norm = sqrt(&norm_sq.add(mlx_rs::array!(1e-12f32))?)?;

        // weight = g * v / ||v||
        self.weight_g
            .as_ref()
            .multiply(self.weight_v.as_ref())?
            .divide(&norm)
    }

    /// Forward pass using NLC (batch, length, channels) format
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let weight = self.weight()?;

        // Use conv1d with computed weight
        // MLX conv expects NLC input and weight in [out, kernel, in] format
        let result = conv1d(
            x,
            &weight,
            self.stride,
            self.padding,
            self.dilation,
            None,  // groups
        )?;

        // Add bias if present
        if let Some(bias) = &self.bias.value {
            result.add(bias)
        } else {
            Ok(result)
        }
    }

    /// Re-normalize weight_v to unit norm (per output channel).
    ///
    /// This should be called after each optimizer update to project weight_v
    /// back onto the constraint manifold where ||v|| = 1 per channel.
    ///
    /// PyTorch's weight_norm applies a gradient hook that does this implicitly.
    /// In MLX, we need to do it explicitly after optimizer.update().
    ///
    /// The computed weight formula is: weight = g * v / ||v||
    /// After normalization, ||v|| = 1, so: weight = g * v
    ///
    /// This means g directly controls the magnitude, and v (now unit norm)
    /// controls the direction. The optimizer can freely update both.
    pub fn normalize_v(&mut self) -> Result<(), Exception> {
        // Compute current norm over kernel and in_channel dims (axes 1 and 2)
        let v = self.weight_v.as_ref();
        let v_squared = v.square()?;
        let norm_sq = sum_axes(&v_squared, &[1, 2], true)?;
        let norm = sqrt(&norm_sq.add(mlx_rs::array!(1e-12f32))?)?;

        // Normalize v to unit norm: v_new = v / ||v||
        let v_normalized = v.divide(&norm)?;
        self.weight_v = Param::new(v_normalized);

        // NOTE: We do NOT scale g here. The gradient update to g already captures
        // the desired magnitude change. Normalizing v just ensures the direction
        // is properly constrained, while g remains as the optimizer set it.

        Ok(())
    }
}

/// Weight-normalized ConvTranspose1d layer
///
/// Stores separate weight_g (magnitude) and weight_v (direction), computing
/// the actual weight as: `weight = g * v / ||v||`
///
/// Note: For ConvTranspose, the magnitude g has shape [in_channels, 1, 1]
/// because the weight is transposed during the operation.
#[derive(Debug, Clone, ModuleParameters)]
pub struct WeightNormConvTranspose1d {
    /// Magnitude parameter: [in_channels, 1, 1] (note: in_channels, not out_channels)
    #[param]
    pub weight_g: Param<Array>,
    /// Direction parameter: [out_channels, kernel_size, in_channels] (MLX format)
    #[param]
    pub weight_v: Param<Array>,
    /// Optional bias: [out_channels]
    #[param]
    pub bias: Param<Option<Array>>,
    /// Input channels
    pub in_channels: i32,
    /// Output channels
    pub out_channels: i32,
    /// Kernel size
    pub kernel_size: i32,
    /// Stride
    pub stride: i32,
    /// Padding
    pub padding: i32,
}

impl WeightNormConvTranspose1d {
    /// Create a new weight-normalized ConvTranspose1d layer
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        padding: i32,
        bias: bool,
    ) -> Result<Self, Exception> {
        // Initialize weight_v with same initialization as ConvTranspose1d
        // MLX ConvTranspose1d format: [out_channels, kernel_size, in_channels]
        let scale = (1.0 / (out_channels * kernel_size) as f32).sqrt();
        let weight_v = mlx_rs::random::uniform::<f32, f32>(
            -scale,
            scale,
            &[out_channels, kernel_size, in_channels],
            None,
        )?;

        // Compute initial weight_g as L2 norm of weight_v over out_channels and kernel dims
        // For ConvTranspose, we norm over axes 0 and 1 (keeping in_channels)
        let v_squared = weight_v.square()?;
        let norm_sq = sum_axes(&v_squared, &[0, 1], true)?;
        // Reshape to [in_channels, 1, 1]
        let weight_g = sqrt(&norm_sq.add(mlx_rs::array!(1e-12f32))?)?
            .transpose_axes(&[2, 0, 1])?;

        let bias_val = if bias {
            Some(Array::zeros::<f32>(&[out_channels])?)
        } else {
            None
        };

        Ok(Self {
            weight_g: Param::new(weight_g),
            weight_v: Param::new(weight_v),
            bias: Param::new(bias_val),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        })
    }

    /// Initialize from existing weight_g and weight_v tensors
    ///
    /// Used when loading pretrained weights that have weight_g/weight_v already.
    pub fn from_weights(
        weight_g: Array,
        weight_v: Array,
        bias: Option<Array>,
        stride: i32,
        padding: i32,
    ) -> Result<Self, Exception> {
        let shape = weight_v.shape();
        let out_channels = shape[0] as i32;
        let kernel_size = shape[1] as i32;
        let in_channels = shape[2] as i32;

        Ok(Self {
            weight_g: Param::new(weight_g),
            weight_v: Param::new(weight_v),
            bias: Param::new(bias),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        })
    }

    /// Compute the actual weight from weight_g and weight_v
    ///
    /// `weight = g * v / ||v||`
    pub fn weight(&self) -> Result<Array, Exception> {
        // For ConvTranspose, norm over out_channels and kernel dims (axes 0, 1)
        let v_squared = self.weight_v.as_ref().square()?;
        let norm_sq = sum_axes(&v_squared, &[0, 1], true)?;
        let norm = sqrt(&norm_sq.add(mlx_rs::array!(1e-12f32))?)?;

        // Transpose weight_g from [in, 1, 1] to [1, 1, in] for broadcasting
        let g_transposed = self.weight_g.as_ref().transpose_axes(&[1, 2, 0])?;

        // weight = g * v / ||v||
        g_transposed
            .multiply(self.weight_v.as_ref())?
            .divide(&norm)
    }

    /// Forward pass using NLC (batch, length, channels) format
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let weight = self.weight()?;

        // Use conv_transpose1d with computed weight
        let result = conv_transpose1d(
            x,
            &weight,
            self.stride,
            self.padding,
            None,  // dilation
            0,     // output_padding
            None,  // groups
        )?;

        // Add bias if present
        if let Some(bias) = &self.bias.value {
            result.add(bias)
        } else {
            Ok(result)
        }
    }

    /// Re-normalize weight_v to unit norm (per input channel).
    ///
    /// This should be called after each optimizer update to project weight_v
    /// back onto the constraint manifold where ||v|| = 1 per channel.
    ///
    /// PyTorch's weight_norm applies a gradient hook that does this implicitly.
    /// In MLX, we need to do it explicitly after optimizer.update().
    ///
    /// For ConvTranspose, the norm is over out_channels and kernel dims (axes 0, 1),
    /// keeping in_channels separate (matching weight_g shape [in_channels, 1, 1]).
    ///
    /// The computed weight formula is: weight = g * v / ||v||
    /// After normalization, ||v|| = 1, so: weight = g * v
    ///
    /// **Important**: We don't scale g because:
    /// - Before normalize: weight = g * v / ||v||
    /// - After normalize:  weight = g * (v/||v||) / 1 = g * v / ||v|| (same!)
    /// So the computed weight is preserved when we don't scale g.
    pub fn normalize_v(&mut self) -> Result<(), Exception> {
        // Compute current norm over out_channels and kernel dims (axes 0, 1)
        let v = self.weight_v.as_ref();
        let v_squared = v.square()?;
        let norm_sq = sum_axes(&v_squared, &[0, 1], true)?;
        let norm = sqrt(&norm_sq.add(mlx_rs::array!(1e-12f32))?)?;

        // Normalize v to unit norm: v_new = v / ||v||
        let v_normalized = v.divide(&norm)?;
        self.weight_v = Param::new(v_normalized);

        // NOTE: We do NOT scale g. The computed weight is preserved because:
        // weight_new = g * v_new / ||v_new|| = g * (v/||v||) / 1 = g * v / ||v|| = weight_old

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::eval;

    #[test]
    fn test_normalize_v_preserves_weight() -> Result<(), Exception> {
        // Create a weight-normalized conv layer
        let mut conv = WeightNormConv1d::new(64, 128, 3, 1, 1, 1, true)?;

        // Compute weight before normalization
        let weight_before = conv.weight()?;
        eval([&weight_before])?;
        let sum_before: f32 = weight_before.sum(false)?.item();

        // Simulate gradient update that changes weight_v magnitude
        // (In real training, optimizer.update() would do this)
        let v_scaled = conv.weight_v.as_ref().multiply(mlx_rs::array!(2.0f32))?;
        conv.weight_v = Param::new(v_scaled);

        // Normalize v (should preserve the computed weight)
        conv.normalize_v()?;
        eval([conv.weight_g.as_ref(), conv.weight_v.as_ref()])?;

        // Compute weight after normalization
        let weight_after = conv.weight()?;
        eval([&weight_after])?;
        let sum_after: f32 = weight_after.sum(false)?.item();

        // Weight should be preserved (within floating point tolerance)
        let diff = (sum_after - sum_before).abs() / sum_before.abs().max(1e-6);
        assert!(diff < 0.01, "Weight changed by {:.2}% after normalize_v", diff * 100.0);

        Ok(())
    }

    #[test]
    fn test_normalize_v_convt_preserves_weight() -> Result<(), Exception> {
        // Create a weight-normalized conv transpose layer
        let mut conv = WeightNormConvTranspose1d::new(512, 256, 16, 8, 4, true)?;

        // Compute weight before normalization
        let weight_before = conv.weight()?;
        eval([&weight_before])?;
        let sum_before: f32 = weight_before.sum(false)?.item();

        // Simulate gradient update that changes weight_v magnitude
        let v_scaled = conv.weight_v.as_ref().multiply(mlx_rs::array!(1.5f32))?;
        conv.weight_v = Param::new(v_scaled);

        // Normalize v (should preserve the computed weight)
        conv.normalize_v()?;
        eval([conv.weight_g.as_ref(), conv.weight_v.as_ref()])?;

        // Compute weight after normalization
        let weight_after = conv.weight()?;
        eval([&weight_after])?;
        let sum_after: f32 = weight_after.sum(false)?.item();

        // Weight should be preserved (within floating point tolerance)
        let diff = (sum_after - sum_before).abs() / sum_before.abs().max(1e-6);
        assert!(diff < 0.01, "Weight changed by {:.2}% after normalize_v", diff * 100.0);

        Ok(())
    }

    #[test]
    fn test_weight_norm_conv1d_shape() -> Result<(), Exception> {
        let conv = WeightNormConv1d::new(64, 128, 3, 1, 1, 1, true)?;

        // Check weight_g shape: [out_channels, 1, 1]
        assert_eq!(conv.weight_g.shape(), &[128, 1, 1]);

        // Check weight_v shape: [out_channels, kernel_size, in_channels]
        assert_eq!(conv.weight_v.shape(), &[128, 3, 64]);

        // Check computed weight shape
        let weight = conv.weight()?;
        eval([&weight])?;
        assert_eq!(weight.shape(), &[128, 3, 64]);

        Ok(())
    }

    #[test]
    fn test_weight_norm_conv1d_forward() -> Result<(), Exception> {
        let mut conv = WeightNormConv1d::new(64, 128, 3, 1, 1, 1, true)?;

        // Input: [batch, length, channels] = [2, 100, 64]
        let x = Array::zeros::<f32>(&[2, 100, 64])?;
        let out = conv.forward(&x)?;
        eval([&out])?;

        // Output should be [2, 100, 128]
        assert_eq!(out.shape(), &[2, 100, 128]);

        Ok(())
    }

    #[test]
    fn test_weight_norm_convt1d_forward() -> Result<(), Exception> {
        let mut conv = WeightNormConvTranspose1d::new(512, 256, 16, 8, 4, true)?;

        // Input: [batch, length, channels] = [2, 32, 512]
        let x = Array::zeros::<f32>(&[2, 32, 512])?;
        let out = conv.forward(&x)?;
        eval([&out])?;

        // Output should be upsampled: [2, 32*8, 256] = [2, 256, 256]
        assert_eq!(out.shape(), &[2, 256, 256]);

        Ok(())
    }
}
