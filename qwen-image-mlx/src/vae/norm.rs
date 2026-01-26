//! RMS Normalization for images/videos
//!
//! Reference: diffusers QwenImageRMS_norm

use mlx_macros::ModuleParameters;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, Param};
use mlx_rs::ops;
use mlx_rs::Array;

/// RMS Normalization for images/videos
/// Normalizes along channel dimension (dim 1)
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageRMSNorm {
    pub eps: f32,
    pub images: bool, // true = 4D (NCHW), false = 5D (NCTHW)

    #[param]
    pub weight: Param<Array>, // [channels]
}

impl QwenImageRMSNorm {
    pub fn new(channels: i32, eps: f32, images: bool) -> Result<Self, Exception> {
        // Weight is 1D, will be broadcast during forward
        let weight = Array::ones::<f32>(&[channels])?;
        Ok(Self {
            eps,
            images,
            weight: Param::new(weight),
        })
    }
}

impl Module<&Array> for QwenImageRMSNorm {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // RMS norm: x / sqrt(mean(x^2) + eps) * weight
        // Equivalent to: x * rsqrt(mean(x^2) + eps) * weight
        let x_squared = ops::multiply(x, x)?;
        let mean_squared = ops::mean_axis(&x_squared, 1, true)?;
        let eps_arr = Array::from_f32(self.eps);
        let variance = ops::add(&mean_squared, &eps_arr)?;
        let inv_std = ops::rsqrt(&variance)?;
        let normalized = ops::multiply(x, &inv_std)?;

        // Broadcast weight to match input dimensions
        // weight shape: [channels] -> [1, channels, 1, 1] or [1, channels, 1, 1, 1]
        let channels = self.weight.dim(0);
        let weight = if x.ndim() == 5 {
            self.weight.reshape(&[1, channels, 1, 1, 1])?
        } else {
            self.weight.reshape(&[1, channels, 1, 1])?
        };

        ops::multiply(&normalized, &weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_5d() {
        let norm = QwenImageRMSNorm::new(64, 1e-6, false).unwrap();
        let x = Array::zeros::<f32>(&[1, 64, 2, 16, 16]).unwrap();
        let mut norm = norm;
        let _y = norm.forward(&x).unwrap();
    }

    #[test]
    fn test_rms_norm_4d() {
        let norm = QwenImageRMSNorm::new(64, 1e-6, true).unwrap();
        let x = Array::zeros::<f32>(&[1, 64, 16, 16]).unwrap();
        let mut norm = norm;
        let _y = norm.forward(&x).unwrap();
    }
}
