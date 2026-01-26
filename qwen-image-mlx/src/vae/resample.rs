//! Resampling module for 2D/3D upsampling and downsampling
//!
//! Reference: diffusers QwenImageResample

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{Conv2d, Conv2dBuilder};
use mlx_rs::ops;
use mlx_rs::Array;

use super::QwenImageCausalConv3D;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResampleMode {
    Upsample3D,
    Upsample2D,
    Downsample3D,
    Downsample2D,
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageResample3D {
    pub mode: ResampleMode,
    #[param]
    pub resample_conv: Conv2d, // 2D conv for spatial resampling
    #[param]
    pub time_conv: Option<QwenImageCausalConv3D>, // Only for 3D modes
}

impl QwenImageResample3D {
    pub fn new(channels: i32, mode: ResampleMode) -> Result<Self, Exception> {
        let (resample_conv, time_conv) = match mode {
            ResampleMode::Upsample3D => {
                let time = QwenImageCausalConv3D::new(
                    channels,
                    channels * 2,
                    (3, 1, 1),
                    (1, 1, 1),
                    (1, 0, 0),
                    true,
                )?;
                let spatial = Conv2dBuilder::new(channels, channels / 2, (3, 3))
                    .stride((1, 1))
                    .padding((1, 1))
                    .bias(true)
                    .build()?;
                (spatial, Some(time))
            }
            ResampleMode::Upsample2D => {
                let spatial = Conv2dBuilder::new(channels, channels / 2, (3, 3))
                    .stride((1, 1))
                    .padding((1, 1))
                    .bias(true)
                    .build()?;
                (spatial, None)
            }
            ResampleMode::Downsample3D => {
                let time = QwenImageCausalConv3D::new(
                    channels,
                    channels,
                    (3, 1, 1),
                    (2, 1, 1),
                    (0, 0, 0),
                    true,
                )?;
                let spatial = Conv2dBuilder::new(channels, channels, (3, 3))
                    .stride((2, 2))
                    .padding((0, 0))
                    .bias(true)
                    .build()?;
                (spatial, Some(time))
            }
            ResampleMode::Downsample2D => {
                let spatial = Conv2dBuilder::new(channels, channels, (3, 3))
                    .stride((2, 2))
                    .padding((0, 0))
                    .bias(true)
                    .build()?;
                (spatial, None)
            }
        };
        Ok(Self {
            mode,
            resample_conv,
            time_conv,
        })
    }

    /// Nearest neighbor 2D upsampling using repeat (matches mflux implementation)
    fn nearest_upsample_2d(x: &Array, scale: i32) -> Result<Array, Exception> {
        // x: [batch, H, W, channels] (NHWC)
        // Use repeat to duplicate rows and columns (like mflux)
        let stream = mlx_rs::Stream::default();

        // Debug: check input
        static DEBUG_UP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_UP.swap(true, std::sync::atomic::Ordering::SeqCst) {
            mlx_rs::transforms::eval([x]).ok();
            eprintln!("[UPSAMPLE] Input shape: {:?}, range: [{:.3}, {:.3}]",
                x.shape(),
                x.min(None).unwrap().item::<f32>(),
                x.max(None).unwrap().item::<f32>());
        }

        let repeated_h = mlx_rs::ops::repeat_axis_device::<f32>(x.clone(), scale, 1, &stream)?;
        let repeated_hw = mlx_rs::ops::repeat_axis_device::<f32>(repeated_h, scale, 2, &stream)?;

        // Debug: check output
        static DEBUG_UP2: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_UP2.swap(true, std::sync::atomic::Ordering::SeqCst) {
            mlx_rs::transforms::eval([&repeated_hw]).ok();
            eprintln!("[UPSAMPLE] Output shape: {:?}, range: [{:.3}, {:.3}]",
                repeated_hw.shape(),
                repeated_hw.min(None).unwrap().item::<f32>(),
                repeated_hw.max(None).unwrap().item::<f32>());
        }

        Ok(repeated_hw)
    }
}

impl Module<&Array> for QwenImageResample3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        self.resample_conv.training_mode(mode);
        if let Some(ref mut tc) = self.time_conv {
            tc.training_mode(mode);
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let mut h = x.clone();

        // For Upsample3D: apply temporal conv FIRST (before spatial operations)
        // time_conv expects the original channel count
        // IMPORTANT: Only apply temporal upsampling for video (T > 1)
        // For single images (T=1), skip temporal processing to avoid artifacts
        if self.mode == ResampleMode::Upsample3D {
            if let Some(ref mut time_conv) = self.time_conv {
                let t = h.dim(2);
                if t > 1 {
                    // Temporal conv doubles channels: C -> 2C
                    h = time_conv.forward(&h)?;
                    // Pixel shuffle in time dimension: [B, 2C, T, H, W] -> [B, C, 2T, H, W]
                    let (b, c, t, hh, ww) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3), h.dim(4));
                    h = h.reshape(&[b, c / 2, 2, t, hh, ww])?;
                    h = h.transpose_axes(&[0, 1, 3, 2, 4, 5])?; // [B, C/2, T, 2, H, W]
                    h = h.reshape(&[b, c / 2, t * 2, hh, ww])?;
                }
                // For T=1, skip temporal processing (no pixel shuffle needed)
            }
        }

        // Apply temporal conv for 3D downsample if T >= 3
        if self.mode == ResampleMode::Downsample3D {
            if let Some(ref mut time_conv) = self.time_conv {
                if h.dim(2) >= 3 {
                    h = time_conv.forward(&h)?;
                }
            }
        }

        let (batch, channels, time, height, width) =
            (h.dim(0), h.dim(1), h.dim(2), h.dim(3), h.dim(4));

        // Reshape to 2D: [B, C, T, H, W] -> [B*T, H, W, C]
        h = h.transpose_axes(&[0, 2, 1, 3, 4])?; // [B, T, C, H, W]
        h = h.reshape(&[batch * time, channels, height, width])?;
        h = h.transpose_axes(&[0, 2, 3, 1])?; // [B*T, H, W, C]

        // Upsample spatial (nearest neighbor 2x)
        if self.mode == ResampleMode::Upsample3D || self.mode == ResampleMode::Upsample2D {
            h = Self::nearest_upsample_2d(&h, 2)?;
        }

        // Pad for downsample
        if self.mode == ResampleMode::Downsample2D || self.mode == ResampleMode::Downsample3D {
            h = ops::pad(
                &h,
                &[(0, 0), (0, 1), (0, 1), (0, 0)],
                Array::from_f32(0.0),
                None, // mode
            )?;
        }

        // Apply spatial conv (expects NHWC)
        h = self.resample_conv.forward(&h)?;

        // Reshape back to 5D: [B*T, H', W', C'] -> [B, C', T, H', W']
        let (new_h, new_w, new_c) = (h.dim(1), h.dim(2), h.dim(3));
        h = h.transpose_axes(&[0, 3, 1, 2])?; // [B*T, C', H', W']
        h = h.reshape(&[batch, time, new_c, new_h, new_w])?;
        h = h.transpose_axes(&[0, 2, 1, 3, 4])?; // [B, C', T, H', W']

        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downsample_2d() {
        let resample = QwenImageResample3D::new(64, ResampleMode::Downsample2D).unwrap();
        let x = Array::zeros::<f32>(&[1, 64, 1, 16, 16]).unwrap();
        let mut resample = resample;
        let y = resample.forward(&x).unwrap();
        // Downsample should halve spatial dims
        assert_eq!(y.dim(3), 8);
        assert_eq!(y.dim(4), 8);
    }
}
