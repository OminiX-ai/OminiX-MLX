//! 3D Causal Convolution
//!
//! Reference: diffusers QwenImageCausalConv3d
//! "causal padding in the time dimension and feature caching for efficient inference"

use mlx_macros::ModuleParameters;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, Param};
use mlx_rs::ops;
use mlx_rs::Array;

/// 3D Causal Convolution
/// Pads temporally only in the "past" direction for causal generation
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageCausalConv3D {
    pub in_channels: i32,
    pub out_channels: i32,
    pub kernel_size: (i32, i32, i32), // (T, H, W)
    pub stride: (i32, i32, i32),
    pub padding: (i32, i32, i32),

    #[param]
    pub weight: Param<Array>, // [out_ch, in_ch, kT, kH, kW]
    #[param]
    pub bias: Param<Option<Array>>, // [out_ch]
}

impl QwenImageCausalConv3D {
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        kernel_size: (i32, i32, i32),
        stride: (i32, i32, i32),
        padding: (i32, i32, i32),
        use_bias: bool,
    ) -> Result<Self, Exception> {
        let (kt, kh, kw) = kernel_size;
        // Xavier initialization
        let scale = (2.0 / (in_channels * kt * kh * kw) as f32).sqrt();
        let weight_shape = &[out_channels, in_channels, kt, kh, kw];
        let random_weight = mlx_rs::random::normal::<f32>(weight_shape, None, None, None)?;
        let weight = ops::multiply(&random_weight, &Array::from_f32(scale))?;

        let bias = if use_bias {
            Some(Array::zeros::<f32>(&[out_channels])?)
        } else {
            None
        };
        Ok(Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weight: Param::new(weight),
            bias: Param::new(bias),
        })
    }

    /// Apply causal padding: only pad "past" in temporal dimension
    /// Standard causal padding: pad_before = kernel_size - 1, pad_after = 0
    fn apply_causal_padding(&self, x: &Array) -> Result<Array, Exception> {
        let (pad_h, pad_w) = (self.padding.1, self.padding.2);
        let (kt, _, _) = self.kernel_size;

        // Causal temporal padding: kernel_size - 1 on the "before" side only
        let temporal_pad = kt - 1;

        if temporal_pad == 0 && pad_h == 0 && pad_w == 0 {
            return Ok(x.clone());
        }

        // x shape: [batch, channels, T, H, W]
        // Pad format for MLX: [(dim0_before, dim0_after), (dim1_before, dim1_after), ...]
        ops::pad(
            x,
            &[
                (0, 0),              // batch
                (0, 0),              // channels
                (temporal_pad, 0),   // time: CAUSAL - only pad before
                (pad_h, pad_h),      // height: symmetric
                (pad_w, pad_w),      // width: symmetric
            ],
            Array::from_f32(0.0),
            None, // mode
        )
    }
}

impl Module<&Array> for QwenImageCausalConv3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // x: [batch, channels, T, H, W] (NCTHW)
        assert_eq!(x.ndim(), 5, "Expected NCTHW format, got {} dimensions", x.ndim());

        // Apply causal padding
        let padded = self.apply_causal_padding(x)?;

        // MLX conv3d expects:
        // input: [N, D, H, W, C_in] (NDHWC)
        // weight: [C_out, D, H, W, C_in]
        // We have NCDHW, need to transpose

        // Transpose from NCTHW to NTHWC
        let input = padded.transpose_axes(&[0, 2, 3, 4, 1])?;

        // Transpose weight from [out, in, kT, kH, kW] to [out, kT, kH, kW, in]
        let weight = self.weight.transpose_axes(&[0, 2, 3, 4, 1])?;

        let (st, sh, sw) = self.stride;

        // Use conv_general with appropriate settings
        // Padding is flat array [pad_d, pad_h, pad_w] since we already applied causal padding
        let y = ops::conv_general(
            &input,
            &weight,
            &[st, sh, sw],        // strides
            &[0, 0, 0],           // padding (already applied via causal padding)
            None,                  // kernel_dilation
            None,                  // input_dilation
            None,                  // groups
            None,                  // flip
        )?;

        // Add bias if present
        let y = if let Some(ref bias) = *self.bias {
            ops::add(&y, bias)?
        } else {
            y
        };

        // Transpose back from NTHWC to NCTHW
        y.transpose_axes(&[0, 4, 1, 2, 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_conv3d_shape() {
        let conv = QwenImageCausalConv3D::new(
            4, 64,
            (3, 3, 3), (1, 1, 1), (1, 1, 1), true,
        ).unwrap();
        let x = Array::zeros::<f32>(&[1, 4, 1, 16, 16]).unwrap();
        let mut conv = conv;
        let y = conv.forward(&x).unwrap();
        // With causal padding of kt-1=2 and symmetric spatial padding of 1,
        // output should maintain spatial dims and time dim
        assert_eq!(y.dim(0), 1); // batch
        assert_eq!(y.dim(1), 64); // out_channels
    }
}
