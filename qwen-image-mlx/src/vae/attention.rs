//! 3D Attention Block for VAE
//!
//! Reference: diffusers QwenImageAttentionBlock
//! "causal self-attention with a single head"

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{Conv2d, Conv2dBuilder};
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use super::QwenImageRMSNorm;

/// 3D Attention Block for VAE
/// Applies self-attention on flattened H*W per time frame
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageAttentionBlock3D {
    pub channels: i32,
    #[param]
    pub norm: QwenImageRMSNorm,
    #[param]
    pub to_qkv: Conv2d, // Projects to Q, K, V combined
    #[param]
    pub proj: Conv2d, // Output projection
}

impl QwenImageAttentionBlock3D {
    pub fn new(channels: i32) -> Result<Self, Exception> {
        Ok(Self {
            channels,
            norm: QwenImageRMSNorm::new(channels, 1e-12, true)?, // images=true (4D)
            to_qkv: Conv2dBuilder::new(channels, channels * 3, (1, 1))
                .stride((1, 1))
                .padding((0, 0))
                .bias(true)
                .build()?,
            proj: Conv2dBuilder::new(channels, channels, (1, 1))
                .stride((1, 1))
                .padding((0, 0))
                .bias(true)
                .build()?,
        })
    }
}

impl Module<&Array> for QwenImageAttentionBlock3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        self.norm.training_mode(mode);
        self.to_qkv.training_mode(mode);
        self.proj.training_mode(mode);
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let identity = x.clone();
        let (batch, channels, time, height, width) =
            (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4));

        // Process each time frame separately
        // Reshape: [B, C, T, H, W] -> [B*T, C, H, W]
        let mut h = x.transpose_axes(&[0, 2, 1, 3, 4])?; // [B, T, C, H, W]
        h = h.reshape(&[batch * time, channels, height, width])?;

        // Apply norm (expects 4D)
        h = self.norm.forward(&h)?;

        // Transpose to NHWC for conv2d
        h = h.transpose_axes(&[0, 2, 3, 1])?; // [B*T, H, W, C]

        // Get Q, K, V
        let qkv = self.to_qkv.forward(&h)?; // [B*T, H, W, 3*C]
        let qkv = qkv.reshape(&[batch * time, height * width, channels * 3])?;

        // Split into Q, K, V
        let q = qkv.index((.., .., ..channels));
        let k = qkv.index((.., .., channels..channels * 2));
        let v = qkv.index((.., .., channels * 2..));

        // Attention: softmax(Q @ K^T / sqrt(C)) @ V
        let scale = (channels as f32).sqrt();
        let k_t = k.transpose_axes(&[0, 2, 1])?; // [B*T, C, H*W]
        let attn_weights = ops::matmul(&q, &k_t)?; // [B*T, H*W, H*W]
        let attn_weights = ops::divide(&attn_weights, &Array::from_f32(scale))?;
        let attn_weights = ops::softmax_axis(&attn_weights, -1, None)?;

        let attn_out = ops::matmul(&attn_weights, &v)?; // [B*T, H*W, C]
        let attn_out = attn_out.reshape(&[batch * time, height, width, channels])?;

        // Output projection
        let attn_out = self.proj.forward(&attn_out)?; // [B*T, H, W, C]

        // Reshape back to [B, C, T, H, W]
        let attn_out = attn_out.transpose_axes(&[0, 3, 1, 2])?; // [B*T, C, H, W]
        let attn_out = attn_out.reshape(&[batch, time, channels, height, width])?;
        let attn_out = attn_out.transpose_axes(&[0, 2, 1, 3, 4])?; // [B, C, T, H, W]

        ops::add(&identity, &attn_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_block() {
        let attn = QwenImageAttentionBlock3D::new(64).unwrap();
        let x = Array::zeros::<f32>(&[1, 64, 1, 8, 8]).unwrap();
        let mut attn = attn;
        let y = attn.forward(&x).unwrap();
        assert_eq!(y.shape(), x.shape());
    }
}
