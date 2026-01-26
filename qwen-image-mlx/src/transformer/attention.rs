//! Joint text-image attention
//!
//! Reference: diffusers QwenDoubleStreamAttnProcessor2_0

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{Linear, LinearBuilder, RmsNorm, RmsNormBuilder};
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use super::rope::apply_rope;

/// Joint text-image attention
/// Reference: diffusers QwenDoubleStreamAttnProcessor2_0
/// "Computing separate QKV projections for each stream"
/// "Concatenating streams for unified attention computation"
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTransformerAttention {
    pub dim: i32,
    pub num_heads: i32,
    pub head_dim: i32,

    // Image projections
    #[param]
    pub to_q: Linear,
    #[param]
    pub to_k: Linear,
    #[param]
    pub to_v: Linear,

    // Text projections
    #[param]
    pub add_q_proj: Linear,
    #[param]
    pub add_k_proj: Linear,
    #[param]
    pub add_v_proj: Linear,

    // QK normalization
    #[param]
    pub norm_q: RmsNorm,
    #[param]
    pub norm_k: RmsNorm,
    #[param]
    pub norm_added_q: RmsNorm,
    #[param]
    pub norm_added_k: RmsNorm,

    // Output projections
    #[param]
    pub attn_to_out: Linear,
    #[param]
    pub to_add_out: Linear,
}

impl QwenTransformerAttention {
    pub fn new(dim: i32, num_heads: i32, head_dim: i32) -> Result<Self, Exception> {
        let inner_dim = num_heads * head_dim;
        Ok(Self {
            dim,
            num_heads,
            head_dim,
            to_q: LinearBuilder::new(dim, inner_dim).build()?,
            to_k: LinearBuilder::new(dim, inner_dim).build()?,
            to_v: LinearBuilder::new(dim, inner_dim).build()?,
            add_q_proj: LinearBuilder::new(dim, inner_dim).build()?,
            add_k_proj: LinearBuilder::new(dim, inner_dim).build()?,
            add_v_proj: LinearBuilder::new(dim, inner_dim).build()?,
            norm_q: RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            norm_k: RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            norm_added_q: RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            norm_added_k: RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            attn_to_out: LinearBuilder::new(inner_dim, dim).build()?,
            to_add_out: LinearBuilder::new(inner_dim, dim).build()?,
        })
    }

    pub fn forward(
        &mut self,
        image_modulated: &Array,
        text_modulated: &Array,
        image_rotary: &(Array, Array),
        text_rotary: &(Array, Array),
        mask: Option<&Array>,
    ) -> Result<(Array, Array), Exception> {
        let (batch, img_seq, _) = (image_modulated.dim(0), image_modulated.dim(1), image_modulated.dim(2));
        let txt_seq = text_modulated.dim(1);

        // Process image QKV
        let img_q = self.to_q
            .forward(image_modulated)?
            .reshape(&[batch, img_seq, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let img_k = self.to_k
            .forward(image_modulated)?
            .reshape(&[batch, img_seq, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let img_v = self.to_v
            .forward(image_modulated)?
            .reshape(&[batch, img_seq, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let img_q = self.norm_q.forward(&img_q)?;
        let img_k = self.norm_k.forward(&img_k)?;

        // Process text QKV
        let txt_q = self.add_q_proj
            .forward(text_modulated)?
            .reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let txt_k = self.add_k_proj
            .forward(text_modulated)?
            .reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let txt_v = self.add_v_proj
            .forward(text_modulated)?
            .reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let txt_q = self.norm_added_q.forward(&txt_q)?;
        let txt_k = self.norm_added_k.forward(&txt_k)?;

        // Apply RoPE separately to image and text
        let (img_q, img_k) = apply_rope(&img_q, &img_k, image_rotary)?;
        let (txt_q, txt_k) = apply_rope(&txt_q, &txt_k, text_rotary)?;

        // Concatenate: [text, image] along sequence dimension
        let joint_q = ops::concatenate_axis(&[&txt_q, &img_q], 2)?;
        let joint_k = ops::concatenate_axis(&[&txt_k, &img_k], 2)?;
        let joint_v = ops::concatenate_axis(&[&txt_v, &img_v], 2)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt().recip();
        let scale_arr = Array::from_f32(scale);

        // Q @ K^T
        let k_t = joint_k.transpose_axes(&[0, 1, 3, 2])?;
        let mut attn_weights = ops::matmul(&joint_q, &k_t)?;
        attn_weights = ops::multiply(&attn_weights, &scale_arr)?;

        // Apply mask if present
        if let Some(m) = mask {
            attn_weights = ops::add(&attn_weights, m)?;
        }

        // Softmax and apply to V
        attn_weights = ops::softmax_axis(&attn_weights, -1, None)?;
        let output = ops::matmul(&attn_weights, &joint_v)?;

        // Transpose back: [B, heads, seq, head_dim] -> [B, seq, heads * head_dim]
        let batch = output.dim(0);
        let seq = output.dim(2);
        let output = output.transpose_axes(&[0, 2, 1, 3])?;
        let output = output.reshape(&[batch, seq, self.num_heads * self.head_dim])?;

        // Split output back to text and image
        let text_seq_len = text_modulated.dim(1);
        let text_out = output.index((.., ..text_seq_len, ..));
        let image_out = output.index((.., text_seq_len.., ..));

        // Project outputs
        let image_out = self.attn_to_out.forward(&image_out)?;
        let text_out = self.to_add_out.forward(&text_out)?;

        Ok((image_out, text_out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention() {
        let mut attn = QwenTransformerAttention::new(64, 4, 16).unwrap();
        let img = Array::zeros::<f32>(&[1, 16, 64]).unwrap();
        let txt = Array::zeros::<f32>(&[1, 10, 64]).unwrap();
        let img_rot = (
            Array::zeros::<f32>(&[16, 8]).unwrap(),
            Array::zeros::<f32>(&[16, 8]).unwrap(),
        );
        let txt_rot = (
            Array::zeros::<f32>(&[10, 8]).unwrap(),
            Array::zeros::<f32>(&[10, 8]).unwrap(),
        );
        let (img_out, txt_out) = attn.forward(&img, &txt, &img_rot, &txt_rot, None).unwrap();
        assert_eq!(img_out.shape(), &[1, 16, 64]);
        assert_eq!(txt_out.shape(), &[1, 10, 64]);
    }
}
