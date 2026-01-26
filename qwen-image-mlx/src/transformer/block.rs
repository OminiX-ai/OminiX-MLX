//! Transformer block for Qwen-Image
//!
//! Reference: diffusers QwenImageTransformerBlock
//! "Processes paired image and text hidden states with shared attention"

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use super::attention::QwenTransformerAttention;
use super::feedforward::QwenFeedForward;
use super::norm::QwenLayerNorm;

/// Transformer block with dual-stream processing
/// Reference: diffusers QwenImageTransformerBlock
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTransformerBlock {
    pub dim: i32,
    pub num_heads: i32,
    pub head_dim: i32,

    // Image stream
    #[param]
    pub norm1: QwenLayerNorm,
    #[param]
    pub ff: QwenFeedForward,

    // Text stream
    #[param]
    pub norm1_context: QwenLayerNorm,
    #[param]
    pub ff_context: QwenFeedForward,

    // Shared attention
    #[param]
    pub attn: QwenTransformerAttention,
}

impl QwenTransformerBlock {
    pub fn new(dim: i32, num_heads: i32, head_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            dim,
            num_heads,
            head_dim,
            norm1: QwenLayerNorm::new(dim)?,
            ff: QwenFeedForward::new(dim)?,
            norm1_context: QwenLayerNorm::new(dim)?,
            ff_context: QwenFeedForward::new(dim)?,
            attn: QwenTransformerAttention::new(dim, num_heads, head_dim)?,
        })
    }

    /// Forward pass with dual streams
    /// Returns (image_output, text_output)
    pub fn forward(
        &mut self,
        image_hidden: &Array,
        text_hidden: &Array,
        temb: &Array,           // Time embedding [batch, dim]
        image_rotary: &(Array, Array),
        text_rotary: &(Array, Array),
        mask: Option<&Array>,
    ) -> Result<(Array, Array), Exception> {
        // Image stream: norm + modulation
        let (img_modulated, img_gate1, img_mod2) = self.norm1.forward(image_hidden, temb)?;

        // Text stream: norm + modulation
        let (txt_modulated, txt_gate1, txt_mod2) = self.norm1_context.forward(text_hidden, temb)?;

        // Joint attention
        let (img_attn_out, txt_attn_out) = self.attn.forward(
            &img_modulated,
            &txt_modulated,
            image_rotary,
            text_rotary,
            mask,
        )?;

        // Image: apply gate and residual
        let img_gate1_expanded = img_gate1.expand_dims(1)?;
        let img_gated = ops::multiply(&img_attn_out, &img_gate1_expanded)?;
        let image_hidden = ops::add(image_hidden, &img_gated)?;

        // Text: apply gate and residual
        let txt_gate1_expanded = txt_gate1.expand_dims(1)?;
        let txt_gated = ops::multiply(&txt_attn_out, &txt_gate1_expanded)?;
        let text_hidden = ops::add(text_hidden, &txt_gated)?;

        // Image FFN with mod2 parameters
        let image_hidden = self.apply_ffn(&image_hidden, &img_mod2, &mut self.ff.clone())?;

        // Text FFN with mod2 parameters
        let text_hidden = self.apply_ffn(&text_hidden, &txt_mod2, &mut self.ff_context.clone())?;

        Ok((image_hidden, text_hidden))
    }

    fn apply_ffn(
        &self,
        hidden: &Array,
        mod2: &Array,
        ff: &mut QwenFeedForward,
    ) -> Result<Array, Exception> {
        use mlx_rs::module::Module;
        use mlx_rs::nn::LayerNormBuilder;

        // Split mod2 into shift2, scale2, gate2
        let shift2 = mod2.index((.., ..self.dim)).expand_dims(1)?;
        let scale2 = mod2.index((.., self.dim..self.dim * 2)).expand_dims(1)?;
        let gate2 = mod2.index((.., self.dim * 2..)).expand_dims(1)?;

        // Apply LayerNorm (non-affine)
        let mut norm = LayerNormBuilder::new(self.dim).eps(1e-6).affine(false).build()?;
        let normed = norm.forward(hidden)?;

        // Modulate: (1 + scale) * normed + shift
        let one = Array::from_f32(1.0f32);
        let scale_factor = ops::add(&one, &scale2)?;
        let modulated = ops::add(&ops::multiply(&normed, &scale_factor)?, &shift2)?;

        // FFN
        let ff_out = ff.forward(&modulated)?;

        // Gate and residual
        let gated = ops::multiply(&ff_out, &gate2)?;
        ops::add(hidden, &gated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_block() {
        let mut block = QwenTransformerBlock::new(64, 4, 16).unwrap();
        let img = Array::zeros::<f32>(&[1, 16, 64]).unwrap();
        let txt = Array::zeros::<f32>(&[1, 10, 64]).unwrap();
        let temb = Array::zeros::<f32>(&[1, 64]).unwrap();
        // RoPE dim should be head_dim/2 = 8
        let img_rot = (
            Array::zeros::<f32>(&[16, 8]).unwrap(),
            Array::zeros::<f32>(&[16, 8]).unwrap(),
        );
        let txt_rot = (
            Array::zeros::<f32>(&[10, 8]).unwrap(),
            Array::zeros::<f32>(&[10, 8]).unwrap(),
        );
        let (img_out, txt_out) = block
            .forward(&img, &txt, &temb, &img_rot, &txt_rot, None)
            .unwrap();
        assert_eq!(img_out.shape(), &[1, 16, 64]);
        assert_eq!(txt_out.shape(), &[1, 10, 64]);
    }
}
