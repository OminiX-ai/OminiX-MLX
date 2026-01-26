//! Adaptive Layer Normalization for DiT blocks
//!
//! Reference: diffusers "Apply modulation to input tensor"

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{self, LayerNorm, LayerNormBuilder, Linear, LinearBuilder};
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

/// Adaptive Layer Normalization for DiT blocks
/// Projects conditioning to shift/scale/gate parameters
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenLayerNorm {
    pub dim: i32,
    #[param]
    pub mod_linear: Linear, // Projects to 6 * dim
    #[param]
    pub norm: LayerNorm, // Non-affine LayerNorm
}

impl QwenLayerNorm {
    pub fn new(dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            dim,
            mod_linear: LinearBuilder::new(dim, 6 * dim).build()?,
            norm: LayerNormBuilder::new(dim).eps(1e-6).affine(false).build()?,
        })
    }

    /// Returns (modulated_hidden, gate, mod2_params)
    /// - modulated_hidden: normalized and modulated by shift1/scale1
    /// - gate: gate1 for attention gating
    /// - mod2_params: [shift2, scale2, gate2] for FFN modulation
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        text_embeddings: &Array,
    ) -> Result<(Array, Array, Array), Exception> {
        // Apply silu then project: [batch, dim] -> [batch, 6*dim]
        let silu_out = nn::silu(text_embeddings)?;
        let mod_params = self.mod_linear.forward(&silu_out)?;

        // Split into mod1 and mod2: each [batch, 3*dim]
        let mod1 = mod_params.index((.., ..self.dim * 3));
        let mod2 = mod_params.index((.., self.dim * 3..));

        // Split mod1 into shift, scale, gate: each [batch, dim]
        let shift1 = mod1.index((.., ..self.dim)).expand_dims(1)?; // [batch, 1, dim]
        let scale1 = mod1.index((.., self.dim..self.dim * 2)).expand_dims(1)?;
        let gate1 = mod1.index((.., self.dim * 2..)); // [batch, dim]

        // Apply normalization and modulation
        let normed = self.norm.forward(hidden_states)?;
        let one = Array::from_f32(1.0f32);
        let scale_factor = ops::add(&one, &scale1)?;
        let modulated = ops::add(&ops::multiply(&normed, &scale_factor)?, &shift1)?;

        Ok((modulated, gate1, mod2))
    }
}

/// Continuous AdaLN for final output normalization
/// Reference: diffusers QwenImageTransformer2DModel "Apply final normalization and output projection"
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenAdaLayerNormContinuous {
    pub embedding_dim: i32,
    #[param]
    pub linear: Linear,
    #[param]
    pub norm: LayerNorm,
}

impl QwenAdaLayerNormContinuous {
    pub fn new(embedding_dim: i32, conditioning_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            embedding_dim,
            linear: LinearBuilder::new(conditioning_dim, embedding_dim * 2).build()?,
            norm: LayerNormBuilder::new(embedding_dim)
                .eps(1e-6)
                .affine(false)
                .build()?,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,
        conditioning: &Array,
    ) -> Result<Array, Exception> {
        // Project conditioning: silu -> linear
        let silu_out = nn::silu(conditioning)?;
        let cond_embeds = self.linear.forward(&silu_out)?;

        // Split into scale and shift
        let scale = cond_embeds.index((.., ..self.embedding_dim));
        let shift = cond_embeds.index((.., self.embedding_dim..));

        // Apply: norm(x) * (1 + scale) + shift
        let normed = self.norm.forward(hidden_states)?;
        let one = Array::from_f32(1.0f32);
        let scale_expanded = ops::add(&one, &scale.expand_dims(1)?)?;
        let shift_expanded = shift.expand_dims(1)?;

        let scaled = ops::multiply(&normed, &scale_expanded)?;
        ops::add(&scaled, &shift_expanded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen_layer_norm() {
        let mut norm = QwenLayerNorm::new(64).unwrap();
        let hidden = Array::zeros::<f32>(&[2, 10, 64]).unwrap();
        let text_embed = Array::zeros::<f32>(&[2, 64]).unwrap();
        let (modulated, gate, mod2) = norm.forward(&hidden, &text_embed).unwrap();
        assert_eq!(modulated.shape(), &[2, 10, 64]);
        assert_eq!(gate.shape(), &[2, 64]);
        assert_eq!(mod2.shape(), &[2, 192]); // 3 * dim
    }

    #[test]
    fn test_ada_layer_norm_continuous() {
        let mut norm = QwenAdaLayerNormContinuous::new(64, 64).unwrap();
        let hidden = Array::zeros::<f32>(&[2, 10, 64]).unwrap();
        let cond = Array::zeros::<f32>(&[2, 64]).unwrap();
        let out = norm.forward(&hidden, &cond).unwrap();
        assert_eq!(out.shape(), &[2, 10, 64]);
    }
}
