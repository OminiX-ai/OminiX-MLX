//! Top-level Qwen-Image Transformer
//!
//! Reference: diffusers QwenImageTransformer2DModel
//! "Main transformer model for Qwen-Image generation"

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{Linear, LinearBuilder};
use mlx_rs::Array;

use super::block::QwenTransformerBlock;
use super::embeddings::QwenTimeTextEmbed;
use super::norm::QwenAdaLayerNormContinuous;
use super::rope::QwenEmbedRope;

/// Configuration for Qwen-Image Transformer
#[derive(Debug, Clone)]
pub struct QwenTransformerConfig {
    pub patch_size: i32,
    pub in_channels: i32,
    pub num_layers: i32,
    pub attention_head_dim: i32,
    pub num_attention_heads: i32,
    pub caption_projection_dim: i32,
    pub pooled_projection_dim: i32,
    pub out_channels: i32,
    pub pos_embed_max_size: i32,
    pub axes_dimensions: [i32; 3],
    pub theta: i32,
}

impl Default for QwenTransformerConfig {
    /// Default config matches Qwen/Qwen-Image from HuggingFace
    fn default() -> Self {
        Self {
            patch_size: 2,
            in_channels: 64,              // patch dimension
            num_layers: 60,               // 60 transformer blocks
            attention_head_dim: 128,
            num_attention_heads: 24,
            caption_projection_dim: 3584, // joint_attention_dim
            pooled_projection_dim: 768,
            out_channels: 16,             // latent channels
            pos_embed_max_size: 192,
            axes_dimensions: [16, 56, 56],
            theta: 10000,
        }
    }
}

impl QwenTransformerConfig {
    pub fn inner_dim(&self) -> i32 {
        self.num_attention_heads * self.attention_head_dim
    }
}

/// Qwen-Image Transformer (DiT-style)
/// Reference: diffusers QwenImageTransformer2DModel
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTransformer {
    pub config: QwenTransformerConfig,

    // Input projections
    #[param]
    pub patch_embed: Linear, // Projects patches to inner_dim
    #[param]
    pub context_embedder: Linear, // Projects text to inner_dim

    // Time embedding
    #[param]
    pub time_text_embed: QwenTimeTextEmbed,

    // Transformer blocks
    #[param]
    pub transformer_blocks: Vec<QwenTransformerBlock>,

    // Output
    #[param]
    pub norm_out: QwenAdaLayerNormContinuous,
    #[param]
    pub proj_out: Linear,

    // RoPE (not a parameter, computed)
    pub rope: QwenEmbedRope,
}

impl QwenTransformer {
    pub fn new(config: QwenTransformerConfig) -> Result<Self, Exception> {
        let inner_dim = config.inner_dim();
        let patch_dim = config.patch_size * config.patch_size * config.in_channels;

        // Input projections
        let patch_embed = LinearBuilder::new(patch_dim, inner_dim).build()?;
        let context_embedder = LinearBuilder::new(config.caption_projection_dim, inner_dim).build()?;

        // Time embedding
        let time_text_embed = QwenTimeTextEmbed::new(256, inner_dim)?;

        // Transformer blocks
        let mut transformer_blocks = Vec::with_capacity(config.num_layers as usize);
        for _ in 0..config.num_layers {
            transformer_blocks.push(QwenTransformerBlock::new(
                inner_dim,
                config.num_attention_heads,
                config.attention_head_dim,
            )?);
        }

        // Output
        let norm_out = QwenAdaLayerNormContinuous::new(inner_dim, inner_dim)?;
        let proj_out = LinearBuilder::new(inner_dim, patch_dim).build()?;

        // RoPE
        let rope = QwenEmbedRope::new(config.theta, config.axes_dimensions, true)?;

        Ok(Self {
            config,
            patch_embed,
            context_embedder,
            time_text_embed,
            transformer_blocks,
            norm_out,
            proj_out,
            rope,
        })
    }

    /// Forward pass
    /// - hidden_states: [batch, channels, frames, height, width] latent
    /// - encoder_hidden_states: [batch, seq_len, caption_dim] text embeddings
    /// - timestep: [batch] diffusion timestep
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        encoder_hidden_states: &Array,
        timestep: &Array,
    ) -> Result<Array, Exception> {
        let _batch = hidden_states.dim(0);
        let frames = hidden_states.dim(2);
        let height = hidden_states.dim(3);
        let width = hidden_states.dim(4);

        // Patchify: [B, C, F, H, W] -> [B, num_patches, patch_dim]
        let hidden_states = self.patchify(hidden_states)?;

        // Project patches
        let hidden_states = self.patch_embed.forward(&hidden_states)?;

        // Project text
        let encoder_hidden_states = self.context_embedder.forward(encoder_hidden_states)?;

        // Time embedding
        let temb = self.time_text_embed.forward(timestep, &hidden_states)?;

        // Compute RoPE
        let patch_h = height / self.config.patch_size;
        let patch_w = width / self.config.patch_size;
        let text_len = encoder_hidden_states.dim(1);
        let ((img_cos, img_sin), (txt_cos, txt_sin)) = self.rope.forward(
            &[(frames, patch_h, patch_w)],
            &[text_len],
        )?;
        let img_rotary = (img_cos, img_sin);
        let txt_rotary = (txt_cos, txt_sin);

        // Build attention mask (optional - for padding)
        let mask = None; // Full attention for now

        // Apply transformer blocks
        let mut hidden_states = hidden_states;
        let mut encoder_hidden_states = encoder_hidden_states;
        for block in &mut self.transformer_blocks {
            let (h, e) = block.forward(
                &hidden_states,
                &encoder_hidden_states,
                &temb,
                &img_rotary,
                &txt_rotary,
                mask,
            )?;
            hidden_states = h;
            encoder_hidden_states = e;
        }

        // Final norm and projection
        let hidden_states = self.norm_out.forward(&hidden_states, &temb)?;
        let hidden_states = self.proj_out.forward(&hidden_states)?;

        // Unpatchify: [B, num_patches, patch_dim] -> [B, C, F, H, W]
        self.unpatchify(&hidden_states, frames, height, width)
    }

    /// Convert [B, C, F, H, W] to patches [B, num_patches, patch_dim]
    fn patchify(&self, x: &Array) -> Result<Array, Exception> {
        let batch = x.dim(0);
        let channels = x.dim(1);
        let frames = x.dim(2);
        let height = x.dim(3);
        let width = x.dim(4);
        let p = self.config.patch_size;

        let patch_h = height / p;
        let patch_w = width / p;

        // Reshape: [B, C, F, H, W] -> [B, C, F, pH, p, pW, p]
        let x = x.reshape(&[batch, channels, frames, patch_h, p, patch_w, p])?;

        // Permute: [B, C, F, pH, p, pW, p] -> [B, F, pH, pW, C, p, p]
        let x = x.transpose_axes(&[0, 2, 3, 5, 1, 4, 6])?;

        // Reshape: [B, F, pH, pW, C, p, p] -> [B, F*pH*pW, C*p*p]
        let num_patches = frames * patch_h * patch_w;
        let patch_dim = channels * p * p;
        x.reshape(&[batch, num_patches, patch_dim])
    }

    /// Convert patches [B, num_patches, patch_dim] back to [B, C, F, H, W]
    fn unpatchify(
        &self,
        x: &Array,
        frames: i32,
        height: i32,
        width: i32,
    ) -> Result<Array, Exception> {
        let batch = x.dim(0);
        let p = self.config.patch_size;
        let channels = self.config.out_channels;

        let patch_h = height / p;
        let patch_w = width / p;

        // Reshape: [B, F*pH*pW, C*p*p] -> [B, F, pH, pW, C, p, p]
        let x = x.reshape(&[batch, frames, patch_h, patch_w, channels, p, p])?;

        // Permute: [B, F, pH, pW, C, p, p] -> [B, C, F, pH, p, pW, p]
        let x = x.transpose_axes(&[0, 4, 1, 2, 5, 3, 6])?;

        // Reshape: [B, C, F, pH, p, pW, p] -> [B, C, F, H, W]
        x.reshape(&[batch, channels, frames, height, width])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = QwenTransformerConfig::default();
        assert_eq!(config.inner_dim(), 3072); // 24 * 128
    }

    #[test]
    fn test_patchify_unpatchify() {
        let config = QwenTransformerConfig {
            patch_size: 2,
            in_channels: 4,
            num_layers: 1,
            attention_head_dim: 16,
            num_attention_heads: 4,
            caption_projection_dim: 64,
            pooled_projection_dim: 64,
            out_channels: 4,
            pos_embed_max_size: 64,
            axes_dimensions: [4, 8, 8],
            theta: 10000,
        };
        let transformer = QwenTransformer::new(config).unwrap();

        let x = Array::zeros::<f32>(&[1, 4, 2, 8, 8]).unwrap();
        let patches = transformer.patchify(&x).unwrap();
        assert_eq!(patches.shape(), &[1, 32, 16]); // 2*4*4 patches, 4*2*2 dim

        let reconstructed = transformer.unpatchify(&patches, 2, 8, 8).unwrap();
        assert_eq!(reconstructed.shape(), &[1, 4, 2, 8, 8]);
    }
}
