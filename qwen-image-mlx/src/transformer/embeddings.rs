//! Timestep embeddings for diffusion
//!
//! Reference: diffusers QwenImageTransformer2DModel "Generate timestep embeddings"

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{self, Linear, LinearBuilder};
use mlx_rs::ops;
use mlx_rs::Dtype;
use mlx_rs::Array;

/// Timestep projection using sinusoidal embeddings
#[derive(Debug, Clone)]
pub struct QwenTimesteps {
    pub projection_dim: i32,
    pub scale: f32,
}

impl QwenTimesteps {
    pub fn new(projection_dim: i32, scale: f32) -> Self {
        Self {
            projection_dim,
            scale,
        }
    }

    pub fn forward(&self, timesteps: &Array) -> Result<Array, Exception> {
        let half_dim = self.projection_dim / 2;
        let max_period: f32 = 10_000.0;

        // Compute frequencies: exp(-log(max_period) * i / half_dim)
        let indices: Vec<f32> = (0..half_dim).map(|i| i as f32).collect();
        let indices = Array::from_slice(&indices, &[half_dim]);
        let log_max = Array::from_f32(-max_period.ln());
        let half_dim_f = Array::from_f32(half_dim as f32);
        let exponent = ops::multiply(&log_max, &ops::divide(&indices, &half_dim_f)?)?;
        let freqs = ops::exp(&exponent)?;

        // Compute embeddings: timesteps[:, None] * freqs[None, :] * scale
        let t = timesteps.as_dtype(Dtype::Float32)?.expand_dims(-1)?;
        let freqs_expanded = freqs.expand_dims(0)?;
        let scale_arr = Array::from_f32(self.scale);
        let emb = ops::multiply(&ops::multiply(&t, &freqs_expanded)?, &scale_arr)?;

        // Concatenate sin and cos
        let sin_emb = ops::sin(&emb)?;
        let cos_emb = ops::cos(&emb)?;

        // Swap halves: [sin, cos] -> [cos, sin] (matches diffusers)
        ops::concatenate_axis(&[&cos_emb, &sin_emb], -1)
    }
}

/// Timestep embedding MLP
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTimestepEmbedding {
    #[param]
    pub linear_1: Linear,
    #[param]
    pub linear_2: Linear,
}

impl QwenTimestepEmbedding {
    pub fn new(projection_dim: i32, inner_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            linear_1: LinearBuilder::new(projection_dim, inner_dim).build()?,
            linear_2: LinearBuilder::new(inner_dim, inner_dim).build()?,
        })
    }
}

impl Module<&Array> for QwenTimestepEmbedding {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        self.linear_1.training_mode(mode);
        self.linear_2.training_mode(mode);
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let h = self.linear_1.forward(x)?;
        let h = nn::silu(&h)?;
        self.linear_2.forward(&h)
    }
}

/// Combined time-text embedding
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTimeTextEmbed {
    pub time_proj: QwenTimesteps,
    #[param]
    pub timestep_embedder: QwenTimestepEmbedding,
}

impl QwenTimeTextEmbed {
    pub fn new(timestep_projection_dim: i32, inner_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            time_proj: QwenTimesteps::new(timestep_projection_dim, 1000.0),
            timestep_embedder: QwenTimestepEmbedding::new(timestep_projection_dim, inner_dim)?,
        })
    }

    pub fn forward(&mut self, timestep: &Array, hidden_states: &Array) -> Result<Array, Exception> {
        let proj = self.time_proj.forward(timestep)?;
        let proj = proj.as_dtype(hidden_states.dtype())?;
        self.timestep_embedder.forward(&proj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timesteps() {
        let timesteps = QwenTimesteps::new(256, 1000.0);
        let t = Array::from_slice(&[0.5f32, 0.8f32], &[2]);
        let emb = timesteps.forward(&t).unwrap();
        assert_eq!(emb.shape(), &[2, 256]);
    }

    #[test]
    fn test_time_text_embed() {
        let mut embed = QwenTimeTextEmbed::new(256, 3072).unwrap();
        let t = Array::from_slice(&[0.5f32, 0.8f32], &[2]);
        let hidden = Array::zeros::<f32>(&[2, 10, 3072]).unwrap();
        let out = embed.forward(&t, &hidden).unwrap();
        assert_eq!(out.shape(), &[2, 3072]);
    }
}
