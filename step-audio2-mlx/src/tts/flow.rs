//! Flow Matching Decoder for Step-Audio 2
//!
//! Implements a CosyVoice2-style Conditional Flow Matching (CFM) decoder
//! that converts semantic features to mel spectrograms.
//!
//! Architecture:
//! - Rectified Flow (linear interpolation)
//! - UNet-like estimator with cross-attention
//! - 10 denoising steps for inference
//! - Output: 80-dim mel spectrogram

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    Array,
};

use crate::error::Result;

/// Flow decoder configuration
#[derive(Debug, Clone)]
pub struct FlowDecoderConfig {
    /// Hidden dimension
    pub hidden_dim: i32,
    /// Number of attention heads
    pub num_heads: i32,
    /// Number of encoder layers
    pub num_encoder_layers: i32,
    /// Number of decoder layers
    pub num_decoder_layers: i32,
    /// Mel spectrogram dimension (output)
    pub mel_dim: i32,
    /// Semantic feature dimension (input)
    pub semantic_dim: i32,
    /// Number of denoising steps for inference
    pub num_steps: i32,
    /// Dropout rate
    pub dropout: f32,
}

impl Default for FlowDecoderConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 512,
            num_heads: 8,
            num_encoder_layers: 4,
            num_decoder_layers: 4,
            mel_dim: 80,
            semantic_dim: 512,
            num_steps: 10,
            dropout: 0.0,
        }
    }
}

/// Timestep embedding using sinusoidal encoding
#[derive(Debug, Clone, ModuleParameters)]
pub struct TimestepEmbedding {
    /// Linear projection
    #[param]
    pub linear1: nn::Linear,
    /// Second projection
    #[param]
    pub linear2: nn::Linear,
    /// Hidden dimension
    pub dim: i32,
}

impl TimestepEmbedding {
    /// Create a new timestep embedding
    pub fn new(dim: i32) -> Result<Self> {
        Ok(Self {
            linear1: nn::LinearBuilder::new(dim, dim * 4).build()?,
            linear2: nn::LinearBuilder::new(dim * 4, dim).build()?,
            dim,
        })
    }

    /// Create sinusoidal positional encoding for timestep
    fn sinusoidal_encoding(&self, t: f32) -> Array {
        let half_dim = self.dim / 2;
        let mut embed = vec![0.0f32; self.dim as usize];

        for i in 0..half_dim {
            let freq = 10000.0f32.powf(-2.0 * i as f32 / self.dim as f32);
            let angle = t * freq;
            embed[i as usize] = angle.sin();
            embed[(i + half_dim) as usize] = angle.cos();
        }

        Array::from_slice(&embed, &[1, self.dim])
    }
}

impl Module<f32> for TimestepEmbedding {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, t: f32) -> std::result::Result<Array, Self::Error> {
        let embed = self.sinusoidal_encoding(t);
        let h = self.linear1.forward(&embed)?;
        let h = nn::silu(&h)?;
        self.linear2.forward(&h)
    }
}

/// Cross-attention layer for conditioning
#[derive(Debug, Clone, ModuleParameters)]
pub struct CrossAttention {
    /// Query projection
    #[param]
    pub q_proj: nn::Linear,
    /// Key projection
    #[param]
    pub k_proj: nn::Linear,
    /// Value projection
    #[param]
    pub v_proj: nn::Linear,
    /// Output projection
    #[param]
    pub out_proj: nn::Linear,
    /// Number of heads
    pub num_heads: i32,
    /// Head dimension
    pub head_dim: i32,
    /// Scale factor
    pub scale: f32,
}

impl CrossAttention {
    /// Create a new cross-attention layer
    pub fn new(query_dim: i32, kv_dim: i32, num_heads: i32) -> Result<Self> {
        let head_dim = query_dim / num_heads;

        Ok(Self {
            q_proj: nn::LinearBuilder::new(query_dim, query_dim).build()?,
            k_proj: nn::LinearBuilder::new(kv_dim, query_dim).build()?,
            v_proj: nn::LinearBuilder::new(kv_dim, query_dim).build()?,
            out_proj: nn::LinearBuilder::new(query_dim, query_dim).build()?,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }
}

impl Module<(&Array, &Array)> for CrossAttention {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, input: (&Array, &Array)) -> std::result::Result<Array, Self::Error> {
        let (query, context) = input;

        let shape = query.shape();
        let (batch, seq_len, _) = (shape[0], shape[1], shape[2]);

        // Project Q, K, V
        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(context)?;
        let v = self.v_proj.forward(context)?;

        // Reshape to multi-head
        let q = q
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let ctx_len = context.shape()[1];
        let k = k
            .reshape(&[batch, ctx_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[batch, ctx_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Scaled dot-product attention
        let attn = mlx_rs::fast::scaled_dot_product_attention(q, k, v, self.scale, None)?;

        // Reshape back
        let attn = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;

        self.out_proj.forward(&attn)
    }
}

/// Flow estimator block (transformer-like)
#[derive(Debug, Clone, ModuleParameters)]
pub struct FlowBlock {
    /// Self-attention layer norm
    #[param]
    pub norm1: nn::LayerNorm,
    /// Self-attention
    #[param]
    pub self_attn: CrossAttention,
    /// Cross-attention layer norm
    #[param]
    pub norm2: nn::LayerNorm,
    /// Cross-attention (for conditioning)
    #[param]
    pub cross_attn: CrossAttention,
    /// FFN layer norm
    #[param]
    pub norm3: nn::LayerNorm,
    /// FFN up projection
    #[param]
    pub ffn_up: nn::Linear,
    /// FFN down projection
    #[param]
    pub ffn_down: nn::Linear,
}

impl FlowBlock {
    /// Create a new flow block
    pub fn new(hidden_dim: i32, num_heads: i32, semantic_dim: i32) -> Result<Self> {
        let ffn_dim = hidden_dim * 4;

        Ok(Self {
            norm1: nn::LayerNormBuilder::new(hidden_dim).build()?,
            self_attn: CrossAttention::new(hidden_dim, hidden_dim, num_heads)?,
            norm2: nn::LayerNormBuilder::new(hidden_dim).build()?,
            cross_attn: CrossAttention::new(hidden_dim, semantic_dim, num_heads)?,
            norm3: nn::LayerNormBuilder::new(hidden_dim).build()?,
            ffn_up: nn::LinearBuilder::new(hidden_dim, ffn_dim).build()?,
            ffn_down: nn::LinearBuilder::new(ffn_dim, hidden_dim).build()?,
        })
    }
}

impl Module<(&Array, &Array)> for FlowBlock {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, input: (&Array, &Array)) -> std::result::Result<Array, Self::Error> {
        let (x, context) = input;

        // Self-attention with residual
        let h = self.norm1.forward(x)?;
        let h = self.self_attn.forward((&h, &h))?;
        let x = x.add(&h)?;

        // Cross-attention with residual
        let h = self.norm2.forward(&x)?;
        let h = self.cross_attn.forward((&h, context))?;
        let x = x.add(&h)?;

        // FFN with residual
        let h = self.norm3.forward(&x)?;
        let h = self.ffn_up.forward(&h)?;
        let h = nn::gelu(&h)?;
        let h = self.ffn_down.forward(&h)?;
        x.add(&h)
    }
}

/// Flow estimator network (UNet-like architecture)
#[derive(Debug, Clone, ModuleParameters)]
pub struct FlowEstimator {
    /// Input projection (mel + noise → hidden)
    #[param]
    pub input_proj: nn::Linear,
    /// Timestep embedding
    #[param]
    pub time_embed: TimestepEmbedding,
    /// Semantic conditioning projection
    #[param]
    pub cond_proj: nn::Linear,
    /// Transformer blocks
    #[param]
    pub blocks: Vec<FlowBlock>,
    /// Output projection (hidden → mel)
    #[param]
    pub output_proj: nn::Linear,
    /// Final layer norm
    #[param]
    pub final_norm: nn::LayerNorm,
    /// Configuration
    pub config: FlowDecoderConfig,
}

impl FlowEstimator {
    /// Create a new flow estimator
    pub fn new(config: FlowDecoderConfig) -> Result<Self> {
        let num_blocks = config.num_encoder_layers + config.num_decoder_layers;

        // Input projection: mel_dim → hidden_dim
        let input_proj = nn::LinearBuilder::new(config.mel_dim, config.hidden_dim).build()?;

        // Timestep embedding
        let time_embed = TimestepEmbedding::new(config.hidden_dim)?;

        // Semantic conditioning projection
        let cond_proj = nn::LinearBuilder::new(config.semantic_dim, config.hidden_dim).build()?;

        // Transformer blocks
        let mut blocks = Vec::new();
        for _ in 0..num_blocks {
            blocks.push(FlowBlock::new(
                config.hidden_dim,
                config.num_heads,
                config.hidden_dim, // After projection
            )?);
        }

        // Output projection
        let output_proj = nn::LinearBuilder::new(config.hidden_dim, config.mel_dim).build()?;

        // Final layer norm
        let final_norm = nn::LayerNormBuilder::new(config.hidden_dim).build()?;

        Ok(Self {
            input_proj,
            time_embed,
            cond_proj,
            blocks,
            output_proj,
            final_norm,
            config,
        })
    }

    /// Forward pass for velocity estimation
    ///
    /// # Arguments
    /// * `x` - Noisy mel spectrogram [B, mel_dim, T]
    /// * `t` - Timestep (0.0 to 1.0)
    /// * `cond` - Semantic conditioning [B, T', semantic_dim]
    pub fn forward_with_time(
        &mut self,
        x: &Array,
        t: f32,
        cond: &Array,
    ) -> std::result::Result<Array, Exception> {
        // x: [B, mel_dim, T] → [B, T, mel_dim]
        let x = x.transpose_axes(&[0, 2, 1])?;

        // Project input
        let h = self.input_proj.forward(&x)?;

        // Add timestep embedding
        // t_embed: [1, hidden_dim] -> expand to [B, T, hidden_dim]
        let t_embed = self.time_embed.forward(t)?;
        // Reshape to [1, 1, hidden_dim] for broadcasting
        let t_embed = t_embed.reshape(&[1, 1, self.config.hidden_dim])?;
        let h = h.add(&t_embed)?;

        // Project conditioning
        let cond = self.cond_proj.forward(cond)?;

        // Apply transformer blocks
        let mut h = h;
        for block in &mut self.blocks {
            h = block.forward((&h, &cond))?;
        }

        // Output projection
        let h = self.final_norm.forward(&h)?;
        let out = self.output_proj.forward(&h)?;

        // [B, T, mel_dim] → [B, mel_dim, T]
        out.transpose_axes(&[0, 2, 1])
    }
}

/// Rectified Flow sampler
///
/// Implements the denoising loop using linear interpolation (rectified flow).
#[derive(Debug, Clone)]
pub struct FlowSampler {
    /// Number of denoising steps
    pub num_steps: i32,
}

impl FlowSampler {
    /// Create a new flow sampler
    pub fn new(num_steps: i32) -> Self {
        Self { num_steps }
    }

    /// Sample from prior (Gaussian noise)
    pub fn sample_prior(&self, shape: &[i32]) -> std::result::Result<Array, Exception> {
        mlx_rs::random::normal::<f32>(shape, None, None, None)
    }

    /// Generate timestep schedule (linear from 1.0 to 0.0)
    pub fn timestep_schedule(&self) -> Vec<f32> {
        let n = self.num_steps as f32;
        (0..=self.num_steps)
            .map(|i| 1.0 - i as f32 / n)
            .collect()
    }

    /// Single denoising step using rectified flow
    ///
    /// x_{t-dt} = x_t - dt * v(x_t, t)
    pub fn step(
        &self,
        x: &Array,
        velocity: &Array,
        t: f32,
        t_next: f32,
    ) -> std::result::Result<Array, Exception> {
        let dt = t - t_next;
        let dt_array = Array::from_slice(&[dt], &[]);

        // Euler step: x_{t-dt} = x_t - dt * v
        let dx = velocity.multiply(&dt_array)?;
        x.subtract(&dx)
    }

    /// Full denoising loop
    ///
    /// # Arguments
    /// * `estimator` - Velocity estimator network
    /// * `x_T` - Initial noise sample
    /// * `cond` - Conditioning features
    pub fn denoise<F>(
        &self,
        mut velocity_fn: F,
        x_t: &Array,
        cond: &Array,
    ) -> std::result::Result<Array, Exception>
    where
        F: FnMut(&Array, f32, &Array) -> std::result::Result<Array, Exception>,
    {
        let schedule = self.timestep_schedule();
        let mut x = x_t.clone();

        for i in 0..self.num_steps as usize {
            let t = schedule[i];
            let t_next = schedule[i + 1];

            // Estimate velocity
            let v = velocity_fn(&x, t, cond)?;

            // Euler step
            x = self.step(&x, &v, t, t_next)?;
        }

        Ok(x)
    }
}

/// Flow matching decoder
#[derive(Debug, Clone, ModuleParameters)]
pub struct FlowDecoder {
    /// Velocity estimator network
    #[param]
    pub estimator: FlowEstimator,
    /// Flow sampler
    pub sampler: FlowSampler,
    /// Configuration
    pub config: FlowDecoderConfig,
}

impl FlowDecoder {
    /// Create a new flow decoder
    pub fn new(config: FlowDecoderConfig) -> Result<Self> {
        Ok(Self {
            estimator: FlowEstimator::new(config.clone())?,
            sampler: FlowSampler::new(config.num_steps),
            config,
        })
    }

    /// Load flow decoder from model directory
    pub fn load(model_dir: impl AsRef<std::path::Path>) -> Result<Self> {
        // TODO: Implement weight loading from safetensors
        // For now, create with default config
        Self::new(FlowDecoderConfig::default())
    }

    /// Generate mel spectrogram from semantic features
    ///
    /// # Arguments
    /// * `semantic_features` - Semantic features from S3Tokenizer [B, T, semantic_dim]
    /// * `prompt_audio` - Optional reference mel for voice cloning [B, mel_dim, T']
    /// * `num_steps` - Override number of denoising steps
    pub fn generate(
        &mut self,
        semantic_features: &Array,
        prompt_audio: Option<&Array>,
        num_steps: Option<i32>,
    ) -> Result<Array> {
        let batch = semantic_features.shape()[0];
        let seq_len = semantic_features.shape()[1];

        // Estimate output mel length (roughly 2x semantic length for 25Hz → 80Hz)
        let mel_len = seq_len * 2;

        // Sample from prior
        let x_t = self
            .sampler
            .sample_prior(&[batch, self.config.mel_dim, mel_len])
            .map_err(|e| crate::error::Error::Inference(format!("Failed to sample prior: {}", e)))?;

        // Prepare conditioning (optionally concatenate prompt)
        let cond = if let Some(prompt) = prompt_audio {
            // Concatenate prompt features with semantic features
            // This is a simplified version; actual implementation may differ
            semantic_features.clone()
        } else {
            semantic_features.clone()
        };

        // Update sampler if num_steps specified
        let sampler = if let Some(steps) = num_steps {
            FlowSampler::new(steps)
        } else {
            self.sampler.clone()
        };

        // Run denoising loop
        let mel = sampler
            .denoise(
                |x, t, c| self.estimator.forward_with_time(x, t, c),
                &x_t,
                &cond,
            )
            .map_err(|e| crate::error::Error::Inference(format!("Denoising failed: {}", e)))?;

        Ok(mel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_decoder_config() {
        let config = FlowDecoderConfig::default();
        assert_eq!(config.mel_dim, 80);
        assert_eq!(config.num_steps, 10);
    }

    #[test]
    fn test_timestep_embedding() {
        let embed = TimestepEmbedding::new(64);
        assert!(embed.is_ok());
    }

    #[test]
    fn test_flow_sampler_schedule() {
        let sampler = FlowSampler::new(10);
        let schedule = sampler.timestep_schedule();

        assert_eq!(schedule.len(), 11); // 10 steps + 1 endpoint
        assert!((schedule[0] - 1.0).abs() < 1e-6);
        assert!((schedule[10] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cross_attention_creation() {
        let attn = CrossAttention::new(256, 512, 8);
        assert!(attn.is_ok());
    }

    #[test]
    fn test_flow_block_creation() {
        let block = FlowBlock::new(256, 8, 512);
        assert!(block.is_ok());
    }

    #[test]
    fn test_flow_estimator_creation() {
        let config = FlowDecoderConfig {
            hidden_dim: 128,
            num_heads: 4,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            ..Default::default()
        };
        let estimator = FlowEstimator::new(config);
        assert!(estimator.is_ok());
    }

    #[test]
    fn test_flow_decoder_creation() {
        let config = FlowDecoderConfig {
            hidden_dim: 128,
            num_heads: 4,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            ..Default::default()
        };
        let decoder = FlowDecoder::new(config);
        assert!(decoder.is_ok());
    }
}
