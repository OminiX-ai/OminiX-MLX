//! Whisper-style audio encoder for Fun-ASR-Nano.
//!
//! The encoder processes mel spectrograms and outputs audio features
//! that can be projected into the LLM's embedding space.

use crate::error::Result;
use mlx_rs::builder::Builder;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::{Module, Param};
use mlx_rs::nn;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

/// Whisper encoder configuration.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WhisperEncoderConfig {
    /// Number of mel filterbank channels (typically 80)
    #[serde(default = "default_n_mels")]
    pub n_mels: i32,

    /// Hidden dimension of the encoder
    #[serde(default = "default_encoder_dim")]
    pub encoder_dim: i32,

    /// Number of encoder layers
    #[serde(default = "default_encoder_layers")]
    pub encoder_layers: i32,

    /// Number of attention heads
    #[serde(default = "default_encoder_heads")]
    pub encoder_heads: i32,

    /// Maximum sequence length
    #[serde(default = "default_max_length")]
    pub max_length: i32,
}

fn default_n_mels() -> i32 { 80 }
fn default_encoder_dim() -> i32 { 1280 }  // Whisper large
fn default_encoder_layers() -> i32 { 32 }
fn default_encoder_heads() -> i32 { 20 }
fn default_max_length() -> i32 { 1500 }  // 30s at 50Hz

impl Default for WhisperEncoderConfig {
    fn default() -> Self {
        Self {
            n_mels: default_n_mels(),
            encoder_dim: default_encoder_dim(),
            encoder_layers: default_encoder_layers(),
            encoder_heads: default_encoder_heads(),
            max_length: default_max_length(),
        }
    }
}

/// Whisper encoder attention layer.
#[derive(Debug, Clone, ModuleParameters)]
pub struct WhisperAttention {
    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub out_proj: nn::Linear,

    pub n_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
}

impl WhisperAttention {
    pub fn new(config: &WhisperEncoderConfig) -> Result<Self> {
        let dim = config.encoder_dim;
        let n_heads = config.encoder_heads;
        let head_dim = dim / n_heads;

        Ok(Self {
            q_proj: nn::LinearBuilder::new(dim, dim).build()?,
            k_proj: nn::LinearBuilder::new(dim, dim).build()?,
            v_proj: nn::LinearBuilder::new(dim, dim).build()?,
            out_proj: nn::LinearBuilder::new(dim, dim).build()?,
            n_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }
}

impl Module<&Array> for WhisperAttention {
    type Output = Array;
    type Error = mlx_rs::error::Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        let shape = x.shape();
        let (batch, seq_len, _) = (shape[0], shape[1], shape[2]);

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [B, n_heads, L, head_dim]
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Scaled dot-product attention
        let attn_out = mlx_rs::fast::scaled_dot_product_attention(
            q, k, v, self.scale, None,
        )?;

        // Reshape back to [B, L, dim]
        let attn_out = attn_out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, self.n_heads * self.head_dim])?;

        self.out_proj.forward(&attn_out)
    }
}

/// Whisper encoder MLP.
#[derive(Debug, Clone, ModuleParameters)]
pub struct WhisperMLP {
    #[param]
    pub fc1: nn::Linear,
    #[param]
    pub fc2: nn::Linear,
}

impl WhisperMLP {
    pub fn new(config: &WhisperEncoderConfig) -> Result<Self> {
        let dim = config.encoder_dim;
        let hidden_dim = dim * 4;

        Ok(Self {
            fc1: nn::LinearBuilder::new(dim, hidden_dim).build()?,
            fc2: nn::LinearBuilder::new(hidden_dim, dim).build()?,
        })
    }
}

impl Module<&Array> for WhisperMLP {
    type Output = Array;
    type Error = mlx_rs::error::Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        let h = self.fc1.forward(x)?;
        let h = nn::gelu(&h)?;
        self.fc2.forward(&h)
    }
}

/// Whisper encoder layer.
#[derive(Debug, Clone, ModuleParameters)]
pub struct WhisperEncoderLayer {
    #[param]
    pub self_attn: WhisperAttention,
    #[param]
    pub self_attn_layer_norm: nn::LayerNorm,
    #[param]
    pub mlp: WhisperMLP,
    #[param]
    pub final_layer_norm: nn::LayerNorm,
}

impl WhisperEncoderLayer {
    pub fn new(config: &WhisperEncoderConfig) -> Result<Self> {
        let dim = config.encoder_dim;

        Ok(Self {
            self_attn: WhisperAttention::new(config)?,
            self_attn_layer_norm: nn::LayerNormBuilder::new(dim).build()?,
            mlp: WhisperMLP::new(config)?,
            final_layer_norm: nn::LayerNormBuilder::new(dim).build()?,
        })
    }
}

impl Module<&Array> for WhisperEncoderLayer {
    type Output = Array;
    type Error = mlx_rs::error::Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        // Self-attention with residual
        let residual = x.clone();
        let x = self.self_attn_layer_norm.forward(x)?;
        let x = self.self_attn.forward(&x)?;
        let x = residual.add(&x)?;

        // MLP with residual
        let residual = x.clone();
        let x = self.final_layer_norm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual.add(&x)
    }
}

/// Whisper audio encoder.
///
/// Processes mel spectrograms and outputs audio features.
#[derive(Debug, Clone, ModuleParameters)]
pub struct WhisperEncoder {
    /// Initial convolution layers
    #[param]
    pub conv1: nn::Conv1d,
    #[param]
    pub conv2: nn::Conv1d,

    /// Positional embedding
    #[param]
    pub positional_embedding: Param<Array>,

    /// Transformer encoder layers
    #[param]
    pub layers: Vec<WhisperEncoderLayer>,

    /// Final layer normalization
    #[param]
    pub ln_post: nn::LayerNorm,

    /// Configuration
    pub config: WhisperEncoderConfig,
}

impl WhisperEncoder {
    /// Create a new Whisper encoder.
    pub fn new(config: WhisperEncoderConfig) -> Result<Self> {
        let n_mels = config.n_mels;
        let dim = config.encoder_dim;
        let n_layers = config.encoder_layers as usize;
        let max_len = config.max_length;

        // Convolutional frontend
        let conv1 = nn::Conv1dBuilder::new(n_mels, dim, 3)
            .padding(1)
            .build()?;
        let conv2 = nn::Conv1dBuilder::new(dim, dim, 3)
            .stride(2)
            .padding(1)
            .build()?;

        // Sinusoidal positional embedding
        let pos_embed = Self::create_positional_embedding(max_len, dim);

        // Transformer layers
        let layers: Result<Vec<_>> = (0..n_layers)
            .map(|_| WhisperEncoderLayer::new(&config))
            .collect();

        Ok(Self {
            conv1,
            conv2,
            positional_embedding: Param::new(pos_embed),
            layers: layers?,
            ln_post: nn::LayerNormBuilder::new(dim).build()?,
            config,
        })
    }

    /// Create sinusoidal positional embedding.
    fn create_positional_embedding(max_len: i32, dim: i32) -> Array {
        let mut pe = vec![0.0f32; (max_len * dim) as usize];

        for pos in 0..max_len {
            for i in 0..dim / 2 {
                let angle = pos as f32 / 10000.0f32.powf(2.0 * i as f32 / dim as f32);
                pe[(pos * dim + 2 * i) as usize] = angle.sin();
                pe[(pos * dim + 2 * i + 1) as usize] = angle.cos();
            }
        }

        Array::from_slice(&pe, &[max_len, dim])
    }

    /// Get the output dimension of the encoder.
    pub fn output_dim(&self) -> i32 {
        self.config.encoder_dim
    }
}

impl Module<&Array> for WhisperEncoder {
    type Output = Array;
    type Error = mlx_rs::error::Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, mel: &Array) -> std::result::Result<Array, Self::Error> {
        // mel: [B, n_mels, T]

        // Convolutional frontend
        let x = self.conv1.forward(mel)?;
        let x = nn::gelu(&x)?;
        let x = self.conv2.forward(&x)?;
        let x = nn::gelu(&x)?;

        // Transpose to [B, T, dim]
        let x = x.transpose_axes(&[0, 2, 1])?;

        // Add positional embedding
        let seq_len = x.shape()[1];
        let pos_embed = self.positional_embedding.as_ref().index((.., ..seq_len, ..));
        let x = x.add(&pos_embed)?;

        // Transformer layers
        let mut x = x;
        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }

        // Final layer norm
        self.ln_post.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let config = WhisperEncoderConfig::default();
        let encoder = WhisperEncoder::new(config);
        assert!(encoder.is_ok());
    }
}
