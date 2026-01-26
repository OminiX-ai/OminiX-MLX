//! Audio adaptor for Fun-ASR-Nano.
//!
//! Projects audio encoder output to LLM input dimension through
//! linear projections and transformer layers.

use crate::error::Result;
use mlx_rs::builder::Builder;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::Array;

/// Adaptor configuration.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AdaptorConfig {
    /// Encoder output dimension
    #[serde(default = "default_encoder_dim")]
    pub encoder_dim: i32,

    /// FFN intermediate dimension
    #[serde(default = "default_ffn_dim")]
    pub ffn_dim: i32,

    /// LLM input dimension
    #[serde(default = "default_llm_dim")]
    pub llm_dim: i32,

    /// Number of transformer layers
    #[serde(default = "default_n_layer")]
    pub n_layer: i32,

    /// Downsample rate (usually 1)
    #[serde(default = "default_downsample_rate")]
    pub downsample_rate: i32,
}

fn default_encoder_dim() -> i32 { 512 }
fn default_ffn_dim() -> i32 { 2048 }
fn default_llm_dim() -> i32 { 1024 }
fn default_n_layer() -> i32 { 2 }
fn default_downsample_rate() -> i32 { 1 }

impl Default for AdaptorConfig {
    fn default() -> Self {
        Self {
            encoder_dim: 512,
            ffn_dim: 2048,
            llm_dim: 1024,
            n_layer: 2,
            downsample_rate: 1,
        }
    }
}

/// Adaptor self-attention layer (separate Q/K/V projections).
#[derive(Debug, Clone, ModuleParameters)]
pub struct AdaptorAttention {
    #[param]
    pub linear_q: nn::Linear,
    #[param]
    pub linear_k: nn::Linear,
    #[param]
    pub linear_v: nn::Linear,
    #[param]
    pub linear_out: nn::Linear,

    pub n_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
}

impl AdaptorAttention {
    pub fn new(dim: i32, n_heads: i32) -> Result<Self> {
        let head_dim = dim / n_heads;

        Ok(Self {
            linear_q: nn::LinearBuilder::new(dim, dim).bias(true).build()?,
            linear_k: nn::LinearBuilder::new(dim, dim).bias(true).build()?,
            linear_v: nn::LinearBuilder::new(dim, dim).bias(true).build()?,
            linear_out: nn::LinearBuilder::new(dim, dim).bias(true).build()?,
            n_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    pub fn forward(&mut self, x: &Array) -> std::result::Result<Array, mlx_rs::error::Exception> {
        let shape = x.shape();
        let (batch, seq_len, _) = (shape[0], shape[1], shape[2]);

        // Separate Q, K, V projections
        let q = self.linear_q.forward(x)?;
        let k = self.linear_k.forward(x)?;
        let v = self.linear_v.forward(x)?;

        // Reshape for multi-head attention
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Scaled dot-product attention
        let attn_out = mlx_rs::fast::scaled_dot_product_attention(
            q, k, v, self.scale, None,
        )?;

        // Reshape back to [batch, seq, dim]
        let attn_out = attn_out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, self.n_heads * self.head_dim])?;

        self.linear_out.forward(&attn_out)
    }
}

/// Adaptor FFN with bottleneck architecture.
#[derive(Debug, Clone, ModuleParameters)]
pub struct AdaptorFFN {
    #[param]
    pub w_1: nn::Linear,
    #[param]
    pub w_2: nn::Linear,
}

impl AdaptorFFN {
    /// Create with bottleneck: dim -> hidden_dim -> dim
    /// For Fun-ASR: 1024 -> 256 -> 1024 (bottleneck)
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self> {
        Ok(Self {
            w_1: nn::LinearBuilder::new(dim, hidden_dim).bias(true).build()?,
            w_2: nn::LinearBuilder::new(hidden_dim, dim).bias(true).build()?,
        })
    }
}

impl Module<&Array> for AdaptorFFN {
    type Output = Array;
    type Error = mlx_rs::error::Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        let h = self.w_1.forward(x)?;
        let h = nn::relu(&h)?;
        self.w_2.forward(&h)
    }
}

/// Adaptor transformer block.
#[derive(Debug, Clone, ModuleParameters)]
pub struct AdaptorBlock {
    #[param]
    pub self_attn: AdaptorAttention,
    #[param]
    pub feed_forward: AdaptorFFN,
    #[param]
    pub norm1: nn::LayerNorm,
    #[param]
    pub norm2: nn::LayerNorm,
}

impl AdaptorBlock {
    pub fn new(dim: i32, n_heads: i32, ffn_hidden_dim: i32) -> Result<Self> {
        Ok(Self {
            self_attn: AdaptorAttention::new(dim, n_heads)?,
            feed_forward: AdaptorFFN::new(dim, ffn_hidden_dim)?,
            norm1: nn::LayerNormBuilder::new(dim).build()?,
            norm2: nn::LayerNormBuilder::new(dim).build()?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> std::result::Result<Array, mlx_rs::error::Exception> {
        // Pre-norm architecture
        let residual = x.clone();
        let h = self.norm1.forward(x)?;
        let h = self.self_attn.forward(&h)?;
        let h = residual.add(&h)?;

        let residual = h.clone();
        let h = self.norm2.forward(&h)?;
        let h = self.feed_forward.forward(&h)?;
        residual.add(&h)
    }
}

/// Audio adaptor.
///
/// Projects audio encoder output to LLM input dimension through:
/// 1. linear1: encoder_dim (512) -> ffn_dim (2048)
/// 2. linear2: ffn_dim (2048) -> llm_dim (1024)
/// 3. transformer blocks: 2 layers of self-attention + FFN
#[derive(Debug, Clone, ModuleParameters)]
pub struct AudioAdaptor {
    /// Input projection: encoder_dim -> ffn_dim
    #[param]
    pub linear1: nn::Linear,

    /// Output projection: ffn_dim -> llm_dim
    #[param]
    pub linear2: nn::Linear,

    /// Transformer blocks
    #[param]
    pub blocks: Vec<AdaptorBlock>,

    pub config: AdaptorConfig,
}

impl AudioAdaptor {
    pub fn new(config: AdaptorConfig) -> Result<Self> {
        // Input projection: encoder_dim (512) -> ffn_dim (2048)
        let linear1 = nn::LinearBuilder::new(config.encoder_dim, config.ffn_dim)
            .bias(true)
            .build()?;

        // Output projection: ffn_dim (2048) -> llm_dim (1024)
        let linear2 = nn::LinearBuilder::new(config.ffn_dim, config.llm_dim)
            .bias(true)
            .build()?;

        // Transformer blocks operating at llm_dim
        let n_heads = 8;  // Typical for 1024-dim
        let ffn_hidden = 256;  // Bottleneck FFN
        let blocks: Result<Vec<_>> = (0..config.n_layer)
            .map(|_| AdaptorBlock::new(config.llm_dim, n_heads, ffn_hidden))
            .collect();

        Ok(Self {
            linear1,
            linear2,
            blocks: blocks?,
            config,
        })
    }
}

impl Module<&Array> for AudioAdaptor {
    type Output = Array;
    type Error = mlx_rs::error::Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        // Input projection: [batch, time, 512] -> [batch, time, 2048]
        let h = self.linear1.forward(x)?;
        let h = nn::relu(&h)?;

        // Output projection: [batch, time, 2048] -> [batch, time, 1024]
        let mut h = self.linear2.forward(&h)?;

        // Transformer blocks
        for block in &mut self.blocks {
            h = block.forward(&h)?;
        }

        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptor_creation() {
        let config = AdaptorConfig::default();
        let adaptor = AudioAdaptor::new(config);
        assert!(adaptor.is_ok());
    }
}
