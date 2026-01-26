//! Whisper-style audio encoder for Step-Audio 2
//!
//! Architecture:
//! - Conv1d (128 → 1280, k=3, p=1) + GELU
//! - Conv1d (1280 → 1280, k=3, s=2, p=1) + GELU (2x downsample)
//! - Positional Embedding (1500, 1280)
//! - Transformer Blocks × 32
//! - AvgPool1d (k=2, s=2) (2x downsample)
//! - LayerNorm
//!
//! Total downsampling: 4x (100Hz mel → 25Hz features)
//!
//! Adapted from funasr-nano-mlx/src/whisper_encoder.rs

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::indexing::IndexOp,
    Array,
};

use crate::config::EncoderConfig;
use crate::error::Result;

// ============================================================================
// AvgPool1d Layer
// ============================================================================

/// 1D Average Pooling layer
///
/// Performs average pooling over a 1D input signal.
/// Input: [B, C, T]
/// Output: [B, C, T/stride]
#[derive(Debug, Clone, ModuleParameters)]
pub struct AvgPool1d {
    pub kernel_size: i32,
    pub stride: i32,
    pub padding: i32,
}

impl AvgPool1d {
    pub fn new(kernel_size: i32, stride: i32) -> Self {
        Self {
            kernel_size,
            stride,
            padding: 0,
        }
    }

    pub fn with_padding(kernel_size: i32, stride: i32, padding: i32) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl Module<&Array> for AvgPool1d {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        // x: [B, L, C] (NLC format for MLX)
        let shape = x.shape();
        let batch = shape[0];
        let length = shape[1];
        let channels = shape[2];

        // Calculate output length
        let padded_len = length + 2 * self.padding;
        let out_length = (padded_len - self.kernel_size) / self.stride + 1;

        // Apply padding if needed
        let x = if self.padding > 0 {
            mlx_rs::ops::pad(x, &[(0, 0), (self.padding, self.padding), (0, 0)], None, None)?
        } else {
            x.clone()
        };

        // Simple averaging approach using reshape and mean
        // For kernel_size=2, stride=2: just reshape and mean over pairs
        if self.kernel_size == 2 && self.stride == 2 && self.padding == 0 {
            // Handle odd length by dropping the last frame
            let effective_len = (length / self.kernel_size) * self.kernel_size;
            let x = if effective_len < length {
                x.index((.., ..effective_len, ..))
            } else {
                x
            };
            let out_length = effective_len / self.kernel_size;
            // Reshape from [B, L, C] to [B, L/2, 2, C] and take mean along axis 2
            let x = x.reshape(&[batch, out_length, self.kernel_size, channels])?;
            return mlx_rs::ops::mean_axis(&x, 2, false);
        }

        // General case: use strided indexing via grouped convolution
        // Create averaging kernel with all channels
        let kernel_val = 1.0 / self.kernel_size as f32;
        let kernel_data: Vec<f32> = vec![kernel_val; (channels * self.kernel_size) as usize];
        // kernel shape for depthwise conv in MLX: [out_channels, kernel_size, in_channels/groups]
        let kernel = Array::from_slice(&kernel_data, &[channels, self.kernel_size, 1]);

        // Apply grouped (depthwise) convolution
        mlx_rs::ops::conv1d(
            &x,
            &kernel,
            self.stride,
            0,  // padding already applied
            1,  // dilation
            channels,  // groups = channels for depthwise
        )
    }
}

// ============================================================================
// Encoder Attention
// ============================================================================

/// Whisper encoder attention layer
#[derive(Debug, Clone, ModuleParameters)]
pub struct EncoderAttention {
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

impl EncoderAttention {
    pub fn new(config: &EncoderConfig) -> Result<Self> {
        let dim = config.n_state;
        let n_heads = config.n_head;
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

impl Module<&Array> for EncoderAttention {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        let shape = x.shape();
        let (batch, seq_len, _) = (shape[0], shape[1], shape[2]);

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [B, n_heads, L, head_dim]
        let q = q
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[batch, seq_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Scaled dot-product attention (no mask for encoder)
        let attn_out = mlx_rs::fast::scaled_dot_product_attention(q, k, v, self.scale, None)?;

        // Reshape back to [B, L, dim]
        let attn_out = attn_out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, self.n_heads * self.head_dim])?;

        self.out_proj.forward(&attn_out)
    }
}

// ============================================================================
// Encoder MLP
// ============================================================================

/// Whisper encoder MLP (4x expansion with GELU)
#[derive(Debug, Clone, ModuleParameters)]
pub struct EncoderMLP {
    #[param]
    pub fc1: nn::Linear,
    #[param]
    pub fc2: nn::Linear,
}

impl EncoderMLP {
    pub fn new(config: &EncoderConfig) -> Result<Self> {
        let dim = config.n_state;
        let hidden_dim = dim * 4;

        Ok(Self {
            fc1: nn::LinearBuilder::new(dim, hidden_dim).build()?,
            fc2: nn::LinearBuilder::new(hidden_dim, dim).build()?,
        })
    }
}

impl Module<&Array> for EncoderMLP {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        let h = self.fc1.forward(x)?;
        let h = nn::gelu(&h)?;
        self.fc2.forward(&h)
    }
}

// ============================================================================
// Encoder Layer
// ============================================================================

/// Whisper encoder layer (pre-norm architecture)
#[derive(Debug, Clone, ModuleParameters)]
pub struct EncoderLayer {
    #[param]
    pub self_attn: EncoderAttention,
    #[param]
    pub self_attn_layer_norm: nn::LayerNorm,
    #[param]
    pub mlp: EncoderMLP,
    #[param]
    pub final_layer_norm: nn::LayerNorm,
}

impl EncoderLayer {
    pub fn new(config: &EncoderConfig) -> Result<Self> {
        let dim = config.n_state;

        Ok(Self {
            self_attn: EncoderAttention::new(config)?,
            self_attn_layer_norm: nn::LayerNormBuilder::new(dim).build()?,
            mlp: EncoderMLP::new(config)?,
            final_layer_norm: nn::LayerNormBuilder::new(dim).build()?,
        })
    }
}

impl Module<&Array> for EncoderLayer {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        // Pre-norm self-attention with residual
        let residual = x.clone();
        let h = self.self_attn_layer_norm.forward(x)?;
        let h = self.self_attn.forward(&h)?;
        let h = residual.add(&h)?;

        // Pre-norm MLP with residual
        let residual = h.clone();
        let h = self.final_layer_norm.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        residual.add(&h)
    }
}

// ============================================================================
// Full Audio Encoder
// ============================================================================

/// Step-Audio 2 Whisper-style audio encoder
///
/// Processes 128-dim mel spectrograms into 1280-dim audio features.
/// Total downsampling: 4x (conv stride 2 + avgpool stride 2)
#[derive(Debug, Clone, ModuleParameters)]
pub struct StepAudio2Encoder {
    /// Conv1: mel_dim → hidden_dim
    #[param]
    pub conv1: nn::Conv1d,
    /// Conv2: hidden_dim → hidden_dim (stride 2)
    #[param]
    pub conv2: nn::Conv1d,
    /// Positional embedding
    #[param]
    pub positional_embedding: Param<Array>,
    /// Transformer encoder layers
    #[param]
    pub layers: Vec<EncoderLayer>,
    /// Final layer normalization
    #[param]
    pub ln_post: nn::LayerNorm,
    /// Average pooling for additional 2x downsampling
    pub avg_pooler: AvgPool1d,
    /// Configuration
    pub config: EncoderConfig,
}

impl StepAudio2Encoder {
    /// Create a new Step-Audio 2 encoder
    pub fn new(config: EncoderConfig) -> Result<Self> {
        let n_mels = config.n_mels;
        let dim = config.n_state;
        let n_layers = config.n_layer as usize;
        let max_len = config.n_ctx;

        // Convolutional frontend
        // Conv1: (n_mels, dim, k=3, p=1) - no stride
        let conv1 = nn::Conv1dBuilder::new(n_mels, dim, 3)
            .padding(1)
            .build()?;

        // Conv2: (dim, dim, k=3, s=2, p=1) - 2x downsample
        let conv2 = nn::Conv1dBuilder::new(dim, dim, 3)
            .stride(2)
            .padding(1)
            .build()?;

        // Sinusoidal positional embedding
        let pos_embed = Self::create_positional_embedding(max_len, dim);

        // Transformer layers
        let layers: Result<Vec<_>> = (0..n_layers)
            .map(|_| EncoderLayer::new(&config))
            .collect();

        // Final layer norm
        let ln_post = nn::LayerNormBuilder::new(dim).build()?;

        // Average pooling: 2x downsample
        let avg_pooler = AvgPool1d::new(2, 2);

        Ok(Self {
            conv1,
            conv2,
            positional_embedding: Param::new(pos_embed),
            layers: layers?,
            ln_post,
            avg_pooler,
            config,
        })
    }

    /// Create sinusoidal positional embedding
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

    /// Get the output dimension
    pub fn output_dim(&self) -> i32 {
        self.config.n_state
    }
}

impl Module<&Array> for StepAudio2Encoder {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, mel: &Array) -> std::result::Result<Array, Self::Error> {
        // mel: [B, n_mels, T] (e.g., [1, 128, 3000])
        // MLX conv1d expects [B, L, C] format, so transpose first
        let mel = mel.transpose_axes(&[0, 2, 1])?;
        // mel: [B, T, n_mels]

        // Convolutional frontend
        let x = self.conv1.forward(&mel)?;
        let x = nn::gelu(&x)?;
        // x: [B, T, dim]
        let x = self.conv2.forward(&x)?;  // 2x downsample
        let x = nn::gelu(&x)?;
        // x: [B, T/2, dim]

        // Add positional embedding
        let seq_len = x.shape()[1];
        let pos_embed = self.positional_embedding.as_ref().index((..seq_len, ..));
        let x = x.add(&pos_embed)?;

        // Transformer layers
        let mut x = x;
        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }

        // Final layer norm
        let x = self.ln_post.forward(&x)?;
        // x: [B, T/2, dim]

        // Average pooling: 2x downsample (now in NLC format)
        let x = self.avg_pooler.forward(&x)?;
        // x: [B, T/4, dim]

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avgpool1d() {
        let mut pool = AvgPool1d::new(2, 2);
        // Input in NLC format: [batch, length, channels] = [1, 4, 2]
        let input = Array::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1, 4, 2],
        );
        let output = pool.forward(&input);
        match &output {
            Ok(_) => {}
            Err(e) => eprintln!("AvgPool1d error: {:?}", e),
        }
        assert!(output.is_ok());
        let output = output.unwrap();
        // Output: [1, 2, 2] (length halved)
        assert_eq!(output.shape(), &[1, 2, 2]);
    }

    #[test]
    fn test_encoder_config() {
        let config = EncoderConfig::default();
        assert_eq!(config.n_mels, 128);
        assert_eq!(config.n_state, 1280);
        assert_eq!(config.n_layer, 32);
    }
}
