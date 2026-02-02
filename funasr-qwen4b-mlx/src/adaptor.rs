//! Audio Adaptor for Qwen4B
//!
//! 4-layer transformer that projects SenseVoice encoder output (512-dim)
//! into Qwen3-4B embedding space (2560-dim).
//!
//! Architecture:
//! - input_proj: Linear(512 → 2048)
//! - transformer: 4 × TransformerEncoderLayer(2048, 8 heads, ffn=8192, ReLU)
//! - output_proj: Linear(2048 → 2560)
//! - norm: LayerNorm(2560)

use mlx_rs::{array, Array};
use mlx_rs::builder::Builder;
use mlx_rs::module::Module;
use mlx_rs::nn;

use crate::error::{Error, Result};

/// Configuration for AudioAdaptorQwen4B
#[derive(Debug, Clone)]
pub struct AdaptorConfig {
    pub input_dim: i32,      // 512 (from SenseVoice)
    pub hidden_dim: i32,     // 2048
    pub output_dim: i32,     // 2560 (Qwen4B embedding)
    pub num_layers: i32,     // 4
    pub num_heads: i32,      // 8
    pub ffn_dim: i32,        // 8192 (hidden_dim * 4)
    pub dropout: f32,        // 0.1 (only used in training)
}

impl Default for AdaptorConfig {
    fn default() -> Self {
        Self {
            input_dim: 512,
            hidden_dim: 2048,
            output_dim: 2560,
            num_layers: 4,
            num_heads: 8,
            ffn_dim: 8192,
            dropout: 0.1,
        }
    }
}

/// Transformer Encoder Layer with GELU activation
/// Matches training script: nn.TransformerEncoderLayer(..., activation="gelu", batch_first=True)
pub struct TransformerEncoderLayer {
    // Self-attention
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub out_proj: nn::Linear,

    // Feed-forward network
    pub ffn_linear1: nn::Linear,
    pub ffn_linear2: nn::Linear,

    // Layer norms (pre-norm style for PyTorch TransformerEncoderLayer)
    pub norm1: nn::LayerNorm,
    pub norm2: nn::LayerNorm,

    // Config
    pub num_heads: i32,
    pub head_dim: i32,
}

impl TransformerEncoderLayer {
    pub fn new(hidden_dim: i32, num_heads: i32, ffn_dim: i32) -> Self {
        let head_dim = hidden_dim / num_heads;

        Self {
            q_proj: nn::Linear::new(hidden_dim, hidden_dim).unwrap(),
            k_proj: nn::Linear::new(hidden_dim, hidden_dim).unwrap(),
            v_proj: nn::Linear::new(hidden_dim, hidden_dim).unwrap(),
            out_proj: nn::Linear::new(hidden_dim, hidden_dim).unwrap(),

            ffn_linear1: nn::Linear::new(hidden_dim, ffn_dim).unwrap(),
            ffn_linear2: nn::Linear::new(ffn_dim, hidden_dim).unwrap(),

            norm1: nn::LayerNormBuilder::new(hidden_dim).eps(1e-5).affine(true).build().unwrap(),
            norm2: nn::LayerNormBuilder::new(hidden_dim).eps(1e-5).affine(true).build().unwrap(),

            num_heads,
            head_dim,
        }
    }

    /// Forward pass with self-attention and FFN
    /// Input shape: [batch, seq_len, hidden_dim]
    ///
    /// Uses POST-norm like PyTorch nn.TransformerEncoderLayer (default):
    /// x = norm1(x + self_attn(x))
    /// x = norm2(x + ffn(x))
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let shape = x.shape();
        let batch_size = shape[0] as i32;
        let seq_len = shape[1] as i32;

        // Self-attention
        // Q, K, V projections (on original x, not normalized)
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let q = q.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
                 .transpose_axes(&[0, 2, 1, 3])?;
        let k = k.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
                 .transpose_axes(&[0, 2, 1, 3])?;
        let v = v.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
                 .transpose_axes(&[0, 2, 1, 3])?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        let scores = mlx_rs::ops::matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?;
        let scores = mlx_rs::ops::divide(&scores, &array!(scale))?;
        let attn_weights = mlx_rs::ops::softmax_axis(&scores, -1, None)?;

        // Apply attention to values
        let attn_out = mlx_rs::ops::matmul(&attn_weights, &v)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_out = attn_out.transpose_axes(&[0, 2, 1, 3])?
                               .reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        // Output projection
        let attn_out = self.out_proj.forward(&attn_out)?;

        // POST-norm: add residual then normalize
        let x = mlx_rs::ops::add(x, &attn_out)?;
        let x = self.norm1.forward(&x)?;

        // Feed-forward with GELU (matches training script: activation="gelu")
        let ffn_out = self.ffn_linear1.forward(&x)?;
        let ffn_out = mlx_rs::nn::gelu(&ffn_out)?;  // GELU
        let ffn_out = self.ffn_linear2.forward(&ffn_out)?;

        // POST-norm: add residual then normalize
        let out = mlx_rs::ops::add(&x, &ffn_out)?;
        let out = self.norm2.forward(&out)?;

        Ok(out)
    }

    /// Set weight by key (called from AudioAdaptorQwen4B::load_weights)
    pub fn set_weight(&mut self, key: &str, value: Array) {
        match key {
            "q_proj.weight" => *self.q_proj.weight = value,
            "k_proj.weight" => *self.k_proj.weight = value,
            "v_proj.weight" => *self.v_proj.weight = value,
            "out_proj.weight" => *self.out_proj.weight = value,
            "ffn_linear1.weight" | "linear1.weight" => *self.ffn_linear1.weight = value,
            "ffn_linear2.weight" | "linear2.weight" => *self.ffn_linear2.weight = value,
            "norm1.weight" => *self.norm1.weight = Some(value),
            "norm2.weight" => *self.norm2.weight = Some(value),
            // Bias handling - need to wrap in Some for Param<Option<Array>>
            "q_proj.bias" => *self.q_proj.bias = Some(value),
            "k_proj.bias" => *self.k_proj.bias = Some(value),
            "v_proj.bias" => *self.v_proj.bias = Some(value),
            "out_proj.bias" => *self.out_proj.bias = Some(value),
            "ffn_linear1.bias" | "linear1.bias" => *self.ffn_linear1.bias = Some(value),
            "ffn_linear2.bias" | "linear2.bias" => *self.ffn_linear2.bias = Some(value),
            "norm1.bias" => *self.norm1.bias = Some(value),
            "norm2.bias" => *self.norm2.bias = Some(value),
            _ => {} // Ignore unknown keys
        }
    }
}

/// Audio Adaptor for Qwen4B
///
/// Matches the PyTorch AudioAdaptorQwen4B architecture:
/// - input_proj: 512 → 2048
/// - 4 transformer encoder layers @ 2048
/// - output_proj: 2048 → 2560
/// - final LayerNorm
pub struct AudioAdaptorQwen4B {
    pub input_proj: nn::Linear,
    pub layers: Vec<TransformerEncoderLayer>,
    pub output_proj: nn::Linear,
    pub norm: nn::LayerNorm,
    pub config: AdaptorConfig,
}

impl AudioAdaptorQwen4B {
    /// Create a new adaptor with default config
    pub fn new() -> Result<Self> {
        Self::with_config(AdaptorConfig::default())
    }

    /// Create adaptor with custom config
    pub fn with_config(config: AdaptorConfig) -> Result<Self> {
        let input_proj = nn::Linear::new(config.input_dim, config.hidden_dim)?;

        let mut layers = Vec::with_capacity(config.num_layers as usize);
        for _ in 0..config.num_layers {
            layers.push(TransformerEncoderLayer::new(
                config.hidden_dim,
                config.num_heads,
                config.ffn_dim,
            ));
        }

        let output_proj = nn::Linear::new(config.hidden_dim, config.output_dim)?;
        let norm = nn::LayerNormBuilder::new(config.output_dim).eps(1e-5).affine(true).build()?;

        Ok(Self {
            input_proj,
            layers,
            output_proj,
            norm,
            config,
        })
    }

    /// Forward pass
    /// Input: [batch, seq_len, 512] (SenseVoice encoder output)
    /// Output: [batch, seq_len, 2560] (Qwen4B embedding space)
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        // Input projection: 512 → 2048
        let mut x = self.input_proj.forward(x)?;

        // 4 transformer encoder layers
        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }

        // Output projection: 2048 → 2560
        let x = self.output_proj.forward(&x)?;

        // Final layer norm
        let x = self.norm.forward(&x)?;

        Ok(x)
    }

    /// Load weights from safetensors file
    /// Expects weights already converted by scripts/convert_weights.py
    /// with key format: input_proj.weight, layers.0.q_proj.weight, etc.
    pub fn load_weights(&mut self, path: &str) -> Result<()> {
        let tensors = Array::load_safetensors(path)
            .map_err(|e| Error::ModelLoad(format!("Failed to load safetensors: {:?}", e)))?;

        for (key, value) in tensors {
            self.set_weight(&key, value);
        }

        Ok(())
    }

    /// Set a single weight by key
    fn set_weight(&mut self, key: &str, value: Array) {
        // Handle input/output projections
        if key.starts_with("input_proj.") {
            let subkey = &key[11..]; // Remove "input_proj."
            if subkey == "weight" {
                *self.input_proj.weight = value;
            } else if subkey == "bias" {
                *self.input_proj.bias = Some(value);
            }
        } else if key.starts_with("output_proj.") {
            let subkey = &key[12..]; // Remove "output_proj."
            if subkey == "weight" {
                *self.output_proj.weight = value;
            } else if subkey == "bias" {
                *self.output_proj.bias = Some(value);
            }
        } else if key.starts_with("norm.") {
            let subkey = &key[5..]; // Remove "norm."
            if subkey == "weight" {
                *self.norm.weight = Some(value);
            } else if subkey == "bias" {
                *self.norm.bias = Some(value);
            }
        } else if key.starts_with("layers.") {
            // Parse layer index: layers.0.q_proj.weight
            let rest = &key[7..]; // Remove "layers."
            if let Some(dot_pos) = rest.find('.') {
                if let Ok(layer_idx) = rest[..dot_pos].parse::<usize>() {
                    if layer_idx < self.layers.len() {
                        let layer_key = &rest[dot_pos + 1..];
                        self.layers[layer_idx].set_weight(layer_key, value);
                    }
                }
            }
        }
    }
}

impl Default for AudioAdaptorQwen4B {
    fn default() -> Self {
        Self::new().expect("Failed to create default AudioAdaptorQwen4B")
    }
}
