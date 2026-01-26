//! Qwen-Image Text Encoder (Qwen2.5-VL variant)
//!
//! This is the text encoder used in Qwen-Image for encoding prompts.
//! Architecture: 28 layers, hidden_size=3584, GQA with 28 q_heads and 4 kv_heads

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    error::Exception,
    module::Param,
    ops::indexing::{IndexOp, take_axis},
    Array,
};

/// Text encoder configuration
#[derive(Debug, Clone)]
pub struct TextEncoderConfig {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,  // 28 query heads
    pub num_key_value_heads: i32,  // 4 kv heads (GQA)
    pub intermediate_size: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub head_dim: i32,
}

impl Default for TextEncoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 3584,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 4,
            intermediate_size: 18944,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            head_dim: 128,
        }
    }
}

/// RMS Normalization
#[derive(Debug)]
pub struct RmsNorm {
    pub weight: Param<Array>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::new(Array::ones::<f32>(&[dim])?),
            eps: 1e-6,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        // RMS norm: x * rsqrt(mean(x^2) + eps) * weight
        let x_sq = mlx_rs::ops::square(x)?;
        let mean = mlx_rs::ops::mean_axes(&x_sq, &[-1], true)?;
        let eps = Array::from_f32(self.eps);
        let norm = mlx_rs::ops::rsqrt(&mlx_rs::ops::add(&mean, &eps)?)?;
        let normalized = mlx_rs::ops::multiply(x, &norm)?;
        mlx_rs::ops::multiply(&normalized, &*self.weight)
    }
}

/// Linear layer with optional bias
#[derive(Debug)]
pub struct Linear {
    pub weight: Param<Array>,
    pub bias: Option<Param<Array>>,
}

impl Linear {
    pub fn new(in_features: i32, out_features: i32, with_bias: bool) -> Result<Self, Exception> {
        let weight = Array::zeros::<f32>(&[out_features, in_features])?;
        let bias = if with_bias {
            Some(Param::new(Array::zeros::<f32>(&[out_features])?))
        } else {
            None
        };
        Ok(Self {
            weight: Param::new(weight),
            bias,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        // x @ weight.T + bias
        let weight_t = (*self.weight).transpose()?;
        let out = mlx_rs::ops::matmul(x, &weight_t)?;
        if let Some(bias) = &self.bias {
            mlx_rs::ops::add(&out, &**bias)
        } else {
            Ok(out)
        }
    }
}

/// Embedding layer
#[derive(Debug)]
pub struct Embedding {
    pub weight: Param<Array>,
}

impl Embedding {
    pub fn new(vocab_size: i32, embed_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::new(Array::zeros::<f32>(&[vocab_size, embed_dim])?),
        })
    }

    pub fn forward(&self, input_ids: &Array) -> Result<Array, Exception> {
        take_axis(&*self.weight, input_ids, 0)
    }
}

/// Rotary Position Embedding
#[derive(Debug)]
pub struct RotaryEmbedding {
    pub inv_freq: Array,
    pub head_dim: i32,
}

impl RotaryEmbedding {
    pub fn new(head_dim: i32, theta: f32) -> Result<Self, Exception> {
        // inv_freq = 1.0 / (theta ^ (i / head_dim)) for i in 0, 2, 4, ...
        let half_dim = head_dim / 2;
        let mut inv_freqs = Vec::with_capacity(half_dim as usize);
        for i in 0..half_dim {
            let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
            inv_freqs.push(freq);
        }
        let inv_freq = Array::from_slice(&inv_freqs, &[half_dim]);

        Ok(Self { inv_freq, head_dim })
    }

    pub fn apply(&self, x: &Array, offset: i32) -> Result<Array, Exception> {
        // x: [batch, heads, seq_len, head_dim]
        let shape = x.shape();
        let seq_len = shape[2];

        // Create position indices
        let positions: Vec<f32> = (offset..offset + seq_len)
            .map(|i| i as f32)
            .collect();
        let pos = Array::from_slice(&positions, &[seq_len]);

        // freqs = positions @ inv_freq.T -> [seq_len, head_dim/2]
        let freqs = mlx_rs::ops::outer(&pos, &self.inv_freq)?;

        // Compute cos and sin
        let cos = mlx_rs::ops::cos(&freqs)?;  // [seq_len, head_dim/2]
        let sin = mlx_rs::ops::sin(&freqs)?;  // [seq_len, head_dim/2]

        // Split x into two halves
        let half = self.head_dim / 2;
        let x1 = x.index((.., .., .., ..half));
        let x2 = x.index((.., .., .., half..));

        // Expand cos/sin for broadcasting: [1, 1, seq_len, head_dim/2]
        let cos = cos.reshape(&[1, 1, seq_len, half])?;
        let sin = sin.reshape(&[1, 1, seq_len, half])?;

        // Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        let rotated_x1 = mlx_rs::ops::subtract(
            &mlx_rs::ops::multiply(&x1, &cos)?,
            &mlx_rs::ops::multiply(&x2, &sin)?,
        )?;
        let rotated_x2 = mlx_rs::ops::add(
            &mlx_rs::ops::multiply(&x1, &sin)?,
            &mlx_rs::ops::multiply(&x2, &cos)?,
        )?;

        // Concatenate back
        mlx_rs::ops::concatenate_axis(&[&rotated_x1, &rotated_x2], -1)
    }
}

/// Attention with Grouped Query Attention (GQA)
#[derive(Debug)]
pub struct TextAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub rope: RotaryEmbedding,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
}

impl TextAttention {
    pub fn new(config: &TextEncoderConfig) -> Result<Self, Exception> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        Ok(Self {
            q_proj: Linear::new(hidden_size, num_heads * head_dim, true)?,
            k_proj: Linear::new(hidden_size, num_kv_heads * head_dim, true)?,
            v_proj: Linear::new(hidden_size, num_kv_heads * head_dim, true)?,
            o_proj: Linear::new(num_heads * head_dim, hidden_size, false)?,
            rope: RotaryEmbedding::new(head_dim, config.rope_theta)?,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
        })
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let q = q.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE
        let q = self.rope.apply(&q, 0)?;
        let k = self.rope.apply(&k, 0)?;

        // Repeat KV heads for GQA
        let num_groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, num_groups)?;
        let v = repeat_kv(&v, num_groups)?;

        // Scaled dot-product attention
        let attn = mlx_rs::ops::matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?;
        let attn = mlx_rs::ops::multiply(&attn, &Array::from_f32(self.scale))?;

        // Apply mask if provided
        let attn = if let Some(mask) = mask {
            mlx_rs::ops::add(&attn, mask)?
        } else {
            attn
        };

        let attn = mlx_rs::ops::softmax_axis(&attn, -1, None)?;

        // Apply attention to values
        let out = mlx_rs::ops::matmul(&attn, &v)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let out = out.transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, -1])?;

        self.o_proj.forward(&out)
    }
}

/// Repeat KV heads for GQA
fn repeat_kv(x: &Array, num_groups: i32) -> Result<Array, Exception> {
    if num_groups == 1 {
        return Ok(x.clone());
    }

    let shape = x.shape();
    let batch = shape[0];
    let num_kv_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    // [batch, kv_heads, seq, head_dim] -> [batch, kv_heads, num_groups, seq, head_dim]
    let x = x.reshape(&[batch, num_kv_heads, 1, seq_len, head_dim])?;
    let x = mlx_rs::ops::broadcast_to(&x, &[batch, num_kv_heads, num_groups, seq_len, head_dim])?;
    x.reshape(&[batch, num_kv_heads * num_groups, seq_len, head_dim])
}

/// MLP with SwiGLU activation
#[derive(Debug)]
pub struct TextMlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl TextMlp {
    pub fn new(config: &TextEncoderConfig) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: Linear::new(config.hidden_size, config.intermediate_size, false)?,
            up_proj: Linear::new(config.hidden_size, config.intermediate_size, false)?,
            down_proj: Linear::new(config.intermediate_size, config.hidden_size, false)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let gate = mlx_rs::nn::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = mlx_rs::ops::multiply(&gate, &up)?;
        self.down_proj.forward(&hidden)
    }
}

/// Transformer layer
#[derive(Debug)]
pub struct TextEncoderLayer {
    pub self_attn: TextAttention,
    pub mlp: TextMlp,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

impl TextEncoderLayer {
    pub fn new(config: &TextEncoderConfig) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: TextAttention::new(config)?,
            mlp: TextMlp::new(config)?,
            input_layernorm: RmsNorm::new(config.hidden_size)?,
            post_attention_layernorm: RmsNorm::new(config.hidden_size)?,
        })
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        // Pre-norm attention
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, mask)?;
        let x = mlx_rs::ops::add(&x, residual)?;

        // Pre-norm MLP
        let residual = &x;
        let hidden = self.post_attention_layernorm.forward(&x)?;
        let hidden = self.mlp.forward(&hidden)?;
        mlx_rs::ops::add(&hidden, residual)
    }
}

/// Qwen Text Encoder
#[derive(Debug)]
pub struct QwenTextEncoder {
    pub config: TextEncoderConfig,
    pub embed_tokens: Embedding,
    pub layers: Vec<TextEncoderLayer>,
    pub norm: RmsNorm,
}

impl QwenTextEncoder {
    pub fn new(config: TextEncoderConfig) -> Result<Self, Exception> {
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for _ in 0..config.num_hidden_layers {
            layers.push(TextEncoderLayer::new(&config)?);
        }

        let norm = RmsNorm::new(config.hidden_size)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Forward pass with attention mask - returns hidden states [batch, seq_len, hidden_size]
    /// attention_mask: [batch, seq_len] with 1 for valid tokens, 0 for padding
    pub fn forward_with_mask(&mut self, input_ids: &Array, attention_mask: &Array) -> Result<Array, Exception> {
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let seq_len = input_ids.dim(1);

        // Create causal mask: lower triangular (0 for visible, -inf for future)
        // Shape: [1, 1, seq_len, seq_len]
        let idx = mlx_rs::ops::arange::<i32, i32>(0, seq_len, 1)?;
        let j = idx.reshape(&[1, seq_len])?;  // [1, seq_len]
        let i = idx.reshape(&[seq_len, 1])?;  // [seq_len, 1]
        // tri_bool[i, j] = j > i (future positions)
        let tri_bool = j.gt(&i)?;  // [seq_len, seq_len]

        let zeros_2d = Array::zeros::<f32>(&[seq_len, seq_len])?;
        let neginf = Array::from_f32(f32::NEG_INFINITY);
        let neginf_2d = mlx_rs::ops::full::<f32>(&[seq_len, seq_len], &neginf)?;
        let causal_mask = mlx_rs::ops::r#where(&tri_bool, &neginf_2d, &zeros_2d)?;  // [seq_len, seq_len]
        let causal_mask = causal_mask.reshape(&[1, 1, seq_len, seq_len])?;  // [1, 1, seq_len, seq_len]

        // Create padding mask: -inf for padding (0), 0 for valid (1)
        // attention_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
        let zeros_mask = Array::zeros::<f32>(&attention_mask.shape())?;
        let neginf_mask = mlx_rs::ops::full::<f32>(&attention_mask.shape(), &neginf)?;
        let one = Array::from_int(1);
        let mask_bool = attention_mask.eq(&one)?;
        let padding_mask = mlx_rs::ops::r#where(&mask_bool, &zeros_mask, &neginf_mask)?;
        let padding_mask = padding_mask.reshape(&[attention_mask.dim(0), 1, 1, seq_len])?;

        // Combine masks: both are additive
        let combined_mask = mlx_rs::ops::add(&causal_mask, &padding_mask)?;

        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, Some(&combined_mask))?;
        }

        self.norm.forward(&hidden)
    }

    /// Forward pass (backward compatible) - returns hidden states [batch, seq_len, hidden_size]
    pub fn forward(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        // Create all-ones attention mask (all valid)
        let seq_len = input_ids.dim(1);
        let attention_mask = Array::ones::<i32>(&[1, seq_len])?;
        self.forward_with_mask(input_ids, &attention_mask)
    }
}

/// Load text encoder weights from safetensors files
pub fn load_text_encoder_weights(
    encoder: &mut QwenTextEncoder,
    weights: HashMap<String, Array>,
) -> Result<(), Exception> {
    // Load embedding
    if let Some(w) = weights.get("encoder.embed_tokens.weight") {
        *encoder.embed_tokens.weight = w.clone();
    }

    // Load layers
    for (i, layer) in encoder.layers.iter_mut().enumerate() {
        let prefix = format!("encoder.layers.{}", i);

        // Input layernorm
        if let Some(w) = weights.get(&format!("{}.input_layernorm.weight", prefix)) {
            *layer.input_layernorm.weight = w.clone();
        }

        // Self attention
        if let Some(w) = weights.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
            *layer.self_attn.q_proj.weight = w.clone();
        }
        if let Some(b) = weights.get(&format!("{}.self_attn.q_proj.bias", prefix)) {
            layer.self_attn.q_proj.bias = Some(Param::new(b.clone()));
        }
        if let Some(w) = weights.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
            *layer.self_attn.k_proj.weight = w.clone();
        }
        if let Some(b) = weights.get(&format!("{}.self_attn.k_proj.bias", prefix)) {
            layer.self_attn.k_proj.bias = Some(Param::new(b.clone()));
        }
        if let Some(w) = weights.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
            *layer.self_attn.v_proj.weight = w.clone();
        }
        if let Some(b) = weights.get(&format!("{}.self_attn.v_proj.bias", prefix)) {
            layer.self_attn.v_proj.bias = Some(Param::new(b.clone()));
        }
        if let Some(w) = weights.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
            *layer.self_attn.o_proj.weight = w.clone();
        }

        // Load rotary embedding inv_freq if present
        if let Some(inv_freq) = weights.get(&format!("{}.self_attn.rotary_emb.inv_freq", prefix)) {
            layer.self_attn.rope.inv_freq = inv_freq.clone();
        }

        // Post attention layernorm
        if let Some(w) = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
            *layer.post_attention_layernorm.weight = w.clone();
        }

        // MLP
        if let Some(w) = weights.get(&format!("{}.mlp.gate_proj.weight", prefix)) {
            *layer.mlp.gate_proj.weight = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.mlp.up_proj.weight", prefix)) {
            *layer.mlp.up_proj.weight = w.clone();
        }
        if let Some(w) = weights.get(&format!("{}.mlp.down_proj.weight", prefix)) {
            *layer.mlp.down_proj.weight = w.clone();
        }
    }

    // Load final norm
    if let Some(w) = weights.get("encoder.norm.weight") {
        *encoder.norm.weight = w.clone();
    }

    Ok(())
}

/// Load text encoder from model directory
pub fn load_text_encoder(model_dir: impl AsRef<Path>) -> Result<QwenTextEncoder, Box<dyn std::error::Error>> {
    let model_dir = model_dir.as_ref();
    let text_encoder_dir = model_dir.join("text_encoder");

    // Load index
    let index_path = text_encoder_dir.join("model.safetensors.index.json");
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: serde_json::Value = serde_json::from_str(&index_content)?;

    // Get all weight files
    let weight_map = index["weight_map"].as_object()
        .ok_or("Invalid index format")?;

    let mut files: std::collections::HashSet<String> = std::collections::HashSet::new();
    for file in weight_map.values() {
        if let Some(f) = file.as_str() {
            files.insert(f.to_string());
        }
    }

    // Load all weights
    let mut all_weights = HashMap::new();
    for file in files {
        let path = text_encoder_dir.join(&file);
        println!("  Loading text encoder: {} ...", file);
        let data = std::fs::read(&path)?;
        let tensors = safetensors::SafeTensors::deserialize(&data)?;

        for (name, tensor) in tensors.tensors() {
            let array = Array::try_from(tensor)?;
            all_weights.insert(name.to_string(), array);
        }
    }

    println!("  Loaded {} text encoder tensors", all_weights.len());

    // Create encoder and load weights
    let config = TextEncoderConfig::default();
    let mut encoder = QwenTextEncoder::new(config)?;
    load_text_encoder_weights(&mut encoder, all_weights)?;

    Ok(encoder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_encoder_creation() {
        let config = TextEncoderConfig::default();
        let encoder = QwenTextEncoder::new(config).unwrap();

        assert_eq!(encoder.layers.len(), 28);
        assert_eq!(encoder.config.hidden_size, 3584);
    }

    #[test]
    fn test_text_encoder_forward() {
        let config = TextEncoderConfig {
            num_hidden_layers: 2,  // Use fewer layers for testing
            ..TextEncoderConfig::default()
        };
        let mut encoder = QwenTextEncoder::new(config).unwrap();

        // Create dummy input_ids
        let input_ids = Array::from_slice(&[1i32, 2, 3, 4, 5], &[1, 5]);

        let output = encoder.forward(&input_ids).unwrap();
        assert_eq!(output.shape(), &[1, 5, 3584]);
    }

    #[test]
    fn test_rms_norm() {
        let norm = RmsNorm::new(64).unwrap();
        let x = Array::from_slice(&vec![1.0f32; 64], &[1, 64]);
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 64]);
    }
}
