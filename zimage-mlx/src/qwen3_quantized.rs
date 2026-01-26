//! Quantized Qwen3 Text Encoder for Z-Image
//!
//! 4-bit quantized version of Qwen3-4B text encoder.
//! Uses QuantizedLinear layers for reduced memory usage (~3GB vs ~12GB).

use std::collections::HashMap;

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    fast::{self, ScaledDotProductAttentionMask},
    module::Module,
    nn::{self, RmsNorm, QuantizedLinear, QuantizedLinearBuilder},
    ops,
    Array,
    Dtype,
};
use mlx_macros::ModuleParameters;

// Re-export config from flux-klein
pub use flux_klein_mlx::qwen3_encoder::Qwen3Config;

// The MLX Qwen3 model uses group_size=32 for 4-bit quantization
const QWEN3_GROUP_SIZE: i32 = 32;

// ============================================================================
// Quantized Attention
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct QuantizedQwen3Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[param]
    pub q_proj: QuantizedLinear,
    #[param]
    pub k_proj: QuantizedLinear,
    #[param]
    pub v_proj: QuantizedLinear,
    #[param]
    pub o_proj: QuantizedLinear,
    #[param]
    pub q_norm: RmsNorm,
    #[param]
    pub k_norm: RmsNorm,
    #[param]
    pub rope: nn::Rope,
}

impl QuantizedQwen3Attention {
    pub fn new(config: &Qwen3Config) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = QuantizedLinearBuilder::new(hidden_size, n_heads * head_dim)
            .group_size(QWEN3_GROUP_SIZE)
            .build()?;
        let k_proj = QuantizedLinearBuilder::new(hidden_size, n_kv_heads * head_dim)
            .group_size(QWEN3_GROUP_SIZE)
            .build()?;
        let v_proj = QuantizedLinearBuilder::new(hidden_size, n_kv_heads * head_dim)
            .group_size(QWEN3_GROUP_SIZE)
            .build()?;
        let o_proj = QuantizedLinearBuilder::new(n_heads * head_dim, hidden_size)
            .group_size(QWEN3_GROUP_SIZE)
            .build()?;

        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()?;

        let rope = nn::RopeBuilder::new(head_dim)
            .base(config.rope_theta)
            .build()?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rope,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape and transpose for multi-head attention
        let queries = self.q_norm.forward(
            &queries
                .reshape(&[b, l, self.n_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?,
        )?;
        let keys = self.k_norm.forward(
            &keys
                .reshape(&[b, l, self.n_kv_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?,
        )?;
        let values = values
            .reshape(&[b, l, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE
        let queries = self.rope.forward(nn::RopeInput::new(&queries))?;
        let keys = self.rope.forward(nn::RopeInput::new(&keys))?;

        // Scaled dot-product attention with combined causal + padding mask
        let output = if let Some(attn_mask) = attention_mask {
            let seq_len = l;

            // Create causal mask [1, 1, seq, seq]
            let i_idx = Array::from_iter(0..seq_len, &[seq_len, 1]);
            let j_idx = Array::from_iter(0..seq_len, &[1, seq_len]);
            let causal_mask = j_idx.le(&i_idx)?;

            // Expand attention mask from [batch, seq] to [batch, 1, 1, seq]
            let padding_mask = attn_mask.reshape(&[b, 1, 1, seq_len])?;
            let padding_mask = padding_mask.as_dtype(Dtype::Bool)?;

            // Combine masks
            let causal_mask = causal_mask.reshape(&[1, 1, seq_len, seq_len])?;
            let combined_mask = ops::logical_and(&causal_mask, &padding_mask)?;

            // Convert to additive mask
            let query_dtype = queries.dtype();
            let combined_float = combined_mask.as_dtype(query_dtype)?;
            let neg_inf = array!(-1e9f32).as_dtype(query_dtype)?;
            let one = array!(1.0f32).as_dtype(query_dtype)?;
            let mask = ops::multiply(&ops::subtract(&one, &combined_float)?, &neg_inf)?;

            fast::scaled_dot_product_attention(
                &queries,
                &keys,
                &values,
                self.scale,
                ScaledDotProductAttentionMask::Array(&mask),
            )?
        } else {
            fast::scaled_dot_product_attention(
                &queries,
                &keys,
                &values,
                self.scale,
                ScaledDotProductAttentionMask::Causal,
            )?
        };

        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, l, -1])?;

        self.o_proj.forward(&output)
    }
}

// ============================================================================
// Quantized MLP
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct QuantizedQwen3Mlp {
    #[param]
    pub gate_proj: QuantizedLinear,
    #[param]
    pub down_proj: QuantizedLinear,
    #[param]
    pub up_proj: QuantizedLinear,
}

impl QuantizedQwen3Mlp {
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        let gate_proj = QuantizedLinearBuilder::new(hidden_size, intermediate_size)
            .group_size(QWEN3_GROUP_SIZE)
            .build()?;
        let down_proj = QuantizedLinearBuilder::new(intermediate_size, hidden_size)
            .group_size(QWEN3_GROUP_SIZE)
            .build()?;
        let up_proj = QuantizedLinearBuilder::new(hidden_size, intermediate_size)
            .group_size(QWEN3_GROUP_SIZE)
            .build()?;

        Ok(Self {
            gate_proj,
            down_proj,
            up_proj,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = nn::silu(self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let hidden = ops::multiply(&gate, &up)?;
        self.down_proj.forward(&hidden)
    }
}

// ============================================================================
// Quantized Transformer Block
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct QuantizedQwen3Block {
    #[param]
    pub self_attn: QuantizedQwen3Attention,
    #[param]
    pub mlp: QuantizedQwen3Mlp,
    #[param]
    pub input_layernorm: RmsNorm,
    #[param]
    pub post_attention_layernorm: RmsNorm,
}

impl QuantizedQwen3Block {
    pub fn new(config: &Qwen3Config) -> Result<Self, Exception> {
        let self_attn = QuantizedQwen3Attention::new(config)?;
        let mlp = QuantizedQwen3Mlp::new(config.hidden_size, config.intermediate_size)?;

        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        // Self attention with residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, attention_mask)?;
        let h = ops::add(x, &attn_out)?;

        // MLP with residual
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        ops::add(&h, &mlp_out)
    }
}

// ============================================================================
// Quantized Qwen3 Text Encoder
// ============================================================================

/// Quantized Qwen3 text encoder for Z-Image
///
/// Uses 4-bit quantized linear layers for reduced memory usage.
#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct QuantizedQwen3TextEncoder {
    pub config: Qwen3Config,

    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub layers: Vec<QuantizedQwen3Block>,
    #[param]
    pub norm: RmsNorm,
}

impl QuantizedQwen3TextEncoder {
    pub fn new(config: Qwen3Config) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers: Result<Vec<_>, Exception> = (0..config.num_hidden_layers)
            .map(|_| QuantizedQwen3Block::new(&config))
            .collect();
        let layers = layers?;

        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Encode text to Z-Image format
    ///
    /// Z-Image uses layer 34 (second-to-last, 0-indexed) as the text embedding,
    /// returning 2560-dim features.
    pub fn encode_zimage(
        &mut self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        // Get embeddings
        let mut h = self.embed_tokens.forward(input_ids)?;

        // Run through layers 0-34 (35 layers, skipping the last one)
        let n_layers_to_run = self.layers.len().saturating_sub(1);
        for layer in self.layers[..n_layers_to_run].iter_mut() {
            h = layer.forward(&h, attention_mask)?;
        }

        Ok(h)
    }

    /// Encode text to FLUX.2-klein format (for compatibility)
    ///
    /// Extracts from layers 8, 17, 26 and concatenates to 7680-dim.
    pub fn encode_flux(
        &mut self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let extract_layers: std::collections::HashSet<usize> = [8, 17, 26].into_iter().collect();

        let mut h = self.embed_tokens.forward(input_ids)?;
        let mut hidden_states = Vec::new();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            h = layer.forward(&h, attention_mask)?;
            if extract_layers.contains(&i) {
                hidden_states.push(h.clone());
            }
        }

        if hidden_states.len() != 3 {
            return Err(Exception::custom(format!(
                "Expected 3 hidden states, got {}",
                hidden_states.len()
            )));
        }

        ops::concatenate_axis(
            &[
                hidden_states[0].clone(),
                hidden_states[1].clone(),
                hidden_states[2].clone(),
            ],
            -1,
        )
    }
}

// ============================================================================
// Weight Loading
// ============================================================================

/// Dequantize 4-bit packed weights
fn dequantize_4bit(weight: &Array, scales: &Array, biases: &Array, group_size: i32) -> Result<Array, Exception> {
    // weight: [out_features, packed_in] as uint32 (8 values per uint32)
    // scales: [out_features, num_groups] as bf16/f32
    // biases: [out_features, num_groups] as bf16/f32
    // Returns: [out_features, in_features] as f32

    // Use MLX's dequantize op with affine mode
    mlx_rs::ops::dequantize(weight, scales, biases, group_size, 4, "affine")
}

/// Sanitize quantized Qwen3 weight keys from MLX format to our model format
///
/// MLX format: model.layers.X.self_attn.q_proj.weight/scales/biases
/// Our format: layers.X.self_attn.q_proj.inner.weight/scales/biases
///
/// QuantizedLinear expects:
/// - .inner.weight (the quantized uint32 weights)
/// - .scales (quantization scales)
/// - .biases (quantization biases)
///
/// Also dequantizes the embedding layer since nn::Embedding doesn't support quantization.
pub fn sanitize_quantized_qwen3_weights(weights: HashMap<String, Array>) -> HashMap<String, Array> {
    let mut sanitized = HashMap::new();

    // First pass: collect all weights, stripping "model." prefix
    let mut temp_weights: HashMap<String, Array> = HashMap::new();
    for (key, value) in &weights {
        let new_key = if key.starts_with("model.") {
            key.strip_prefix("model.").unwrap().to_string()
        } else {
            key.clone()
        };

        if new_key.starts_with("lm_head") {
            continue;
        }

        temp_weights.insert(new_key, value.clone());
    }

    // Second pass: dequantize embedding if needed (nn::Embedding doesn't support quantization)
    if temp_weights.contains_key("embed_tokens.weight")
        && temp_weights.contains_key("embed_tokens.scales")
        && temp_weights.contains_key("embed_tokens.biases")
    {
        let weight = temp_weights.get("embed_tokens.weight").unwrap();
        let scales = temp_weights.get("embed_tokens.scales").unwrap();
        let biases = temp_weights.get("embed_tokens.biases").unwrap();

        // Calculate group_size from weight and scales shapes
        // weight is packed: [vocab, hidden/8], scales: [vocab, hidden/group_size]
        let packed_dim = weight.dim(1);
        let num_groups = scales.dim(1);
        let hidden_size = packed_dim * 8; // 8 values per uint32
        let group_size = hidden_size / num_groups;

        if let Ok(dequantized) = dequantize_4bit(weight, scales, biases, group_size) {
            sanitized.insert("embed_tokens.weight".to_string(), dequantized);
            temp_weights.remove("embed_tokens.weight");
            temp_weights.remove("embed_tokens.scales");
            temp_weights.remove("embed_tokens.biases");
        }
    }

    // Third pass: rename QuantizedLinear weight keys
    // MLX format: .weight -> Our format: .inner.weight
    // The QuantizedLinear struct has inner: Linear which holds the quantized weight
    for (key, value) in temp_weights {
        // For quantized layers (q_proj, k_proj, v_proj, o_proj, gate_proj, down_proj, up_proj),
        // rename .weight to .inner.weight
        let new_key = if key.ends_with("_proj.weight") {
            key.replace("_proj.weight", "_proj.inner.weight")
        } else {
            key
        };
        sanitized.insert(new_key, value);
    }

    sanitized
}

/// Load quantized Qwen3 text encoder from safetensors file
pub fn load_quantized_qwen3_encoder(
    weights: HashMap<String, Array>,
    config: Qwen3Config,
) -> Result<QuantizedQwen3TextEncoder, Exception> {
    use mlx_rs::module::ModuleParameters;

    // Create model
    let mut encoder = QuantizedQwen3TextEncoder::new(config)?;

    // Sanitize weights
    let weights = sanitize_quantized_qwen3_weights(weights);

    // Convert to Rc<str> keys for update_flattened
    let weights_rc: HashMap<std::rc::Rc<str>, Array> = weights
        .into_iter()
        .map(|(k, v)| (std::rc::Rc::from(k.as_str()), v))
        .collect();

    // Load weights
    encoder.update_flattened(weights_rc);

    Ok(encoder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_qwen3_encoder_creation() {
        let config = Qwen3Config {
            hidden_size: 64,
            num_hidden_layers: 2,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-6,
            vocab_size: 1000,
            max_position_embeddings: 512,
            rope_theta: 10000.0,
            head_dim: 16,
        };

        let encoder = QuantizedQwen3TextEncoder::new(config);
        assert!(encoder.is_ok());
    }
}
