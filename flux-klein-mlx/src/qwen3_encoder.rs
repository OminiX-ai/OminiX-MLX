//! Qwen3 Text Encoder for FLUX.2-klein
//!
//! FLUX.2-klein uses Qwen3-4B as its text encoder, extracting hidden states
//! from layers 8, 17, and 26 (1-indexed) to create 7680-dim embeddings.
//!
//! Based on mlx-rs-lm's Qwen3 implementation.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    fast::{self, ScaledDotProductAttentionMask},
    module::Module,
    nn::{self, Linear, RmsNorm},
    ops,
    Array,
    Dtype,
};
use mlx_macros::ModuleParameters;
use serde::Deserialize;

// ============================================================================
// Configuration
// ============================================================================

/// Qwen3 model configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,
    pub rope_theta: f32,
    pub head_dim: i32,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        // Qwen3-4B configuration (from FLUX.2-klein-4B text_encoder/config.json)
        Self {
            hidden_size: 2560,
            num_hidden_layers: 36,
            intermediate_size: 9728,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            rms_norm_eps: 1e-6,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rope_theta: 1000000.0,
            head_dim: 128,  // From config.json, NOT 80
        }
    }
}

// ============================================================================
// Attention
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Qwen3Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[param]
    pub q_proj: Linear,
    #[param]
    pub k_proj: Linear,
    #[param]
    pub v_proj: Linear,
    #[param]
    pub o_proj: Linear,
    #[param]
    pub q_norm: RmsNorm,
    #[param]
    pub k_norm: RmsNorm,
    #[param]
    pub rope: nn::Rope,
}

impl Qwen3Attention {
    pub fn new(config: &Qwen3Config) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = nn::LinearBuilder::new(hidden_size, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, hidden_size)
            .bias(false)
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

    /// Forward pass with optional attention mask
    ///
    /// If attention_mask is provided, it should have shape [batch, seq_len] with
    /// 1 for real tokens and 0 for padded tokens.
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
            // Create combined mask: causal mask + padding mask
            // Causal: mask positions where j > i
            // Padding: mask positions where attention_mask[j] == 0
            let seq_len = l;

            // Create causal mask [1, 1, seq, seq]
            let i_idx = Array::from_iter(0..seq_len, &[seq_len, 1]);
            let j_idx = Array::from_iter(0..seq_len, &[1, seq_len]);
            let causal_mask = j_idx.le(&i_idx)?;  // True where j <= i (can attend)

            // Expand attention mask from [batch, seq] to [batch, 1, 1, seq] for broadcasting
            let padding_mask = attn_mask.reshape(&[b, 1, 1, seq_len])?;
            let padding_mask = padding_mask.as_dtype(Dtype::Bool)?;

            // Combine: can attend only if causal AND not padding
            // causal_mask: [seq, seq] -> [1, 1, seq, seq]
            let causal_mask = causal_mask.reshape(&[1, 1, seq_len, seq_len])?;
            let combined_mask = ops::logical_and(&causal_mask, &padding_mask)?;

            // Convert to float: True->0, False->-inf
            // Match the dtype of queries (bfloat16) for the mask
            let query_dtype = queries.dtype();
            let combined_float = combined_mask.as_dtype(query_dtype)?;
            let neg_inf = array!(-1e9f32).as_dtype(query_dtype)?;
            let one = array!(1.0f32).as_dtype(query_dtype)?;
            let mask = ops::multiply(&ops::subtract(&one, &combined_float)?, &neg_inf)?;

            // Use additive mask (Array variant expects additive mask where 0 = attend, -inf = mask)
            fast::scaled_dot_product_attention(
                &queries,
                &keys,
                &values,
                self.scale,
                ScaledDotProductAttentionMask::Array(&mask),
            )?
        } else {
            // No attention mask provided - use causal only
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

/// Create a causal mask for attention (lower triangular)
/// Returns mask with shape [1, 1, seq_len, seq_len] where:
/// - 0 for positions that can be attended
/// - -inf for positions that should be masked
fn create_causal_mask(seq_len: i32) -> Result<Array, Exception> {
    // Create indices
    let i = Array::from_iter(0..seq_len, &[seq_len, 1]);
    let j = Array::from_iter(0..seq_len, &[1, seq_len]);

    // mask[i,j] = 0 if j <= i, else -inf
    // This is the lower triangular mask (j <= i means attend)
    let mask = j.le(&i)?;  // True where j <= i (can attend)
    let mask = mask.as_dtype(Dtype::Float32)?;

    // Convert: 1 (attend) -> 0, 0 (mask) -> -inf
    let neg_inf = Array::from_slice(&[f32::NEG_INFINITY], &[1]);
    let one = Array::from_slice(&[1.0f32], &[1]);
    let mask = ops::multiply(&ops::subtract(&one, &mask)?, &neg_inf)?;

    // Add batch and head dimensions: [seq, seq] -> [1, 1, seq, seq]
    mask.reshape(&[1, 1, seq_len, seq_len])
}

// ============================================================================
// MLP
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Qwen3Mlp {
    #[param]
    pub gate_proj: Linear,
    #[param]
    pub down_proj: Linear,
    #[param]
    pub up_proj: Linear,
}

impl Qwen3Mlp {
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(intermediate_size, hidden_size)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(hidden_size, intermediate_size)
            .bias(false)
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
// Transformer Block
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Qwen3Block {
    #[param]
    pub self_attn: Qwen3Attention,
    #[param]
    pub mlp: Qwen3Mlp,
    #[param]
    pub input_layernorm: RmsNorm,
    #[param]
    pub post_attention_layernorm: RmsNorm,
}

impl Qwen3Block {
    pub fn new(config: &Qwen3Config) -> Result<Self, Exception> {
        let self_attn = Qwen3Attention::new(config)?;
        let mlp = Qwen3Mlp::new(config.hidden_size, config.intermediate_size)?;

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
// Qwen3 Text Encoder
// ============================================================================

/// Qwen3 model adapted for text encoding in FLUX.2-klein
///
/// Extracts hidden states from specific layers (8, 17, 26 in 1-indexed)
/// and stacks them to create 7680-dim embeddings.
#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Qwen3TextEncoder {
    pub config: Qwen3Config,
    /// Layers from which to extract hidden states (0-indexed)
    pub extract_layers: Vec<usize>,

    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub layers: Vec<Qwen3Block>,
    #[param]
    pub norm: RmsNorm,
}

impl Qwen3TextEncoder {
    /// Create a new Qwen3 text encoder
    pub fn new(config: Qwen3Config) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers: Result<Vec<_>, Exception> = (0..config.num_hidden_layers)
            .map(|_| Qwen3Block::new(&config))
            .collect();
        let layers = layers?;

        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        // FLUX.2-klein extracts from layers 9, 18, 27 (1-indexed)
        // Convert to 0-indexed: 8, 17, 26
        // This matches flux.c's QWEN3_OUTPUT_LAYER_{1,2,3}
        let extract_layers = vec![8, 17, 26];

        Ok(Self {
            config,
            extract_layers,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Forward pass returning hidden states from specified layers
    ///
    /// attention_mask: Optional mask with shape [batch, seq_len], 1 for real tokens, 0 for padding
    /// Returns a vector of hidden states from layers specified in extract_layers
    pub fn forward_with_hidden_states(
        &mut self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Vec<Array>, Exception> {
        // Get embeddings
        let mut h = self.embed_tokens.forward(input_ids)?;

        // Collect hidden states from specified layers
        let mut hidden_states = Vec::new();
        let extract_set: std::collections::HashSet<_> = self.extract_layers.iter().collect();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Pass attention mask to each layer
            h = layer.forward(&h, attention_mask)?;

            if extract_set.contains(&i) {
                // Extract raw hidden states without normalization (matches flux.c)
                hidden_states.push(h.clone());
            }
        }

        Ok(hidden_states)
    }

    /// Encode text to FLUX.2-klein format
    ///
    /// attention_mask: Optional mask with shape [batch, seq_len], 1 for real tokens, 0 for padding
    /// Returns stacked hidden states with shape [batch, seq, hidden_size * 3]
    /// where hidden_size * 3 = 2560 * 3 = 7680
    pub fn encode(
        &mut self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let hidden_states = self.forward_with_hidden_states(input_ids, attention_mask)?;

        // Stack along the last dimension: [batch, seq, hidden*3]
        if hidden_states.len() != 3 {
            return Err(Exception::custom(format!(
                "Expected 3 hidden states, got {}",
                hidden_states.len()
            )));
        }

        // Concatenate along the hidden dimension
        ops::concatenate_axis(
            &[
                hidden_states[0].clone(),
                hidden_states[1].clone(),
                hidden_states[2].clone(),
            ],
            -1,
        )
    }

    /// Encode text to Z-Image format
    ///
    /// Z-Image uses layer 34 (second-to-last, 0-indexed) as the text embedding,
    /// returning 2560-dim features instead of FLUX's 7680-dim concatenated features.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `attention_mask` - Optional mask [batch, seq_len], 1 for real tokens, 0 for padding
    ///
    /// # Returns
    /// Hidden states from layer 34 with shape [batch, seq, 2560]
    pub fn encode_zimage(
        &mut self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        // Get embeddings
        let mut h = self.embed_tokens.forward(input_ids)?;

        // Run through layers 0-34 (35 layers, skipping the last one)
        // This matches MLX_z-image's `return hidden_states[-2]`
        // In Python, hidden_states = [embeddings, layer0_out, layer1_out, ..., layer35_out]
        // hidden_states[-2] = layer34_out (second to last)
        let n_layers_to_run = self.layers.len().saturating_sub(1);
        for layer in self.layers[..n_layers_to_run].iter_mut() {
            h = layer.forward(&h, attention_mask)?;
        }

        // Return output after layer 34 (0-indexed), which is hidden_states[-2] in Python
        Ok(h)
    }
}

// ============================================================================
// Weight Loading
// ============================================================================

/// Sanitize Qwen3 weight keys from HuggingFace format to our model format
pub fn sanitize_qwen3_weights(weights: HashMap<String, Array>) -> HashMap<String, Array> {
    let mut sanitized = HashMap::new();

    for (key, value) in weights {
        // Remove "model." prefix if present
        let new_key = if key.starts_with("model.") {
            key.strip_prefix("model.").unwrap().to_string()
        } else {
            key.clone()
        };

        // Skip lm_head (not needed for text encoding)
        if new_key.starts_with("lm_head") {
            continue;
        }

        // Map layer keys: model.layers.X.Y -> layers.X.Y
        let new_key = new_key
            .replace("self_attn.", "self_attn.")
            .replace("mlp.", "mlp.");

        sanitized.insert(new_key, value);
    }

    sanitized
}

/// Load Qwen3 text encoder from a directory
pub fn load_qwen3_encoder(model_dir: impl AsRef<Path>) -> crate::Result<Qwen3TextEncoder> {
    let model_dir = model_dir.as_ref();

    // Load config
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
        crate::FluxError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Failed to read config.json: {}", e),
        ))
    })?;

    let config: Qwen3Config = serde_json::from_str(&config_str).map_err(|e| {
        crate::FluxError::Model(format!("Failed to parse config.json: {}", e))
    })?;

    // Create model
    let encoder = Qwen3TextEncoder::new(config)?;

    // Note: Weight loading should be done separately using update_flattened
    // after sanitizing the weights

    Ok(encoder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_config() {
        let config = Qwen3Config::default();
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_hidden_layers, 36);
    }

    #[test]
    fn test_qwen3_encoder_creation() {
        // Use smaller config for testing
        let config = Qwen3Config {
            hidden_size: 64,
            num_hidden_layers: 4,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-6,
            vocab_size: 1000,
            max_position_embeddings: 512,
            rope_theta: 10000.0,
            head_dim: 16,
        };

        let encoder = Qwen3TextEncoder::new(config);
        assert!(encoder.is_ok());
    }
}
