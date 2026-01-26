//! Qwen2.5-7B LLM backbone for Step-Audio 2
//!
//! Configuration:
//! - hidden_size: 3584
//! - intermediate_size: 18944
//! - num_hidden_layers: 28
//! - num_attention_heads: 28
//! - num_key_value_heads: 4 (GQA 7:1)
//! - vocab_size: 158720 (text + audio tokens)
//!
//! Key differences from standard Qwen2:
//! - Extended vocabulary for audio tokens (151696-158256)
//! - tie_word_embeddings: false (separate lm_head)
//!
//! Adapted from qwen3-mlx/src/qwen2.rs

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters as ModuleParametersTrait},
    nn,
    Array,
};
use serde::Deserialize;

use mlx_rs_core::{
    AttentionMask, KVCache, KeyValueCache,
    create_attention_mask, initialize_rope,
};

use crate::config::LLMConfig;
use crate::error::{Error, Result};

// ============================================================================
// Attention
// ============================================================================

/// Qwen2 Attention with GQA support
///
/// Uses bias in Q/K/V projections, no QK-norm (unlike Qwen3)
#[derive(Debug, Clone, ModuleParameters)]
pub struct Attention {
    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub o_proj: nn::Linear,
    #[param]
    pub rope: nn::Rope,

    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
}

impl Attention {
    pub fn new(config: &LLMConfig) -> Result<Self> {
        let dim = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = dim / n_heads;
        let scale = (head_dim as f32).sqrt().recip();

        // Qwen2 uses bias=True for q/k/v projections
        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(true)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(true)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(true)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(false)
            .build()?;

        // Initialize RoPE
        let rope = initialize_rope(
            head_dim,
            config.rope_theta,
            false, // traditional
            &None, // rope_scaling
            config.max_position_embeddings,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
        })
    }

    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> std::result::Result<Array, Exception> {
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        // Project Q, K, V
        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape to [B, L, n_heads, head_dim] then transpose to [B, n_heads, L, head_dim]
        let mut queries = queries
            .reshape(&[B, L, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = keys
            .reshape(&[B, L, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut values = values
            .reshape(&[B, L, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE and update cache
        if let Some(cache) = cache {
            let offset = cache.offset();
            let q_input = nn::RopeInputBuilder::new(&queries)
                .offset(offset)
                .build()?;
            queries = self.rope.forward(q_input)?;
            let k_input = nn::RopeInputBuilder::new(&keys)
                .offset(offset)
                .build()?;
            keys = self.rope.forward(k_input)?;

            (keys, values) = cache.update_and_fetch(keys, values)?;
        } else {
            queries = self.rope.forward(nn::RopeInput::new(&queries))?;
            keys = self.rope.forward(nn::RopeInput::new(&keys))?;
        }

        // Scaled dot-product attention
        let sdpa_mask = if mask.is_some() {
            mask.map(mlx_rs::fast::ScaledDotProductAttentionMask::Array)
        } else if L > 1 {
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };

        let output = mlx_rs::fast::scaled_dot_product_attention(
            queries, keys, values, self.scale, sdpa_mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }
}

// ============================================================================
// MLP (SwiGLU)
// ============================================================================

/// Qwen2 MLP with SwiGLU activation
#[derive(Debug, Clone, ModuleParameters)]
pub struct MLP {
    #[param]
    pub gate_proj: nn::Linear,
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
}

impl MLP {
    pub fn new(config: &LLMConfig) -> Result<Self> {
        let dim = config.hidden_size;
        let hidden_dim = config.intermediate_size;

        Ok(Self {
            gate_proj: nn::LinearBuilder::new(dim, hidden_dim).bias(false).build()?,
            up_proj: nn::LinearBuilder::new(dim, hidden_dim).bias(false).build()?,
            down_proj: nn::LinearBuilder::new(hidden_dim, dim).bias(false).build()?,
        })
    }
}

impl Module<&Array> for MLP {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = nn::silu(&gate)?.multiply(&up)?;
        self.down_proj.forward(&activated)
    }
}

// ============================================================================
// Transformer Block
// ============================================================================

/// Qwen2 Transformer Block
#[derive(Debug, Clone, ModuleParameters)]
pub struct TransformerBlock {
    #[param]
    pub self_attn: Attention,
    #[param]
    pub mlp: MLP,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    pub fn new(config: &LLMConfig) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(config)?,
            mlp: MLP::new(config)?,
            input_layernorm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> std::result::Result<Array, Exception> {
        // Pre-norm attention
        let h = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&h, mask, cache)?;
        let h = x.add(&attn_out)?;

        // Pre-norm MLP
        let mlp_in = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&mlp_in)?;
        h.add(&mlp_out)
    }
}

// ============================================================================
// Full LLM Model
// ============================================================================

/// Qwen2.5-7B Language Model for Step-Audio 2
#[derive(Debug, Clone, ModuleParameters)]
pub struct StepAudio2LLM {
    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub layers: Vec<TransformerBlock>,
    #[param]
    pub norm: nn::RmsNorm,
    #[param]
    pub lm_head: nn::Linear,

    pub config: LLMConfig,
}

impl StepAudio2LLM {
    pub fn new(config: LLMConfig) -> Result<Self> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;

        let layers: Result<Vec<_>> = (0..config.num_hidden_layers)
            .map(|_| TransformerBlock::new(&config))
            .collect();

        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        // Step-Audio 2 uses separate lm_head (tie_word_embeddings: false)
        let lm_head = nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
            .bias(false)
            .build()?;

        Ok(Self {
            embed_tokens,
            layers: layers?,
            norm,
            lm_head,
            config,
        })
    }

    /// Get token embeddings without running the full model
    pub fn get_token_embeddings(&mut self, tokens: &Array) -> std::result::Result<Array, Exception> {
        self.embed_tokens.forward(tokens)
    }

    /// Forward pass with token IDs
    pub fn forward(
        &mut self,
        tokens: &Array,
        cache: &mut Vec<Option<KVCache>>,
    ) -> std::result::Result<Array, Exception> {
        let h = self.embed_tokens.forward(tokens)?;
        self.forward_embeddings(&h, cache)
    }

    /// Forward pass with pre-computed embeddings (for audio injection)
    pub fn forward_embeddings(
        &mut self,
        h: &Array,
        cache: &mut Vec<Option<KVCache>>,
    ) -> std::result::Result<Array, Exception> {
        // Create attention mask if needed
        let mask = match create_attention_mask(h, cache, Some(true))? {
            Some(AttentionMask::Array(a)) => Some(a),
            _ => None,
        };

        // Initialize cache if empty
        if cache.is_empty() {
            *cache = (0..self.layers.len())
                .map(|_| Some(KVCache::default()))
                .collect();
        }

        // Run through transformer layers
        let mut h = h.clone();
        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(&h, mask.as_ref(), c.as_mut())?;
        }

        // Final norm and LM head
        let h = self.norm.forward(&h)?;
        self.lm_head.forward(&h)
    }
}

// ============================================================================
// Weight Loading
// ============================================================================

/// Load LLM weights from safetensors files
pub fn load_llm_weights(
    model: &mut StepAudio2LLM,
    model_dir: &Path,
) -> Result<()> {
    // Check for weight index file
    let weights_index = model_dir.join("model.safetensors.index.json");

    if weights_index.exists() {
        // Multi-file weights
        let json = std::fs::read_to_string(&weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();

        for weight_file in weight_files {
            let weights_path = model_dir.join(weight_file);
            load_weights_from_file(model, &weights_path)?;
        }
    } else {
        // Single file weights
        let weights_path = model_dir.join("model.safetensors");
        if weights_path.exists() {
            load_weights_from_file(model, &weights_path)?;
        } else {
            return Err(Error::WeightLoad(format!(
                "No weights found in {}",
                model_dir.display()
            )));
        }
    }

    Ok(())
}

fn load_weights_from_file(model: &mut StepAudio2LLM, path: &Path) -> Result<()> {
    let loaded = Array::load_safetensors(path)?;
    let mut params = model.parameters_mut().flatten();

    for (st_key, value) in loaded {
        let rust_key = map_weight_key(&st_key);
        if let Some(param) = params.get_mut(&*rust_key) {
            **param = value;
        }
    }

    Ok(())
}

fn map_weight_key(key: &str) -> std::rc::Rc<str> {
    // Map HuggingFace weight names to our parameter names
    let key = key
        .replace("model.layers.", "layers.")
        .replace(".self_attn.", ".self_attn.")
        .replace(".mlp.gate_proj.", ".mlp.gate_proj.")
        .replace(".mlp.up_proj.", ".mlp.up_proj.")
        .replace(".mlp.down_proj.", ".mlp.down_proj.")
        .replace("model.embed_tokens.", "embed_tokens.")
        .replace("model.norm.", "norm.");

    std::rc::Rc::from(key)
}

#[derive(Debug, Clone, Deserialize)]
struct WeightMap {
    weight_map: HashMap<String, String>,
}

// ============================================================================
// Generation
// ============================================================================

/// Sample from logits
pub fn sample(logits: &Array, temperature: f32) -> std::result::Result<Array, Exception> {
    if temperature == 0.0 {
        mlx_rs::ops::indexing::argmax_axis(logits, -1, false)
    } else {
        let scaled = logits.multiply(&array!(1.0 / temperature))?;
        mlx_rs::random::categorical(&scaled, None, None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_creation() {
        let config = LLMConfig::default();
        // This would be too large for a unit test, so just test config
        assert_eq!(config.hidden_size, 3584);
        assert_eq!(config.vocab_size, 158720);
    }
}
