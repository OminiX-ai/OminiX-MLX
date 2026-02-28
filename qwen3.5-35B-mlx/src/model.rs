use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::{
    error::Exception,
    module::{Module, ModuleParameters, Param},
    nn,
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;

use mlx_rs_core::{
    cache::{KVCache, KeyValueCache},
    error::Error,
    utils::initialize_rope,
};

use crate::attention::{GatedAttention, GatedAttentionInput};
use crate::cache::{HybridCache, RecurrentState};
use crate::config::{ModelArgs, TextConfig};
use crate::deltanet::GatedDeltaNet;
use crate::mlp::Mlp;

// ============================================================================
// Layer Types
// ============================================================================

pub enum AttentionLayer {
    FullAttention(GatedAttention),
    LinearAttention(GatedDeltaNet),
}

pub struct TransformerBlock {
    pub attention: AttentionLayer,
    pub mlp: Mlp,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut HybridCache,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;

        let attn_out = match (&mut self.attention, cache) {
            (AttentionLayer::FullAttention(attn), HybridCache::KV(kv_cache)) => {
                attn.forward(GatedAttentionInput {
                    x: &normed,
                    mask,
                    cache: Some(kv_cache),
                })?
            }
            (AttentionLayer::LinearAttention(delta), HybridCache::Recurrent(rec_cache)) => {
                let L = normed.shape()[1];
                if L > 1 {
                    delta.forward_prefill(&normed, rec_cache)?
                } else {
                    delta.forward_step(&normed, rec_cache)?
                }
            }
            _ => return Err(Exception::custom("Cache type mismatch with layer type")),
        };

        let h = x.add(attn_out)?;
        let mlp_out = self.mlp.forward(&self.post_attention_layernorm.forward(&h)?)?;
        h.add(mlp_out)
    }
}

// ============================================================================
// Full Model
// ============================================================================

pub struct Qwen35TextModel {
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    pub layers: Vec<TransformerBlock>,
    pub norm: nn::RmsNorm,
    pub layer_types: Vec<String>,
}

pub struct Model {
    pub args: ModelArgs,
    pub text_model: Qwen35TextModel,
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<HybridCache>,
    ) -> Result<Array, Exception> {
        let mut h = self.text_model.embed_tokens.forward(inputs)?;

        // Create attention mask (only needed for full attention layers)
        // Use the first KV cache offset for mask creation
        let T = h.shape()[1];
        let mask = if T > 1 {
            let offset = cache
                .iter()
                .find_map(|c| match c {
                    HybridCache::KV(kv) => Some(kv.offset()),
                    _ => None,
                })
                .unwrap_or(0);
            // Create causal mask as array for multi-token input
            let mask = mlx_rs_core::utils::create_causal_mask(T, Some(offset), None, None)?;
            Some(mask)
        } else {
            None
        };

        // Initialize caches on first call
        if cache.is_empty() {
            for layer_type in &self.text_model.layer_types {
                if layer_type == "full_attention" {
                    cache.push(HybridCache::KV(KVCache::new()));
                } else {
                    cache.push(HybridCache::Recurrent(RecurrentState::new()));
                }
            }
        }

        for (layer, c) in self.text_model.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(&h, mask.as_ref(), c)?;
        }

        h = self.text_model.norm.forward(&h)?;

        // LM head
        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&h),
            None => match &mut self.text_model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(&h),
                MaybeQuantized::Quantized(qe) => qe.as_linear(&h),
            },
        }
    }
}

// ============================================================================
// Weight Loading
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub weight_map: HashMap<String, String>,
}

fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>, Error> {
    let weights_index = model_dir.join("model.safetensors.index.json");

    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();

        let mut all_weights: HashMap<String, Array> = HashMap::new();
        for weight_file in weight_files {
            let path = model_dir.join(weight_file);
            let loaded = Array::load_safetensors(&path)?;
            all_weights.extend(loaded);
        }
        Ok(all_weights)
    } else {
        // Single safetensors file
        let path = model_dir.join("model.safetensors");
        let loaded = Array::load_safetensors(&path)?;
        Ok(loaded)
    }
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

fn make_quantized_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedLinear, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;

    let inner = nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(None),
    };

    let mut ql = nn::QuantizedLinear {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };
    ql.freeze_parameters(true);
    Ok(ql)
}

fn make_quantized_embedding(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedEmbedding, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;

    let inner = nn::Embedding {
        weight: Param::new(weight),
    };

    let mut qe = nn::QuantizedEmbedding {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };
    qe.freeze_parameters(true);
    Ok(qe)
}

fn load_rms_norm(
    weights: &HashMap<String, Array>,
    key: &str,
    eps: f32,
) -> Result<nn::RmsNorm, Error> {
    Ok(nn::RmsNorm {
        weight: Param::new(get_weight(weights, key)?),
        eps,
    })
}

/// Detect the weight key prefix (VLM vs standalone text model).
fn detect_prefix(weights: &HashMap<String, Array>) -> &'static str {
    if weights.keys().any(|k| k.starts_with("language_model.")) {
        "language_model.model"
    } else {
        "model"
    }
}

fn detect_lm_head_prefix(weights: &HashMap<String, Array>) -> &'static str {
    if weights.contains_key("language_model.lm_head.weight") {
        "language_model.lm_head"
    } else {
        "lm_head"
    }
}

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();

    // Load config
    let config_file = std::fs::File::open(model_dir.join("config.json"))?;
    let args: ModelArgs = serde_json::from_reader(config_file)?;
    let tc = &args.text_config;

    let quant = args.quantization();
    let (group_size, bits) = match quant {
        Some(q) => (q.group_size, q.bits),
        None => return Err(Error::Model("Only quantized models are supported".to_string())),
    };

    // Validate config consistency
    if tc.layer_types.len() != tc.num_hidden_layers as usize {
        return Err(Error::Model(format!(
            "layer_types length ({}) != num_hidden_layers ({})",
            tc.layer_types.len(),
            tc.num_hidden_layers
        )));
    }

    let weights = load_all_weights(model_dir)?;
    let prefix = detect_prefix(&weights);
    let lm_head_prefix = detect_lm_head_prefix(&weights);

    // Build layers
    let mut layers = Vec::with_capacity(tc.num_hidden_layers as usize);
    for i in 0..tc.num_hidden_layers {
        let layer_prefix = format!("{}.layers.{}", prefix, i);
        let layer_type = &tc.layer_types[i as usize];

        let attention = if layer_type == "full_attention" {
            AttentionLayer::FullAttention(load_gated_attention(
                &weights,
                &layer_prefix,
                tc,
                group_size,
                bits,
            )?)
        } else {
            AttentionLayer::LinearAttention(load_gated_deltanet(
                &weights,
                &layer_prefix,
                tc,
                group_size,
                bits,
            )?)
        };

        let mlp = Mlp {
            gate_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights,
                &format!("{}.mlp.gate_proj", layer_prefix),
                group_size,
                bits,
            )?),
            down_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights,
                &format!("{}.mlp.down_proj", layer_prefix),
                group_size,
                bits,
            )?),
            up_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights,
                &format!("{}.mlp.up_proj", layer_prefix),
                group_size,
                bits,
            )?),
        };

        let block = TransformerBlock {
            attention,
            mlp,
            input_layernorm: load_rms_norm(
                &weights,
                &format!("{}.input_layernorm.weight", layer_prefix),
                tc.rms_norm_eps,
            )?,
            post_attention_layernorm: load_rms_norm(
                &weights,
                &format!("{}.post_attention_layernorm.weight", layer_prefix),
                tc.rms_norm_eps,
            )?,
        };

        layers.push(block);
    }

    // Embed tokens
    let embed_tokens = MaybeQuantized::Quantized(make_quantized_embedding(
        &weights,
        &format!("{}.embed_tokens", prefix),
        group_size,
        bits,
    )?);

    // Final norm
    let norm = load_rms_norm(
        &weights,
        &format!("{}.norm.weight", prefix),
        tc.rms_norm_eps,
    )?;

    // LM head
    let lm_head = if !args.tie_word_embeddings {
        Some(MaybeQuantized::Quantized(make_quantized_linear(
            &weights,
            lm_head_prefix,
            group_size,
            bits,
        )?))
    } else {
        None
    };

    let text_model = Qwen35TextModel {
        embed_tokens,
        layers,
        norm,
        layer_types: tc.layer_types.clone(),
    };

    Ok(Model {
        args,
        text_model,
        lm_head,
    })
}

fn load_gated_attention(
    weights: &HashMap<String, Array>,
    layer_prefix: &str,
    tc: &TextConfig,
    group_size: i32,
    bits: i32,
) -> Result<GatedAttention, Error> {
    let attn_prefix = format!("{}.self_attn", layer_prefix);

    let rope_dims = (tc.head_dim as f32 * tc.rope_parameters.partial_rotary_factor) as i32;
    let rope = initialize_rope(
        rope_dims,
        tc.rope_parameters.rope_theta,
        false,
        &None,
        tc.max_position_embeddings,
    )?;

    Ok(GatedAttention {
        n_heads: tc.num_attention_heads,
        n_kv_heads: tc.num_key_value_heads,
        head_dim: tc.head_dim,
        scale: (tc.head_dim as f32).sqrt().recip(),
        q_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.q_proj", attn_prefix),
            group_size,
            bits,
        )?),
        k_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.k_proj", attn_prefix),
            group_size,
            bits,
        )?),
        v_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.v_proj", attn_prefix),
            group_size,
            bits,
        )?),
        o_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.o_proj", attn_prefix),
            group_size,
            bits,
        )?),
        q_norm: load_rms_norm(
            weights,
            &format!("{}.q_norm.weight", attn_prefix),
            tc.rms_norm_eps,
        )?,
        k_norm: load_rms_norm(
            weights,
            &format!("{}.k_norm.weight", attn_prefix),
            tc.rms_norm_eps,
        )?,
        rope,
    })
}

fn load_gated_deltanet(
    weights: &HashMap<String, Array>,
    layer_prefix: &str,
    tc: &TextConfig,
    group_size: i32,
    bits: i32,
) -> Result<GatedDeltaNet, Error> {
    let attn_prefix = format!("{}.linear_attn", layer_prefix);

    let num_k_heads = tc.linear_num_key_heads;
    let num_v_heads = tc.linear_num_value_heads;
    if num_v_heads % num_k_heads != 0 {
        return Err(Error::Model(format!(
            "linear_num_value_heads ({}) must be divisible by linear_num_key_heads ({})",
            num_v_heads, num_k_heads
        )));
    }
    let key_head_dim = tc.linear_key_head_dim;
    let value_head_dim = tc.linear_value_head_dim;
    let key_dim = num_k_heads * key_head_dim;
    let value_dim = num_v_heads * value_head_dim;
    let conv_dim = key_dim * 2 + value_dim;
    let conv_kernel_size = tc.linear_conv_kernel_dim;

    Ok(GatedDeltaNet {
        in_proj_qkv: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.in_proj_qkv", attn_prefix),
            group_size,
            bits,
        )?),
        in_proj_z: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.in_proj_z", attn_prefix),
            group_size,
            bits,
        )?),
        in_proj_a: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.in_proj_a", attn_prefix),
            group_size,
            bits,
        )?),
        in_proj_b: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.in_proj_b", attn_prefix),
            group_size,
            bits,
        )?),
        conv1d_weight: Param::new(get_weight(
            weights,
            &format!("{}.conv1d.weight", attn_prefix),
        )?),
        a_log: Param::new(get_weight(
            weights,
            &format!("{}.A_log", attn_prefix),
        )?),
        dt_bias: Param::new(get_weight(
            weights,
            &format!("{}.dt_bias", attn_prefix),
        )?),
        norm: load_rms_norm(
            weights,
            &format!("{}.norm.weight", attn_prefix),
            tc.rms_norm_eps,
        )?,
        out_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.out_proj", attn_prefix),
            group_size,
            bits,
        )?),
        num_k_heads,
        num_v_heads,
        key_head_dim,
        value_head_dim,
        key_dim,
        value_dim,
        conv_dim,
        conv_kernel_size,
    })
}
