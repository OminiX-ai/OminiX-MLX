//! Qwen3-4B LLM backbone for FunASR-Qwen4B.
//!
//! This module provides the Qwen3-4B language model used for text generation.
//! Based on the Qwen implementation in funasr-nano-mlx, adapted for 4B model.

use crate::error::{Error, Result};
use mlx_rs_core::{initialize_rope, KeyValueCache, KVCache};
use mlx_rs::fast::ScaledDotProductAttentionMask;
use mlx_rs::builder::Builder;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::{Module, ModuleParameters};
use mlx_rs::nn;
use mlx_rs::transforms::eval;
use mlx_rs::Array;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

/// Qwen3-4B model configuration.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen4BConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: i32,
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl Default for Qwen4BConfig {
    fn default() -> Self {
        // Qwen3-4B configuration from HuggingFace config.json
        Self {
            hidden_size: 2560,
            num_hidden_layers: 36,
            intermediate_size: 9728,        // ~3.8x hidden_size
            num_attention_heads: 32,        // 32 heads
            num_key_value_heads: 8,         // GQA: 32/8 = 4 Q heads per KV head
            head_dim: 128,                  // 4096 / 32 = 128
            vocab_size: 151936,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 40960,
            rope_theta: 1000000.0,
            rope_scaling: None,
            tie_word_embeddings: true,
        }
    }
}

/// Qwen3-4B attention with Group Query Attention (GQA).
#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen4BAttention {
    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub o_proj: nn::Linear,
    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: nn::RmsNorm,
    #[param]
    pub rope: nn::Rope,

    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
}

impl Qwen4BAttention {
    pub fn new(config: &Qwen4BConfig) -> Result<Self> {
        let dim = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        let rope = initialize_rope(
            head_dim,
            config.rope_theta,
            false,
            &config.rope_scaling.as_ref().map(|m| {
                m.iter()
                    .map(|(k, v)| {
                        let fos = match v {
                            serde_json::Value::Number(n) => {
                                mlx_rs_core::FloatOrString::Float(n.as_f64().unwrap_or(1.0) as f32)
                            }
                            serde_json::Value::String(s) => {
                                mlx_rs_core::FloatOrString::String(s.clone())
                            }
                            _ => mlx_rs_core::FloatOrString::Float(1.0),
                        };
                        (k.clone(), fos)
                    })
                    .collect()
            }),
            config.max_position_embeddings,
        )?;

        Ok(Self {
            q_proj: nn::LinearBuilder::new(dim, n_heads * head_dim).bias(false).build()?,
            k_proj: nn::LinearBuilder::new(dim, n_kv_heads * head_dim).bias(false).build()?,
            v_proj: nn::LinearBuilder::new(dim, n_kv_heads * head_dim).bias(false).build()?,
            o_proj: nn::LinearBuilder::new(n_heads * head_dim, dim).bias(false).build()?,
            q_norm: nn::RmsNormBuilder::new(head_dim).eps(config.rms_norm_eps).build()?,
            k_norm: nn::RmsNormBuilder::new(head_dim).eps(config.rms_norm_eps).build()?,
            rope,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        cache: &mut Option<KVCache>,
        mask: Option<&Array>,
    ) -> std::result::Result<Array, mlx_rs::error::Exception> {
        let shape = x.shape();
        let (batch, seq_len, _) = (shape[0], shape[1], shape[2]);

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [B, L, n_heads, head_dim]
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;

        // Apply Q/K normalization
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Transpose to [B, n_heads, L, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE
        let (q, k, v) = if let Some(cache) = cache.as_mut() {
            let offset = cache.offset();
            let q = self.rope.forward(
                nn::RopeInputBuilder::new(&q).offset(offset).build()?,
            )?;
            let k = self.rope.forward(
                nn::RopeInputBuilder::new(&k).offset(offset).build()?,
            )?;
            let (k, v) = cache.update_and_fetch(k, v)?;
            (q, k, v)
        } else {
            let q = self.rope.forward(nn::RopeInput::new(&q))?;
            let k = self.rope.forward(nn::RopeInput::new(&k))?;
            (q, k, v)
        };

        // Scaled dot-product attention
        let attn_out = match mask {
            Some(m) => mlx_rs::fast::scaled_dot_product_attention(
                q, k, v, self.scale, ScaledDotProductAttentionMask::Array(m),
            )?,
            None if seq_len > 1 => mlx_rs::fast::scaled_dot_product_attention(
                q, k, v, self.scale, ScaledDotProductAttentionMask::Causal,
            )?,
            None => mlx_rs::fast::scaled_dot_product_attention(
                q, k, v, self.scale, None::<ScaledDotProductAttentionMask>,
            )?,
        };

        // Reshape back to [B, L, dim]
        let attn_out = attn_out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, self.n_heads * self.head_dim])?;

        self.o_proj.forward(&attn_out)
    }
}

/// Qwen3-4B MLP with SwiGLU activation.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen4BMLP {
    #[param]
    pub gate_proj: nn::Linear,
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
}

impl Qwen4BMLP {
    pub fn new(config: &Qwen4BConfig) -> Result<Self> {
        let dim = config.hidden_size;
        let hidden_dim = config.intermediate_size;

        Ok(Self {
            gate_proj: nn::LinearBuilder::new(dim, hidden_dim).bias(false).build()?,
            up_proj: nn::LinearBuilder::new(dim, hidden_dim).bias(false).build()?,
            down_proj: nn::LinearBuilder::new(hidden_dim, dim).bias(false).build()?,
        })
    }
}

impl Module<&Array> for Qwen4BMLP {
    type Output = Array;
    type Error = mlx_rs::error::Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = nn::silu(&gate)?.multiply(&up)?;
        self.down_proj.forward(&activated)
    }
}

/// Qwen3-4B transformer block.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen4BBlock {
    #[param]
    pub self_attn: Qwen4BAttention,
    #[param]
    pub mlp: Qwen4BMLP,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl Qwen4BBlock {
    pub fn new(config: &Qwen4BConfig) -> Result<Self> {
        Ok(Self {
            self_attn: Qwen4BAttention::new(config)?,
            mlp: Qwen4BMLP::new(config)?,
            input_layernorm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
        })
    }

    pub fn forward_with_cache(
        &mut self,
        x: &Array,
        cache: &mut Option<KVCache>,
        mask: Option<&Array>,
    ) -> std::result::Result<Array, mlx_rs::error::Exception> {
        // Self-attention with residual
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward_with_cache(&x, cache, mask)?;
        let x = residual.add(&x)?;

        // MLP with residual
        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual.add(&x)
    }
}

/// Qwen3-4B language model.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen4BModel {
    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub layers: Vec<Qwen4BBlock>,
    #[param]
    pub norm: nn::RmsNorm,
    #[param]
    pub lm_head: nn::Linear,

    pub config: Qwen4BConfig,
}

impl Qwen4BModel {
    /// Create a new Qwen3-4B model.
    pub fn new(config: Qwen4BConfig) -> Result<Self> {
        let n_layers = config.num_hidden_layers as usize;

        let layers: Result<Vec<_>> = (0..n_layers)
            .map(|_| Qwen4BBlock::new(&config))
            .collect();

        Ok(Self {
            embed_tokens: nn::Embedding::new(config.vocab_size, config.hidden_size)?,
            layers: layers?,
            norm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
            lm_head: nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
                .bias(false)
                .build()?,
            config,
        })
    }

    /// Get the embedding dimension (2560 for Qwen3-4B).
    pub fn hidden_size(&self) -> i32 {
        self.config.hidden_size
    }

    /// Forward pass with token inputs.
    pub fn forward_tokens(
        &mut self,
        tokens: &Array,
        cache: &mut Vec<Option<KVCache>>,
    ) -> std::result::Result<Array, mlx_rs::error::Exception> {
        let h = self.embed_tokens.forward(tokens)?;
        self.forward_embeddings(&h, cache)
    }

    /// Forward pass with embedding inputs (for multimodal).
    pub fn forward_embeddings(
        &mut self,
        embeddings: &Array,
        cache: &mut Vec<Option<KVCache>>,
    ) -> std::result::Result<Array, mlx_rs::error::Exception> {
        // Ensure cache has correct size
        if cache.len() != self.layers.len() {
            cache.resize_with(self.layers.len(), || Some(KVCache::new()));
        }

        let mut h = embeddings.clone();

        for (layer, layer_cache) in self.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward_with_cache(&h, layer_cache, None)?;
        }

        h = self.norm.forward(&h)?;
        self.lm_head.forward(&h)
    }

    /// Get token embeddings (for multimodal injection).
    pub fn get_token_embeddings(&mut self, tokens: &Array) -> std::result::Result<Array, mlx_rs::error::Exception> {
        self.embed_tokens.forward(tokens)
    }

    /// Load weights from safetensors file.
    ///
    /// Expects weights in MLX format (converted by scripts/convert_qwen4b_weights.py):
    /// - embed_tokens.weight
    /// - layers.X.self_attn.q_proj.weight
    /// - layers.X.self_attn.k_proj.weight
    /// - ...
    /// - norm.weight
    /// - lm_head.weight
    pub fn load_weights(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(Error::ModelLoad(format!(
                "Model file not found: {}. Please convert weights first.",
                path.display()
            )));
        }

        eprintln!("Loading Qwen4B weights from {}...", path.display());

        // Load safetensors
        let loaded = Array::load_safetensors(path)
            .map_err(|e| Error::ModelLoad(format!("Failed to load safetensors: {:?}", e)))?;

        // Get mutable parameters
        let mut params = self.parameters_mut().flatten();

        let mut loaded_count = 0;
        let mut skipped_keys = Vec::new();

        for (st_key, value) in loaded {
            // Map safetensors key to Rust parameter name
            let rust_key = Self::map_safetensors_key(&st_key);

            if let Some(param) = params.get_mut(&*rust_key) {
                **param = value;
                loaded_count += 1;
            } else {
                skipped_keys.push(st_key.clone());
            }
        }

        eprintln!("Loaded {} parameters", loaded_count);
        if !skipped_keys.is_empty() {
            eprintln!(
                "Skipped {} keys: {:?}",
                skipped_keys.len(),
                &skipped_keys[..skipped_keys.len().min(10)]
            );
        }

        // Evaluate loaded parameters
        eval(params.values().map(|v| &**v))?;

        // Drop params before handling tied embeddings
        drop(params);

        // Handle tied embeddings: lm_head shares weights with embed_tokens
        if self.config.tie_word_embeddings && skipped_keys.iter().any(|k| k == "lm_head.weight") {
            eprintln!("Tying lm_head.weight to embed_tokens.weight");
            let embed_weight = self.embed_tokens.weight.as_ref().clone();
            *self.lm_head.weight = embed_weight;
            eval([&*self.lm_head.weight])?;
        }

        Ok(())
    }

    /// Map safetensors key to Rust parameter name.
    ///
    /// HuggingFace keys (after removing "model." prefix):
    /// - embed_tokens.weight
    /// - layers.X.self_attn.q_proj.weight
    /// - layers.X.self_attn.k_proj.weight
    /// - layers.X.self_attn.v_proj.weight
    /// - layers.X.self_attn.o_proj.weight
    /// - layers.X.self_attn.q_norm.weight
    /// - layers.X.self_attn.k_norm.weight
    /// - layers.X.mlp.gate_proj.weight
    /// - layers.X.mlp.up_proj.weight
    /// - layers.X.mlp.down_proj.weight
    /// - layers.X.input_layernorm.weight
    /// - layers.X.post_attention_layernorm.weight
    /// - norm.weight
    /// - lm_head.weight
    ///
    /// Rust parameter names match this structure exactly.
    fn map_safetensors_key(st_key: &str) -> Rc<str> {
        // The conversion script already strips "model." prefix,
        // so keys should match our parameter names directly
        Rc::from(st_key)
    }

    /// Load weights from a directory containing sharded safetensors.
    pub fn load_weights_from_dir(&mut self, model_dir: impl AsRef<Path>) -> Result<()> {
        let model_dir = model_dir.as_ref();

        // Look for consolidated model_mlx.safetensors first
        let consolidated_path = model_dir.join("model_mlx.safetensors");
        if consolidated_path.exists() {
            return self.load_weights(&consolidated_path);
        }

        // Otherwise load sharded files
        let mut all_weights: HashMap<String, Array> = HashMap::new();

        for entry in std::fs::read_dir(model_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map(|e| e == "safetensors").unwrap_or(false) {
                eprintln!("Loading {}...", path.file_name().unwrap().to_string_lossy());
                let loaded = Array::load_safetensors(&path)
                    .map_err(|e| Error::ModelLoad(format!("Failed to load safetensors: {:?}", e)))?;

                for (key, value) in loaded {
                    // Strip "model." prefix if present
                    let key = if key.starts_with("model.") {
                        key[6..].to_string()
                    } else {
                        key
                    };
                    all_weights.insert(key, value);
                }
            }
        }

        if all_weights.is_empty() {
            return Err(Error::ModelLoad(format!(
                "No safetensors files found in {}",
                model_dir.display()
            )));
        }

        // Get mutable parameters and load
        let mut params = self.parameters_mut().flatten();
        let mut loaded_count = 0;

        for (key, value) in all_weights {
            let rust_key = Self::map_safetensors_key(&key);
            if let Some(param) = params.get_mut(&*rust_key) {
                **param = value;
                loaded_count += 1;
            }
        }

        eprintln!("Loaded {} parameters from sharded files", loaded_count);

        // Evaluate loaded parameters
        eval(params.values().map(|v| &**v))?;

        // Drop params before handling tied embeddings
        drop(params);

        // Handle tied embeddings: lm_head shares weights with embed_tokens
        if self.config.tie_word_embeddings {
            eprintln!("Tying lm_head.weight to embed_tokens.weight");
            let embed_weight = self.embed_tokens.weight.as_ref().clone();
            *self.lm_head.weight = embed_weight;
            eval([&*self.lm_head.weight])?;
        }

        Ok(())
    }
}
