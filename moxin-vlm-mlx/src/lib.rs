//! # moxin-vlm-mlx
//!
//! Moxin-7B Vision-Language Model inference on Apple Silicon with MLX.
//!
//! ## Architecture
//!
//! ```text
//! Image (224x224)
//!   |-> DINOv2 ViT-L/14 -> [B, 256, 1024]
//!   |-> SigLIP ViT-SO400M -> [B, 256, 1152]
//!              | concat
//!        [B, 256, 2176]
//!              | FusedMLPProjector
//!        [B, 256, 4096]  (256 visual tokens)
//!              |
//!   BOS + [visual tokens] + text tokens
//!              | Mistral-7B decoder (36 layers)
//!        logits -> autoregressive generation
//! ```

pub mod error;
pub mod projector;
pub mod vision;

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::{
    array,
    builder::Builder,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::indexing::IndexOp,
    quantization::{MaybeQuantized, Quantizable},
    Array,
};
use serde::Deserialize;

use error::{Error, Result};
use projector::FusedMLPProjector;
use vision::{ViTConfig, ViTEncoder};

// Re-exports
pub use mlx_rs_core::{
    cache::{ConcatKeyValueCache, KVCache, KeyValueCache},
    load_tokenizer,
};

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct VLMConfig {
    #[serde(default)]
    pub text_config: Option<MistralConfig>,
    #[serde(default = "default_image_sizes")]
    pub image_sizes: Vec<i32>,
    #[serde(default = "default_image_token_index")]
    pub image_token_index: i32,
}

fn default_image_sizes() -> Vec<i32> {
    vec![224, 224]
}
fn default_image_token_index() -> i32 {
    32000
}

#[derive(Debug, Clone, Deserialize)]
pub struct MistralConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    #[serde(default = "default_num_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_num_heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_kv_heads")]
    pub num_key_value_heads: i32,
    #[serde(default = "default_intermediate")]
    pub intermediate_size: i32,
    #[serde(default = "default_vocab")]
    pub vocab_size: i32,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_hidden_size() -> i32 { 4096 }
fn default_num_layers() -> i32 { 36 }
fn default_num_heads() -> i32 { 32 }
fn default_kv_heads() -> i32 { 8 }
fn default_intermediate() -> i32 { 14336 }
fn default_vocab() -> i32 { 32064 }
fn default_rms_eps() -> f32 { 1e-5 }
fn default_rope_theta() -> f32 { 10000.0 }

impl Default for MistralConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 14336,
            vocab_size: 32064,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
        }
    }
}

// ============================================================================
// Mistral Decoder Components
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct LLMAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub rope: nn::Rope,
}

pub struct LLMAttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut C,
}

impl<C: KeyValueCache> Module<LLMAttentionInput<'_, C>> for LLMAttention {
    type Output = Array;
    type Error = mlx_rs::error::Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: LLMAttentionInput<'_, C>) -> std::result::Result<Array, Self::Error> {
        let LLMAttentionInput { x, mask, cache } = input;
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        let mut queries = self.q_proj.forward(x)?
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = self.k_proj.forward(x)?
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = self.v_proj.forward(x)?
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q_input = nn::RopeInputBuilder::new(&queries)
            .offset(cache.offset())
            .build()?;
        queries = self.rope.forward(q_input)?;
        let k_input = nn::RopeInputBuilder::new(&keys)
            .offset(cache.offset())
            .build()?;
        keys = self.rope.forward(k_input)?;

        let (keys, values) = cache.update_and_fetch(keys, values)?;

        use mlx_rs_core::SdpaMask;
        let sdpa_mask = match mask {
            Some(m) => Some(SdpaMask::Array(m)),
            None if L > 1 => Some(SdpaMask::Causal),
            None => None,
        };

        let output = mlx_rs_core::scaled_dot_product_attention(
            queries,
            keys,
            values,
            Some(cache),
            self.scale,
            sdpa_mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        <nn::Rope as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct LLMFeedForward {
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
}

impl Module<&Array> for LLMFeedForward {
    type Output = Array;
    type Error = mlx_rs::error::Exception;

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = nn::silu(gate)?.multiply(up)?;
        self.down_proj.forward(&activated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct LLMBlock {
    #[param]
    pub self_attn: LLMAttention,
    #[param]
    pub mlp: LLMFeedForward,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl<C: KeyValueCache> Module<LLMAttentionInput<'_, C>> for LLMBlock {
    type Output = Array;
    type Error = mlx_rs::error::Exception;

    fn forward(&mut self, input: LLMAttentionInput<'_, C>) -> std::result::Result<Array, Self::Error> {
        let LLMAttentionInput { x, mask, cache } = input;
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(LLMAttentionInput {
            x: &normed,
            mask,
            cache,
        })?;
        let h = x.add(&attn_out)?;
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        h.add(&mlp_out)
    }

    fn training_mode(&mut self, mode: bool) {
        <LLMAttention as Module<LLMAttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

// ============================================================================
// Moxin VLM
// ============================================================================

#[derive(Debug)]
pub struct MoxinVLM {
    pub dino: ViTEncoder,
    pub siglip: ViTEncoder,
    pub projector: FusedMLPProjector,
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<LLMBlock>,
    pub norm: nn::RmsNorm,
    pub lm_head: MaybeQuantized<nn::Linear>,
    pub config: MistralConfig,
}

impl MoxinVLM {
    /// Full VLM forward: encode image + text -> logits
    ///
    /// - `dino_image`: [B, 224, 224, 3] NHWC, ImageNet-normalized
    /// - `siglip_image`: [B, 224, 224, 3] NHWC, unit-normalized
    /// - `input_ids`: [B, seq_len] token IDs (including BOS)
    pub fn forward<C: KeyValueCache + Default>(
        &mut self,
        dino_image: &Array,
        siglip_image: &Array,
        input_ids: &Array,
        cache: &mut Vec<C>,
    ) -> Result<Array> {
        // 1. Vision encoding
        let dino_features = self.dino.forward(dino_image)?;
        let siglip_features = self.siglip.forward(siglip_image)?;
        let fused_vision =
            mlx_rs::ops::concatenate_axis(&[&dino_features, &siglip_features], -1)?;

        // 2. Project to LLM space
        let visual_tokens = self.projector.forward(&fused_vision)?;

        // 3. Get text embeddings
        let text_embeds = self.embed_tokens.forward(input_ids)?;

        // 4. Assemble: BOS + visual_tokens + rest_of_text
        let bos_embed = text_embeds.index((.., ..1, ..));
        let rest_embed = text_embeds.index((.., 1.., ..));
        let fused_embed = mlx_rs::ops::concatenate_axis(
            &[&bos_embed, &visual_tokens, &rest_embed],
            1,
        )?;

        // 5. Decoder forward
        self.decoder_forward(&fused_embed, cache)
    }

    /// Text-only decode (single token, cached)
    pub fn decode_token<C: KeyValueCache + Default>(
        &mut self,
        token: &Array,
        cache: &mut Vec<C>,
    ) -> Result<Array> {
        let token = if token.ndim() == 1 {
            token.reshape(&[-1, 1])?
        } else {
            token.clone()
        };
        let embed = self.embed_tokens.forward(&token)?;
        self.decoder_forward(&embed, cache)
    }

    fn decoder_forward<C: KeyValueCache + Default>(
        &mut self,
        embeddings: &Array,
        cache: &mut Vec<C>,
    ) -> Result<Array> {
        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| C::default()).collect();
        }

        let mut h = embeddings.clone();
        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(LLMAttentionInput {
                x: &h,
                mask: None,
                cache: c,
            })?;
        }
        h = self.norm.forward(&h)?;
        Ok(self.lm_head.forward(&h)?)
    }

    /// Quantize the LLM decoder linear layers (keeps vision + projector in BF16).
    ///
    /// Only the Mistral-7B decoder is quantized since it dominates memory and
    /// compute. Vision encoders have dimensions (e.g. SigLIP 4304) that aren't
    /// cleanly divisible by common group sizes, and they only run once at prefill.
    pub fn quantize(self, group_size: i32, bits: i32) -> Result<Self> {
        let layers = self
            .layers
            .into_iter()
            .map(|block| quantize_llm_block(block, group_size, bits))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let lm_head = self.lm_head.try_into_quantized(group_size, bits)?;

        Ok(MoxinVLM {
            dino: self.dino,
            siglip: self.siglip,
            projector: self.projector,
            embed_tokens: self.embed_tokens,
            layers,
            norm: self.norm,
            lm_head,
            config: self.config,
        })
    }
}

fn quantize_llm_block(
    block: LLMBlock,
    group_size: i32,
    bits: i32,
) -> std::result::Result<LLMBlock, Error> {
    Ok(LLMBlock {
        self_attn: LLMAttention {
            n_heads: block.self_attn.n_heads,
            n_kv_heads: block.self_attn.n_kv_heads,
            head_dim: block.self_attn.head_dim,
            scale: block.self_attn.scale,
            q_proj: block.self_attn.q_proj.try_into_quantized(group_size, bits)?,
            k_proj: block.self_attn.k_proj.try_into_quantized(group_size, bits)?,
            v_proj: block.self_attn.v_proj.try_into_quantized(group_size, bits)?,
            o_proj: block.self_attn.o_proj.try_into_quantized(group_size, bits)?,
            rope: block.self_attn.rope,
        },
        mlp: LLMFeedForward {
            gate_proj: block.mlp.gate_proj.try_into_quantized(group_size, bits)?,
            up_proj: block.mlp.up_proj.try_into_quantized(group_size, bits)?,
            down_proj: block.mlp.down_proj.try_into_quantized(group_size, bits)?,
        },
        input_layernorm: block.input_layernorm,
        post_attention_layernorm: block.post_attention_layernorm,
    })
}

// ============================================================================
// Image preprocessing helpers
// ============================================================================

/// Normalize an image array for DINOv2 (ImageNet statistics).
///
/// Input: [B, H, W, 3] float in [0, 1]
pub fn normalize_dino(img: &Array) -> Result<Array> {
    let mean = Array::from_slice(&[0.485f32, 0.456, 0.406], &[1, 1, 1, 3]);
    let std = Array::from_slice(&[0.229f32, 0.224, 0.225], &[1, 1, 1, 3]);
    Ok(img.subtract(&mean)?.divide(&std)?)
}

/// Normalize an image array for SigLIP (unit normalization).
///
/// Input: [B, H, W, 3] float in [0, 1]
pub fn normalize_siglip(img: &Array) -> Result<Array> {
    let mean = Array::from_slice(&[0.5f32, 0.5, 0.5], &[1, 1, 1, 3]);
    let std = Array::from_slice(&[0.5f32, 0.5, 0.5], &[1, 1, 1, 3]);
    Ok(img.subtract(&mean)?.divide(&std)?)
}

// ============================================================================
// Generation
// ============================================================================

pub fn sample(logits: &Array, temp: f32) -> std::result::Result<Array, mlx_rs::error::Exception> {
    if temp == 0.0 {
        mlx_rs::argmax_axis!(logits, -1)
    } else {
        let logits = logits.multiply(array!(1.0 / temp))?;
        mlx_rs::categorical!(logits)
    }
}

/// Token generator for VLM inference.
pub struct Generate<'a, C> {
    vlm: &'a mut MoxinVLM,
    cache: &'a mut Vec<C>,
    temp: f32,
    state: GenerateState,
}

enum GenerateState {
    Prefill {
        dino_image: Array,
        siglip_image: Array,
        input_ids: Array,
    },
    Decode {
        next_token: Array,
    },
    Done,
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    pub fn new(
        vlm: &'a mut MoxinVLM,
        cache: &'a mut Vec<C>,
        temp: f32,
        dino_image: Array,
        siglip_image: Array,
        input_ids: Array,
    ) -> Self {
        Self {
            vlm,
            cache,
            temp,
            state: GenerateState::Prefill {
                dino_image,
                siglip_image,
                input_ids,
            },
        }
    }
}

impl<C: KeyValueCache + Default> Iterator for Generate<'_, C> {
    type Item = Result<Array>;

    fn next(&mut self) -> Option<Self::Item> {
        let state = std::mem::replace(&mut self.state, GenerateState::Done);

        match state {
            GenerateState::Prefill {
                dino_image,
                siglip_image,
                input_ids,
            } => {
                let logits = match self.vlm.forward(
                    &dino_image,
                    &siglip_image,
                    &input_ids,
                    self.cache,
                ) {
                    Ok(l) => l,
                    Err(e) => return Some(Err(e)),
                };
                let last_logits = logits.index((.., -1, ..));
                let token = match sample(&last_logits, self.temp) {
                    Ok(t) => t,
                    Err(e) => return Some(Err(e.into())),
                };
                if let Err(e) = mlx_rs::transforms::eval([&token]) {
                    return Some(Err(e.into()));
                }
                self.state = GenerateState::Decode {
                    next_token: token.clone(),
                };
                Some(Ok(token))
            }
            GenerateState::Decode { next_token } => {
                let logits = match self.vlm.decode_token(&next_token, self.cache) {
                    Ok(l) => l,
                    Err(e) => return Some(Err(e)),
                };
                let last_logits = logits.index((.., -1, ..));
                let token = match sample(&last_logits, self.temp) {
                    Ok(t) => t,
                    Err(e) => return Some(Err(e.into())),
                };
                if let Err(e) = mlx_rs::transforms::eval([&token]) {
                    return Some(Err(e.into()));
                }
                self.state = GenerateState::Decode {
                    next_token: token.clone(),
                };
                Some(Ok(token))
            }
            GenerateState::Done => None,
        }
    }
}

// ============================================================================
// Model loading
// ============================================================================

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

fn make_linear(weight: Array, bias: Option<Array>) -> MaybeQuantized<nn::Linear> {
    MaybeQuantized::new(nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(bias),
    })
}

fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>> {
    // Try sharded safetensors
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let json = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&json)?;
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| Error::InvalidConfig("Invalid weight index".to_string()))?;

        let files: HashSet<&str> = weight_map.values().filter_map(|v| v.as_str()).collect();
        let mut all: HashMap<String, Array> = HashMap::new();
        for file in files {
            let path = model_dir.join(file);
            let loaded = Array::load_safetensors(&path)
                .map_err(Error::MlxIo)?;
            all.extend(loaded);
        }
        return Ok(all);
    }

    // Try single safetensors
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Array::load_safetensors(&single).map_err(Error::MlxIo);
    }

    Err(Error::InvalidConfig(
        "No model.safetensors found".to_string(),
    ))
}

/// Load the Moxin-7B VLM from a directory containing safetensors + config.json.
pub fn load_model(model_dir: impl AsRef<Path>) -> Result<MoxinVLM> {
    let model_dir = model_dir.as_ref();

    // Load config
    let config_path = model_dir.join("config.json");
    let llm_config = if config_path.exists() {
        let json = std::fs::read_to_string(&config_path)?;
        let vlm_config: VLMConfig = serde_json::from_str(&json)?;
        vlm_config.text_config.unwrap_or_default()
    } else {
        MistralConfig::default()
    };

    eprintln!("Loading Moxin-7B VLM...");
    let weights = load_all_weights(model_dir)?;

    // Auto-detect weight key prefix for vision backbone
    // HF-converted format uses featurizer (DINOv2) + fused_featurizer (SigLIP)
    let vision_prefix = if weights.keys().any(|k| k.starts_with("vision_backbone.featurizer.blocks.")) {
        (
            "vision_backbone.featurizer",
            "vision_backbone.fused_featurizer",
        )
    } else {
        (
            "vision_backbone.featurizer.0",
            "vision_backbone.featurizer.1",
        )
    };

    // Auto-detect LLM prefix
    let llm_prefix = if weights.keys().any(|k| k.starts_with("language_model.")) {
        "language_model"
    } else {
        "llm_backbone.llm"
    };

    // Load vision encoders
    eprintln!("  Loading DINOv2 ViT-L/14...");
    let dino = vision::load_vit_encoder(&weights, vision_prefix.0, ViTConfig::dinov2_large())?;

    eprintln!("  Loading SigLIP ViT-SO400M/14...");
    let siglip = vision::load_vit_encoder(&weights, vision_prefix.1, ViTConfig::siglip_so400m())?;

    // Load projector
    eprintln!("  Loading projector...");
    let projector = projector::load_projector(&weights, "projector")?;

    // Load LLM decoder
    eprintln!(
        "  Loading Mistral-7B decoder ({} layers)...",
        llm_config.num_hidden_layers
    );

    let head_dim = llm_config.hidden_size / llm_config.num_attention_heads;
    let n_heads = llm_config.num_attention_heads;
    let n_kv_heads = llm_config.num_key_value_heads;

    let embed_tokens = nn::Embedding {
        weight: Param::new(get_weight(
            &weights,
            &format!("{}.model.embed_tokens.weight", llm_prefix),
        )?),
    };

    let mut layers = Vec::with_capacity(llm_config.num_hidden_layers as usize);
    for i in 0..llm_config.num_hidden_layers {
        let lp = format!("{}.model.layers.{}", llm_prefix, i);

        let attention = LLMAttention {
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            q_proj: make_linear(
                get_weight(&weights, &format!("{}.self_attn.q_proj.weight", lp))?,
                None,
            ),
            k_proj: make_linear(
                get_weight(&weights, &format!("{}.self_attn.k_proj.weight", lp))?,
                None,
            ),
            v_proj: make_linear(
                get_weight(&weights, &format!("{}.self_attn.v_proj.weight", lp))?,
                None,
            ),
            o_proj: make_linear(
                get_weight(&weights, &format!("{}.self_attn.o_proj.weight", lp))?,
                None,
            ),
            rope: nn::RopeBuilder::new(head_dim)
                .traditional(false)
                .base(llm_config.rope_theta)
                .build()
                .unwrap(),
        };

        let mlp = LLMFeedForward {
            gate_proj: make_linear(
                get_weight(&weights, &format!("{}.mlp.gate_proj.weight", lp))?,
                None,
            ),
            up_proj: make_linear(
                get_weight(&weights, &format!("{}.mlp.up_proj.weight", lp))?,
                None,
            ),
            down_proj: make_linear(
                get_weight(&weights, &format!("{}.mlp.down_proj.weight", lp))?,
                None,
            ),
        };

        layers.push(LLMBlock {
            self_attn: attention,
            mlp,
            input_layernorm: nn::RmsNorm {
                weight: Param::new(get_weight(
                    &weights,
                    &format!("{}.input_layernorm.weight", lp),
                )?),
                eps: llm_config.rms_norm_eps,
            },
            post_attention_layernorm: nn::RmsNorm {
                weight: Param::new(get_weight(
                    &weights,
                    &format!("{}.post_attention_layernorm.weight", lp),
                )?),
                eps: llm_config.rms_norm_eps,
            },
        });
    }

    let norm = nn::RmsNorm {
        weight: Param::new(get_weight(
            &weights,
            &format!("{}.model.norm.weight", llm_prefix),
        )?),
        eps: llm_config.rms_norm_eps,
    };

    let lm_head = if llm_config.tie_word_embeddings {
        make_linear(
            get_weight(&weights, &format!("{}.model.embed_tokens.weight", llm_prefix))?,
            None,
        )
    } else {
        make_linear(
            get_weight(&weights, &format!("{}.lm_head.weight", llm_prefix))?,
            None,
        )
    };

    eprintln!("Model loaded successfully.");

    Ok(MoxinVLM {
        dino,
        siglip,
        projector,
        embed_tokens,
        layers,
        norm,
        lm_head,
        config: llm_config,
    })
}
