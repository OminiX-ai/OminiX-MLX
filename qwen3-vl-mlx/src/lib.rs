//! # qwen3-vl-mlx
//!
//! Qwen3-VL Vision-Language Model inference on Apple Silicon with MLX.
//!
//! ## Architecture
//!
//! ```text
//! Image (H x W x 3)
//!   |-> Conv3d PatchEmbed -> [N_patches, 1024]
//!   |-> + pos_embed
//!   |-> 24 VisionBlocks (LN + QKV attn + LN + GELU MLP)
//!   |     DeepStack mergers at blocks [5, 11, 17] -> intermediate [N/4, 2560]
//!   |-> PatchMerger (final) -> [N/4, 2560]
//!   |-> sum all deepstack + final -> [N_visual, 2560]
//!
//! Qwen3 LM decoder (36 layers, GQA 32/8, q_norm/k_norm, RoPE theta=5M)
//!   - embed_tokens (quantized 8-bit)
//!   - Replace image token positions with visual features
//!   - Generate text autoregressively
//! ```

pub mod error;

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::{
    array,
    builder::Builder,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::indexing::IndexOp,
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;

use error::{Error, Result};

pub use mlx_rs_core::cache::{ConcatKeyValueCache, KVCache, KeyValueCache};

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_depth")]
    pub depth: usize,
    #[serde(default = "default_vision_hidden")]
    pub hidden_size: i32,
    #[serde(default = "default_num_heads")]
    pub num_heads: i32,
    #[serde(default = "default_intermediate")]
    pub intermediate_size: i32,
    #[serde(default = "default_patch_size")]
    pub patch_size: i32,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: i32,
    #[serde(default = "default_in_channels")]
    pub in_channels: i32,
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: i32,
    #[serde(default = "default_num_pos_emb")]
    pub num_position_embeddings: i32,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: i32,
    #[serde(default = "default_deepstack_indexes")]
    pub deepstack_visual_indexes: Vec<usize>,
}

fn default_depth() -> usize { 24 }
fn default_vision_hidden() -> i32 { 1024 }
fn default_num_heads() -> i32 { 16 }
fn default_intermediate() -> i32 { 4096 }
fn default_patch_size() -> i32 { 16 }
fn default_temporal_patch_size() -> i32 { 2 }
fn default_in_channels() -> i32 { 3 }
fn default_out_hidden_size() -> i32 { 2560 }
fn default_num_pos_emb() -> i32 { 2304 }
fn default_spatial_merge_size() -> i32 { 2 }
fn default_deepstack_indexes() -> Vec<usize> { vec![5, 11, 17] }

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            depth: 24,
            hidden_size: 1024,
            num_heads: 16,
            intermediate_size: 4096,
            patch_size: 16,
            temporal_patch_size: 2,
            in_channels: 3,
            out_hidden_size: 2560,
            num_position_embeddings: 2304,
            spatial_merge_size: 2,
            deepstack_visual_indexes: vec![5, 11, 17],
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    #[serde(default = "default_num_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_num_attn_heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_kv_heads")]
    pub num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    #[serde(default = "default_text_intermediate")]
    pub intermediate_size: i32,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_vocab_size() -> i32 { 151936 }
fn default_hidden_size() -> i32 { 2560 }
fn default_num_layers() -> i32 { 36 }
fn default_num_attn_heads() -> i32 { 32 }
fn default_kv_heads() -> i32 { 8 }
fn default_head_dim() -> i32 { 128 }
fn default_text_intermediate() -> i32 { 9728 }
fn default_rms_eps() -> f32 { 1e-6 }
fn default_rope_theta() -> f64 { 5_000_000.0 }

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 2560,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 9728,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            tie_word_embeddings: true,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Qwen3VLConfig {
    #[serde(default)]
    pub vision_config: VisionConfig,
    #[serde(default)]
    pub text_config: TextConfig,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: i32,
    #[serde(default = "default_vision_start_token_id")]
    pub vision_start_token_id: i32,
    #[serde(default = "default_vision_end_token_id")]
    pub vision_end_token_id: i32,
}

fn default_image_token_id() -> i32 { 151655 }
fn default_vision_start_token_id() -> i32 { 151652 }
fn default_vision_end_token_id() -> i32 { 151653 }

impl Default for Qwen3VLConfig {
    fn default() -> Self {
        Self {
            vision_config: VisionConfig::default(),
            text_config: TextConfig::default(),
            image_token_id: 151655,
            vision_start_token_id: 151652,
            vision_end_token_id: 151653,
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
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
) -> Result<MaybeQuantized<nn::Linear>> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases_arr = get_weight(weights, &format!("{}.biases", prefix))?;
    let inner = nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(None),
    };
    Ok(MaybeQuantized::Quantized(nn::QuantizedLinear {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases_arr),
        inner,
    }))
}

fn make_quantized_embedding(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<MaybeQuantized<nn::Embedding>> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases_arr = get_weight(weights, &format!("{}.biases", prefix))?;
    let inner = nn::Embedding {
        weight: Param::new(weight),
    };
    Ok(MaybeQuantized::Quantized(nn::QuantizedEmbedding {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases_arr),
        inner,
    }))
}

// ============================================================================
// Vision Encoder
// ============================================================================

fn apply_gelu(x: Array) -> std::result::Result<Array, mlx_rs::error::Exception> {
    nn::gelu_approximate(x)
}

/// PatchEmbed: Conv3d to extract image patches.
/// Weight: [out_channels, temporal, H_patch, W_patch, in_channels] = [1024, 2, 16, 16, 3]
pub struct PatchEmbed {
    pub proj: nn::Conv3d,
}

impl PatchEmbed {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        // x: [1, T, H, W, C] (channels last)
        let out = self.proj.forward(x)?;
        // out: [1, T', H', W', 1024] where T'=1, H'=H/16, W'=W/16
        let shape = out.shape();
        let n = shape[2] * shape[3]; // H' * W'
        // Reshape to [n, 1024]
        Ok(out.reshape(&[n, shape[4]])?)
    }
}

/// Vision block LayerNorm (with bias).
pub struct VisionLayerNorm {
    pub weight: Array,
    pub bias: Array,
    pub eps: f32,
}

impl VisionLayerNorm {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        // Use mlx_rs layernorm fast path
        let w = Some(&self.weight);
        let b = Some(&self.bias);
        Ok(mlx_rs::fast::layer_norm(x, w, b, self.eps)?)
    }
}

/// Vision self-attention (full attention, no windowing).
/// Uses combined QKV projection then split.
pub struct VisionAttention {
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    // QKV combined [3*hidden, hidden]
    pub qkv: nn::Linear,
    pub proj: nn::Linear,
}

impl VisionAttention {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let shape = x.shape();
        let n = shape[0]; // sequence length
        let hidden = shape[1];

        // QKV projection: [N, 3*hidden]
        let qkv = self.qkv.forward(x)?;
        // Split: each [N, hidden]
        let parts = qkv.split(3, -1)?;
        let (q, k, v) = (&parts[0], &parts[1], &parts[2]);

        // Reshape to [num_heads, N, head_dim]
        let q = q
            .reshape(&[1, n, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?; // [1, H, N, D]
        let k = k
            .reshape(&[1, n, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[1, n, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Use MLX SDPA — avoids materializing full attention matrix
        let out = mlx_rs::fast::scaled_dot_product_attention(q, k, v, self.scale, None)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[n, hidden])?;

        Ok(self.proj.forward(&out)?)
    }
}

/// Vision MLP: linear_fc1 (GELU) + linear_fc2
pub struct VisionMlp {
    pub linear_fc1: nn::Linear,
    pub linear_fc2: nn::Linear,
}

impl VisionMlp {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let h = apply_gelu(self.linear_fc1.forward(x)?)?;
        Ok(self.linear_fc2.forward(&h)?)
    }
}

/// One vision transformer block: LN1 + Attn + LN2 + MLP (pre-norm).
pub struct VisionBlock {
    pub norm1: VisionLayerNorm,
    pub norm2: VisionLayerNorm,
    pub attn: VisionAttention,
    pub mlp: VisionMlp,
}

impl VisionBlock {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let normed = self.norm1.forward(x)?;
        let attn_out = self.attn.forward(&normed)?;
        let h = x.add(&attn_out)?;
        let normed2 = self.norm2.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed2)?;
        Ok(h.add(&mlp_out)?)
    }
}

/// PatchMerger with LayerNorm before reshape (final merger).
/// norm is applied on hidden_size (1024), then reshape to 4*hidden_size.
pub struct FinalMerger {
    pub norm: VisionLayerNorm,     // on hidden_size
    pub linear_fc1: nn::Linear,   // [4096, 4096]
    pub linear_fc2: nn::Linear,   // [2560, 4096]
}

impl FinalMerger {
    /// x: [N, hidden_size], returns [N/4, out_hidden_size]
    pub fn forward(&mut self, x: &Array, spatial_merge_size: i32) -> Result<Array> {
        let normed = self.norm.forward(x)?;
        // Reshape to [N/4, 4*hidden]
        let shape = normed.shape();
        let n = shape[0];
        let d = shape[1];
        let m = spatial_merge_size * spatial_merge_size;
        let merged = normed.reshape(&[n / m, m * d])?;
        // MLP
        let h = apply_gelu(self.linear_fc1.forward(&merged)?)?;
        Ok(self.linear_fc2.forward(&h)?)
    }
}

/// DeepStack PatchMerger: LayerNorm after reshape (on 4*hidden_size).
pub struct DeepStackMerger {
    pub norm: VisionLayerNorm,    // on 4*hidden_size
    pub linear_fc1: nn::Linear,  // [4096, 4096]
    pub linear_fc2: nn::Linear,  // [2560, 4096]
}

impl DeepStackMerger {
    /// x: [N, hidden_size], returns [N/4, out_hidden_size]
    pub fn forward(&mut self, x: &Array, spatial_merge_size: i32) -> Result<Array> {
        let shape = x.shape();
        let n = shape[0];
        let d = shape[1];
        let m = spatial_merge_size * spatial_merge_size;
        // Reshape to [N/4, 4*hidden]
        let merged = x.reshape(&[n / m, m * d])?;
        let normed = self.norm.forward(&merged)?;
        let h = apply_gelu(self.linear_fc1.forward(&normed)?)?;
        Ok(self.linear_fc2.forward(&h)?)
    }
}

/// Full vision tower (VisionModel).
pub struct VisionTower {
    pub patch_embed: PatchEmbed,
    pub pos_embed: nn::Embedding,
    pub blocks: Vec<VisionBlock>,
    pub deepstack_mergers: Vec<DeepStackMerger>,
    pub merger: FinalMerger,
    pub config: VisionConfig,
}

impl VisionTower {
    /// Encode an image.
    ///
    /// - `pixels`: [1, temporal, H, W, 3] float32 in [0, 1] (channels last)
    /// - `h_patches`, `w_patches`: number of patches in each spatial dim
    ///
    /// Returns visual features [N_visual, out_hidden_size].
    pub fn forward(&mut self, pixels: &Array, h_patches: i32, w_patches: i32) -> Result<Array> {
        // Patch embedding: [N, hidden_size]
        let mut hidden = self.patch_embed.forward(pixels)?;

        // Add positional embeddings
        let n = h_patches * w_patches;
        let pos_ids: Vec<i32> = (0..n).collect();
        let pos_ids_arr = Array::from_slice(&pos_ids, &[n]);
        let pos_emb = self.pos_embed.forward(&pos_ids_arr)?;
        hidden = hidden.add(&pos_emb)?;

        // Run blocks with DeepStack collection
        let deepstack_indexes = &self.config.deepstack_visual_indexes.clone();
        let spatial_merge = self.config.spatial_merge_size;
        let mut deepstack_idx = 0usize;
        let mut deepstack_outputs: Vec<Array> = Vec::new();

        for (i, block) in self.blocks.iter_mut().enumerate() {
            hidden = block.forward(&hidden)?;
            if deepstack_idx < deepstack_indexes.len() && i == deepstack_indexes[deepstack_idx] {
                let ds_out = self.deepstack_mergers[deepstack_idx].forward(&hidden, spatial_merge)?;
                deepstack_outputs.push(ds_out);
                deepstack_idx += 1;
            }
        }

        // Final merger
        let final_out = self.merger.forward(&hidden, spatial_merge)?;

        // Sum deepstack outputs with final
        let mut result = final_out;
        for ds in deepstack_outputs {
            result = result.add(&ds)?;
        }

        Ok(result)
    }
}

// ============================================================================
// Language Model (Qwen3 decoder)
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
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: nn::RmsNorm,
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
            .reshape(&[B, L, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        queries = self.q_norm.forward(&queries)?;

        let mut keys = self.k_proj.forward(x)?
            .reshape(&[B, L, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        keys = self.k_norm.forward(&keys)?;

        let values = self.v_proj.forward(x)?
            .reshape(&[B, L, self.n_kv_heads, self.head_dim])?
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
            queries, keys, values, Some(cache), self.scale, sdpa_mask,
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
        self.q_norm.training_mode(mode);
        self.k_norm.training_mode(mode);
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
            x: &normed, mask, cache,
        })?;
        let h = x.add(&attn_out)?;
        let normed2 = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed2)?;
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
// Qwen3-VL Model
// ============================================================================

pub struct Qwen3VL {
    pub vision: VisionTower,
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    pub layers: Vec<LLMBlock>,
    pub norm: nn::RmsNorm,
    pub config: Qwen3VLConfig,
}

impl Qwen3VL {
    /// Encode image and return visual feature tokens.
    pub fn encode_image(&mut self, image: &Array, h_patches: i32, w_patches: i32) -> Result<Array> {
        self.vision.forward(image, h_patches, w_patches)
    }

    /// Full forward pass for the language model portion given embeddings.
    fn lm_forward<C: KeyValueCache + Default>(
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
                x: &h, mask: None, cache: c,
            })?;
        }
        h = self.norm.forward(&h)?;

        // LM head: use embed_tokens as linear (tied weights)
        let logits = match &mut self.embed_tokens {
            MaybeQuantized::Original(emb) => emb.as_linear(&h)?,
            MaybeQuantized::Quantized(qemb) => qemb.as_linear(&h)?,
        };
        Ok(logits)
    }

    /// Prefill: encode image + text prompt, return logits for next token.
    pub fn prefill<C: KeyValueCache + Default>(
        &mut self,
        input_ids: &[i32],
        visual_features: &Array, // [N_visual, hidden_size]
        cache: &mut Vec<C>,
    ) -> Result<Array> {
        let image_token_id = self.config.image_token_id;

        // Build embeddings: embed text tokens, replace image placeholders with visual features
        let text_ids: Vec<i32> = input_ids
            .iter()
            .filter(|&&t| t != image_token_id)
            .cloned()
            .collect();

        let text_ids_arr = Array::from_slice(&text_ids, &[1, text_ids.len() as i32]);
        let text_embeds = match &mut self.embed_tokens {
            MaybeQuantized::Original(emb) => emb.forward(&text_ids_arr)?,
            MaybeQuantized::Quantized(qemb) => qemb.forward(&text_ids_arr)?,
        };
        // text_embeds: [1, text_len, hidden_size]

        // Find image token position and splice in visual features
        let mut result_embeds: Vec<Array> = Vec::new();
        let mut text_embed_idx = 0i32;
        let vs = visual_features.shape();
        let n_visual = vs[0];
        let hidden_size = vs[1];

        for (seq_idx, &token) in input_ids.iter().enumerate() {
            if token == image_token_id {
                // Insert visual features (all at once when we hit the first image token)
                // Count consecutive image tokens to know how many there should be
                let end = input_ids[seq_idx..]
                    .iter()
                    .take_while(|&&t| t == image_token_id)
                    .count();
                if end == n_visual as usize {
                    result_embeds.push(visual_features.clone());
                    // Skip ahead - handled by continue below
                }
                break; // Will re-do via positional approach
            }
        }

        // Simpler approach: find the image block and reconstruct
        result_embeds.clear();
        let mut in_image_block = false;
        let mut image_inserted = false;
        let mut text_so_far: Vec<i32> = Vec::new();
        let mut after_image: Vec<i32> = Vec::new();
        for &token in input_ids {
            if token == image_token_id && !image_inserted {
                in_image_block = true;
            } else if in_image_block && token != image_token_id {
                in_image_block = false;
                image_inserted = true;
                after_image.push(token);
            } else if !image_inserted && !in_image_block {
                text_so_far.push(token);
            } else if image_inserted {
                after_image.push(token);
            }
        }

        // Build combined embeddings
        if image_inserted || in_image_block {
            // [pre_image_tokens] [visual_features] [after_image_tokens]
            let mut parts: Vec<Array> = Vec::new();

            if !text_so_far.is_empty() {
                let pre_arr = Array::from_slice(&text_so_far, &[1, text_so_far.len() as i32]);
                let pre_emb = match &mut self.embed_tokens {
                    MaybeQuantized::Original(emb) => emb.forward(&pre_arr)?,
                    MaybeQuantized::Quantized(qemb) => qemb.forward(&pre_arr)?,
                };
                parts.push(pre_emb);
            }

            // Add visual features: [N_visual, hidden] → [1, N_visual, hidden]
            let vis = visual_features.reshape(&[1, n_visual, hidden_size])?;
            parts.push(vis);

            if !after_image.is_empty() {
                let post_arr = Array::from_slice(&after_image, &[1, after_image.len() as i32]);
                let post_emb = match &mut self.embed_tokens {
                    MaybeQuantized::Original(emb) => emb.forward(&post_arr)?,
                    MaybeQuantized::Quantized(qemb) => qemb.forward(&post_arr)?,
                };
                parts.push(post_emb);
            }

            let combined = if parts.len() == 1 {
                parts.remove(0)
            } else {
                let refs: Vec<&Array> = parts.iter().collect();
                mlx_rs::ops::concatenate_axis(&refs, 1)?
            };

            self.lm_forward(&combined, cache)
        } else {
            // No image tokens - pure text
            let ids_arr = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            let embeds = match &mut self.embed_tokens {
                MaybeQuantized::Original(emb) => emb.forward(&ids_arr)?,
                MaybeQuantized::Quantized(qemb) => qemb.forward(&ids_arr)?,
            };
            self.lm_forward(&embeds, cache)
        }
    }

    /// Decode a single token (used after prefill).
    pub fn decode_token<C: KeyValueCache + Default>(
        &mut self,
        token: i32,
        cache: &mut Vec<C>,
    ) -> Result<Array> {
        let token_arr = Array::from_slice(&[token], &[1, 1]);
        let embed = match &mut self.embed_tokens {
            MaybeQuantized::Original(emb) => emb.forward(&token_arr)?,
            MaybeQuantized::Quantized(qemb) => qemb.forward(&token_arr)?,
        };
        self.lm_forward(&embed, cache)
    }
}

// ============================================================================
// Image preprocessing
// ============================================================================

/// Preprocess an image for Qwen3-VL inference.
///
/// Returns `(pixel_array, h_patches, w_patches)` where:
/// - `pixel_array`: [1, temporal_patch_size, H, W, 3] float32 in [0,1]
/// - `h_patches`: H / patch_size
/// - `w_patches`: W / patch_size
pub fn preprocess_image(
    img: &image::DynamicImage,
    patch_size: i32,
    temporal_patch_size: i32,
) -> Result<(Array, i32, i32)> {
    // Resize to a resolution that's divisible by patch_size
    let target = 448i32; // 448 / 16 = 28 patches per side
    let img = img.resize_exact(target as u32, target as u32, image::imageops::FilterType::CatmullRom);
    let rgb = img.to_rgb8();

    let h = target;
    let w = target;
    let h_patches = h / patch_size;
    let w_patches = w / patch_size;

    // Normalize [0, 255] → [0, 1]
    let pixels: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| p.0.iter().map(|&v| v as f32 / 255.0))
        .collect();

    // Build [1, T, H, W, 3] where T = temporal_patch_size (repeat frame)
    // Single image: duplicate the frame temporal_patch_size times
    let frame_pixels = pixels.len(); // H * W * 3
    let mut temporal_pixels = Vec::with_capacity(temporal_patch_size as usize * frame_pixels);
    for _ in 0..temporal_patch_size {
        temporal_pixels.extend_from_slice(&pixels);
    }

    let arr = Array::from_slice(
        &temporal_pixels,
        &[1, temporal_patch_size, h, w, 3],
    );

    Ok((arr, h_patches, w_patches))
}

// ============================================================================
// Sampling
// ============================================================================

pub fn sample(logits: &Array, temp: f32) -> std::result::Result<Array, mlx_rs::error::Exception> {
    if temp == 0.0 {
        mlx_rs::argmax_axis!(logits, -1)
    } else {
        let scaled = logits.multiply(array!(1.0 / temp))?;
        mlx_rs::categorical!(scaled)
    }
}

// ============================================================================
// Model loading
// ============================================================================

fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>> {
    // Try single file first
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Array::load_safetensors(&single).map_err(Error::MlxIo);
    }

    // Sharded safetensors
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
            let loaded = Array::load_safetensors(&path).map_err(Error::MlxIo)?;
            all.extend(loaded);
        }
        return Ok(all);
    }

    Err(Error::InvalidConfig("No model.safetensors found".to_string()))
}

fn load_vision_layer_norm(
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<VisionLayerNorm> {
    Ok(VisionLayerNorm {
        weight: get_weight(weights, &format!("{}.weight", prefix))?,
        bias: get_weight(weights, &format!("{}.bias", prefix))?,
        eps: 1e-6,
    })
}

fn load_vision_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<nn::Linear> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let bias = get_weight(weights, &format!("{}.bias", prefix))?;
    Ok(nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(Some(bias)),
    })
}

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Qwen3VL> {
    let model_dir = model_dir.as_ref();

    // Load config
    let config_path = model_dir.join("config.json");
    let config: Qwen3VLConfig = if config_path.exists() {
        let json = std::fs::read_to_string(&config_path)?;
        serde_json::from_str(&json)?
    } else {
        Qwen3VLConfig::default()
    };

    let vis_cfg = &config.vision_config;
    let txt_cfg = &config.text_config;

    // Quantization config from config.json
    let (group_size, bits) = {
        let json = std::fs::read_to_string(&config_path)?;
        let v: serde_json::Value = serde_json::from_str(&json)?;
        let gs = v["quantization"]["group_size"].as_i64().unwrap_or(64) as i32;
        let b = v["quantization"]["bits"].as_i64().unwrap_or(8) as i32;
        (gs, b)
    };

    eprintln!("Loading Qwen3-VL weights (8-bit quantized)...");
    let weights = load_all_weights(model_dir)?;

    // ---- Vision Tower ----
    eprintln!("  Loading vision tower...");
    let vp = "vision_tower";

    // PatchEmbed Conv3d
    let conv_weight = get_weight(&weights, &format!("{}.patch_embed.proj.weight", vp))?;
    let conv_bias = get_weight(&weights, &format!("{}.patch_embed.proj.bias", vp))?;
    // Weight shape: [out_ch, T, H, W, in_ch] = [1024, 2, 16, 16, 3]
    let patch_embed = PatchEmbed {
        proj: nn::Conv3d {
            weight: Param::new(conv_weight),
            bias: Param::new(Some(conv_bias)),
            stride: (
                vis_cfg.temporal_patch_size,
                vis_cfg.patch_size,
                vis_cfg.patch_size,
            ),
            padding: (0, 0, 0),
            dilation: (1, 1, 1),
            groups: 1,
        },
    };

    // Positional embedding
    let pos_embed_weight = get_weight(&weights, &format!("{}.pos_embed.weight", vp))?;
    let pos_embed = nn::Embedding {
        weight: Param::new(pos_embed_weight),
    };

    // VisionBlocks
    let head_dim = vis_cfg.hidden_size / vis_cfg.num_heads;
    let mut blocks = Vec::with_capacity(vis_cfg.depth);
    for i in 0..vis_cfg.depth {
        let bp = format!("{}.blocks.{}", vp, i);

        let qkv_weight = get_weight(&weights, &format!("{}.attn.qkv.weight", bp))?;
        let qkv_bias = get_weight(&weights, &format!("{}.attn.qkv.bias", bp))?;
        let proj_weight = get_weight(&weights, &format!("{}.attn.proj.weight", bp))?;
        let proj_bias = get_weight(&weights, &format!("{}.attn.proj.bias", bp))?;

        let attn = VisionAttention {
            num_heads: vis_cfg.num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            qkv: nn::Linear {
                weight: Param::new(qkv_weight),
                bias: Param::new(Some(qkv_bias)),
            },
            proj: nn::Linear {
                weight: Param::new(proj_weight),
                bias: Param::new(Some(proj_bias)),
            },
        };

        let mlp = VisionMlp {
            linear_fc1: load_vision_linear(&weights, &format!("{}.mlp.linear_fc1", bp))?,
            linear_fc2: load_vision_linear(&weights, &format!("{}.mlp.linear_fc2", bp))?,
        };

        blocks.push(VisionBlock {
            norm1: load_vision_layer_norm(&weights, &format!("{}.norm1", bp))?,
            norm2: load_vision_layer_norm(&weights, &format!("{}.norm2", bp))?,
            attn,
            mlp,
        });
    }

    // DeepStack mergers
    let mut deepstack_mergers = Vec::with_capacity(vis_cfg.deepstack_visual_indexes.len());
    for i in 0..vis_cfg.deepstack_visual_indexes.len() {
        let dp = format!("{}.deepstack_merger_list.{}", vp, i);
        deepstack_mergers.push(DeepStackMerger {
            norm: load_vision_layer_norm(&weights, &format!("{}.norm", dp))?,
            linear_fc1: load_vision_linear(&weights, &format!("{}.linear_fc1", dp))?,
            linear_fc2: load_vision_linear(&weights, &format!("{}.linear_fc2", dp))?,
        });
    }

    // Final merger
    let mp = format!("{}.merger", vp);
    let merger = FinalMerger {
        norm: load_vision_layer_norm(&weights, &format!("{}.norm", mp))?,
        linear_fc1: load_vision_linear(&weights, &format!("{}.linear_fc1", mp))?,
        linear_fc2: load_vision_linear(&weights, &format!("{}.linear_fc2", mp))?,
    };

    let vision = VisionTower {
        patch_embed,
        pos_embed,
        blocks,
        deepstack_mergers,
        merger,
        config: vis_cfg.clone(),
    };

    // ---- Language Model ----
    eprintln!(
        "  Loading Qwen3 decoder ({} layers, 8-bit quantized)...",
        txt_cfg.num_hidden_layers
    );
    let lm = "language_model.model";

    let embed_tokens = make_quantized_embedding(
        &weights,
        &format!("{}.embed_tokens", lm),
        group_size,
        bits,
    )?;

    let n_heads = txt_cfg.num_attention_heads;
    let n_kv_heads = txt_cfg.num_key_value_heads;
    let head_dim_lm = txt_cfg.head_dim;
    let rope_theta = txt_cfg.rope_theta as f32;

    let mut layers = Vec::with_capacity(txt_cfg.num_hidden_layers as usize);
    for i in 0..txt_cfg.num_hidden_layers {
        let lp = format!("{}.layers.{}", lm, i);

        let attention = LLMAttention {
            n_heads,
            n_kv_heads,
            head_dim: head_dim_lm,
            scale: (head_dim_lm as f32).powf(-0.5),
            q_proj: make_quantized_linear(
                &weights, &format!("{}.self_attn.q_proj", lp), group_size, bits,
            )?,
            k_proj: make_quantized_linear(
                &weights, &format!("{}.self_attn.k_proj", lp), group_size, bits,
            )?,
            v_proj: make_quantized_linear(
                &weights, &format!("{}.self_attn.v_proj", lp), group_size, bits,
            )?,
            o_proj: make_quantized_linear(
                &weights, &format!("{}.self_attn.o_proj", lp), group_size, bits,
            )?,
            q_norm: nn::RmsNorm {
                weight: Param::new(get_weight(
                    &weights,
                    &format!("{}.self_attn.q_norm.weight", lp),
                )?),
                eps: txt_cfg.rms_norm_eps,
            },
            k_norm: nn::RmsNorm {
                weight: Param::new(get_weight(
                    &weights,
                    &format!("{}.self_attn.k_norm.weight", lp),
                )?),
                eps: txt_cfg.rms_norm_eps,
            },
            rope: nn::RopeBuilder::new(head_dim_lm)
                .traditional(false)
                .base(rope_theta)
                .build()
                .unwrap(),
        };

        let mlp = LLMFeedForward {
            gate_proj: make_quantized_linear(
                &weights, &format!("{}.mlp.gate_proj", lp), group_size, bits,
            )?,
            up_proj: make_quantized_linear(
                &weights, &format!("{}.mlp.up_proj", lp), group_size, bits,
            )?,
            down_proj: make_quantized_linear(
                &weights, &format!("{}.mlp.down_proj", lp), group_size, bits,
            )?,
        };

        layers.push(LLMBlock {
            self_attn: attention,
            mlp,
            input_layernorm: nn::RmsNorm {
                weight: Param::new(get_weight(
                    &weights,
                    &format!("{}.input_layernorm.weight", lp),
                )?),
                eps: txt_cfg.rms_norm_eps,
            },
            post_attention_layernorm: nn::RmsNorm {
                weight: Param::new(get_weight(
                    &weights,
                    &format!("{}.post_attention_layernorm.weight", lp),
                )?),
                eps: txt_cfg.rms_norm_eps,
            },
        });
    }

    let norm = nn::RmsNorm {
        weight: Param::new(get_weight(&weights, &format!("{}.norm.weight", lm))?),
        eps: txt_cfg.rms_norm_eps,
    };

    eprintln!("Qwen3-VL loaded successfully.");

    Ok(Qwen3VL {
        vision,
        embed_tokens,
        layers,
        norm,
        config,
    })
}

// ============================================================================
// High-level generate function
// ============================================================================

/// Build input token IDs for Qwen3-VL vision chat.
///
/// Format:
/// ```text
/// <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
/// <|im_start|>user\n<|vision_start|><|image_pad|>...<|vision_end|>\n{prompt}<|im_end|>\n
/// <|im_start|>assistant\n
/// ```
pub fn build_chat_tokens(
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    n_image_tokens: usize,
    config: &Qwen3VLConfig,
) -> Result<Vec<i32>> {
    // Special token IDs for Qwen VL
    let im_start = 151644i32;  // <|im_start|>
    let im_end = 151645i32;    // <|im_end|>
    let newline = 198i32;      // \n
    let image_token_id = config.image_token_id;
    let vision_start = config.vision_start_token_id;
    let vision_end = config.vision_end_token_id;

    // Encode text parts (skip special tokens)
    let encode = |text: &str| -> Result<Vec<i32>> {
        let enc = tokenizer.encode(text, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
    };

    let system_text = encode("system")?;
    let system_content = encode("You are a helpful assistant.")?;
    let user_text = encode("user")?;
    let prompt_tokens = encode(prompt)?;
    let assistant_text = encode("assistant")?;

    let mut tokens: Vec<i32> = Vec::new();

    // System turn
    tokens.push(im_start);
    tokens.extend_from_slice(&system_text);
    tokens.push(newline);
    tokens.extend_from_slice(&system_content);
    tokens.push(im_end);
    tokens.push(newline);

    // User turn
    tokens.push(im_start);
    tokens.extend_from_slice(&user_text);
    tokens.push(newline);
    // Vision segment
    tokens.push(vision_start);
    for _ in 0..n_image_tokens {
        tokens.push(image_token_id);
    }
    tokens.push(vision_end);
    tokens.push(newline);
    // User text
    tokens.extend_from_slice(&prompt_tokens);
    tokens.push(im_end);
    tokens.push(newline);

    // Assistant start
    tokens.push(im_start);
    tokens.extend_from_slice(&assistant_text);
    tokens.push(newline);

    Ok(tokens)
}

/// Run Qwen3-VL inference: image + prompt → generated text.
///
/// `tokenizer_json` is the raw bytes of `tokenizer.json` — passed as bytes to
/// avoid cross-crate `tokenizers` version conflicts.
pub fn generate(
    model: &mut Qwen3VL,
    image_bytes: &[u8],
    prompt: &str,
    tokenizer_json: &[u8],
    max_tokens: usize,
    temperature: f32,
) -> Result<String> {
    let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_json)
        .map_err(|e| Error::Tokenizer(e.to_string()))?;

    let img = image::load_from_memory(image_bytes)?;

    let patch_size = model.config.vision_config.patch_size;
    let temporal_patch_size = model.config.vision_config.temporal_patch_size;
    let spatial_merge = model.config.vision_config.spatial_merge_size;

    let (pixels, h_patches, w_patches) = preprocess_image(&img, patch_size, temporal_patch_size)?;
    let n_visual_tokens = (h_patches * w_patches / (spatial_merge * spatial_merge)) as usize;

    // Encode image
    let visual_features = model.encode_image(&pixels, h_patches, w_patches)?;
    mlx_rs::transforms::eval([&visual_features])?;

    // Build input tokens
    let input_ids = build_chat_tokens(&tokenizer, prompt, n_visual_tokens, &model.config)?;

    // EOS token
    let eos_token_id = 151645i32; // <|im_end|>

    // Prefill
    let mut cache: Vec<ConcatKeyValueCache> = Vec::new();
    let logits = model.prefill(&input_ids, &visual_features, &mut cache)?;

    // Sample first token
    let last_logits = logits.index((.., -1, ..));
    let mut token = sample(&last_logits, temperature)?;
    mlx_rs::transforms::eval([&token])?;

    let mut generated_ids: Vec<u32> = Vec::new();

    for _ in 0..max_tokens {
        let token_id = token.item::<i32>();
        if token_id == eos_token_id || token_id == 151643 {
            break;
        }
        generated_ids.push(token_id as u32);

        let logits = model.decode_token(token_id, &mut cache)?;
        let next_logits = logits.index((.., -1, ..));
        token = sample(&next_logits, temperature)?;
        mlx_rs::transforms::eval([&token])?;
    }

    let text = tokenizer
        .decode(&generated_ids, true)
        .map_err(|e| Error::Tokenizer(e.to_string()))?;

    Ok(text)
}
