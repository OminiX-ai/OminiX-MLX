//! # deepseek-ocr2-mlx
//!
//! DeepSeek-OCR-2 Vision-Language Model inference on Apple Silicon with MLX.
//!
//! ## Architecture
//!
//! ```text
//! Image (1024x1024 global + 768x768 crops)
//!   |-> SAM ViT-B/16 (12 blocks, 768-dim, rel pos, window attn)
//!   |     -> Neck (768->256->512->896)
//!   |-> Qwen2 Decoder-as-Encoder (24 layers, 896-dim, mixed attention)
//!   |     -> [B, num_queries, 896]
//!   |-> Linear Projector (896 -> 1280)
//!   |-> view_seperator (learned 1280-dim)
//!   |
//!   BOS + text + [visual tokens] + text
//!   |-> DeepSeek-V2 LLM (12 layers, 1280-dim, MoE)
//!   |     Layer 0: dense MLP (intermediate=6848)
//!   |     Layers 1-11: MoE (64 experts, top-6, 2 shared)
//!   |-> lm_head -> logits
//! ```

pub mod error;
pub mod qwen2_encoder;
pub mod vision;

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp},
    Array, Dtype,
};
use serde::Deserialize;

use error::{Error, Result};
use qwen2_encoder::Qwen2Encoder;
use vision::ImageEncoderViT;

pub use mlx_rs_core::{
    cache::{ConcatKeyValueCache, KVCache, KeyValueCache},
    fused_swiglu,
    utils::SdpaMask,
};

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct DeepseekOCR2Config {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: i32,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub tie_word_embeddings: bool,

    // MoE config
    #[serde(default = "default_moe_intermediate_size")]
    pub moe_intermediate_size: i32,
    #[serde(default = "default_n_routed_experts")]
    pub n_routed_experts: i32,
    #[serde(default = "default_n_shared_experts")]
    pub n_shared_experts: i32,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: i32,
    #[serde(default = "default_first_k_dense_replace")]
    pub first_k_dense_replace: i32,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
    #[serde(default = "default_n_group")]
    pub n_group: i32,
    #[serde(default = "default_topk_group")]
    pub topk_group: i32,
    #[serde(default = "default_topk_method")]
    pub topk_method: String,

    #[serde(default)]
    pub use_mla: bool,
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: i32,
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: i32,
}

fn default_hidden_size() -> i32 { 1280 }
fn default_num_hidden_layers() -> i32 { 12 }
fn default_num_attention_heads() -> i32 { 10 }
fn default_num_key_value_heads() -> i32 { 10 }
fn default_intermediate_size() -> i32 { 6848 }
fn default_vocab_size() -> i32 { 129280 }
fn default_rms_norm_eps() -> f32 { 1e-6 }
fn default_rope_theta() -> f32 { 10000.0 }
fn default_max_position_embeddings() -> i32 { 8192 }
fn default_moe_intermediate_size() -> i32 { 896 }
fn default_n_routed_experts() -> i32 { 64 }
fn default_n_shared_experts() -> i32 { 2 }
fn default_num_experts_per_tok() -> i32 { 6 }
fn default_first_k_dense_replace() -> i32 { 1 }
fn default_routed_scaling_factor() -> f32 { 1.0 }
fn default_n_group() -> i32 { 1 }
fn default_topk_group() -> i32 { 1 }
fn default_topk_method() -> String { "greedy".to_string() }
fn default_bos_token_id() -> i32 { 0 }
fn default_eos_token_id() -> i32 { 1 }

// ============================================================================
// LLM Components
// ============================================================================

/// Standard MLP with SwiGLU activation
#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct MLP {
    #[param]
    pub gate_proj: nn::Linear,
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
}

impl Module<&Array> for MLP {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = fused_swiglu(&up, &gate)?;
        self.down_proj.forward(&activated)
    }
}

/// MoE Gate for expert routing (softmax scoring, greedy top-k)
#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct MoEGate {
    pub top_k: i32,
    pub n_routed_experts: i32,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,

    #[param]
    pub weight: Param<Array>,
}

impl MoEGate {
    /// Route tokens to top-k experts.
    /// Returns (expert_indices [B, L, k], expert_weights [B, L, k])
    pub fn route(&self, x: &Array) -> std::result::Result<(Array, Array), Exception> {
        // x: [B, L, D] -> logits: [B, L, n_experts]
        let logits = x.as_dtype(Dtype::Float32)?.matmul(&(*self.weight).as_dtype(Dtype::Float32)?.t())?;

        // Softmax scoring
        let scores = ops::softmax_axis(&logits, -1, true)?;

        // Greedy top-k selection
        let neg_scores = scores.negative()?;
        let partitioned_inds = ops::argpartition_axis(&neg_scores, self.top_k - 1, -1)?;
        let inds = partitioned_inds.index((.., .., ..self.top_k));
        let selected_scores = ops::indexing::take_along_axis(&scores, &inds, -1)?;

        // Normalize and scale
        let final_scores = if self.norm_topk_prob && self.top_k > 1 {
            let denom = selected_scores.sum_axis(-1, true)?.add(array!(1e-20f32))?;
            selected_scores.divide(&denom)?.multiply(array!(self.routed_scaling_factor))?
        } else {
            selected_scores.multiply(array!(self.routed_scaling_factor))?
        };

        Ok((inds, final_scores))
    }
}

/// Mixture of Experts block
#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct MoE {
    pub num_experts_per_tok: i32,
    pub n_routed_experts: i32,

    #[param]
    pub gate: MoEGate,
    #[param]
    pub experts: Vec<MLP>,
    #[param]
    pub shared_experts: Option<MLP>,
}

impl Module<&Array> for MoE {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Exception> {
        let identity = x.clone();
        let (topk_idx, topk_weight) = self.gate.route(x)?;

        // Evaluate topk_idx and topk_weight to get concrete values
        mlx_rs::transforms::eval([&topk_idx, &topk_weight].into_iter())?;

        let shape = x.shape();
        let b = shape[0] as i32;
        let l = shape[1] as i32;
        let d = shape[2] as i32;
        let x_flat = x.reshape(&[-1, d])?;

        // MoE inference: sort tokens by expert, process, unsort
        let flat_topk_idx = topk_idx.reshape(&[-1])?;
        let n_tokens = b * l;

        // Sort by expert index for coalesced access
        let sort_order = ops::argsort(&flat_topk_idx)?;
        let sorted_idx = ops::indexing::take_axis(&flat_topk_idx, &sort_order, 0)?;

        // Map each expert slot to its token: slot i -> token sort_order[i] / top_k
        let token_indices = sort_order.floor_divide(array!(self.num_experts_per_tok))?;
        let sorted_tokens = ops::indexing::take_axis(&x_flat, &token_indices, 0)?;

        // Process each expert's tokens - cast to i32 since argpartition returns uint32
        let sorted_idx = sorted_idx.as_dtype(Dtype::Int32)?;
        mlx_rs::transforms::eval([&sorted_idx].into_iter())?;
        let sorted_idx_data: Vec<i32> = sorted_idx.as_slice().to_vec();
        let total_slots = sorted_idx_data.len();

        let mut outputs = Vec::new();
        let mut start = 0;
        while start < total_slots {
            let expert_id = sorted_idx_data[start];
            let mut end = start + 1;
            while end < total_slots && sorted_idx_data[end] == expert_id {
                end += 1;
            }

            let expert_tokens = sorted_tokens.index(start as i32..end as i32);
            let expert_out = self.experts[expert_id as usize].forward(&expert_tokens)?;
            outputs.push(expert_out);

            start = end;
        }

        // Concatenate and unsort
        let all_outputs = if outputs.is_empty() {
            ops::zeros_like(&sorted_tokens)?
        } else {
            ops::concatenate_axis(
                &outputs.iter().collect::<Vec<_>>(),
                0,
            )?
        };

        // Unsort back to original order
        let inv_order = ops::argsort(&sort_order)?;
        let unsorted = ops::indexing::take_axis(&all_outputs, &inv_order, 0)?;

        // Weight by routing scores: [n_tokens * top_k, D] -> [n_tokens, top_k, D]
        let weighted = unsorted.reshape(&[n_tokens, self.num_experts_per_tok, d])?;
        let weights = topk_weight.reshape(&[n_tokens, self.num_experts_per_tok, 1])?;
        let y = weighted.multiply(&weights)?.sum_axis(1, false)?.as_dtype(x.dtype())?;
        let y = y.reshape(&[b, l, d])?;

        // Add shared experts
        if let Some(ref mut shared) = self.shared_experts {
            let shared_out = shared.forward(&identity)?;
            return y.add(&shared_out);
        }

        Ok(y)
    }
}

/// LLM Attention (standard MHA with RoPE)
#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct LLMAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

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
}

pub struct LLMAttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut C,
}

impl<C> Module<LLMAttentionInput<'_, C>> for LLMAttention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;
    fn training_mode(&mut self, _: bool) {}

    fn forward(&mut self, input: LLMAttentionInput<'_, C>) -> std::result::Result<Self::Output, Self::Error> {
        let LLMAttentionInput { x, mask, cache } = input;

        let shape = x.shape();
        let b = shape[0] as i32;
        let l = shape[1] as i32;

        let q = self.q_proj.forward(x)?
            .reshape(&[b, l, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self.k_proj.forward(x)?
            .reshape(&[b, l, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self.v_proj.forward(x)?
            .reshape(&[b, l, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE with cache offset
        let q_input = nn::RopeInputBuilder::new(&q)
            .offset(cache.offset())
            .build()?;
        let q = self.rope.forward(q_input)?;
        let k_input = nn::RopeInputBuilder::new(&k)
            .offset(cache.offset())
            .build()?;
        let k = self.rope.forward(k_input)?;

        // Update KV cache
        let (k, v) = cache.update_and_fetch(k, v)?;

        // Determine mask
        let sdpa_mask = match mask {
            Some(m) => Some(SdpaMask::Array(m)),
            None if l > 1 => Some(SdpaMask::Causal),
            None => None,
        };

        let output = mlx_rs_core::utils::scaled_dot_product_attention(
            q, k, v, Some(cache), self.scale, sdpa_mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, l, -1])?;

        self.o_proj.forward(&output)
    }
}

/// Decoder layer (dense or MoE)
#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct DecoderLayer {
    pub is_moe: bool,

    #[param]
    pub self_attn: LLMAttention,
    #[param]
    pub mlp: Option<MLP>,
    #[param]
    pub moe: Option<MoE>,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl<C> Module<LLMAttentionInput<'_, C>> for DecoderLayer
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;
    fn training_mode(&mut self, _: bool) {}

    fn forward(&mut self, input: LLMAttentionInput<'_, C>) -> std::result::Result<Self::Output, Self::Error> {
        let LLMAttentionInput { x, mask, cache } = input;

        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(LLMAttentionInput {
            x: &normed,
            mask,
            cache,
        })?;
        let h = x.add(&attn_out)?;

        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = if self.is_moe {
            self.moe.as_mut().unwrap().forward(&normed)?
        } else {
            self.mlp.as_mut().unwrap().forward(&normed)?
        };

        h.add(&mlp_out)
    }
}

// ============================================================================
// Full Model
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct DeepseekOCR2 {
    pub config: DeepseekOCR2Config,

    // Vision components
    #[param]
    pub sam_model: ImageEncoderViT,
    #[param]
    pub qwen2_model: Qwen2Encoder,
    #[param]
    pub projector: nn::Linear,
    #[param]
    pub view_seperator: Param<Array>,

    // LLM components
    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
    #[param]
    pub lm_head: nn::Linear,
}

impl DeepseekOCR2 {
    /// Encode image patches through SAM + Qwen2 + Projector pipeline.
    /// Returns concatenated visual features with view separator.
    ///
    /// crop_images: [N, H, W, 3] crop patches (768x768), can be empty
    /// global_image: [1, H, W, 3] global view (1024x1024)
    pub fn encode_image(
        &mut self,
        crop_images: Option<&Array>,
        global_image: &Array,
    ) -> Result<Array> {
        // Encode global view
        let global_feat = self.sam_model.forward(global_image)?;
        let global_feat = self.qwen2_model.forward_vision(&global_feat)?;
        let global_feat = self.projector.forward(&global_feat)?;

        let d = global_feat.shape()[2] as i32;
        let global_flat = global_feat.reshape(&[-1, d])?;

        // Encode crop patches if present
        let features = if let Some(crops) = crop_images {
            let crop_feat = self.sam_model.forward(crops)?;
            let crop_feat = self.qwen2_model.forward_vision(&crop_feat)?;
            let crop_feat = self.projector.forward(&crop_feat)?;
            let crop_flat = crop_feat.reshape(&[-1, d])?;

            // [local | global | view_separator]
            let sep = (*self.view_seperator).reshape(&[1, d])?;
            ops::concatenate_axis(&[&crop_flat, &global_flat, &sep], 0)?
        } else {
            // [global | view_separator]
            let sep = (*self.view_seperator).reshape(&[1, d])?;
            ops::concatenate_axis(&[&global_flat, &sep], 0)?
        };

        Ok(features)
    }

    /// Run the LLM forward pass.
    /// input_ids: [B, L], visual_features already embedded in inputs_embeds
    pub fn forward_llm<C: KeyValueCache>(
        &mut self,
        inputs_embeds: &Array,
        cache: &mut Vec<C>,
    ) -> Result<Array> {
        let mut h = inputs_embeds.clone();

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(LLMAttentionInput {
                x: &h,
                mask: None,
                cache: c,
            })?;
        }

        let h = self.norm.forward(&h)?;
        let logits = self.lm_head.forward(&h)?;
        Ok(logits)
    }

    /// Prepare inputs with visual tokens scattered into the right positions.
    pub fn prepare_inputs(
        &mut self,
        input_ids: &Array,
        images_seq_mask: &Array,
        visual_features: &Array,
    ) -> Result<Array> {
        // Get text embeddings
        let inputs_embeds = self.embed_tokens.forward(input_ids)?;

        // Scatter visual features into image token positions
        // images_seq_mask: [B, L] bool mask
        // visual_features: [N, D] where N = number of True positions
        let shape = inputs_embeds.shape();
        let b = shape[0] as i32;
        let l = shape[1] as i32;
        let d = shape[2] as i32;

        // Create visual embed tensor of same shape, with visual features at masked positions
        // For simplicity with batch_size=1:
        let embeds_flat = inputs_embeds.reshape(&[-1, d])?;
        let mask_flat = images_seq_mask.reshape(&[-1])?;
        mlx_rs::transforms::eval([&mask_flat].into_iter())?;

        let mask_data: Vec<bool> = mask_flat.as_slice().to_vec();
        let n_visual = mask_data.iter().filter(|&&x| x).count();

        if n_visual > 0 && n_visual == visual_features.shape()[0] as usize {
            // Build new embeddings by replacing masked positions
            let mut parts = Vec::new();
            let mut vis_idx = 0i32;

            let mut i = 0;
            while i < mask_data.len() {
                if mask_data[i] {
                    // Find contiguous run of True
                    let start = i;
                    while i < mask_data.len() && mask_data[i] {
                        i += 1;
                    }
                    let count = (i - start) as i32;
                    parts.push(visual_features.index(vis_idx..vis_idx + count));
                    vis_idx += count;
                } else {
                    // Find contiguous run of False
                    let start = i;
                    while i < mask_data.len() && !mask_data[i] {
                        i += 1;
                    }
                    let count = (i - start) as i32;
                    parts.push(embeds_flat.index(start as i32..start as i32 + count));
                }
            }

            let new_embeds = ops::concatenate_axis(
                &parts.iter().collect::<Vec<_>>(),
                0,
            )?;
            Ok(new_embeds.reshape(&[b, l, d])?)
        } else {
            Ok(inputs_embeds)
        }
    }

    /// Single token decode step (for autoregressive generation).
    pub fn decode_token<C: KeyValueCache>(
        &mut self,
        token: &Array,
        cache: &mut Vec<C>,
    ) -> Result<Array> {
        let embeds = self.embed_tokens.forward(token)?;
        self.forward_llm(&embeds, cache)
    }

    /// Initialize KV cache for all layers.
    pub fn init_cache(&self) -> Vec<KVCache> {
        (0..self.config.num_hidden_layers)
            .map(|_| KVCache::default())
            .collect()
    }
}

// ============================================================================
// Generation
// ============================================================================

pub enum GenerateState {
    Prefill { embeds: Array },
    Decode { next_token: Array },
    Done,
}

pub struct Generate<'a> {
    pub model: &'a mut DeepseekOCR2,
    pub cache: &'a mut Vec<KVCache>,
    pub temp: f32,
    pub state: GenerateState,
    pub eos_token_id: i32,
    pub repetition_penalty: f32,
    pub repetition_context_size: usize,
    pub generated_tokens: Vec<i32>,
}

impl Generate<'_> {
    fn apply_repetition_penalty(&self, logits: &Array) -> std::result::Result<Array, Exception> {
        if self.repetition_penalty == 1.0 || self.generated_tokens.is_empty() {
            return Ok(logits.clone());
        }

        // Use recent context window
        let tokens = if self.repetition_context_size > 0 && self.generated_tokens.len() > self.repetition_context_size {
            &self.generated_tokens[self.generated_tokens.len() - self.repetition_context_size..]
        } else {
            &self.generated_tokens
        };

        // Collect unique token ids
        let mut seen = std::collections::HashSet::new();
        let unique_ids: Vec<i32> = tokens.iter().copied().filter(|t| seen.insert(*t)).collect();

        if unique_ids.is_empty() {
            return Ok(logits.clone());
        }

        // Materialize logits to CPU for penalty application
        mlx_rs::transforms::eval(std::iter::once(logits))?;
        let logits_data: Vec<f32> = logits.as_slice().to_vec();
        let mut penalized = logits_data;

        let penalty = self.repetition_penalty;
        for &tid in &unique_ids {
            let idx = tid as usize;
            if idx < penalized.len() {
                if penalized[idx] > 0.0 {
                    penalized[idx] /= penalty;
                } else {
                    penalized[idx] *= penalty;
                }
            }
        }

        Ok(Array::from_slice(&penalized, logits.shape()))
    }

    fn sample_with_penalty(&mut self, logits: &Array) -> std::result::Result<Array, Exception> {
        let logits = self.apply_repetition_penalty(logits)?;
        let token = sample(&logits, self.temp)?;
        let val: i32 = token.item();
        self.generated_tokens.push(val);
        Ok(token)
    }

    /// Detect if generation is stuck in a repeating n-gram pattern.
    /// Returns true if any n-gram of size 4..=32 has repeated 3+ times consecutively.
    fn is_repeating(&self) -> bool {
        let tokens = &self.generated_tokens;
        let len = tokens.len();
        if len < 12 { return false; }

        // Check n-gram sizes from 4 to 32
        for n in 4..=32.min(len / 3) {
            let tail = &tokens[len - n..];
            let mut repeats = 1;
            let mut pos = len - n;
            while pos >= n {
                pos -= n;
                if &tokens[pos..pos + n] == tail {
                    repeats += 1;
                    if repeats >= 3 {
                        return true;
                    }
                } else {
                    break;
                }
            }
        }
        false
    }
}

impl Iterator for Generate<'_> {
    type Item = Result<Array>;

    fn next(&mut self) -> Option<Self::Item> {
        match std::mem::replace(&mut self.state, GenerateState::Done) {
            GenerateState::Prefill { embeds } => {
                let logits = match self.model.forward_llm(&embeds, self.cache) {
                    Ok(l) => l,
                    Err(e) => return Some(Err(e)),
                };
                let last_logits = logits.index((.., -1, ..));
                let token = match self.sample_with_penalty(&last_logits) {
                    Ok(t) => t,
                    Err(e) => return Some(Err(e.into())),
                };

                // Check EOS
                {
                    let val: i32 = token.item();
                    if val == self.eos_token_id {
                        self.state = GenerateState::Done;
                        return Some(Ok(token));
                    }
                }

                self.state = GenerateState::Decode {
                    next_token: token.clone(),
                };
                Some(Ok(token))
            }
            GenerateState::Decode { next_token } => {
                // Check for n-gram repetition before generating more
                if self.is_repeating() {
                    self.state = GenerateState::Done;
                    return None;
                }

                let token_input = match next_token.reshape(&[1, 1]) {
                    Ok(t) => t,
                    Err(e) => return Some(Err(e.into())),
                };
                let logits = match self.model.decode_token(&token_input, self.cache) {
                    Ok(l) => l,
                    Err(e) => return Some(Err(e)),
                };
                let last_logits = logits.index((.., -1, ..));
                let token = match self.sample_with_penalty(&last_logits) {
                    Ok(t) => t,
                    Err(e) => return Some(Err(e.into())),
                };

                // Check EOS
                {
                    let val: i32 = token.item();
                    if val == self.eos_token_id {
                        self.state = GenerateState::Done;
                        return Some(Ok(token));
                    }
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

pub fn sample(logits: &Array, temp: f32) -> std::result::Result<Array, Exception> {
    let token = if temp == 0.0 {
        mlx_rs::argmax_axis!(logits, -1)?
    } else {
        let scaled = logits.multiply(array!(1.0 / temp))?;
        mlx_rs::categorical!(&scaled)?
    };
    // argmax/categorical return uint32, cast to int32 for consistency
    token.as_dtype(Dtype::Int32)
}

// ============================================================================
// Image preprocessing
// ============================================================================

/// Image token ID used in the tokenizer
pub const IMAGE_TOKEN_ID: i32 = 128815;

/// Preprocess and tokenize a prompt with image.
/// Returns (input_ids, images_seq_mask) where image token positions are marked.
pub fn tokenize_prompt(
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    has_image: bool,
    base_size: i32,
    image_size: i32,
    crop_ratio: (i32, i32),
) -> Result<(Vec<i32>, Vec<bool>)> {
    let patch_size = 16;
    let downsample_ratio = 4;

    // Format conversation (matches Python's get_conv_template format)
    let formatted = format!(
        "<|User|>: {}\n\n<|Assistant|>:",
        prompt
    );

    let text_splits: Vec<&str> = formatted.split("<image>").collect();

    let mut token_ids: Vec<i32> = Vec::new();
    let mut seq_mask: Vec<bool> = Vec::new();

    if has_image && text_splits.len() > 1 {
        // Text before <image>
        let pre_tokens = tokenizer
            .encode(text_splits[0], false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        for &id in pre_tokens.get_ids() {
            token_ids.push(id as i32);
            seq_mask.push(false);
        }

        // Image tokens
        let num_queries_base = (base_size / patch_size / downsample_ratio) as i32;
        let num_queries = (image_size / patch_size / downsample_ratio) as i32;

        // Global view tokens
        for _ in 0..num_queries_base * num_queries_base {
            token_ids.push(IMAGE_TOKEN_ID);
            seq_mask.push(true);
        }
        // Separator
        token_ids.push(IMAGE_TOKEN_ID);
        seq_mask.push(true);

        // Crop tokens (if crops exist)
        let (w_crops, h_crops) = crop_ratio;
        if w_crops > 1 || h_crops > 1 {
            for _ in 0..num_queries * w_crops * num_queries * h_crops {
                token_ids.push(IMAGE_TOKEN_ID);
                seq_mask.push(true);
            }
        }

        // Text after <image>
        let remaining = text_splits[1..].join("<image>");
        let post_tokens = tokenizer
            .encode(remaining.as_str(), false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        for &id in post_tokens.get_ids() {
            token_ids.push(id as i32);
            seq_mask.push(false);
        }
    } else {
        // Pure text
        let tokens = tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        for &id in tokens.get_ids() {
            token_ids.push(id as i32);
            seq_mask.push(false);
        }
    }

    // Prepend BOS
    token_ids.insert(0, 0); // bos_token_id = 0
    seq_mask.insert(0, false);

    Ok((token_ids, seq_mask))
}

/// Find best aspect ratio for dynamic cropping.
pub fn find_best_crop_ratio(
    width: u32,
    height: u32,
    min_num: u32,
    max_num: u32,
) -> (i32, i32) {
    let aspect = width as f64 / height as f64;

    let mut best_ratio = (1i32, 1i32);
    let mut best_diff = f64::MAX;

    for n in min_num..=max_num {
        for i in 1..=n {
            for j in 1..=n {
                if i * j <= max_num && i * j >= min_num {
                    let target_aspect = i as f64 / j as f64;
                    let diff = (aspect - target_aspect).abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best_ratio = (i as i32, j as i32);
                    }
                }
            }
        }
    }

    best_ratio
}

// ============================================================================
// Weight loading
// ============================================================================

#[derive(Debug, Deserialize)]
struct WeightMap {
    weight_map: HashMap<String, String>,
}

fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>> {
    let index_path = model_dir.join("model.safetensors.index.json");

    if index_path.exists() {
        let json = std::fs::read_to_string(&index_path)?;
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
        let path = model_dir.join("model.safetensors");
        let loaded = Array::load_safetensors(&path)?;
        Ok(loaded)
    }
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::Model(format!("Weight not found: {}", key)))
}

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<tokenizers::Tokenizer> {
    let path = model_dir.as_ref().join("tokenizer.json");
    tokenizers::Tokenizer::from_file(path).map_err(|e| Error::Tokenizer(e.to_string()))
}

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<DeepseekOCR2> {
    let model_dir = model_dir.as_ref();

    // Load config
    let config_path = model_dir.join("config.json");
    let config: DeepseekOCR2Config =
        serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;

    // Load weights
    eprintln!("Loading weights...");
    let weights = load_all_weights(model_dir)?;
    eprintln!("Loaded {} weight tensors", weights.len());

    // Load SAM encoder
    eprintln!("Loading SAM encoder...");
    let sam_model = vision::load_sam_encoder(&weights, "model.sam_model")?;

    // Load Qwen2 encoder
    eprintln!("Loading Qwen2 encoder...");
    let qwen2_model = qwen2_encoder::load_qwen2_encoder(&weights, "model.qwen2_model")?;

    // Load projector (linear: 896 -> 1280)
    let projector = nn::Linear {
        weight: Param::new(get_weight(&weights, "model.projector.layers.weight")?),
        bias: Param::new(Some(get_weight(&weights, "model.projector.layers.bias")?)),
    };

    // View separator
    let view_seperator = get_weight(&weights, "model.view_seperator")?;

    // Embed tokens
    let embed_tokens = nn::Embedding {
        weight: Param::new(get_weight(&weights, "model.embed_tokens.weight")?),
    };

    // LLM layers
    eprintln!("Loading LLM decoder ({} layers)...", config.num_hidden_layers);
    let head_dim = config.hidden_size / config.num_attention_heads;

    let mut layers = Vec::new();
    for i in 0..config.num_hidden_layers {
        let lp = format!("model.layers.{}", i);
        let is_moe = i >= config.first_k_dense_replace;

        // RoPE: DeepSeek-V2 uses standard stride-based RoPE (non-traditional)
        // The Python code does rearrange+rotate_half which is equivalent to stride-based
        let rope = nn::RopeBuilder::new(head_dim)
            .base(config.rope_theta)
            .traditional(false)
            .build()
            .map_err(|e| Error::Model(format!("RoPE error: {:?}", e)))?;

        let attn = LLMAttention {
            n_heads: config.num_attention_heads,
            n_kv_heads: config.num_key_value_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            q_proj: nn::Linear {
                weight: Param::new(get_weight(&weights, &format!("{}.self_attn.q_proj.weight", lp))?),
                bias: Param::new(None),
            },
            k_proj: nn::Linear {
                weight: Param::new(get_weight(&weights, &format!("{}.self_attn.k_proj.weight", lp))?),
                bias: Param::new(None),
            },
            v_proj: nn::Linear {
                weight: Param::new(get_weight(&weights, &format!("{}.self_attn.v_proj.weight", lp))?),
                bias: Param::new(None),
            },
            o_proj: nn::Linear {
                weight: Param::new(get_weight(&weights, &format!("{}.self_attn.o_proj.weight", lp))?),
                bias: Param::new(None),
            },
            rope,
        };

        let input_layernorm = nn::RmsNorm {
            weight: Param::new(get_weight(&weights, &format!("{}.input_layernorm.weight", lp))?),
            eps: config.rms_norm_eps,
        };

        let post_attention_layernorm = nn::RmsNorm {
            weight: Param::new(get_weight(
                &weights,
                &format!("{}.post_attention_layernorm.weight", lp),
            )?),
            eps: config.rms_norm_eps,
        };

        let (mlp, moe) = if is_moe {
            // MoE layer
            let mut experts = Vec::new();
            for e in 0..config.n_routed_experts {
                let ep = format!("{}.mlp.experts.{}", lp, e);
                experts.push(MLP {
                    gate_proj: nn::Linear {
                        weight: Param::new(get_weight(&weights, &format!("{}.gate_proj.weight", ep))?),
                        bias: Param::new(None),
                    },
                    up_proj: nn::Linear {
                        weight: Param::new(get_weight(&weights, &format!("{}.up_proj.weight", ep))?),
                        bias: Param::new(None),
                    },
                    down_proj: nn::Linear {
                        weight: Param::new(get_weight(&weights, &format!("{}.down_proj.weight", ep))?),
                        bias: Param::new(None),
                    },
                });
            }

            // Shared experts
            let shared = MLP {
                gate_proj: nn::Linear {
                    weight: Param::new(get_weight(
                        &weights,
                        &format!("{}.mlp.shared_experts.gate_proj.weight", lp),
                    )?),
                    bias: Param::new(None),
                },
                up_proj: nn::Linear {
                    weight: Param::new(get_weight(
                        &weights,
                        &format!("{}.mlp.shared_experts.up_proj.weight", lp),
                    )?),
                    bias: Param::new(None),
                },
                down_proj: nn::Linear {
                    weight: Param::new(get_weight(
                        &weights,
                        &format!("{}.mlp.shared_experts.down_proj.weight", lp),
                    )?),
                    bias: Param::new(None),
                },
            };

            // Gate
            let gate = MoEGate {
                top_k: config.num_experts_per_tok,
                n_routed_experts: config.n_routed_experts,
                routed_scaling_factor: config.routed_scaling_factor,
                norm_topk_prob: config.norm_topk_prob,
                weight: Param::new(get_weight(
                    &weights,
                    &format!("{}.mlp.gate.weight", lp),
                )?),
            };

            let moe = MoE {
                num_experts_per_tok: config.num_experts_per_tok,
                n_routed_experts: config.n_routed_experts,
                gate,
                experts,
                shared_experts: Some(shared),
            };

            (None, Some(moe))
        } else {
            // Dense layer
            let mlp = MLP {
                gate_proj: nn::Linear {
                    weight: Param::new(get_weight(&weights, &format!("{}.mlp.gate_proj.weight", lp))?),
                    bias: Param::new(None),
                },
                up_proj: nn::Linear {
                    weight: Param::new(get_weight(&weights, &format!("{}.mlp.up_proj.weight", lp))?),
                    bias: Param::new(None),
                },
                down_proj: nn::Linear {
                    weight: Param::new(get_weight(&weights, &format!("{}.mlp.down_proj.weight", lp))?),
                    bias: Param::new(None),
                },
            };
            (Some(mlp), None)
        };

        layers.push(DecoderLayer {
            is_moe,
            self_attn: attn,
            mlp,
            moe,
            input_layernorm,
            post_attention_layernorm,
        });
    }

    // Final norm
    let norm = nn::RmsNorm {
        weight: Param::new(get_weight(&weights, "model.norm.weight")?),
        eps: config.rms_norm_eps,
    };

    // LM head
    let lm_head = nn::Linear {
        weight: Param::new(get_weight(&weights, "lm_head.weight")?),
        bias: Param::new(None),
    };

    eprintln!("Model loaded successfully!");

    Ok(DeepseekOCR2 {
        config,
        sam_model,
        qwen2_model,
        projector,
        view_seperator: Param::new(view_seperator),
        embed_tokens,
        layers,
        norm,
        lm_head,
    })
}
