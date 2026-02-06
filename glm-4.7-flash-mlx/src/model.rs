//! GLM-4.7-Flash model implementation with MLA (Multi-head Latent Attention)
//!
//! This module implements the GLM-4.7-Flash architecture (Glm4MoeLite) with:
//! - Absorbed MLA: kv_b_proj absorbed into embed_q (query-side) and unembed_out (output-side)
//! - Compressed KV cache: 576 floats/token/layer (kv_lora_rank=512 + qk_rope_head_dim=64)
//! - Mixture of Experts with top-k routing (shared + routed experts)
//! - 6-bit quantization support

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    argmax_axis, array,
    builder::Builder,
    categorical,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters as ModuleParametersTrait, ModuleParametersExt, Param},
    nn,
    ops::{
        concatenate_axis, expand_dims,
        indexing::{IndexOp, NewAxis, take_axis, take_along_axis},
        sigmoid,
    },
    Array, Dtype,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use mlx_rs_core::{
    cache::KeyValueCache,
    error::Error,
    fused_swiglu,
    utils::SdpaMask,
};

// ============================================================================
// Configuration
// ============================================================================

/// Quantization configuration for the model
#[derive(Debug, Clone, Deserialize, Default)]
pub struct QuantizationConfig {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
}

fn default_group_size() -> i32 { 64 }
fn default_bits() -> i32 { 6 }

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    pub vocab_size: i32,

    // MLA (Multi-head Latent Attention) specific fields
    pub kv_lora_rank: i32,
    pub q_lora_rank: i32,
    pub qk_nope_head_dim: i32,
    pub qk_rope_head_dim: i32,
    pub v_head_dim: i32,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    // MoE specific fields
    #[serde(default)]
    pub moe_intermediate_size: i32,
    #[serde(default)]
    pub n_routed_experts: i32,
    #[serde(default)]
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

    /// Quantization config (present for quantized models)
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

fn default_rms_norm_eps() -> f32 { 1e-6 }
fn default_rope_theta() -> f32 { 1000000.0 }
fn default_num_experts_per_tok() -> i32 { 4 }
fn default_first_k_dense_replace() -> i32 { 1 }
fn default_routed_scaling_factor() -> f32 { 1.0 }
fn default_n_group() -> i32 { 1 }
fn default_topk_group() -> i32 { 1 }

// ============================================================================
// QuantizedMultiLinear — per-head quantized projection
// ============================================================================

/// Per-head quantized linear projection with 3D weights.
///
/// Used for `embed_q` and `unembed_out` in the absorbed MLA formulation.
/// Weights have shape `[num_groups, output_dims, input_dims]` (quantized).
/// Uses `gather_qmm` with indices `[0, 1, ..., num_groups-1]` for per-head operation.
#[derive(Debug, Clone, ModuleParameters)]
pub struct QuantizedMultiLinear {
    pub num_groups: i32,
    pub group_size: i32,
    pub bits: i32,

    #[param]
    pub weight: Param<Array>,
    #[param]
    pub scales: Param<Array>,
    #[param]
    pub biases: Param<Array>,

    /// Precomputed indices [0, 1, ..., num_groups-1] for gather_qmm
    pub indices: Array,
}

impl QuantizedMultiLinear {
    /// Apply the per-head quantized projection.
    ///
    /// Input: `[..., num_groups, input_dims]`
    /// Output: `[..., num_groups, output_dims]`
    pub fn apply(&self, x: &Array) -> Result<Array, Exception> {
        mlx_rs::ops::gather_qmm(
            x,
            &*self.weight,
            &*self.scales,
            &*self.biases,
            None::<&Array>,
            Some(&self.indices),
            true,
            self.group_size,
            self.bits,
            None::<&str>,
            false,
        )
    }
}

// ============================================================================
// MLA (Multi-head Latent Attention)
// ============================================================================

/// Multi-head Latent Attention with absorbed KV projections.
///
/// Key idea: `kv_b_proj` is split and absorbed into:
/// - `embed_q`: per-head projection on query side (absorbs K portion)
/// - `unembed_out`: per-head projection on output side (absorbs V portion)
///
/// KV cache stores only compressed representation:
/// - keys: `[B, 1, L, kv_lora_rank + qk_rope_head_dim]` = `[B, 1, L, 576]`
/// - values: `[B, 1, L, kv_lora_rank]` = `[B, 1, L, 512]`
#[derive(Debug, Clone, ModuleParameters)]
pub struct MLAttention {
    pub n_heads: i32,
    pub kv_lora_rank: i32,
    pub qk_nope_head_dim: i32,
    pub qk_rope_head_dim: i32,
    pub v_head_dim: i32,
    pub scale: f32,

    // Query path: x → q_a_proj → q_a_layernorm → q_b_proj → split → embed_q + RoPE
    #[param]
    pub q_a_proj: nn::QuantizedLinear,
    #[param]
    pub q_a_layernorm: nn::RmsNorm,
    #[param]
    pub q_b_proj: nn::QuantizedLinear,
    #[param]
    pub embed_q: QuantizedMultiLinear,

    // KV path: x → kv_a_proj_with_mqa → split → kv_a_layernorm + RoPE
    #[param]
    pub kv_a_proj_with_mqa: nn::QuantizedLinear,
    #[param]
    pub kv_a_layernorm: nn::RmsNorm,

    // Output path: SDPA → unembed_out → reshape → o_proj
    #[param]
    pub unembed_out: QuantizedMultiLinear,
    #[param]
    pub o_proj: nn::QuantizedLinear,

    #[param]
    pub rope: nn::Rope,
}

pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut C,
}

impl<C> Module<AttentionInput<'_, C>> for MLAttention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;

        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        // === Query path ===
        // x → q_a_proj → q_a_layernorm → q_b_proj
        let q = self.q_a_proj.forward(x)?;
        let q = self.q_a_layernorm.forward(&q)?;
        let q = self.q_b_proj.forward(&q)?;

        // Reshape and transpose: [B, L, n_heads*(nope+rope)] → [B, n_heads, L, nope+rope]
        let head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim;
        let q = q.reshape(&[B as i32, L as i32, self.n_heads, head_dim])?;
        let q = q.transpose_axes(&[0, 2, 1, 3])?;

        // Split into nope and rope components
        let q_nope = q.index((.., .., .., ..self.qk_nope_head_dim));
        let q_pe = q.index((.., .., .., self.qk_nope_head_dim..));

        // Absorb kv_b_proj K into query: [B, n_heads, L, nope] → [B, n_heads, L, kv_lora_rank]
        let q_nope = self.embed_q.apply(&q_nope)?;

        // Apply RoPE to query PE component
        let q_pe_input = nn::RopeInputBuilder::new(&q_pe)
            .offset(cache.offset())
            .build()?;
        let q_pe = self.rope.forward(q_pe_input)?;

        // Concatenate: [B, n_heads, L, kv_lora_rank + rope_dim]
        let queries = concatenate_axis(&[q_nope, q_pe], -1)?;

        // === KV path ===
        // x → kv_a_proj_with_mqa: [B, L, kv_lora_rank + rope_dim]
        let kv = self.kv_a_proj_with_mqa.forward(x)?;

        // Split into latent and rope components
        let kv_latent = kv.index((.., .., ..self.kv_lora_rank));
        let k_pe = kv.index((.., .., self.kv_lora_rank..));

        // Normalize latent
        let kv_latent = self.kv_a_layernorm.forward(&kv_latent)?;

        // Add head dimension: [B, L, D] → [B, 1, L, D]
        let kv_latent = expand_dims(&kv_latent, 1)?;
        let k_pe = expand_dims(&k_pe, 1)?;

        // Apply RoPE to key PE component
        let k_pe_input = nn::RopeInputBuilder::new(&k_pe)
            .offset(cache.offset())
            .build()?;
        let k_pe = self.rope.forward(k_pe_input)?;

        // keys: [B, 1, L, kv_lora_rank + rope_dim] (shared across all heads)
        let keys = concatenate_axis(&[&kv_latent, &k_pe], -1)?;
        // values: [B, 1, L, kv_lora_rank] (latent only)
        let values = kv_latent;

        // Update cache and get full history
        let (keys, values) = cache.update_and_fetch(keys, values)?;

        // === Attention ===
        // queries: [B, n_heads, L, kv_lora_rank+rope_dim]
        // keys:    [B, 1, S, kv_lora_rank+rope_dim] — broadcasts to n_heads
        // values:  [B, 1, S, kv_lora_rank] — broadcasts to n_heads
        let sdpa_mask = match mask {
            Some(m) => Some(SdpaMask::Array(m)),
            None if L > 1 => Some(SdpaMask::Causal),
            None => None,
        };

        let output = mlx_rs_core::utils::scaled_dot_product_attention(
            queries, keys, values, Some(cache), self.scale, sdpa_mask,
        )?;
        // output: [B, n_heads, L, kv_lora_rank]

        // === Output path ===
        // Absorb kv_b_proj V into output: [B, n_heads, L, kv_lora_rank] → [B, n_heads, L, v_head_dim]
        let output = self.unembed_out.apply(&output)?;

        // Reshape: [B, n_heads, L, v_head_dim] → [B, L, n_heads * v_head_dim]
        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B as i32, L as i32, -1])?;

        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_a_proj.training_mode(mode);
        self.q_b_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        self.kv_a_proj_with_mqa.training_mode(mode);
        <nn::Rope as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

// ============================================================================
// MLP and MoE (shared with glm4-moe-mlx)
// ============================================================================

/// Standard MLP (used for dense layers and shared experts)
#[derive(Debug, Clone, ModuleParameters)]
pub struct MLP {
    #[param]
    pub gate_proj: nn::QuantizedLinear,
    #[param]
    pub up_proj: nn::QuantizedLinear,
    #[param]
    pub down_proj: nn::QuantizedLinear,
}

impl Module<&Array> for MLP {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = fused_swiglu(&up, &gate)?;
        self.down_proj.forward(&activated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

/// MoE Gate for expert routing
#[derive(Debug, Clone, ModuleParameters)]
pub struct MoEGate {
    pub top_k: i32,
    pub n_routed_experts: i32,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    pub n_group: i32,
    pub topk_group: i32,

    #[param]
    pub weight: Param<Array>,
    #[param]
    pub e_score_correction_bias: Param<Array>,
}

impl MoEGate {
    /// Returns (expert_indices, expert_weights) for top-k routing
    pub fn route(&self, x: &Array) -> Result<(Array, Array), Exception> {
        let gates = x.matmul(&(*self.weight).t())?;

        let orig_scores = sigmoid(&gates.as_dtype(Dtype::Float32)?)?;
        let scores_with_bias = orig_scores.add(&*self.e_score_correction_bias)?;

        let neg_scores = scores_with_bias.negative()?;
        let partitioned_inds = mlx_rs::ops::argpartition_axis(&neg_scores, self.top_k - 1, -1)?;
        let inds = partitioned_inds.index((.., .., ..self.top_k));
        let selected_scores = take_along_axis(&orig_scores, &inds, -1)?;

        let scaling_arr = array!(self.routed_scaling_factor);
        let final_scores = if self.norm_topk_prob && self.top_k > 1 {
            let denom = selected_scores.sum_axis(-1, true)?;
            let normalized = selected_scores.divide(&denom)?;
            normalized.multiply(&scaling_arr)?
        } else {
            selected_scores.multiply(&scaling_arr)?
        };

        Ok((inds, final_scores))
    }
}

/// Quantized Switch Linear for MoE experts
/// Stores stacked weights for all experts: [n_experts, out_dim, in_dim]
#[derive(Debug, Clone, ModuleParameters)]
pub struct QuantizedSwitchLinear {
    pub num_experts: i32,
    pub input_dims: i32,
    pub output_dims: i32,
    pub group_size: i32,
    pub bits: i32,

    #[param]
    pub weight: Param<Array>,
    #[param]
    pub scales: Param<Array>,
    #[param]
    pub biases: Param<Array>,
}

impl QuantizedSwitchLinear {
    pub fn apply(&self, x: &Array, indices: &Array, sorted_indices: bool) -> Result<Array, Exception> {
        mlx_rs::ops::gather_qmm(
            x,
            &*self.weight,
            &*self.scales,
            &*self.biases,
            None::<&Array>,
            Some(indices),
            true,
            self.group_size,
            self.bits,
            None::<&str>,
            sorted_indices,
        )
    }
}

/// Sort tokens by their expert indices for coalesced memory access.
fn gather_sort(x: &Array, indices: &Array) -> Result<(Array, Array, Array), Exception> {
    let indices_shape = indices.shape();
    let m = *indices_shape.last().unwrap() as i32;

    let indices_flat = indices.flatten(None, None)?;
    let order = mlx_rs::ops::argsort(&indices_flat)?;
    let inv_order = mlx_rs::ops::argsort(&order)?;

    let x_shape = x.shape();
    let d = *x_shape.last().unwrap() as i32;
    let x_flat = x.reshape(&[-1, 1, d])?;

    let token_order = order.floor_divide(mlx_rs::array!(m))?;
    let x_sorted = take_axis(&x_flat, &token_order, 0)?;
    let indices_sorted = take_axis(&indices_flat, &order, 0)?;

    Ok((x_sorted, indices_sorted, inv_order))
}

/// Unsort the output back to original token order.
fn scatter_unsort(x: &Array, inv_order: &Array, original_shape: &[i32]) -> Result<Array, Exception> {
    let x_shape = x.shape();
    let d = *x_shape.last().unwrap() as i32;

    let x_flat = x.reshape(&[-1, d])?;
    let x_unsorted = take_axis(&x_flat, inv_order, 0)?;

    let mut new_shape: Vec<i32> = original_shape.to_vec();
    new_shape.push(1);
    new_shape.push(d);
    x_unsorted.reshape(&new_shape)
}

/// SwitchGLU MLP for routed experts
#[derive(Debug, Clone, ModuleParameters)]
pub struct SwitchGLU {
    #[param]
    pub gate_proj: QuantizedSwitchLinear,
    #[param]
    pub up_proj: QuantizedSwitchLinear,
    #[param]
    pub down_proj: QuantizedSwitchLinear,
}

impl SwitchGLU {
    pub fn forward_experts(&mut self, x: &Array, indices: &Array) -> Result<Array, Exception> {
        let indices_shape = indices.shape();
        let b = indices_shape[0];
        let l = indices_shape[1];
        let k = indices_shape[2];

        let x_expanded = expand_dims(x, -2)?;
        let x_expanded = expand_dims(&x_expanded, -2)?;

        let indices_size = b * l * k;
        let do_sort = indices_size >= 64;

        if do_sort {
            let (x_sorted, indices_sorted, inv_order) = gather_sort(&x_expanded, indices)?;

            let gate = self.gate_proj.apply(&x_sorted, &indices_sorted, true)?;
            let up = self.up_proj.apply(&x_sorted, &indices_sorted, true)?;
            let activated = fused_swiglu(&up, &gate)?;
            let output = self.down_proj.apply(&activated, &indices_sorted, true)?;
            let output_unsorted = scatter_unsort(&output, &inv_order, &[b as i32, l as i32, k as i32])?;

            let shape = output_unsorted.shape();
            output_unsorted.reshape(&[shape[0] as i32, shape[1] as i32, shape[2] as i32, shape[4] as i32])
        } else {
            let gate = self.gate_proj.apply(&x_expanded, indices, false)?;
            let up = self.up_proj.apply(&x_expanded, indices, false)?;
            let activated = fused_swiglu(&up, &gate)?;
            let output = self.down_proj.apply(&activated, indices, false)?;

            let shape = output.shape();
            if shape.len() == 5 {
                output.reshape(&[shape[0] as i32, shape[1] as i32, shape[2] as i32, shape[4] as i32])
            } else {
                Ok(output)
            }
        }
    }
}

/// Mixture of Experts block
#[derive(Debug, Clone, ModuleParameters)]
pub struct MoE {
    pub num_experts_per_tok: i32,
    pub has_shared_experts: bool,

    #[param]
    pub gate: MoEGate,
    #[param]
    pub switch_mlp: SwitchGLU,
    #[param]
    pub shared_experts: Option<MLP>,
}

impl Module<&Array> for MoE {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let (indices, scores) = self.gate.route(x)?;

        let expert_out = self.switch_mlp.forward_experts(x, &indices)?;

        let scores_expanded = scores.index((.., .., .., NewAxis));
        let weighted = expert_out.multiply(&scores_expanded)?;
        let mut y = weighted.sum_axis(2, false)?.as_dtype(x.dtype())?;

        if let Some(ref mut shared) = self.shared_experts {
            let shared_out = shared.forward(x)?;
            y = y.add(&shared_out)?;
        }

        Ok(y)
    }

    fn training_mode(&mut self, mode: bool) {
        if let Some(ref mut shared) = self.shared_experts {
            shared.training_mode(mode);
        }
    }
}

// ============================================================================
// Decoder Layer, Language Model, Model
// ============================================================================

/// Decoder layer (can be dense or MoE) with MLA attention
#[derive(Debug, Clone, ModuleParameters)]
pub struct DecoderLayer {
    pub layer_idx: i32,
    pub is_moe: bool,

    #[param]
    pub self_attn: MLAttention,
    #[param]
    pub mlp: Option<MLP>,
    #[param]
    pub moe: Option<MoE>,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl<C> Module<AttentionInput<'_, C>> for DecoderLayer
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;

        let normed = self.input_layernorm.forward(x)?;
        let attn_input = AttentionInput {
            x: &normed,
            mask,
            cache,
        };
        let attn_out = self.self_attn.forward(attn_input)?;
        let h = x.add(&attn_out)?;

        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = if self.is_moe {
            self.moe.as_mut().unwrap().forward(&normed)?
        } else {
            self.mlp.as_mut().unwrap().forward(&normed)?
        };

        h.add(&mlp_out)
    }

    fn training_mode(&mut self, mode: bool) {
        <MLAttention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        if let Some(ref mut mlp) = self.mlp {
            mlp.training_mode(mode);
        }
        if let Some(ref mut moe) = self.moe {
            moe.training_mode(mode);
        }
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct LanguageModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[param]
    pub embed_tokens: nn::QuantizedEmbedding,
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}

pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<C>,
}

impl<C> Module<ModelInput<'_, C>> for LanguageModel
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput { inputs, mask, cache } = input;

        let mut h = self.embed_tokens.forward(inputs)?;

        let mask = mask.cloned();

        assert!(!cache.is_empty(), "Cache must be pre-allocated with init_cache()");

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c,
            };
            h = layer.forward(layer_input)?;
        }

        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <DecoderLayer as Module<AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Model {
    pub args: ModelArgs,

    #[param]
    pub model: LanguageModel,

    #[param]
    pub lm_head: nn::QuantizedLinear,
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self.model.forward(input)?;
        self.lm_head.forward(&out)
    }

    fn training_mode(&mut self, mode: bool) {
        <LanguageModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        self.lm_head.training_mode(mode);
    }
}

// ============================================================================
// Loading functions
// ============================================================================

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let model_args_filename = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(model_args_filename)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;
    Ok(model_args)
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>, Error> {
    let weights_index = model_dir.join("model.safetensors.index.json");
    let json = std::fs::read_to_string(weights_index)?;
    let weight_map: WeightMap = serde_json::from_str(&json)?;

    let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();

    let mut all_weights: HashMap<String, Array> = HashMap::new();

    for weight_file in weight_files {
        let weights_filename = model_dir.join(weight_file);
        let loaded = Array::load_safetensors(&weights_filename)?;
        all_weights.extend(loaded);
    }

    Ok(all_weights)
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    weights.get(key)
        .cloned()
        .ok_or_else(|| Error::Model(format!("Weight not found: {}", key)))
}

fn get_weight_optional(weights: &HashMap<String, Array>, key: &str) -> Option<Array> {
    weights.get(key).cloned()
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
    let linear_bias = get_weight_optional(weights, &format!("{}.bias", prefix));

    let inner = nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(linear_bias),
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

/// Absorb kv_b_proj into per-head embed_q and unembed_out projections.
///
/// kv_b_proj maps compressed latent to full K_nope and V per head:
///   weight shape: [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
///
/// Absorption splits this into:
///   embed_q:     [n_heads, kv_lora_rank, qk_nope_head_dim] — absorbs K portion into Q
///   unembed_out: [n_heads, v_head_dim, kv_lora_rank]       — absorbs V portion into output
fn convert_kv_b_proj(
    weights: &HashMap<String, Array>,
    prefix: &str,
    n_heads: i32,
    qk_nope_head_dim: i32,
    v_head_dim: i32,
    kv_lora_rank: i32,
    group_size: i32,
    bits: i32,
) -> Result<(QuantizedMultiLinear, QuantizedMultiLinear), Error> {
    let weight_q = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;

    // Dequantize: [n_heads*(nope+v), kv_lora_rank] → float
    let weight = mlx_rs::ops::dequantize(
        &weight_q, &scales, &biases, group_size, bits, None::<&str>,
    )?;

    // Reshape to [n_heads, nope+v, kv_lora_rank]
    let weight = weight.reshape(&[n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank])?;

    // Split into K and V portions
    let k_weight = weight.index((.., ..qk_nope_head_dim, ..));   // [n_heads, nope, lora_rank]
    let v_weight = weight.index((.., qk_nope_head_dim.., ..));   // [n_heads, v_dim, lora_rank]

    // embed_q: absorb K into Q side
    // Q_absorbed = Q_nope @ W_K  where W_K per head is [nope, lora_rank]
    // For gather_qmm(transpose=True): output = x @ W.T, so W = W_K.T = [lora_rank, nope]
    let embed_q_weight = k_weight.transpose_axes(&[0, 2, 1])?;  // [n_heads, lora_rank, nope]

    // unembed_out: absorb V into output side
    // output = sdpa_out @ W_V.T  where W_V per head is [v_dim, lora_rank]
    // For gather_qmm(transpose=True): output = x @ W.T, so W = W_V = [v_dim, lora_rank]
    let unembed_out_weight = v_weight;  // [n_heads, v_dim, lora_rank]

    // Re-quantize both 3D tensors
    let (eq_wq, eq_scales, eq_biases) = mlx_rs::ops::quantize(
        &embed_q_weight, group_size, bits, None::<&str>,
    )?;
    let (uo_wq, uo_scales, uo_biases) = mlx_rs::ops::quantize(
        &unembed_out_weight, group_size, bits, None::<&str>,
    )?;

    // Force evaluation before wrapping in Params
    mlx_rs::transforms::eval([
        &eq_wq, &eq_scales, &eq_biases,
        &uo_wq, &uo_scales, &uo_biases,
    ])?;

    let idx_vec: Vec<u32> = (0..n_heads as u32).collect();

    let mut embed_q = QuantizedMultiLinear {
        num_groups: n_heads,
        group_size,
        bits,
        weight: Param::new(eq_wq),
        scales: Param::new(eq_scales),
        biases: Param::new(eq_biases),
        indices: Array::from(idx_vec.as_slice()),
    };
    embed_q.freeze_parameters(true);

    let mut unembed_out = QuantizedMultiLinear {
        num_groups: n_heads,
        group_size,
        bits,
        weight: Param::new(uo_wq),
        scales: Param::new(uo_scales),
        biases: Param::new(uo_biases),
        indices: Array::from(idx_vec.as_slice()),
    };
    unembed_out.freeze_parameters(true);

    Ok((embed_q, unembed_out))
}

fn make_quantized_switch_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<QuantizedSwitchLinear, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;

    let shape = weight.shape();
    let num_experts = shape[0] as i32;
    let output_dims = shape[1] as i32;
    let scales_shape = scales.shape();
    let input_dims = (scales_shape[2] as i32) * group_size;

    Ok(QuantizedSwitchLinear {
        num_experts,
        input_dims,
        output_dims,
        group_size,
        bits,
        weight: Param::new(weight),
        scales: Param::new(scales),
        biases: Param::new(biases),
    })
}

fn make_quantized_mlp(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<MLP, Error> {
    Ok(MLP {
        gate_proj: make_quantized_linear(
            weights, &format!("{}.gate_proj", prefix), group_size, bits
        )?,
        up_proj: make_quantized_linear(
            weights, &format!("{}.up_proj", prefix), group_size, bits
        )?,
        down_proj: make_quantized_linear(
            weights, &format!("{}.down_proj", prefix), group_size, bits
        )?,
    })
}

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let args = get_model_args(model_dir)?;

    let quant_config = args.quantization.as_ref()
        .ok_or_else(|| Error::Model("GLM-4.7-Flash requires quantized model".to_string()))?;
    let group_size = quant_config.group_size;
    let bits = quant_config.bits;

    eprintln!("Loading weights for {}-bit quantized model...", bits);
    let weights = load_all_weights(model_dir)?;

    let rope_dim = args.qk_rope_head_dim;

    let mut layers = Vec::with_capacity(args.num_hidden_layers as usize);

    for i in 0..args.num_hidden_layers {
        let layer_prefix = format!("model.layers.{}", i);
        let is_moe = i >= args.first_k_dense_replace;

        // Absorb kv_b_proj into per-head embed_q and unembed_out
        let (embed_q, unembed_out) = convert_kv_b_proj(
            &weights, &format!("{}.self_attn.kv_b_proj", layer_prefix),
            args.num_attention_heads, args.qk_nope_head_dim, args.v_head_dim,
            args.kv_lora_rank, group_size, bits,
        )?;

        if i % 10 == 0 {
            eprintln!("  Loading layer {}/{}...", i, args.num_hidden_layers);
        }

        // Build MLA attention
        let attention = MLAttention {
            n_heads: args.num_attention_heads,
            kv_lora_rank: args.kv_lora_rank,
            qk_nope_head_dim: args.qk_nope_head_dim,
            qk_rope_head_dim: args.qk_rope_head_dim,
            v_head_dim: args.v_head_dim,
            scale: ((args.qk_nope_head_dim + args.qk_rope_head_dim) as f32).sqrt().recip(),

            q_a_proj: make_quantized_linear(
                &weights, &format!("{}.self_attn.q_a_proj", layer_prefix), group_size, bits
            )?,
            q_a_layernorm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.self_attn.q_a_layernorm.weight", layer_prefix))?),
                eps: args.rms_norm_eps,
            },
            q_b_proj: make_quantized_linear(
                &weights, &format!("{}.self_attn.q_b_proj", layer_prefix), group_size, bits
            )?,
            embed_q,

            kv_a_proj_with_mqa: make_quantized_linear(
                &weights, &format!("{}.self_attn.kv_a_proj_with_mqa", layer_prefix), group_size, bits
            )?,
            kv_a_layernorm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.self_attn.kv_a_layernorm.weight", layer_prefix))?),
                eps: args.rms_norm_eps,
            },

            unembed_out,
            o_proj: make_quantized_linear(
                &weights, &format!("{}.self_attn.o_proj", layer_prefix), group_size, bits
            )?,

            rope: nn::RopeBuilder::new(rope_dim)
                .base(args.rope_theta)
                .traditional(true)
                .build()
                .unwrap(),
        };

        let (mlp, moe) = if is_moe {
            let gate = MoEGate {
                top_k: args.num_experts_per_tok,
                n_routed_experts: args.n_routed_experts,
                routed_scaling_factor: args.routed_scaling_factor,
                norm_topk_prob: args.norm_topk_prob,
                n_group: args.n_group,
                topk_group: args.topk_group,
                weight: Param::new(get_weight(&weights, &format!("{}.mlp.gate.weight", layer_prefix))?),
                e_score_correction_bias: Param::new(get_weight(&weights, &format!("{}.mlp.gate.e_score_correction_bias", layer_prefix))?),
            };

            let switch_mlp = SwitchGLU {
                gate_proj: make_quantized_switch_linear(
                    &weights, &format!("{}.mlp.switch_mlp.gate_proj", layer_prefix), group_size, bits
                )?,
                up_proj: make_quantized_switch_linear(
                    &weights, &format!("{}.mlp.switch_mlp.up_proj", layer_prefix), group_size, bits
                )?,
                down_proj: make_quantized_switch_linear(
                    &weights, &format!("{}.mlp.switch_mlp.down_proj", layer_prefix), group_size, bits
                )?,
            };

            let shared_experts = if args.n_shared_experts > 0 {
                Some(make_quantized_mlp(
                    &weights, &format!("{}.mlp.shared_experts", layer_prefix), group_size, bits
                )?)
            } else {
                None
            };

            let moe = MoE {
                num_experts_per_tok: args.num_experts_per_tok,
                has_shared_experts: args.n_shared_experts > 0,
                gate,
                switch_mlp,
                shared_experts,
            };

            (None, Some(moe))
        } else {
            let mlp = make_quantized_mlp(
                &weights, &format!("{}.mlp", layer_prefix), group_size, bits
            )?;
            (Some(mlp), None)
        };

        let layer = DecoderLayer {
            layer_idx: i,
            is_moe,
            self_attn: attention,
            mlp,
            moe,
            input_layernorm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.input_layernorm.weight", layer_prefix))?),
                eps: args.rms_norm_eps,
            },
            post_attention_layernorm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.post_attention_layernorm.weight", layer_prefix))?),
                eps: args.rms_norm_eps,
            },
        };

        layers.push(layer);
    }

    let language_model = LanguageModel {
        vocab_size: args.vocab_size,
        num_hidden_layers: args.num_hidden_layers,
        embed_tokens: make_quantized_embedding(
            &weights, "model.embed_tokens", group_size, bits
        )?,
        layers,
        norm: nn::RmsNorm {
            weight: Param::new(get_weight(&weights, "model.norm.weight")?),
            eps: args.rms_norm_eps,
        },
    };

    let lm_head = make_quantized_linear(
        &weights, "lm_head", group_size, bits
    )?;

    let model = Model {
        args,
        model: language_model,
        lm_head,
    };

    model.eval()?;

    Ok(model)
}

// ============================================================================
// Generation
// ============================================================================

pub fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    match temp {
        0.0 => argmax_axis!(logits, -1).map_err(Into::into),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical!(logits).map_err(Into::into)
        }
    }
}

pub struct Generate<'a, C> {
    model: &'a mut Model,
    cache: &'a mut Vec<C>,
    temp: f32,
    state: GenerateState<'a>,
}

/// Initialize KV cache for a model with the given number of layers
pub fn init_cache<C: KeyValueCache + Default>(num_layers: usize) -> Vec<C> {
    (0..num_layers).map(|_| C::default()).collect()
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<C>,
        temp: f32,
        prompt_token: &'a Array,
    ) -> Self {
        if cache.is_empty() {
            *cache = init_cache(model.model.num_hidden_layers as usize);
        }
        Self {
            model,
            cache,
            temp,
            state: GenerateState::Prefill { prompt_token },
        }
    }
}

/// State machine for pipelined token generation
pub enum GenerateState<'a> {
    Prefill { prompt_token: &'a Array },
    FirstDecode { y: Array },
    Pipelined { current_y: Array },
    Done,
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    fn compute_next(&mut self, y: &Array) -> Result<Array, Exception> {
        let inputs = y.index((.., NewAxis));
        let input = ModelInput {
            inputs: &inputs,
            mask: None,
            cache: self.cache,
        };
        let logits = self.model.forward(input)?;
        sample(&logits, self.temp)
    }
}

impl<'a, C> Iterator for Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        let state = std::mem::replace(&mut self.state, GenerateState::Done);

        match state {
            GenerateState::Prefill { prompt_token } => {
                let input = ModelInput {
                    inputs: prompt_token,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));

                tri!(mlx_rs::transforms::async_eval([&y]));
                tri!(mlx_rs::transforms::eval([&y]));

                let next_y = tri!(self.compute_next(&y));
                tri!(mlx_rs::transforms::async_eval([&next_y]));

                self.state = GenerateState::Pipelined { current_y: next_y };
                Some(Ok(y))
            }
            GenerateState::FirstDecode { y } => {
                self.state = GenerateState::Done;
                Some(Ok(y))
            }
            GenerateState::Pipelined { current_y } => {
                let next_y = tri!(self.compute_next(&current_y));
                tri!(mlx_rs::transforms::async_eval([&next_y]));

                self.state = GenerateState::Pipelined { current_y: next_y };
                Some(Ok(current_y))
            }
            GenerateState::Done => None,
        }
    }
}
