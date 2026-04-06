//! Gemma 4 text-only model implementation.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{
        self,
        indexing::{take_along_axis, take_axis, IndexOp, NewAxis},
    },
    Array, Dtype,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use mlx_rs_core::{
    cache::KeyValueCache,
    error::Error,
    sampler::{DefaultSampler, Sampler},
    utils::{create_causal_mask, scaled_dot_product_attention, SdpaMask},
};

// ============================================================================
// Configuration
// ============================================================================

/// Deserialize a JSON value that may be `null` or missing as `0`.
fn nullable_i32<'de, D>(deserializer: D) -> Result<i32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<i32>::deserialize(deserializer).map(|opt| opt.unwrap_or(0))
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4Config {
    pub model_type: String,
    pub text_config: Gemma4TextConfig,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4TextConfig {
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub attention_k_eq_v: bool,
    pub enable_moe_block: bool,
    #[serde(default, deserialize_with = "nullable_i32")]
    pub global_head_dim: i32,
    pub head_dim: i32,
    pub hidden_activation: String,
    pub hidden_size: i32,
    #[serde(default)]
    pub hidden_size_per_layer_input: i32,
    pub intermediate_size: i32,
    pub layer_types: Vec<String>,
    pub max_position_embeddings: i32,
    #[serde(default, deserialize_with = "nullable_i32")]
    pub moe_intermediate_size: i32,
    pub num_attention_heads: i32,
    #[serde(default, deserialize_with = "nullable_i32")]
    pub num_experts: i32,
    #[serde(default, deserialize_with = "nullable_i32")]
    pub num_global_key_value_heads: i32,
    pub num_hidden_layers: i32,
    #[serde(default)]
    pub num_kv_shared_layers: i32,
    pub num_key_value_heads: i32,
    pub rms_norm_eps: f32,
    pub rope_parameters: RopeParameters,
    pub sliding_window: i32,
    pub tie_word_embeddings: bool,
    #[serde(default, deserialize_with = "nullable_i32")]
    pub top_k_experts: i32,
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    pub vocab_size: i32,
    #[serde(default)]
    pub vocab_size_per_layer_input: i32,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub sliding_attention: RopeSpec,
    pub full_attention: RopeSpec,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeSpec {
    pub rope_theta: f32,
    pub rope_type: String,
    #[serde(default)]
    pub partial_rotary_factor: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

impl LayerType {
    fn from_config(layer_type: &str) -> Result<Self, Error> {
        match layer_type {
            "sliding_attention" => Ok(Self::SlidingAttention),
            "full_attention" => Ok(Self::FullAttention),
            other => Err(Error::Model(format!(
                "Unsupported Gemma4 layer type: {other}"
            ))),
        }
    }

    fn is_sliding(self) -> bool {
        matches!(self, Self::SlidingAttention)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GemmaActivation {
    GeluPytorchTanh,
    Silu,
}

impl GemmaActivation {
    fn from_name(name: &str) -> Result<Self, Error> {
        match name {
            "gelu_pytorch_tanh" => Ok(Self::GeluPytorchTanh),
            "silu" | "swish" => Ok(Self::Silu),
            other => Err(Error::Model(format!(
                "Unsupported Gemma4 activation: {other}"
            ))),
        }
    }

    fn apply(self, x: &Array) -> Result<Array, Exception> {
        match self {
            Self::GeluPytorchTanh => nn::gelu_approximate(x),
            Self::Silu => nn::silu(x),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UnscaledRmsNorm {
    pub eps: f32,
}

impl UnscaledRmsNorm {
    pub fn new(eps: f32) -> Self {
        Self { eps }
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let x_f32 = x.as_dtype(Dtype::Float32)?;
        let variance = x_f32.square()?.mean_axis(-1, true)?;
        let scale = variance.add(&array!(self.eps))?.rsqrt()?;
        x_f32.multiply(&scale)?.as_dtype(x.dtype())
    }
}

// ============================================================================
// Rotary embeddings
// ============================================================================

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum RotaryLayout {
    BatchHeadsSeqDim,
    BatchSeqHeadsDim,
}

#[derive(Debug, Clone)]
enum GemmaRope {
    Standard(nn::Rope),
    Proportional(ProportionalRope),
}

impl GemmaRope {
    fn apply(&mut self, x: &Array, offset: i32) -> Result<Array, Exception> {
        match self {
            Self::Standard(rope) => {
                rope.forward(nn::RopeInputBuilder::new(x).offset(offset).build()?)
            }
            Self::Proportional(rope) => rope.apply(x, offset),
        }
    }

    #[allow(dead_code)]
    fn apply_with_layout(
        &mut self,
        x: &Array,
        offset: i32,
        layout: RotaryLayout,
    ) -> Result<Array, Exception> {
        match self {
            Self::Standard(rope) => match x.shape().len() {
                3 => rope.forward(nn::RopeInputBuilder::new(x).offset(offset).build()?),
                4 => match layout {
                    RotaryLayout::BatchHeadsSeqDim => {
                        rope.forward(nn::RopeInputBuilder::new(x).offset(offset).build()?)
                    }
                    RotaryLayout::BatchSeqHeadsDim => {
                        let transposed = x.transpose_axes(&[0, 2, 1, 3])?;
                        let rotated = rope.forward(
                            nn::RopeInputBuilder::new(&transposed)
                                .offset(offset)
                                .build()?,
                        )?;
                        rotated.transpose_axes(&[0, 2, 1, 3])
                    }
                },
                ndim => Err(Exception::custom(format!(
                    "Gemma standard RoPE expects 3D or 4D input, got {ndim}D"
                ))),
            },
            Self::Proportional(rope) => rope.apply_with_layout(x, offset, layout),
        }
    }

    fn training_mode(&mut self, mode: bool) {
        if let Self::Standard(rope) = self {
            <nn::Rope as Module<nn::RopeInput>>::training_mode(rope, mode);
        }
    }
}

#[derive(Debug, Clone)]
struct ProportionalRope {
    inv_freq: Array,
    head_dim: i32,
}

impl ProportionalRope {
    fn new(head_dim: i32, theta: f32, partial_rotary_factor: f32) -> Self {
        let half_dim = head_dim / 2;
        let rope_angles = ((partial_rotary_factor * head_dim as f32) / 2.0).floor() as i32;
        let rope_angles = rope_angles.clamp(0, half_dim);
        let nope_angles = half_dim - rope_angles;

        let mut inv_freq = Vec::with_capacity(half_dim as usize);
        for i in 0..rope_angles {
            inv_freq.push(1.0 / theta.powf((2 * i) as f32 / head_dim as f32));
        }
        inv_freq.extend(std::iter::repeat_n(0.0, nope_angles as usize));

        Self {
            inv_freq: Array::from_slice(&inv_freq, &[half_dim]),
            head_dim,
        }
    }

    fn apply(&self, x: &Array, offset: i32) -> Result<Array, Exception> {
        self.apply_with_layout(x, offset, RotaryLayout::BatchHeadsSeqDim)
    }

    #[allow(dead_code)]
    fn apply_with_layout(
        &self,
        x: &Array,
        offset: i32,
        layout: RotaryLayout,
    ) -> Result<Array, Exception> {
        match x.shape().len() {
            3 => self.apply_3d(x, offset),
            4 => self.apply_4d(x, offset, layout),
            ndim => Err(Exception::custom(format!(
                "Gemma proportional RoPE expects 3D or 4D input, got {ndim}D"
            ))),
        }
    }

    fn apply_3d(&self, x: &Array, offset: i32) -> Result<Array, Exception> {
        let seq_len = x.shape()[1];
        let (cos, sin) = self.cos_sin(seq_len, offset, x.dtype(), &[1, seq_len, self.head_dim])?;
        apply_rotary_pos_emb(x, &cos, &sin)
    }

    fn apply_4d(&self, x: &Array, offset: i32, layout: RotaryLayout) -> Result<Array, Exception> {
        let (seq_len, shape) = match layout {
            RotaryLayout::BatchHeadsSeqDim => (x.shape()[2], [1, 1, x.shape()[2], self.head_dim]),
            RotaryLayout::BatchSeqHeadsDim => (x.shape()[1], [1, x.shape()[1], 1, self.head_dim]),
        };
        let (cos, sin) = self.cos_sin(seq_len, offset, x.dtype(), &shape)?;
        apply_rotary_pos_emb(x, &cos, &sin)
    }

    fn cos_sin(
        &self,
        seq_len: i32,
        offset: i32,
        dtype: Dtype,
        shape: &[i32],
    ) -> Result<(Array, Array), Exception> {
        let positions = ops::arange::<_, f32>(offset, offset + seq_len, 1)?;
        let freqs = ops::outer(&positions, &self.inv_freq)?;
        let emb = ops::concatenate_axis(&[&freqs, &freqs], -1)?;
        let cos = ops::cos(&emb)?.as_dtype(dtype)?.reshape(shape)?;
        let sin = ops::sin(&emb)?.as_dtype(dtype)?.reshape(shape)?;
        Ok((cos, sin))
    }
}

fn rotate_half(x: &Array) -> Result<Array, Exception> {
    let half_dim = x.shape()[x.shape().len() - 1] / 2;
    let rotated = match x.shape().len() {
        3 => {
            let x1 = x.index((.., .., ..half_dim));
            let x2 = x.index((.., .., half_dim..));
            let neg_x2 = x2.negative()?;
            ops::concatenate_axis(&[&neg_x2, &x1], -1)?
        }
        4 => {
            let x1 = x.index((.., .., .., ..half_dim));
            let x2 = x.index((.., .., .., half_dim..));
            let neg_x2 = x2.negative()?;
            ops::concatenate_axis(&[&neg_x2, &x1], -1)?
        }
        ndim => {
            return Err(Exception::custom(format!(
                "Gemma rotate_half expects 3D or 4D input, got {ndim}D"
            )))
        }
    };
    Ok(rotated)
}

fn apply_rotary_pos_emb(x: &Array, cos: &Array, sin: &Array) -> Result<Array, Exception> {
    let rotated = rotate_half(x)?;
    x.multiply(cos)?.add(&rotated.multiply(sin)?)
}

// ============================================================================
// Attention
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct Attention {
    pub layer_idx: i32,
    pub layer_type: LayerType,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub sliding_window: Option<i32>,
    /// If true, this layer shares KV from an earlier layer (no k/v projections).
    pub is_kv_shared: bool,

    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: Option<nn::Linear>,
    #[param]
    pub v_proj: Option<nn::Linear>,
    #[param]
    pub o_proj: nn::Linear,
    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: Option<nn::RmsNorm>,
    rope: GemmaRope,

    pub v_norm: Option<UnscaledRmsNorm>,
}

pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut C,
    /// Pre-computed shared KV from an earlier layer (for KV-shared layers).
    pub shared_kv: Option<(Array, Array)>,
}

impl<C> Module<AttentionInput<'_, C>> for Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput {
            x,
            mask,
            cache,
            shared_kv,
        } = input;

        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        let offset = cache.offset();

        let queries = self.q_proj.forward(x)?;
        let mut queries =
            self.q_norm
                .forward(&queries.reshape(&[B, L, self.n_heads, self.head_dim])?)?;
        queries = queries.transpose_axes(&[0, 2, 1, 3])?;

        let is_shared_kv = shared_kv.is_some();
        let (keys, values) = if let Some((shared_k, shared_v)) = shared_kv {
            // KV-shared layer: use pre-computed KV from reference layer.
            // Keys already have RoPE applied and are in [B, H, S, D] format.
            // Do NOT update the cache — the reference layer manages it.
            (shared_k, shared_v)
        } else {
            // Normal layer: compute K/V from projections
            let k_proj = self.k_proj.as_mut().expect("k_proj required for non-shared layer");
            let raw_keys = k_proj.forward(x)?;
            let raw_values = match self.v_proj.as_mut() {
                Some(v_proj) => v_proj.forward(x)?,
                None => raw_keys.clone(),
            };

            let k_norm = self.k_norm.as_mut().expect("k_norm required for non-shared layer");
            let mut keys =
                k_norm.forward(&raw_keys.reshape(&[B, L, self.n_kv_heads, self.head_dim])?)?;
            let mut values = match self.v_norm.as_ref() {
                Some(v_norm) => {
                    v_norm.forward(&raw_values.reshape(&[B, L, self.n_kv_heads, self.head_dim])?)?
                }
                None => raw_values.reshape(&[B, L, self.n_kv_heads, self.head_dim])?,
            };

            keys = keys.transpose_axes(&[0, 2, 1, 3])?;
            values = values.transpose_axes(&[0, 2, 1, 3])?;

            keys = self.rope.apply(&keys, offset)?;

            cache.update_and_fetch(keys, values)?
        };

        // For shared KV layers, the reference layer already incremented the
        // cache offset via update_and_fetch. Derive the correct query RoPE
        // offset from the KV sequence length instead.
        let rope_offset = if is_shared_kv {
            (keys.shape()[2] - L) as i32
        } else {
            offset
        };
        queries = self.rope.apply(&queries, rope_offset)?;

        let sliding_mask = match (mask, self.sliding_window) {
            (None, Some(window)) => {
                Some(create_causal_mask(L, Some(rope_offset), Some(window), None)?)
            }
            _ => None,
        };
        let sdpa_mask = match mask {
            Some(m) => Some(SdpaMask::Array(m)),
            None => match sliding_mask.as_ref() {
                Some(m) => Some(SdpaMask::Array(m)),
                None if L > 1 => Some(SdpaMask::Causal),
                None => None,
            },
        };

        let output = scaled_dot_product_attention::<C>(
            queries, keys, values, None::<C>, self.scale, sdpa_mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        if let Some(ref mut k_proj) = self.k_proj {
            k_proj.training_mode(mode);
        }
        if let Some(ref mut v_proj) = self.v_proj {
            v_proj.training_mode(mode);
        }
        self.o_proj.training_mode(mode);
        self.q_norm.training_mode(mode);
        if let Some(ref mut k_norm) = self.k_norm {
            k_norm.training_mode(mode);
        }
        self.rope.training_mode(mode);
    }
}

// ============================================================================
// Feedforward / MoE
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct DenseMlp {
    #[param]
    pub gate_proj: nn::Linear,
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,

    pub activation: GemmaActivation,
}

impl Module<&Array> for DenseMlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = self.activation.apply(&gate)?.multiply(&up)?;
        self.down_proj.forward(&activated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Router {
    pub hidden_size: i32,
    pub top_k_experts: i32,
    pub scalar_root_size: f32,

    #[param]
    pub proj: nn::Linear,
    #[param]
    pub scale: Param<Array>,
    #[param]
    pub per_expert_scale: Param<Array>,

    pub norm: UnscaledRmsNorm,
}

impl Router {
    pub fn forward(&mut self, hidden_states: &Array) -> Result<(Array, Array, Array), Exception> {
        let hidden_states = self.norm.forward(hidden_states)?;
        let scale = (&*self.scale).as_dtype(hidden_states.dtype())?;
        let hidden_states = hidden_states
            .multiply(&scale)?
            .multiply(&array!(self.scalar_root_size))?;

        let expert_scores = self
            .proj
            .forward(&hidden_states)?
            .as_dtype(Dtype::Float32)?;
        let router_probabilities = ops::softmax_axis(&expert_scores, -1, Some(true))?;

        let neg_scores = router_probabilities.negative()?;
        let partitioned_indices = ops::argpartition_axis(&neg_scores, self.top_k_experts - 1, -1)?;
        let top_k_index = partitioned_indices.index((.., ..self.top_k_experts));

        let mut top_k_weights = take_along_axis(&router_probabilities, &top_k_index, -1)?;
        let denom = top_k_weights.sum_axis(-1, true)?;
        top_k_weights = top_k_weights.divide(&denom)?;

        let per_expert_scale = take_axis(
            &(&*self.per_expert_scale).as_dtype(Dtype::Float32)?,
            &top_k_index,
            0,
        )?;
        top_k_weights = top_k_weights.multiply(&per_expert_scale)?;

        Ok((router_probabilities, top_k_weights, top_k_index))
    }

    pub fn training_mode(&mut self, mode: bool) {
        self.proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Experts {
    pub hidden_size: i32,
    pub intermediate_size: i32,

    #[param]
    pub gate_up_proj: Param<Array>,
    #[param]
    pub down_proj: Param<Array>,

    pub activation: GemmaActivation,
}

impl Experts {
    pub fn forward_topk(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
    ) -> Result<Array, Exception> {
        let hidden_dtype = hidden_states.dtype();
        let n = hidden_states.shape()[0];
        let k = top_k_index.shape()[1];

        let hidden_states = hidden_states.reshape(&[n, 1, 1, self.hidden_size])?;

        let gate_up = take_axis(&*self.gate_up_proj, top_k_index, 0)?;
        let gate_up = gate_up.transpose_axes(&[0, 1, 3, 2])?;
        let projected =
            hidden_states
                .matmul(&gate_up)?
                .reshape(&[n, k, 2 * self.intermediate_size])?;
        let split = projected.split(2, -1)?;
        let gate = self.activation.apply(&split[0])?;
        let up = &split[1];

        let activated = gate
            .multiply(up)?
            .reshape(&[n, k, 1, self.intermediate_size])?;

        let down = take_axis(&*self.down_proj, top_k_index, 0)?;
        let down = down.transpose_axes(&[0, 1, 3, 2])?;
        let expert_out = activated
            .matmul(&down)?
            .reshape(&[n, k, self.hidden_size])?
            .as_dtype(Dtype::Float32)?;

        let weighted = expert_out.multiply(&top_k_weights.index((.., .., NewAxis)))?;
        weighted.sum_axis(1, false)?.as_dtype(hidden_dtype)
    }

    pub fn training_mode(&mut self, _mode: bool) {}
}

// ============================================================================
// Decoder / model
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct DecoderLayer {
    pub layer_idx: i32,
    pub enable_moe_block: bool,

    #[param]
    pub self_attn: Attention,
    #[param]
    pub mlp: DenseMlp,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub pre_feedforward_layernorm: nn::RmsNorm,
    #[param]
    pub post_feedforward_layernorm: nn::RmsNorm,
    #[param]
    pub router: Option<Router>,
    #[param]
    pub experts: Option<Experts>,
    #[param]
    pub post_feedforward_layernorm_1: Option<nn::RmsNorm>,
    #[param]
    pub post_feedforward_layernorm_2: Option<nn::RmsNorm>,
    #[param]
    pub pre_feedforward_layernorm_2: Option<nn::RmsNorm>,
    #[param]
    pub layer_scalar: Param<Array>,

    // Per-layer embeddings (PLE) — present when hidden_size_per_layer_input > 0
    #[param]
    pub per_layer_input_gate: Option<nn::Linear>,
    #[param]
    pub per_layer_projection: Option<nn::Linear>,
    #[param]
    pub post_per_layer_input_norm: Option<nn::RmsNorm>,

    pub activation: GemmaActivation,
}

pub struct DecoderLayerInput<'a, C> {
    pub hidden_states: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut C,
    /// Shared KV from a reference layer (for KV-shared layers).
    pub shared_kv: Option<(Array, Array)>,
    /// Per-layer embedding input for this layer.
    pub per_layer_input: Option<&'a Array>,
}

impl<C> Module<DecoderLayerInput<'_, C>> for DecoderLayer
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: DecoderLayerInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let DecoderLayerInput {
            hidden_states,
            mask,
            cache,
            shared_kv,
            per_layer_input,
        } = input;

        let residual = hidden_states.clone();

        let attn_in = self.input_layernorm.forward(hidden_states)?;
        let attn_out = self.self_attn.forward(AttentionInput {
            x: &attn_in,
            mask,
            cache,
            shared_kv,
        })?;
        let attn_out = self.post_attention_layernorm.forward(&attn_out)?;
        let mut hidden_states = residual.add(&attn_out)?;

        let residual = hidden_states.clone();
        let dense_hidden = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let mut ff_out = self.mlp.forward(&dense_hidden)?;

        if self.enable_moe_block {
            let mlp_branch = self
                .post_feedforward_layernorm_1
                .as_mut()
                .expect("MoE layer missing post_feedforward_layernorm_1")
                .forward(&ff_out)?;

            let hidden_states_flat = residual.reshape(&[-1, residual.shape()[2]])?;
            let (_, top_k_weights, top_k_index) = self
                .router
                .as_mut()
                .expect("MoE layer missing router")
                .forward(&hidden_states_flat)?;
            let moe_in = self
                .pre_feedforward_layernorm_2
                .as_mut()
                .expect("MoE layer missing pre_feedforward_layernorm_2")
                .forward(&hidden_states_flat)?;
            let moe_out = self
                .experts
                .as_mut()
                .expect("MoE layer missing experts")
                .forward_topk(&moe_in, &top_k_index, &top_k_weights)?;
            let moe_out = moe_out.reshape(&residual.shape())?;
            let moe_out = self
                .post_feedforward_layernorm_2
                .as_mut()
                .expect("MoE layer missing post_feedforward_layernorm_2")
                .forward(&moe_out)?;

            ff_out = mlp_branch.add(&moe_out)?;
        }

        let ff_out = self.post_feedforward_layernorm.forward(&ff_out)?;
        hidden_states = residual.add(&ff_out)?;

        // Per-layer embeddings (PLE): gated residual from auxiliary embedding
        if let (Some(gate), Some(proj), Some(norm), Some(ple_input)) = (
            self.per_layer_input_gate.as_mut(),
            self.per_layer_projection.as_mut(),
            self.post_per_layer_input_norm.as_mut(),
            per_layer_input,
        ) {
            let residual = hidden_states.clone();
            let gated = self.activation.apply(&gate.forward(&hidden_states)?)?;
            let gated = gated.multiply(ple_input)?;
            let projected = norm.forward(&proj.forward(&gated)?)?;
            hidden_states = residual.add(&projected)?;
        }

        // Layer scalar applied AFTER PLE (HF applies it as the final operation)
        hidden_states = hidden_states.multiply(&*self.layer_scalar)?;

        Ok(hidden_states)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
        self.pre_feedforward_layernorm.training_mode(mode);
        self.post_feedforward_layernorm.training_mode(mode);
        if let Some(ref mut router) = self.router {
            router.training_mode(mode);
        }
        if let Some(ref mut experts) = self.experts {
            experts.training_mode(mode);
        }
        if let Some(ref mut norm) = self.post_feedforward_layernorm_1 {
            norm.training_mode(mode);
        }
        if let Some(ref mut norm) = self.post_feedforward_layernorm_2 {
            norm.training_mode(mode);
        }
        if let Some(ref mut norm) = self.pre_feedforward_layernorm_2 {
            norm.training_mode(mode);
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct LanguageModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,
    pub hidden_size_per_layer_input: i32,

    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,

    // Per-layer embeddings (PLE)
    #[param]
    pub embed_tokens_per_layer: Option<nn::Embedding>,
    #[param]
    pub per_layer_model_projection: Option<nn::Linear>,
    #[param]
    pub per_layer_projection_norm: Option<nn::RmsNorm>,

    /// Maps each layer index to its cache slot. Shared layers point to the
    /// same slot as their reference layer.
    pub kv_cache_map: Vec<usize>,
    /// For layers that store full-length KV for sharing, (layer_idx, cache_slot).
    pub kv_store_layers: HashSet<usize>,

    pub has_moe: bool,
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

    #[allow(non_snake_case)]
    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;

        assert!(
            !cache.is_empty(),
            "Cache must be pre-allocated with init_cache() before calling Gemma4 forward",
        );

        let mut hidden_states = self.embed_tokens.forward(inputs)?;

        // Gemma models scale embeddings by sqrt(hidden_size), cast through BF16 for
        // numerical fidelity with Google's reference implementation.
        let hidden_dim = hidden_states.shape()[2] as f32;
        let scale = Array::from(hidden_dim.sqrt())
            .as_dtype(Dtype::Bfloat16)?
            .as_dtype(hidden_states.dtype())?;
        hidden_states = hidden_states.multiply(&scale)?;

        // Compute per-layer embeddings (PLE) if enabled
        let per_layer_inputs = if self.hidden_size_per_layer_input > 0 {
            let ple_dim = self.hidden_size_per_layer_input;
            let num_layers = self.num_hidden_layers;

            // Auxiliary embedding: embed_tokens_per_layer(input_ids)
            let ple_embed = self
                .embed_tokens_per_layer
                .as_mut()
                .expect("PLE embed_tokens_per_layer missing")
                .forward(inputs)?;
            // Scale by sqrt(ple_dim), cast through BF16
            let ple_scale = Array::from((ple_dim as f32).sqrt())
                .as_dtype(Dtype::Bfloat16)?
                .as_dtype(ple_embed.dtype())?;
            let ple_embed = ple_embed.multiply(&ple_scale)?;
            let B = ple_embed.shape()[0];
            let L = ple_embed.shape()[1];
            let ple_embed = ple_embed.reshape(&[B, L, num_layers, ple_dim])?;

            // Model projection: project main embeddings to per-layer space
            let proj = self
                .per_layer_model_projection
                .as_mut()
                .expect("PLE per_layer_model_projection missing")
                .forward(&hidden_states)?;
            let proj_scale = array!((hidden_dim).powf(-0.5));
            let proj = proj.multiply(&proj_scale)?;
            let proj = proj.reshape(&[B, L, num_layers, ple_dim])?;
            let proj = self
                .per_layer_projection_norm
                .as_mut()
                .expect("PLE per_layer_projection_norm missing")
                .forward(&proj)?;

            // Combine: (projection + embedding) * 2^-0.5
            let combined = proj.add(&ple_embed)?;
            let input_scale = array!(std::f32::consts::FRAC_1_SQRT_2);
            Some(combined.multiply(&input_scale)?)
        } else {
            None
        };

        // During prefill with MoE, eval each layer to free expert gather intermediates.
        let is_prefill = inputs.shape()[1] > 1;

        // Track shared KV: layers that store full-length KV for later sharing
        let mut shared_kv_store: HashMap<usize, (Array, Array)> = HashMap::new();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let cache_slot = self.kv_cache_map[i];
            let is_shared = layer.self_attn.is_kv_shared;

            // For KV-shared layers, retrieve pre-computed KV from the store
            let shared_kv = if is_shared {
                shared_kv_store.get(&cache_slot).cloned()
            } else {
                None
            };

            // Extract per-layer input slice for this layer
            let ple_slice = per_layer_inputs
                .as_ref()
                .map(|p| p.index((.., .., i as i32, ..)));

            hidden_states = layer.forward(DecoderLayerInput {
                hidden_states: &hidden_states,
                mask,
                cache: &mut cache[cache_slot],
                shared_kv,
                per_layer_input: ple_slice.as_ref(),
            })?;

            // If this layer stores KV for sharing, snapshot the sliced KV
            if self.kv_store_layers.contains(&i) {
                if let Some(kv) = cache[cache_slot].current_kv() {
                    shared_kv_store.insert(cache_slot, kv);
                }
            }

            if is_prefill && self.has_moe {
                mlx_rs::transforms::eval([&hidden_states])?;
            }
        }

        self.norm.forward(&hidden_states)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <DecoderLayer as Module<DecoderLayerInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Model {
    pub args: Gemma4TextConfig,

    #[param]
    pub model: LanguageModel,
    #[param]
    pub lm_head: Option<nn::Linear>,
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self.model.forward(input)?;
        let mut logits = match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&out)?,
            None => self.model.embed_tokens.as_linear(&out)?,
        };
        if let Some(softcap) = self.args.final_logit_softcapping {
            let cap = array!(softcap);
            logits = ops::tanh(&logits.divide(&cap)?)?.multiply(&cap)?;
        }
        Ok(logits)
    }

    fn training_mode(&mut self, mode: bool) {
        <LanguageModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(ref mut lm_head) = self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

// ============================================================================
// Loading
// ============================================================================

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<Gemma4Config, Error> {
    let model_args_filename = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(model_args_filename)?;
    serde_json::from_reader(file).map_err(Into::into)
}

fn keep_language_weight(key: &str) -> bool {
    key.starts_with("model.language_model.")
        || matches!(
            key,
            "lm_head.weight" | "model.lm_head.weight" | "model.language_model.lm_head.weight"
        )
}

fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>, Error> {
    let weights_index = model_dir.join("model.safetensors.index.json");
    let single_file = model_dir.join("model.safetensors");

    let weight_files: Vec<std::path::PathBuf> = if weights_index.exists() {
        // Sharded: read index to find relevant shard files
        let json = std::fs::read_to_string(&weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        weight_map
            .weight_map
            .iter()
            .filter(|(key, _)| keep_language_weight(key))
            .map(|(_, file)| model_dir.join(file))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    } else if single_file.exists() {
        // Single file: load it directly
        vec![single_file]
    } else {
        return Err(Error::Model(
            "No model.safetensors or model.safetensors.index.json found".to_string(),
        ));
    };

    let mut all_weights = HashMap::new();
    for weights_filename in weight_files {
        let loaded = Array::load_safetensors(&weights_filename)?;
        for (key, value) in loaded {
            if keep_language_weight(&key) {
                all_weights.insert(key, value);
            }
        }
    }

    Ok(all_weights)
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::Model(format!("Weight not found: {key}")))
}

fn get_weight_optional(weights: &HashMap<String, Array>, key: &str) -> Option<Array> {
    weights.get(key).cloned()
}

fn get_first_weight(weights: &HashMap<String, Array>, keys: &[&str]) -> Option<Array> {
    keys.iter()
        .find_map(|key| get_weight_optional(weights, key))
}

fn make_linear(weight: Array) -> nn::Linear {
    nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(None::<Array>),
    }
}

fn make_rms_norm(weight: Array, eps: f32) -> nn::RmsNorm {
    // HF Gemma4 weights are stored as actual scale values (initialized to ones),
    // NOT as zero-centered offsets. Use them directly.
    nn::RmsNorm {
        weight: Param::new(weight),
        eps,
    }
}

fn build_rope(config: &Gemma4TextConfig, layer_type: LayerType, head_dim: i32) -> GemmaRope {
    let rope = match layer_type {
        LayerType::SlidingAttention => &config.rope_parameters.sliding_attention,
        LayerType::FullAttention => &config.rope_parameters.full_attention,
    };

    match rope.rope_type.as_str() {
        "default" => GemmaRope::Standard(
            nn::RopeBuilder::new(head_dim)
                .base(rope.rope_theta)
                .traditional(false)
                .build()
                .expect("RopeBuilder is infallible for known Gemma4 dimensions"),
        ),
        "proportional" => GemmaRope::Proportional(ProportionalRope::new(
            head_dim,
            rope.rope_theta,
            rope.partial_rotary_factor,
        )),
        other => panic!("Unsupported Gemma4 rope type: {other}"),
    }
}

/// Load model with config overrides applied before building.
/// The `overrides` closure receives the text config for mutation.
pub fn load_model_with_overrides(
    model_dir: impl AsRef<Path>,
    overrides: impl FnOnce(&mut Gemma4TextConfig),
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let config = get_model_args(model_dir)?;
    let mut args = config.text_config.clone();
    overrides(&mut args);
    load_model_inner(model_dir, config, args)
}

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let config = get_model_args(model_dir)?;
    let args = config.text_config.clone();
    load_model_inner(model_dir, config, args)
}

fn load_model_inner(model_dir: &Path, config: Gemma4Config, args: Gemma4TextConfig) -> Result<Model, Error> {

    if args.use_double_wide_mlp {
        return Err(Error::Model(
            "Gemma4 double-wide MLP is not implemented yet".to_string(),
        ));
    }
    for (layer_name, rope) in [
        ("sliding_attention", &args.rope_parameters.sliding_attention),
        ("full_attention", &args.rope_parameters.full_attention),
    ] {
        if !matches!(rope.rope_type.as_str(), "default" | "proportional") {
            return Err(Error::Model(format!(
                "Unsupported Gemma4 {layer_name} rope type: {}",
                rope.rope_type
            )));
        }
    }

    let activation = GemmaActivation::from_name(&args.hidden_activation)?;
    let weights = load_all_weights(model_dir)?;

    // Infer global_head_dim from weight shapes when config has null/0.
    // Full attention layers have larger q_proj (more dims per head).
    let mut args = args;
    if args.global_head_dim == 0 {
        // Find the first full_attention layer and infer head_dim from q_proj shape.
        for (i, lt) in args.layer_types.iter().enumerate() {
            if lt == "full_attention" {
                let key = format!("model.language_model.layers.{i}.self_attn.q_proj.weight");
                if let Some(w) = weights.get(&key) {
                    let q_out = w.shape()[0]; // [n_heads * head_dim, hidden_size]
                    args.global_head_dim = q_out / args.num_attention_heads;
                    break;
                }
            }
        }
        // Fallback: use head_dim
        if args.global_head_dim == 0 {
            args.global_head_dim = args.head_dim;
        }
    }

    // Compute KV sharing: which layers share KV from earlier layers
    let num_layers = args.num_hidden_layers as usize;
    let first_shared = num_layers.saturating_sub(args.num_kv_shared_layers as usize);
    let non_shared_types: Vec<&str> = args.layer_types[..first_shared]
        .iter()
        .map(|s| s.as_str())
        .collect();

    // Build cache mapping: each layer → cache slot index
    // Non-shared layers get sequential unique slots.
    // Shared layers point to the slot of the last non-shared layer of the same type.
    let mut kv_cache_map = Vec::with_capacity(num_layers);
    let mut kv_store_layers = HashSet::new();
    let mut next_slot = 0usize;
    let mut layer_to_slot: Vec<usize> = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        if i < first_shared {
            // Non-shared layer: gets its own cache slot
            layer_to_slot.push(next_slot);
            kv_cache_map.push(next_slot);
            next_slot += 1;
        } else {
            // Shared layer: find the last non-shared layer of the same type
            let layer_type_str = args.layer_types[i].as_str();
            let ref_idx = non_shared_types
                .iter()
                .rposition(|t| *t == layer_type_str)
                .ok_or_else(|| {
                    Error::Model(format!(
                        "No reference layer of type {layer_type_str} for shared layer {i}"
                    ))
                })?;
            kv_cache_map.push(layer_to_slot[ref_idx]);
            // Mark the reference layer as one that stores full-length KV
            kv_store_layers.insert(ref_idx);
        }
    }
    let num_cache_slots = next_slot;

    let mut layers = Vec::with_capacity(num_layers);
    for layer_idx_u in 0..num_layers {
        let layer_idx = layer_idx_u as i32;
        let layer_type = LayerType::from_config(&args.layer_types[layer_idx_u])?;
        let layer_prefix = format!("model.language_model.layers.{layer_idx}");
        let is_kv_shared = layer_idx_u >= first_shared && args.num_kv_shared_layers > 0;

        let head_dim = if !layer_type.is_sliding() && args.global_head_dim > 0 {
            args.global_head_dim
        } else {
            args.head_dim
        };
        // For E4B: num_global_key_value_heads is null/0, use num_key_value_heads for all
        let n_kv_heads = if !layer_type.is_sliding() && args.num_global_key_value_heads > 0 {
            args.num_global_key_value_heads
        } else {
            args.num_key_value_heads
        };

        let attention = Attention {
            layer_idx,
            layer_type,
            n_heads: args.num_attention_heads,
            n_kv_heads,
            head_dim,
            scale: 1.0,
            sliding_window: if layer_type.is_sliding() {
                Some(args.sliding_window)
            } else {
                None
            },
            is_kv_shared,
            q_proj: make_linear(get_weight(
                &weights,
                &format!("{layer_prefix}.self_attn.q_proj.weight"),
            )?),
            // KV-shared layers don't have k/v projection weights
            k_proj: if is_kv_shared {
                None
            } else {
                Some(make_linear(get_weight(
                    &weights,
                    &format!("{layer_prefix}.self_attn.k_proj.weight"),
                )?))
            },
            v_proj: if is_kv_shared {
                None
            } else {
                get_weight_optional(
                    &weights,
                    &format!("{layer_prefix}.self_attn.v_proj.weight"),
                )
                .map(make_linear)
            },
            o_proj: make_linear(get_weight(
                &weights,
                &format!("{layer_prefix}.self_attn.o_proj.weight"),
            )?),
            q_norm: make_rms_norm(
                get_weight(&weights, &format!("{layer_prefix}.self_attn.q_norm.weight"))?,
                args.rms_norm_eps,
            ),
            k_norm: if is_kv_shared {
                None
            } else {
                Some(make_rms_norm(
                    get_weight(&weights, &format!("{layer_prefix}.self_attn.k_norm.weight"))?,
                    args.rms_norm_eps,
                ))
            },
            rope: build_rope(&args, layer_type, head_dim),
            v_norm: if is_kv_shared {
                None
            } else {
                Some(UnscaledRmsNorm::new(args.rms_norm_eps))
            },
        };

        let mlp = DenseMlp {
            gate_proj: make_linear(get_weight(
                &weights,
                &format!("{layer_prefix}.mlp.gate_proj.weight"),
            )?),
            up_proj: make_linear(get_weight(
                &weights,
                &format!("{layer_prefix}.mlp.up_proj.weight"),
            )?),
            down_proj: make_linear(get_weight(
                &weights,
                &format!("{layer_prefix}.mlp.down_proj.weight"),
            )?),
            activation,
        };

        let (router, experts, post_ff_ln_1, post_ff_ln_2, pre_ff_ln_2) = if args.enable_moe_block {
            let router = Router {
                hidden_size: args.hidden_size,
                top_k_experts: args.top_k_experts,
                scalar_root_size: (args.hidden_size as f32).sqrt().recip(),
                proj: make_linear(get_weight(
                    &weights,
                    &format!("{layer_prefix}.router.proj.weight"),
                )?),
                scale: Param::new(get_weight(
                    &weights,
                    &format!("{layer_prefix}.router.scale"),
                )?),
                per_expert_scale: Param::new(get_weight(
                    &weights,
                    &format!("{layer_prefix}.router.per_expert_scale"),
                )?),
                norm: UnscaledRmsNorm::new(args.rms_norm_eps),
            };

            let experts = Experts {
                hidden_size: args.hidden_size,
                intermediate_size: args.moe_intermediate_size,
                gate_up_proj: Param::new(get_weight(
                    &weights,
                    &format!("{layer_prefix}.experts.gate_up_proj"),
                )?),
                down_proj: Param::new(get_weight(
                    &weights,
                    &format!("{layer_prefix}.experts.down_proj"),
                )?),
                activation,
            };

            (
                Some(router),
                Some(experts),
                Some(make_rms_norm(
                    get_weight(
                        &weights,
                        &format!("{layer_prefix}.post_feedforward_layernorm_1.weight"),
                    )?,
                    args.rms_norm_eps,
                )),
                Some(make_rms_norm(
                    get_weight(
                        &weights,
                        &format!("{layer_prefix}.post_feedforward_layernorm_2.weight"),
                    )?,
                    args.rms_norm_eps,
                )),
                Some(make_rms_norm(
                    get_weight(
                        &weights,
                        &format!("{layer_prefix}.pre_feedforward_layernorm_2.weight"),
                    )?,
                    args.rms_norm_eps,
                )),
            )
        } else {
            (None, None, None, None, None)
        };

        // PLE per-layer weights
        let per_layer_input_gate = if args.hidden_size_per_layer_input > 0 {
            Some(make_linear(get_weight(
                &weights,
                &format!("{layer_prefix}.per_layer_input_gate.weight"),
            )?))
        } else {
            None
        };
        let per_layer_projection = if args.hidden_size_per_layer_input > 0 {
            Some(make_linear(get_weight(
                &weights,
                &format!("{layer_prefix}.per_layer_projection.weight"),
            )?))
        } else {
            None
        };
        let post_per_layer_input_norm = if args.hidden_size_per_layer_input > 0 {
            Some(make_rms_norm(
                get_weight(
                    &weights,
                    &format!("{layer_prefix}.post_per_layer_input_norm.weight"),
                )?,
                args.rms_norm_eps,
            ))
        } else {
            None
        };

        layers.push(DecoderLayer {
            layer_idx,
            enable_moe_block: args.enable_moe_block,
            self_attn: attention,
            mlp,
            input_layernorm: make_rms_norm(
                get_weight(&weights, &format!("{layer_prefix}.input_layernorm.weight"))?,
                args.rms_norm_eps,
            ),
            post_attention_layernorm: make_rms_norm(
                get_weight(
                    &weights,
                    &format!("{layer_prefix}.post_attention_layernorm.weight"),
                )?,
                args.rms_norm_eps,
            ),
            pre_feedforward_layernorm: make_rms_norm(
                get_weight(
                    &weights,
                    &format!("{layer_prefix}.pre_feedforward_layernorm.weight"),
                )?,
                args.rms_norm_eps,
            ),
            post_feedforward_layernorm: make_rms_norm(
                get_weight(
                    &weights,
                    &format!("{layer_prefix}.post_feedforward_layernorm.weight"),
                )?,
                args.rms_norm_eps,
            ),
            router,
            experts,
            post_feedforward_layernorm_1: post_ff_ln_1,
            post_feedforward_layernorm_2: post_ff_ln_2,
            pre_feedforward_layernorm_2: pre_ff_ln_2,
            layer_scalar: Param::new(get_weight(
                &weights,
                &format!("{layer_prefix}.layer_scalar"),
            )?),
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            activation,
        });
    }

    // PLE model-level weights
    let embed_tokens_per_layer = if args.hidden_size_per_layer_input > 0 {
        Some(nn::Embedding {
            weight: Param::new(get_weight(
                &weights,
                "model.language_model.embed_tokens_per_layer.weight",
            )?),
        })
    } else {
        None
    };
    let per_layer_model_projection = if args.hidden_size_per_layer_input > 0 {
        Some(make_linear(get_weight(
            &weights,
            "model.language_model.per_layer_model_projection.weight",
        )?))
    } else {
        None
    };
    let per_layer_projection_norm = if args.hidden_size_per_layer_input > 0 {
        Some(make_rms_norm(
            get_weight(
                &weights,
                "model.language_model.per_layer_projection_norm.weight",
            )?,
            args.rms_norm_eps,
        ))
    } else {
        None
    };

    let language_model = LanguageModel {
        vocab_size: args.vocab_size,
        num_hidden_layers: args.num_hidden_layers,
        hidden_size_per_layer_input: args.hidden_size_per_layer_input,
        embed_tokens: nn::Embedding {
            weight: Param::new(get_weight(
                &weights,
                "model.language_model.embed_tokens.weight",
            )?),
        },
        layers,
        norm: make_rms_norm(
            get_weight(&weights, "model.language_model.norm.weight")?,
            args.rms_norm_eps,
        ),
        embed_tokens_per_layer,
        per_layer_model_projection,
        per_layer_projection_norm,
        kv_cache_map,
        kv_store_layers,
        has_moe: args.enable_moe_block,
    };

    let lm_head = get_first_weight(
        &weights,
        &[
            "lm_head.weight",
            "model.lm_head.weight",
            "model.language_model.lm_head.weight",
        ],
    )
    .map(make_linear);
    if lm_head.is_none() && !args.tie_word_embeddings && !config.tie_word_embeddings {
        return Err(Error::Model(
            "Gemma4 lm_head weights are missing and embeddings are not tied".to_string(),
        ));
    }

    Ok(Model {
        args,
        model: language_model,
        lm_head,
    })
}

// ============================================================================
// Generation
// ============================================================================

pub fn init_cache<C: KeyValueCache + Default>(num_layers: usize) -> Vec<C> {
    (0..num_layers).map(|_| C::default()).collect()
}

pub struct Generate<'a, C, S: Sampler = DefaultSampler> {
    model: &'a mut Model,
    cache: &'a mut Vec<C>,
    sampler: S,
    temp: f32,
    state: GenerateState<'a>,
    token_count: usize,
}

pub enum GenerateState<'a> {
    Prefill { prompt_token: &'a Array },
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

impl<'a, C> Generate<'a, C, DefaultSampler>
where
    C: KeyValueCache + Default,
{
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<C>,
        temp: f32,
        prompt_token: &'a Array,
    ) -> Self {
        Self::with_sampler(model, cache, temp, prompt_token, DefaultSampler)
    }
}

impl<'a, C, S: Sampler> Generate<'a, C, S>
where
    C: KeyValueCache + Default,
{
    pub fn with_sampler(
        model: &'a mut Model,
        cache: &'a mut Vec<C>,
        temp: f32,
        prompt_token: &'a Array,
        sampler: S,
    ) -> Self {
        if cache.is_empty() {
            // Number of cache slots may be less than num_layers due to KV sharing
            let num_slots = *model.model.kv_cache_map.iter().max().unwrap_or(&0) + 1;
            *cache = init_cache(num_slots);
        }

        Self {
            model,
            cache,
            sampler,
            temp,
            state: GenerateState::Prefill { prompt_token },
            token_count: 0,
        }
    }

    fn compute_next(&mut self, y: &Array) -> Result<Array, Exception> {
        let inputs = y.index((.., NewAxis));
        let input = ModelInput {
            inputs: &inputs,
            mask: None,
            cache: self.cache,
        };
        let logits = self.model.forward(input)?;
        // Select last token before sampling to keep output shape [B], not [B, 1].
        // Without this, the shape accumulates an extra dim each iteration,
        // causing hidden_dim to be computed as 1 instead of hidden_size.
        self.sampler.sample(&logits.index((.., -1, ..)), self.temp)
    }
}

impl<'a, C, S: Sampler> Iterator for Generate<'a, C, S>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        let state = std::mem::replace(&mut self.state, GenerateState::Done);

        match state {
            GenerateState::Prefill { prompt_token } => {
                // Chunked prefill: process the prompt in chunks to limit peak
                // GPU memory from MoE expert gather intermediates. Each chunk
                // updates the KV cache, so attention can look back at all
                // previously processed tokens.
                const PREFILL_CHUNK: i32 = 32;
                let seq_len = prompt_token.shape()[1];

                if seq_len > PREFILL_CHUNK {
                    let mut pos = 0;
                    while pos < seq_len {
                        let end = (pos + PREFILL_CHUNK).min(seq_len);
                        let chunk = prompt_token.index((.., pos..end));
                        let input = ModelInput {
                            inputs: &chunk,
                            mask: None,
                            cache: self.cache,
                        };
                        let logits = tri!(self.model.forward(input));
                        // Eval to free intermediates before next chunk
                        tri!(mlx_rs::transforms::eval([&logits]));
                        pos = end;

                        // On last chunk, sample from the final logits
                        if pos >= seq_len {
                            let y = tri!(self.sampler.sample(&logits.index((.., -1, ..)), self.temp));
                            tri!(mlx_rs::transforms::eval([&y]));

                            let next_y = tri!(self.compute_next(&y));
                            tri!(mlx_rs::transforms::async_eval([&next_y]));

                            self.state = GenerateState::Pipelined { current_y: next_y };
                            return Some(Ok(y));
                        }
                    }
                    unreachable!()
                } else {
                    let input = ModelInput {
                        inputs: prompt_token,
                        mask: None,
                        cache: self.cache,
                    };
                    let logits = tri!(self.model.forward(input));
                    let y = tri!(self.sampler.sample(&logits.index((.., -1, ..)), self.temp));

                    tri!(mlx_rs::transforms::async_eval([&y]));
                    tri!(mlx_rs::transforms::eval([&y]));

                    let next_y = tri!(self.compute_next(&y));
                    tri!(mlx_rs::transforms::async_eval([&next_y]));

                    self.state = GenerateState::Pipelined { current_y: next_y };
                    Some(Ok(y))
                }
            }
            GenerateState::Pipelined { current_y } => {
                let next_y = tri!(self.compute_next(&current_y));
                tri!(mlx_rs::transforms::async_eval([&next_y]));

                // Per-layer cache eval materializes lazy index chains from
                // update_and_fetch so the next forward pass starts clean.
                for c in self.cache.iter() {
                    tri!(c.eval());
                }

                // Periodically release completed computation graph memory.
                self.token_count += 1;
                if self.token_count % 256 == 0 {
                    unsafe {
                        mlx_sys::mlx_clear_cache();
                    }
                }

                self.state = GenerateState::Pipelined { current_y: next_y };
                Some(Ok(current_y))
            }
            GenerateState::Done => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_text_config() -> Gemma4TextConfig {
        Gemma4TextConfig {
            attention_bias: false,
            attention_dropout: 0.0,
            attention_k_eq_v: true,
            enable_moe_block: true,
            global_head_dim: 16,
            head_dim: 4,
            hidden_activation: "gelu_pytorch_tanh".to_string(),
            hidden_size: 16,
            hidden_size_per_layer_input: 0,
            intermediate_size: 32,
            layer_types: vec!["full_attention".to_string()],
            max_position_embeddings: 128,
            moe_intermediate_size: 8,
            num_attention_heads: 2,
            num_experts: 4,
            num_global_key_value_heads: 1,
            num_hidden_layers: 1,
            num_kv_shared_layers: 0,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_parameters: RopeParameters {
                sliding_attention: RopeSpec {
                    rope_theta: 10_000.0,
                    rope_type: "default".to_string(),
                    partial_rotary_factor: 1.0,
                },
                full_attention: RopeSpec {
                    rope_theta: 1_000_000.0,
                    rope_type: "proportional".to_string(),
                    partial_rotary_factor: 0.25,
                },
            },
            sliding_window: 32,
            tie_word_embeddings: true,
            top_k_experts: 2,
            use_double_wide_mlp: false,
            vocab_size: 256,
            vocab_size_per_layer_input: 0,
            final_logit_softcapping: Some(30.0),
        }
    }

    fn make_test_input(seq_len: i32, head_dim: i32) -> Array {
        let mut values = vec![0.0_f32; seq_len as usize * head_dim as usize];
        let row_start = ((seq_len - 1) * head_dim) as usize;
        for i in 0..head_dim as usize {
            values[row_start + i] = (i + 1) as f32;
        }
        Array::from_slice(&values, &[1, seq_len, head_dim])
    }

    fn make_blnh_test_input(seq_len: i32, n_heads: i32, head_dim: i32) -> Array {
        let mut values = Vec::with_capacity((seq_len * n_heads * head_dim) as usize);
        for pos in 0..seq_len {
            for head in 0..n_heads {
                for dim in 0..head_dim {
                    values.push(((pos + 1) * 100 + (head + 1) * 10 + dim + 1) as f32);
                }
            }
        }
        Array::from_slice(&values, &[1, seq_len, n_heads, head_dim])
    }

    fn rope_angles(head_dim: i32, partial_rotary_factor: f32) -> i32 {
        let rope_angles = (partial_rotary_factor * head_dim as f32 / 2.0).floor() as i32;
        rope_angles.clamp(0, head_dim / 2)
    }

    fn hf_reference_proportional_rope_blnh(
        input: &Array,
        head_dim: i32,
        theta: f32,
        partial_rotary_factor: f32,
        offset: i32,
    ) -> Array {
        let shape = input.shape();
        let batch = shape[0] as usize;
        let seq_len = shape[1] as usize;
        let n_heads = shape[2] as usize;
        let head_dim_usize = head_dim as usize;
        let half_dim = head_dim_usize / 2;
        let rope_angles = rope_angles(head_dim, partial_rotary_factor) as usize;

        let mut inv_freq = vec![0.0_f32; half_dim];
        for (i, freq) in inv_freq.iter_mut().enumerate().take(rope_angles) {
            *freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
        }

        let input = input.as_slice::<f32>();
        let mut output = vec![0.0_f32; input.len()];
        for b in 0..batch {
            for pos in 0..seq_len {
                let position = (offset as usize + pos) as f32;
                for head in 0..n_heads {
                    let base = ((b * seq_len + pos) * n_heads + head) * head_dim_usize;
                    for lane in 0..half_dim {
                        let angle = position * inv_freq[lane];
                        let cos = angle.cos();
                        let sin = angle.sin();
                        let x1 = input[base + lane];
                        let x2 = input[base + half_dim + lane];
                        output[base + lane] = x1 * cos - x2 * sin;
                        output[base + half_dim + lane] = x2 * cos + x1 * sin;
                    }
                }
            }
        }

        Array::from_slice(&output, shape)
    }

    fn max_abs_diff(lhs: &Array, rhs: &Array) -> f32 {
        lhs.subtract(rhs)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>()
    }

    fn noop_lane_diff(
        input: &Array,
        result: &Array,
        token_idx: i32,
        head_dim: i32,
        partial_rotary_factor: f32,
    ) -> f32 {
        let half_dim = head_dim / 2;
        let rope_angles = rope_angles(head_dim, partial_rotary_factor);

        let input_first = input.index((0, token_idx, rope_angles..half_dim));
        let result_first = result.index((0, token_idx, rope_angles..half_dim));
        let input_second = input.index((0, token_idx, (half_dim + rope_angles)..head_dim));
        let result_second = result.index((0, token_idx, (half_dim + rope_angles)..head_dim));

        let first_diff = result_first
            .subtract(&input_first)
            .unwrap()
            .abs()
            .unwrap()
            .sum(None)
            .unwrap()
            .item::<f32>();
        let second_diff = result_second
            .subtract(&input_second)
            .unwrap()
            .abs()
            .unwrap()
            .sum(None)
            .unwrap()
            .item::<f32>();

        first_diff + second_diff
    }

    #[test]
    fn full_attention_prefill_should_preserve_nope_lanes() {
        let config = test_text_config();
        let mut rope = build_rope(&config, LayerType::FullAttention, config.global_head_dim);
        let input = make_test_input(2, config.global_head_dim);
        let result = rope.apply(&input, 0).unwrap();

        let diff = noop_lane_diff(
            &input,
            &result,
            1,
            config.global_head_dim,
            config.rope_parameters.full_attention.partial_rotary_factor,
        );

        assert!(
            diff < 1e-6,
            "expected proportional RoPE to leave Gemma4 nope lanes unchanged during prefill, diff={diff}"
        );
    }

    #[test]
    fn full_attention_decode_should_preserve_nope_lanes_with_offset() {
        let config = test_text_config();
        let mut rope = build_rope(&config, LayerType::FullAttention, config.global_head_dim);
        let input = make_test_input(1, config.global_head_dim);
        let result = rope.apply(&input, 7).unwrap();

        let diff = noop_lane_diff(
            &input,
            &result,
            0,
            config.global_head_dim,
            config.rope_parameters.full_attention.partial_rotary_factor,
        );

        assert!(
            diff < 1e-6,
            "expected proportional RoPE to leave Gemma4 nope lanes unchanged during decode, diff={diff}"
        );
    }

    #[test]
    fn full_attention_prefill_should_match_hf_reference_on_blnh_layout() {
        const HF_ROPE_TOLERANCE: f32 = 5e-5;

        let config = test_text_config();
        let mut rope = build_rope(&config, LayerType::FullAttention, config.global_head_dim);
        let input = make_blnh_test_input(3, config.num_attention_heads, config.global_head_dim);
        let expected = hf_reference_proportional_rope_blnh(
            &input,
            config.global_head_dim,
            config.rope_parameters.full_attention.rope_theta,
            config.rope_parameters.full_attention.partial_rotary_factor,
            0,
        );
        let result = rope
            .apply_with_layout(&input, 0, RotaryLayout::BatchSeqHeadsDim)
            .unwrap();

        let diff = max_abs_diff(&result, &expected);
        assert!(
            diff < HF_ROPE_TOLERANCE,
            "expected BLNH proportional RoPE to match HF reference during prefill, diff={diff}"
        );
    }

    #[test]
    fn full_attention_decode_should_match_hf_reference_on_blnh_layout() {
        const HF_ROPE_TOLERANCE: f32 = 5e-5;

        let config = test_text_config();
        let mut rope = build_rope(&config, LayerType::FullAttention, config.global_head_dim);
        let input = make_blnh_test_input(1, config.num_attention_heads, config.global_head_dim);
        let expected = hf_reference_proportional_rope_blnh(
            &input,
            config.global_head_dim,
            config.rope_parameters.full_attention.rope_theta,
            config.rope_parameters.full_attention.partial_rotary_factor,
            11,
        );
        let result = rope
            .apply_with_layout(&input, 11, RotaryLayout::BatchSeqHeadsDim)
            .unwrap();

        let diff = max_abs_diff(&result, &expected);
        assert!(
            diff < HF_ROPE_TOLERANCE,
            "expected BLNH proportional RoPE to match HF reference during decode, diff={diff}"
        );
    }
}
