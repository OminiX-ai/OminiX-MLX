//! Qwen3-MoE model implementation
//!
//! Qwen3 with Mixture of Experts (MoE) support.
//! Uses top-k routing with optional normalization.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    argmax_axis, array,
    builder::Builder,
    categorical,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters as ModuleParametersTrait, ModuleParametersExt, Param},
    nn,
    ops::{
        self,
        indexing::{IndexOp, NewAxis, take_along_axis, take_axis},
    },
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use mlx_rs_core::{
    KeyValueCache,
    Error,
    fused_swiglu,
    create_attention_mask,
    initialize_rope,
    FloatOrString,
    AttentionMask,
    SdpaMask,
};

/// Quantization configuration for the model
#[derive(Debug, Clone, Deserialize, Default)]
pub struct QuantizationConfig {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
}

fn default_group_size() -> i32 { 64 }
fn default_bits() -> i32 { 4 }

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rope_theta: f32,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: i32,
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,

    // MoE specific
    pub num_experts: i32,
    pub num_experts_per_tok: i32,
    #[serde(default)]
    pub decoder_sparse_step: i32,
    #[serde(default)]
    pub mlp_only_layers: Vec<i32>,
    pub moe_intermediate_size: i32,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,

    /// Quantization config (present for quantized models)
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

fn default_tie_word_embeddings() -> bool { false }
fn default_norm_topk_prob() -> bool { false }

/// Qwen3-MoE Attention with q/k normalization
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub scale: f32,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: nn::RmsNorm,
    #[param]
    pub rope: nn::Rope,
}

impl Attention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(false)
            .build()?;

        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;

        let rope = initialize_rope(
            head_dim,
            args.rope_theta,
            false,
            &args.rope_scaling,
            args.max_position_embeddings,
        )?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            scale,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            q_norm,
            k_norm,
            rope,
        })
    }
}

pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
}

impl<C> Module<AttentionInput<'_, C>> for Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, mut cache } = input;

        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let mut queries = self.q_norm.forward(
            &queries
                .reshape(&[B, L, self.n_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?,
        )?;
        let mut keys = self.k_norm.forward(
            &keys
                .reshape(&[B, L, self.n_kv_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?,
        )?;
        let mut values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        if let Some(cache) = cache.as_mut() {
            let q_input = nn::RopeInputBuilder::new(&queries)
                .offset(cache.offset())
                .build()?;
            queries = self.rope.forward(q_input)?;
            let k_input = nn::RopeInputBuilder::new(&keys)
                .offset(cache.offset())
                .build()?;
            keys = self.rope.forward(k_input)?;

            (keys, values) = cache.update_and_fetch(keys, values)?;
        } else {
            queries = self.rope.forward(nn::RopeInput::new(&queries))?;
            keys = self.rope.forward(nn::RopeInput::new(&keys))?;
        }

        // Determine mask mode: use Causal for prefill (L > 1), None for decode (L == 1)
        let sdpa_mask = match mask {
            Some(m) => Some(SdpaMask::Array(m)),
            None if L > 1 => Some(SdpaMask::Causal),
            None => None,
        };

        let output = mlx_rs_core::scaled_dot_product_attention(
            queries, keys, values, cache, self.scale, sdpa_mask,
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

/// Standard MLP (for non-MoE layers)
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mlp {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
}

impl Mlp {
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(hidden_dim, dim)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;

        Ok(Self {
            gate_proj: MaybeQuantized::Original(gate_proj),
            down_proj: MaybeQuantized::Original(down_proj),
            up_proj: MaybeQuantized::Original(up_proj),
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = nn::silu(&gate)?.multiply(&up)?;
        self.down_proj.forward(&activated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
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
    /// Apply gather_qmm with already-expanded input.
    /// x: [..., groups, D], indices: [..., k] -> output: [..., k, out_dim]
    pub fn apply(&self, x: &Array, indices: &Array, sorted_indices: bool) -> Result<Array, Exception> {
        mlx_rs::ops::gather_qmm(
            x,
            &*self.weight,
            &*self.scales,
            &*self.biases,
            None::<&Array>,        // lhs_indices
            Some(indices),         // rhs_indices - expert selection
            true,                  // transpose
            self.group_size,
            self.bits,
            None::<&str>,          // mode
            sorted_indices,        // sorted_indices
        )
    }
}

/// Sort tokens by their expert indices for coalesced memory access.
fn gather_sort(x: &Array, indices: &Array) -> Result<(Array, Array, Array), Exception> {
    let indices_shape = indices.shape();
    let m = *indices_shape.last().unwrap() as i32;  // k (num experts per token)

    // Flatten indices: [B, L, k] -> [B*L*k]
    let indices_flat = indices.flatten(None, None)?;

    // Get sort order: argsort gives indices that would sort the array
    let order = mlx_rs::ops::argsort(&indices_flat)?;

    // Get inverse order for unsorting later
    let inv_order = mlx_rs::ops::argsort(&order)?;

    // Flatten x from [B, L, 1, 1, D] to [B*L, 1, D] then reorder
    let x_shape = x.shape();
    let d = *x_shape.last().unwrap() as i32;
    let x_flat = x.reshape(&[-1, 1, d])?;  // [B*L, 1, D]

    // Reorder x: x_flat[order // m] selects the token for each sorted position
    let token_order = order.floor_divide(mlx_rs::array!(m))?;

    // Use take_axis to gather elements along axis 0
    let x_sorted = take_axis(&x_flat, &token_order, 0)?;

    // Reorder indices using take_axis
    let indices_sorted = take_axis(&indices_flat, &order, 0)?;

    Ok((x_sorted, indices_sorted, inv_order))
}

/// Unsort the output back to original token order.
fn scatter_unsort(x: &Array, inv_order: &Array, original_shape: &[i32]) -> Result<Array, Exception> {
    let x_shape = x.shape();
    let d = *x_shape.last().unwrap() as i32;

    // Flatten to [B*L*k, D] for indexing
    let x_flat = x.reshape(&[-1, d])?;

    // Reorder back to original order using take_axis
    let x_unsorted = take_axis(&x_flat, inv_order, 0)?;

    // Reshape to original shape [B, L, k, 1, D]
    let mut new_shape: Vec<i32> = original_shape.to_vec();
    new_shape.push(1);
    new_shape.push(d);
    x_unsorted.reshape(&new_shape)
}

/// SwitchGLU MLP for routed experts (efficient batched computation)
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
    /// Apply SwitchGLU experts using efficient gather_qmm operations.
    /// x: [B, L, D], indices: [B, L, k] -> output: [B, L, k, D]
    pub fn forward_experts(&mut self, x: &Array, indices: &Array) -> Result<Array, Exception> {
        let indices_shape = indices.shape();
        let b = indices_shape[0];
        let l = indices_shape[1];
        let k = indices_shape[2];

        // Expand x: [B, L, D] -> [B, L, 1, 1, D]
        let x_expanded = mlx_rs::ops::expand_dims(x, -2)?;  // [B, L, 1, D]
        let x_expanded = mlx_rs::ops::expand_dims(&x_expanded, -2)?;  // [B, L, 1, 1, D]

        // Use sorting optimization for coalesced memory access when >= 64 tokens
        let indices_size = b * l * k;
        let do_sort = indices_size >= 64;

        if do_sort {
            // Sort tokens by expert indices for better memory access
            let (x_sorted, indices_sorted, inv_order) = gather_sort(&x_expanded, indices)?;

            // Gate and Up projections with sorted data
            let gate = self.gate_proj.apply(&x_sorted, &indices_sorted, true)?;
            let up = self.up_proj.apply(&x_sorted, &indices_sorted, true)?;

            // SwiGLU activation using fused Metal kernel
            let activated = fused_swiglu(&up, &gate)?;

            // Down projection
            let output = self.down_proj.apply(&activated, &indices_sorted, true)?;

            // Unsort back to original order
            let output_unsorted = scatter_unsort(&output, &inv_order, &[b as i32, l as i32, k as i32])?;

            // Squeeze: [B, L, k, 1, D] -> [B, L, k, D]
            let shape = output_unsorted.shape();
            output_unsorted.reshape(&[shape[0] as i32, shape[1] as i32, shape[2] as i32, shape[4] as i32])
        } else {
            // No sorting for small batches
            let gate = self.gate_proj.apply(&x_expanded, indices, false)?;
            let up = self.up_proj.apply(&x_expanded, indices, false)?;

            // SwiGLU activation
            let activated = fused_swiglu(&up, &gate)?;

            // Down projection
            let output = self.down_proj.apply(&activated, indices, false)?;

            // Squeeze: [B, L, k, 1, D] -> [B, L, k, D]
            let shape = output.shape();
            if shape.len() == 5 {
                output.reshape(&[shape[0] as i32, shape[1] as i32, shape[2] as i32, shape[4] as i32])
            } else {
                Ok(output)
            }
        }
    }
}

/// MoE block with top-k routing (efficient batched computation)
#[derive(Debug, Clone, ModuleParameters)]
pub struct MoeBlock {
    pub num_experts: i32,
    pub top_k: i32,
    pub norm_topk_prob: bool,

    #[param]
    pub gate: MaybeQuantized<nn::Linear>,  // Gate: [hidden_size] -> [num_experts]
    #[param]
    pub switch_mlp: SwitchGLU,
}

impl Module<&Array> for MoeBlock {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        // x shape: [batch, seq_len, dim]

        // Compute gate scores: [B, L, num_experts]
        let gates = self.gate.forward(x)?;
        let gates = ops::softmax_axis(&gates, -1, true)?;

        // Top-k selection using argpartition
        let neg_gates = gates.negative()?;
        let partitioned_inds = ops::argpartition_axis(&neg_gates, self.top_k - 1, -1)?;
        let top_k_indices = partitioned_inds.index((.., .., ..self.top_k));

        // Gather scores for selected experts
        let mut top_k_scores = take_along_axis(&gates, &top_k_indices, -1)?;

        // Normalize scores if needed
        if self.norm_topk_prob && self.top_k > 1 {
            let score_sum = top_k_scores.sum_axis(-1, true)?;
            top_k_scores = top_k_scores.divide(&score_sum)?;
        }

        // Apply experts using efficient gather_qmm (batched computation)
        let expert_out = self.switch_mlp.forward_experts(x, &top_k_indices)?;

        // Weight by scores: [B, L, k, D] * [B, L, k, 1] -> sum over k
        let scores_expanded = top_k_scores.index((.., .., .., NewAxis));
        let weighted = expert_out.multiply(&scores_expanded)?;
        weighted.sum_axis(2, false)
    }

    fn training_mode(&mut self, _mode: bool) {
        // SwitchGLU has frozen quantized weights
    }
}

/// Transformer block that can use either MLP or MoE
#[derive(Debug, Clone, ModuleParameters)]
pub struct TransformerBlock {
    pub num_attention_heads: i32,
    pub hidden_size: i32,

    #[param]
    pub self_attn: Attention,
    #[param]
    pub mlp: Option<Mlp>,
    #[param]
    pub moe: Option<MoeBlock>,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    /// Check if a layer should use MoE (sparse) or dense MLP
    pub fn is_moe_layer(args: &ModelArgs, layer_idx: i32) -> bool {
        if args.mlp_only_layers.contains(&layer_idx) {
            false
        } else if args.decoder_sparse_step > 0 {
            (layer_idx + 1) % args.decoder_sparse_step == 0
        } else {
            true // Default to MoE for all layers
        }
    }
}

impl<C> Module<AttentionInput<'_, C>> for TransformerBlock
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;

        let self_attn_input = AttentionInput {
            x: &self.input_layernorm.forward(x)?,
            mask,
            cache,
        };
        let r = self.self_attn.forward(self_attn_input)?;
        let h = x.add(r)?;

        let out = self.post_attention_layernorm.forward(&h)?;

        let out = if let Some(moe) = &mut self.moe {
            moe.forward(&out)?
        } else if let Some(mlp) = &mut self.mlp {
            mlp.forward(&out)?
        } else {
            return Err(Exception::custom("Layer has neither MLP nor MoE"));
        };

        h.add(out)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        if let Some(mlp) = &mut self.mlp {
            mlp.training_mode(mode);
        }
        if let Some(moe) = &mut self.moe {
            moe.training_mode(mode);
        }
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen3MoeModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[param]
    pub layers: Vec<TransformerBlock>,
    #[param]
    pub norm: nn::RmsNorm,
}


pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<ModelInput<'_, C>> for Qwen3MoeModel
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput { inputs, mask, cache } = input;

        let mut h = self.embed_tokens.forward(inputs)?;

        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => match create_attention_mask(&h, cache, Some(true))? {
                Some(AttentionMask::Array(a)) => Some(a),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("Only `Array` mask is supported"))
                }
                None => None,
            },
        };

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c.as_mut(),
            };
            h = layer.forward(layer_input)?;
        }

        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <TransformerBlock as Module<AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Model {
    pub args: ModelArgs,

    #[param]
    pub model: Qwen3MoeModel,

    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self.model.forward(input)?;

        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&out),
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(embed_tokens) => embed_tokens.as_linear(&out),
                MaybeQuantized::Quantized(q_embed_tokens) => q_embed_tokens.as_linear(&out),
            },
        }
    }

    fn training_mode(&mut self, mode: bool) {
        <Qwen3MoeModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

// =================== Loading ===================

pub fn load_qwen3_moe_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn get_qwen3_moe_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
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

pub fn load_qwen3_moe_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_qwen3_moe_model_args(model_dir)?;

    // Qwen3-MoE requires quantized model for efficient SwitchGLU
    if model_args.quantization.is_none() {
        return Err(Error::Model(
            "Qwen3-MoE requires quantized model. Use mlx_lm to quantize the model first.".to_string()
        ));
    }

    load_qwen3_moe_model_quantized(model_dir, &model_args)
}

/// Load all weight arrays from safetensors files
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

/// Helper to get a weight array by key
fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    weights.get(key)
        .cloned()
        .ok_or_else(|| Error::Model(format!("Weight not found: {}", key)))
}

/// Create a QuantizedLinear from weight arrays
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

/// Create a QuantizedEmbedding from weight arrays
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
) -> Result<Mlp, Error> {
    Ok(Mlp {
        gate_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights, &format!("{}.gate_proj", prefix), group_size, bits
        )?),
        up_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights, &format!("{}.up_proj", prefix), group_size, bits
        )?),
        down_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights, &format!("{}.down_proj", prefix), group_size, bits
        )?),
    })
}

/// Load a quantized Qwen3 MoE model with efficient SwitchGLU experts
fn load_qwen3_moe_model_quantized(model_dir: &Path, args: &ModelArgs) -> Result<Model, Error> {
    let quant_config = args.quantization.as_ref()
        .ok_or_else(|| Error::Model("No quantization config".to_string()))?;
    let group_size = quant_config.group_size;
    let bits = quant_config.bits;

    eprintln!("Loading weights for {}-bit quantized Qwen3-MoE model...", bits);
    let weights = load_all_weights(model_dir)?;

    let mut layers = Vec::with_capacity(args.num_hidden_layers as usize);

    for i in 0..args.num_hidden_layers {
        let layer_prefix = format!("model.layers.{}", i);
        let is_moe = TransformerBlock::is_moe_layer(args, i);

        // Build attention
        let self_attn = Attention {
            n_heads: args.num_attention_heads,
            n_kv_heads: args.num_key_value_heads,
            scale: (args.head_dim as f32).sqrt().recip(),
            q_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.self_attn.q_proj", layer_prefix), group_size, bits
            )?),
            k_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.self_attn.k_proj", layer_prefix), group_size, bits
            )?),
            v_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.self_attn.v_proj", layer_prefix), group_size, bits
            )?),
            o_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.self_attn.o_proj", layer_prefix), group_size, bits
            )?),
            q_norm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.self_attn.q_norm.weight", layer_prefix))?),
                eps: args.rms_norm_eps,
            },
            k_norm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.self_attn.k_norm.weight", layer_prefix))?),
                eps: args.rms_norm_eps,
            },
            rope: initialize_rope(
                args.head_dim,
                args.rope_theta,
                false,
                &args.rope_scaling,
                args.max_position_embeddings,
            )?,
        };

        let (mlp, moe) = if is_moe {
            // Build MoE with efficient SwitchGLU
            // Gate uses 8-bit quantization for better routing accuracy
            let gate = MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.mlp.gate", layer_prefix), group_size, 8
            )?);

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

            let moe = MoeBlock {
                num_experts: args.num_experts,
                top_k: args.num_experts_per_tok,
                norm_topk_prob: args.norm_topk_prob,
                gate,
                switch_mlp,
            };

            (None, Some(moe))
        } else {
            // Dense MLP
            let mlp = make_quantized_mlp(
                &weights, &format!("{}.mlp", layer_prefix), group_size, bits
            )?;
            (Some(mlp), None)
        };

        let layer = TransformerBlock {
            num_attention_heads: args.num_attention_heads,
            hidden_size: args.hidden_size,
            self_attn,
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

    let qwen3_model = Qwen3MoeModel {
        vocab_size: args.vocab_size,
        num_hidden_layers: args.num_hidden_layers,
        embed_tokens: MaybeQuantized::Quantized(make_quantized_embedding(
            &weights, "model.embed_tokens", group_size, bits
        )?),
        layers,
        norm: nn::RmsNorm {
            weight: Param::new(get_weight(&weights, "model.norm.weight")?),
            eps: args.rms_norm_eps,
        },
    };

    let lm_head = if !args.tie_word_embeddings {
        Some(MaybeQuantized::Quantized(make_quantized_linear(
            &weights, "lm_head", group_size, bits
        )?))
    } else {
        None
    };

    let model = Model {
        args: args.clone(),
        model: qwen3_model,
        lm_head,
    };

    model.eval()?;

    Ok(model)
}

// =================== Generation ===================

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
    cache: &'a mut Vec<Option<C>>,
    temp: f32,
    state: GenerateState<'a>,
    prefetched: Option<Array>,
    token_count: usize,
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<Option<C>>,
        temp: f32,
        prompt_token: &'a Array,
    ) -> Self {
        Self {
            model,
            cache,
            temp,
            state: GenerateState::Prefill { prompt_token },
            prefetched: None,
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
        sample(&logits, self.temp)
    }
}

pub enum GenerateState<'a> {
    Prefill { prompt_token: &'a Array },
    Decode,
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

impl<'a, C> Iterator for Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        use mlx_rs::transforms::async_eval;

        match &self.state {
            GenerateState::Prefill { prompt_token } => {
                use mlx_rs::transforms::eval;

                let input = ModelInput {
                    inputs: prompt_token,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));

                let _ = async_eval([&y]);
                let next_y = tri!(self.compute_next(&y));
                let _ = async_eval([&next_y]);
                let _ = eval([&y]);

                self.prefetched = Some(next_y);
                self.state = GenerateState::Decode;
                self.token_count = 1;

                Some(Ok(y))
            }
            GenerateState::Decode => {
                let current = self.prefetched.take()?;
                let next_y = tri!(self.compute_next(&current));
                let _ = async_eval([&next_y]);

                self.prefetched = Some(next_y);

                self.token_count += 1;
                if self.token_count % 256 == 0 {
                    unsafe { mlx_sys::mlx_clear_cache(); }
                }

                Some(Ok(current))
            }
        }
    }
}
