//! Mixtral MoE Model Implementation
//!
//! Mixtral is a Mixture of Experts (MoE) model from Mistral AI.
//! Uses top-k routing with softmax scores on selected experts.

use std::{collections::{HashMap, HashSet}, path::Path};

use mlx_rs::{
    argmax_axis, array, categorical,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters as ModuleParametersTrait, ModuleParametersExt, Param},
    nn, ops,
    ops::indexing::{IndexOp, NewAxis, take_along_axis, take_axis},
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use mlx_rs_core::{
    cache::KeyValueCache,
    error::Error,
    fused_swiglu,
    utils::{create_attention_mask, scaled_dot_product_attention, AttentionMask, SdpaMask},
};

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone, Deserialize, Default)]
pub struct QuantizationConfig {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
}

fn default_group_size() -> i32 { 64 }
fn default_bits() -> i32 { 4 }
fn default_vocab_size() -> i32 { 32000 }
fn default_hidden_size() -> i32 { 4096 }
fn default_intermediate_size() -> i32 { 14336 }
fn default_num_hidden_layers() -> i32 { 32 }
fn default_num_attention_heads() -> i32 { 32 }
fn default_num_experts_per_tok() -> i32 { 2 }
fn default_num_local_experts() -> i32 { 8 }
fn default_rms_norm_eps() -> f32 { 1e-5 }
fn default_rope_theta() -> f32 { 1e6 }

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: i32,
    pub num_key_value_heads: Option<i32>,
    #[serde(default = "default_num_local_experts")]
    pub num_local_experts: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_traditional: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

impl ModelArgs {
    pub fn num_key_value_heads(&self) -> i32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }
}

// ============================================================================
// Attention
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct Attention {
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

pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
}

impl<C: KeyValueCache> Module<AttentionInput<'_, C>> for Attention {
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_, C>) -> std::result::Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, mut cache } = input;

        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let mut queries = queries.reshape(&[B, L, self.n_heads, -1])?.transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = keys.reshape(&[B, L, self.n_kv_heads, -1])?.transpose_axes(&[0, 2, 1, 3])?;
        let mut values = values.reshape(&[B, L, self.n_kv_heads, -1])?.transpose_axes(&[0, 2, 1, 3])?;

        if let Some(cache) = cache.as_mut() {
            let offset = cache.offset();
            queries = self.rope.forward(nn::RopeInputBuilder::new(&queries).offset(offset).build()?)?;
            keys = self.rope.forward(nn::RopeInputBuilder::new(&keys).offset(offset).build()?)?;
            (keys, values) = cache.update_and_fetch(keys, values)?;
        } else {
            queries = self.rope.forward(nn::RopeInput::new(&queries))?;
            keys = self.rope.forward(nn::RopeInput::new(&keys))?;
        }

        let sdpa_mask = match mask {
            Some(m) => Some(SdpaMask::Array(m)),
            None if L > 1 => Some(SdpaMask::Causal),
            None => None,
        };

        let output = scaled_dot_product_attention::<C>(
            queries, keys, values, None, self.scale, sdpa_mask)?
            .transpose_axes(&[0, 2, 1, 3])?.reshape(&[B, L, -1])?;

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

// ============================================================================
// Quantized Switch Linear for MoE
// ============================================================================

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
    pub fn apply(&self, x: &Array, indices: &Array, sorted_indices: bool) -> std::result::Result<Array, Exception> {
        ops::gather_qmm(
            x, &*self.weight, &*self.scales, &*self.biases,
            None::<&Array>, Some(indices), true,
            self.group_size, self.bits, None::<&str>, sorted_indices,
        )
    }
}

fn gather_sort(x: &Array, indices: &Array) -> std::result::Result<(Array, Array, Array), Exception> {
    let m = *indices.shape().last().unwrap() as i32;
    let indices_flat = indices.flatten(None, None)?;
    let order = ops::argsort(&indices_flat)?;
    let inv_order = ops::argsort(&order)?;

    let d = *x.shape().last().unwrap() as i32;
    let x_flat = x.reshape(&[-1, 1, d])?;

    let token_order = order.floor_divide(array!(m))?;
    let x_sorted = take_axis(&x_flat, &token_order, 0)?;
    let indices_sorted = take_axis(&indices_flat, &order, 0)?;

    Ok((x_sorted, indices_sorted, inv_order))
}

fn scatter_unsort(x: &Array, inv_order: &Array, original_shape: &[i32]) -> std::result::Result<Array, Exception> {
    let d = *x.shape().last().unwrap() as i32;
    let x_unsorted = take_axis(&x.reshape(&[-1, d])?, inv_order, 0)?;
    let mut new_shape: Vec<i32> = original_shape.to_vec();
    new_shape.extend([1, d]);
    x_unsorted.reshape(&new_shape)
}

// ============================================================================
// SwitchGLU MLP for routed experts
// ============================================================================

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
    pub fn forward_experts(&mut self, x: &Array, indices: &Array) -> std::result::Result<Array, Exception> {
        let indices_shape = indices.shape();
        let b = indices_shape[0];
        let l = indices_shape[1];
        let k = indices_shape[2];

        let x_expanded = ops::expand_dims(&ops::expand_dims(x, -2)?, -2)?;
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

// ============================================================================
// Mixtral Sparse MoE Block
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct MixtralSparseMoeBlock {
    pub num_experts: i32,
    pub num_experts_per_tok: i32,

    #[param]
    pub gate: MaybeQuantized<nn::Linear>,
    #[param]
    pub switch_mlp: SwitchGLU,
}

impl Module<&Array> for MixtralSparseMoeBlock {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> std::result::Result<Self::Output, Self::Error> {
        let gates = self.gate.forward(x)?;
        let k = self.num_experts_per_tok;
        let neg_gates = gates.negative()?;
        let partitioned_inds = ops::argpartition_axis(&neg_gates, k - 1, -1)?;
        let inds = partitioned_inds.index((.., .., ..k));
        let selected_gates = take_along_axis(&gates, &inds, -1)?;
        let scores = ops::softmax_axis(&selected_gates, -1, true)?;

        let y = self.switch_mlp.forward_experts(x, &inds)?;
        let scores_expanded = scores.index((.., .., .., NewAxis));
        y.multiply(&scores_expanded)?.sum_axis(2, false)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate.training_mode(mode);
    }
}

// ============================================================================
// Decoder Layer
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct DecoderLayer {
    #[param]
    pub self_attn: Attention,
    #[param]
    pub block_sparse_moe: MixtralSparseMoeBlock,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl<C: KeyValueCache> Module<AttentionInput<'_, C>> for DecoderLayer {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_, C>) -> std::result::Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;
        let attn_input = AttentionInput {
            x: &self.input_layernorm.forward(x)?,
            mask,
            cache,
        };
        let h = x.add(self.self_attn.forward(attn_input)?)?;
        let r = self.block_sparse_moe.forward(&self.post_attention_layernorm.forward(&h)?)?;
        h.add(r)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.block_sparse_moe.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

// ============================================================================
// Mixtral Model
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct MixtralModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}

pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C: KeyValueCache + Default> Module<ModelInput<'_, C>> for MixtralModel {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> std::result::Result<Self::Output, Self::Error> {
        let ModelInput { inputs, mask, cache } = input;
        let mut h = self.embed_tokens.forward(inputs)?;

        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => match create_attention_mask(&h, cache, Some(true))? {
                Some(AttentionMask::Array(a)) => Some(a),
                Some(AttentionMask::Causal) => return Err(Exception::custom("Only `Array` mask is supported")),
                None => None,
            },
        };

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(AttentionInput { x: &h, mask: mask.as_ref(), cache: c.as_mut() })?;
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

// ============================================================================
// Full Model
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct Model {
    pub args: ModelArgs,

    #[param]
    pub model: MixtralModel,
    #[param]
    pub lm_head: MaybeQuantized<nn::Linear>,
}

impl Model {
    pub fn model_type(&self) -> &str { &self.args.model_type }
}

impl<C: KeyValueCache + Default> Module<ModelInput<'_, C>> for Model {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> std::result::Result<Self::Output, Self::Error> {
        self.lm_head.forward(&self.model.forward(input)?)
    }

    fn training_mode(&mut self, mode: bool) {
        <MixtralModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        self.lm_head.training_mode(mode);
    }
}

// ============================================================================
// Model Loading
// ============================================================================

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    Tokenizer::from_file(model_dir.as_ref().join("tokenizer.json")).map_err(Into::into)
}

pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
    Ok(serde_json::from_reader(file)?)
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

    let mut all_weights: HashMap<String, Array> = HashMap::new();
    for weight_file in weight_map.weight_map.values().collect::<HashSet<_>>() {
        all_weights.extend(Array::load_safetensors(&model_dir.join(weight_file))?);
    }
    Ok(all_weights)
}

fn sanitize_weights(weights: &mut HashMap<String, Array>, args: &ModelArgs) -> Result<(), Error> {
    if !weights.contains_key("model.layers.0.block_sparse_moe.experts.0.w1.weight") {
        return Ok(());
    }

    let mappings = [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")];

    for layer_idx in 0..args.num_hidden_layers {
        let prefix = format!("model.layers.{}", layer_idx);

        for (old_name, new_name) in &mappings {
            for component in &["weight", "scales", "biases"] {
                let first_key = format!("{}.block_sparse_moe.experts.0.{}.{}", prefix, old_name, component);
                if !weights.contains_key(&first_key) { continue; }

                let expert_arrays: Vec<Array> = (0..args.num_local_experts)
                    .map(|e| {
                        let key = format!("{}.block_sparse_moe.experts.{}.{}.{}", prefix, e, old_name, component);
                        weights.remove(&key).ok_or_else(|| Error::WeightNotFound(key))
                    })
                    .collect::<Result<Vec<_>, Error>>()?;

                let stacked = ops::stack_axis(&expert_arrays.iter().collect::<Vec<_>>(), 0)?;
                weights.insert(format!("{}.block_sparse_moe.switch_mlp.{}.{}", prefix, new_name, component), stacked);
            }
        }
    }
    Ok(())
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    weights.get(key).cloned().ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

fn make_quantized_linear(weights: &HashMap<String, Array>, prefix: &str, group_size: i32, bits: i32) -> Result<nn::QuantizedLinear, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;

    let inner = nn::Linear { weight: Param::new(weight), bias: Param::new(None) };
    let mut ql = nn::QuantizedLinear { group_size, bits, scales: Param::new(scales), biases: Param::new(biases), inner };
    ql.freeze_parameters(true);
    Ok(ql)
}

fn make_quantized_embedding(weights: &HashMap<String, Array>, prefix: &str, group_size: i32, bits: i32) -> Result<nn::QuantizedEmbedding, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;

    let inner = nn::Embedding { weight: Param::new(weight) };
    let mut qe = nn::QuantizedEmbedding { group_size, bits, scales: Param::new(scales), biases: Param::new(biases), inner };
    qe.freeze_parameters(true);
    Ok(qe)
}

fn make_quantized_switch_linear(weights: &HashMap<String, Array>, prefix: &str, group_size: i32, bits: i32) -> Result<QuantizedSwitchLinear, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;

    let shape = weight.shape();
    let num_experts = shape[0] as i32;
    let output_dims = shape[1] as i32;
    let input_dims = (scales.shape()[2] as i32) * group_size;

    Ok(QuantizedSwitchLinear {
        num_experts, input_dims, output_dims, group_size, bits,
        weight: Param::new(weight), scales: Param::new(scales), biases: Param::new(biases),
    })
}

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let args = get_model_args(model_dir)?;

    if args.quantization.is_none() {
        return Err(Error::Model("Mixtral requires quantized model. Use mlx_lm to quantize first.".to_string()));
    }

    let quant_config = args.quantization.as_ref().unwrap();
    let (group_size, bits) = (quant_config.group_size, quant_config.bits);

    let mut weights = load_all_weights(model_dir)?;
    sanitize_weights(&mut weights, &args)?;

    let head_dim = args.head_dim();
    let n_kv_heads = args.num_key_value_heads();

    let layers: Vec<DecoderLayer> = (0..args.num_hidden_layers).map(|i| {
        let prefix = format!("model.layers.{}", i);

        let self_attn = Attention {
            n_heads: args.num_attention_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
            q_proj: MaybeQuantized::Quantized(make_quantized_linear(&weights, &format!("{}.self_attn.q_proj", prefix), group_size, bits)?),
            k_proj: MaybeQuantized::Quantized(make_quantized_linear(&weights, &format!("{}.self_attn.k_proj", prefix), group_size, bits)?),
            v_proj: MaybeQuantized::Quantized(make_quantized_linear(&weights, &format!("{}.self_attn.v_proj", prefix), group_size, bits)?),
            o_proj: MaybeQuantized::Quantized(make_quantized_linear(&weights, &format!("{}.self_attn.o_proj", prefix), group_size, bits)?),
            rope: nn::RopeBuilder::new(head_dim).base(args.rope_theta).traditional(args.rope_traditional).build().unwrap(),
        };

        let block_sparse_moe = MixtralSparseMoeBlock {
            num_experts: args.num_local_experts,
            num_experts_per_tok: args.num_experts_per_tok,
            gate: MaybeQuantized::Quantized(make_quantized_linear(&weights, &format!("{}.block_sparse_moe.gate", prefix), group_size, bits)?),
            switch_mlp: SwitchGLU {
                gate_proj: make_quantized_switch_linear(&weights, &format!("{}.block_sparse_moe.switch_mlp.gate_proj", prefix), group_size, bits)?,
                up_proj: make_quantized_switch_linear(&weights, &format!("{}.block_sparse_moe.switch_mlp.up_proj", prefix), group_size, bits)?,
                down_proj: make_quantized_switch_linear(&weights, &format!("{}.block_sparse_moe.switch_mlp.down_proj", prefix), group_size, bits)?,
            },
        };

        Ok(DecoderLayer {
            self_attn,
            block_sparse_moe,
            input_layernorm: nn::RmsNorm { weight: Param::new(get_weight(&weights, &format!("{}.input_layernorm.weight", prefix))?), eps: args.rms_norm_eps },
            post_attention_layernorm: nn::RmsNorm { weight: Param::new(get_weight(&weights, &format!("{}.post_attention_layernorm.weight", prefix))?), eps: args.rms_norm_eps },
        })
    }).collect::<Result<Vec<_>, Error>>()?;

    let model = Model {
        args: args.clone(),
        model: MixtralModel {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            embed_tokens: MaybeQuantized::Quantized(make_quantized_embedding(&weights, "model.embed_tokens", group_size, bits)?),
            layers,
            norm: nn::RmsNorm { weight: Param::new(get_weight(&weights, "model.norm.weight")?), eps: args.rms_norm_eps },
        },
        lm_head: MaybeQuantized::Quantized(make_quantized_linear(&weights, "lm_head", group_size, bits)?),
    };

    model.eval()?;
    Ok(model)
}

// ============================================================================
// Generation
// ============================================================================

pub fn sample(logits: &Array, temp: f32) -> std::result::Result<Array, Exception> {
    match temp {
        0.0 => argmax_axis!(logits, -1).map_err(Into::into),
        _ => categorical!(logits.multiply(array!(1.0 / temp))?).map_err(Into::into),
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

pub enum GenerateState<'a> {
    Prefill { prompt_token: &'a Array },
    Decode,
}

impl<'a, C: KeyValueCache + Default> Generate<'a, C> {
    pub fn new(model: &'a mut Model, cache: &'a mut Vec<Option<C>>, temp: f32, prompt_token: &'a Array) -> Self {
        Self { model, cache, temp, state: GenerateState::Prefill { prompt_token }, prefetched: None, token_count: 0 }
    }

    fn compute_next(&mut self, y: &Array) -> std::result::Result<Array, Exception> {
        let input = ModelInput { inputs: &y.index((.., NewAxis)), mask: None, cache: self.cache };
        sample(&self.model.forward(input)?, self.temp)
    }
}

macro_rules! tri {
    ($expr:expr) => { match $expr { Ok(val) => val, Err(e) => return Some(Err(e.into())) } };
}

impl<'a, C: KeyValueCache + Default> Iterator for Generate<'a, C> {
    type Item = std::result::Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        use mlx_rs::transforms::{async_eval, eval};

        match &self.state {
            GenerateState::Prefill { prompt_token } => {
                let input = ModelInput { inputs: prompt_token, mask: None, cache: self.cache };
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
