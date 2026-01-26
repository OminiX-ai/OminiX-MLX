//! Qwen3 Model Implementation

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    argmax_axis, array, categorical,
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters as ModuleParametersTrait, ModuleParametersExt, Param},
    nn,
    ops::indexing::{IndexOp, NewAxis},
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use mlx_rs_core::{
    cache::KeyValueCache,
    error::Error,
    utils::{
        create_attention_mask, initialize_rope, scaled_dot_product_attention,
        AttentionMask, FloatOrString, SdpaMask,
    },
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
    pub max_position_embeddings: i32,
    pub rope_theta: f32,
    pub head_dim: i32,
    pub tie_word_embeddings: bool,
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

// ============================================================================
// Attention
// ============================================================================

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
    pub fn new(args: &ModelArgs) -> Result<Self, Error> {
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
    fn forward(&mut self, input: AttentionInput<'_, C>) -> std::result::Result<Self::Output, Self::Error> {
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

        let sdpa_mask = match mask {
            Some(m) => Some(SdpaMask::Array(m)),
            None if L > 1 => Some(SdpaMask::Causal),
            None => None,
        };

        let output = scaled_dot_product_attention::<C>(
            queries, keys, values, None, self.scale, sdpa_mask,
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

// ============================================================================
// MLP
// ============================================================================

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
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Error> {
        let gate_proj = nn::LinearBuilder::new(dim, hidden_dim).bias(false).build()?;
        let down_proj = nn::LinearBuilder::new(hidden_dim, dim).bias(false).build()?;
        let up_proj = nn::LinearBuilder::new(dim, hidden_dim).bias(false).build()?;

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

    fn forward(&mut self, input: &Array) -> std::result::Result<Self::Output, Self::Error> {
        let activated = nn::silu(self.gate_proj.forward(input)?)?
            .multiply(self.up_proj.forward(input)?)?;
        self.down_proj.forward(&activated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

// ============================================================================
// Transformer Block
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct TransformerBlock {
    pub num_attention_heads: i32,
    pub hidden_size: i32,

    #[quantizable]
    #[param]
    pub self_attn: Attention,
    #[quantizable]
    #[param]
    pub mlp: Mlp,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    pub fn new(args: &ModelArgs) -> Result<Self, Error> {
        Ok(Self {
            num_attention_heads: args.num_attention_heads,
            hidden_size: args.hidden_size,
            self_attn: Attention::new(args)?,
            mlp: Mlp::new(args.hidden_size, args.intermediate_size)?,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

impl<C> Module<AttentionInput<'_, C>> for TransformerBlock
where
    C: KeyValueCache,
{
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
        let r = self.mlp.forward(&self.post_attention_layernorm.forward(&h)?)?;
        h.add(r)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

// ============================================================================
// Qwen3 Model
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Qwen3Model {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<TransformerBlock>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl Qwen3Model {
    pub fn new(args: &ModelArgs) -> Result<Self, Error> {
        assert!(args.vocab_size.is_positive());

        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            embed_tokens: MaybeQuantized::Original(
                nn::Embedding::new(args.vocab_size, args.hidden_size)?
            ),
            layers: (0..args.num_hidden_layers)
                .map(|_| TransformerBlock::new(args))
                .collect::<Result<Vec<_>, Error>>()?,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<ModelInput<'_, C>> for Qwen3Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> std::result::Result<Self::Output, Self::Error> {
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

// ============================================================================
// Full Model with LM Head
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    pub model: Qwen3Model,

    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Error> {
        let model = Qwen3Model::new(&args)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        } else {
            None
        };

        Ok(Self { args, model, lm_head })
    }

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

    fn forward(&mut self, input: ModelInput<'_, C>) -> std::result::Result<Self::Output, Self::Error> {
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
        <Qwen3Model as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

// ============================================================================
// Model Loading
// ============================================================================

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
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

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_model_args(model_dir)?;

    if model_args.quantization.is_some() {
        return load_model_quantized(model_dir, &model_args);
    }

    let mut model = Model::new(model_args)?;

    let weights_index = model_dir.join("model.safetensors.index.json");
    let json = std::fs::read_to_string(weights_index)?;
    let weight_map: WeightMap = serde_json::from_str(&json)?;

    let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();

    for weight_file in weight_files {
        let weights_filename = model_dir.join(weight_file);
        model.load_safetensors(&weights_filename)?;
    }

    Ok(model)
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

fn load_model_quantized(model_dir: &Path, args: &ModelArgs) -> Result<Model, Error> {
    let quant_config = args.quantization.as_ref()
        .ok_or_else(|| Error::Model("No quantization config".to_string()))?;
    let group_size = quant_config.group_size;
    let bits = quant_config.bits;

    let weights = load_all_weights(model_dir)?;

    let mut layers = Vec::with_capacity(args.num_hidden_layers as usize);

    for i in 0..args.num_hidden_layers {
        let prefix = format!("model.layers.{}", i);

        let attention = Attention {
            n_heads: args.num_attention_heads,
            n_kv_heads: args.num_key_value_heads,
            scale: (args.head_dim as f32).sqrt().recip(),
            q_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.self_attn.q_proj", prefix), group_size, bits
            )?),
            k_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.self_attn.k_proj", prefix), group_size, bits
            )?),
            v_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.self_attn.v_proj", prefix), group_size, bits
            )?),
            o_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.self_attn.o_proj", prefix), group_size, bits
            )?),
            q_norm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.self_attn.q_norm.weight", prefix))?),
                eps: args.rms_norm_eps,
            },
            k_norm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.self_attn.k_norm.weight", prefix))?),
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

        let mlp = Mlp {
            gate_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.mlp.gate_proj", prefix), group_size, bits
            )?),
            down_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.mlp.down_proj", prefix), group_size, bits
            )?),
            up_proj: MaybeQuantized::Quantized(make_quantized_linear(
                &weights, &format!("{}.mlp.up_proj", prefix), group_size, bits
            )?),
        };

        let block = TransformerBlock {
            num_attention_heads: args.num_attention_heads,
            hidden_size: args.hidden_size,
            self_attn: attention,
            mlp,
            input_layernorm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.input_layernorm.weight", prefix))?),
                eps: args.rms_norm_eps,
            },
            post_attention_layernorm: nn::RmsNorm {
                weight: Param::new(get_weight(&weights, &format!("{}.post_attention_layernorm.weight", prefix))?),
                eps: args.rms_norm_eps,
            },
        };

        layers.push(block);
    }

    let qwen3_model = Qwen3Model {
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

// ============================================================================
// Generation
// ============================================================================

pub fn sample(logits: &Array, temp: f32) -> std::result::Result<Array, Exception> {
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

pub enum GenerateState<'a> {
    Prefill { prompt_token: &'a Array },
    Decode,
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

    fn compute_next(&mut self, y: &Array) -> std::result::Result<Array, Exception> {
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
    type Item = std::result::Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        use mlx_rs::transforms::{async_eval, eval};

        match &self.state {
            GenerateState::Prefill { prompt_token } => {
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
                let _ = mlx_rs::transforms::async_eval([&next_y]);

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
