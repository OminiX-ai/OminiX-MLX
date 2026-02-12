use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    argmax_axis, array, categorical,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt},
    nn,
    Array,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use mlx_rs_core::error::Error;

use crate::attention::{
    HybridAttention, LayerCache, LightningAttention, SparseAttention,
};
use crate::config::ModelArgs;

// ============================================================================
// MLP
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Mlp {
    #[param]
    pub gate_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
    #[param]
    pub up_proj: nn::Linear,
}

impl Mlp {
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(dim, hidden_dim).bias(false).build()?;
        let down_proj = nn::LinearBuilder::new(hidden_dim, dim).bias(false).build()?;
        let up_proj = nn::LinearBuilder::new(dim, hidden_dim).bias(false).build()?;
        Ok(Self {
            gate_proj,
            down_proj,
            up_proj,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let activated = nn::silu(self.gate_proj.forward(x)?)?.multiply(self.up_proj.forward(x)?)?;
        self.down_proj.forward(&activated)
    }
}

// ============================================================================
// Decoder Layer
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct DecoderLayer {
    pub residual_scale: f32,

    #[param]
    pub self_attn: HybridAttention,
    #[param]
    pub mlp: Mlp,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl DecoderLayer {
    pub fn new(args: &ModelArgs, layer_idx: usize) -> Result<Self, Exception> {
        let attn = if args.is_sparse_layer(layer_idx) {
            HybridAttention::Sparse(SparseAttention::new(args)?)
        } else {
            HybridAttention::Lightning(LightningAttention::new(args)?)
        };

        Ok(Self {
            residual_scale: args.residual_scale(),
            self_attn: attn,
            mlp: Mlp::new(args.hidden_size, args.intermediate_size)?,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut LayerCache,
    ) -> Result<Array, Exception> {
        // Pre-norm attention + scaled residual
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, mask, cache)?;
        let h = residual.add(h.multiply(array!(self.residual_scale))?)?;

        // Pre-norm MLP + scaled residual
        let residual = h.clone();
        let r = self.post_attention_layernorm.forward(&h)?;
        let r = self.mlp.forward(&r)?;
        residual.add(r.multiply(array!(self.residual_scale))?)
    }
}

// ============================================================================
// Model
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct MiniCPMSALAModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,
    pub scale_emb: f32,

    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl MiniCPMSALAModel {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let layers = (0..args.num_hidden_layers as usize)
            .map(|i| DecoderLayer::new(args, i))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            scale_emb: args.scale_emb,
            embed_tokens: nn::Embedding::new(args.vocab_size, args.hidden_size)?,
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }

    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        caches: &mut [LayerCache],
    ) -> Result<Array, Exception> {
        // Embed with muP scaling
        let mut h = self
            .embed_tokens
            .forward(inputs)?
            .multiply(array!(self.scale_emb))?;

        // Create attention mask for sparse layers (lightning layers don't use it)
        let mask = if mask.is_some() {
            mask.cloned()
        } else {
            let L = h.shape()[1];
            if L > 1 {
                // For prefill, create causal mask
                // We need an explicit array mask since some layers are sparse and some aren't
                let mask = create_additive_causal_mask(L, &caches[0])?;
                Some(mask)
            } else {
                None
            }
        };

        for (layer, cache) in self.layers.iter_mut().zip(caches.iter_mut()) {
            h = layer.forward(&h, mask.as_ref(), cache)?;
        }

        self.norm.forward(&h)
    }
}

/// Create an additive causal mask for attention.
/// Returns [seq_len, total_len] with 0 where attending and -inf where masked.
fn create_additive_causal_mask(
    seq_len: i32,
    first_cache: &LayerCache,
) -> Result<Array, Exception> {
    let offset = first_cache.offset();
    let total_len = offset + seq_len;

    // Build causal mask: 0 where attending, -inf where masked
    let row_idx = mlx_rs::ops::arange::<_, i32>(offset, offset + seq_len, None)?
        .reshape(&[seq_len, 1])?;
    let col_idx = mlx_rs::ops::arange::<_, i32>(0, total_len, None)?
        .reshape(&[1, total_len])?;
    let causal = mlx_rs::ops::ge(&row_idx, &col_idx)?;
    let mask = mlx_rs::ops::r#where(
        &causal,
        &array!(0.0_f32),
        &array!(f32::NEG_INFINITY),
    )?;

    Ok(mask)
}

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Model {
    pub args: ModelArgs,
    pub logits_scale: f32,

    #[param]
    pub model: MiniCPMSALAModel,
    #[param]
    pub lm_head: Option<nn::Linear>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = MiniCPMSALAModel::new(&args)?;
        let logits_scale = args.logits_scale();

        let lm_head = if !args.tie_word_embeddings {
            Some(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            )
        } else {
            None
        };

        Ok(Self {
            args,
            logits_scale,
            model,
            lm_head,
        })
    }

    pub fn forward(
        &mut self,
        inputs: &Array,
        caches: &mut [LayerCache],
    ) -> Result<Array, Exception> {
        let out = self.model.forward(inputs, None, caches)?;

        // muP logits scaling: lm_head(hidden / logits_scale)
        let scaled = out.multiply(array!(1.0 / self.logits_scale))?;

        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&scaled),
            None => self.model.embed_tokens.as_linear(&scaled),
        }
    }
}

// ============================================================================
// Sampling
// ============================================================================

pub fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    match temp {
        t if t <= 0.0 => argmax_axis!(logits, -1).map_err(Into::into),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical!(logits).map_err(Into::into)
        }
    }
}

// ============================================================================
// Weight Loading
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
    Ok(serde_json::from_reader(file)?)
}

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_model_args(model_dir)?;

    let mut model = Model::new(model_args).map_err(Error::Mlx)?;

    let weights_index = model_dir.join("model.safetensors.index.json");
    if weights_index.exists() {
        // Sharded model
        let json = std::fs::read_to_string(&weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();

        for weight_file in weight_files {
            let path = model_dir.join(weight_file);
            model.load_safetensors(&path)?;
        }
    } else {
        // Single file
        let path = model_dir.join("model.safetensors");
        model.load_safetensors(&path)?;
    }

    Ok(model)
}
