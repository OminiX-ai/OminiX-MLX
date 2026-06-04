//! Gemma 4 full text model: embed (×√hidden) → 48 blocks → final norm → tied lm_head → softcap.
//! TODO(M1-Task3): forward pass is implemented in Task 3.

use std::path::Path;

use mlx_rs::nn;

use crate::block::TransformerBlock;
use crate::config::{LayerKind, ModelArgs, QuantConfig};
use crate::error::{Error, Result};
use crate::norm::GemmaRmsNorm;
use crate::weights::{get_weight, load_all_weights, make_quantized_embedding};

/// Gemma 4 full text model (struct only; forward is Task 3).
#[allow(dead_code)] // TODO(M1-Task3): all fields used by forward; remove this attr then.
pub struct Gemma4TextModel {
    pub(crate) embed_tokens: nn::QuantizedEmbedding,
    pub(crate) embed_scale: f32,
    pub(crate) layers: Vec<TransformerBlock>,
    pub(crate) norm: GemmaRmsNorm,
    pub(crate) final_logit_softcapping: f32,
    pub(crate) layer_types: Vec<LayerKind>,
    pub(crate) sliding_window: i32,
}

/// Load the full Gemma 4 text model from `model_dir`.
///
/// Expects `config.json` and `model.safetensors.index.json` (plus shards) in that directory.
pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Gemma4TextModel> {
    let dir = model_dir.as_ref();

    let cfg = std::fs::read_to_string(dir.join("config.json"))?;
    let args = ModelArgs::from_config_str(&cfg)?;
    let quant = QuantConfig::from_config_str(&cfg)?
        .ok_or_else(|| Error::Config("expected quantization config".into()))?;
    let weights = load_all_weights(dir)?;

    // Embedding — (bits, group_size) from quant_for, passed as (group_size, bits) to helper.
    let ep = "language_model.model.embed_tokens";
    let (eb, eg) = quant.quant_for(ep); // eb = bits, eg = group_size
    let embed_tokens = make_quantized_embedding(&weights, ep, eg, eb)?;

    // 48 decoder layers.
    let mut layers = Vec::with_capacity(args.num_hidden_layers as usize);
    for i in 0..args.num_hidden_layers {
        layers.push(TransformerBlock::from_weights(&weights, &args, &quant, i)?);
    }

    // Final RMSNorm.
    let norm = GemmaRmsNorm::from_weight(
        get_weight(&weights, "language_model.model.norm.weight")?,
        args.rms_norm_eps,
    );

    Ok(Gemma4TextModel {
        embed_tokens,
        embed_scale: (args.hidden_size as f32).sqrt(),
        layers,
        norm,
        final_logit_softcapping: args.final_logit_softcapping,
        layer_types: args.layer_types.clone(),
        sliding_window: args.sliding_window,
    })
}
