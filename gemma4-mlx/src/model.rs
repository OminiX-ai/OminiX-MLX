//! Gemma 4 full text model: embed (×√hidden) → 48 blocks → final norm → tied lm_head → softcap.

use std::path::Path;

use mlx_rs::{module::Module, nn, ops::indexing::IndexOp, Array};

use crate::block::TransformerBlock;
use crate::config::{LayerKind, ModelArgs, QuantConfig};
use crate::error::{Error, Result};
use crate::mask::{full_causal_mask, sliding_window_mask};
use crate::norm::GemmaRmsNorm;
use crate::weights::{get_weight, load_all_weights, make_quantized_embedding};

/// Gemma 4 full text model.
pub struct Gemma4TextModel {
    pub(crate) embed_tokens: nn::QuantizedEmbedding,
    pub(crate) embed_scale: f32,
    pub(crate) layers: Vec<TransformerBlock>,
    pub(crate) norm: GemmaRmsNorm,
    pub(crate) final_logit_softcapping: f32,
    pub(crate) layer_types: Vec<LayerKind>,
    pub(crate) sliding_window: i32,
}

impl Gemma4TextModel {
    /// Forward pass: tokens [B, L] → logits [B, *, vocab_size].
    ///
    /// `last_only`: if true, slice hidden to the last position before the lm_head projection,
    /// returning logits of shape [B, 1, vocab_size] rather than [B, L, vocab_size].
    pub fn forward(&mut self, tokens: &Array, last_only: bool) -> Result<Array> {
        // Embed tokens and scale by √hidden_size.
        let mut h = self.embed_tokens.forward(tokens)?; // [B, L, hidden]
        h = h.multiply(&Array::from_f32(self.embed_scale))?;

        let seq_len = h.shape()[1]; // L

        // Build both mask variants once; choose per-layer below.
        let full_mask = full_causal_mask(seq_len, 0)?;
        let sliding_mask = sliding_window_mask(seq_len, 0, self.sliding_window)?;

        // Per-layer forward with the appropriate mask kind.
        for i in 0..self.layers.len() {
            let mask = match self.layer_types[i] {
                LayerKind::Global => &full_mask,
                LayerKind::Sliding => &sliding_mask,
            };
            h = self.layers[i].forward(&h, mask)?;
        }

        // Final layer norm.
        h = self.norm.forward(&h)?;

        // Optionally keep only the last token's hidden state → [B, 1, hidden].
        if last_only {
            h = h.index((.., -1_i32.., ..));
        }

        // Tied output projection: embed_tokens.as_linear → [B, *, vocab_size].
        let mut logits = self.embed_tokens.as_linear(&h)?;

        // Logit soft-capping: logits = cap * tanh(logits / cap).
        if self.final_logit_softcapping > 0.0 {
            let cap = Array::from_f32(self.final_logit_softcapping);
            logits = mlx_rs::ops::tanh(&logits.divide(&cap)?)?
                .multiply(&cap)?;
        }

        Ok(logits)
    }
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
