//! Gemma 4 full text model: embed (×√hidden) → 48 blocks → final norm → tied lm_head → softcap.

use std::path::Path;

use mlx_rs::{module::Module, nn, ops::indexing::IndexOp, Array};
use mlx_rs_core::cache::{KVCache, KeyValueCache};

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
    /// Create one fresh KVCache per layer. Offsets start at 0.
    pub fn new_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    /// Cached forward for prefill (L>1, fresh caches) AND decode (L==1, warm caches).
    /// `caches` must have one entry per layer.
    #[allow(non_snake_case)]
    pub fn forward_cached(&mut self, tokens: &Array, caches: &mut [KVCache], last_only: bool) -> Result<Array> {
        let L = tokens.shape()[1];
        let mut h = self.embed_tokens.forward(tokens)?;
        h = h.multiply(&Array::from_f32(self.embed_scale))?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let off = caches[i].offset();
            let mask = decode_mask(self.layer_types[i], L, off, self.sliding_window)?;
            h = layer.forward_cached(&h, mask.as_ref(), &mut caches[i])?;
        }

        h = self.norm.forward(&h)?;

        if last_only {
            h = h.index((.., -1_i32.., ..));
        }

        let mut logits = self.embed_tokens.as_linear(&h)?;

        if self.final_logit_softcapping > 0.0 {
            let cap = Array::from_f32(self.final_logit_softcapping);
            logits = mlx_rs::ops::tanh(&logits.divide(&cap)?)?.multiply(&cap)?;
        }

        Ok(logits)
    }

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

/// Per-layer attention mask given query length L at cache offset `off`.
///
/// Prefill (L>1): always return a mask — causal for Global, windowed-causal for Sliding.
/// Decode  (L==1): Global layers see all cached keys → no mask (SDPA runs unmasked);
///                 Sliding layers restrict to the last `window` keys → explicit mask.
#[allow(non_snake_case)]
fn decode_mask(kind: LayerKind, L: i32, off: i32, window: i32) -> Result<Option<Array>> {
    if L > 1 {
        // Prefill: causal mask over the fresh sequence (off==0 on a cold cache).
        Ok(Some(match kind {
            LayerKind::Global  => full_causal_mask(L, off)?,
            LayerKind::Sliding => sliding_window_mask(L, off, window)?,
        }))
    } else {
        // Decode (L==1): single query at absolute position `off`.
        // Global: attends to all cached keys [0..off] → no mask needed.
        // Sliding: query at `off` must only see keys in [off-window, off] →
        //   sliding_window_mask(1, off, window) produces shape [1, off+1] where
        //   visible[0, k] = (off >= k) && (off <= k + window)
        //                 = k in [off-window, off]   ✓
        match kind {
            LayerKind::Global  => Ok(None),
            LayerKind::Sliding if off <= window => Ok(None),
            LayerKind::Sliding => Ok(Some(sliding_window_mask(1, off, window)?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_decode_before_window_needs_no_mask() {
        let mask = decode_mask(LayerKind::Sliding, 1, 1024, 1024).unwrap();

        assert!(mask.is_none());
    }

    #[test]
    fn sliding_decode_after_window_masks_old_keys() {
        let mask = decode_mask(LayerKind::Sliding, 1, 1025, 1024).unwrap().unwrap();
        let v: Vec<bool> = mask.as_slice::<bool>().to_vec();

        assert_eq!(v.len(), 1026);
        assert!(!v[0]);
        assert!(v[1]);
        assert!(v[1025]);
    }
}
