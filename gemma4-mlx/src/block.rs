//! Gemma 4 decoder layer (TransformerBlock).
//!
//! Replicates mlx-vlm `DecoderLayer.__call__` for the 12B model (per-layer-input
//! gating OFF, MoE OFF):
//! ```text
//! residual = x
//! h = input_layernorm(x)
//! h = self_attn(h, mask)
//! h = post_attention_layernorm(h)
//! h = residual + h
//! residual = h
//! h = pre_feedforward_layernorm(h)
//! h = mlp(h)
//! h = post_feedforward_layernorm(h)
//! h = residual + h
//! h = h * layer_scalar
//! ```

use std::collections::HashMap;

use mlx_rs::{ops::multiply, Array};
use mlx_rs_core::cache::KVCache;

use crate::attention::Attention;
use crate::config::{ModelArgs, QuantConfig};
use crate::error::Result;
use crate::mlp::Mlp;
use crate::norm::GemmaRmsNorm;
use crate::weights::get_weight;

/// One Gemma 4 decoder layer: 4 RMSNorms, two residuals, trailing `layer_scalar`.
pub struct TransformerBlock {
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    pre_feedforward_layernorm: GemmaRmsNorm,
    post_feedforward_layernorm: GemmaRmsNorm,
    self_attn: Attention,
    mlp: Mlp,
    layer_scalar: Array,
}

impl TransformerBlock {
    /// Build the decoder layer at `layer_idx` from pre-loaded weights + config.
    pub fn from_weights(
        weights: &HashMap<String, Array>,
        args: &ModelArgs,
        quant: &QuantConfig,
        layer_idx: i32,
    ) -> Result<Self> {
        let base = format!("language_model.model.layers.{layer_idx}");
        let norm = |name: &str| -> Result<GemmaRmsNorm> {
            Ok(GemmaRmsNorm::from_weight(
                get_weight(weights, &format!("{base}.{name}.weight"))?,
                args.rms_norm_eps,
            ))
        };

        Ok(Self {
            input_layernorm: norm("input_layernorm")?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
            pre_feedforward_layernorm: norm("pre_feedforward_layernorm")?,
            post_feedforward_layernorm: norm("post_feedforward_layernorm")?,
            self_attn: Attention::from_weights(weights, args, quant, layer_idx)?,
            mlp: Mlp::from_weights(weights, quant, layer_idx)?,
            layer_scalar: get_weight(weights, &format!("{base}.layer_scalar"))?,
        })
    }

    /// Forward. `x` is `[B, L, hidden]`, `mask` a bool `[L, L]` (true=visible).
    pub fn forward(&mut self, x: &Array, mask: &Array) -> Result<Array> {
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, mask)?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = residual.add(&h)?;

        let residual = h.clone();
        let hh = self.pre_feedforward_layernorm.forward(&h)?;
        let hh = self.mlp.forward(&hh)?;
        let hh = self.post_feedforward_layernorm.forward(&hh)?;
        let hh = residual.add(&hh)?;

        Ok(multiply(&hh, &self.layer_scalar)?)
    }

    /// Cache-aware forward for single-token (or chunked) decode. Identical to
    /// [`TransformerBlock::forward`] except attention threads `cache` (and the
    /// RoPE offset derived from it) through [`Attention::attend`]. `mask` is
    /// optional — `None` lets SDPA run unmasked (single-token decode).
    pub fn forward_cached(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KVCache,
    ) -> Result<Array> {
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.attend(&h, mask, Some(cache))?;
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = residual.add(&h)?;

        let residual = h.clone();
        let hh = self.pre_feedforward_layernorm.forward(&h)?;
        let hh = self.mlp.forward(&hh)?;
        let hh = self.post_feedforward_layernorm.forward(&hh)?;
        let hh = residual.add(&hh)?;

        Ok(multiply(&hh, &self.layer_scalar)?)
    }
}
