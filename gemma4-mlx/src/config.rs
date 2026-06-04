//! Gemma 4 text config parsing.
use std::collections::HashMap;
use serde::Deserialize;
use serde_json::Value;
use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind { Sliding, Global }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType { Default, Proportional }

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeSpec {
    pub rope_type: RopeType,
    pub theta: f32,
    /// Only meaningful for Proportional; Default uses 1.0 (full-dim rotation).
    pub partial_rotary_factor: f32,
}

#[derive(Debug, Clone)]
pub struct ModelArgs {
    pub num_hidden_layers: i32,
    pub hidden_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub global_head_dim: i32,
    pub num_global_key_value_heads: i32,
    pub attention_k_eq_v: bool,
    pub intermediate_size: i32,
    pub vocab_size: i32,
    pub tie_word_embeddings: bool,
    pub rms_norm_eps: f32,
    pub final_logit_softcapping: f32,
    pub sliding_window: i32,
    pub max_position_embeddings: i32,
    pub layer_types: Vec<LayerKind>,
    pub rope_sliding: RopeSpec,
    pub rope_global: RopeSpec,
}

#[derive(Deserialize)]
struct Root { text_config: TextConfig }

#[derive(Deserialize)]
struct TextConfig {
    num_hidden_layers: i32,
    hidden_size: i32,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    head_dim: i32,
    global_head_dim: i32,
    num_global_key_value_heads: i32,
    // Gemma 4 12B sets this true (global layers share K/V). Default true so an
    // absent field doesn't silently flip global-layer behavior.
    #[serde(default = "default_true")] attention_k_eq_v: bool,
    intermediate_size: i32,
    vocab_size: i32,
    tie_word_embeddings: bool,
    rms_norm_eps: f32,
    #[serde(default)] final_logit_softcapping: f32,
    sliding_window: i32,
    max_position_embeddings: i32,
    layer_types: Vec<String>,
    rope_parameters: RopeParameters,
}

#[derive(Deserialize)]
struct RopeParameters { full_attention: RopeRaw, sliding_attention: RopeRaw }

#[derive(Deserialize)]
struct RopeRaw {
    rope_type: String,
    rope_theta: f32,
    #[serde(default = "default_partial")] partial_rotary_factor: f32,
}
fn default_partial() -> f32 { 1.0 }
fn default_true() -> bool { true }

impl ModelArgs {
    pub fn from_config_str(s: &str) -> Result<Self> {
        let root: Root = serde_json::from_str(s)?;
        let t = root.text_config;

        let layer_types = t.layer_types.iter().map(|s| match s.as_str() {
            "sliding_attention" => Ok(LayerKind::Sliding),
            "full_attention"    => Ok(LayerKind::Global),
            other               => Err(Error::Config(format!("unknown layer_type {other}"))),
        }).collect::<Result<Vec<_>>>()?;

        if layer_types.len() != t.num_hidden_layers as usize {
            return Err(Error::Config(format!(
                "layer_types has {} entries but num_hidden_layers is {}",
                layer_types.len(), t.num_hidden_layers
            )));
        }

        let parse_rope = |r: &RopeRaw| -> Result<RopeSpec> {
            let rope_type = match r.rope_type.as_str() {
                // mlx-rs-core treats "linear" as plain "default" scaling.
                "default" | "linear" => RopeType::Default,
                "proportional"       => RopeType::Proportional,
                other                => return Err(Error::Config(format!("unknown rope_type {other}"))),
            };
            Ok(RopeSpec { rope_type, theta: r.rope_theta, partial_rotary_factor: r.partial_rotary_factor })
        };

        Ok(ModelArgs {
            num_hidden_layers:         t.num_hidden_layers,
            hidden_size:               t.hidden_size,
            num_attention_heads:       t.num_attention_heads,
            num_key_value_heads:       t.num_key_value_heads,
            head_dim:                  t.head_dim,
            global_head_dim:           t.global_head_dim,
            num_global_key_value_heads: t.num_global_key_value_heads,
            attention_k_eq_v:          t.attention_k_eq_v,
            intermediate_size:         t.intermediate_size,
            vocab_size:                t.vocab_size,
            tie_word_embeddings:       t.tie_word_embeddings,
            rms_norm_eps:              t.rms_norm_eps,
            final_logit_softcapping:   t.final_logit_softcapping,
            sliding_window:            t.sliding_window,
            max_position_embeddings:   t.max_position_embeddings,
            layer_types,
            rope_sliding: parse_rope(&t.rope_parameters.sliding_attention)?,
            rope_global:  parse_rope(&t.rope_parameters.full_attention)?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub default_bits: i32,
    pub default_group_size: i32,
    /// Full weight prefix -> (bits, group_size)
    overrides: HashMap<String, (i32, i32)>,
}

impl QuantConfig {
    /// Accepts either `quantization` or `quantization_config`; returns Ok(None) if neither present.
    pub fn from_config_str(s: &str) -> Result<Option<Self>> {
        let v: Value = serde_json::from_str(s)?;
        let q = v.get("quantization").or_else(|| v.get("quantization_config"));
        let Some(q) = q else { return Ok(None) };
        let obj = q.as_object().ok_or_else(|| Error::Config("quantization not object".into()))?;

        let default_bits = obj.get("bits").and_then(|x| x.as_i64()).unwrap_or(4) as i32;
        let default_group_size = obj.get("group_size").and_then(|x| x.as_i64()).unwrap_or(64) as i32;

        let mut overrides = HashMap::new();
        for (k, val) in obj {
            // Heuristic: any nested object with both bits+group_size is a per-module
            // override keyed by exact module prefix. String values like "mode":"affine"
            // are skipped by as_object(); "mode" excluded by name as belt-and-suspenders.
            // A future non-module {bits,group_size} object would be a harmless false
            // positive (no one calls quant_for with its key).
            if let Some(o) = val.as_object() {
                if let (Some(b), Some(g)) = (o.get("bits").and_then(|x| x.as_i64()),
                                             o.get("group_size").and_then(|x| x.as_i64())) {
                    if k != "mode" {
                        overrides.insert(k.clone(), (b as i32, g as i32));
                    }
                }
            }
        }
        Ok(Some(QuantConfig { default_bits, default_group_size, overrides }))
    }

    /// Returns (bits, group_size) for the given module prefix (without
    /// .weight/.scales/.biases). Lookup is **exact equality** — not a
    /// prefix/glob match; an unknown key falls back to the defaults.
    /// Glob/regex matching is intentionally not implemented (YAGNI).
    pub fn quant_for(&self, prefix: &str) -> (i32, i32) {
        self.overrides.get(prefix)
            .copied()
            .unwrap_or((self.default_bits, self.default_group_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> ModelArgs {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/config.text.json");
        let s = std::fs::read_to_string(path).unwrap();
        ModelArgs::from_config_str(&s).unwrap()
    }

    #[test]
    fn parses_dims_and_layer_pattern() {
        let a = fixture();
        assert_eq!(a.num_hidden_layers, 6);
        assert_eq!(a.hidden_size, 3840);
        assert_eq!(a.head_dim, 256);
        assert_eq!(a.global_head_dim, 512);
        assert_eq!(a.num_key_value_heads, 8);
        assert_eq!(a.num_global_key_value_heads, 1);
        assert!(a.attention_k_eq_v);
        assert_eq!(a.final_logit_softcapping, 30.0);
        assert_eq!(a.layer_types[0], LayerKind::Sliding);
        assert_eq!(a.layer_types[4], LayerKind::Sliding);
        assert_eq!(a.layer_types[5], LayerKind::Global);
    }

    #[test]
    fn parses_dual_rope() {
        let a = fixture();
        assert_eq!(a.rope_sliding.rope_type, RopeType::Default);
        assert_eq!(a.rope_sliding.theta, 10000.0);
        assert_eq!(a.rope_global.rope_type, RopeType::Proportional);
        assert_eq!(a.rope_global.theta, 1_000_000.0);
        assert!((a.rope_global.partial_rotary_factor - 0.25).abs() < 1e-6);
    }

    #[test]
    fn quant_for_resolves_per_module_overrides() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/config.text.json");
        let s = std::fs::read_to_string(path).unwrap();
        let q = QuantConfig::from_config_str(&s).unwrap().unwrap();
        assert_eq!(q.quant_for("language_model.model.layers.5.self_attn.q_proj"), (4, 64));
        assert_eq!(q.quant_for("language_model.model.layers.5.mlp.gate_proj"), (8, 64));
        assert_eq!(q.quant_for("language_model.model.layers.5.mlp.up_proj"), (8, 64));
    }
}
