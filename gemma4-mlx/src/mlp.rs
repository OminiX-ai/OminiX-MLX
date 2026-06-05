//! Gemma 4 GeGLU MLP.
//!
//! Replicates mlx-vlm `MLP`:
//! ```text
//! down_proj(gelu_approx(gate_proj(x)) * up_proj(x))
//! ```
//! `gelu_approx` is the tanh ("pytorch_tanh") GELU. For the 12B model all three
//! projections (gate/up/down) are 8-bit quantized; `quant_for(prefix)` resolves
//! the per-module (bits, group_size).

use std::collections::HashMap;

use mlx_rs::{module::Module, nn, ops::multiply, Array};

use crate::config::QuantConfig;
use crate::error::Result;
use crate::weights::make_quantized_linear;

/// Gemma 4 gated MLP (GeGLU): `down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))`.
pub struct Mlp {
    gate_proj: nn::QuantizedLinear,
    up_proj: nn::QuantizedLinear,
    down_proj: nn::QuantizedLinear,
}

impl Mlp {
    /// Build the MLP for layer `layer_idx` from pre-loaded weights.
    ///
    /// Weight key prefix is `language_model.model.layers.{layer_idx}.mlp.*`.
    pub fn from_weights(
        weights: &HashMap<String, Array>,
        quant: &QuantConfig,
        layer_idx: i32,
    ) -> Result<Self> {
        let base = format!("language_model.model.layers.{layer_idx}.mlp");
        let load = |name: &str| -> Result<nn::QuantizedLinear> {
            let prefix = format!("{base}.{name}");
            let (bits, group_size) = quant.quant_for(&prefix);
            make_quantized_linear(weights, &prefix, group_size, bits)
        };
        Ok(Self {
            gate_proj: load("gate_proj")?,
            up_proj: load("up_proj")?,
            down_proj: load("down_proj")?,
        })
    }

    /// Forward: `down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))`.
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = nn::gelu_approximate(&gate)?;
        let gated = multiply(&activated, &up)?;
        Ok(self.down_proj.forward(&gated)?)
    }
}
