//! Fused MLP projector for mapping vision features to LLM embedding space.

use std::collections::HashMap;

use mlx_rs::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    quantization::{MaybeQuantized, Quantizable},
    Array,
};

use crate::error::Error;

/// 3-layer MLP with GELU activations (Prismatic "fused-gelu-mlp" projector).
///
/// Maps fused DINOv2+SigLIP features (2176-dim) to LLM embedding space (4096-dim).
#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct FusedMLPProjector {
    #[param]
    pub fc1: MaybeQuantized<nn::Linear>,
    #[param]
    pub fc2: MaybeQuantized<nn::Linear>,
    #[param]
    pub fc3: MaybeQuantized<nn::Linear>,
}

impl Module<&Array> for FusedMLPProjector {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let h = self.fc1.forward(x)?;
        let h = nn::gelu(h)?;
        let h = self.fc2.forward(&h)?;
        let h = nn::gelu(h)?;
        self.fc3.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.fc1.training_mode(mode);
        self.fc2.training_mode(mode);
        self.fc3.training_mode(mode);
    }
}

impl FusedMLPProjector {
    pub fn quantize(self, group_size: i32, bits: i32) -> Result<Self, Exception> {
        Ok(Self {
            fc1: self.fc1.try_into_quantized(group_size, bits)?,
            fc2: self.fc2.try_into_quantized(group_size, bits)?,
            fc3: self.fc3.try_into_quantized(group_size, bits)?,
        })
    }
}

/// Load projector weights from a weight map.
pub fn load_projector(
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<FusedMLPProjector, Error> {
    let fc1 = load_linear(weights, prefix, "fc1", Some("0"))?;
    let fc2 = load_linear(weights, prefix, "fc2", Some("2"))?;
    let fc3 = load_linear(weights, prefix, "fc3", Some("4"))?;

    Ok(FusedMLPProjector { fc1, fc2, fc3 })
}

fn load_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    name: &str,
    fallback_idx: Option<&str>,
) -> Result<MaybeQuantized<nn::Linear>, Error> {
    let w_key = format!("{}.{}.weight", prefix, name);
    let b_key = format!("{}.{}.bias", prefix, name);

    let weight = if let Some(w) = weights.get(&w_key) {
        w.clone()
    } else if let Some(idx) = fallback_idx {
        let fallback_key = format!("{}.{}.weight", prefix, idx);
        weights
            .get(&fallback_key)
            .cloned()
            .ok_or_else(|| Error::WeightNotFound(format!("{} or {}", w_key, fallback_key)))?
    } else {
        return Err(Error::WeightNotFound(w_key));
    };

    let bias = weights.get(&b_key).cloned().or_else(|| {
        fallback_idx
            .and_then(|idx| weights.get(&format!("{}.{}.bias", prefix, idx)).cloned())
    });

    Ok(MaybeQuantized::new(nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(bias),
    }))
}
