use anyhow::{Context, Result};
use mlx_rs::{ops, Array};
use std::{collections::HashMap, path::Path};

pub(crate) struct Weights(HashMap<String, Array>);

impl Weights {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("model.safetensors");
        Ok(Self(
            Array::load_safetensors(&path)
                .with_context(|| format!("loading {}", path.display()))?,
        ))
    }

    pub fn take(&mut self, key: &str) -> Result<Array> {
        self.0
            .remove(key)
            .with_context(|| format!("missing PaddleOCR tensor {key}"))
    }
}

pub(crate) struct Linear {
    pub weight: Array,
    pub bias: Option<Array>,
}

impl Linear {
    pub fn load(weights: &mut Weights, prefix: &str, bias: bool) -> Result<Self> {
        Ok(Self {
            weight: weights.take(&format!("{prefix}.weight"))?,
            bias: bias
                .then(|| weights.take(&format!("{prefix}.bias")))
                .transpose()?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        Ok(match &self.bias {
            Some(b) => ops::addmm(b, x, self.weight.t(), None, None)?,
            None => ops::matmul(x, self.weight.t())?,
        })
    }
}
