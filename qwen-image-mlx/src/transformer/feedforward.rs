//! Feed-forward network with GELU activation (for DiT blocks)
//!
//! Reference: diffusers QwenImageTransformerBlock "Independent MLP layers per stream"
//! Note: Uses GELU, not SwiGLU like the text encoder

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn::{self, Linear, LinearBuilder};
use mlx_rs::Array;

/// Feed-forward network with GELU activation
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenFeedForward {
    #[param]
    pub mlp_in: Linear,
    #[param]
    pub mlp_out: Linear,
}

impl QwenFeedForward {
    pub fn new(dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            mlp_in: LinearBuilder::new(dim, 4 * dim).build()?,
            mlp_out: LinearBuilder::new(4 * dim, dim).build()?,
        })
    }
}

impl Module<&Array> for QwenFeedForward {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        self.mlp_in.training_mode(mode);
        self.mlp_out.training_mode(mode);
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let h = self.mlp_in.forward(x)?;
        let h = nn::gelu_approximate(&h)?; // GELU, not SiLU!
        self.mlp_out.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedforward() {
        let mut ff = QwenFeedForward::new(64).unwrap();
        let x = Array::zeros::<f32>(&[2, 10, 64]).unwrap();
        let y = ff.forward(&x).unwrap();
        assert_eq!(y.shape(), x.shape());
    }
}
