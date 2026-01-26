//! Audio-to-LLM adaptor
//!
//! Projects audio encoder features to LLM embedding space.
//!
//! Architecture:
//! - Conv1d (1280 → 1280, k=3, s=2, p=1) + GELU (2x downsample)
//! - Linear (1280 → 2048) + ReLU
//! - Linear (2048 → 3584)
//!
//! Total: 25Hz → 12.5Hz, 1280 dim → 3584 dim
//!
//! Note: Uses ReLU for linear layers (matching Python implementation).

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    Array,
};

use crate::config::AdaptorConfig;
use crate::error::Result;

/// Audio-to-LLM adaptor
///
/// Projects 1280-dim encoder features to 3584-dim LLM embedding space
/// with additional 2x temporal downsampling.
#[derive(Debug, Clone, ModuleParameters)]
pub struct StepAudio2Adaptor {
    /// Conv1d for temporal downsampling (stride=2)
    #[param]
    pub conv: nn::Conv1d,
    /// First linear projection (1280 → 2048)
    #[param]
    pub linear1: nn::Linear,
    /// Second linear projection (2048 → 3584)
    #[param]
    pub linear2: nn::Linear,
    /// Configuration
    pub config: AdaptorConfig,
}

impl StepAudio2Adaptor {
    /// Create a new adaptor
    pub fn new(config: AdaptorConfig) -> Result<Self> {
        let encoder_dim = config.encoder_dim;
        let hidden_dim = config.hidden_dim;
        let llm_dim = config.llm_dim;
        let kernel_size = config.kernel_size;
        let stride = config.stride;

        // Conv1d: (encoder_dim, encoder_dim, k=3, s=2, p=1)
        // This provides 2x temporal downsampling
        let padding = (kernel_size - 1) / 2;
        let conv = nn::Conv1dBuilder::new(encoder_dim, encoder_dim, kernel_size)
            .stride(stride)
            .padding(padding)
            .build()?;

        // Linear layers for dimension projection
        let linear1 = nn::LinearBuilder::new(encoder_dim, hidden_dim)
            .bias(true)
            .build()?;
        let linear2 = nn::LinearBuilder::new(hidden_dim, llm_dim)
            .bias(true)
            .build()?;

        Ok(Self {
            conv,
            linear1,
            linear2,
            config,
        })
    }

    /// Get output dimension
    pub fn output_dim(&self) -> i32 {
        self.config.llm_dim
    }
}

impl Module<&Array> for StepAudio2Adaptor {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        // x: [B, T, encoder_dim] from encoder
        // MLX Conv1d expects [B, L, C] format which matches our input

        // Conv1d with stride 2 (temporal downsampling)
        let x = self.conv.forward(x)?;
        let x = nn::gelu(&x)?;
        // x: [B, T/2, encoder_dim]

        // Linear projections with ReLU (matching Python implementation)
        let x = self.linear1.forward(&x)?;
        let x = nn::relu(&x)?;  // ReLU, not GELU!
        let x = self.linear2.forward(&x)?;

        // x: [B, T/2, llm_dim]
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptor_creation() {
        let config = AdaptorConfig::default();
        let adaptor = StepAudio2Adaptor::new(config);
        assert!(adaptor.is_ok());
    }

    #[test]
    fn test_adaptor_forward() {
        let config = AdaptorConfig::default();
        let mut adaptor = StepAudio2Adaptor::new(config).unwrap();

        // Input: [1, 100, 1280]
        let input = Array::zeros::<f32>(&[1, 100, 1280]).unwrap();
        let output = adaptor.forward(&input);

        match &output {
            Ok(_) => {}
            Err(e) => eprintln!("Adaptor forward error: {:?}", e),
        }
        assert!(output.is_ok());
        let output = output.unwrap();
        // Output should be [1, 50, 3584] (2x downsampling)
        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 50); // T/2
        assert_eq!(output.shape()[2], 3584);
    }
}
