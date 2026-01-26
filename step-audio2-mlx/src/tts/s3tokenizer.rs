//! S3Tokenizer for Step-Audio 2
//!
//! The S3Tokenizer (Speech Semantic Tokenizer) converts:
//! - Audio waveforms → discrete codes (encode)
//! - Discrete codes → semantic features (decode)
//!
//! Step-Audio 2 uses a CosyVoice2 speech tokenizer with:
//! - Frame rate: 25 Hz
//! - Codebook size: 6561 entries
//! - ONNX model file: speech_tokenizer_v2_25hz.onnx
//!
//! This implementation provides both ONNX runtime integration (when available)
//! and a pure MLX fallback for the decode path.

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    Array,
};

use crate::error::{Error, Result};

/// S3Tokenizer configuration
#[derive(Debug, Clone)]
pub struct S3TokenizerConfig {
    /// Frame rate in Hz (tokens per second)
    pub frame_rate: i32,
    /// Codebook size
    pub codebook_size: i32,
    /// Hidden dimension for embeddings
    pub hidden_dim: i32,
    /// ONNX model path (optional)
    pub onnx_path: Option<String>,
}

impl Default for S3TokenizerConfig {
    fn default() -> Self {
        Self {
            frame_rate: 25,
            codebook_size: 6561,
            hidden_dim: 512,
            onnx_path: None,
        }
    }
}

/// Vector quantizer codebook
///
/// Maps discrete codes to continuous embeddings.
#[derive(Debug, Clone, ModuleParameters)]
pub struct Codebook {
    /// Embedding table [codebook_size, hidden_dim]
    #[param]
    pub embeddings: nn::Embedding,
    /// Codebook size
    pub size: i32,
    /// Hidden dimension
    pub dim: i32,
}

impl Codebook {
    /// Create a new codebook
    pub fn new(size: i32, dim: i32) -> Result<Self> {
        Ok(Self {
            embeddings: nn::Embedding::new(size, dim)?,
            size,
            dim,
        })
    }

    /// Look up embeddings for codes
    pub fn lookup(&mut self, codes: &Array) -> std::result::Result<Array, Exception> {
        self.embeddings.forward(codes)
    }
}

/// Post-network for refining codebook embeddings
///
/// A small transformer-based network that refines the raw codebook
/// embeddings into richer semantic features.
#[derive(Debug, Clone, ModuleParameters)]
pub struct PostNet {
    /// Layer normalization
    #[param]
    pub norm: nn::LayerNorm,
    /// Self-attention layers
    #[param]
    pub layers: Vec<PostNetLayer>,
    /// Output projection
    #[param]
    pub output_proj: nn::Linear,
}

/// Single PostNet layer
#[derive(Debug, Clone, ModuleParameters)]
pub struct PostNetLayer {
    /// Pre-attention norm
    #[param]
    pub norm1: nn::LayerNorm,
    /// Self-attention Q projection
    #[param]
    pub q_proj: nn::Linear,
    /// Self-attention K projection
    #[param]
    pub k_proj: nn::Linear,
    /// Self-attention V projection
    #[param]
    pub v_proj: nn::Linear,
    /// Self-attention output projection
    #[param]
    pub out_proj: nn::Linear,
    /// Pre-FFN norm
    #[param]
    pub norm2: nn::LayerNorm,
    /// FFN up projection
    #[param]
    pub ffn_up: nn::Linear,
    /// FFN down projection
    #[param]
    pub ffn_down: nn::Linear,
    /// Number of heads
    pub num_heads: i32,
    /// Head dimension
    pub head_dim: i32,
}

impl PostNetLayer {
    /// Create a new PostNet layer
    pub fn new(hidden_dim: i32, num_heads: i32) -> Result<Self> {
        let head_dim = hidden_dim / num_heads;
        let ffn_dim = hidden_dim * 4;

        Ok(Self {
            norm1: nn::LayerNormBuilder::new(hidden_dim).build()?,
            q_proj: nn::LinearBuilder::new(hidden_dim, hidden_dim).build()?,
            k_proj: nn::LinearBuilder::new(hidden_dim, hidden_dim).build()?,
            v_proj: nn::LinearBuilder::new(hidden_dim, hidden_dim).build()?,
            out_proj: nn::LinearBuilder::new(hidden_dim, hidden_dim).build()?,
            norm2: nn::LayerNormBuilder::new(hidden_dim).build()?,
            ffn_up: nn::LinearBuilder::new(hidden_dim, ffn_dim).build()?,
            ffn_down: nn::LinearBuilder::new(ffn_dim, hidden_dim).build()?,
            num_heads,
            head_dim,
        })
    }
}

impl Module<&Array> for PostNetLayer {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        let shape = x.shape();
        let (batch, seq_len, _) = (shape[0], shape[1], shape[2]);

        // Self-attention with residual
        let residual = x.clone();
        let h = self.norm1.forward(x)?;

        let q = self.q_proj.forward(&h)?;
        let k = self.k_proj.forward(&h)?;
        let v = self.v_proj.forward(&h)?;

        // Multi-head reshape
        let q = q
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let scale = (self.head_dim as f32).powf(-0.5);
        let attn = mlx_rs::fast::scaled_dot_product_attention(q, k, v, scale, None)?;

        let attn = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;

        let h = self.out_proj.forward(&attn)?;
        let x = residual.add(&h)?;

        // FFN with residual
        let residual = x.clone();
        let h = self.norm2.forward(&x)?;
        let h = self.ffn_up.forward(&h)?;
        let h = nn::gelu(&h)?;
        let h = self.ffn_down.forward(&h)?;

        residual.add(&h)
    }
}

impl PostNet {
    /// Create a new PostNet
    pub fn new(hidden_dim: i32, num_layers: i32, num_heads: i32) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(PostNetLayer::new(hidden_dim, num_heads)?);
        }

        Ok(Self {
            norm: nn::LayerNormBuilder::new(hidden_dim).build()?,
            layers,
            output_proj: nn::LinearBuilder::new(hidden_dim, hidden_dim).build()?,
        })
    }
}

impl Module<&Array> for PostNet {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        let mut h = x.clone();

        for layer in &mut self.layers {
            h = layer.forward(&h)?;
        }

        let h = self.norm.forward(&h)?;
        self.output_proj.forward(&h)
    }
}

/// S3Tokenizer for speech semantic tokenization
///
/// This implementation provides:
/// - `decode`: codes → semantic features (using MLX codebook + postnet)
/// - `encode`: (future) audio → codes (would require ONNX runtime)
#[derive(Debug, Clone, ModuleParameters)]
pub struct S3Tokenizer {
    /// Vector quantizer codebook
    #[param]
    pub codebook: Codebook,
    /// Post-processing network
    #[param]
    pub postnet: PostNet,
    /// Configuration
    pub config: S3TokenizerConfig,
}

impl S3Tokenizer {
    /// Create a new S3Tokenizer
    pub fn new(config: S3TokenizerConfig) -> Result<Self> {
        Ok(Self {
            codebook: Codebook::new(config.codebook_size, config.hidden_dim)?,
            postnet: PostNet::new(config.hidden_dim, 4, 8)?,
            config,
        })
    }

    /// Load S3Tokenizer from ONNX model path
    pub fn load(onnx_path: impl AsRef<std::path::Path>) -> Result<Self> {
        // TODO: Implement ONNX model loading
        // For now, create with default config
        let config = S3TokenizerConfig {
            onnx_path: Some(onnx_path.as_ref().to_string_lossy().to_string()),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Load codebook weights from safetensors
    pub fn load_codebook(&mut self, weights_path: impl AsRef<std::path::Path>) -> Result<()> {
        // TODO: Implement weight loading
        // The codebook embeddings would be loaded from:
        // - model.codebook.embeddings or
        // - quantizer.embedding
        Ok(())
    }

    /// Decode discrete codes to semantic features
    ///
    /// # Arguments
    /// * `codes` - Codebook indices [0, codebook_size)
    ///
    /// # Returns
    /// Semantic features [B, T, hidden_dim]
    pub fn decode(&mut self, codes: &[i32]) -> Result<Array> {
        if codes.is_empty() {
            return Err(Error::Inference("Empty codes array".to_string()));
        }

        // Validate codes
        for &code in codes {
            if code < 0 || code >= self.config.codebook_size {
                return Err(Error::Inference(format!(
                    "Invalid code {}: must be in [0, {})",
                    code, self.config.codebook_size
                )));
            }
        }

        // Convert to Array [1, T]
        let codes_array = Array::from_slice(codes, &[1, codes.len() as i32]);

        // Look up embeddings
        let embeddings = self
            .codebook
            .lookup(&codes_array)
            .map_err(|e| Error::Inference(format!("Codebook lookup failed: {}", e)))?;

        // Apply postnet for refinement
        let features = self
            .postnet
            .forward(&embeddings)
            .map_err(|e| Error::Inference(format!("PostNet forward failed: {}", e)))?;

        Ok(features)
    }

    /// Decode with batched codes
    pub fn decode_batch(&mut self, codes: &Array) -> Result<Array> {
        // Validate shape
        if codes.ndim() != 2 {
            return Err(Error::Inference(format!(
                "Expected 2D codes array, got {}D",
                codes.ndim()
            )));
        }

        // Look up embeddings
        let embeddings = self
            .codebook
            .lookup(codes)
            .map_err(|e| Error::Inference(format!("Codebook lookup failed: {}", e)))?;

        // Apply postnet
        let features = self
            .postnet
            .forward(&embeddings)
            .map_err(|e| Error::Inference(format!("PostNet forward failed: {}", e)))?;

        Ok(features)
    }

    /// Encode audio to discrete codes (requires ONNX runtime)
    ///
    /// Note: This is a placeholder. Full implementation would require
    /// ONNX runtime integration to run the encoder portion of the model.
    #[allow(unused)]
    pub fn encode(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<i32>> {
        Err(Error::Inference(
            "S3Tokenizer encode requires ONNX runtime (not yet implemented)".to_string(),
        ))
    }

    /// Get frame rate (tokens per second)
    pub fn frame_rate(&self) -> i32 {
        self.config.frame_rate
    }

    /// Get codebook size
    pub fn codebook_size(&self) -> i32 {
        self.config.codebook_size
    }

    /// Get hidden dimension
    pub fn hidden_dim(&self) -> i32 {
        self.config.hidden_dim
    }

    /// Estimate output duration from code count
    pub fn estimate_duration_secs(&self, num_codes: usize) -> f32 {
        num_codes as f32 / self.config.frame_rate as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s3tokenizer_config() {
        let config = S3TokenizerConfig::default();
        assert_eq!(config.frame_rate, 25);
        assert_eq!(config.codebook_size, 6561);
    }

    #[test]
    fn test_codebook_creation() {
        let codebook = Codebook::new(6561, 512);
        assert!(codebook.is_ok());
    }

    #[test]
    fn test_postnet_layer_creation() {
        let layer = PostNetLayer::new(512, 8);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_postnet_creation() {
        let postnet = PostNet::new(512, 4, 8);
        assert!(postnet.is_ok());
    }

    #[test]
    fn test_s3tokenizer_creation() {
        let tokenizer = S3Tokenizer::new(S3TokenizerConfig::default());
        assert!(tokenizer.is_ok());
    }

    #[test]
    fn test_estimate_duration() {
        let tokenizer = S3Tokenizer::new(S3TokenizerConfig::default()).unwrap();
        let duration = tokenizer.estimate_duration_secs(250); // 10 seconds at 25Hz
        assert!((duration - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_decode_empty_codes() {
        let mut tokenizer = S3Tokenizer::new(S3TokenizerConfig::default()).unwrap();
        let result = tokenizer.decode(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_invalid_code() {
        let mut tokenizer = S3Tokenizer::new(S3TokenizerConfig::default()).unwrap();
        let result = tokenizer.decode(&[0, 100, 7000]); // 7000 is out of range
        assert!(result.is_err());
    }
}
