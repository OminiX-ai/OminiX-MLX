//! FunASR speech recognition on Apple Silicon using MLX
//!
//! This crate provides GPU-accelerated Chinese speech recognition using the
//! Paraformer-large model from FunASR, optimized for Apple Silicon via MLX.
//!
//! # Features
//!
//! - **Non-autoregressive ASR**: Predicts all tokens in parallel (18x+ real-time)
//! - **Pure Rust**: No Python dependencies at runtime
//! - **GPU Accelerated**: Metal GPU via MLX for all operations
//! - **High Quality**: FunASR-compatible audio preprocessing
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use funasr_mlx::{load_model, parse_cmvn_file, transcribe};
//! use funasr_mlx::audio::{load_wav, resample};
//!
//! // Load audio
//! let (samples, sample_rate) = load_wav("audio.wav")?;
//! let samples = resample(&samples, sample_rate, 16000);
//!
//! // Load model
//! let mut model = load_model("paraformer.safetensors")?;
//! let (addshift, rescale) = parse_cmvn_file("am.mvn")?;
//! model.set_cmvn(addshift, rescale);
//!
//! // Transcribe
//! let audio = mlx_rs::Array::from_slice(&samples, &[samples.len() as i32]);
//! let token_ids = model.transcribe(&audio)?;
//! ```
//!
//! # Model Files
//!
//! You need to download and convert the FunASR Paraformer-large model:
//!
//! 1. **Weights**: `paraformer.safetensors` (converted from FunASR PyTorch)
//! 2. **CMVN**: `am.mvn` (from FunASR model directory)
//! 3. **Vocabulary**: `tokens.txt` or `vocab.txt` (8404 tokens)
//!
//! # Architecture
//!
//! The Paraformer model consists of:
//!
//! - **Mel Frontend**: 80-bin mel spectrogram with LFR stacking (7 frames, stride 6)
//! - **SAN-M Encoder**: 50-layer self-attention with FSMN memory enhancement
//! - **CIF Predictor**: Continuous integrate-and-fire for acoustic alignment
//! - **Bidirectional Decoder**: 16-layer transformer decoder

pub mod audio;
pub mod error;
pub mod paraformer;

// Re-export main types for convenience
pub use error::{Error, Result};
pub use paraformer::{
    load_model, load_model_with_config, parse_cmvn_file, DecoderInput, MelFrontend, Paraformer,
    ParaformerConfig,
};

/// Vocabulary for decoding token IDs to text
pub struct Vocabulary {
    tokens: Vec<String>,
}

impl Vocabulary {
    /// Load vocabulary from a text file (one token per line)
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let tokens: Vec<String> = content.lines().map(|s| s.to_string()).collect();
        Ok(Self { tokens })
    }

    /// Decode token IDs to text, filtering special tokens
    pub fn decode(&self, token_ids: &[i32]) -> String {
        token_ids
            .iter()
            .filter_map(|&id| {
                let id = id as usize;
                if id < self.tokens.len() {
                    let token = &self.tokens[id];
                    // Filter special tokens
                    if token == "<blank>"
                        || token == "<s>"
                        || token == "</s>"
                        || token == "<unk>"
                        || token == "<pad>"
                    {
                        None
                    } else {
                        Some(token.clone())
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get the number of tokens in vocabulary
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// High-level transcription function
///
/// # Arguments
/// * `model` - Loaded Paraformer model with CMVN set
/// * `audio` - Audio samples as f32 in range [-1, 1]
/// * `vocab` - Vocabulary for decoding
///
/// # Returns
/// Transcribed text
pub fn transcribe(
    model: &mut Paraformer,
    audio: &[f32],
    vocab: &Vocabulary,
) -> Result<String> {
    use mlx_rs::transforms::eval;

    let audio_array = mlx_rs::Array::from_slice(audio, &[audio.len() as i32]);
    let token_ids = model.transcribe(&audio_array)?;
    eval([&token_ids])?;

    let token_ids_vec: Vec<i32> = token_ids
        .try_as_slice::<i32>()
        .map_err(|_| Error::Audio("Failed to get token IDs".into()))?
        .to_vec();

    Ok(vocab.decode(&token_ids_vec))
}
