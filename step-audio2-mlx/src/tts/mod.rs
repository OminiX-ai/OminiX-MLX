//! TTS (Text-to-Speech) Decoder for Step-Audio 2
//!
//! This module implements the complete TTS pipeline:
//! 1. Audio token extraction from LLM output
//! 2. S3Tokenizer: Mel spectrogram → audio codes (MLX-accelerated)
//! 3. Flow decoder: Audio codes → mel spectrogram (includes codebook + encoder + DiT)
//! 4. HiFi-GAN vocoder: Mel spectrogram → waveform
//!
//! Pipeline for voice cloning:
//! ```text
//! Reference Audio
//!     → Mel Spectrogram (audio.rs)
//!     → S3Tokenizer (MLX-accelerated)
//!     → Audio Codes [0-6560]
//!     → Flow Decoder (codebook + conformer + DiT)
//!     → 80-dim Mel Spectrogram
//!     → HiFi-GAN (256x upsample)
//!     → 24kHz Waveform
//! ```

pub mod audio_tokens;
pub mod flow;
pub mod hifigan;
pub mod s3tokenizer;
pub mod s3tokenizer_mlx;

pub use audio_tokens::{extract_audio_tokens, AudioTokenExtractor};
pub use flow::{FlowDecoder, FlowDecoderConfig};
pub use hifigan::{HiFiGAN, HiFiGANConfig};
pub use s3tokenizer::{S3Tokenizer, S3TokenizerConfig};
pub use s3tokenizer_mlx::{S3TokenizerMLX, S3TokenizerMLXConfig};

use std::path::Path;
use crate::error::{Error, Result};

/// TTS decoder configuration
#[derive(Debug, Clone)]
pub struct TTSDecoderConfig {
    /// Flow decoder configuration
    pub flow: FlowDecoderConfig,
    /// HiFi-GAN configuration
    pub hifigan: HiFiGANConfig,
    /// Output sample rate
    pub output_sample_rate: i32,
}

impl Default for TTSDecoderConfig {
    fn default() -> Self {
        Self {
            flow: FlowDecoderConfig::default(),
            hifigan: HiFiGANConfig::default(),
            output_sample_rate: 24000,
        }
    }
}

/// Complete TTS decoder pipeline
pub struct TTSDecoder {
    /// Flow decoder (codebook + encoder + DiT)
    pub flow: FlowDecoder,
    /// HiFi-GAN vocoder
    pub hifigan: HiFiGAN,
    /// Configuration
    pub config: TTSDecoderConfig,
    /// Whether the decoder is loaded
    pub loaded: bool,
}

impl TTSDecoder {
    /// Create a new TTS decoder with default (unloaded) weights
    pub fn new(config: TTSDecoderConfig) -> Result<Self> {
        Ok(Self {
            flow: FlowDecoder::new(config.flow.clone())?,
            hifigan: HiFiGAN::new(config.hifigan.clone())?,
            config,
            loaded: false,
        })
    }

    /// Load TTS decoder from model directory
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Check for converted weights
        let tts_dir = model_dir.join("tts_mlx");
        if !tts_dir.exists() {
            return Err(Error::ModelLoad(format!(
                "TTS weights not found. Run scripts/convert_tts_weights.py first. Expected: {:?}",
                tts_dir
            )));
        }

        // Load Flow decoder
        let flow = FlowDecoder::load(model_dir)?;

        // Load HiFi-GAN
        let hifigan = HiFiGAN::load(model_dir)?;

        Ok(Self {
            flow,
            hifigan,
            config: TTSDecoderConfig::default(),
            loaded: true,
        })
    }

    /// Synthesize audio from audio token codes
    ///
    /// # Arguments
    /// * `codes` - Audio codebook indices (0-6560)
    ///
    /// # Returns
    /// Audio waveform at 24kHz as Vec<f32>
    pub fn synthesize(&mut self, codes: &[i32]) -> Result<Vec<f32>> {
        if codes.is_empty() {
            return Ok(vec![]);
        }

        // Validate codes
        for &code in codes {
            if code < 0 || code >= 6561 {
                return Err(Error::Inference(format!(
                    "Invalid audio code {}: must be in [0, 6561)",
                    code
                )));
            }
        }

        // Flow decoder: codes → mel spectrogram
        let mel = self.flow.generate(codes)?;

        // HiFi-GAN: mel → waveform
        let waveform = self.hifigan.synthesize(&mel)?;

        // Convert to Vec<f32>
        // waveform: [1, 1, T] -> flatten
        let audio_data: Vec<f32> = waveform.as_slice::<f32>().to_vec();

        Ok(audio_data)
    }

    /// Synthesize audio from raw audio tokens (with offset)
    ///
    /// # Arguments
    /// * `audio_tokens` - Audio token IDs from LLM (range 151696-158256)
    ///
    /// # Returns
    /// Audio waveform at 24kHz as Vec<f32>
    pub fn synthesize_from_tokens(&mut self, audio_tokens: &[i32]) -> Result<Vec<f32>> {
        // Extract codes from tokens
        let codes = extract_audio_tokens(audio_tokens);
        if codes.is_empty() {
            return Ok(vec![]);
        }

        self.synthesize(&codes)
    }

    /// Get output sample rate
    pub fn sample_rate(&self) -> i32 {
        self.config.output_sample_rate
    }

    /// Check if decoder is loaded
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tts_config_default() {
        let config = TTSDecoderConfig::default();
        assert_eq!(config.output_sample_rate, 24000);
    }

    #[test]
    fn test_tts_decoder_creation() {
        let config = TTSDecoderConfig::default();
        let decoder = TTSDecoder::new(config);
        assert!(decoder.is_ok());
        assert!(!decoder.unwrap().is_loaded());
    }
}
