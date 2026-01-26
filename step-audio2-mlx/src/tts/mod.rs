//! TTS (Text-to-Speech) Decoder for Step-Audio 2
//!
//! This module implements the complete TTS pipeline:
//! 1. Audio token extraction from LLM output
//! 2. S3Tokenizer: Audio tokens → semantic features
//! 3. Flow decoder: Semantic features → mel spectrogram
//! 4. HiFi-GAN vocoder: Mel spectrogram → waveform
//!
//! Pipeline:
//! ```text
//! Audio Tokens [151696-158256]
//!     → S3Tokenizer (decode codes → features)
//!     → Flow Decoder (10 steps, rectified flow)
//!     → 80-dim Mel Spectrogram
//!     → HiFi-GAN (256x upsample)
//!     → 24kHz Waveform
//! ```

pub mod audio_tokens;
pub mod flow;
pub mod hifigan;
pub mod s3tokenizer;

pub use audio_tokens::{extract_audio_tokens, AudioTokenExtractor};
pub use flow::{FlowDecoder, FlowDecoderConfig};
pub use hifigan::{HiFiGAN, HiFiGANConfig};
pub use s3tokenizer::{S3Tokenizer, S3TokenizerConfig};

use crate::error::{Error, Result};
use mlx_rs::module::Module;
use mlx_rs::Array;

/// TTS decoder configuration
#[derive(Debug, Clone)]
pub struct TTSDecoderConfig {
    /// S3Tokenizer configuration
    pub s3tokenizer: S3TokenizerConfig,
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
            s3tokenizer: S3TokenizerConfig::default(),
            flow: FlowDecoderConfig::default(),
            hifigan: HiFiGANConfig::default(),
            output_sample_rate: 24000,
        }
    }
}

/// Complete TTS decoder pipeline
pub struct TTSDecoder {
    /// S3Tokenizer for code-to-feature conversion
    pub s3tokenizer: S3Tokenizer,
    /// Flow matching decoder
    pub flow: FlowDecoder,
    /// HiFi-GAN vocoder
    pub hifigan: HiFiGAN,
    /// Configuration
    pub config: TTSDecoderConfig,
}

impl TTSDecoder {
    /// Create a new TTS decoder
    pub fn new(config: TTSDecoderConfig) -> Result<Self> {
        Ok(Self {
            s3tokenizer: S3Tokenizer::new(config.s3tokenizer.clone())?,
            flow: FlowDecoder::new(config.flow.clone())?,
            hifigan: HiFiGAN::new(config.hifigan.clone())?,
            config,
        })
    }

    /// Load TTS decoder from model directory
    pub fn load(model_dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Load S3Tokenizer (ONNX model)
        let s3tokenizer = S3Tokenizer::load(model_dir.join("speech_tokenizer_v2_25hz.onnx"))?;

        // Load Flow decoder weights
        let flow = FlowDecoder::load(model_dir.join("flow"))?;

        // Load HiFi-GAN weights
        let hifigan = HiFiGAN::load(model_dir.join("hifigan"))?;

        Ok(Self {
            s3tokenizer,
            flow,
            hifigan,
            config: TTSDecoderConfig::default(),
        })
    }

    /// Synthesize audio from audio tokens
    ///
    /// # Arguments
    /// * `audio_tokens` - Audio token IDs from LLM (range 151696-158256)
    /// * `prompt_audio` - Optional reference audio for voice cloning
    ///
    /// # Returns
    /// Audio waveform at 24kHz
    pub fn synthesize(
        &mut self,
        audio_tokens: &[i32],
        prompt_audio: Option<&Array>,
    ) -> Result<Vec<f32>> {
        if audio_tokens.is_empty() {
            return Ok(vec![]);
        }

        // 1. Extract codebook indices from audio tokens
        let codes = extract_audio_tokens(audio_tokens);
        if codes.is_empty() {
            return Ok(vec![]);
        }

        // 2. S3Tokenizer: codes → semantic features
        let semantic_features = self.s3tokenizer.decode(&codes)?;

        // 3. Flow decoder: semantic features → mel spectrogram
        let mel = self.flow.generate(&semantic_features, prompt_audio, None)?;

        // 4. HiFi-GAN: mel → waveform
        let waveform = self.hifigan.forward(&mel)
            .map_err(|e| Error::Inference(format!("HiFi-GAN forward failed: {}", e)))?;

        // Convert to Vec<f32>
        let audio_data: Vec<f32> = waveform.as_slice::<f32>().to_vec();

        Ok(audio_data)
    }

    /// Synthesize audio from text (requires external text-to-token conversion)
    pub fn synthesize_from_codes(&mut self, codes: &[i32]) -> Result<Vec<f32>> {
        if codes.is_empty() {
            return Ok(vec![]);
        }

        // S3Tokenizer: codes → semantic features
        let semantic_features = self.s3tokenizer.decode(codes)?;

        // Flow decoder: semantic features → mel spectrogram
        let mel = self.flow.generate(&semantic_features, None, None)?;

        // HiFi-GAN: mel → waveform
        let waveform = self.hifigan.forward(&mel)
            .map_err(|e| Error::Inference(format!("HiFi-GAN forward failed: {}", e)))?;

        // Convert to Vec<f32>
        let audio_data: Vec<f32> = waveform.as_slice::<f32>().to_vec();

        Ok(audio_data)
    }

    /// Get output sample rate
    pub fn sample_rate(&self) -> i32 {
        self.config.output_sample_rate
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
}
