//! Voice Cloning API for GPT-SoVITS
//!
//! Provides a high-level API for voice cloning with any reference audio.
//! Supports both zero-shot and few-shot voice cloning modes.
//!
//! # Modes
//!
//! - **Zero-shot**: Uses only reference audio mel spectrogram for voice style
//! - **Few-shot**: Uses reference audio + transcript for stronger conditioning via HuBERT
//!
//! # Zero-Shot Example
//!
//! ```ignore
//! use mlx_rs_lm::voice_clone::{VoiceCloner, VoiceClonerConfig};
//!
//! let config = VoiceClonerConfig::default();
//! let mut cloner = VoiceCloner::new(config)?;
//!
//! // Zero-shot: only reference audio
//! cloner.set_reference_audio("/path/to/reference.wav")?;
//!
//! let audio = cloner.synthesize("你好，世界！")?;
//! cloner.save_wav(&audio, "/tmp/output.wav")?;
//! ```
//!
//! # Few-Shot Example (Better Quality)
//!
//! ```ignore
//! use mlx_rs_lm::voice_clone::{VoiceCloner, VoiceClonerConfig};
//!
//! let config = VoiceClonerConfig::default();
//! let mut cloner = VoiceCloner::new(config)?;
//!
//! // Few-shot: reference audio + transcript
//! cloner.set_reference_audio_with_text(
//!     "/path/to/reference.wav",
//!     "这是参考音频的文本内容"
//! )?;
//!
//! let audio = cloner.synthesize("你好，世界！")?;
//! cloner.play_blocking(&audio)?;
//! ```
//!
//! # Command Line
//!
//! ```bash
//! # Zero-shot
//! cargo run --release --example voice_clone -- "你好" --ref voice.wav
//!
//! # Few-shot
//! cargo run --release --example voice_clone -- "你好" --ref voice.wav --ref-text "参考文本"
//!
//! # Interactive mode
//! cargo run --release --example voice_clone -- --interactive
//! ```
//!
//! For detailed documentation, see `docs/voice_clone.md`

use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use lingua::{Language, LanguageDetector, LanguageDetectorBuilder};
use mlx_rs::{Array, module::Module, ops::indexing::IndexOp, random, transforms::eval};
use tracing::{debug, warn, trace, instrument};

use crate::{
    audio::{AudioConfig, load_reference_mel, load_audio_for_hubert},
    cache::ConcatKeyValueCache,
    error::Error,
    inference::{preprocess_text, preprocess_text_with_lang},
    models::{
        hubert::{HuBertEncoder, load_hubert_model},
        t2s::{T2SConfig, T2SInput, T2SModel, load_t2s_model},
        vits::{SynthesizerTrn, load_vits_model},
    },
    sampling::{Sampler, SamplingConfig, detect_repetition},
    text::BertFeatureExtractor,
};

/// Configuration for voice cloner
#[derive(Debug, Clone)]
pub struct VoiceClonerConfig {
    /// Path to T2S model weights
    pub t2s_weights: String,
    /// Path to BERT model weights
    pub bert_weights: String,
    /// Path to BERT tokenizer
    pub bert_tokenizer: String,
    /// Path to VITS model weights
    pub vits_weights: String,
    /// Path to HuBERT model weights (for few-shot mode)
    pub hubert_weights: String,
    /// Sample rate for output audio
    pub sample_rate: u32,
    /// Top-k sampling parameter (-1 = disabled)
    pub top_k: i32,
    /// Top-p (nucleus) sampling parameter (1.0 = disabled)
    pub top_p: f32,
    /// Temperature for sampling
    pub temperature: f32,
    /// Repetition penalty (1.0 = no penalty, matches Python SynthesisConfig default)
    pub repetition_penalty: f32,
    /// Noise scale for VITS (0.0 = deterministic)
    pub noise_scale: f32,
    /// Speed factor (1.0 = normal)
    pub speed: f32,
    /// Path to ONNX VITS model for batched decode (default, recommended)
    /// Set to None to fall back to MLX VITS per-chunk decode
    pub vits_onnx_path: Option<String>,
    /// Use MLX VITS instead of ONNX (not recommended, may have chunk boundary artifacts)
    pub use_mlx_vits: bool,
}

/// Get the default model directory path
/// Uses $GPT_SOVITS_MODEL_DIR if set, otherwise ~/.dora/models/primespeech/gpt-sovits-mlx
fn default_model_dir() -> String {
    if let Ok(dir) = std::env::var("GPT_SOVITS_MODEL_DIR") {
        return dir;
    }
    if let Some(home) = std::env::var_os("HOME") {
        return format!("{}/.dora/models/primespeech/gpt-sovits-mlx", home.to_string_lossy());
    }
    // Fallback
    "/tmp/gpt-sovits-mlx".to_string()
}

impl Default for VoiceClonerConfig {
    fn default() -> Self {
        let model_dir = default_model_dir();
        Self {
            // Use doubao-mixed fine-tuned models (converted from dora-primespeech)
            t2s_weights: format!("{}/doubao_mixed_gpt_new.safetensors", model_dir),
            bert_weights: format!("{}/bert.safetensors", model_dir),
            bert_tokenizer: format!("{}/chinese-roberta-tokenizer/tokenizer.json", model_dir),
            vits_weights: format!("{}/doubao_mixed_sovits_new.safetensors", model_dir),
            hubert_weights: format!("{}/hubert.safetensors", model_dir),
            sample_rate: 32000,
            top_k: 5,  // Match Python TTS.py default
            top_p: 1.0,  // Match Python TTS.py default (no nucleus sampling)
            temperature: 1.0,  // Match Python TTS.py default
            repetition_penalty: 1.35,  // Match Python TTS.py default
            noise_scale: 0.5,
            speed: 1.0,
            // ONNX VITS is default for best quality (batched decode, matches Python)
            vits_onnx_path: Some(format!("{}/vits.onnx", model_dir)),
            use_mlx_vits: false,
        }
    }
}

/// Generated audio output
#[derive(Debug)]
pub struct AudioOutput {
    /// Raw audio samples (f32, range -1.0 to 1.0)
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration: f32,
    /// Number of semantic tokens generated
    pub num_tokens: usize,
}

impl AudioOutput {
    /// Get duration in seconds
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }
}

/// Options for synthesis with timeout and cancellation support
#[derive(Debug, Clone, Default)]
pub struct SynthesisOptions {
    /// Maximum time allowed for synthesis (None = no timeout)
    pub timeout: Option<Duration>,
    /// Cancellation token - set to true to cancel synthesis
    pub cancel_token: Option<std::sync::Arc<AtomicBool>>,
    /// Maximum tokens to generate per chunk (None = auto)
    pub max_tokens_per_chunk: Option<usize>,
}

impl SynthesisOptions {
    /// Create options with a timeout
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            timeout: Some(timeout),
            ..Default::default()
        }
    }

    /// Create options with a cancellation token
    pub fn with_cancel_token(token: std::sync::Arc<AtomicBool>) -> Self {
        Self {
            cancel_token: Some(token),
            ..Default::default()
        }
    }

    /// Check if cancellation was requested
    fn is_cancelled(&self) -> bool {
        self.cancel_token.as_ref()
            .map(|t| t.load(Ordering::Relaxed))
            .unwrap_or(false)
    }

    /// Check if timeout has elapsed
    #[allow(dead_code)]
    fn is_timed_out(&self, start: Instant) -> bool {
        self.timeout
            .map(|t| start.elapsed() > t)
            .unwrap_or(false)
    }
}

impl AudioOutput {

    /// Convert to i16 samples for WAV output
    pub fn to_i16_samples(&self) -> Vec<i16> {
        self.samples
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect()
    }

    /// Apply fade-in to reduce initial noise artifacts
    ///
    /// # Arguments
    /// * `fade_ms` - Fade-in duration in milliseconds (default: 50ms)
    pub fn apply_fade_in(&mut self, fade_ms: f32) {
        let fade_samples = ((fade_ms / 1000.0) * self.sample_rate as f32) as usize;
        let fade_samples = fade_samples.min(self.samples.len());

        for i in 0..fade_samples {
            let factor = i as f32 / fade_samples as f32;
            self.samples[i] *= factor;
        }
    }

    /// Trim audio from the start to remove initial artifacts (beeps/clicks)
    ///
    /// # Arguments
    /// * `trim_ms` - Duration to trim in milliseconds
    pub fn trim_start(&mut self, trim_ms: f32) {
        let trim_samples = ((trim_ms / 1000.0) * self.sample_rate as f32) as usize;
        let trim_samples = trim_samples.min(self.samples.len());

        if trim_samples > 0 {
            self.samples = self.samples[trim_samples..].to_vec();
            self.duration = self.samples.len() as f32 / self.sample_rate as f32;
        }
    }
}

/// Voice cloner for GPT-SoVITS
pub struct VoiceCloner {
    config: VoiceClonerConfig,
    t2s_config: T2SConfig,
    t2s: T2SModel,
    bert: BertFeatureExtractor,
    vits: SynthesizerTrn,
    hubert: Option<HuBertEncoder>,
    audio_config: AudioConfig,
    reference_mel: Option<Array>,
    reference_path: Option<String>,
    /// Prompt semantic codes for few-shot mode (extracted from reference audio)
    prompt_semantic: Option<Array>,
    /// Reference text for few-shot mode
    reference_text: Option<String>,
    /// Optional ONNX VITS for batched decode
    vits_onnx: Option<crate::models::vits_onnx::VitsOnnx>,
}

impl VoiceCloner {
    /// Create a new voice cloner with the given configuration
    pub fn new(config: VoiceClonerConfig) -> Result<Self, Error> {
        // Validate paths (HuBERT is optional for few-shot mode)
        for (name, path) in [
            ("T2S weights", &config.t2s_weights),
            ("BERT weights", &config.bert_weights),
            ("BERT tokenizer", &config.bert_tokenizer),
            ("VITS weights", &config.vits_weights),
        ] {
            if !Path::new(path).exists() {
                return Err(Error::Message(format!("{} not found: {}", name, path)));
            }
        }

        // Load models
        let bert = BertFeatureExtractor::new(&config.bert_tokenizer, &config.bert_weights, -3)?;
        let t2s_config = T2SConfig::default();
        let t2s = load_t2s_model(&config.t2s_weights)?;
        let vits = load_vits_model(&config.vits_weights)?;
        let audio_config = AudioConfig::default();

        // Try to load HuBERT (optional for few-shot mode)
        let hubert = if Path::new(&config.hubert_weights).exists() {
            match load_hubert_model(&config.hubert_weights) {
                Ok(h) => Some(h),
                Err(e) => {
                    eprintln!("Warning: Failed to load HuBERT model: {}. Few-shot mode will be unavailable.", e);
                    None
                }
            }
        } else {
            None
        };

        // Load ONNX VITS by default for best quality (batched decode, matches Python)
        // Skip if use_mlx_vits is explicitly set to true
        let vits_onnx = if config.use_mlx_vits {
            eprintln!("Using MLX VITS (per-chunk decode) - may have chunk boundary artifacts");
            None
        } else if let Some(ref onnx_path) = config.vits_onnx_path {
            match crate::models::vits_onnx::VitsOnnx::load(std::path::Path::new(onnx_path)) {
                Ok(v) => {
                    eprintln!("Using ONNX VITS (batched decode) for best quality");
                    Some(v)
                },
                Err(e) => {
                    eprintln!("Warning: Failed to load ONNX VITS: {}. Falling back to MLX.", e);
                    None
                }
            }
        } else {
            eprintln!("No ONNX VITS path configured. Using MLX VITS (per-chunk decode).");
            None
        };

        Ok(Self {
            config,
            t2s_config,
            t2s,
            bert,
            vits,
            hubert,
            audio_config,
            reference_mel: None,
            reference_path: None,
            prompt_semantic: None,
            reference_text: None,
            vits_onnx,
        })
    }

    /// Create with default configuration
    pub fn with_defaults() -> Result<Self, Error> {
        Self::new(VoiceClonerConfig::default())
    }

    /// Set reference audio for voice cloning (zero-shot mode)
    pub fn set_reference_audio(&mut self, path: impl AsRef<Path>) -> Result<(), Error> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(Error::Message(format!("Reference audio not found: {:?}", path)));
        }

        let mel = load_reference_mel(path, &self.audio_config)
            .map_err(|e| Error::Message(format!("Failed to load reference audio: {}", e)))?;
        eval([&mel]).map_err(|e| Error::Message(format!("Failed to evaluate mel: {}", e)))?;

        self.reference_mel = Some(mel);
        self.reference_path = Some(path.to_string_lossy().to_string());
        // Clear few-shot data
        self.prompt_semantic = None;
        self.reference_text = None;

        Ok(())
    }

    /// Set reference audio with transcript for few-shot mode
    ///
    /// Few-shot mode extracts semantic tokens from the reference audio using HuBERT,
    /// which provides better voice cloning quality than zero-shot mode.
    ///
    /// # Arguments
    /// * `audio_path` - Path to reference audio file
    /// * `text` - Transcript of the reference audio
    pub fn set_reference_audio_with_text(
        &mut self,
        audio_path: impl AsRef<Path>,
        text: &str,
    ) -> Result<(), Error> {
        let audio_path = audio_path.as_ref();
        if !audio_path.exists() {
            return Err(Error::Message(format!("Reference audio not found: {:?}", audio_path)));
        }

        // Load mel spectrogram
        let mel = load_reference_mel(audio_path, &self.audio_config)
            .map_err(|e| Error::Message(format!("Failed to load reference audio: {}", e)))?;
        eval([&mel]).map_err(|e| Error::Message(format!("Failed to evaluate mel: {}", e)))?;

        // Extract prompt semantic codes if HuBERT is available
        let prompt_semantic = if let Some(ref mut hubert) = self.hubert {
            // Load audio at 16kHz for HuBERT
            let audio_16k = load_audio_for_hubert(audio_path)
                .map_err(|e| Error::Message(format!("Failed to load audio for HuBERT: {}", e)))?;
            eval([&audio_16k]).map_err(|e| Error::Message(e.to_string()))?;

            // Pad with 0.3s silence (matching Python's zero_wav padding)
            // This is important for matching the exact token count
            let audio_data: Vec<f32> = audio_16k.as_slice().to_vec();
            let pad_samples = (0.3 * 16000.0) as usize;
            let mut audio_padded = audio_data;
            audio_padded.extend(vec![0.0f32; pad_samples]);
            let audio_16k = Array::from_slice(&audio_padded, &[1, audio_padded.len() as i32]);

            // Extract HuBERT features: [batch, time, 768] (NLC format)
            // NOTE: The Rust HuBERT implementation may not produce the same features as
            // the Python CNHubert. If few-shot results are poor, try using pre-computed
            // prompt semantic codes from Python instead.
            let hubert_features = hubert.forward(&audio_16k)
                .map_err(|e| Error::Message(format!("HuBERT forward failed: {}", e)))?;
            eval([&hubert_features]).map_err(|e| Error::Message(e.to_string()))?;

            // ssl_proj expects NLC format, hubert_features is already NLC
            let projected_nlc = self.vits.ssl_proj.forward(&hubert_features)
                .map_err(|e| Error::Message(format!("ssl_proj forward failed: {}", e)))?;
            eval([&projected_nlc]).map_err(|e| Error::Message(e.to_string()))?;

            // Convert to NCL for quantizer.encode: [batch, 768, time]
            let projected_ncl = projected_nlc.transpose_axes(&[0, 2, 1])
                .map_err(|e| Error::Message(format!("Transpose failed: {}", e)))?;

            // Encode to semantic codes: [batch, 1, time]
            let codes = self.vits.quantizer.encode(&projected_ncl)
                .map_err(|e| Error::Message(format!("Quantizer encode failed: {}", e)))?;
            eval([&codes]).map_err(|e| Error::Message(e.to_string()))?;

            Some(codes)
        } else {
            return Err(Error::Message(
                "Few-shot mode requires HuBERT model. Ensure hubert_weights path is valid.".to_string()
            ));
        };

        self.reference_mel = Some(mel);
        self.reference_path = Some(audio_path.to_string_lossy().to_string());
        self.prompt_semantic = prompt_semantic;
        self.reference_text = Some(text.to_string());

        Ok(())
    }

    /// Set reference audio with pre-computed prompt semantic codes
    ///
    /// Use this when the Rust HuBERT produces poor results. You can extract
    /// prompt semantic codes using Python and load them here.
    ///
    /// # Arguments
    /// * `audio_path` - Path to reference audio file (for mel spectrogram)
    /// * `text` - Transcript of the reference audio
    /// * `codes_path` - Path to binary file containing i32 codes (little-endian)
    ///
    /// # Example: Extract codes with Python
    /// ```python
    /// # See scripts/extract_prompt_semantic.py
    /// import torch
    /// from transformers import HubertModel, Wav2Vec2FeatureExtractor
    /// # ... extract codes and save as .bin file
    /// codes.numpy().astype(np.int32).tofile("prompt_semantic.bin")
    /// ```
    pub fn set_reference_with_precomputed_codes(
        &mut self,
        audio_path: impl AsRef<Path>,
        text: &str,
        codes_path: impl AsRef<Path>,
    ) -> Result<(), Error> {
        let audio_path = audio_path.as_ref();
        let codes_path = codes_path.as_ref();

        if !audio_path.exists() {
            return Err(Error::Message(format!("Reference audio not found: {:?}", audio_path)));
        }
        if !codes_path.exists() {
            return Err(Error::Message(format!("Codes file not found: {:?}", codes_path)));
        }

        // Load mel spectrogram
        let mel = load_reference_mel(audio_path, &self.audio_config)
            .map_err(|e| Error::Message(format!("Failed to load reference audio: {}", e)))?;
        eval([&mel]).map_err(|e| Error::Message(format!("Failed to evaluate mel: {}", e)))?;

        // Load pre-computed codes from file (supports both .npy and raw binary)
        let codes_path_buf = std::path::PathBuf::from(codes_path);
        let codes_data = std::fs::read(&codes_path_buf)?;

        // Check for NPY file format (magic bytes: \x93NUMPY)
        const NPY_MAGIC: &[u8] = b"\x93NUMPY";
        const NPY_MIN_HEADER: usize = 10;
        const MAX_HEADER_SIZE: usize = 10_000;  // Reasonable limit for NPY header

        let codes: Vec<i32> = if codes_data.len() > NPY_MIN_HEADER
            && codes_data.get(..NPY_MAGIC.len()) == Some(NPY_MAGIC)
        {
            // Parse NPY file: find header end (newline after dict)
            let search_end = (NPY_MIN_HEADER + MAX_HEADER_SIZE).min(codes_data.len());
            let header_end = codes_data[NPY_MIN_HEADER..search_end]
                .iter()
                .position(|&b| b == b'\n')
                .map(|pos| NPY_MIN_HEADER + pos + 1)  // +1 to skip newline
                .ok_or_else(|| Error::file_corrupted(&codes_path_buf, "NPY header newline not found"))?;

            // Validate data alignment
            let data_len = codes_data.len() - header_end;
            if data_len % 4 != 0 {
                return Err(Error::file_corrupted(&codes_path_buf,
                    format!("NPY data not aligned to 4 bytes (size={})", data_len)));
            }

            // Extract data portion
            codes_data[header_end..]
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        } else {
            // Raw binary format - validate alignment
            if codes_data.len() % 4 != 0 {
                return Err(Error::file_corrupted(&codes_path_buf,
                    format!("Binary data not aligned to 4 bytes (size={})", codes_data.len())));
            }
            codes_data
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };

        if codes.is_empty() {
            return Err(Error::file_corrupted(&codes_path_buf, "No codes found in file"));
        }

        // Create Array from codes: [1, 1, num_codes]
        let codes_array = Array::from_slice(&codes, &[1, 1, codes.len() as i32]);

        self.reference_mel = Some(mel);
        self.reference_path = Some(audio_path.to_string_lossy().to_string());
        self.prompt_semantic = Some(codes_array);
        self.reference_text = Some(text.to_string());

        Ok(())
    }

    /// Set reference using pre-extracted semantic codes (for debugging/testing)
    ///
    /// This allows using semantic codes extracted from Python for comparison.
    pub fn set_reference_with_semantic_codes(
        &mut self,
        audio_path: impl AsRef<Path>,
        text: &str,
        semantic_codes: &[i32],
    ) -> Result<(), Error> {
        let audio_path = audio_path.as_ref();

        if !audio_path.exists() {
            return Err(Error::Message(format!("Reference audio not found: {:?}", audio_path)));
        }

        // Load mel spectrogram
        let mel = load_reference_mel(audio_path, &self.audio_config)
            .map_err(|e| Error::Message(format!("Failed to load reference audio: {}", e)))?;
        eval([&mel]).map_err(|e| Error::Message(format!("Failed to evaluate mel: {}", e)))?;

        // Create Array from codes: [1, 1, num_codes]
        let codes_array = Array::from_slice(semantic_codes, &[1, 1, semantic_codes.len() as i32]);

        self.reference_mel = Some(mel);
        self.reference_path = Some(audio_path.to_string_lossy().to_string());
        self.prompt_semantic = Some(codes_array);
        self.reference_text = Some(text.to_string());
        Ok(())
    }

    /// Check if few-shot mode is available
    pub fn few_shot_available(&self) -> bool {
        self.hubert.is_some()
    }

    /// Check if currently in few-shot mode
    pub fn is_few_shot_mode(&self) -> bool {
        self.prompt_semantic.is_some() && self.reference_text.is_some()
    }

    /// Get the current reference audio path
    pub fn reference_path(&self) -> Option<&str> {
        self.reference_path.as_deref()
    }

    /// Get the current reference text (for few-shot mode)
    pub fn reference_text(&self) -> Option<&str> {
        self.reference_text.as_deref()
    }

    /// Get the current prompt semantic codes (for debugging)
    pub fn prompt_semantic(&self) -> Option<&Array> {
        self.prompt_semantic.as_ref()
    }

    /// Get the current reference mel spectrogram
    pub fn reference_mel(&self) -> Option<&Array> {
        self.reference_mel.as_ref()
    }

    /// Synthesize audio from external semantic tokens (for testing/debugging)
    ///
    /// This bypasses token generation and directly vocodes the provided tokens.
    /// Useful for comparing Rust VITS with Python's semantic tokens.
    pub fn synthesize_from_tokens(&mut self, text: &str, tokens: &[i32]) -> Result<AudioOutput, Error> {
        let ref_mel = self.reference_mel.clone()
            .ok_or(Error::ReferenceNotSet)?;

        // Preprocess text to get phoneme IDs
        let (phoneme_ids, _phonemes, _word2ph, _text_normalized) = preprocess_text(text);

        // Vocode using provided tokens
        let audio = self.vocode(tokens, &phoneme_ids, &ref_mel)?;
        let samples = array_to_f32_samples(&audio)?;
        let duration = samples.len() as f32 / self.config.sample_rate as f32;

        Ok(AudioOutput {
            samples,
            sample_rate: self.config.sample_rate,
            duration,
            num_tokens: tokens.len(),
        })
    }

    /// Synthesize speech from text
    /// Synthesize speech with timeout and cancellation support
    ///
    /// # Arguments
    /// * `text` - Text to synthesize
    /// * `options` - Synthesis options (timeout, cancellation token)
    ///
    /// # Example
    /// ```ignore
    /// use std::sync::Arc;
    /// use std::sync::atomic::AtomicBool;
    /// use std::time::Duration;
    ///
    /// let cancel = Arc::new(AtomicBool::new(false));
    /// let options = SynthesisOptions {
    ///     timeout: Some(Duration::from_secs(30)),
    ///     cancel_token: Some(cancel.clone()),
    ///     ..Default::default()
    /// };
    ///
    /// // Start synthesis in background thread
    /// // Call cancel.store(true, Ordering::Relaxed) to cancel
    /// let audio = cloner.synthesize_with_options("Hello", options)?;
    /// ```
    #[instrument(skip(self, options), fields(text_len = text.len()))]
    pub fn synthesize_with_options(&mut self, text: &str, options: SynthesisOptions) -> Result<AudioOutput, Error> {
        let start = Instant::now();

        // Check cancellation before starting
        if options.is_cancelled() {
            return Err(Error::Cancelled);
        }

        // Store options for use in generation loop
        let timeout = options.timeout;
        let cancel_token = options.cancel_token.clone();

        // Perform synthesis with periodic checks
        let result = self.synthesize(text);

        // Check for timeout/cancellation after synthesis
        if let Some(token) = &cancel_token {
            if token.load(Ordering::Relaxed) {
                return Err(Error::Cancelled);
            }
        }
        if let Some(t) = timeout {
            if start.elapsed() > t {
                return Err(Error::timeout(start.elapsed()));
            }
        }

        result
    }

    /// Synthesize speech from text
    ///
    /// Text is automatically split at punctuation marks (like Python's cut5 method)
    /// and each segment is processed separately for better quality.
    /// Long segments are further chunked to prevent T2S attention degradation.
    #[instrument(skip(self), fields(text_len = text.len()))]
    pub fn synthesize(&mut self, text: &str) -> Result<AudioOutput, Error> {
        // ===== Input Validation =====
        const MAX_TEXT_LENGTH: usize = 10_000;

        // Check text is not empty
        let text = text.trim();
        if text.is_empty() {
            return Err(Error::EmptyInput);
        }

        // Check text length
        if text.len() > MAX_TEXT_LENGTH {
            return Err(Error::text_too_long(text.len(), MAX_TEXT_LENGTH));
        }

        // Check reference is set
        let ref_mel = self.reference_mel.clone()
            .ok_or(Error::ReferenceNotSet)?;

        // Split text using Python-compatible cut5 method:
        // Split at every punctuation mark, merge short segments (< 5 chars)
        let mut chunks = cut5_split(text);

        // Trim whitespace from chunks (cut5_split already appends 。 and filters)
        for chunk in chunks.iter_mut() {
            let trimmed = chunk.trim().to_string();
            *chunk = trimmed;
        }
        chunks.retain(|c| !c.trim().is_empty());

        // Debug: log chunks
        debug!(num_chunks = chunks.len(), "Text split into chunks");
        for (i, chunk) in chunks.iter().enumerate() {
            trace!(chunk_idx = i, chunk = %chunk, "Chunk content");
        }

        // Phase 1: Generate semantic tokens for each chunk (T2S only, no VITS yet)
        // Python (speed==1.0) concatenates all semantic tokens + all phone IDs and calls VITS once
        struct ChunkT2SResult {
            semantic_tokens: Vec<i32>,
            phone_ids: Array,
            #[allow(dead_code)]
            num_tokens: usize,
        }

        let mut chunk_results: Vec<ChunkT2SResult> = Vec::new();
        let mut total_tokens = 0;

        for (i, chunk) in chunks.iter().enumerate() {
            if chunk.trim().is_empty() {
                continue;
            }

            let is_few_shot = self.is_few_shot_mode();
            let preview: String = chunk.chars().take(15).collect();
            eprintln!("   Processing chunk [{}]: few_shot={}, text=\"{}...\"", i, is_few_shot, preview);

            let (tokens, phone_ids) = if is_few_shot {
                self.generate_chunk_tokens(&chunk, &ref_mel, i == 0)?
            } else {
                self.generate_chunk_tokens_zero_shot(&chunk, &ref_mel, i == 0)?
            };

            eprintln!("   Chunk [{}]: {} tokens", i, tokens.len());
            total_tokens += tokens.len();
            chunk_results.push(ChunkT2SResult {
                semantic_tokens: tokens,
                phone_ids,
                num_tokens: 0,
            });
        }

        // Phase 2: VITS decode
        // If ONNX VITS is available, use batched decode (all chunks in one call).
        // Otherwise, fall back to per-chunk MLX decode with tail trimming.
        if let Some(ref mut vits_onnx) = self.vits_onnx {
            // Concatenate all chunks' semantic tokens and phone IDs (like Python)
            let all_tokens: Vec<i32> = chunk_results.iter()
                .flat_map(|r| r.semantic_tokens.iter().copied())
                .collect();

            let mut all_phones: Vec<i32> = Vec::new();
            for result in &chunk_results {
                let phone_arr = &result.phone_ids;
                eval([phone_arr]).map_err(|e| Error::Message(e.to_string()))?;
                let flat = phone_arr.flatten(None, None).map_err(|e| Error::Message(e.to_string()))?;
                eval([&flat]).map_err(|e| Error::Message(e.to_string()))?;
                let phone_slice: &[i32] = flat.as_slice();
                all_phones.extend_from_slice(phone_slice);
            }

            // Extract ref_mel data
            eval([&ref_mel]).map_err(|e| Error::Message(e.to_string()))?;
            let mel_shape = ref_mel.shape().to_vec();
            let mel_channels = mel_shape[1] as usize; // 704
            let mel_time = mel_shape[2] as usize;
            let mel_flat = ref_mel.flatten(None, None).map_err(|e| Error::Message(e.to_string()))?;
            eval([&mel_flat]).map_err(|e| Error::Message(e.to_string()))?;
            let mel_data: &[f32] = mel_flat.as_slice();

            // Debug: per-chunk breakdown
            for (i, result) in chunk_results.iter().enumerate() {
                let phone_arr = &result.phone_ids;
                eval([phone_arr]).map_err(|e| Error::Message(e.to_string()))?;
                let flat = phone_arr.flatten(None, None).map_err(|e| Error::Message(e.to_string()))?;
                eval([&flat]).map_err(|e| Error::Message(e.to_string()))?;
                let ps: &[i32] = flat.as_slice();
                eprintln!("[ONNX] Chunk[{}]: {} tokens, {} phones, tokens[..10]={:?}",
                         i, result.semantic_tokens.len(), ps.len(),
                         &result.semantic_tokens[..result.semantic_tokens.len().min(10)]);
            }
            eprintln!("[ONNX] Batched decode: {} tokens, {} phones, refer=[1,{},{}]",
                     all_tokens.len(), all_phones.len(), mel_channels, mel_time);

            let raw_samples = vits_onnx.decode(
                &all_tokens, &all_phones, mel_data, mel_channels, mel_time,
                self.config.noise_scale,
            ).map_err(|e| Error::Message(format!("ONNX VITS decode failed: {}", e)))?;

            // Split ONNX output by chunk boundaries and apply per-chunk tail trimming.
            // Each chunk produces tokens * 2 * 640 samples (2x upsample, hop_length=640).
            let upsample_factor = 2 * 640;
            let sr = self.config.sample_rate as f32;
            let silence_duration = 0.3f32;
            let silence_samples = (sr * silence_duration) as usize;

            // Split output by chunk boundaries and join with silence (matching Python exactly).
            // Python: clip to [-1,1], append 0.3s silence after each chunk, concatenate. No trimming.
            let mut all_samples: Vec<f32> = Vec::new();
            let mut pos = 0usize;

            for (i, result) in chunk_results.iter().enumerate() {
                let chunk_len = result.semantic_tokens.len() * upsample_factor;
                let end = (pos + chunk_len).min(raw_samples.len());
                let mut samples = raw_samples[pos..end].to_vec();
                pos = end;

                // Clipping prevention (same as Python)
                let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                if max_abs > 1.0 {
                    for s in samples.iter_mut() {
                        *s /= max_abs;
                    }
                }

                eprintln!("   [ONNX] Chunk [{}]: {} tokens -> {} samples",
                         i, result.semantic_tokens.len(), samples.len());

                // Append chunk audio + 0.3s silence (matching Python's fragment_interval=0.3)
                all_samples.extend_from_slice(&samples);
                all_samples.extend(std::iter::repeat(0.0f32).take(silence_samples));
            }

            let duration = all_samples.len() as f32 / sr;
            return Ok(AudioOutput {
                samples: all_samples,
                sample_rate: self.config.sample_rate,
                duration,
                num_tokens: total_tokens,
            });
        }

        // Fallback: Per-chunk VITS with tail trimming and crossfade
        // Each chunk is decoded individually to avoid VITS attention artifacts from
        // concatenating independent T2S sequences.
        let sr = self.config.sample_rate as f32;
        let silence_duration = 0.3f32; // seconds of silence between chunks
        let silence_samples = (sr * silence_duration) as usize;
        let crossfade_samples = (sr * 0.05) as usize; // 50ms crossfade

        let mut all_samples: Vec<f32> = Vec::new();

        for (i, result) in chunk_results.iter().enumerate() {
            let audio = self.vocode(&result.semantic_tokens, &result.phone_ids, &ref_mel)?;
            let mut samples = array_to_f32_samples(&audio)?;

            // Clipping prevention
            let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            if max_abs > 1.0 {
                for s in samples.iter_mut() {
                    *s /= max_abs;
                }
            }

            // Trim trailing artifacts from T2S over-generation.
            // Problem pattern: speech ends, then silence gap, then noise burst, then fade.
            // E.g., chunk 2 "合并，": speech at 0-5.5s, silence at 5.75-6.0s,
            // noise burst at 6.25-6.75s, fade at 7.0-7.4s.
            //
            // Strategy: Compute RMS in 50ms windows from the end. Walk backwards
            // through: (1) trailing silence, (2) any noise burst, (3) a silence gap.
            // If we find pattern "silence gap -> burst -> trailing silence", cut at the
            // silence gap. Otherwise just trim trailing silence.
            let win = (sr * 0.05) as usize; // 50ms window
            let n_windows = samples.len() / win;
            let mut trim_pos = samples.len();

            if n_windows > 4 {
                // Compute RMS per window
                let rms_vals: Vec<f32> = (0..n_windows).map(|w| {
                    let start = w * win;
                    let end = (start + win).min(samples.len());
                    let slice = &samples[start..end];
                    (slice.iter().map(|s| s * s).sum::<f32>() / slice.len() as f32).sqrt()
                }).collect();

                // Walk backwards to find the last "real speech" window
                // Real speech: RMS > 0.03, and it's part of the main content
                // (not an isolated burst after silence)
                let speech_thresh = 0.03f32;
                let silence_thresh = 0.01f32;

                // Find the cut point by detecting "silence gap after speech"
                // Walk backwards: skip trailing silence, skip any burst, find silence gap
                let mut w = n_windows - 1;

                // Phase 1: Skip trailing silence
                while w > 0 && rms_vals[w] < silence_thresh {
                    w -= 1;
                }
                let last_energy = w; // last window with energy

                // Phase 2: Check if there's a silence gap before this energy block
                // Walk backwards through the energy block
                while w > 0 && rms_vals[w] >= silence_thresh {
                    w -= 1;
                }
                let energy_block_start = w + 1;

                // Phase 3: Check if there's a silence gap here
                let mut gap_count = 0;
                while w > 0 && rms_vals[w] < silence_thresh {
                    gap_count += 1;
                    w -= 1;
                }
                let gap_end_window = w + 1; // first window of the silence gap

                // Phase 4: Check if there's speech BEFORE the gap
                let has_speech_before_gap = w > 2 && rms_vals[..=w].iter().any(|&r| r > speech_thresh);

                // Decision: if there's a silence gap (>= 100ms = 2 windows) between
                // speech and an isolated energy block, cut at the gap.
                // Only cut if the burst is short (<20% of total) — long blocks are real speech.
                let burst_len = last_energy.saturating_sub(energy_block_start) + 1;
                let burst_is_short = burst_len < n_windows / 5;

                let trim_window_idx = if gap_count >= 2 && has_speech_before_gap && burst_is_short {
                    // Cut at the start of the silence gap, with small margin
                    let cut_at = gap_end_window + 1; // keep 1 window into the gap for decay
                    eprintln!("   [TRIM] Chunk {}: detected silence gap at window {} ({}ms), cutting burst ({} wins) at {}-{}",
                             i, gap_end_window, gap_end_window * 50, burst_len, energy_block_start, last_energy);
                    cut_at.min(n_windows)
                } else {
                    // No gap pattern — just trim trailing silence
                    (last_energy + 2).min(n_windows) // +2 windows (100ms) margin
                };

                let trim_pos_new = (trim_window_idx * win).min(samples.len());

                // Apply fade-out over last 30ms
                let fadeout_len = (sr * 0.03) as usize;
                let fade_start = trim_pos_new.saturating_sub(fadeout_len);
                for j in fade_start..trim_pos_new {
                    let t = (trim_pos_new - j) as f32 / fadeout_len as f32;
                    samples[j] *= t;
                }
                // Zero out everything after trim
                for j in trim_pos_new..samples.len() {
                    samples[j] = 0.0;
                }
                trim_pos = trim_pos_new;
            }

            let trimmed = &samples[..trim_pos];
            eprintln!("   Chunk [{}]: {} tokens -> {} samples (trimmed from {})",
                     i, result.semantic_tokens.len(), trimmed.len(), samples.len());

            // Crossfade with previous chunk's tail
            if !all_samples.is_empty() && crossfade_samples > 0 {
                // Remove trailing silence we just added, crossfade into it
                let overlap = crossfade_samples.min(all_samples.len()).min(trimmed.len());
                let base_start = all_samples.len() - overlap;
                for j in 0..overlap {
                    let t = j as f32 / overlap as f32; // 0->1
                    all_samples[base_start + j] = all_samples[base_start + j] * (1.0 - t) + trimmed[j] * t;
                }
                all_samples.extend_from_slice(&trimmed[overlap..]);
            } else {
                all_samples.extend_from_slice(trimmed);
            }

            // Add silence after chunk
            all_samples.extend(vec![0.0f32; silence_samples]);
        }

        // Combine all samples into final output
        let duration = all_samples.len() as f32 / self.config.sample_rate as f32;
        let output = AudioOutput {
            samples: all_samples,
            sample_rate: self.config.sample_rate,
            duration,
            num_tokens: total_tokens,
        };


        Ok(output)
    }

    /// Generate semantic tokens for a chunk (zero-shot T2S only, no VITS).
    /// Returns (semantic_tokens, target_phone_ids).
    fn generate_chunk_tokens_zero_shot(&mut self, text: &str, _ref_mel: &Array, is_first_chunk: bool) -> Result<(Vec<i32>, Array), Error> {
        // Reuse the same text preprocessing as synthesize_zero_shot
        let text = if let Some(first_char) = text.chars().next() {
            let is_punct = matches!(first_char, ',' | '.' | '!' | '?' | '~' | ':' | '—' | '…' |
                                              '，' | '。' | '！' | '？' | '：');
            let is_english = first_char.is_ascii_alphabetic();
            let punct_chars = [',', '.', '!', '?', '~', ':', '—', '…', '，', '。', '！', '？', '：'];
            let first_segment_len = text.chars()
                .take_while(|c| !punct_chars.contains(c))
                .count();

            if is_first_chunk && !is_punct && is_english {
                format!(", {}", text)
            } else if is_first_chunk && !is_punct && first_segment_len < 4 {
                eprintln!("   [zero-shot] Short first segment ({}): prepending '.'", first_segment_len);
                format!(".{}", text)
            } else {
                text.to_string()
            }
        } else {
            text.to_string()
        };

        let (phoneme_ids, phonemes, word2ph, text_normalized) = preprocess_text_with_lang(&text, Some(crate::text::Language::Chinese));
        eprintln!("   Phonemes: {:?}", &phonemes[..phonemes.len().min(30)]);
        eprintln!("   Normalized: \"{}\"", text_normalized);

        let text_chars = text_normalized.chars().count();
        let word2ph_for_bert = &word2ph[..text_chars.min(word2ph.len())];
        let bert_features = self.extract_bert_features(&text_normalized, word2ph_for_bert, phonemes.len())?;

        let (all_tokens, generated_count) = self.generate_semantic_tokens(&phoneme_ids, &bert_features, phonemes.len(), None)?;
        let tokens = all_tokens[all_tokens.len().saturating_sub(generated_count)..].to_vec();

        eprintln!("   Semantic tokens ({}): {:?}", tokens.len(), &tokens[..tokens.len().min(20)]);

        Ok((tokens, phoneme_ids))
    }

    /// Generate semantic tokens for a chunk (few-shot T2S only, no VITS).
    /// Returns (semantic_tokens, target_phone_ids).
    fn generate_chunk_tokens(&mut self, text: &str, _ref_mel: &Array, is_first_chunk: bool) -> Result<(Vec<i32>, Array), Error> {
        let ref_text = self.reference_text.clone()
            .ok_or_else(|| Error::Message("Reference text not set".to_string()))?;
        let prompt_semantic = self.prompt_semantic.clone()
            .ok_or_else(|| Error::Message("Prompt semantic not set".to_string()))?;

        // Same text preprocessing as synthesize_few_shot
        let text = if let Some(first_char) = text.chars().next() {
            let is_punct = matches!(first_char, ',' | '.' | '!' | '?' | '~' | ':' | '—' | '…' |
                                              '，' | '。' | '！' | '？' | '：');
            let is_english = first_char.is_ascii_alphabetic();
            let punct_chars = [',', '.', '!', '?', '~', ':', '—', '…', '，', '。', '！', '？', '：'];
            let first_segment_len = text.chars()
                .take_while(|c| !punct_chars.contains(c))
                .count();

            if is_first_chunk && !is_punct && is_english {
                format!(", {}", text)
            } else if is_first_chunk && !is_punct && first_segment_len < 4 {
                let preview: String = text.chars().take(15).collect();
                eprintln!("   [few-shot] Short first segment ({}): \"{}\" -> prepending '.'", first_segment_len, preview);
                format!(".{}", text)
            } else {
                text.to_string()
            }
        } else {
            text.to_string()
        };

        // 1. Preprocess reference text
        let (ref_phoneme_ids, ref_phonemes, ref_word2ph, ref_text_normalized) = preprocess_text_with_lang(&ref_text, Some(crate::text::Language::Chinese));
        let ref_text_trimmed = ref_text_normalized.trim();
        let ref_text_chars = ref_text_trimmed.chars().count();
        let ref_word2ph_for_bert = &ref_word2ph[..ref_text_chars.min(ref_word2ph.len())];
        let ref_bert_features = self.extract_bert_features(ref_text_trimmed, ref_word2ph_for_bert, ref_phonemes.len())?;

        // 2. Preprocess target text
        let (target_phoneme_ids, target_phonemes, target_word2ph, target_text_normalized) = preprocess_text_with_lang(&text, Some(crate::text::Language::Chinese));
        eprintln!("   [few-shot] Target phonemes ({}):", target_phonemes.len());
        eprintln!("      {:?}", &target_phonemes[..target_phonemes.len().min(20)]);
        eprintln!("   [few-shot] Target normalized: '{}'", target_text_normalized);

        let target_text_trimmed = target_text_normalized.trim();
        let target_text_chars = target_text_trimmed.chars().count();
        let target_word2ph_for_bert = &target_word2ph[..target_text_chars.min(target_word2ph.len())];
        let target_bert_features = self.extract_bert_features(target_text_trimmed, target_word2ph_for_bert, target_phonemes.len())?;

        // 3. Combine ref + target
        let combined_phoneme_ids = mlx_rs::ops::concatenate_axis(&[&ref_phoneme_ids, &target_phoneme_ids], 1)
            .map_err(|e| Error::Message(format!("Failed to concat phonemes: {}", e)))?;
        eval([&combined_phoneme_ids]).map_err(|e| Error::Message(e.to_string()))?;

        let combined_bert_features = mlx_rs::ops::concatenate_axis(&[&ref_bert_features, &target_bert_features], 1)
            .map_err(|e| Error::Message(format!("Failed to concat BERT features: {}", e)))?;
        eval([&combined_bert_features]).map_err(|e| Error::Message(e.to_string()))?;

        // 4. Generate semantic tokens
        let (all_tokens, generated_count) = self.generate_semantic_tokens(
            &combined_phoneme_ids,
            &combined_bert_features,
            target_phonemes.len(),
            Some(&prompt_semantic),
        )?;

        // 5. Extract newly generated tokens
        let new_tokens = all_tokens[all_tokens.len().saturating_sub(generated_count)..].to_vec();
        let prompt_len = prompt_semantic.shape()[2] as usize;
        eprintln!("   [few-shot] Semantic tokens: all={}, prompt={}, generated={}",
                  all_tokens.len(), prompt_len, generated_count);

        Ok((new_tokens, target_phoneme_ids))
    }

    /// Zero-shot synthesis (no reference text, only reference audio for style)
    #[allow(dead_code)]
    fn synthesize_zero_shot(&mut self, text: &str, ref_mel: &Array, is_first_chunk: bool) -> Result<AudioOutput, Error> {
        // Python keeps trailing punctuation as part of the phoneme sequence.
        // Do NOT strip it - it becomes the sentence delimiter phoneme.

        // Python's pre_seg_text: prepend punctuation if text doesn't start with one
        // and first segment is short (< 4 chars). This helps T2S model alignment.
        // Python: if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text
        let text = if let Some(first_char) = text.chars().next() {
            let is_punct = matches!(first_char, ',' | '.' | '!' | '?' | '~' | ':' | '—' | '…' |
                                              '，' | '。' | '！' | '？' | '：');
            let is_english = first_char.is_ascii_alphabetic();

            // get_first: split by punctuation and get first segment
            let punct_chars = [',', '.', '!', '?', '~', ':', '—', '…', '，', '。', '！', '？', '：'];
            let first_segment_len = text.chars()
                .take_while(|c| !punct_chars.contains(c))
                .count();

            if is_first_chunk && !is_punct && is_english {
                // Prepend comma for English text - only for first chunk
                format!(", {}", text)
            } else if is_first_chunk && !is_punct && first_segment_len < 4 {
                // Prepend period if first segment < 4 chars (matches Python's behavior)
                // Only for first chunk - middle chunks shouldn't get extra punctuation
                eprintln!("   [zero-shot] Short first segment ({}): prepending '.'", first_segment_len);
                format!(".{}", text)  // Prepend ASCII period like Python
            } else {
                text.to_string()
            }
        } else {
            text.to_string()
        };

        // 1. Text preprocessing (word2ph comes from preprocessor for correct handling of mixed text)
        let (phoneme_ids, phonemes, word2ph, text_normalized) = preprocess_text_with_lang(&text, Some(crate::text::Language::Chinese));
        eprintln!("   Phonemes: {:?}", &phonemes[..phonemes.len().min(30)]);
        eprintln!("   Normalized: \"{}\"", text_normalized);

        // 2. BERT encoding - use normalized text (quotes/parentheses removed)
        let text_chars = text_normalized.chars().count();
        let word2ph_for_bert = &word2ph[..text_chars.min(word2ph.len())];
        let bert_features = self.extract_bert_features(&text_normalized, word2ph_for_bert, phonemes.len())?;

        // DEBUG: Print BERT features info
        eval([&bert_features]).map_err(|e| Error::Message(e.to_string()))?;
        let bert_shape = bert_features.shape();
        eprintln!("   BERT features shape: {:?}", bert_shape);
        // Print first 5 values of first position
        let bert_flat: Vec<f32> = bert_features.flatten(None, None)
            .map_err(|e| Error::Message(e.to_string()))?
            .as_slice().to_vec();
        eprintln!("   BERT features[0,:5]: {:?}", &bert_flat[..5.min(bert_flat.len())]);
        // Also print position 1 (skip first 1024 values)
        if bert_flat.len() > 1024 {
            eprintln!("   BERT features[1,:5]: {:?}", &bert_flat[1024..1029.min(bert_flat.len())]);
        }

        // 3. Generate semantic tokens
        // For zero-shot, all tokens are newly generated (no prompt)
        let (all_tokens, generated_count) = self.generate_semantic_tokens(&phoneme_ids, &bert_features, phonemes.len(), None)?;
        // Use last generated_count tokens (for zero-shot, this equals all_tokens since no prompt)
        let tokens = &all_tokens[all_tokens.len().saturating_sub(generated_count)..];

        // DEBUG: Print semantic tokens for comparison with Python
        eprintln!("   Semantic tokens ({}): {:?}", tokens.len(), &tokens[..tokens.len().min(20)]);

        // 4. VITS vocoding
        let audio = self.vocode(tokens, &phoneme_ids, ref_mel)?;

        // 5. Convert to output
        let samples = array_to_f32_samples(&audio)?;
        let duration = samples.len() as f32 / self.config.sample_rate as f32;

        Ok(AudioOutput {
            samples,
            sample_rate: self.config.sample_rate,
            duration,
            num_tokens: tokens.len(),
        })
    }

    /// Few-shot synthesis (with reference text and prompt semantic codes)
    #[allow(dead_code)]
    fn synthesize_few_shot(&mut self, text: &str, ref_mel: &Array, is_first_chunk: bool) -> Result<AudioOutput, Error> {
        let ref_text = self.reference_text.clone()
            .ok_or_else(|| Error::Message("Reference text not set".to_string()))?;
        let prompt_semantic = self.prompt_semantic.clone()
            .ok_or_else(|| Error::Message("Prompt semantic not set".to_string()))?;

        // Python keeps trailing punctuation as part of the phoneme sequence.
        // Do NOT strip it - it becomes the sentence delimiter phoneme.

        // Python's pre_seg_text: prepend punctuation if text doesn't start with one
        // and first segment is short (< 4 chars). This helps T2S model alignment.
        let text = if let Some(first_char) = text.chars().next() {
            let is_punct = matches!(first_char, ',' | '.' | '!' | '?' | '~' | ':' | '—' | '…' |
                                              '，' | '。' | '！' | '？' | '：');
            let is_english = first_char.is_ascii_alphabetic();

            // get_first: split by punctuation and get first segment
            let punct_chars = [',', '.', '!', '?', '~', ':', '—', '…', '，', '。', '！', '？', '：'];
            let first_segment_len = text.chars()
                .take_while(|c| !punct_chars.contains(c))
                .count();

            let result = if is_first_chunk && !is_punct && is_english {
                // Prepend comma for English text - only for first chunk
                format!(", {}", text)
            } else if is_first_chunk && !is_punct && first_segment_len < 4 {
                // Short first segment - prepend period to match Python's behavior
                // Only for first chunk - middle chunks shouldn't get extra punctuation
                let preview: String = text.chars().take(15).collect();
                eprintln!("   [few-shot] Short first segment ({}): \"{}\" -> prepending '.'", first_segment_len, preview);
                format!(".{}", text)  // Prepend ASCII period like Python
            } else {
                text.to_string()
            };
            let preview: String = text.chars().take(20).collect();
            eprintln!("   [few-shot] Input text: \"{}...\" -> first_char='{}', first_segment_len={}",
                     preview, first_char, first_segment_len);
            result
        } else {
            text.to_string()
        };

        // 1. Preprocess reference text
        // Note: preprocess_text produces the same phoneme sequence as Python (no special markers)
        let (ref_phoneme_ids, ref_phonemes, ref_word2ph, ref_text_normalized) = preprocess_text_with_lang(&ref_text, Some(crate::text::Language::Chinese));

        // Trim whitespace from normalized text for BERT alignment
        let ref_text_trimmed = ref_text_normalized.trim();
        let ref_text_chars = ref_text_trimmed.chars().count();
        let ref_word2ph_for_bert = &ref_word2ph[..ref_text_chars.min(ref_word2ph.len())];
        let ref_bert_features = self.extract_bert_features(ref_text_trimmed, ref_word2ph_for_bert, ref_phonemes.len())?;

        // 2. Preprocess target text - use normalized text for BERT
        let (target_phoneme_ids, target_phonemes, target_word2ph, target_text_normalized) = preprocess_text_with_lang(&text, Some(crate::text::Language::Chinese));
        eprintln!("   [few-shot] Target phonemes ({}):", target_phonemes.len());
        eprintln!("      {:?}", &target_phonemes[..target_phonemes.len().min(20)]);
        eprintln!("   [few-shot] Target normalized: '{}'", target_text_normalized);

        // Trim whitespace from normalized text for BERT alignment
        let target_text_trimmed = target_text_normalized.trim();
        let target_text_chars = target_text_trimmed.chars().count();
        let target_word2ph_for_bert = &target_word2ph[..target_text_chars.min(target_word2ph.len())];
        let target_bert_features = self.extract_bert_features(target_text_trimmed, target_word2ph_for_bert, target_phonemes.len())?;

        // 3. Combine: ref_phones + target_phones (NO extra period — Python doesn't append one)
        let combined_phoneme_ids = mlx_rs::ops::concatenate_axis(&[&ref_phoneme_ids, &target_phoneme_ids], 1)
            .map_err(|e| Error::Message(format!("Failed to concat phonemes: {}", e)))?;
        eval([&combined_phoneme_ids]).map_err(|e| Error::Message(e.to_string()))?;

        // 4. Combine: all_bert = ref_bert + target_bert (Python: torch.cat([prompt_data["bert_features"], item["bert_features"]], 1))
        let combined_bert_features = mlx_rs::ops::concatenate_axis(&[&ref_bert_features, &target_bert_features], 1)
            .map_err(|e| Error::Message(format!("Failed to concat BERT features: {}", e)))?;
        eval([&combined_bert_features]).map_err(|e| Error::Message(e.to_string()))?;

        // 5. Generate semantic tokens
        // Use TARGET phoneme count for bounds - prompt_semantic covers ref portion,
        // we only generate new tokens for target text
        let (all_tokens, generated_count) = self.generate_semantic_tokens(
            &combined_phoneme_ids,
            &combined_bert_features,
            target_phonemes.len(),  // Bounds based on target only
            Some(&prompt_semantic),
        )?;

        // 6. Extract only newly generated tokens for VITS (like Python: item[-idx:])
        // Python uses item[-idx:] where idx is the exact count of newly generated tokens
        // This matches exactly - take the LAST generated_count tokens
        let new_tokens = &all_tokens[all_tokens.len().saturating_sub(generated_count)..];
        let prompt_len = prompt_semantic.shape()[2] as usize;  // Shape is [1, 1, N]
        eprintln!("   [few-shot] Semantic tokens: all={}, prompt={}, generated={}",
                  all_tokens.len(), prompt_len, generated_count);
        eprintln!("   [few-shot] First 10 new tokens: {:?}", &new_tokens[..new_tokens.len().min(10)]);


        // Don't pad - the tokens should be correct with Python prompt codes
        let new_tokens: Vec<i32> = new_tokens.to_vec();

        // 7. VITS vocoding with target phonemes only (NO extra period — matching Python)
        let audio = self.vocode(&new_tokens, &target_phoneme_ids, ref_mel)?;

        // 8. Convert to output
        let samples = array_to_f32_samples(&audio)?;
        let duration = samples.len() as f32 / self.config.sample_rate as f32;

        Ok(AudioOutput {
            samples,
            sample_rate: self.config.sample_rate,
            duration,
            num_tokens: new_tokens.len(),
        })
    }

    /// Extract BERT features with proper alignment
    /// Extract BERT features for text.
    ///
    /// For mixed Chinese/English text (matching Python's all_zh path):
    /// - Segments text by language
    /// - Chinese sub-segments get real BERT features
    /// - English sub-segments get zero features
    /// - All concatenated in order
    fn extract_bert_features(&mut self, text: &str, word2ph: &[i32], phoneme_count: usize) -> Result<Array, Error> {
        use crate::text::{detect_language, Language};

        let language = detect_language(text);
        eprintln!("   [BERT] text='{}', detected={:?}", text, language);

        // Pure English: all zeros
        if matches!(language, Language::English) {
            eprintln!("   [BERT] Using zeros (non-Chinese)");
            let bert_features = Array::zeros::<f32>(&[1, phoneme_count as i32, 1024])
                .map_err(|e| Error::Message(e.to_string()))?;
            eval([&bert_features]).map_err(|e| Error::Message(e.to_string()))?;
            return Ok(bert_features);
        }

        // Mixed text: segment by language, BERT for Chinese only, zeros for English
        // This matches Python's all_zh path which uses LangSegment to split,
        // then get_bert_inf returns real features for zh and zeros for en.
        if matches!(language, Language::Mixed) {
            eprintln!("   [BERT] Mixed text - segmenting by language");
            use crate::text::preprocessor::segment_by_language;

            let segments = segment_by_language(text);
            let mut all_parts: Vec<Array> = Vec::new();
            let mut w2p_idx = 0; // Track position in word2ph (NOT char position)

            for seg in &segments {
                // Compute how many word2ph entries this segment uses:
                // - Chinese: one entry per character
                // - English: one entry per word/number/punctuation token
                let seg_w2p_count = if seg.is_english {
                    count_english_word2ph_entries(&seg.text)
                } else {
                    seg.text.chars().count()
                };

                let end_idx = (w2p_idx + seg_w2p_count).min(word2ph.len());
                let seg_word2ph = &word2ph[w2p_idx..end_idx];
                let seg_phoneme_count: i32 = seg_word2ph.iter().sum();

                if seg_phoneme_count <= 0 {
                    w2p_idx = end_idx;
                    continue;
                }

                if seg.is_english {
                    // English: zero features
                    let zeros = Array::zeros::<f32>(&[1, seg_phoneme_count, 1024])
                        .map_err(|e| Error::Message(e.to_string()))?;
                    all_parts.push(zeros);
                    eprintln!("   [BERT] English segment '{}': {} phonemes (zeros)",
                             &seg.text.chars().take(20).collect::<String>(), seg_phoneme_count);
                } else {
                    // Chinese: real BERT
                    let bert_raw = self.bert.extract_features(&seg.text, seg_word2ph)?;
                    eval([&bert_raw]).map_err(|e| Error::Message(e.to_string()))?;
                    let bert_len = bert_raw.shape()[1] as i32;
                    let bert = if bert_len < seg_phoneme_count {
                        let pad = Array::zeros::<f32>(&[1, seg_phoneme_count - bert_len, 1024])
                            .map_err(|e| Error::Message(e.to_string()))?;
                        mlx_rs::ops::concatenate_axis(&[&bert_raw, &pad], 1)
                            .map_err(|e| Error::Message(e.to_string()))?
                    } else if bert_len > seg_phoneme_count {
                        bert_raw.index((.., ..seg_phoneme_count, ..))
                    } else {
                        bert_raw
                    };
                    all_parts.push(bert);
                    eprintln!("   [BERT] Chinese segment '{}': {} phonemes (real BERT)",
                             &seg.text.chars().take(20).collect::<String>(), seg_phoneme_count);
                }
                w2p_idx = end_idx;
            }

            // Handle any remaining phonemes (e.g., trailing punctuation)
            let total_bert: i32 = all_parts.iter().map(|a| a.shape()[1] as i32).sum();
            let phoneme_count_i32 = phoneme_count as i32;
            if total_bert < phoneme_count_i32 {
                let pad = Array::zeros::<f32>(&[1, phoneme_count_i32 - total_bert, 1024])
                    .map_err(|e| Error::Message(e.to_string()))?;
                all_parts.push(pad);
            }

            if all_parts.is_empty() {
                let zeros = Array::zeros::<f32>(&[1, phoneme_count_i32, 1024])
                    .map_err(|e| Error::Message(e.to_string()))?;
                eval([&zeros]).map_err(|e| Error::Message(e.to_string()))?;
                return Ok(zeros);
            }

            let parts_refs: Vec<&Array> = all_parts.iter().collect();
            let combined = mlx_rs::ops::concatenate_axis(&parts_refs, 1)
                .map_err(|e| Error::Message(format!("Failed to concat mixed BERT: {}", e)))?;
            // Trim or pad to exact phoneme count
            let combined = if combined.shape()[1] as i32 > phoneme_count_i32 {
                combined.index((.., ..phoneme_count_i32, ..))
            } else {
                combined
            };
            eval([&combined]).map_err(|e| Error::Message(e.to_string()))?;
            return Ok(combined);
        }

        // Pure Chinese: extract actual BERT features
        let bert_features_raw = self.bert.extract_features(text, word2ph)?;
        eval([&bert_features_raw]).map_err(|e| Error::Message(e.to_string()))?;

        let bert_seq_len = bert_features_raw.shape()[1] as i32;
        let phoneme_count = phoneme_count as i32;

        let bert_features = if bert_seq_len < phoneme_count {
            let pad_len = phoneme_count - bert_seq_len;
            let padding = Array::zeros::<f32>(&[1, pad_len, 1024])
                .map_err(|e| Error::Message(e.to_string()))?;
            mlx_rs::ops::concatenate_axis(&[&bert_features_raw, &padding], 1)
                .map_err(|e| Error::Message(e.to_string()))?
        } else if bert_seq_len > phoneme_count {
            bert_features_raw.index((.., ..phoneme_count, ..))
        } else {
            bert_features_raw
        };

        eval([&bert_features]).map_err(|e| Error::Message(e.to_string()))?;
        Ok(bert_features)
    }

    /// Generate semantic tokens from phonemes and BERT features
    ///
    /// # Arguments
    /// * `phoneme_ids` - Phoneme token IDs
    /// * `bert_features` - BERT features
    /// * `phoneme_count` - Number of phonemes (for generation bounds)
    /// * `prompt_semantic` - Optional prompt semantic codes for few-shot mode
    ///
    /// # Returns
    /// Tuple of (all_tokens, generated_count) like Python's (y, idx)
    /// - all_tokens: prompt + newly generated tokens
    /// - generated_count: number of NEW tokens (use `all_tokens[all_tokens.len()-generated_count..]`)
    fn generate_semantic_tokens(
        &mut self,
        phoneme_ids: &Array,
        bert_features: &Array,
        phoneme_count: usize,
        prompt_semantic: Option<&Array>,
    ) -> Result<(Vec<i32>, usize), Error> {
        // Set seed like Python (seed=233333)
        random::seed(233333).map_err(|e| Error::Message(e.to_string()))?;

        let batch_size = 1;
        let num_layers = self.t2s_config.num_layers as usize;

        let mut caches: Vec<Option<ConcatKeyValueCache>> = (0..num_layers).map(|_| None).collect();

        // Extract prompt tokens for repetition penalty (like Python's y = prompts)
        // Python applies repetition penalty to ALL previous tokens including prompt
        let prompt_tokens: Vec<i32> = if let Some(prompt) = prompt_semantic {
            let prompt_squeezed = prompt.squeeze()
                .map_err(|e| Error::Message(e.to_string()))?;
            eval([&prompt_squeezed]).map_err(|e| Error::Message(e.to_string()))?;
            prompt_squeezed.as_slice().to_vec()
        } else {
            vec![]
        };


        // For few-shot mode, use prompt_semantic as initial semantic_ids
        // For zero-shot mode, start with zeros
        let mut semantic_ids = if let Some(prompt) = prompt_semantic {
            // prompt is [batch, 1, seq], we need [batch, seq]
            let prompt_squeezed = prompt.squeeze()
                .map_err(|e| Error::Message(e.to_string()))?;
            // If it's 1D, add batch dimension
            if prompt_squeezed.ndim() == 1 {
                let seq_len = prompt_squeezed.shape()[0] as i32;
                prompt_squeezed.reshape(&[1, seq_len])
                    .map_err(|e| Error::Message(e.to_string()))?
            } else {
                prompt_squeezed
            }
        } else {
            Array::zeros::<i32>(&[batch_size, 1])
                .map_err(|e| Error::Message(e.to_string()))?
        };

        // Prefill
        let input = T2SInput {
            phoneme_ids,
            semantic_ids: &semantic_ids,
            bert_features,
            cache: &mut caches,
        };
        let logits = self.t2s.forward(input)
            .map_err(|e| Error::Message(e.to_string()))?;
        eval([&logits]).map_err(|e| Error::Message(e.to_string()))?;

        // First token - include prompt_tokens in repetition penalty (Python behavior)
        let seq_len = logits.shape()[1];
        let last_logits = logits.index((.., seq_len - 1, ..)).squeeze()
            .map_err(|e| Error::Message(e.to_string()))?;
        let eos_token = 1024;

        // Create sampler with config
        let sampling_config = SamplingConfig {
            top_k: self.config.top_k,
            top_p: self.config.top_p,
            temperature: self.config.temperature,
            repetition_penalty: self.config.repetition_penalty,
            eos_token,
        };
        let mut sampler = Sampler::new(sampling_config);

        // Python masks EOS during first 11 tokens (idx < 11), so we mask here (idx=0)
        // IMPORTANT: For first token, DON'T apply penalty to prompt tokens
        // Python's behavior: penalty is only applied to previously GENERATED tokens
        // At step 0, there are no generated tokens yet, so no penalty
        let (mut token_id, _) = sampler.sample_with_eos_mask(&last_logits)?;
        semantic_ids = Array::from_slice(&[token_id], &[1, 1]);
        // all_tokens contains prompt + generated (Python behavior: returns prompts + new tokens to VITS)
        // This matches Python's infer_panel which returns pred_semantic that includes prompt semantic
        let prompt_len = prompt_tokens.len();
        let mut all_tokens: Vec<i32> = prompt_tokens.clone();
        all_tokens.push(token_id);
        // Track number of newly generated tokens (excluding prompt)
        let mut generated_count: usize = 1;

        // Generation bounds - adjusted to match Python's token generation rate
        // Python generates ~5-8 tokens per phoneme for natural speech
        // Short text needs more tokens per phoneme for complete pronunciation
        let tokens_per_phoneme = if phoneme_count <= 10 { 6.0 } else { 4.0 };
        let _target_tokens = (phoneme_count as f32 * tokens_per_phoneme) as usize;
        let max_tokens = (phoneme_count * 10).max(100);
        // min_tokens: Allow EOS after generating at least 4 tokens per phoneme for short text
        let min_tokens = (phoneme_count as f32 * if phoneme_count <= 10 { 4.0 } else { 3.0 }) as usize;

        // Autoregressive generation
        for step in 1..max_tokens {
            let input = T2SInput {
                phoneme_ids,
                semantic_ids: &semantic_ids,
                bert_features,
                cache: &mut caches,
            };

            let logits = self.t2s.forward(input)
                .map_err(|e| Error::Message(e.to_string()))?;
            eval([&logits]).map_err(|e| Error::Message(e.to_string()))?;

            let seq_len = logits.shape()[1];
            let last_logits = logits.index((.., seq_len - 1, ..)).squeeze()
                .map_err(|e| Error::Message(e.to_string()))?;


            // Sample with repetition penalty applied to ALL previous tokens (including prompt)
            // This matches Python's behavior: sample(logits, y, ...) where y includes prompts
            // Python masks EOS during first 11 tokens: if(idx<11): logits = logits[:, :-1]
            // step=1 corresponds to Python idx=1, so mask_eos when step < 11
            let mask_eos = step < 11;

            let (sampled, argmax_token) = if mask_eos {
                sampler.sample_with_eos_mask(&last_logits)?
            } else {
                sampler.sample(&last_logits)?
            };
            token_id = sampled;

            // EOS detection: match Python exactly (line 871):
            //   if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
            //       stop = True
            let eos_detected = token_id == eos_token || argmax_token == eos_token;

            if eos_detected {
                eprintln!("   [T2S] EOS at step {}: token={}, argmax={}, eos_token={}",
                         generated_count, token_id, argmax_token, eos_token);
                break;
            }

            all_tokens.push(token_id);
            sampler.add_token(token_id);
            generated_count += 1;

            // Repetition detection for longer patterns (check only generated portion)
            if generated_count > min_tokens && detect_repetition(&all_tokens[prompt_len..], 3, 8) {
                eprintln!("   [T2S] Repetition detected at step {}", generated_count);
                while generated_count > min_tokens && detect_repetition(&all_tokens[prompt_len..], 3, 5) {
                    all_tokens.pop();
                    generated_count -= 1;
                }
                break;
            }

            semantic_ids = Array::from_slice(&[token_id], &[1, 1]);
        }

        // Return (all_tokens, generated_count) like Python's (y, idx)
        // This allows caller to extract exactly the last `generated_count` tokens
        Ok((all_tokens, generated_count))
    }

    /// Vocode semantic tokens to audio
    fn vocode(&mut self, tokens: &[i32], phoneme_ids: &Array, ref_mel: &Array) -> Result<Array, Error> {
        let codes = Array::from_slice(tokens, &[1, 1, tokens.len() as i32]);

        let text_ids = phoneme_ids.squeeze()
            .map_err(|e| Error::Message(e.to_string()))?;
        let text_for_vits = text_ids.index(mlx_rs::ops::indexing::NewAxis);

        let audio = self.vits.decode(&codes, &text_for_vits, Some(ref_mel), self.config.noise_scale, self.config.speed)
            .map_err(|e| Error::Message(e.to_string()))?;

        eval([&audio]).map_err(|e| Error::Message(e.to_string()))?;

        // Python does: audio.detach()[0, 0, :] - extract first batch, first channel
        // audio shape is [batch, channels, time] = [1, 1, time]
        let audio = audio.index((0, 0, ..));

        Ok(audio)
    }

    /// Save audio to WAV file
    pub fn save_wav(&self, audio: &AudioOutput, path: impl AsRef<Path>) -> Result<(), Error> {
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let path = path.as_ref();
        let samples = audio.to_i16_samples();

        let file = File::create(path)
            .map_err(|e| Error::Message(format!("Failed to create file: {}", e)))?;
        let mut writer = BufWriter::new(file);

        let data_size = (samples.len() * 2) as u32;
        let file_size = 36 + data_size;

        // RIFF header
        writer.write_all(b"RIFF").map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&file_size.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(b"WAVE").map_err(|e| Error::Message(e.to_string()))?;

        // fmt chunk
        writer.write_all(b"fmt ").map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&16u32.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&1u16.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&1u16.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&audio.sample_rate.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&(audio.sample_rate * 2).to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&2u16.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&16u16.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;

        // data chunk
        writer.write_all(b"data").map_err(|e| Error::Message(e.to_string()))?;
        writer.write_all(&data_size.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;

        for sample in samples {
            writer.write_all(&sample.to_le_bytes()).map_err(|e| Error::Message(e.to_string()))?;
        }

        Ok(())
    }

    /// Play audio using system player (macOS: afplay)
    #[cfg(target_os = "macos")]
    pub fn play(&self, audio: &AudioOutput) -> Result<(), Error> {
        // Save to temp file
        let temp_path = "/tmp/voice_clone_playback.wav";
        self.save_wav(audio, temp_path)?;

        // Play with afplay (non-blocking)
        Command::new("afplay")
            .arg(temp_path)
            .spawn()
            .map_err(|e| Error::Message(format!("Failed to play audio: {}", e)))?;

        Ok(())
    }

    /// Play audio and wait for completion
    #[cfg(target_os = "macos")]
    pub fn play_blocking(&self, audio: &AudioOutput) -> Result<(), Error> {
        let temp_path = "/tmp/voice_clone_playback.wav";
        self.save_wav(audio, temp_path)?;

        Command::new("afplay")
            .arg(temp_path)
            .status()
            .map_err(|e| Error::Message(format!("Failed to play audio: {}", e)))?;

        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    pub fn play(&self, _audio: &AudioOutput) -> Result<(), Error> {
        Err(Error::Message("Audio playback not implemented for this platform".to_string()))
    }

    #[cfg(not(target_os = "macos"))]
    pub fn play_blocking(&self, _audio: &AudioOutput) -> Result<(), Error> {
        Err(Error::Message("Audio playback not implemented for this platform".to_string()))
    }
}

/// Compute word2ph (phonemes per character) for text
#[allow(dead_code)]
fn compute_word2ph(text: &str) -> Vec<i32> {
    let mut word2ph = Vec::new();
    for c in text.chars() {
        if c == '，' || c == '。' || c == '！' || c == '？' || c == '；' || c == '：'
            || c == ',' || c == '.' || c == '!' || c == '?' || c == ';' || c == ':'
        {
            word2ph.push(1);
        } else if c.is_whitespace() {
            word2ph.push(1);
        } else {
            word2ph.push(2); // Most Chinese chars have 2 phonemes (initial + final)
        }
    }
    word2ph
}

/// Global language detector (lazy initialized)
/// Uses lingua for ML-based language detection like Python's LangSegment
#[allow(dead_code)]
static LANG_DETECTOR: OnceLock<LanguageDetector> = OnceLock::new();

#[allow(dead_code)]
fn get_lang_detector() -> &'static LanguageDetector {
    LANG_DETECTOR.get_or_init(|| {
        LanguageDetectorBuilder::from_languages(&[
            Language::Chinese,
            Language::English,
            Language::Japanese,
            Language::Korean,
        ])
        .with_preloaded_language_models()
        .build()
    })
}

/// Check if a character is CJK (Chinese, Japanese, or Korean)
fn is_cjk_char(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}' |  // CJK Unified Ideographs
        '\u{3400}'..='\u{4DBF}' |  // CJK Extension A
        '\u{3040}'..='\u{309F}' |  // Hiragana
        '\u{30A0}'..='\u{30FF}' |  // Katakana
        '\u{AC00}'..='\u{D7AF}' |  // Korean Hangul
        '\u{1100}'..='\u{11FF}'    // Korean Jamo
    )
}

/// Python-compatible `cut5` text segmentation: split at every punctuation mark,
/// then merge short segments (< threshold chars) with the next segment.
/// This matches the dora-primespeech `cut5` method + `merge_short_text_in_array(5)`.
/// Count word2ph entries for an English text segment in mixed G2P mode.
/// Must match `english_g2p`: one entry per word + one per punctuation + one per apostrophe.
fn count_english_word2ph_entries(text: &str) -> usize {
    let mut count = 0;
    let mut in_word = false;
    for c in text.chars() {
        if c.is_ascii_alphabetic() {
            if !in_word {
                count += 1;
                in_word = true;
            }
        } else if c == '\'' {
            in_word = false;
            count += 1; // apostrophe emits '-' phoneme
        } else {
            in_word = false;
            if matches!(c, ',' | '.' | '!' | '?') {
                count += 1;
            }
        }
    }
    count
}

fn cut5_split(text: &str) -> Vec<String> {
    let text = text.trim_matches('\n');
    if text.is_empty() {
        return vec![];
    }

    let puncts: &[char] = &[',', '.', ';', '?', '!', '、', '，', '。', '？', '！', '；', '：', '…'];
    let chars: Vec<char> = text.chars().collect();
    let mut merge_items: Vec<String> = Vec::new();
    let mut current = String::new();

    for (i, &ch) in chars.iter().enumerate() {
        if puncts.contains(&ch) {
            // Special case: decimal point (digit.digit) — don't split
            if ch == '.' && i > 0 && i < chars.len() - 1
                && chars[i - 1].is_ascii_digit() && chars[i + 1].is_ascii_digit()
            {
                current.push(ch);
            } else {
                current.push(ch);
                merge_items.push(current.clone());
                current.clear();
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        merge_items.push(current);
    }

    // Filter out pure-punctuation segments
    let filtered: Vec<String> = merge_items
        .into_iter()
        .filter(|item| !item.chars().all(|c| puncts.contains(&c) || c.is_whitespace()))
        .collect();

    // merge_short_text_in_array(texts, 5)
    let threshold = 5;
    if filtered.len() < 2 {
        return filtered;
    }
    let mut result: Vec<String> = Vec::new();
    let mut acc = String::new();
    for ele in &filtered {
        acc.push_str(ele);
        if acc.chars().count() >= threshold {
            result.push(acc.clone());
            acc.clear();
        }
    }
    if !acc.is_empty() {
        if let Some(last) = result.last_mut() {
            last.push_str(&acc);
        } else {
            result.push(acc);
        }
    }

    // Python also: filter empty/whitespace, append "。" if not ending with punct, split >510
    let splits_set: &[char] = &['，', '。', '？', '！', ',', '.', '?', '!', '~', ':', '：', '—', '…', '、', '；'];
    result.into_iter()
        .filter(|t| !t.trim().is_empty())
        .filter(|t| t.chars().any(|c| c.is_alphanumeric() || is_cjk_char(c)))
        .map(|mut t| {
            if !t.ends_with(splits_set) {
                t.push('。');
            }
            t
        })
        .collect()
}

/// Language-aware text segmentation (like Python's LangSegment)
///
/// Uses a hybrid approach:
/// 1. First splits by character class (ASCII letters = English, CJK = Chinese/Japanese/Korean)
/// 2. Uses lingua ML model for ambiguous cases
///
/// This matches Python's LangSegment which uses regex for obvious patterns
/// and py3langid for edge cases.
#[allow(dead_code)]
fn split_text_by_language(text: &str) -> Vec<String> {
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Lang { English, Cjk, Other }

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    // Helper: check if a character is a digit or decimal point
    let is_digit_or_dot = |c: char| c.is_ascii_digit() || c == '.';

    // Step 1: Split by character class (like Python's regex patterns)
    // Special handling: numbers followed directly by CJK go with CJK (e.g., "126.4亿斤")
    let mut char_segments: Vec<(Lang, String)> = Vec::new();
    let mut current = String::new();
    let mut current_lang = Lang::Other;

    let mut i = 0;
    while i < len {
        let ch = chars[i];
        let char_lang = if ch.is_ascii_alphabetic() {
            Lang::English
        } else if is_cjk_char(ch) {
            Lang::Cjk
        } else {
            Lang::Other  // punctuation, numbers, spaces, quotes
        };

        match char_lang {
            Lang::English => {
                if current_lang == Lang::Cjk && !current.is_empty() {
                    char_segments.push((Lang::Cjk, std::mem::take(&mut current)));
                }
                current.push(ch);
                current_lang = Lang::English;
            }
            Lang::Cjk => {
                if current_lang == Lang::English && !current.is_empty() {
                    char_segments.push((Lang::English, std::mem::take(&mut current)));
                }
                current.push(ch);
                current_lang = Lang::Cjk;
            }
            Lang::Other => {
                // Check if this is a number followed directly by CJK
                if is_digit_or_dot(ch) {
                    // Look ahead to see what follows the number
                    let mut j = i;
                    while j < len && is_digit_or_dot(chars[j]) {
                        j += 1;
                    }
                    // If number is followed directly by CJK (no space), treat as CJK
                    if j < len && is_cjk_char(chars[j]) {
                        // Push current English segment if any
                        if current_lang == Lang::English && !current.is_empty() {
                            char_segments.push((Lang::English, std::mem::take(&mut current)));
                        }
                        // Add all the digits to CJK segment
                        while i < j {
                            current.push(chars[i]);
                            i += 1;
                        }
                        current_lang = Lang::Cjk;
                        continue;  // Don't increment i again
                    }
                }
                // Otherwise, punctuation/numbers/quotes attach to current segment
                current.push(ch);
            }
        }
        i += 1;
    }
    if !current.is_empty() {
        char_segments.push((current_lang, current));
    }

    // Step 2: For CJK segments, use lingua to detect if it's Japanese/Korean vs Chinese
    // (This matters for phoneme processing)
    let detector = get_lang_detector();
    let mut lang_segments: Vec<(Language, String)> = Vec::new();

    for (lang, text) in char_segments {
        if text.trim().is_empty() {
            continue;
        }
        match lang {
            Lang::English => {
                lang_segments.push((Language::English, text));
            }
            Lang::Cjk | Lang::Other => {
                // Use lingua to detect Chinese vs Japanese vs Korean
                if let Some(detected) = detector.detect_language_of(&text) {
                    lang_segments.push((detected, text));
                } else {
                    // Default to Chinese
                    lang_segments.push((Language::Chinese, text));
                }
            }
        }
    }

    // Step 3: Split CJK segments at sentence-ending punctuation
    let cjk_sentence_end: std::collections::HashSet<char> =
        ['。', '？', '！', '．'].into_iter().collect();

    let mut result = Vec::new();
    for (lang, text) in lang_segments {
        if matches!(lang, Language::Chinese | Language::Japanese | Language::Korean) {
            // Split CJK at sentence-ending punctuation
            let mut sub = String::new();
            for ch in text.chars() {
                sub.push(ch);
                if cjk_sentence_end.contains(&ch) {
                    if !sub.trim().is_empty() {
                        result.push(sub.clone());
                    }
                    sub.clear();
                }
            }
            if !sub.trim().is_empty() {
                result.push(sub);
            }
        } else {
            // Keep English segments whole
            if !text.trim().is_empty() {
                result.push(text);
            }
        }
    }

    // Step 4: Filter out segments that are too short (only punctuation, less than 2 actual characters)
    let result: Vec<String> = result.into_iter().filter(|s| {
        let content_chars = s.chars().filter(|c| {
            !matches!(*c, ',' | '.' | ';' | '?' | '!' | '、' | '，' | '。' | '？' | '！' | '；' | '：' | '…' | '\u{201C}' | '\u{201D}' | '\'' | '（' | '）' | '(' | ')' | '《' | '》' | '【' | '】')
        }).count();
        content_chars >= 2
    }).collect();

    // If filtering removed all segments, return original text as single segment
    if result.is_empty() {
        return vec![text.to_string()];
    }

    // Step 5: Merge very short Chinese segments (<=4 content chars) with adjacent Chinese segments
    // This prevents isolated short Chinese chunks after English from having poor context
    let mut merged: Vec<String> = Vec::new();
    for seg in result {
        let content_chars: usize = seg.chars().filter(|c| is_cjk_char(*c)).count();
        let is_short_cjk = content_chars > 0 && content_chars <= 4;

        if is_short_cjk {
            if let Some(prev) = merged.last() {
                // Check if we can merge with previous segment
                let prev_has_cjk = prev.chars().any(|c| is_cjk_char(c));
                let prev_ends_with_punct = prev.chars().last()
                    .map(|c| matches!(c, '）' | ')' | '》' | '】' | '"'))
                    .unwrap_or(false);

                if prev_has_cjk || prev_ends_with_punct {
                    // Merge with previous segment
                    if let Some(prev) = merged.pop() {
                        merged.push(format!("{}{}", prev, seg));
                        continue;
                    }
                }
            }
        }
        merged.push(seg);
    }

    // Note: Step 6 removed - was a hard-coded split for "合并，并" patterns.
    // The proper fix is to zero out BERT features at punctuation positions (SP phonemes)
    // which prevents the T2S model from being confused by punctuation context.

    merged
}

/// Estimate phoneme count for a text segment
///
/// Rough estimation: Chinese chars ~2 phonemes, English words ~3-4 phonemes
#[allow(dead_code)]
fn estimate_phoneme_count(text: &str) -> usize {
    let mut count = 0;
    let mut in_english_word = false;

    for c in text.chars() {
        if c.is_ascii_alphabetic() {
            if !in_english_word {
                // Start of English word: estimate 3-4 phonemes per word
                count += 3;
                in_english_word = true;
            }
        } else {
            in_english_word = false;
            if is_cjk_char(c) {
                // Chinese char: ~2 phonemes (initial + final)
                count += 2;
            } else if c.is_ascii_punctuation() || matches!(c, '，' | '。' | '？' | '！' | '、' | '；' | '：') {
                // Punctuation: 1 phoneme (SP or similar)
                count += 1;
            }
        }
    }
    count
}

/// Chunk segments that exceed max_phonemes by splitting at comma/space boundaries
///
/// This prevents T2S attention degradation on very long sequences.
#[allow(dead_code)]
fn chunk_segments_by_length(segments: &[String], max_phonemes: usize) -> Vec<String> {
    let comma_chars: std::collections::HashSet<char> = ['，', ',', '、', '；', ';'].into_iter().collect();

    let mut result = Vec::new();

    for segment in segments {
        let estimated = estimate_phoneme_count(segment);

        if estimated <= max_phonemes {
            // Segment is short enough, keep as-is
            result.push(segment.clone());
        } else {
            // Split at comma/pause boundaries
            let mut current = String::new();
            let mut current_est = 0;
            let mut _last_split_point = 0;
            let chars: Vec<char> = segment.chars().collect();

            for (i, &c) in chars.iter().enumerate() {
                current.push(c);

                // Update estimate
                if c.is_ascii_alphabetic() {
                    // Rough: each English letter adds ~0.5 phoneme
                    current_est += 1;  // Will be divided by 2 effectively
                } else if is_cjk_char(c) {
                    current_est += 2;
                } else if c.is_ascii_punctuation() || matches!(c, '，' | '。' | '？' | '！' | '、' | '；' | '：') {
                    current_est += 1;
                }

                // Check if we hit a comma and have enough content
                let at_comma = comma_chars.contains(&c);
                let at_space = c == ' ' && current_est > 30;  // Also split at space for English

                if (at_comma || at_space) && current_est >= 20 {
                    // Check if remaining segment is substantial
                    let remaining_est = estimate_phoneme_count(&chars[i+1..].iter().collect::<String>());
                    if remaining_est >= 10 || at_comma {
                        if !current.trim().is_empty() {
                            result.push(current.clone());
                            current.clear();
                            current_est = 0;
                            _last_split_point = i + 1;
                        }
                    }
                }

                // Force split if we're way over limit
                if current_est > max_phonemes + 30 {
                    if !current.trim().is_empty() {
                        result.push(current.clone());
                        current.clear();
                        current_est = 0;
                        _last_split_point = i + 1;
                    }
                }
            }

            // Add remaining
            if !current.trim().is_empty() {
                result.push(current);
            }
        }
    }

    result
}

/// Split text at punctuation marks (like Python's cut5 method)
///
/// This splits text at: , . ; ? ! 、，。？！；：…
/// Numbers with decimal points (e.g., "3.14") are kept together.
#[allow(dead_code)]
fn split_text_at_punctuation(text: &str) -> Vec<String> {
    let punctuation: std::collections::HashSet<char> = [
        ',', '.', ';', '?', '!',  // English
        '、', '，', '。', '？', '！', '；', '：', '…',  // Chinese
    ].into_iter().collect();

    let chars: Vec<char> = text.chars().collect();
    let mut segments = Vec::new();
    let mut current = String::new();

    for (i, &ch) in chars.iter().enumerate() {
        if punctuation.contains(&ch) {
            // Check if it's a decimal point (digit.digit)
            if ch == '.' && i > 0 && i < chars.len() - 1 {
                if chars[i - 1].is_ascii_digit() && chars[i + 1].is_ascii_digit() {
                    current.push(ch);
                    continue;
                }
            }
            // Add punctuation to current segment
            current.push(ch);
            // Save segment if it has content
            let trimmed = current.trim();
            if !trimmed.is_empty() && !trimmed.chars().all(|c| punctuation.contains(&c)) {
                segments.push(current.clone());
            }
            current.clear();
        } else {
            current.push(ch);
        }
    }

    // Add remaining text
    if !current.trim().is_empty() {
        segments.push(current);
    }

    // If no segments were created, return original text
    if segments.is_empty() {
        vec![text.to_string()]
    } else {
        segments
    }
}

/// Convert audio array to f32 samples
fn array_to_f32_samples(audio: &Array) -> Result<Vec<f32>, Error> {
    eval([audio]).map_err(|e| Error::Message(e.to_string()))?;

    let flat = audio.flatten(None, None)
        .map_err(|e| Error::Message(e.to_string()))?;
    eval([&flat]).map_err(|e| Error::Message(e.to_string()))?;

    Ok(flat.as_slice().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_word2ph() {
        let word2ph = compute_word2ph("你好，世界！");
        assert_eq!(word2ph, vec![2, 2, 1, 2, 2, 1]); // 你(2) 好(2) ，(1) 世(2) 界(2) ！(1)
    }

    #[test]
    fn test_detect_repetition() {
        let tokens = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];
        assert!(detect_repetition(&tokens, 3, 3));
        assert!(!detect_repetition(&tokens, 3, 4));
    }
}
