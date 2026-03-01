use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use crate::error::Result;

// ============================================================================
// Model type detection
// ============================================================================

/// Qwen3-TTS model variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// Base model — supports voice cloning via speaker encoder
    Base,
    /// CustomVoice model — 9 preset speakers, no voice cloning
    CustomVoice,
    /// VoiceDesign model — text-described voice characteristics
    VoiceDesign,
}

impl ModelType {
    /// Parse from the `tts_model_type` config field.
    pub fn from_config_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "custom_voice" | "customvoice" => Self::CustomVoice,
            "voice_design" | "voicedesign" => Self::VoiceDesign,
            "base" => Self::Base,
            _ => {
                tracing::warn!("Unknown tts_model_type '{}', assuming Base", s);
                Self::Base
            }
        }
    }

    /// Whether this model supports preset speakers (CustomVoice only).
    pub fn supports_preset_speakers(&self) -> bool {
        matches!(self, Self::CustomVoice)
    }

    /// Whether this model supports voice cloning (Base only).
    pub fn supports_voice_cloning(&self) -> bool {
        matches!(self, Self::Base)
    }

    /// Whether this model supports voice design via text instructions.
    pub fn supports_voice_design(&self) -> bool {
        matches!(self, Self::VoiceDesign)
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Base => write!(f, "Base"),
            Self::CustomVoice => write!(f, "CustomVoice"),
            Self::VoiceDesign => write!(f, "VoiceDesign"),
        }
    }
}

// ============================================================================
// Top-level config
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsConfig {
    pub model_type: String,
    pub tts_model_type: String,
    pub tts_model_size: String,

    pub assistant_token_id: u32,
    pub im_start_token_id: u32,
    pub im_end_token_id: u32,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,

    pub talker_config: TalkerConfig,

    /// Speaker encoder config (Base model only, for voice cloning)
    #[serde(default)]
    pub speaker_encoder_config: Option<SpeakerEncoderJsonConfig>,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,
}

/// Minimal speaker encoder config as it appears in config.json
#[derive(Debug, Clone, Deserialize)]
pub struct SpeakerEncoderJsonConfig {
    #[serde(default = "default_enc_dim")]
    pub enc_dim: i32,
    #[serde(default = "default_speaker_sample_rate")]
    pub sample_rate: u32,
}

fn default_enc_dim() -> i32 { 2048 }
fn default_speaker_sample_rate() -> u32 { 24000 }

impl Qwen3TtsConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let file = std::fs::File::open(model_dir.join("config.json"))?;
        Ok(serde_json::from_reader(file)?)
    }

    pub fn quant_config(&self) -> Option<&QuantizationConfig> {
        self.quantization.as_ref().or(self.quantization_config.as_ref())
    }

    /// Detect the model type from config.
    pub fn model_type(&self) -> ModelType {
        ModelType::from_config_str(&self.tts_model_type)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
    #[serde(default = "default_mode")]
    pub mode: String,
}

fn default_group_size() -> i32 { 64 }
fn default_bits() -> i32 { 8 }
fn default_mode() -> String { "affine".to_string() }

// ============================================================================
// Talker config
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct TalkerConfig {
    pub model_type: String,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub hidden_act: String,
    pub vocab_size: i32,
    pub text_hidden_size: i32,
    pub text_vocab_size: i32,
    pub num_code_groups: i32,
    pub max_position_embeddings: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub position_id_per_seconds: i32,

    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,

    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_pad_id: u32,
    pub codec_think_id: u32,
    pub codec_nothink_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,

    #[serde(default)]
    pub codec_language_id: HashMap<String, u32>,
    #[serde(default)]
    pub spk_id: HashMap<String, u32>,

    pub code_predictor_config: CodePredictorConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScalingConfig {
    #[serde(default)]
    pub interleaved: bool,
    #[serde(default)]
    pub mrope_section: Vec<i32>,
    #[serde(default)]
    pub rope_type: String,
}

// ============================================================================
// Code predictor config
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct CodePredictorConfig {
    pub model_type: String,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub vocab_size: i32,
    pub num_code_groups: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: i32,
}

impl CodePredictorConfig {
    pub fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }
}

// ============================================================================
// Speech tokenizer config
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct SpeechTokenizerConfig {
    pub model_type: String,
    pub input_sample_rate: u32,
    pub output_sample_rate: u32,
    pub decode_upsample_rate: u32,
    pub encode_downsample_rate: u32,
    pub decoder_config: DecoderConfig,
}

impl SpeechTokenizerConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let file = std::fs::File::open(model_dir.join("speech_tokenizer").join("config.json"))?;
        Ok(serde_json::from_reader(file)?)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub latent_dim: i32,
    pub codebook_dim: i32,
    pub codebook_size: i32,
    pub decoder_dim: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub num_hidden_layers: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: i32,
    pub max_position_embeddings: i32,
    pub num_quantizers: i32,
    pub num_semantic_quantizers: i32,
    pub semantic_codebook_size: i32,
    pub vector_quantization_hidden_dimension: i32,
    pub upsample_rates: Vec<i32>,
    pub upsampling_ratios: Vec<i32>,
    #[serde(default = "default_layer_scale")]
    pub layer_scale_initial_scale: f32,
}

fn default_layer_scale() -> f32 { 0.01 }

// ============================================================================
// Generation config
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    #[serde(default = "default_true")]
    pub do_sample: bool,
    #[serde(default = "default_temp")]
    pub temperature: f32,
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_rep_penalty")]
    pub repetition_penalty: f32,
    #[serde(default = "default_max_tokens")]
    pub max_new_tokens: i32,

    /// Speed factor: > 1.0 = faster speech, < 1.0 = slower speech.
    /// Controls how fast text tokens are fed to the model during generation.
    /// Default 1.0 = one text token per codec frame (natural speed).
    #[serde(default = "default_speed")]
    pub speed_factor: f32,

    #[serde(default = "default_true")]
    pub subtalker_dosample: bool,
    #[serde(default = "default_temp")]
    pub subtalker_temperature: f32,
    #[serde(default = "default_top_k")]
    pub subtalker_top_k: i32,
    #[serde(default = "default_top_p")]
    pub subtalker_top_p: f32,
}

impl GenerationConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("generation_config.json");
        if path.exists() {
            let file = std::fs::File::open(path)?;
            Ok(serde_json::from_reader(file)?)
        } else {
            Ok(Self::default())
        }
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            do_sample: true,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
            max_new_tokens: 8192,
            speed_factor: 1.0,
            subtalker_dosample: true,
            subtalker_temperature: 0.9,
            subtalker_top_k: 50,
            subtalker_top_p: 1.0,
        }
    }
}

fn default_true() -> bool { true }
fn default_temp() -> f32 { 0.9 }
fn default_top_k() -> i32 { 50 }
fn default_top_p() -> f32 { 1.0 }
fn default_rep_penalty() -> f32 { 1.05 }
fn default_max_tokens() -> i32 { 8192 }
fn default_speed() -> f32 { 1.0 }
