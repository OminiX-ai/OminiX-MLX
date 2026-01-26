//! Configuration types for Step-Audio 2

use serde::{Deserialize, Serialize};

/// Main model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepAudio2Config {
    /// Audio encoder configuration
    pub encoder: EncoderConfig,
    /// Audio-to-LLM adaptor configuration
    pub adaptor: AdaptorConfig,
    /// LLM configuration
    pub llm: LLMConfig,
    /// Audio processing configuration
    #[serde(default)]
    pub audio: AudioConfig,
}

impl Default for StepAudio2Config {
    fn default() -> Self {
        Self {
            encoder: EncoderConfig::default(),
            adaptor: AdaptorConfig::default(),
            llm: LLMConfig::default(),
            audio: AudioConfig::default(),
        }
    }
}

/// Whisper-style audio encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Number of mel filterbank channels
    #[serde(default = "default_n_mels")]
    pub n_mels: i32,
    /// Maximum context length
    #[serde(default = "default_n_ctx")]
    pub n_ctx: i32,
    /// Hidden state dimension
    #[serde(default = "default_n_state")]
    pub n_state: i32,
    /// Number of attention heads
    #[serde(default = "default_n_head")]
    pub n_head: i32,
    /// Number of transformer layers
    #[serde(default = "default_n_layer")]
    pub n_layer: i32,
}

fn default_n_mels() -> i32 { 128 }
fn default_n_ctx() -> i32 { 1500 }
fn default_n_state() -> i32 { 1280 }
fn default_n_head() -> i32 { 20 }
fn default_n_layer() -> i32 { 32 }

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            n_mels: default_n_mels(),
            n_ctx: default_n_ctx(),
            n_state: default_n_state(),
            n_head: default_n_head(),
            n_layer: default_n_layer(),
        }
    }
}

/// Audio-to-LLM adaptor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptorConfig {
    /// Input dimension from encoder
    #[serde(default = "default_encoder_dim")]
    pub encoder_dim: i32,
    /// Hidden dimension
    #[serde(default = "default_hidden_dim")]
    pub hidden_dim: i32,
    /// Output dimension (LLM hidden size)
    #[serde(default = "default_llm_dim")]
    pub llm_dim: i32,
    /// Convolution kernel size
    #[serde(default = "default_kernel_size")]
    pub kernel_size: i32,
    /// Convolution stride (downsampling factor)
    #[serde(default = "default_stride")]
    pub stride: i32,
}

fn default_encoder_dim() -> i32 { 1280 }
fn default_hidden_dim() -> i32 { 2048 }
fn default_llm_dim() -> i32 { 3584 }
fn default_kernel_size() -> i32 { 3 }
fn default_stride() -> i32 { 2 }

impl Default for AdaptorConfig {
    fn default() -> Self {
        Self {
            encoder_dim: default_encoder_dim(),
            hidden_dim: default_hidden_dim(),
            llm_dim: default_llm_dim(),
            kernel_size: default_kernel_size(),
            stride: default_stride(),
        }
    }
}

/// Qwen2.5-7B LLM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    /// Hidden size
    #[serde(default = "default_hidden_size")]
    pub hidden_size: i32,
    /// Intermediate size in MLP
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: i32,
    /// Number of hidden layers
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: i32,
    /// Number of attention heads
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    /// Number of key-value heads (for GQA)
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: i32,
    /// Vocabulary size (text + audio tokens)
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    /// Maximum position embeddings
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    /// RoPE theta
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// RMS norm epsilon
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// Whether to tie word embeddings
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}

fn default_hidden_size() -> i32 { 3584 }
fn default_intermediate_size() -> i32 { 18944 }
fn default_num_hidden_layers() -> i32 { 28 }
fn default_num_attention_heads() -> i32 { 28 }
fn default_num_key_value_heads() -> i32 { 4 }
fn default_vocab_size() -> i32 { 158720 }
fn default_max_position_embeddings() -> i32 { 16384 }
fn default_rope_theta() -> f32 { 1000000.0 }
fn default_rms_norm_eps() -> f32 { 1e-6 }
fn default_tie_word_embeddings() -> bool { false }

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            num_key_value_heads: default_num_key_value_heads(),
            vocab_size: default_vocab_size(),
            max_position_embeddings: default_max_position_embeddings(),
            rope_theta: default_rope_theta(),
            rms_norm_eps: default_rms_norm_eps(),
            tie_word_embeddings: default_tie_word_embeddings(),
        }
    }
}

/// Audio processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Input sample rate
    #[serde(default = "default_sample_rate")]
    pub sample_rate: i32,
    /// FFT size
    #[serde(default = "default_n_fft")]
    pub n_fft: i32,
    /// Hop length
    #[serde(default = "default_hop_length")]
    pub hop_length: i32,
    /// Number of mel bins
    #[serde(default = "default_audio_n_mels")]
    pub n_mels: i32,
    /// Minimum frequency for mel filterbank
    #[serde(default = "default_fmin")]
    pub fmin: f32,
    /// Maximum frequency for mel filterbank
    #[serde(default = "default_fmax")]
    pub fmax: Option<f32>,
}

fn default_sample_rate() -> i32 { 16000 }
fn default_n_fft() -> i32 { 400 }
fn default_hop_length() -> i32 { 160 }
fn default_audio_n_mels() -> i32 { 128 }
fn default_fmin() -> f32 { 0.0 }
fn default_fmax() -> Option<f32> { Some(8000.0) }

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: default_sample_rate(),
            n_fft: default_n_fft(),
            hop_length: default_hop_length(),
            n_mels: default_audio_n_mels(),
            fmin: default_fmin(),
            fmax: default_fmax(),
        }
    }
}

/// Token ID constants
pub mod tokens {
    /// Start of audio token range
    pub const AUDIO_TOKEN_START: i32 = 151696;
    /// End of audio token range
    pub const AUDIO_TOKEN_END: i32 = 158256;
    /// Audio codebook size
    pub const AUDIO_CODEBOOK_SIZE: i32 = 6561;
    /// End of text token (<|endoftext|>)
    pub const EOS_TOKEN: i32 = 151643;
    /// End of turn token (<|EOT|>) - the actual stopping token
    pub const EOT_TOKEN: i32 = 151665;
    /// IM start token
    pub const IM_START_TOKEN: i32 = 151644;
    /// IM end token
    pub const IM_END_TOKEN: i32 = 151645;
    /// Audio start placeholder (where audio features are inserted)
    pub const AUDIO_START_TOKEN: i32 = 151688;
    /// Audio end token
    pub const AUDIO_END_TOKEN: i32 = 151689;
    /// Audio patch placeholder (legacy alias)
    pub const AUDIO_PATCH_TOKEN: i32 = 151688;

    /// Check if token is an audio token
    pub fn is_audio_token(token_id: i32) -> bool {
        token_id >= AUDIO_TOKEN_START && token_id <= AUDIO_TOKEN_END
    }

    /// Convert audio token ID to codebook index
    pub fn token_to_code(token_id: i32) -> i32 {
        token_id - AUDIO_TOKEN_START
    }

    /// Convert codebook index to audio token ID
    pub fn code_to_token(code: i32) -> i32 {
        code + AUDIO_TOKEN_START
    }
}
