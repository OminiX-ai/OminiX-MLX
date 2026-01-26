//! # Step-Audio 2 MLX
//!
//! Step-Audio 2 mini implementation in MLX for Apple Silicon.
//!
//! Step-Audio 2 is an end-to-end multimodal large language model for
//! bidirectional audio understanding and generation.
//!
//! ## Features
//!
//! - **ASR**: Automatic Speech Recognition (speech → text)
//! - **TTS**: Text-to-Speech synthesis (text → speech)
//! - **S2TT**: Speech-to-Text Translation
//! - **S2ST**: Speech-to-Speech Translation
//! - **Think Mode**: Extended reasoning before response
//! - **Voice Cloning**: Clone voice from reference audio
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use step_audio2_mlx::StepAudio2;
//!
//! // Load model
//! let mut model = StepAudio2::load("path/to/Step-Audio-2-mini")?;
//!
//! // ASR: Speech to text
//! let text = model.transcribe("audio.wav")?;
//! println!("Transcription: {}", text);
//!
//! // TTS: Text to speech (requires tts feature)
//! #[cfg(feature = "tts")]
//! {
//!     let audio = model.synthesize("Hello, world!")?;
//!     model.save_audio(&audio, "output.wav")?;
//! }
//! ```
//!
//! ## Architecture
//!
//! ```text
//! Audio (16kHz)
//!     → Mel Spectrogram (128 bins)
//!     → Whisper-style Encoder (32 layers, 1280 dim)
//!     → Adaptor (Conv1d + Linear → 3584 dim)
//!     → Qwen2.5-7B LLM (28 layers)
//!     → Text tokens + Audio tokens
//!     → [TTS: S3Tokenizer → Flow Decoder → HiFi-GAN]
//!     → Audio output (24kHz)
//! ```
//!
//! ## Model Variants
//!
//! | Variant | Size | Use Case |
//! |---------|------|----------|
//! | mini-Base | 8B | Fine-tuning foundation |
//! | mini | 8B | Production inference |
//! | mini-Think | 8B | Extended reasoning |

// Re-export core types
pub use mlx_rs_core::{KVCache, KeyValueCache};

pub mod config;
pub mod error;

// Phase 1: ASR
pub mod audio;
pub mod encoder;
pub mod adaptor;
pub mod llm;

// Phase 2: Think Mode
pub mod think;

// Phase 3: TTS (optional)
#[cfg(feature = "tts")]
pub mod tts;

// Phase 4: Integration
pub mod model;
pub mod pipeline;
pub mod tools;

// Re-export main types
pub use config::{StepAudio2Config, EncoderConfig, LLMConfig};
pub use error::{Error, Result};
pub use model::StepAudio2;
pub use pipeline::{StepAudio2Pipeline, PipelineConfig, SamplingConfig, ChatResponse};
pub use think::{ThinkConfig, ThinkOutput, ThinkState, ThinkModeHandler};
pub use tools::{Tool, ToolManager, ToolCall, ToolResult, WebSearchTool, CalculatorTool};

// TTS exports (feature-gated)
#[cfg(feature = "tts")]
pub use tts::{
    TTSDecoder, TTSDecoderConfig,
    HiFiGAN, HiFiGANConfig,
    FlowDecoder, FlowDecoderConfig,
    S3Tokenizer, S3TokenizerConfig,
    AudioTokenExtractor,
};

/// Model variant type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVariant {
    /// Base model (pre-training only)
    Base,
    /// Standard model (pre-training + SFT + RL)
    Standard,
    /// Think model (+ reasoning RL)
    Think,
}

impl Default for ModelVariant {
    fn default() -> Self {
        Self::Standard
    }
}

/// Supported languages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Chinese,
    English,
    Japanese,
    Cantonese,
    Arabic,
}

impl Language {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Chinese => "zh",
            Self::English => "en",
            Self::Japanese => "ja",
            Self::Cantonese => "yue",
            Self::Arabic => "ar",
        }
    }
}
