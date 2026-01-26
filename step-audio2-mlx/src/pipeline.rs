//! High-level inference pipeline for Step-Audio 2
//!
//! Provides a unified API for all Step-Audio 2 capabilities:
//! - ASR (Automatic Speech Recognition)
//! - TTS (Text-to-Speech)
//! - S2ST (Speech-to-Speech Translation)
//! - Think mode (extended reasoning)
//! - Tool calling (web search, calculator)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use step_audio2_mlx::{StepAudio2Pipeline, PipelineConfig};
//!
//! // Load pipeline
//! let mut pipeline = StepAudio2Pipeline::load("path/to/model", PipelineConfig::default())?;
//!
//! // ASR: Speech to text
//! let text = pipeline.transcribe_file("audio.wav")?;
//!
//! // TTS: Text to speech (requires tts feature)
//! #[cfg(feature = "tts")]
//! let audio = pipeline.synthesize("Hello, world!")?;
//!
//! // Think mode
//! let response = pipeline.think_file("question.wav")?;
//! println!("Thinking: {}", response.thinking.unwrap_or_default());
//! println!("Response: {}", response.response_text);
//! ```

use std::path::Path;

use crate::error::{Error, Result};
use crate::model::StepAudio2;
use crate::think::{ThinkConfig, ThinkOutput};
use crate::tools::{ToolManager, ToolCall, ToolResult};

#[cfg(feature = "tts")]
use crate::tts::{TTSDecoder, TTSDecoderConfig, extract_audio_tokens};

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Enable TTS output
    pub enable_tts: bool,
    /// Enable think mode
    pub enable_think: bool,
    /// Enable tool calling
    pub enable_tools: bool,
    /// Think mode configuration
    pub think_config: ThinkConfig,
    /// Sampling configuration
    pub sampling: SamplingConfig,
    /// Output sample rate for TTS (default: 24000)
    pub output_sample_rate: u32,
    /// Maximum tool call iterations
    pub max_tool_iterations: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enable_tts: false,
            enable_think: false,
            enable_tools: false,
            think_config: ThinkConfig::disabled(),
            sampling: SamplingConfig::default(),
            output_sample_rate: 24000,
            max_tool_iterations: 5,
        }
    }
}

impl PipelineConfig {
    /// Create a config for ASR-only mode
    pub fn asr_only() -> Self {
        Self::default()
    }

    /// Create a config for think mode (Step-Audio 2 mini-Think)
    pub fn with_think() -> Self {
        Self {
            enable_think: true,
            think_config: ThinkConfig::default(),
            ..Default::default()
        }
    }

    /// Create a config with tool calling enabled
    pub fn with_tools() -> Self {
        Self {
            enable_tools: true,
            ..Default::default()
        }
    }

    /// Create a config with TTS enabled
    #[cfg(feature = "tts")]
    pub fn with_tts() -> Self {
        Self {
            enable_tts: true,
            ..Default::default()
        }
    }

    /// Create a full-featured config (think + TTS + tools)
    #[cfg(feature = "tts")]
    pub fn full() -> Self {
        Self {
            enable_tts: true,
            enable_think: true,
            enable_tools: true,
            think_config: ThinkConfig::default(),
            sampling: SamplingConfig::default(),
            output_sample_rate: 24000,
            max_tool_iterations: 5,
        }
    }

    /// Enable/disable think mode
    pub fn think(mut self, enabled: bool) -> Self {
        self.enable_think = enabled;
        if enabled {
            self.think_config = ThinkConfig::default();
        } else {
            self.think_config = ThinkConfig::disabled();
        }
        self
    }

    /// Enable/disable tool calling
    pub fn tools(mut self, enabled: bool) -> Self {
        self.enable_tools = enabled;
        self
    }

    /// Set sampling configuration
    pub fn with_sampling(mut self, sampling: SamplingConfig) -> Self {
        self.sampling = sampling;
        self
    }
}

/// Sampling configuration
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature (0.0 = greedy)
    pub temperature: f32,
    /// Top-k filtering (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus) filtering
    pub top_p: f32,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Repetition penalty
    pub repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            max_tokens: 512,
            repetition_penalty: 1.0,
        }
    }
}

impl SamplingConfig {
    /// Greedy sampling (deterministic)
    pub fn greedy() -> Self {
        Self::default()
    }

    /// Creative sampling (higher temperature)
    pub fn creative() -> Self {
        Self {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.9,
            ..Default::default()
        }
    }

    /// Balanced sampling
    pub fn balanced() -> Self {
        Self {
            temperature: 0.5,
            top_k: 40,
            top_p: 0.95,
            ..Default::default()
        }
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set top-k
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set top-p
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }
}

/// Chat response
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// Text response
    pub text: String,
    /// Audio response (if TTS enabled)
    pub audio: Option<Vec<f32>>,
    /// Audio sample rate
    pub audio_sample_rate: Option<u32>,
    /// Thinking content (if think mode enabled)
    pub thinking: Option<String>,
    /// Tool calls made during response
    pub tool_calls: Vec<ToolCall>,
    /// Tool results
    pub tool_results: Vec<ToolResult>,
    /// Number of tokens generated
    pub tokens_generated: usize,
}

impl ChatResponse {
    /// Create a text-only response
    pub fn text_only(text: String, tokens: usize) -> Self {
        Self {
            text,
            audio: None,
            audio_sample_rate: None,
            thinking: None,
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            tokens_generated: tokens,
        }
    }

    /// Create a response from think output
    pub fn from_think_output(output: ThinkOutput) -> Self {
        Self {
            text: output.response_text,
            audio: None,
            audio_sample_rate: None,
            thinking: output.thinking,
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            tokens_generated: output.total_tokens,
        }
    }

    /// Create a response with audio
    #[cfg(feature = "tts")]
    pub fn with_audio(
        text: String,
        audio: Vec<f32>,
        sample_rate: u32,
        tokens: usize,
    ) -> Self {
        Self {
            text,
            audio: Some(audio),
            audio_sample_rate: Some(sample_rate),
            thinking: None,
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            tokens_generated: tokens,
        }
    }

    /// Check if response has audio
    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }

    /// Check if response has thinking
    pub fn has_thinking(&self) -> bool {
        self.thinking.is_some()
    }

    /// Check if tool calls were made
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

/// Conversation turn
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    /// Role (user/assistant/system)
    pub role: String,
    /// Text content
    pub text: Option<String>,
    /// Audio content (samples)
    pub audio: Option<Vec<f32>>,
    /// Audio sample rate
    pub audio_sample_rate: Option<u32>,
}

impl ConversationTurn {
    /// Create a user turn with text
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            text: Some(text.into()),
            audio: None,
            audio_sample_rate: None,
        }
    }

    /// Create a user turn with audio
    pub fn user_audio(audio: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            role: "user".to_string(),
            text: None,
            audio: Some(audio),
            audio_sample_rate: Some(sample_rate),
        }
    }

    /// Create an assistant turn
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            text: Some(text.into()),
            audio: None,
            audio_sample_rate: None,
        }
    }

    /// Create a system turn
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            text: Some(text.into()),
            audio: None,
            audio_sample_rate: None,
        }
    }
}

/// Conversation context for multi-turn dialogue
#[derive(Debug, Clone, Default)]
pub struct Conversation {
    /// Conversation history
    pub turns: Vec<ConversationTurn>,
    /// System prompt
    pub system_prompt: Option<String>,
}

impl Conversation {
    /// Create a new conversation
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with system prompt
    pub fn with_system_prompt(prompt: impl Into<String>) -> Self {
        Self {
            turns: Vec::new(),
            system_prompt: Some(prompt.into()),
        }
    }

    /// Add a turn
    pub fn add_turn(&mut self, turn: ConversationTurn) {
        self.turns.push(turn);
    }

    /// Add user text
    pub fn add_user_text(&mut self, text: impl Into<String>) {
        self.turns.push(ConversationTurn::user_text(text));
    }

    /// Add user audio
    pub fn add_user_audio(&mut self, audio: Vec<f32>, sample_rate: u32) {
        self.turns.push(ConversationTurn::user_audio(audio, sample_rate));
    }

    /// Add assistant response
    pub fn add_assistant(&mut self, text: impl Into<String>) {
        self.turns.push(ConversationTurn::assistant(text));
    }

    /// Get number of turns
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Clear conversation history
    pub fn clear(&mut self) {
        self.turns.clear();
    }
}

/// Step-Audio 2 pipeline
pub struct StepAudio2Pipeline {
    /// Underlying model
    pub model: StepAudio2,
    /// Pipeline configuration
    pub config: PipelineConfig,
    /// Tool manager
    pub tools: ToolManager,
    /// TTS decoder (optional)
    #[cfg(feature = "tts")]
    pub tts: Option<TTSDecoder>,
    /// Conversation context
    pub conversation: Conversation,
}

impl StepAudio2Pipeline {
    /// Create new pipeline
    pub fn new(model: StepAudio2, config: PipelineConfig) -> Self {
        let tools = if config.enable_tools {
            ToolManager::with_defaults()
        } else {
            ToolManager::new()
        };

        Self {
            model,
            config,
            tools,
            #[cfg(feature = "tts")]
            tts: None,
            conversation: Conversation::new(),
        }
    }

    /// Create pipeline with default config
    pub fn with_defaults(model: StepAudio2) -> Self {
        Self::new(model, PipelineConfig::default())
    }

    /// Load pipeline from model directory
    pub fn load(
        model_dir: impl AsRef<Path>,
        config: PipelineConfig,
    ) -> Result<Self> {
        let model = StepAudio2::load(&model_dir)?;

        let mut pipeline = Self::new(model, config);

        // Load TTS decoder if enabled
        #[cfg(feature = "tts")]
        if pipeline.config.enable_tts {
            pipeline.tts = Some(TTSDecoder::load(&model_dir)?);
        }

        Ok(pipeline)
    }

    /// ASR: Speech to text
    pub fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<String> {
        self.model.transcribe_samples(audio, sample_rate)
    }

    /// ASR: Speech to text from file
    pub fn transcribe_file(&mut self, audio_path: impl AsRef<Path>) -> Result<String> {
        self.model.transcribe(audio_path)
    }

    /// TTS: Text to speech
    #[cfg(feature = "tts")]
    pub fn synthesize(&mut self, text: &str) -> Result<Vec<f32>> {
        let tts = self.tts.as_mut()
            .ok_or_else(|| Error::TTS("TTS decoder not loaded".to_string()))?;

        // TODO: Generate audio tokens from text using LLM
        // For now, return empty
        Err(Error::TTS("Text-to-speech not yet implemented".to_string()))
    }

    /// TTS: Synthesize from audio tokens
    #[cfg(feature = "tts")]
    pub fn synthesize_from_tokens(&mut self, audio_tokens: &[i32]) -> Result<Vec<f32>> {
        let tts = self.tts.as_mut()
            .ok_or_else(|| Error::TTS("TTS decoder not loaded".to_string()))?;

        tts.synthesize(audio_tokens, None)
    }

    /// Process audio with think mode
    pub fn think(&mut self, audio: &[f32], sample_rate: u32) -> Result<ThinkOutput> {
        let think_config = if self.config.enable_think {
            self.config.think_config.clone()
        } else {
            ThinkConfig::disabled()
        };

        self.model.think_and_respond_samples(audio, sample_rate, think_config)
    }

    /// Process audio file with think mode
    pub fn think_file(&mut self, audio_path: impl AsRef<Path>) -> Result<ThinkOutput> {
        let think_config = if self.config.enable_think {
            self.config.think_config.clone()
        } else {
            ThinkConfig::disabled()
        };

        self.model.think_and_respond(audio_path, think_config)
    }

    /// Chat with audio input
    pub fn chat_audio(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<ChatResponse> {
        // Add to conversation
        self.conversation.add_user_audio(audio.to_vec(), sample_rate);

        // Process with think mode if enabled
        if self.config.enable_think {
            let output = self.think(audio, sample_rate)?;
            let mut response = ChatResponse::from_think_output(output);

            // Handle tool calls if enabled
            if self.config.enable_tools {
                self.handle_tool_calls(&mut response)?;
            }

            // Synthesize audio if TTS enabled
            #[cfg(feature = "tts")]
            if self.config.enable_tts {
                if let Some(ref tts) = self.tts {
                    // TODO: Get audio tokens from generation and synthesize
                }
            }

            // Add assistant response to conversation
            self.conversation.add_assistant(&response.text);

            Ok(response)
        } else {
            // Simple transcription
            let text = self.transcribe(audio, sample_rate)?;
            let response = ChatResponse::text_only(text.clone(), 0);

            self.conversation.add_assistant(&text);

            Ok(response)
        }
    }

    /// Chat with text input
    pub fn chat_text(&mut self, text: &str) -> Result<ChatResponse> {
        self.conversation.add_user_text(text);

        // TODO: Implement text-only generation
        Err(Error::Inference("Text-only input not yet supported".to_string()))
    }

    /// Chat with audio/text input (unified API)
    pub fn chat(
        &mut self,
        audio: Option<&[f32]>,
        sample_rate: Option<u32>,
        text: Option<&str>,
    ) -> Result<ChatResponse> {
        match (audio, sample_rate, text) {
            (Some(audio), Some(rate), _) => self.chat_audio(audio, rate),
            (None, None, Some(text)) => self.chat_text(text),
            _ => Err(Error::Inference("Must provide either audio or text input".to_string())),
        }
    }

    /// Handle tool calls in response
    fn handle_tool_calls(&mut self, response: &mut ChatResponse) -> Result<()> {
        if !self.config.enable_tools {
            return Ok(());
        }

        // Check for tool calls in the response
        let mut iterations = 0;
        while self.tools.has_tool_call(&response.text) && iterations < self.config.max_tool_iterations {
            // Parse tool calls
            let calls = self.tools.parse_all_tool_calls(&response.text);
            if calls.is_empty() {
                break;
            }

            // Execute tools
            let results = self.tools.execute_all(&calls);

            // Store calls and results
            response.tool_calls.extend(calls);
            response.tool_results.extend(results.clone());

            // TODO: Feed results back to model for continuation
            // For now, just append results to response
            for result in &results {
                response.text.push_str("\n\n");
                response.text.push_str(&result.format_for_model());
            }

            iterations += 1;
        }

        Ok(())
    }

    /// S2ST: Speech-to-speech translation
    #[cfg(feature = "tts")]
    pub fn translate_speech(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        _target_lang: &str,
    ) -> Result<ChatResponse> {
        // For S2ST, we:
        // 1. Transcribe input
        // 2. Generate response with audio tokens
        // 3. Synthesize audio

        let response = self.chat_audio(audio, sample_rate)?;

        // TODO: Ensure TTS output is generated
        Ok(response)
    }

    /// Update pipeline configuration
    pub fn set_config(&mut self, config: PipelineConfig) {
        self.config = config;

        // Update tool manager
        if self.config.enable_tools && self.tools.tool_names().is_empty() {
            self.tools = ToolManager::with_defaults();
        }
    }

    /// Enable/disable think mode
    pub fn set_think_mode(&mut self, enabled: bool) {
        self.config.enable_think = enabled;
        if enabled {
            self.config.think_config = ThinkConfig::default();
        } else {
            self.config.think_config = ThinkConfig::disabled();
        }
    }

    /// Enable/disable tool calling
    pub fn set_tools(&mut self, enabled: bool) {
        self.config.enable_tools = enabled;
        if enabled && self.tools.tool_names().is_empty() {
            self.tools = ToolManager::with_defaults();
        }
    }

    /// Register a custom tool
    pub fn register_tool(&mut self, tool: Box<dyn crate::tools::Tool>) {
        self.tools.register(tool);
    }

    /// Get tool prompt for model
    pub fn get_tool_prompt(&self) -> String {
        self.tools.generate_tool_prompt()
    }

    /// Clear conversation history
    pub fn clear_conversation(&mut self) {
        self.conversation.clear();
    }

    /// Set system prompt
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.conversation.system_prompt = Some(prompt.into());
    }

    /// Get output sample rate
    pub fn output_sample_rate(&self) -> u32 {
        self.config.output_sample_rate
    }
}

/// Save audio to WAV file
pub fn save_audio(
    audio: &[f32],
    sample_rate: u32,
    path: impl AsRef<Path>,
) -> Result<()> {
    crate::audio::save_wav(audio, sample_rate, path)
}

/// Load audio from WAV file
pub fn load_audio(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    crate::audio::load_wav(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!(!config.enable_tts);
        assert!(!config.enable_think);
        assert!(!config.enable_tools);
    }

    #[test]
    fn test_pipeline_config_with_think() {
        let config = PipelineConfig::with_think();
        assert!(config.enable_think);
        assert!(config.think_config.enabled);
    }

    #[test]
    fn test_pipeline_config_builder() {
        let config = PipelineConfig::default()
            .think(true)
            .tools(true)
            .with_sampling(SamplingConfig::creative());

        assert!(config.enable_think);
        assert!(config.enable_tools);
        assert!(config.sampling.temperature > 0.0);
    }

    #[test]
    fn test_sampling_config() {
        let config = SamplingConfig::greedy();
        assert_eq!(config.temperature, 0.0);

        let config = SamplingConfig::creative();
        assert!(config.temperature > 0.5);

        let config = SamplingConfig::balanced();
        assert!(config.temperature > 0.0);
        assert!(config.temperature < 1.0);
    }

    #[test]
    fn test_chat_response() {
        let response = ChatResponse::text_only("Hello".to_string(), 5);
        assert_eq!(response.text, "Hello");
        assert!(!response.has_audio());
        assert!(!response.has_thinking());
        assert!(!response.has_tool_calls());
    }

    #[test]
    fn test_conversation() {
        let mut conv = Conversation::new();
        assert!(conv.is_empty());

        conv.add_user_text("Hello");
        conv.add_assistant("Hi there!");

        assert_eq!(conv.len(), 2);
        assert_eq!(conv.turns[0].role, "user");
        assert_eq!(conv.turns[1].role, "assistant");
    }

    #[test]
    fn test_conversation_with_system() {
        let conv = Conversation::with_system_prompt("You are a helpful assistant.");
        assert!(conv.system_prompt.is_some());
    }

    #[test]
    fn test_conversation_turn() {
        let turn = ConversationTurn::user_text("Hello");
        assert_eq!(turn.role, "user");
        assert_eq!(turn.text, Some("Hello".to_string()));
        assert!(turn.audio.is_none());

        let turn = ConversationTurn::user_audio(vec![0.0; 100], 16000);
        assert_eq!(turn.role, "user");
        assert!(turn.text.is_none());
        assert!(turn.audio.is_some());
    }
}
