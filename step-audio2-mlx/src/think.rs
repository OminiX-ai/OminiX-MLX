//! Think mode handler
//!
//! Implements extended reasoning with `<think>...</think>` tags.
//!
//! Think mode allows the model to reason before responding:
//! 1. Generate until `</think>`
//! 2. Extract thinking content
//! 3. Continue generating response (text + audio)
//!
//! Step-Audio 2 mini-Think uses this for complex reasoning tasks.

use crate::config::tokens;

/// Think mode configuration
#[derive(Debug, Clone)]
pub struct ThinkConfig {
    /// Whether think mode is enabled
    pub enabled: bool,
    /// Think start tag
    pub think_start: String,
    /// Think end tag
    pub think_end: String,
    /// Maximum tokens for thinking phase
    pub max_think_tokens: usize,
    /// Maximum tokens for response phase
    pub max_response_tokens: usize,
    /// Whether to include thinking in output
    pub include_thinking: bool,
}

impl Default for ThinkConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            think_start: "<think>".to_string(),
            think_end: "</think>".to_string(),
            max_think_tokens: 2048,
            max_response_tokens: 512,
            include_thinking: true,
        }
    }
}

impl ThinkConfig {
    /// Create a config with think mode disabled
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create a config with custom token limits
    pub fn with_limits(max_think_tokens: usize, max_response_tokens: usize) -> Self {
        Self {
            max_think_tokens,
            max_response_tokens,
            ..Default::default()
        }
    }
}

/// Output from think mode generation
#[derive(Debug, Clone)]
pub struct ThinkOutput {
    /// Content inside `<think>...</think>` tags
    pub thinking: Option<String>,
    /// Text response after thinking
    pub response_text: String,
    /// Audio tokens (if TTS enabled)
    pub audio_tokens: Option<Vec<i32>>,
    /// Total tokens generated
    pub total_tokens: usize,
    /// Tokens used for thinking
    pub think_tokens: usize,
    /// Tokens used for response
    pub response_tokens: usize,
}

impl ThinkOutput {
    /// Create a new think output (no thinking)
    pub fn text_only(response_text: String, tokens: usize) -> Self {
        Self {
            thinking: None,
            response_text,
            audio_tokens: None,
            total_tokens: tokens,
            think_tokens: 0,
            response_tokens: tokens,
        }
    }

    /// Create a new think output with thinking
    pub fn with_thinking(
        thinking: String,
        response_text: String,
        think_tokens: usize,
        response_tokens: usize,
    ) -> Self {
        Self {
            thinking: Some(thinking),
            response_text,
            audio_tokens: None,
            total_tokens: think_tokens + response_tokens,
            think_tokens,
            response_tokens,
        }
    }
}

/// Think tag parser state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkState {
    /// Before any think tags
    Initial,
    /// Inside `<think>` tag, accumulating thinking content
    Thinking,
    /// After `</think>` tag, in response phase
    Responding,
    /// Generation complete
    Done,
}

/// Think mode handler for managing think tag detection and parsing
#[derive(Debug, Clone)]
pub struct ThinkModeHandler {
    /// Configuration
    config: ThinkConfig,
    /// Current state
    state: ThinkState,
    /// Accumulated thinking tokens
    think_tokens: Vec<i32>,
    /// Accumulated response tokens
    response_tokens: Vec<i32>,
    /// Buffer for detecting tags (accumulated text)
    text_buffer: String,
    /// Whether we've seen the start tag
    seen_start: bool,
    /// Whether we've seen the end tag
    seen_end: bool,
}

impl ThinkModeHandler {
    /// Create a new think mode handler
    pub fn new(config: ThinkConfig) -> Self {
        let initial_state = if config.enabled {
            ThinkState::Initial
        } else {
            ThinkState::Responding
        };

        Self {
            config,
            state: initial_state,
            think_tokens: Vec::new(),
            response_tokens: Vec::new(),
            text_buffer: String::new(),
            seen_start: false,
            seen_end: false,
        }
    }

    /// Reset handler state for new generation
    pub fn reset(&mut self) {
        self.state = if self.config.enabled {
            ThinkState::Initial
        } else {
            ThinkState::Responding
        };
        self.think_tokens.clear();
        self.response_tokens.clear();
        self.text_buffer.clear();
        self.seen_start = false;
        self.seen_end = false;
    }

    /// Get current state
    pub fn state(&self) -> ThinkState {
        self.state
    }

    /// Check if generation should stop
    pub fn should_stop(&self, token_id: i32) -> bool {
        // Stop on EOS or IM_END
        if token_id == tokens::EOS_TOKEN || token_id == tokens::IM_END_TOKEN {
            return true;
        }

        // Check token limits
        match self.state {
            ThinkState::Thinking => {
                self.think_tokens.len() >= self.config.max_think_tokens
            }
            ThinkState::Responding => {
                self.response_tokens.len() >= self.config.max_response_tokens
            }
            ThinkState::Done => true,
            _ => false,
        }
    }

    /// Process a generated token
    ///
    /// Returns true if the token was consumed (is part of a tag)
    pub fn process_token(&mut self, token_id: i32, token_text: &str) -> bool {
        // Update text buffer for tag detection
        self.text_buffer.push_str(token_text);

        match self.state {
            ThinkState::Initial => {
                // Looking for <think> tag
                if self.text_buffer.contains(&self.config.think_start) {
                    self.seen_start = true;
                    self.state = ThinkState::Thinking;
                    // Clear buffer after finding start tag
                    if let Some(pos) = self.text_buffer.find(&self.config.think_start) {
                        let after = pos + self.config.think_start.len();
                        self.text_buffer = self.text_buffer[after..].to_string();
                    }
                    return true; // Tag token consumed
                }
                // If we get too many tokens without finding start tag,
                // assume no think mode and switch to responding
                if self.text_buffer.len() > 50 {
                    self.state = ThinkState::Responding;
                    self.response_tokens.push(token_id);
                }
                false
            }
            ThinkState::Thinking => {
                // Looking for </think> tag while accumulating thinking content
                if self.text_buffer.contains(&self.config.think_end) {
                    self.seen_end = true;
                    self.state = ThinkState::Responding;
                    // Clear buffer after finding end tag
                    if let Some(pos) = self.text_buffer.find(&self.config.think_end) {
                        // Content before end tag goes to think tokens
                        self.text_buffer = self.text_buffer[pos + self.config.think_end.len()..].to_string();
                    }
                    return true; // Tag token consumed
                }
                self.think_tokens.push(token_id);
                false
            }
            ThinkState::Responding => {
                // Accumulate response tokens
                self.response_tokens.push(token_id);

                // Check for audio tokens (switch to audio generation mode)
                if tokens::is_audio_token(token_id) {
                    // Continue accumulating, will be processed later
                }
                false
            }
            ThinkState::Done => {
                // Already done, consume token
                true
            }
        }
    }

    /// Mark generation as complete
    pub fn finish(&mut self) {
        self.state = ThinkState::Done;
    }

    /// Get thinking tokens
    pub fn thinking_tokens(&self) -> &[i32] {
        &self.think_tokens
    }

    /// Get response tokens
    pub fn response_tokens(&self) -> &[i32] {
        &self.response_tokens
    }

    /// Build the final output
    pub fn build_output(&self, decode_fn: impl Fn(&[i32]) -> String) -> ThinkOutput {
        let thinking = if self.config.include_thinking && !self.think_tokens.is_empty() {
            Some(decode_fn(&self.think_tokens))
        } else {
            None
        };

        // Separate text tokens from audio tokens in response
        let (text_tokens, audio_tokens): (Vec<i32>, Vec<i32>) = self
            .response_tokens
            .iter()
            .partition(|&&t| !tokens::is_audio_token(t));

        let response_text = decode_fn(&text_tokens);
        let audio_tokens = if audio_tokens.is_empty() {
            None
        } else {
            Some(audio_tokens)
        };

        ThinkOutput {
            thinking,
            response_text,
            audio_tokens,
            total_tokens: self.think_tokens.len() + self.response_tokens.len(),
            think_tokens: self.think_tokens.len(),
            response_tokens: self.response_tokens.len(),
        }
    }

    /// Check if we're in thinking phase
    pub fn is_thinking(&self) -> bool {
        self.state == ThinkState::Thinking
    }

    /// Check if we're in response phase
    pub fn is_responding(&self) -> bool {
        self.state == ThinkState::Responding
    }

    /// Get the appropriate max tokens for current phase
    pub fn current_max_tokens(&self) -> usize {
        match self.state {
            ThinkState::Initial => self.config.max_think_tokens,
            ThinkState::Thinking => self.config.max_think_tokens,
            ThinkState::Responding => self.config.max_response_tokens,
            ThinkState::Done => 0,
        }
    }
}

/// Parse think tags from a complete text string
///
/// Returns (thinking_content, response_content)
pub fn parse_think_tags(text: &str, config: &ThinkConfig) -> (Option<String>, String) {
    if !config.enabled {
        return (None, text.to_string());
    }

    // Find think tags
    let start_pos = text.find(&config.think_start);
    let end_pos = text.find(&config.think_end);

    match (start_pos, end_pos) {
        (Some(start), Some(end)) if start < end => {
            // Extract thinking content
            let think_start = start + config.think_start.len();
            let thinking = text[think_start..end].trim().to_string();

            // Extract response (everything after </think>)
            let response_start = end + config.think_end.len();
            let response = text[response_start..].trim().to_string();

            (Some(thinking), response)
        }
        _ => {
            // No valid think tags, treat everything as response
            (None, text.to_string())
        }
    }
}

/// Format a prompt for think mode
///
/// Adds the think start tag if think mode is enabled
pub fn format_think_prompt(prompt: &str, config: &ThinkConfig) -> String {
    if config.enabled {
        format!("{}{}", prompt, config.think_start)
    } else {
        prompt.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_think_config_default() {
        let config = ThinkConfig::default();
        assert!(config.enabled);
        assert_eq!(config.think_start, "<think>");
        assert_eq!(config.think_end, "</think>");
    }

    #[test]
    fn test_think_config_disabled() {
        let config = ThinkConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_parse_think_tags() {
        let config = ThinkConfig::default();

        // With think tags
        let text = "<think>I need to analyze this</think>The answer is 42";
        let (thinking, response) = parse_think_tags(text, &config);
        assert_eq!(thinking, Some("I need to analyze this".to_string()));
        assert_eq!(response, "The answer is 42");

        // Without think tags
        let text = "The answer is 42";
        let (thinking, response) = parse_think_tags(text, &config);
        assert_eq!(thinking, None);
        assert_eq!(response, "The answer is 42");
    }

    #[test]
    fn test_think_mode_handler_states() {
        let config = ThinkConfig::default();
        let mut handler = ThinkModeHandler::new(config);

        assert_eq!(handler.state(), ThinkState::Initial);

        // Process <think> tag
        handler.process_token(1, "<think>");
        assert_eq!(handler.state(), ThinkState::Thinking);

        // Process thinking content
        handler.process_token(2, "reasoning");
        assert_eq!(handler.state(), ThinkState::Thinking);
        assert_eq!(handler.thinking_tokens().len(), 1);

        // Process </think> tag
        handler.process_token(3, "</think>");
        assert_eq!(handler.state(), ThinkState::Responding);

        // Process response
        handler.process_token(4, "answer");
        assert_eq!(handler.response_tokens().len(), 1);
    }

    #[test]
    fn test_think_mode_disabled() {
        let config = ThinkConfig::disabled();
        let mut handler = ThinkModeHandler::new(config);

        // Should start in responding state when disabled
        assert_eq!(handler.state(), ThinkState::Responding);

        // All tokens go to response
        handler.process_token(1, "hello");
        assert_eq!(handler.response_tokens().len(), 1);
        assert_eq!(handler.thinking_tokens().len(), 0);
    }

    #[test]
    fn test_format_think_prompt() {
        let config = ThinkConfig::default();

        let prompt = "What is 2+2?";
        let formatted = format_think_prompt(prompt, &config);
        assert_eq!(formatted, "What is 2+2?<think>");

        let config_disabled = ThinkConfig::disabled();
        let formatted = format_think_prompt(prompt, &config_disabled);
        assert_eq!(formatted, "What is 2+2?");
    }

    #[test]
    fn test_think_output() {
        let output = ThinkOutput::with_thinking(
            "thinking...".to_string(),
            "response".to_string(),
            10,
            5,
        );

        assert_eq!(output.thinking, Some("thinking...".to_string()));
        assert_eq!(output.response_text, "response");
        assert_eq!(output.total_tokens, 15);
        assert_eq!(output.think_tokens, 10);
        assert_eq!(output.response_tokens, 5);
    }
}
