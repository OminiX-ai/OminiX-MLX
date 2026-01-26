//! Audio token handling utilities
//!
//! Step-Audio 2 uses a specialized vocabulary where:
//! - Text tokens: 0 - 151695
//! - Audio tokens: 151696 - 158256 (6561 codes)
//!
//! The audio tokens correspond to a CosyVoice2 codebook with 6561 entries.
//! These tokens are interleaved with text during generation.

use crate::config::tokens;

/// Extract audio token codes from a sequence of tokens
///
/// Filters to only audio tokens and converts to codebook indices.
///
/// # Arguments
/// * `tokens` - Raw token IDs from the LLM
///
/// # Returns
/// Codebook indices (0-6560) for audio tokens only
pub fn extract_audio_tokens(token_ids: &[i32]) -> Vec<i32> {
    token_ids
        .iter()
        .filter(|&&t| tokens::is_audio_token(t))
        .map(|&t| tokens::token_to_code(t))
        .collect()
}

/// Extract text tokens from a mixed sequence
pub fn extract_text_tokens(token_ids: &[i32]) -> Vec<i32> {
    token_ids
        .iter()
        .filter(|&&t| !tokens::is_audio_token(t))
        .copied()
        .collect()
}

/// Separate text and audio tokens from a mixed sequence
///
/// # Returns
/// (text_tokens, audio_codes)
pub fn separate_tokens(token_ids: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let mut text_tokens = Vec::new();
    let mut audio_codes = Vec::new();

    for &token_id in token_ids {
        if tokens::is_audio_token(token_id) {
            audio_codes.push(tokens::token_to_code(token_id));
        } else {
            text_tokens.push(token_id);
        }
    }

    (text_tokens, audio_codes)
}

/// Convert codebook indices back to audio token IDs
pub fn codes_to_tokens(codes: &[i32]) -> Vec<i32> {
    codes.iter().map(|&c| tokens::code_to_token(c)).collect()
}

/// Audio token extractor with streaming support
#[derive(Debug, Clone)]
pub struct AudioTokenExtractor {
    /// Accumulated audio codes
    codes: Vec<i32>,
    /// Whether we've seen any audio tokens
    has_audio: bool,
    /// Minimum codes before yielding (for batching)
    min_batch_size: usize,
}

impl Default for AudioTokenExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioTokenExtractor {
    /// Create a new extractor
    pub fn new() -> Self {
        Self {
            codes: Vec::new(),
            has_audio: false,
            min_batch_size: 1,
        }
    }

    /// Create extractor with minimum batch size
    pub fn with_batch_size(min_batch_size: usize) -> Self {
        Self {
            codes: Vec::new(),
            has_audio: false,
            min_batch_size,
        }
    }

    /// Process a token and return accumulated codes if batch is ready
    ///
    /// # Arguments
    /// * `token_id` - Token ID from LLM
    ///
    /// # Returns
    /// Some(codes) if a batch is ready, None otherwise
    pub fn process(&mut self, token_id: i32) -> Option<Vec<i32>> {
        if tokens::is_audio_token(token_id) {
            self.has_audio = true;
            self.codes.push(tokens::token_to_code(token_id));

            if self.codes.len() >= self.min_batch_size {
                let batch = std::mem::take(&mut self.codes);
                return Some(batch);
            }
        }
        None
    }

    /// Process multiple tokens at once
    pub fn process_batch(&mut self, token_ids: &[i32]) -> Option<Vec<i32>> {
        for &token_id in token_ids {
            if tokens::is_audio_token(token_id) {
                self.has_audio = true;
                self.codes.push(tokens::token_to_code(token_id));
            }
        }

        if self.codes.len() >= self.min_batch_size {
            let batch = std::mem::take(&mut self.codes);
            return Some(batch);
        }
        None
    }

    /// Flush remaining codes
    pub fn flush(&mut self) -> Option<Vec<i32>> {
        if self.codes.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.codes))
        }
    }

    /// Check if any audio tokens have been seen
    pub fn has_audio(&self) -> bool {
        self.has_audio
    }

    /// Get count of pending codes
    pub fn pending_count(&self) -> usize {
        self.codes.len()
    }

    /// Reset the extractor
    pub fn reset(&mut self) {
        self.codes.clear();
        self.has_audio = false;
    }
}

/// Validate audio codes are within valid range
pub fn validate_codes(codes: &[i32]) -> bool {
    codes
        .iter()
        .all(|&c| c >= 0 && c < tokens::AUDIO_CODEBOOK_SIZE)
}

/// Statistics about audio content in a token sequence
#[derive(Debug, Clone, Default)]
pub struct AudioTokenStats {
    /// Total number of tokens
    pub total_tokens: usize,
    /// Number of audio tokens
    pub audio_tokens: usize,
    /// Number of text tokens
    pub text_tokens: usize,
    /// Audio token ratio (0.0 - 1.0)
    pub audio_ratio: f32,
    /// Estimated audio duration in seconds (25Hz frame rate)
    pub estimated_duration_secs: f32,
}

impl AudioTokenStats {
    /// Compute statistics from a token sequence
    pub fn from_tokens(token_ids: &[i32]) -> Self {
        let total_tokens = token_ids.len();
        let audio_tokens = token_ids
            .iter()
            .filter(|&&t| tokens::is_audio_token(t))
            .count();
        let text_tokens = total_tokens - audio_tokens;

        let audio_ratio = if total_tokens > 0 {
            audio_tokens as f32 / total_tokens as f32
        } else {
            0.0
        };

        // Audio tokens at 25Hz frame rate
        let estimated_duration_secs = audio_tokens as f32 / 25.0;

        Self {
            total_tokens,
            audio_tokens,
            text_tokens,
            audio_ratio,
            estimated_duration_secs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_audio_tokens() {
        let tokens = vec![
            100,      // text
            151696,   // audio code 0
            200,      // text
            151700,   // audio code 4
            151696 + 6560, // audio code 6560
        ];

        let codes = extract_audio_tokens(&tokens);
        assert_eq!(codes, vec![0, 4, 6560]);
    }

    #[test]
    fn test_separate_tokens() {
        let tokens = vec![100, 151696, 200, 151700];
        let (text, audio) = separate_tokens(&tokens);
        assert_eq!(text, vec![100, 200]);
        assert_eq!(audio, vec![0, 4]);
    }

    #[test]
    fn test_audio_token_extractor() {
        let mut extractor = AudioTokenExtractor::with_batch_size(3);

        assert!(extractor.process(100).is_none()); // text
        assert!(extractor.process(151696).is_none()); // audio 0, need 2 more
        assert!(extractor.process(151697).is_none()); // audio 1, need 1 more

        let batch = extractor.process(151698); // audio 2, batch ready
        assert_eq!(batch, Some(vec![0, 1, 2]));

        // Add one more and flush
        assert!(extractor.process(151699).is_none());
        let remaining = extractor.flush();
        assert_eq!(remaining, Some(vec![3]));
    }

    #[test]
    fn test_audio_token_stats() {
        let tokens = vec![100, 151696, 200, 151700, 151701];
        let stats = AudioTokenStats::from_tokens(&tokens);

        assert_eq!(stats.total_tokens, 5);
        assert_eq!(stats.audio_tokens, 3);
        assert_eq!(stats.text_tokens, 2);
        assert!((stats.audio_ratio - 0.6).abs() < 0.01);
        assert!((stats.estimated_duration_secs - 0.12).abs() < 0.01);
    }

    #[test]
    fn test_validate_codes() {
        assert!(validate_codes(&[0, 100, 6560]));
        assert!(!validate_codes(&[0, 100, 6561])); // out of range
        assert!(!validate_codes(&[-1, 100])); // negative
    }

    #[test]
    fn test_codes_to_tokens() {
        let codes = vec![0, 4, 6560];
        let tokens = codes_to_tokens(&codes);
        assert_eq!(tokens, vec![151696, 151700, 158256]);
    }
}
