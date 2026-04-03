//! Backend trait definitions for Qwen3-TTS.
//!
//! Each compute backend (MLX, GGML/Ascend) implements these traits.
//! The shared generation logic in this crate calls these traits,
//! making it backend-agnostic.

use crate::error::Result;

// ============================================================================
// Talker backend — transformer forward pass + embedding construction
// ============================================================================

/// The core transformer inference backend.
///
/// Handles embedding lookups, text projection, transformer forward passes,
/// and code prediction. This is the performance-critical path.
pub trait TalkerBackend {
    /// Reset KV caches before a new generation.
    fn reset_caches(&mut self);

    /// Set RoPE speed factor for EOS steering (speed control).
    fn set_rope_speed_factor(&mut self, factor: f32);

    /// Build projected text embedding for a single token: text_proj(text_embed(token_id)).
    /// Returns a flat f32 vector of length `hidden_size`.
    fn text_embed(&mut self, token_id: u32) -> Result<Vec<f32>>;

    /// Build projected text embeddings for a batch of tokens.
    /// Returns flat f32 of length `tokens.len() * hidden_size`.
    fn text_embeds_batch(&mut self, token_ids: &[u32]) -> Result<Vec<f32>>;

    /// Build codec embedding for a single control token.
    /// Returns a flat f32 vector of length `hidden_size`.
    fn codec_embed(&mut self, codec_token: u32) -> Result<Vec<f32>>;

    /// Build the generation-step embedding: text_proj(text_token) + codec_embed_sum(prev_codes).
    /// `text_embed` is a pre-computed projected text embedding of length `hidden_size`.
    /// `prev_codes` are the 16 codebook values from the previous frame.
    /// Returns flat f32 of length `hidden_size`.
    fn generation_embed(
        &mut self,
        text_embed: &[f32],
        prev_codes: &[u32; 16],
    ) -> Result<Vec<f32>>;

    /// Build batched prefill embedding for all positions at once.
    ///
    /// The first `no_codec_positions` positions get text projection only (role prefix).
    /// Remaining positions get text_proj(tts_pad) + codec_embed(codec_tokens[i]).
    ///
    /// Returns flat f32 of length `(no_codec_positions + codec_tokens.len()) * hidden_size`.
    fn batched_prefill_embed(
        &mut self,
        text_only_tokens: &[u32],
        codec_tokens: &[u32],
    ) -> Result<Vec<f32>>;

    /// Forward pass through the transformer backbone.
    ///
    /// `input_embeds`: flat f32 of length `seq_len * hidden_size`.
    /// Returns `(logits, hidden_state)` — both for the LAST position only.
    /// `logits`: length `vocab_size`, `hidden`: length `hidden_size`.
    fn forward_step(
        &mut self,
        input_embeds: &[f32],
        seq_len: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)>;

    /// Generate sub-codes (codebooks 1-15) from hidden state + code0.
    fn predict_sub_codes(
        &mut self,
        hidden: &[f32],
        code0: u32,
    ) -> Result<Vec<u32>>;

    /// Codec vocabulary size (typically 3072).
    fn vocab_size(&self) -> usize;

    /// Hidden size (typically 2048).
    fn hidden_size(&self) -> usize;
}

// ============================================================================
// Speech decoder backend — codec frames to audio waveform
// ============================================================================

/// Converts codec frames to audio samples via the speech tokenizer decoder.
pub trait SpeechDecoderBackend {
    /// Decode codec frames to audio waveform.
    ///
    /// `codes`: N frames, each with 16 codebook values.
    /// Returns f32 audio samples in [-1, 1] at the output sample rate.
    fn decode(&mut self, codes: &[[u32; 16]]) -> Result<Vec<f32>>;

    /// Output sample rate (typically 24000 Hz).
    fn sample_rate(&self) -> u32;
}

// ============================================================================
// Speaker encoder backend — reference audio to speaker embedding
// ============================================================================

/// Extracts a speaker embedding vector from reference audio.
/// Used for x-vector voice cloning (Base model).
pub trait SpeakerEncoderBackend {
    /// Extract speaker embedding from audio samples.
    ///
    /// `audio`: f32 samples at `sample_rate` Hz.
    /// Returns embedding vector (typically 2048 dims for 1.7B model).
    fn extract_embedding(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<f32>>;
}

// ============================================================================
// Speech encoder backend — audio to codec frames (for ICL clone)
// ============================================================================

/// Encodes audio into codec frames (used by ICL voice cloning).
/// Not required for x-vector clone or CustomVoice modes.
pub trait SpeechEncoderBackend {
    /// Encode audio samples to codec frames.
    ///
    /// Returns N frames, each with 16 codebook values.
    fn encode(&mut self, audio: &[f32], sample_rate: u32) -> Result<Vec<[u32; 16]>>;
}
