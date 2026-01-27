//! Step-Audio 2 model
//!
//! Main model struct integrating encoder, adaptor, and LLM.
//!
//! # Performance Optimization
//!
//! MLX automatically caches compiled Metal kernels internally. For best performance:
//! 1. Call `warmup()` before benchmarking to trigger JIT compilation
//! 2. Use consistent input shapes to maximize kernel cache hits
//! 3. Audio preprocessing is GPU-accelerated (see `audio.rs`)

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use mlx_rs::{
    Array,
    error::Exception,
    module::{Module, ModuleParameters},
    ops::indexing::IndexOp,
};
use mlx_rs_core::KVCache;
use tokenizers::Tokenizer;

use crate::adaptor::StepAudio2Adaptor;
use crate::audio::{load_audio_mel, load_wav, samples_to_mel, resample, compute_mel_spectrogram, MAX_AUDIO_DURATION_SECS};
use crate::config::{tokens, AudioConfig, StepAudio2Config};
use crate::encoder::StepAudio2Encoder;
use crate::error::{Error, Result};
use crate::llm::{apply_repetition_penalty, load_llm_weights, sample, StepAudio2LLM};
use crate::think::{ThinkConfig, ThinkModeHandler, ThinkOutput};

/// Step-Audio 2 model
pub struct StepAudio2 {
    /// Audio encoder (Whisper-style)
    pub encoder: StepAudio2Encoder,
    /// Audio-to-LLM adaptor
    pub adaptor: StepAudio2Adaptor,
    /// LLM backbone (Qwen2.5-7B)
    pub llm: StepAudio2LLM,
    /// Model configuration
    pub config: StepAudio2Config,
    /// KV cache for generation
    cache: Vec<Option<KVCache>>,
    /// Tokenizer for text encoding/decoding
    tokenizer: Option<Arc<Tokenizer>>,
    /// Whether model has been warmed up (JIT compiled)
    warmed_up: bool,
}

impl StepAudio2 {
    /// Create a new model from configuration
    pub fn new(config: StepAudio2Config) -> Result<Self> {
        let encoder = StepAudio2Encoder::new(config.encoder.clone())?;
        let adaptor = StepAudio2Adaptor::new(config.adaptor.clone())?;
        let llm = StepAudio2LLM::new(config.llm.clone())?;

        Ok(Self {
            encoder,
            adaptor,
            llm,
            config,
            cache: Vec::new(),
            tokenizer: None,
            warmed_up: false,
        })
    }

    /// Warmup the model by running a forward pass with dummy input.
    ///
    /// This triggers MLX's JIT compilation of Metal kernels, so subsequent
    /// inference calls will be faster. Call this once before benchmarking
    /// or when latency of the first call matters.
    ///
    /// Returns the estimated audio duration (in seconds) used for warmup.
    pub fn warmup(&mut self) -> Result<f32> {
        if self.warmed_up {
            return Ok(0.0);
        }

        // Create dummy mel spectrogram (~1 second of audio)
        // Shape: [1, n_mels, n_frames] where n_frames ≈ sample_rate / hop_length
        let n_mels = self.config.audio.n_mels;
        let n_frames = 100; // ~1 second at 16kHz with hop_length=160

        let dummy_mel = Array::zeros::<f32>(&[1, n_mels, n_frames])?;

        // Run encoder + adaptor (triggers JIT for 32 encoder layers + adaptor)
        let _ = self.encode_audio(&dummy_mel)?;

        // Run a few LLM forward steps (triggers JIT for transformer layers)
        self.reset_cache();
        let dummy_embed = Array::zeros::<f32>(&[1, 10, self.config.llm.hidden_size])?;
        let _ = self.llm.forward_embeddings(&dummy_embed, &mut self.cache)?;

        self.warmed_up = true;
        self.reset_cache();

        Ok(1.0) // ~1 second equivalent
    }

    /// Check if model has been warmed up
    pub fn is_warmed_up(&self) -> bool {
        self.warmed_up
    }

    /// Load model from directory
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Load config
        let config = Self::load_config(model_dir)?;

        // Create model
        let mut model = Self::new(config)?;

        // Load weights
        model.load_weights(model_dir)?;

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            match Tokenizer::from_file(&tokenizer_path) {
                Ok(tok) => model.tokenizer = Some(Arc::new(tok)),
                Err(e) => eprintln!("Warning: Failed to load tokenizer: {}", e),
            }
        }

        Ok(model)
    }

    /// Load configuration from model directory
    fn load_config(model_dir: &Path) -> Result<StepAudio2Config> {
        let config_path = model_dir.join("config.json");

        if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)?;
            // Try to parse as HuggingFace config and convert
            let hf_config: serde_json::Value = serde_json::from_str(&config_str)?;

            // Build config from HF values
            let config = StepAudio2Config {
                encoder: crate::config::EncoderConfig {
                    n_mels: hf_config.get("encoder_config")
                        .and_then(|c| c.get("n_mels"))
                        .and_then(|v| v.as_i64())
                        .unwrap_or(128) as i32,
                    n_ctx: hf_config.get("encoder_config")
                        .and_then(|c| c.get("n_ctx"))
                        .and_then(|v| v.as_i64())
                        .unwrap_or(1500) as i32,
                    n_state: hf_config.get("encoder_config")
                        .and_then(|c| c.get("n_state"))
                        .and_then(|v| v.as_i64())
                        .unwrap_or(1280) as i32,
                    n_head: hf_config.get("encoder_config")
                        .and_then(|c| c.get("n_head"))
                        .and_then(|v| v.as_i64())
                        .unwrap_or(20) as i32,
                    n_layer: hf_config.get("encoder_config")
                        .and_then(|c| c.get("n_layer"))
                        .and_then(|v| v.as_i64())
                        .unwrap_or(32) as i32,
                },
                adaptor: crate::config::AdaptorConfig::default(),
                llm: crate::config::LLMConfig {
                    hidden_size: hf_config.get("hidden_size")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(3584) as i32,
                    intermediate_size: hf_config.get("intermediate_size")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(18944) as i32,
                    num_hidden_layers: hf_config.get("num_hidden_layers")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(28) as i32,
                    num_attention_heads: hf_config.get("num_attention_heads")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(28) as i32,
                    num_key_value_heads: hf_config.get("num_key_value_heads")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(4) as i32,
                    vocab_size: hf_config.get("vocab_size")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(158720) as i32,
                    max_position_embeddings: hf_config.get("max_position_embeddings")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(16384) as i32,
                    rope_theta: hf_config.get("rope_theta")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1000000.0) as f32,
                    rms_norm_eps: hf_config.get("rms_norm_eps")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1e-6) as f32,
                    tie_word_embeddings: hf_config.get("tie_word_embeddings")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                },
                audio: AudioConfig::default(),
            };

            Ok(config)
        } else {
            // Use default config
            Ok(StepAudio2Config::default())
        }
    }

    /// Load weights from model directory
    fn load_weights(&mut self, model_dir: &Path) -> Result<()> {
        // Load encoder weights
        self.load_encoder_weights(model_dir)?;

        // Load adaptor weights
        self.load_adaptor_weights(model_dir)?;

        // Load LLM weights
        load_llm_weights(&mut self.llm, model_dir)?;

        Ok(())
    }

    /// Load encoder weights
    fn load_encoder_weights(&mut self, model_dir: &Path) -> Result<()> {
        let weights_index = model_dir.join("model.safetensors.index.json");

        if weights_index.exists() {
            let json = std::fs::read_to_string(&weights_index)?;
            let weight_map: WeightMap = serde_json::from_str(&json)?;
            let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();

            for weight_file in weight_files {
                let weights_path = model_dir.join(weight_file);
                self.load_encoder_weights_from_file(&weights_path)?;
            }
        } else {
            let weights_path = model_dir.join("model.safetensors");
            if weights_path.exists() {
                self.load_encoder_weights_from_file(&weights_path)?;
            }
        }

        Ok(())
    }

    fn load_encoder_weights_from_file(&mut self, path: &Path) -> Result<()> {
        let loaded = Array::load_safetensors(path)?;
        let mut params = self.encoder.parameters_mut().flatten();

        for (st_key, value) in loaded {
            // Map encoder weight keys (e.g., "encoder.conv1.weight" -> "conv1.weight")
            if st_key.starts_with("encoder.") {
                let rust_key = st_key.strip_prefix("encoder.").unwrap();
                let rust_key = map_encoder_key(rust_key);

                // Transpose conv1d weights from PyTorch [out, in, kernel] to MLX [out, kernel, in]
                let value = if rust_key.contains("conv") && rust_key.ends_with("weight") && value.ndim() == 3 {
                    value.transpose_axes(&[0, 2, 1])?
                } else {
                    value
                };

                if let Some(param) = params.get_mut(&*rust_key) {
                    **param = value;
                }
            }
        }

        Ok(())
    }

    /// Load adaptor weights
    fn load_adaptor_weights(&mut self, model_dir: &Path) -> Result<()> {
        let weights_index = model_dir.join("model.safetensors.index.json");

        if weights_index.exists() {
            let json = std::fs::read_to_string(&weights_index)?;
            let weight_map: WeightMap = serde_json::from_str(&json)?;
            let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();

            for weight_file in weight_files {
                let weights_path = model_dir.join(weight_file);
                self.load_adaptor_weights_from_file(&weights_path)?;
            }
        } else {
            let weights_path = model_dir.join("model.safetensors");
            if weights_path.exists() {
                self.load_adaptor_weights_from_file(&weights_path)?;
            }
        }

        Ok(())
    }

    fn load_adaptor_weights_from_file(&mut self, path: &Path) -> Result<()> {
        let loaded = Array::load_safetensors(path)?;
        let mut params = self.adaptor.parameters_mut().flatten();

        for (st_key, value) in loaded {
            // Map adaptor weight keys (e.g., "adapter.conv.weight" -> "conv.weight")
            if st_key.starts_with("adapter.") || st_key.starts_with("adaptor.") || st_key.starts_with("audio_projector.") {
                let rust_key = if st_key.starts_with("adapter.") {
                    st_key.strip_prefix("adapter.").unwrap().to_string()
                } else if st_key.starts_with("adaptor.") {
                    st_key.strip_prefix("adaptor.").unwrap().to_string()
                } else {
                    st_key.strip_prefix("audio_projector.").unwrap().to_string()
                };
                let rust_key = map_adaptor_key(&rust_key);

                // Transpose conv1d weights from PyTorch [out, in, kernel] to MLX [out, kernel, in]
                let value = if rust_key.contains("conv") && rust_key.ends_with("weight") && value.ndim() == 3 {
                    value.transpose_axes(&[0, 2, 1])?
                } else {
                    value
                };

                if let Some(param) = params.get_mut(&*rust_key) {
                    **param = value;
                }
            }
        }

        Ok(())
    }

    /// Reset KV cache for new generation
    pub fn reset_cache(&mut self) {
        self.cache.clear();
    }

    /// Process audio through encoder and adaptor
    ///
    /// MLX automatically caches compiled kernels, so repeated calls with
    /// the same shapes will benefit from kernel caching.
    fn encode_audio(&mut self, mel: &Array) -> std::result::Result<Array, Exception> {
        // mel: [B, n_mels, T]
        let encoded = self.encoder.forward(mel)?;
        // encoded: [B, T/4, encoder_dim]
        let projected = self.adaptor.forward(&encoded)?;
        // projected: [B, T/8, llm_dim]
        Ok(projected)
    }

    /// Build ASR prompt tokens
    /// Returns (prompt_tokens, audio_insert_position)
    fn build_asr_prompt(&self, audio_len: i32) -> (Vec<i32>, usize) {
        // Prompt format:
        // <|im_start|>user\n<audio_start>[placeholder tokens x audio_len]<|im_end|>\n<|im_start|>assistant\n
        //
        // Token IDs (from tokenizer):
        // <|im_start|> = 151644
        // <|im_end|> = 151645
        // <audio_start> = 151688
        // "user" = 872 (from Qwen tokenizer)
        // "assistant" = 77091
        // "\n" = 198
        //
        // The Python model inserts audio AFTER <audio_start>, at position (marker_pos + 1)
        // So we add placeholder tokens after <audio_start> that will be replaced

        let mut prompt = vec![
            tokens::IM_START_TOKEN,  // <|im_start|>
            872,                      // user
            198,                      // \n
            tokens::AUDIO_START_TOKEN, // <audio_start> - marker token (kept, not replaced)
        ];

        // Position where audio features will be inserted (AFTER the audio_start marker)
        let audio_insert_pos = prompt.len();

        // Add placeholder tokens for audio length (will be replaced with audio features)
        // Using audio_start token as placeholder (common practice)
        for _ in 0..audio_len {
            prompt.push(tokens::AUDIO_START_TOKEN); // Placeholder
        }

        // Continue prompt after audio
        prompt.extend_from_slice(&[
            tokens::IM_END_TOKEN,    // <|im_end|>
            198,                      // \n
            tokens::IM_START_TOKEN,  // <|im_start|>
            77091,                    // assistant
            198,                      // \n
        ]);

        (prompt, audio_insert_pos)
    }

    /// Generate text tokens using placeholder-based audio insertion
    ///
    /// Performance notes:
    /// - MLX lazy evaluation batches operations within each iteration
    /// - item() forces synchronization to get token value for stopping conditions
    /// - KV cache is updated incrementally, avoiding recomputation
    /// - For longer generations, consider speculative decoding
    fn generate_text(
        &mut self,
        audio_features: &Array,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<i32>> {
        self.reset_cache();

        // Get audio sequence length
        let audio_len = audio_features.shape()[1];

        // Build prompt with audio placeholder
        let (prompt_tokens, audio_insert_pos) = self.build_asr_prompt(audio_len);

        // Get embeddings for prompt tokens
        let prompt_array = Array::from_slice(
            &prompt_tokens.iter().map(|&t| t).collect::<Vec<i32>>(),
            &[1, prompt_tokens.len() as i32],
        );
        let embeddings = self.llm.get_token_embeddings(&prompt_array)?;

        // Insert audio features at placeholder position
        // embeddings: [1, seq_len, hidden_dim]
        // audio_features: [1, audio_len, hidden_dim]
        let total_len = embeddings.shape()[1];

        // Get parts before and after audio insertion
        let before_audio = embeddings.index((.., ..audio_insert_pos as i32, ..));
        let after_audio_start = (audio_insert_pos + audio_len as usize) as i32;
        let after_audio = if after_audio_start < total_len {
            Some(embeddings.index((.., after_audio_start.., ..)))
        } else {
            None
        };

        // Concatenate: [before] + [audio_features] + [after]
        let combined = if let Some(after) = after_audio {
            mlx_rs::ops::concatenate_axis(&[&before_audio, audio_features, &after], 1)?
        } else {
            mlx_rs::ops::concatenate_axis(&[&before_audio, audio_features], 1)?
        };

        // Run prefill with combined embeddings (this is the heavy computation)
        let logits = self.llm.forward_embeddings(&combined, &mut self.cache)?;

        // Get last token logits
        let seq_len = logits.shape()[1];
        let last_logits = logits.index((.., seq_len - 1, ..));

        // Sample first token - item() forces evaluation
        let mut token = sample(&last_logits, temperature)?;
        let mut token_id = token.item::<i32>();

        let mut output_tokens = vec![token_id];

        // Autoregressive generation loop
        // Each iteration: embed -> forward -> sample -> check stopping
        // MLX batches lazy operations within each iteration automatically
        for _ in 1..max_tokens {
            // Check stopping conditions (requires evaluated token_id)
            if token_id == tokens::EOS_TOKEN
               || token_id == tokens::EOT_TOKEN
               || token_id == tokens::IM_END_TOKEN {
                break;
            }

            // Check if token is audio token (shouldn't happen in ASR mode)
            if tokens::is_audio_token(token_id) {
                break;
            }

            // Single token forward pass (incremental decoding)
            let token_array = Array::from_slice(&[token_id], &[1, 1]);
            let token_embed = self.llm.get_token_embeddings(&token_array)?;
            let logits = self.llm.forward_embeddings(&token_embed, &mut self.cache)?;
            let last_logits = logits.index((.., 0, ..));

            // Sample next token - item() is the synchronization point
            token = sample(&last_logits, temperature)?;
            token_id = token.item::<i32>();
            output_tokens.push(token_id);
        }

        Ok(output_tokens)
    }

    /// Decode token IDs to text
    /// Decode token IDs to text using the tokenizer
    fn decode_tokens(&self, tokens: &[i32]) -> String {
        if let Some(tokenizer) = &self.tokenizer {
            // Convert i32 tokens to u32 for tokenizer
            let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            match tokenizer.decode(&tokens_u32, true) {
                Ok(text) => text,
                Err(e) => {
                    eprintln!("Warning: Tokenizer decode error: {}", e);
                    format!("[tokens: {:?}]", tokens)
                }
            }
        } else {
            // Fallback: just return token IDs
            format!("[tokens: {:?}]", tokens)
        }
    }

    /// Transcribe audio file (ASR)
    pub fn transcribe(&mut self, audio_path: impl AsRef<Path>) -> Result<String> {
        // Load and process audio
        let mel = load_audio_mel(&audio_path, &self.config.audio)?;

        // Encode audio
        let audio_features = self.encode_audio(&mel)?;

        // Generate text tokens
        let tokens = self.generate_text(&audio_features, 512, 0.0)?;

        // Decode tokens to text
        let text = self.decode_tokens(&tokens);

        Ok(text)
    }

    /// Transcribe audio samples (ASR)
    pub fn transcribe_samples(&mut self, samples: &[f32], sample_rate: u32) -> Result<String> {
        // Convert samples to mel spectrogram
        let mel = samples_to_mel(samples, sample_rate, &self.config.audio)?;

        // Encode audio
        let audio_features = self.encode_audio(&mel)?;

        // Generate text tokens
        let tokens = self.generate_text(&audio_features, 2048, 0.0)?;

        // Decode tokens to text
        let text = self.decode_tokens(&tokens);

        Ok(text)
    }

    /// Transcribe long audio by chunking into segments
    ///
    /// Splits audio at the encoder's max context (15s) and transcribes each chunk.
    pub fn transcribe_long(&mut self, audio_path: impl AsRef<Path>) -> Result<String> {
        let (samples, src_rate) = load_wav(&audio_path)?;
        let target_rate = self.config.audio.sample_rate as u32;
        let samples = if src_rate != target_rate {
            resample(&samples, src_rate, target_rate)
        } else {
            samples
        };

        let max_samples = (MAX_AUDIO_DURATION_SECS * target_rate as f32) as usize;
        let total_duration = samples.len() as f32 / target_rate as f32;

        if samples.len() <= max_samples {
            // Short enough, single pass
            return self.transcribe_samples(&samples, target_rate);
        }

        eprintln!("Long audio ({:.1}s), chunking into {:.0}s segments...",
            total_duration, MAX_AUDIO_DURATION_SECS);

        let mut results = Vec::new();
        let mut offset = 0usize;
        let mut chunk_idx = 0;

        while offset < samples.len() {
            let end = (offset + max_samples).min(samples.len());
            let chunk = &samples[offset..end];
            let chunk_duration = chunk.len() as f32 / target_rate as f32;

            eprintln!("  Chunk {}: {:.1}s - {:.1}s ({:.1}s)",
                chunk_idx,
                offset as f32 / target_rate as f32,
                end as f32 / target_rate as f32,
                chunk_duration);

            let text = self.transcribe_samples(chunk, target_rate)?;
            results.push(text);

            offset = end;
            chunk_idx += 1;
        }

        Ok(results.join("\n"))
    }

    /// Build prompt for TTS (text input → audio output)
    fn build_tts_prompt(&self, text: &str) -> Vec<i32> {
        // TTS prompt format for Step-Audio 2:
        // <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
        // The model should generate audio tokens after this

        let mut prompt = vec![
            tokens::IM_START_TOKEN,  // <|im_start|>
            872,                      // user
            198,                      // \n
        ];

        // Encode text
        if let Some(tokenizer) = &self.tokenizer {
            if let Ok(encoding) = tokenizer.encode(text, false) {
                prompt.extend(encoding.get_ids().iter().map(|&id| id as i32));
            }
        }

        // Continue with assistant turn
        prompt.extend_from_slice(&[
            tokens::IM_END_TOKEN,    // <|im_end|>
            198,                      // \n
            tokens::IM_START_TOKEN,  // <|im_start|>
            77091,                    // assistant
            198,                      // \n
        ]);

        prompt
    }

    /// Generate audio tokens from text prompt
    #[cfg(feature = "tts")]
    fn generate_audio_tokens(
        &mut self,
        prompt_tokens: &[i32],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<i32>> {
        self.reset_cache();

        // Get embeddings for prompt
        let prompt_array = Array::from_slice(
            &prompt_tokens.iter().map(|&t| t).collect::<Vec<i32>>(),
            &[1, prompt_tokens.len() as i32],
        );
        let embeddings = self.llm.get_token_embeddings(&prompt_array)?;

        // Run prefill
        let logits = self.llm.forward_embeddings(&embeddings, &mut self.cache)?;

        // Get last token logits
        let seq_len = logits.shape()[1];
        let last_logits = logits.index((.., seq_len - 1, ..));

        // Sample first token
        let mut token = sample(&last_logits, temperature)?;
        let mut token_id = token.item::<i32>();

        let mut output_tokens = vec![token_id];
        let mut audio_token_count = 0;

        // Generation loop - collect audio tokens
        for _ in 1..max_tokens {
            // Check stopping conditions
            if token_id == tokens::EOS_TOKEN
               || token_id == tokens::EOT_TOKEN
               || token_id == tokens::IM_END_TOKEN {
                break;
            }

            // Count audio tokens for stopping condition
            if tokens::is_audio_token(token_id) {
                audio_token_count += 1;
            }

            // Single token forward pass
            let token_array = Array::from_slice(&[token_id], &[1, 1]);
            let token_embed = self.llm.get_token_embeddings(&token_array)?;
            let logits = self.llm.forward_embeddings(&token_embed, &mut self.cache)?;
            let last_logits = logits.index((.., 0, ..));

            // Sample next token
            token = sample(&last_logits, temperature)?;
            token_id = token.item::<i32>();
            output_tokens.push(token_id);
        }

        Ok(output_tokens)
    }

    /// Synthesize speech from text (TTS)
    ///
    /// Returns audio waveform at 24kHz
    #[cfg(feature = "tts")]
    pub fn synthesize(&mut self, text: &str) -> Result<Vec<f32>> {
        use crate::tts::{TTSDecoder, extract_audio_tokens};

        // Build TTS prompt
        let prompt_tokens = self.build_tts_prompt(text);

        // Generate tokens (including audio tokens)
        let generated_tokens = self.generate_audio_tokens(&prompt_tokens, 2048, 0.7)?;

        // Extract audio codes from generated tokens
        let audio_codes = extract_audio_tokens(&generated_tokens);

        if audio_codes.is_empty() {
            return Err(Error::Inference("No audio tokens generated".to_string()));
        }

        // Load TTS decoder if not already loaded
        // Note: In a real implementation, this would be cached
        let model_dir = Path::new("./Step-Audio-2-mini");
        let mut tts = TTSDecoder::load(model_dir)?;

        // Synthesize audio from codes
        tts.synthesize(&audio_codes)
    }

    /// Speech-to-speech: transcribe audio and generate speech response
    ///
    /// Takes audio input, processes it, and returns audio output
    #[cfg(feature = "tts")]
    pub fn speech_to_speech(
        &mut self,
        audio_path: impl AsRef<Path>,
    ) -> Result<(String, Vec<f32>)> {
        use crate::tts::{TTSDecoder, extract_audio_tokens};

        // Load and process audio
        let mel = load_audio_mel(&audio_path, &self.config.audio)?;

        // Encode audio
        let audio_features = self.encode_audio(&mel)?;

        // Generate response (text + audio tokens)
        // Use a modified prompt that encourages audio output
        let generated_tokens = self.generate_with_audio(&audio_features, 2048, 0.7)?;

        // Separate text and audio tokens
        let (text_tokens, audio_codes) = tokens::separate_tokens(&generated_tokens);

        // Decode text
        let text = self.decode_tokens(&text_tokens);

        // Synthesize audio if we have audio codes
        let audio = if !audio_codes.is_empty() {
            let model_dir = Path::new("./Step-Audio-2-mini");
            let mut tts = TTSDecoder::load(model_dir)?;
            tts.synthesize(&audio_codes)?
        } else {
            vec![]
        };

        Ok((text, audio))
    }

    /// Build prompt for speech-to-speech (audio input → audio output)
    /// Returns (prompt_tokens, audio_insert_position)
    #[cfg(feature = "tts")]
    fn build_speech_to_speech_prompt(&self, audio_len: i32) -> (Vec<i32>, usize) {
        // Speech-to-speech prompt format:
        // <|im_start|>user\n<audio_start>[audio features]<|im_end|>\n<|im_start|>assistant\n<audio_start>
        // The trailing <audio_start> signals the model to generate audio tokens

        let mut prompt = vec![
            tokens::IM_START_TOKEN,  // <|im_start|>
            872,                      // user
            198,                      // \n
            tokens::AUDIO_START_TOKEN, // <audio_start>
        ];

        let audio_insert_pos = prompt.len();

        // Placeholders for audio
        for _ in 0..audio_len {
            prompt.push(tokens::AUDIO_START_TOKEN);
        }

        // End user turn, start assistant with audio signal
        prompt.extend_from_slice(&[
            tokens::IM_END_TOKEN,    // <|im_end|>
            198,                      // \n
            tokens::IM_START_TOKEN,  // <|im_start|>
            77091,                    // assistant
            198,                      // \n
            tokens::AUDIO_START_TOKEN, // <audio_start> - signal to generate audio
        ]);

        (prompt, audio_insert_pos)
    }

    /// Generate response with potential audio tokens
    #[cfg(feature = "tts")]
    fn generate_with_audio(
        &mut self,
        audio_features: &Array,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<i32>> {
        self.reset_cache();

        let audio_len = audio_features.shape()[1];
        let (prompt_tokens, audio_insert_pos) = self.build_speech_to_speech_prompt(audio_len);

        // Get embeddings
        let prompt_array = Array::from_slice(
            &prompt_tokens.iter().map(|&t| t).collect::<Vec<i32>>(),
            &[1, prompt_tokens.len() as i32],
        );
        let embeddings = self.llm.get_token_embeddings(&prompt_array)?;

        // Insert audio features
        let total_len = embeddings.shape()[1];
        let before_audio = embeddings.index((.., ..audio_insert_pos as i32, ..));
        let after_audio_start = (audio_insert_pos + audio_len as usize) as i32;
        let after_audio = if after_audio_start < total_len {
            Some(embeddings.index((.., after_audio_start.., ..)))
        } else {
            None
        };

        let combined = if let Some(after) = after_audio {
            mlx_rs::ops::concatenate_axis(&[&before_audio, audio_features, &after], 1)?
        } else {
            mlx_rs::ops::concatenate_axis(&[&before_audio, audio_features], 1)?
        };

        // Run prefill
        let logits = self.llm.forward_embeddings(&combined, &mut self.cache)?;

        let seq_len = logits.shape()[1];
        let last_logits = logits.index((.., seq_len - 1, ..));

        let mut token = sample(&last_logits, temperature)?;
        let mut token_id = token.item::<i32>();

        let mut output_tokens = vec![token_id];
        let mut audio_token_count = 0;

        // Generation loop - allow audio tokens
        for _ in 1..max_tokens {
            // Track audio token count for logging
            if tokens::is_audio_token(token_id) {
                audio_token_count += 1;
            }

            // Stop on end tokens
            if token_id == tokens::EOS_TOKEN
               || token_id == tokens::EOT_TOKEN
               || token_id == tokens::IM_END_TOKEN
               || token_id == tokens::AUDIO_END_TOKEN {
                break;
            }

            let token_array = Array::from_slice(&[token_id], &[1, 1]);
            let token_embed = self.llm.get_token_embeddings(&token_array)?;
            let logits = self.llm.forward_embeddings(&token_embed, &mut self.cache)?;
            let last_logits = logits.index((.., 0, ..));

            token = sample(&last_logits, temperature)?;
            token_id = token.item::<i32>();
            output_tokens.push(token_id);
        }

        // Log how many audio tokens were generated
        if audio_token_count > 0 {
            eprintln!("  Generated {} audio tokens", audio_token_count);
        }

        Ok(output_tokens)
    }

    /// Speech-to-speech translation
    #[cfg(feature = "tts")]
    pub fn translate_speech(
        &mut self,
        audio_path: impl AsRef<Path>,
        _target_lang: &str,
    ) -> Result<Vec<f32>> {
        // For now, just do speech-to-speech without translation
        let (_, audio) = self.speech_to_speech(audio_path)?;
        Ok(audio)
    }

    /// Save audio samples to WAV file
    pub fn save_audio(&self, samples: &[f32], path: impl AsRef<Path>) -> Result<()> {
        crate::audio::save_wav(samples, 24000, path)
    }

    // ========================================================================
    // Think Mode Support (Phase 2)
    // ========================================================================

    /// Process audio with think mode (for Step-Audio 2 mini-Think)
    ///
    /// This method enables extended reasoning before responding.
    /// The model first generates thinking content inside `<think>...</think>` tags,
    /// then generates the actual response.
    pub fn think_and_respond(
        &mut self,
        audio_path: impl AsRef<Path>,
        think_config: ThinkConfig,
    ) -> Result<ThinkOutput> {
        // Load and process audio
        let mel = load_audio_mel(&audio_path, &self.config.audio)?;

        // Encode audio
        let audio_features = self.encode_audio(&mel)?;

        // Generate with think mode
        self.generate_with_think(&audio_features, think_config, 0.7)
    }

    /// Process audio samples with think mode
    pub fn think_and_respond_samples(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        think_config: ThinkConfig,
    ) -> Result<ThinkOutput> {
        // Convert samples to mel spectrogram
        let mel = samples_to_mel(samples, sample_rate, &self.config.audio)?;

        // Encode audio
        let audio_features = self.encode_audio(&mel)?;

        // Generate with think mode
        self.generate_with_think(&audio_features, think_config, 0.7)
    }

    /// Generate tokens with think mode handling
    fn generate_with_think(
        &mut self,
        audio_features: &Array,
        think_config: ThinkConfig,
        temperature: f32,
    ) -> Result<ThinkOutput> {
        self.reset_cache();

        let mut handler = ThinkModeHandler::new(think_config);
        let max_total_tokens = handler.current_max_tokens() * 2; // thinking + response

        // Run prefill with audio features
        let logits = self.llm.forward_embeddings(audio_features, &mut self.cache)?;

        // Get last token logits
        let seq_len = logits.shape()[1];
        let last_logits = logits.index((.., seq_len - 1, ..));

        // Sample first token
        let mut token = sample(&last_logits, temperature)?;
        let mut token_id = token.item::<i32>();

        // Process first token through handler
        let token_text = self.decode_single_token(token_id);
        handler.process_token(token_id, &token_text);

        // Generate remaining tokens
        let mut total_generated = 1;
        while total_generated < max_total_tokens {
            // Check stopping conditions
            if handler.should_stop(token_id) {
                break;
            }

            // Check for EOS
            if token_id == tokens::EOS_TOKEN || token_id == tokens::IM_END_TOKEN {
                break;
            }

            // Get embedding for current token
            let token_array = Array::from_slice(&[token_id], &[1, 1]);
            let token_embed = self.llm.get_token_embeddings(&token_array)?;

            // Forward through LLM
            let logits = self.llm.forward_embeddings(&token_embed, &mut self.cache)?;
            let last_logits = logits.index((.., 0, ..));

            // Use different temperature based on phase
            let phase_temp = if handler.is_thinking() {
                temperature // Higher temperature for creative thinking
            } else {
                temperature * 0.5 // Lower temperature for precise responses
            };

            // Sample next token
            token = sample(&last_logits, phase_temp)?;
            token_id = token.item::<i32>();

            // Process token through handler
            let token_text = self.decode_single_token(token_id);
            handler.process_token(token_id, &token_text);

            total_generated += 1;
        }

        handler.finish();

        // Build output using our decode function
        let output = handler.build_output(|tokens| self.decode_tokens(tokens));

        Ok(output)
    }

    /// Decode a single token to text
    fn decode_single_token(&self, token_id: i32) -> String {
        // TODO: Implement proper tokenizer decoding
        // For now, return a placeholder
        // In a real implementation, this would use the Qwen tokenizer
        if tokens::is_audio_token(token_id) {
            format!("[audio:{}]", tokens::token_to_code(token_id))
        } else if token_id == tokens::EOS_TOKEN {
            "<eos>".to_string()
        } else if token_id == tokens::IM_START_TOKEN {
            "<|im_start|>".to_string()
        } else if token_id == tokens::IM_END_TOKEN {
            "<|im_end|>".to_string()
        } else {
            // Placeholder - would need actual tokenizer
            format!("[{}]", token_id)
        }
    }
}

/// Map encoder weight keys from HuggingFace format
fn map_encoder_key(key: &str) -> std::rc::Rc<str> {
    let key = key
        .replace("blocks.", "layers.")
        .replace(".attn.query.", ".self_attn.q_proj.")
        .replace(".attn.key.", ".self_attn.k_proj.")
        .replace(".attn.value.", ".self_attn.v_proj.")
        .replace(".attn.out.", ".self_attn.out_proj.")
        .replace(".attn_ln.", ".self_attn_layer_norm.")
        .replace(".mlp_ln.", ".final_layer_norm.")
        .replace(".mlp.0.", ".mlp.fc1.")
        .replace(".mlp.2.", ".mlp.fc2.")
        .replace("after_norm.", "ln_post.")
        .replace("positional_embedding.weight", "positional_embedding");

    std::rc::Rc::from(key)
}

/// Map adaptor weight keys from HuggingFace format
fn map_adaptor_key(key: &str) -> std::rc::Rc<str> {
    let key = key
        .replace("proj.0.", "conv.")
        .replace("proj.2.", "linear1.")
        .replace("proj.4.", "linear2.");

    std::rc::Rc::from(key)
}

#[derive(Debug, Clone, serde::Deserialize)]
struct WeightMap {
    weight_map: std::collections::HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let config = StepAudio2Config::default();
        // Creating the full model would use too much memory for tests
        // Just verify the config is valid
        assert_eq!(config.encoder.n_mels, 128);
        assert_eq!(config.llm.hidden_size, 3584);
    }

    #[test]
    fn test_key_mapping() {
        // Test encoder key mapping
        let key = "blocks.0.attn.query.weight";
        let mapped = map_encoder_key(key);
        assert_eq!(&*mapped, "layers.0.self_attn.q_proj.weight");

        let key = "blocks.0.attn.key.weight";
        let mapped = map_encoder_key(key);
        assert_eq!(&*mapped, "layers.0.self_attn.k_proj.weight");
    }
}
