//! Combined FunASR-Qwen4B model for ASR + translation.
//!
//! Pipeline: Audio → SenseVoice → Adaptor → Qwen3-4B → Text

use crate::adaptor::AudioAdaptorQwen4B;
use crate::audio::{load_wav, resample, AudioConfig, compute_mel_spectrogram, apply_lfr};
use crate::error::{Error, Result};
use crate::sensevoice_encoder::{SenseVoiceEncoder, SenseVoiceEncoderConfig};

use mlx_rs::Array;
use mlx_rs::module::Module;
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::quantization::MaybeQuantized;
use qwen3_mlx::{
    Model as Qwen3Model, KVCache, Generate, load_model, load_tokenizer,
    AttentionInput, sample, create_attention_mask, AttentionMask,
};
use std::path::Path;
use tokenizers::Tokenizer;

// ── Qwen3 special token IDs ──────────────────────────────────────────────────
const TOKEN_EOS: u32 = 151643;             // <|endoftext|>
const TOKEN_IM_START: u32 = 151644;        // <|im_start|>
const TOKEN_IM_END: u32 = 151645;          // <|im_end|>
const TOKEN_START_OF_SPEECH: u32 = 151646; // <|startofspeech|>
const TOKEN_END_OF_SPEECH: u32 = 151647;   // <|endofspeech|>

// ── Generation parameters ────────────────────────────────────────────────────
/// Tokens generated per second of audio (Chinese transcription).
const TOKENS_PER_SEC_CHINESE: f32 = 5.0;
/// Tokens generated per second of audio (English translation).
const TOKENS_PER_SEC_ENGLISH: f32 = 8.0;
/// Minimum number of tokens to generate regardless of audio duration.
const MIN_GENERATED_TOKENS: usize = 30;
/// Maximum number of tokens to generate regardless of audio duration.
const MAX_GENERATED_TOKENS: usize = 400;

// ── Repetition detection thresholds ──────────────────────────────────────────
/// Minimum block size (in chars) to qualify as a repeated block.
const MIN_REPEAT_BLOCK_CHARS: usize = 30;
/// Minimum text length (in chars) before checking for repeated blocks.
const MIN_TEXT_LEN_FOR_REPEAT_DETECTION: usize = 60;
/// Minimum common prefix length to consider as text-level repetition.
const MIN_COMMON_CHARS: usize = 20;
/// Minimum text length (in chars) before checking for text-level repetition.
const MIN_TEXT_LEN_FOR_TEXT_REPETITION: usize = 40;

/// Speech tokens for multimodal prompts
pub struct SpeechTokens {
    pub start_of_speech: u32,
    pub end_of_speech: u32,
    pub im_start: u32,
    pub im_end: u32,
    pub eos: u32,
}

impl Default for SpeechTokens {
    fn default() -> Self {
        Self {
            start_of_speech: TOKEN_START_OF_SPEECH,
            end_of_speech: TOKEN_END_OF_SPEECH,
            im_start: TOKEN_IM_START,
            im_end: TOKEN_IM_END,
            eos: TOKEN_EOS,
        }
    }
}

/// Combined FunASR-Qwen4B model
pub struct FunASRQwen4B {
    pub encoder: SenseVoiceEncoder,
    pub adaptor: AudioAdaptorQwen4B,
    pub llm: Qwen3Model,
    pub tokenizer: Tokenizer,
    pub audio_config: AudioConfig,
    pub speech_tokens: SpeechTokens,
    #[cfg(feature = "punctuation")]
    pub punc_model: Option<funasr_mlx::punctuation::PunctuationModel>,
}

impl FunASRQwen4B {
    /// Load model from directory
    ///
    /// Auto-discovers model files in the directory:
    /// - SenseVoice encoder: `sensevoice_iic.safetensors`, `sensevoice/encoder.safetensors`, or `model.safetensors`
    /// - Adaptor: `adaptor_phase2_final.safetensors`, `adaptor.safetensors`, or any `adaptor*.safetensors`
    /// - Qwen3-4B: `models/Qwen3-4B-8bit/`, `models/Qwen3-4B-4bit/`, `models/Qwen3-4B/`, or `qwen3-4b/`
    pub fn load(model_dir: &str) -> Result<Self> {
        let model_path = Path::new(model_dir);

        // Load SenseVoice encoder
        let mut encoder = SenseVoiceEncoder::new(SenseVoiceEncoderConfig::default())?;

        // Try to load encoder weights from various locations
        let encoder_paths = [
            model_path.join("sensevoice_iic.safetensors"),
            model_path.join("sensevoice").join("encoder.safetensors"),
            model_path.join("model.safetensors"),
            std::path::PathBuf::from(std::env::var("SENSEVOICE_WEIGHTS").unwrap_or_default()),
            dirs::home_dir().unwrap_or_default().join(".OminiX/models/funasr-nano/model.safetensors"),
        ];

        let mut encoder_loaded = false;
        for encoder_path in &encoder_paths {
            if encoder_path.exists() {
                match encoder.load_weights(encoder_path) {
                    Ok(_) => {
                        eprintln!("Loaded SenseVoice encoder from {:?}", encoder_path);
                        encoder_loaded = true;
                        break;
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load encoder from {:?}: {:?}", encoder_path, e);
                    }
                }
            }
        }

        if !encoder_loaded {
            eprintln!("Warning: SenseVoice encoder weights not loaded. Audio transcription may not work.");
        }

        // Load adaptor - try multiple naming conventions (prefer Phase 3 over Phase 2)
        let adaptor_candidates = [
            model_path.join("adaptor_phase3_final.safetensors"),
            model_path.join("models").join("adaptor_phase3_final.safetensors"),
            model_path.join("adaptor_phase2_final.safetensors"),
            model_path.join("models").join("adaptor_phase2_final.safetensors"),
            model_path.join("adaptor.safetensors"),
        ];
        let mut adaptor = AudioAdaptorQwen4B::new()?;
        let mut adaptor_loaded = false;
        for adaptor_path in &adaptor_candidates {
            if adaptor_path.exists() {
                adaptor.load_weights(adaptor_path.to_str().unwrap())?;
                eprintln!("Loaded adaptor from {:?}", adaptor_path);
                adaptor_loaded = true;
                break;
            }
        }
        if !adaptor_loaded {
            // Scan for any adaptor*.safetensors
            if let Ok(entries) = std::fs::read_dir(model_path) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let name = name.to_string_lossy();
                    if name.starts_with("adaptor") && name.ends_with(".safetensors") {
                        adaptor.load_weights(entry.path().to_str().unwrap())?;
                        eprintln!("Loaded adaptor from {:?}", entry.path());
                        adaptor_loaded = true;
                        break;
                    }
                }
            }
        }
        if !adaptor_loaded {
            eprintln!("Warning: Adaptor weights not loaded.");
        }

        // Load Qwen3-4B - try multiple paths (prefer quantized)
        let qwen_candidates = [
            model_path.join("models").join("Qwen3-4B-4bit"),
            model_path.join("models").join("Qwen3-4B-8bit"),
            model_path.join("models").join("Qwen3-4B"),
            model_path.join("qwen3-4b"),
        ];
        let qwen_path = qwen_candidates.iter()
            .find(|p| p.join("config.json").exists())
            .ok_or_else(|| Error::ModelLoad(format!(
                "No Qwen3-4B model found. Searched: {:?}", qwen_candidates
            )))?;
        eprintln!("Loading Qwen3 from {:?}", qwen_path);

        let llm = load_model(qwen_path)
            .map_err(|e| Error::ModelLoad(format!("Failed to load Qwen3: {:?}", e)))?;

        // Load tokenizer
        let tokenizer = load_tokenizer(qwen_path)
            .map_err(|e| Error::Tokenizer(format!("Failed to load tokenizer: {:?}", e)))?;

        // Create audio config
        let audio_config = AudioConfig::default();

        // Load punctuation model if available
        #[cfg(feature = "punctuation")]
        let punc_model = {
            let punc_candidates = [
                model_path.join("punc_ct"),
                model_path.join("models").join("punc_ct"),
                // Common system-wide location
                dirs::home_dir().unwrap_or_default()
                    .join("home/VoiceDialogue11/assets/models/asr/funasr/punc_ct-transformer_cn-en-common-vocab471067-large"),
                dirs::home_dir().unwrap_or_default()
                    .join(".OminiX/models/punc_ct"),
            ];
            let mut loaded = None;
            for punc_path in &punc_candidates {
                if punc_path.join("model_quant.onnx").exists() || punc_path.join("model.onnx").exists() {
                    match funasr_mlx::punctuation::PunctuationModel::load(punc_path) {
                        Ok(m) => {
                            eprintln!("Loaded punctuation model from {:?}", punc_path);
                            loaded = Some(m);
                            break;
                        }
                        Err(e) => {
                            eprintln!("Warning: Failed to load punctuation model from {:?}: {:?}", punc_path, e);
                        }
                    }
                }
            }
            if loaded.is_none() {
                eprintln!("Note: No punctuation model found. Output will lack punctuation.");
            }
            loaded
        };

        Ok(Self {
            encoder,
            adaptor,
            llm,
            tokenizer,
            audio_config,
            speech_tokens: SpeechTokens::default(),
            #[cfg(feature = "punctuation")]
            punc_model,
        })
    }

    /// Transcribe audio file to Chinese text
    pub fn transcribe(&mut self, audio_path: &str) -> Result<String> {
        // Load and preprocess audio
        let (samples, sample_rate) = load_wav(audio_path)?;
        self.transcribe_samples(&samples, sample_rate)
    }

    /// Transcribe long audio by splitting into chunks
    ///
    /// Splits audio into `chunk_secs`-second segments (default 30s)
    /// and transcribes each independently. Returns concatenated result.
    pub fn transcribe_long(&mut self, audio_path: &str, chunk_secs: f32) -> Result<String> {
        let (samples, sample_rate) = load_wav(audio_path)?;
        self.transcribe_long_samples(&samples, sample_rate, chunk_secs)
    }

    /// Transcribe long audio samples by splitting into chunks (Chinese)
    pub fn transcribe_long_samples(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        chunk_secs: f32,
    ) -> Result<String> {
        self.process_long_samples(samples, sample_rate, chunk_secs, "语音转写成中文：", "")
    }

    /// Translate long audio to English by splitting into chunks
    pub fn translate_long(&mut self, audio_path: &str, chunk_secs: f32) -> Result<String> {
        let (samples, sample_rate) = load_wav(audio_path)?;
        self.translate_long_samples(&samples, sample_rate, chunk_secs)
    }

    /// Translate long audio samples to English by splitting into chunks
    pub fn translate_long_samples(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        chunk_secs: f32,
    ) -> Result<String> {
        self.process_long_samples(samples, sample_rate, chunk_secs, "Translate the speech to English:", " ")
    }

    /// Process long audio samples with arbitrary prompt by splitting into chunks
    fn process_long_samples(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        chunk_secs: f32,
        prompt: &str,
        separator: &str,
    ) -> Result<String> {
        let chunk_size = (chunk_secs * sample_rate as f32) as usize;
        let total_chunks = (samples.len() + chunk_size - 1) / chunk_size;
        let mut results: Vec<String> = Vec::new();

        for (i, chunk) in samples.chunks(chunk_size).enumerate() {
            if chunk.len() < (sample_rate as usize / 10) {
                break; // skip chunks shorter than 100ms
            }
            eprint!("\r  Chunk {}/{}", i + 1, total_chunks);
            let text = self.transcribe_samples_with_prompt(chunk, sample_rate, prompt)?;
            if !text.is_empty() {
                results.push(text);
            }
        }
        eprintln!();

        let joined = results.join(separator);

        // Final pass: remove repeated blocks in the joined text
        Ok(Self::remove_repeated_blocks(&joined))
    }

    /// Remove repeated blocks of text from the joined output.
    ///
    /// After chunked processing, the model sometimes regenerates content that
    /// was already transcribed in a previous chunk. This finds blocks of >=30
    /// chars that appear more than once and keeps only the first occurrence.
    fn remove_repeated_blocks(text: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        if len < MIN_TEXT_LEN_FOR_REPEAT_DETECTION {
            return text.to_string();
        }

        // Find repeated blocks: for each position, check if a block starting
        // there also appears earlier in the text.
        let min_block = MIN_REPEAT_BLOCK_CHARS;
        let mut skip_ranges: Vec<(usize, usize)> = Vec::new();

        // Scan through the text looking for blocks that duplicate earlier content
        let mut pos = min_block;
        while pos < len {
            // Check if chars[pos..pos+min_block] appears earlier
            let end = (pos + min_block).min(len);
            let block: String = chars[pos..end].iter().collect();

            let prefix: String = chars[..pos].iter().collect();
            if let Some(first_pos) = prefix.find(&block) {
                // Found a duplicate block. Extend it to find the full overlap.
                let mut match_len = min_block;
                while pos + match_len < len
                    && first_pos + match_len < pos
                    && chars[first_pos + match_len] == chars[pos + match_len]
                {
                    match_len += 1;
                }

                // Only skip if the repeated block is substantial (>=30 chars)
                if match_len >= min_block {
                    skip_ranges.push((pos, pos + match_len));
                    pos += match_len;
                    continue;
                }
            }
            pos += 1;
        }

        if skip_ranges.is_empty() {
            return text.to_string();
        }

        // Build result, skipping the duplicate ranges
        let mut result = String::new();
        let mut i = 0;
        for (start, end) in &skip_ranges {
            if i < *start {
                let segment: String = chars[i..*start].iter().collect();
                result.push_str(&segment);
            }
            i = *end;
        }
        if i < len {
            let segment: String = chars[i..].iter().collect();
            result.push_str(&segment);
        }

        result
    }

    /// Transcribe raw audio samples to Chinese text
    ///
    /// This is useful for streaming audio or when you already have the samples.
    pub fn transcribe_samples(&mut self, samples: &[f32], sample_rate: u32) -> Result<String> {
        self.transcribe_samples_with_prompt(samples, sample_rate, "语音转写成中文：")
    }

    /// Translate audio directly to English (single pass, no Chinese intermediate)
    ///
    /// Uses Qwen3-4B's multilingual capability to generate English text
    /// directly from audio features.
    pub fn translate_audio_to_english(&mut self, audio_path: &str) -> Result<String> {
        let (samples, sample_rate) = load_wav(audio_path)?;
        self.translate_samples_to_english(&samples, sample_rate)
    }

    /// Translate raw audio samples directly to English
    pub fn translate_samples_to_english(&mut self, samples: &[f32], sample_rate: u32) -> Result<String> {
        self.transcribe_samples_with_prompt(
            samples,
            sample_rate,
            "Translate the speech to English:",
        )
    }

    /// Transcribe/translate audio with a custom system prompt
    ///
    /// The prompt is placed in the system turn of the ChatML template.
    /// Examples:
    /// - "语音转写成中文：" (Chinese transcription)
    /// - "Translate the speech to English:" (direct translation)
    /// - "Transcribe and summarize:" (custom task)
    pub fn transcribe_samples_with_prompt(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        prompt: &str,
    ) -> Result<String> {
        // Resample to 16kHz if needed
        let samples = if sample_rate != 16000 {
            resample(samples, sample_rate, 16000)?
        } else {
            samples.to_vec()
        };

        // Compute duration-proportional max_tokens to prevent hallucination
        // Chinese: ~3-5 chars/sec ≈ 3-5 tokens/sec
        // English: ~3-4 words/sec ≈ 5-8 tokens/sec
        let duration_secs = samples.len() as f32 / 16000.0;
        let is_chinese = prompt.contains("中文");
        let tokens_per_sec = if is_chinese { TOKENS_PER_SEC_CHINESE } else { TOKENS_PER_SEC_ENGLISH };
        let max_tokens = ((duration_secs * tokens_per_sec) as usize).max(MIN_GENERATED_TOKENS).min(MAX_GENERATED_TOKENS);

        // Compute mel spectrogram
        let mel = compute_mel_spectrogram(&samples, &self.audio_config)?;

        // Apply LFR (Low Frame Rate) transformation
        let mel_lfr = apply_lfr(&mel, 7, 6)?;

        // Encode with SenseVoice
        let encoder_out = self.encoder.forward(&mel_lfr)?;

        // Project to Qwen4B embedding space
        let adapted = self.adaptor.forward(&encoder_out)?;

        // Generate text with duration-proportional limit
        let text = self.generate_from_audio_features(&adapted, prompt, max_tokens)?;

        Ok(text)
    }

    /// Process audio to get adapted features without generating text
    ///
    /// This is useful for batched processing or when you want more control
    /// over the generation step.
    pub fn encode_audio(&mut self, samples: &[f32], sample_rate: u32) -> Result<Array> {
        // Resample to 16kHz if needed
        let samples = if sample_rate != 16000 {
            resample(samples, sample_rate, 16000)?
        } else {
            samples.to_vec()
        };

        // Compute mel spectrogram
        let mel = compute_mel_spectrogram(&samples, &self.audio_config)?;

        // Apply LFR (Low Frame Rate) transformation
        let mel_lfr = apply_lfr(&mel, 7, 6)?;

        // Encode with SenseVoice
        let encoder_out = self.encoder.forward(&mel_lfr)?;

        // Project to Qwen4B embedding space
        self.adaptor.forward(&encoder_out)
    }

    /// Transcribe and translate to English
    pub fn transcribe_and_translate(&mut self, audio_path: &str) -> Result<(String, String)> {
        // First transcribe to Chinese
        let chinese = self.transcribe(audio_path)?;

        // Then translate to English
        let english = self.translate(&chinese)?;

        Ok((chinese, english))
    }

    /// Translate Chinese text to English
    pub fn translate(&mut self, chinese: &str) -> Result<String> {
        let prompt = format!(
            "<|im_start|>user\nTranslate to English: {}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            chinese
        );

        self.generate_text(&prompt, 200)
    }

    /// Generate text from audio features (multimodal)
    ///
    /// Uses bare prompt format matching Phase 3 LoRA training:
    /// [prompt_tokens | audio_features | no_think_tokens]
    /// The no_think prefix (`<think>\n\n</think>\n\n`) bypasses Qwen3's
    /// thinking mode so the model generates content directly.
    fn generate_from_audio_features(
        &mut self,
        audio_features: &Array,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<String> {
        // Encode the bare prompt (e.g., "语音转写成中文：" or "Translate the speech to English:")
        let prompt_encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| Error::Tokenizer(format!("Tokenization failed: {}", e)))?;

        let prompt_tokens: Vec<i32> = prompt_encoding.get_ids().iter()
            .map(|&t| t as i32)
            .collect();

        // Get prompt embeddings
        let token_array = Array::from_slice(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
        let prompt_embed = self.get_token_embeddings(&token_array)?;
        mlx_rs::transforms::eval([&prompt_embed])?;

        // Build suffix to append after audio features
        // Chinese: anti-hallucination instruction to only output transcription
        // English: no-think prefix + anti-hallucination instruction
        let is_chinese_prompt = prompt.contains("中文");
        let suffix_text = if is_chinese_prompt {
            "只输出转写文字，不要分析或解释："
        } else {
            "<think>\n\n</think>\n\n"
        };
        let suffix_encoding = self.tokenizer.encode(suffix_text, false)
            .map_err(|e| Error::Tokenizer(format!("Tokenization failed: {}", e)))?;
        let suffix_tokens: Vec<i32> = suffix_encoding.get_ids().iter()
            .map(|&t| t as i32)
            .collect();
        let suffix_array = Array::from_slice(&suffix_tokens, &[1, suffix_tokens.len() as i32]);
        let suffix_embed = self.get_token_embeddings(&suffix_array)?;
        mlx_rs::transforms::eval([&suffix_embed])?;

        let combined = mlx_rs::ops::concatenate_axis(
            &[&prompt_embed, audio_features, &suffix_embed],
            1,
        ).map_err(|e| Error::ModelLoad(format!("Concat failed: {:?}", e)))?;
        mlx_rs::transforms::eval([&combined])?;

        // Generate tokens from combined embeddings
        self.generate_from_embeddings(&combined, max_tokens)
    }

    /// Get token embeddings (for multimodal injection)
    fn get_token_embeddings(&mut self, tokens: &Array) -> Result<Array> {
        match &mut self.llm.model.embed_tokens {
            MaybeQuantized::Original(embed) => embed.forward(tokens)
                .map_err(|e| Error::ModelLoad(format!("Embed failed: {:?}", e))),
            MaybeQuantized::Quantized(embed) => embed.forward(tokens)
                .map_err(|e| Error::ModelLoad(format!("Embed failed: {:?}", e))),
        }
    }

    /// Forward pass with embedding inputs (for multimodal)
    ///
    /// Runs embeddings through transformer layers, returns logits.
    fn forward_embeddings(
        &mut self,
        embeddings: &Array,
        cache: &mut Vec<Option<KVCache>>,
    ) -> Result<Array> {
        // Initialize cache if empty
        if cache.is_empty() {
            *cache = (0..self.llm.model.layers.len())
                .map(|_| Some(KVCache::default()))
                .collect();
        }

        // Create attention mask
        let mask = match create_attention_mask(embeddings, cache, Some(true))
            .map_err(|e| Error::ModelLoad(format!("Mask creation failed: {:?}", e)))?
        {
            Some(AttentionMask::Array(m)) => Some(m),
            _ => None,
        };

        // Forward through transformer layers
        let mut h = embeddings.clone();
        for (layer, c) in self.llm.model.layers.iter_mut().zip(cache.iter_mut()) {
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c.as_mut(),
            };
            h = layer.forward(layer_input)
                .map_err(|e| Error::ModelLoad(format!("Layer forward failed: {:?}", e)))?;
        }

        // Apply final norm
        h = self.llm.model.norm.forward(&h)
            .map_err(|e| Error::ModelLoad(format!("Norm failed: {:?}", e)))?;

        // Get logits (tied embeddings or lm_head)
        match &mut self.llm.lm_head {
            Some(lm_head) => lm_head.forward(&h)
                .map_err(|e| Error::ModelLoad(format!("LM head failed: {:?}", e))),
            None => {
                match &mut self.llm.model.embed_tokens {
                    MaybeQuantized::Original(embed) => embed.as_linear(&h)
                        .map_err(|e| Error::ModelLoad(format!("Tied embed failed: {:?}", e))),
                    MaybeQuantized::Quantized(embed) => embed.as_linear(&h)
                        .map_err(|e| Error::ModelLoad(format!("Tied embed failed: {:?}", e))),
                }
            }
        }
    }

    /// Generate text from pre-computed embeddings
    ///
    /// Uses greedy decoding with n-gram repetition detection and
    /// text-level dedup post-processing.
    fn generate_from_embeddings(
        &mut self,
        embeddings: &Array,
        max_tokens: usize,
    ) -> Result<String> {
        let temperature: f32 = 0.0; // greedy decoding for ASR
        let top_k: usize = 20;
        let presence_penalty: f32 = 0.0;

        let mut cache: Vec<Option<KVCache>> = Vec::new();
        let mut tokens: Vec<i32> = Vec::new();

        // Prefill: forward pass with full embeddings
        let logits = self.forward_embeddings(embeddings, &mut cache)?;

        // Sample from last position
        let last_logits = logits.index((.., -1, ..));
        let token = Self::sample_top_k_p(&last_logits, temperature, top_k, &tokens, presence_penalty)?;
        mlx_rs::transforms::eval([&token])?;
        let mut token_id: i32 = token.item();

        for _step in 0..max_tokens {
            // Check for EOS tokens
            if token_id == self.speech_tokens.eos as i32 || token_id == self.speech_tokens.im_end as i32 {
                break;
            }

            tokens.push(token_id);

            // N-gram repetition detection (patterns 1-512 tokens)
            if tokens.len() >= 4 {
                let mut repeat_n: usize = 0;
                let max_n = 512.min(tokens.len() / 2);
                'outer: for n in 1..=max_n {
                    let reps_needed: usize = if n <= 2 { 3 } else { 2 };
                    if tokens.len() >= n * reps_needed {
                        let tail = &tokens[tokens.len() - n * reps_needed..];
                        let pattern = &tail[tail.len() - n..];
                        let all_match = (0..reps_needed).all(|i| {
                            &tail[i * n..(i + 1) * n] == pattern
                        });
                        if all_match {
                            repeat_n = n;
                            break 'outer;
                        }
                    }
                }
                if repeat_n > 0 {
                    let pattern: Vec<i32> = tokens[tokens.len() - repeat_n..].to_vec();
                    let mut pos = tokens.len();
                    while pos >= repeat_n {
                        if tokens[pos - repeat_n..pos] == pattern[..] {
                            pos -= repeat_n;
                        } else {
                            break;
                        }
                    }
                    // Keep only the first occurrence, remove all repeated copies
                    tokens.truncate(pos);
                    break;
                }
            }

            // Get embedding for next step
            let token_array = Array::from_slice(&[token_id], &[1, 1]);
            let h = self.get_token_embeddings(&token_array)?;

            // Forward through LLM
            let logits = self.forward_embeddings(&h, &mut cache)?;

            // Sample from last position with penalties
            let last_logits = logits.index((.., -1, ..));
            let token = Self::sample_top_k_p(&last_logits, temperature, top_k, &tokens, presence_penalty)?;
            mlx_rs::transforms::eval([&token])?;
            token_id = token.item();
        }

        // Decode tokens to text
        let token_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        let text = self.tokenizer.decode(&token_ids, true)
            .map_err(|e| Error::Tokenizer(format!("Decoding failed: {}", e)))?;

        // Post-process: remove <think>...</think> tags
        let text = if let Some(think_end) = text.find("</think>") {
            text[think_end + 8..].trim().to_string()
        } else if text.starts_with("<think>") {
            text[7..].trim().to_string()
        } else {
            text
        };

        // Post-process: remove text-level repetition
        // Check if the text contains a repeated block (>20 chars)
        let text = Self::remove_text_repetition(&text);

        // Post-process: cut off meta-commentary that the model sometimes generates
        // after the actual transcription (e.g., "这段文字看起来像是...")
        let text = Self::remove_meta_commentary(&text);

        // Post-process: add punctuation if model is available
        #[cfg(feature = "punctuation")]
        let text = if let Some(ref mut punc) = self.punc_model {
            match punc.punctuate(&text) {
                Ok(punctuated) => punctuated,
                Err(_) => text,  // silently fall back to unpunctuated
            }
        } else {
            text
        };

        Ok(text)
    }

    /// Remove text-level repetition from generated output.
    ///
    /// Checks if the text contains a repeated block (min 20 chars).
    /// If the second half of the text starts with a prefix of the first half,
    /// truncate to keep only the first occurrence.
    fn remove_text_repetition(text: &str) -> String {
        let text = text.trim();
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        if len < MIN_TEXT_LEN_FOR_TEXT_REPETITION {
            return text.to_string();
        }

        // Try to find a repeated block starting from the middle
        // Check if text[i..] starts with text[0..i] for i around len/2
        for start in (len / 3)..=(len * 2 / 3) {
            let suffix: String = chars[start..].iter().collect();
            let prefix: String = chars[..suffix.len().min(chars.len() - start)].iter().collect();
            // Check if suffix starts with at least MIN_COMMON_CHARS of prefix
            let common_len = suffix.chars().zip(prefix.chars())
                .take_while(|(a, b)| a == b)
                .count();
            if common_len >= MIN_COMMON_CHARS && common_len as f64 / (len - start) as f64 > 0.8 {
                // The repeated block starts at `start`, keep only text[..start]
                let result: String = chars[..start].iter().collect();
                return result.trim().to_string();
            }
        }

        text.to_string()
    }

    /// Remove meta-commentary that the model sometimes generates after transcription.
    ///
    /// The 8-bit quantized Qwen3 model sometimes enters analysis mode after generating
    /// the actual transcription, producing text like "这段文字看起来像是..." or
    /// "以下是我对这段文字的..." which is not part of the audio content.
    fn remove_meta_commentary(text: &str) -> String {
        // Chinese meta-commentary markers
        let zh_markers = [
            "这段文字看起来",
            "这段话看起来",
            "这段话有点长",
            "这段话的中文",
            "这段文字主要",
            "这段文字似乎",
            "这段文字确实",
            "这段文字中有",
            "这段文字是",
            "看起来你提供的",
            "看起来这段",
            "看起来作者",
            "以下是我对",
            "以下是整理",
            "以下是翻译",
            "以下是可能",
            "不过，这段",
            "不过，原文",
            "不过，内容",
            "不过，根据",
            "可能的正确",
            "可能的原文",
            "可能的中文",
            "可能需要更多",
            "如果需要",
            "如果你有",
            "如果你需要",
            "如果你能",
            "整理后的版本",
            "转写后的中文",
            "翻译内容：",
            "我来帮你整理",
            "我将尝试",
            "我尝试解析",
            "我尝试将",
            "我需要先",
            "我需要将",
            "总结来说",
            "总结：",
            "总的来说",
            "可能涉及",
            "可能是在",
            "首先，我需要",
            "首先，原文",
            "首先，这段",
            "接下来，我",
            "**转写",
            "**翻译",
            "**整理",
            "**解释",
            "---\n",
            "用户输入",
            "用户的问题",
            "用户可能",
            "需要进一步",
            "按照要求",
            "以下是转写",
            "转写后的文字",
            "转写后的中文",
            "转写成中文",
            "请转写成中文",
            "当转写成中文",
            "根据用户提供",
            "根据用户输入",
            "不进行任何分析",
            "不进行分析",
            "逻辑上，为什么",
            "逻辑上，这家",
        ];

        // English meta-commentary markers
        let en_markers = [
            "Here's a translation",
            "This text is a bit",
            "This seems like",
            "This is a",
            "The text you provided",
            "The text provided",
            "The main idea",
            "When translating",
            "Hmm, this seems",
            "First, I'll",
            "Let me try",
            "I need to",
            "I think",
            "The user",
            "Okay, let's see",
            "It seems like",
            "Putting it all together",
            "So, when we talk",
            "But the user",
        ];

        let text_trimmed = text.trim();

        // Strip suffix instruction text that sometimes leaks into output
        let text_trimmed = text_trimmed
            .replace("只输出转写文字，不要分析或解释：", "")
            .replace("只输出转写文字，不要分析或解释", "")
            .replace("只输出转写文字", "");
        let text_trimmed = text_trimmed.trim();

        // Markers that indicate meta-commentary even at the start of output
        // (these are never part of legitimate transcription)
        let strict_markers = [
            "按照要求",
            "以下是转写",
            "转写后的文字",
            "转写后的中文",
            "转写成中文",
            "请转写成中文",
            "当转写成中文",
            "根据用户提供",
            "根据用户输入",
            "不进行任何分析",
            "不进行分析",
            "逻辑上，为什么",
            "逻辑上，这家",
        ];

        // Find the earliest meta-commentary marker
        let mut earliest_pos = text_trimmed.len();

        // Strict markers: cut at any position (pos >= 0)
        for marker in &strict_markers {
            if let Some(pos) = text_trimmed.find(marker) {
                if pos < earliest_pos {
                    earliest_pos = pos;
                }
            }
        }

        // Regular markers: only cut if not at the very start (pos > 10)
        for marker in zh_markers.iter().chain(en_markers.iter()) {
            if let Some(pos) = text_trimmed.find(marker) {
                if pos > 10 && pos < earliest_pos {
                    earliest_pos = pos;
                }
            }
        }

        if earliest_pos < text_trimmed.len() {
            return text_trimmed[..earliest_pos].trim().to_string();
        }

        text_trimmed.to_string()
    }

    /// Sample with top-k filtering and presence penalty
    fn sample_top_k_p(
        logits: &Array,
        temperature: f32,
        top_k: usize,
        generated_tokens: &[i32],
        presence_penalty: f32,
    ) -> Result<Array> {
        if temperature == 0.0 && (presence_penalty == 0.0 || generated_tokens.is_empty()) {
            return sample(logits, 0.0)
                .map_err(|e| Error::ModelLoad(format!("Sample failed: {:?}", e)));
        }

        let shape = logits.shape();
        let vocab_size = *shape.last().unwrap() as usize;

        // Apply presence penalty
        let mut modified = logits.clone();
        if presence_penalty > 0.0 && !generated_tokens.is_empty() {
            let mut penalty_data = vec![0.0f32; vocab_size];
            for &tok in generated_tokens {
                if (tok as usize) < vocab_size {
                    penalty_data[tok as usize] = presence_penalty;
                }
            }
            let penalty = Array::from_slice(&penalty_data, &[1, vocab_size as i32]);
            modified = mlx_rs::ops::subtract(&modified, &penalty)
                .map_err(|e| Error::ModelLoad(format!("Penalty failed: {:?}", e)))?;
        }

        // Scale by temperature
        modified = modified.multiply(mlx_rs::array!(1.0 / temperature))
            .map_err(|e| Error::ModelLoad(format!("Scale failed: {:?}", e)))?;

        // Top-k filtering
        if top_k > 0 && top_k < vocab_size {
            let topk_vals = mlx_rs::ops::indexing::topk_axis_device(
                &modified, top_k as i32, -1, mlx_rs::StreamOrDevice::default()
            ).map_err(|e| Error::ModelLoad(format!("Top-k failed: {:?}", e)))?;
            let threshold = topk_vals.index((.., (top_k as i32 - 1)));
            let threshold = threshold.reshape(&[1, 1])
                .map_err(|e| Error::ModelLoad(format!("Reshape failed: {:?}", e)))?;
            let mask = modified.ge(&threshold)
                .map_err(|e| Error::ModelLoad(format!("Compare failed: {:?}", e)))?;
            let neg_inf = mlx_rs::array!(f32::NEG_INFINITY);
            modified = mlx_rs::ops::r#where(&mask, &modified, &neg_inf)
                .map_err(|e| Error::ModelLoad(format!("Where failed: {:?}", e)))?;
        }

        sample(&modified, 1.0)
            .map_err(|e| Error::ModelLoad(format!("Sample failed: {:?}", e)))
    }

    /// Generate text from prompt using qwen3-mlx
    pub fn generate_text(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| Error::Tokenizer(format!("Tokenization failed: {}", e)))?;

        let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);

        let mut cache: Vec<Option<KVCache>> = Vec::new();
        let mut generated_tokens: Vec<u32> = Vec::new();

        let generator = Generate::<KVCache>::new(&mut self.llm, &mut cache, 0.0, &prompt_tokens);

        for token_result in generator.take(max_tokens) {
            let token = token_result
                .map_err(|e| Error::ModelLoad(format!("Generation failed: {:?}", e)))?;
            let token_id: u32 = token.item();

            // Check for EOS
            if token_id == self.speech_tokens.eos || token_id == self.speech_tokens.im_end {
                break;
            }

            generated_tokens.push(token_id);
        }

        let text = self.tokenizer.decode(&generated_tokens, true)
            .map_err(|e| Error::Tokenizer(format!("Decoding failed: {}", e)))?;

        Ok(text)
    }

    /// Simple text completion (for testing)
    pub fn complete(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String> {
        let encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| Error::Tokenizer(format!("Tokenization failed: {}", e)))?;

        let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);

        let mut cache: Vec<Option<KVCache>> = Vec::new();
        let mut generated_tokens: Vec<u32> = Vec::new();

        let generator = Generate::<KVCache>::new(&mut self.llm, &mut cache, temperature, &prompt_tokens);

        for token_result in generator.take(max_tokens) {
            let token = token_result
                .map_err(|e| Error::ModelLoad(format!("Generation failed: {:?}", e)))?;
            let token_id: u32 = token.item();

            if token_id == self.speech_tokens.eos || token_id == self.speech_tokens.im_end {
                break;
            }

            generated_tokens.push(token_id);
        }

        let text = self.tokenizer.decode(&generated_tokens, true)
            .map_err(|e| Error::Tokenizer(format!("Decoding failed: {}", e)))?;

        Ok(text)
    }
}
