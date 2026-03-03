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

// ── Repetition detection thresholds ──────────────────────────────────────────
/// Minimum block size (in chars) to qualify as a repeated block.
const MIN_REPEAT_BLOCK_CHARS: usize = 30;
/// Minimum text length (in chars) before checking for repeated blocks.
const MIN_TEXT_LEN_FOR_REPEAT_DETECTION: usize = 60;
/// Minimum common prefix length to consider as text-level repetition.
const MIN_COMMON_CHARS: usize = 20;
/// Minimum text length (in chars) before checking for text-level repetition.
const MIN_TEXT_LEN_FOR_TEXT_REPETITION: usize = 40;

// ── Hallucination detection patterns (data-driven, add new patterns here) ───
//
// These patterns indicate the 8-bit quantized model has mode-collapsed into
// "helpful AI assistant" behavior instead of transcribing speech. Adding new
// patterns here is all that's needed to handle new hallucination types.

/// Phrases that indicate a hallucination block when found at a sentence boundary.
/// If text after a period/sentence boundary starts with any of these, everything
/// from that boundary onward is excised.
const HALLUCINATION_MARKERS: &[&str] = &[
    "你好，我是",
    "我是一个",
    "作为一个",
    "好的，",
    "首先",
    "以下是",
    "根据您",
    "请问",
    "您好",
    "让我",
    "需要注意",
    "这段音频",
    "这段语音",
    "这是一段",
    "总结一下",
    "希望这",
    "如果您",
    "感谢您",
    "很高兴",
    "I'm an AI",
    "I am an AI",
    "As an AI",
    "Let me ",
    "Here is",
    "Here's ",
    "Based on",
    "The audio",
    "The speech",
    "In summary",
];

/// Phrases that indicate meta-commentary (model talking about itself/the task
/// rather than transcribing). If found anywhere in the text, the sentence
/// containing it is removed.
const META_COMMENTARY_MARKERS: &[&str] = &[
    "AI助手",
    "AI语言模型",
    "语言模型",
    "人工智能",
    "自我介绍",
    "我来帮",
    "我来为您",
    "我来分析",
    "请告诉我",
    "有什么可以帮",
    "处理用户",
    "用户的查询",
    "用户的请求",
    "用户提供",
    "用户提到",
    "用户的问题",
    "这段话的转写",
    "这段音频的转写",
    "转写结果",
    "chain of thought",
    "chain-of-thought",
    "step by step",
];

/// Configuration for audio transcription/translation.
///
/// Controls generation parameters, quality heuristics, and
/// long-form chunking behavior. Use `Default::default()` for robust
/// real-world transcription, or `TranscribeConfig::greedy()` for
/// benchmark-optimal CER on clean audio.
#[derive(Debug, Clone)]
pub struct TranscribeConfig {
    /// Sampling temperature (0.0 = greedy, 0.6 = Qwen3 recommended).
    /// Greedy gives best CER on clean benchmarks but causes repetition
    /// on real-world audio. Default: 0.6.
    pub temperature: f32,
    /// Top-k sampling (0 = disabled). Default: 20.
    pub top_k: usize,
    /// Presence penalty for previously generated tokens. Default: 1.0.
    pub presence_penalty: f32,

    /// Tokens generated per second of audio (Chinese). Default: 5.0.
    pub tokens_per_sec_chinese: f32,
    /// Tokens generated per second of audio (English). Default: 8.0.
    pub tokens_per_sec_english: f32,
    /// Minimum tokens to generate regardless of duration. Default: 30.
    pub min_tokens: usize,
    /// Maximum tokens to generate regardless of duration. Default: 400.
    pub max_tokens: usize,

    /// Enable entropy monitoring to detect degenerate generation. Default: true.
    pub entropy_monitoring: bool,
    /// Shannon entropy threshold below which generation is considered degenerate.
    /// Default: 0.05.
    pub entropy_threshold: f32,
    /// Consecutive low-entropy steps before breaking. Default: 15.
    pub entropy_window: usize,

    /// Enable embedding collapse detection. Default: true.
    pub detect_embedding_collapse: bool,
    /// Cosine similarity threshold for embedding collapse. Default: 0.95.
    pub embedding_collapse_threshold: f32,

    /// Enable VAD to skip silent chunks in long-form processing. Default: true.
    pub vad_enabled: bool,
    /// Silence threshold in dB for VAD. Default: -40.0.
    pub silence_threshold_db: f32,
    /// Overlap in seconds between adjacent chunks. Default: 2.0.
    pub chunk_overlap_secs: f32,
    /// Enable fuzzy cross-chunk deduplication. Default: true.
    pub fuzzy_dedup: bool,
}

impl Default for TranscribeConfig {
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_k: 20,
            presence_penalty: 1.0,
            tokens_per_sec_chinese: 5.0,
            tokens_per_sec_english: 8.0,
            min_tokens: 30,
            max_tokens: 400,
            entropy_monitoring: true,
            entropy_threshold: 0.05,
            entropy_window: 15,
            detect_embedding_collapse: true,
            embedding_collapse_threshold: 0.95,
            vad_enabled: true,
            silence_threshold_db: -40.0,
            chunk_overlap_secs: 2.0,
            fuzzy_dedup: true,
        }
    }
}

impl TranscribeConfig {
    /// Greedy decoding config (best CER on clean benchmark audio like AISHELL).
    /// Disables sampling heuristics for maximum accuracy.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            presence_penalty: 0.0,
            entropy_monitoring: false,
            detect_embedding_collapse: false,
            vad_enabled: false,
            chunk_overlap_secs: 0.0,
            fuzzy_dedup: false,
            ..Default::default()
        }
    }
}

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

    /// Transcribe audio file with custom config
    pub fn transcribe_with_config(&mut self, audio_path: &str, config: &TranscribeConfig) -> Result<String> {
        let (samples, sample_rate) = load_wav(audio_path)?;
        self.transcribe_samples_with_config(&samples, sample_rate, "语音转写成中文：", config)
    }

    /// Transcribe long audio with custom config (includes VAD, overlap, fuzzy dedup)
    pub fn transcribe_long_with_config(
        &mut self,
        audio_path: &str,
        chunk_secs: f32,
        config: &TranscribeConfig,
    ) -> Result<String> {
        let (samples, sample_rate) = load_wav(audio_path)?;
        self.transcribe_long_samples_with_config(&samples, sample_rate, chunk_secs, config)
    }

    /// Transcribe long audio samples with custom config (includes VAD, overlap, fuzzy dedup)
    pub fn transcribe_long_samples_with_config(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        chunk_secs: f32,
        config: &TranscribeConfig,
    ) -> Result<String> {
        self.process_long_samples_with_config(samples, sample_rate, chunk_secs, "语音转写成中文：", "", config)
    }

    /// Translate long audio to English with custom config
    pub fn translate_long_with_config(
        &mut self,
        audio_path: &str,
        chunk_secs: f32,
        config: &TranscribeConfig,
    ) -> Result<String> {
        let (samples, sample_rate) = load_wav(audio_path)?;
        self.translate_long_samples_with_config(&samples, sample_rate, chunk_secs, config)
    }

    /// Translate long audio samples to English with custom config
    pub fn translate_long_samples_with_config(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        chunk_secs: f32,
        config: &TranscribeConfig,
    ) -> Result<String> {
        self.process_long_samples_with_config(samples, sample_rate, chunk_secs, "Translate the speech to English:", " ", config)
    }

    /// Process long audio samples with arbitrary prompt by splitting into chunks.
    /// Uses default greedy config for backward compatibility.
    fn process_long_samples(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        chunk_secs: f32,
        prompt: &str,
        separator: &str,
    ) -> Result<String> {
        let mut config = TranscribeConfig::greedy();
        config.chunk_overlap_secs = 0.0;
        config.vad_enabled = false;
        config.fuzzy_dedup = false;
        self.process_long_samples_with_config(samples, sample_rate, chunk_secs, prompt, separator, &config)
    }

    /// Process long audio samples with full config (VAD, overlap, fuzzy dedup).
    fn process_long_samples_with_config(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        chunk_secs: f32,
        prompt: &str,
        separator: &str,
        config: &TranscribeConfig,
    ) -> Result<String> {
        let chunk_size = (chunk_secs * sample_rate as f32) as usize;
        let overlap_size = (config.chunk_overlap_secs * sample_rate as f32) as usize;
        let step_size = if overlap_size >= chunk_size {
            chunk_size // no overlap if overlap >= chunk
        } else {
            chunk_size - overlap_size
        };

        // Build chunk boundaries
        let mut chunk_ranges: Vec<(usize, usize)> = Vec::new();
        let mut start = 0usize;
        while start < samples.len() {
            let end = (start + chunk_size).min(samples.len());
            if end - start < (sample_rate as usize / 10) {
                break; // skip chunks shorter than 100ms
            }
            chunk_ranges.push((start, end));
            start += step_size;
        }

        let total_chunks = chunk_ranges.len();
        let mut results: Vec<String> = Vec::new();

        for (i, &(start, end)) in chunk_ranges.iter().enumerate() {
            let chunk = &samples[start..end];

            // VAD: skip silent chunks
            if config.vad_enabled && crate::audio::is_silent(chunk, config.silence_threshold_db) {
                eprint!("\r  Chunk {}/{} (silent, skipped)", i + 1, total_chunks);
                continue;
            }

            eprint!("\r  Chunk {}/{}", i + 1, total_chunks);
            let text = self.transcribe_samples_with_config(chunk, sample_rate, prompt, config)?;
            if !text.is_empty() {
                // Trim overlap with previous chunk's text
                if overlap_size > 0 && !results.is_empty() {
                    let trimmed = Self::trim_overlap(results.last().unwrap(), &text);
                    if !trimmed.is_empty() {
                        results.push(trimmed);
                    }
                } else {
                    results.push(text);
                }
            }
        }
        eprintln!();

        let joined = results.join(separator);

        // Final pass: remove repeated blocks (fuzzy or exact), then excise hallucination
        let deduped = if config.fuzzy_dedup {
            Self::remove_repeated_blocks_fuzzy(&joined)
        } else {
            Self::remove_repeated_blocks(&joined)
        };
        let cleaned = Self::excise_hallucination_blocks(&deduped);
        let cleaned = Self::remove_meta_commentary(&cleaned);
        Ok(cleaned)
    }

    /// Trim overlapping text between adjacent chunk transcriptions.
    /// Finds the longest suffix of `prev` that matches a prefix of `current`.
    fn trim_overlap(prev: &str, current: &str) -> String {
        if prev.is_empty() || current.is_empty() {
            return current.to_string();
        }
        let prev_chars: Vec<char> = prev.chars().collect();
        let curr_chars: Vec<char> = current.chars().collect();
        let max_check = prev_chars.len().min(curr_chars.len()).min(100);
        let mut best_overlap = 0;
        for len in (5..=max_check).rev() {
            let suffix = &prev_chars[prev_chars.len() - len..];
            let prefix = &curr_chars[..len];
            if suffix == prefix {
                best_overlap = len;
                break;
            }
        }
        if best_overlap > 0 {
            curr_chars[best_overlap..].iter().collect()
        } else {
            current.to_string()
        }
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

    /// Check if a character is CJK/ASCII punctuation.
    fn is_dedup_punctuation(c: char) -> bool {
        matches!(c,
            '。' | '，' | '、' | '？' | '！' | '；' | '：' | '\u{201C}' | '\u{201D}' |
            '\u{2018}' | '\u{2019}' | '【' | '】' | '《' | '》' | '（' | '）' | '—' |
            '…' | '·' | '～' | '「' | '」' |
            '.' | ',' | '?' | '!' | ';' | ':' | '"' | '\'' | '(' | ')' |
            '[' | ']' | '{' | '}' | '-' | '/' | '~'
        )
    }

    /// Remove repeated blocks with fuzzy matching (strips punctuation before matching).
    ///
    /// Handles the case where adjacent chunks produce identical text
    /// but with different punctuation (e.g., "你好，" vs "你好。").
    fn remove_repeated_blocks_fuzzy(text: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        if len < MIN_TEXT_LEN_FOR_REPEAT_DETECTION {
            return text.to_string();
        }

        // Build normalized text (no punctuation) with index mapping back to original
        let mut norm_to_orig: Vec<usize> = Vec::new();
        for (i, &c) in chars.iter().enumerate() {
            if !Self::is_dedup_punctuation(c) && !c.is_whitespace() {
                norm_to_orig.push(i);
            }
        }
        let norm_chars: Vec<char> = norm_to_orig.iter().map(|&i| chars[i]).collect();
        let norm_len = norm_chars.len();
        let min_block = MIN_REPEAT_BLOCK_CHARS / 2; // lower threshold for normalized

        if norm_len < min_block * 2 {
            return text.to_string();
        }

        // Same algorithm as remove_repeated_blocks but on normalized chars
        let mut skip_ranges: Vec<(usize, usize)> = Vec::new();
        let mut pos = min_block;
        while pos < norm_len {
            let end = (pos + min_block).min(norm_len);
            let block: String = norm_chars[pos..end].iter().collect();
            let prefix: String = norm_chars[..pos].iter().collect();
            if let Some(first_pos) = prefix.find(&block) {
                let mut match_len = min_block;
                while pos + match_len < norm_len
                    && first_pos + match_len < pos
                    && norm_chars[first_pos + match_len] == norm_chars[pos + match_len]
                {
                    match_len += 1;
                }
                if match_len >= min_block {
                    // Map back to original char indices
                    let orig_start = norm_to_orig[pos];
                    let orig_end = if pos + match_len < norm_to_orig.len() {
                        // Include any trailing punctuation after the matched region
                        let raw_end = norm_to_orig[pos + match_len - 1] + 1;
                        // Extend past trailing punctuation/whitespace
                        let mut ext = raw_end;
                        while ext < len && (Self::is_dedup_punctuation(chars[ext]) || chars[ext].is_whitespace()) {
                            ext += 1;
                        }
                        ext
                    } else {
                        len
                    };
                    skip_ranges.push((orig_start, orig_end));
                    pos += match_len;
                    continue;
                }
            }
            pos += 1;
        }

        if skip_ranges.is_empty() {
            return text.to_string();
        }

        // Build result, skipping duplicate ranges
        let mut result = String::new();
        let mut i = 0;
        for (start, end) in &skip_ranges {
            if i < *start {
                let segment: String = chars[i..*start].iter().collect();
                result.push_str(&segment);
            }
            if *end > i {
                i = *end;
            }
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
        self.transcribe_samples_with_config(samples, sample_rate, prompt, &TranscribeConfig::greedy())
    }

    /// Transcribe/translate audio with a custom prompt and config.
    pub fn transcribe_samples_with_config(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        prompt: &str,
        config: &TranscribeConfig,
    ) -> Result<String> {
        // Resample to 16kHz if needed
        let samples = if sample_rate != 16000 {
            resample(samples, sample_rate, 16000)?
        } else {
            samples.to_vec()
        };

        // Compute duration-proportional max_tokens to prevent hallucination
        let duration_secs = samples.len() as f32 / 16000.0;
        let is_chinese = prompt.contains("中文");
        let tokens_per_sec = if is_chinese { config.tokens_per_sec_chinese } else { config.tokens_per_sec_english };
        let max_tokens = ((duration_secs * tokens_per_sec) as usize)
            .max(config.min_tokens)
            .min(config.max_tokens);

        // Compute mel spectrogram
        let mel = compute_mel_spectrogram(&samples, &self.audio_config)?;

        // Apply LFR (Low Frame Rate) transformation
        let mel_lfr = apply_lfr(&mel, 7, 6)?;

        // Encode with SenseVoice
        let encoder_out = self.encoder.forward(&mel_lfr)?;

        // Project to Qwen4B embedding space
        let adapted = self.adaptor.forward(&encoder_out)?;

        // Generate text with duration-proportional limit
        let text = self.generate_from_audio_features(&adapted, prompt, max_tokens, config)?;

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
    /// Embedding layout:
    ///   [prompt_tokens | audio_features | suffix_tokens]
    ///
    /// The suffix controls model behavior at generation time. For robust mode,
    /// it includes `<think>\n\n</think>\n\n` to bypass Qwen3's thinking mode
    /// and prevent chain-of-thought hallucination, followed by an instruction
    /// to output only transcription text.
    fn generate_from_audio_features(
        &mut self,
        audio_features: &Array,
        prompt: &str,
        max_tokens: usize,
        config: &TranscribeConfig,
    ) -> Result<String> {
        // Embedding collapse detection: skip chunk if embeddings are degenerate
        if config.detect_embedding_collapse {
            if Self::check_embedding_collapse(audio_features, config.embedding_collapse_threshold) {
                return Ok(String::new());
            }
        }

        // 1. Task prompt (e.g., "语音转写成中文：")
        let prompt_embed = self.encode_text_to_embeddings(prompt)?;

        // 2. Suffix: determines model behavior at generation time
        //    Chinese: anti-hallucination instruction suffix
        //    English: no-think prefix to bypass Qwen3 thinking mode
        let is_chinese = prompt.contains("中文");
        let suffix_text = if is_chinese {
            "只输出转写文字，不要分析或解释："
        } else {
            "<think>\n\n</think>\n\n"
        };
        let suffix_embed = self.encode_text_to_embeddings(suffix_text)?;

        // Concatenate: [prompt | audio | suffix]
        let combined = mlx_rs::ops::concatenate_axis(
            &[&prompt_embed, audio_features, &suffix_embed],
            1,
        ).map_err(|e| Error::ModelLoad(format!("Concat failed: {:?}", e)))?;
        mlx_rs::transforms::eval([&combined])?;

        // Generate tokens from combined embeddings
        self.generate_from_embeddings(&combined, max_tokens, config)
    }

    /// Encode a text string into token embeddings [1, seq_len, hidden_dim].
    fn encode_text_to_embeddings(&mut self, text: &str) -> Result<Array> {
        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| Error::Tokenizer(format!("Tokenization failed: {}", e)))?;
        let token_ids: Vec<i32> = encoding.get_ids().iter()
            .map(|&t| t as i32)
            .collect();
        let token_array = Array::from_slice(&token_ids, &[1, token_ids.len() as i32]);
        let embed = self.get_token_embeddings(&token_array)?;
        mlx_rs::transforms::eval([&embed])?;
        Ok(embed)
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
    /// Uses configurable sampling with n-gram repetition detection,
    /// entropy monitoring, and text-level dedup post-processing.
    fn generate_from_embeddings(
        &mut self,
        embeddings: &Array,
        max_tokens: usize,
        config: &TranscribeConfig,
    ) -> Result<String> {
        let temperature = config.temperature;
        let top_k = config.top_k;
        let presence_penalty = config.presence_penalty;

        let mut cache: Vec<Option<KVCache>> = Vec::new();
        let mut tokens: Vec<i32> = Vec::new();
        let mut low_entropy_count: usize = 0;

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

            // Entropy monitoring: detect degenerate generation
            if config.entropy_monitoring {
                let entropy = Self::compute_entropy(&last_logits);
                if entropy < config.entropy_threshold {
                    low_entropy_count += 1;
                    if low_entropy_count >= config.entropy_window {
                        break;
                    }
                } else {
                    low_entropy_count = 0;
                }
            }

            let token = Self::sample_top_k_p(&last_logits, temperature, top_k, &tokens, presence_penalty)?;
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
        let text = Self::remove_text_repetition(&text);

        // Post-process: strip leaked suffix, excise hallucination blocks, remove meta-commentary
        let text = Self::strip_leaked_instructions(&text);
        let text = Self::excise_hallucination_blocks(&text);
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

    /// Strip our own injected suffix text that sometimes leaks into the output.
    fn strip_leaked_instructions(text: &str) -> String {
        text.replace("只输出转写文字，不要分析或解释：", "")
            .replace("只输出转写文字，不要分析或解释", "")
            .replace("只输出转写文字", "")
            .replace("输出转写文字，不要分析或解释", "")
            .trim()
            .to_string()
    }

    /// Remove meta-commentary sentences (model talking about itself/the task).
    ///
    /// Uses the `META_COMMENTARY_MARKERS` pattern list. Any sentence containing
    /// a marker is removed. To handle new hallucination types, add patterns to
    /// the const array at the top of this file.
    fn remove_meta_commentary(text: &str) -> String {
        // Split on sentence-ending punctuation, check each sentence
        let mut result = String::new();
        let mut current_sentence = String::new();

        for c in text.chars() {
            current_sentence.push(c);
            if matches!(c, '。' | '！' | '？' | '.' | '!' | '?' | '\n') {
                let has_meta = META_COMMENTARY_MARKERS.iter()
                    .any(|marker| current_sentence.contains(marker));
                if !has_meta {
                    result.push_str(&current_sentence);
                }
                current_sentence.clear();
            }
        }

        // Handle trailing text without sentence-ending punctuation
        if !current_sentence.is_empty() {
            let has_meta = META_COMMENTARY_MARKERS.iter()
                .any(|marker| current_sentence.contains(marker));
            if !has_meta {
                result.push_str(&current_sentence);
            }
        }

        result.trim().to_string()
    }

    /// Excise hallucination blocks from transcription output.
    ///
    /// When the 8-bit quantized model mode-collapses, it generates "helpful AI
    /// assistant" text mid-transcription. This function scans for sentence
    /// boundaries (。！？.!?) and checks if the text after a boundary starts
    /// with any `HALLUCINATION_MARKERS` pattern. If so, everything from that
    /// boundary onward is cut.
    ///
    /// To handle new hallucination types, add patterns to the const array at
    /// the top of this file.
    fn excise_hallucination_blocks(text: &str) -> String {
        let text = text.trim();
        if text.is_empty() {
            return String::new();
        }

        // Check if the entire text starts with a hallucination marker
        let trimmed = text.trim_start();
        for marker in HALLUCINATION_MARKERS {
            if trimmed.starts_with(marker) {
                return String::new();
            }
        }

        // Scan for sentence boundaries and check what follows
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();

        for i in 0..len {
            if !matches!(chars[i], '。' | '！' | '？' | '.' | '!' | '?') {
                continue;
            }

            // Found sentence boundary — check what follows
            let rest_start = i + 1;
            if rest_start >= len {
                break;
            }

            // Skip whitespace after punctuation
            let rest: String = chars[rest_start..].iter().collect();
            let rest_trimmed = rest.trim_start();

            for marker in HALLUCINATION_MARKERS {
                if rest_trimmed.starts_with(marker) {
                    // Cut everything from this boundary onward
                    let result: String = chars[..=i].iter().collect();
                    return result.trim().to_string();
                }
            }
        }

        text.to_string()
    }

    /// Compute Shannon entropy of logits distribution.
    /// Returns f32::MAX on error so generation continues safely.
    fn compute_entropy(logits: &Array) -> f32 {
        let flat = match logits.reshape(&[-1]) {
            Ok(f) => f,
            Err(_) => return f32::MAX,
        };
        if mlx_rs::transforms::eval([&flat]).is_err() {
            return f32::MAX;
        }
        let vals: &[f32] = flat.as_slice();
        if vals.is_empty() {
            return f32::MAX;
        }
        let max_val = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = vals.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        if sum_exp == 0.0 {
            return 0.0;
        }
        exp_vals.iter()
            .map(|&e| {
                let p = e / sum_exp;
                if p > 0.0 { -p * p.ln() } else { 0.0 }
            })
            .sum()
    }

    /// Check if adaptor output embeddings have collapsed (all audio tokens ≈ same vector).
    /// Returns true if mean pairwise cosine similarity > threshold.
    fn check_embedding_collapse(embeddings: &Array, threshold: f32) -> bool {
        if mlx_rs::transforms::eval([embeddings]).is_err() {
            return false;
        }
        let shape = embeddings.shape();
        if shape.len() < 3 || (shape[1] as usize) < 4 {
            return false;
        }
        let seq_len = shape[1] as usize;
        let indices = [0usize, seq_len / 3, 2 * seq_len / 3, seq_len - 1];
        let embeds: Vec<Vec<f32>> = indices.iter().filter_map(|&i| {
            let e = embeddings.index((.., i as i32, ..));
            mlx_rs::transforms::eval([&e]).ok()?;
            let f = e.reshape(&[-1]).ok()?;
            mlx_rs::transforms::eval([&f]).ok()?;
            Some(f.as_slice::<f32>().to_vec())
        }).collect();
        if embeds.len() < 2 {
            return false;
        }
        let mut total_cos = 0.0f32;
        let mut count = 0;
        for i in 0..embeds.len() {
            for j in (i + 1)..embeds.len() {
                let dot: f32 = embeds[i].iter().zip(embeds[j].iter()).map(|(a, b)| a * b).sum();
                let na: f32 = embeds[i].iter().map(|v| v * v).sum::<f32>().sqrt();
                let nb: f32 = embeds[j].iter().map(|v| v * v).sum::<f32>().sqrt();
                if na > 0.0 && nb > 0.0 {
                    total_cos += dot / (na * nb);
                    count += 1;
                }
            }
        }
        if count == 0 {
            return false;
        }
        total_cos / count as f32 > threshold
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
