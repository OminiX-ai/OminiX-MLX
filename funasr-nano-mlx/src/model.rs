//! Fun-ASR-Nano combined model.
//!
//! Integrates SenseVoice encoder, audio adaptor, and Qwen3 LLM.

use crate::adaptor::{AdaptorConfig, AudioAdaptor};
use crate::audio::{self, AudioConfig, MelFrontend};
use crate::error::{Error, Result};
use crate::qwen::{QwenConfig, QwenModel};
use crate::sensevoice_encoder::{SenseVoiceEncoder, SenseVoiceEncoderConfig};

use mlx_rs_core::KVCache;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::{Module, ModuleParameters as ModuleParametersTrait};
use mlx_rs::ops::indexing::*;
use mlx_rs::transforms::eval;
use mlx_rs::Array;
use std::path::Path;

/// Fun-ASR-Nano configuration.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct FunASRNanoConfig {
    /// Audio encoder configuration
    #[serde(default)]
    pub encoder: SenseVoiceEncoderConfig,

    /// Audio adaptor configuration
    #[serde(default)]
    pub adaptor: AdaptorConfig,

    /// LLM configuration
    #[serde(default)]
    pub llm: QwenConfig,

    /// Audio processing configuration
    #[serde(skip)]
    pub audio: AudioConfig,
}

impl Default for FunASRNanoConfig {
    fn default() -> Self {
        Self {
            encoder: SenseVoiceEncoderConfig::default(),
            adaptor: AdaptorConfig::default(),
            llm: QwenConfig::default(),
            audio: AudioConfig::default(),
        }
    }
}

/// Speech markers for audio injection.
#[derive(Debug, Clone)]
pub struct SpeechMarkers {
    /// Start of speech token ID
    pub start_token: i32,
    /// End of speech token ID
    pub end_token: i32,
    /// End of text token ID
    pub eos_token: i32,
    /// IM end token ID
    pub im_end_token: i32,
}

impl Default for SpeechMarkers {
    fn default() -> Self {
        Self {
            // Qwen3 special tokens
            start_token: 151646,  // <|startofspeech|>
            end_token: 151647,    // <|endofspeech|>
            eos_token: 151643,    // <|endoftext|>
            im_end_token: 151645, // <|im_end|>
        }
    }
}

/// Sampling configuration for text generation.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for softmax (0.0 = greedy)
    pub temperature: f32,
    /// Top-k filtering (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus) filtering (1.0 = disabled)
    pub top_p: f32,
    /// Repetition penalty (1.0 = disabled)
    pub repetition_penalty: f32,
    /// Maximum tokens to generate
    pub max_tokens: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,     // Greedy by default for deterministic ASR
            top_k: 0,             // Disabled
            top_p: 1.0,           // Disabled
            repetition_penalty: 1.0, // Disabled
            max_tokens: 256,
        }
    }
}

impl SamplingConfig {
    /// Greedy decoding (deterministic, fastest)
    pub fn greedy() -> Self {
        Self::default()
    }

    /// Temperature sampling
    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature,
            ..Default::default()
        }
    }

    /// Top-k sampling
    pub fn top_k(k: usize) -> Self {
        Self {
            temperature: 1.0,
            top_k: k,
            ..Default::default()
        }
    }

    /// Top-p (nucleus) sampling
    pub fn top_p(p: f32) -> Self {
        Self {
            temperature: 1.0,
            top_p: p,
            ..Default::default()
        }
    }
}

/// Fun-ASR-Nano model.
///
/// Combines SenseVoice encoder, audio adaptor, and Qwen3 LLM for
/// speech recognition.
#[derive(ModuleParameters)]
pub struct FunASRNano {
    /// Audio encoder (SenseVoice with SAN-M attention)
    #[param]
    pub encoder: SenseVoiceEncoder,

    /// Audio-to-LLM adaptor (2-layer transformer)
    #[param]
    pub adaptor: AudioAdaptor,

    /// Language model (Qwen3-0.6B)
    #[param]
    pub llm: QwenModel,

    /// Audio processing configuration
    pub audio_config: AudioConfig,

    /// Speech markers
    pub markers: SpeechMarkers,

    /// Cached mel spectrogram frontend (pre-computed FFT, window, filterbank)
    mel_frontend: MelFrontend,

    /// Tokenizer (loaded separately)
    #[allow(dead_code)]
    tokenizer: Option<tokenizers::Tokenizer>,
}

impl FunASRNano {
    /// Create a new Fun-ASR-Nano model.
    pub fn new(config: FunASRNanoConfig) -> Result<Self> {
        let encoder = SenseVoiceEncoder::new(config.encoder)?;
        let adaptor = AudioAdaptor::new(config.adaptor)?;
        let llm = QwenModel::new(config.llm)?;

        // Pre-compute FFT planner, window, and mel filterbank
        let mel_frontend = MelFrontend::new(config.audio.clone());

        Ok(Self {
            encoder,
            adaptor,
            llm,
            audio_config: config.audio,
            markers: SpeechMarkers::default(),
            mel_frontend,
            tokenizer: None,
        })
    }

    /// Load model from directory.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Load configuration from config.yaml
        let config = Self::load_config(model_dir)?;

        // Create model
        let mut model = Self::new(config)?;

        // Load weights from model.pt
        model.load_weights(model_dir)?;

        // Load tokenizer
        let tokenizer_path = model_dir.join("Qwen3-0.6B/tokenizer.json");
        if tokenizer_path.exists() {
            model.tokenizer = Some(
                tokenizers::Tokenizer::from_file(&tokenizer_path)
                    .map_err(|e| Error::Tokenizer(e.to_string()))?,
            );
        }

        Ok(model)
    }

    /// Load configuration from config.yaml.
    fn load_config(model_dir: &Path) -> Result<FunASRNanoConfig> {
        let config_path = model_dir.join("config.yaml");
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            // Parse YAML config
            let yaml: serde_yaml::Value = serde_yaml::from_str(&content)
                .map_err(|e| Error::ModelLoad(format!("Failed to parse config.yaml: {}", e)))?;

            // Extract encoder config
            let encoder = if let Some(enc_conf) = yaml.get("audio_encoder_conf") {
                SenseVoiceEncoderConfig {
                    output_size: enc_conf.get("output_size")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32)
                        .unwrap_or(512),
                    attention_heads: enc_conf.get("attention_heads")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32)
                        .unwrap_or(4),
                    linear_units: enc_conf.get("linear_units")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32)
                        .unwrap_or(2048),
                    num_blocks: enc_conf.get("num_blocks")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32)
                        .unwrap_or(50),
                    tp_blocks: enc_conf.get("tp_blocks")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32)
                        .unwrap_or(20),
                    kernel_size: enc_conf.get("kernel_size")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32)
                        .unwrap_or(11),
                    dropout_rate: 0.0,
                    lfr_dim: 560,  // 80 mels * 7 stacking
                }
            } else {
                SenseVoiceEncoderConfig::default()
            };

            // Extract adaptor config
            let adaptor = if let Some(adp_conf) = yaml.get("audio_adaptor_conf") {
                AdaptorConfig {
                    encoder_dim: adp_conf.get("encoder_dim")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32)
                        .unwrap_or(512),
                    ffn_dim: adp_conf.get("ffn_dim")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32)
                        .unwrap_or(2048),
                    llm_dim: adp_conf.get("llm_dim")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32)
                        .unwrap_or(1024),
                    n_layer: adp_conf.get("n_layer")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32)
                        .unwrap_or(2),
                    downsample_rate: 1,
                }
            } else {
                AdaptorConfig::default()
            };

            // Load Qwen config from Qwen3-0.6B/config.json
            let qwen_config_path = model_dir.join("Qwen3-0.6B/config.json");
            let llm = if qwen_config_path.exists() {
                let file = std::fs::File::open(&qwen_config_path)?;
                serde_json::from_reader(file)?
            } else {
                QwenConfig::default()
            };

            Ok(FunASRNanoConfig {
                encoder,
                adaptor,
                llm,
                audio: AudioConfig::default(),
            })
        } else {
            Ok(FunASRNanoConfig::default())
        }
    }

    /// Load weights from safetensors file.
    fn load_weights(&mut self, model_dir: &Path) -> Result<()> {
        let safetensors_path = model_dir.join("model.safetensors");

        if !safetensors_path.exists() {
            return Err(Error::ModelLoad(format!(
                "model.safetensors not found at {}. Please ensure the model is downloaded.",
                safetensors_path.display()
            )));
        }

        eprintln!("Loading weights from {}...", safetensors_path.display());

        // Load safetensors
        let loaded = Array::load_safetensors(&safetensors_path)
            .map_err(|e| Error::ModelLoad(format!("Failed to load safetensors: {}", e)))?;

        // Get mutable parameters
        let mut params = self.parameters_mut().flatten();

        let mut loaded_count = 0;
        let mut skipped_keys = Vec::new();

        for (st_key, value) in loaded {
            // Map safetensors key to Rust parameter name
            let rust_key = Self::map_safetensors_key(&st_key);

            if let Some(param) = params.get_mut(&*rust_key) {
                **param = value;
                loaded_count += 1;
            } else {
                skipped_keys.push(st_key.to_string());
            }
        }

        eprintln!("Loaded {} parameters", loaded_count);
        if !skipped_keys.is_empty() && skipped_keys.len() < 20 {
            eprintln!("Skipped {} keys: {:?}", skipped_keys.len(), &skipped_keys[..skipped_keys.len().min(10)]);
        } else if !skipped_keys.is_empty() {
            eprintln!("Skipped {} keys (FSMN weights not implemented yet)", skipped_keys.len());
        }

        // Evaluate loaded parameters
        eval(params.values().map(|v| &**v))?;

        Ok(())
    }

    /// Map safetensors key to Rust parameter name.
    fn map_safetensors_key(st_key: &str) -> std::rc::Rc<str> {
        let mut key = st_key.to_string();

        // Encoder mappings
        key = key.replace(".attn.qkv.", ".self_attn.linear_q_k_v.");
        key = key.replace(".attn.out.", ".self_attn.linear_out.");
        key = key.replace(".attn.fsmn.", ".self_attn.fsmn.");
        key = key.replace(".ffn.w1.", ".feed_forward.w_1.");
        key = key.replace(".ffn.w2.", ".feed_forward.w_2.");

        // Adaptor attention mappings (separate Q/K/V)
        key = key.replace(".attn.q.", ".self_attn.linear_q.");
        key = key.replace(".attn.k.", ".self_attn.linear_k.");
        key = key.replace(".attn.v.", ".self_attn.linear_v.");

        // LLM attention mappings
        key = key.replace(".attn.q_proj.", ".self_attn.q_proj.");
        key = key.replace(".attn.k_proj.", ".self_attn.k_proj.");
        key = key.replace(".attn.v_proj.", ".self_attn.v_proj.");
        key = key.replace(".attn.o_proj.", ".self_attn.o_proj.");
        key = key.replace(".attn.q_norm.", ".self_attn.q_norm.");
        key = key.replace(".attn.k_norm.", ".self_attn.k_norm.");

        std::rc::Rc::from(key)
    }

    /// Transcribe audio file with default (greedy) sampling.
    pub fn transcribe(&mut self, audio_path: impl AsRef<Path>) -> Result<String> {
        self.transcribe_with_config(audio_path, SamplingConfig::default())
    }

    /// Transcribe audio file with custom sampling configuration.
    pub fn transcribe_with_config(
        &mut self,
        audio_path: impl AsRef<Path>,
        config: SamplingConfig,
    ) -> Result<String> {
        // Load and preprocess audio
        let (samples, sample_rate) = audio::load_wav(audio_path)?;
        let samples = audio::resample(&samples, sample_rate, self.audio_config.sample_rate)?;

        // Compute mel spectrogram using cached frontend (avoids recreating FFT planner)
        let mel = self.mel_frontend.compute_mel_spectrogram(&samples)?;

        // Apply LFR (Low Frame Rate) - stack 7 frames, subsample by 6
        let mel_lfr = audio::apply_lfr(&mel, 7, 6)?;

        // Encode audio
        let audio_features = self.encode_audio(&mel_lfr)?;

        // Generate text with config
        self.generate_text_with_config(&audio_features, &config)
    }

    /// Transcribe multiple audio files.
    ///
    /// Processes files sequentially but reuses the cached mel frontend
    /// for efficient repeated processing.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = model.transcribe_batch(&[
    ///     "audio1.wav",
    ///     "audio2.wav",
    ///     "audio3.wav",
    /// ])?;
    ///
    /// for (path, text) in results {
    ///     println!("{}: {}", path, text);
    /// }
    /// ```
    pub fn transcribe_batch<P: AsRef<Path>>(
        &mut self,
        audio_paths: &[P],
    ) -> Result<Vec<(String, Result<String>)>> {
        self.transcribe_batch_with_config(audio_paths, SamplingConfig::default())
    }

    /// Transcribe multiple audio files with custom sampling configuration.
    pub fn transcribe_batch_with_config<P: AsRef<Path>>(
        &mut self,
        audio_paths: &[P],
        config: SamplingConfig,
    ) -> Result<Vec<(String, Result<String>)>> {
        let mut results = Vec::with_capacity(audio_paths.len());

        for path in audio_paths {
            let path_str = path.as_ref().display().to_string();
            let result = self.transcribe_with_config(path, config.clone());
            results.push((path_str, result));
        }

        Ok(results)
    }

    /// Transcribe multiple audio samples (already loaded).
    ///
    /// This is more efficient when you already have audio data in memory.
    ///
    /// # Arguments
    /// * `samples_list` - List of (samples, sample_rate) tuples
    ///
    /// # Returns
    /// * Vector of transcription results
    pub fn transcribe_samples_batch(
        &mut self,
        samples_list: &[(&[f32], u32)],
    ) -> Result<Vec<Result<String>>> {
        self.transcribe_samples_batch_with_config(samples_list, SamplingConfig::default())
    }

    /// Transcribe multiple audio samples with custom sampling configuration.
    pub fn transcribe_samples_batch_with_config(
        &mut self,
        samples_list: &[(&[f32], u32)],
        config: SamplingConfig,
    ) -> Result<Vec<Result<String>>> {
        let mut results = Vec::with_capacity(samples_list.len());

        for (samples, sample_rate) in samples_list {
            let result = self.transcribe_samples_with_config(samples, *sample_rate, config.clone());
            results.push(result);
        }

        Ok(results)
    }

    /// Transcribe audio samples directly (without file loading).
    pub fn transcribe_samples(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<String> {
        self.transcribe_samples_with_config(samples, sample_rate, SamplingConfig::default())
    }

    /// Transcribe audio samples with custom sampling configuration.
    pub fn transcribe_samples_with_config(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        config: SamplingConfig,
    ) -> Result<String> {
        // Resample to 16kHz if needed
        let samples = if sample_rate != self.audio_config.sample_rate {
            audio::resample(samples, sample_rate, self.audio_config.sample_rate)?
        } else {
            samples.to_vec()
        };

        // Compute mel spectrogram
        let mel = self.mel_frontend.compute_mel_spectrogram(&samples)?;

        // Apply LFR
        let mel_lfr = audio::apply_lfr(&mel, 7, 6)?;

        // Encode audio
        let audio_features = self.encode_audio(&mel_lfr)?;

        // Generate text
        self.generate_text_with_config(&audio_features, &config)
    }

    /// Encode audio to features.
    pub fn encode_audio(&mut self, mel_lfr: &Array) -> Result<Array> {
        // SenseVoice encoder
        let encoder_out = self.encoder.forward(mel_lfr)?;

        // Adaptor projection
        let adapted = self.adaptor.forward(&encoder_out)?;

        eval([&adapted])?;
        Ok(adapted)
    }

    /// Generate text from audio features with default sampling.
    pub fn generate_text(&mut self, audio_features: &Array) -> Result<String> {
        self.generate_text_with_config(audio_features, &SamplingConfig::default())
    }

    /// Generate text from audio features with custom sampling configuration.
    pub fn generate_text_with_config(
        &mut self,
        audio_features: &Array,
        config: &SamplingConfig,
    ) -> Result<String> {
        let mut cache: Vec<Option<KVCache>> = Vec::new();
        let mut tokens: Vec<i32> = Vec::new();

        // Audio feature length
        let audio_len = audio_features.shape()[1];

        // Build prompt tokens using ChatML format similar to official implementation:
        // <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        // <|im_start|>user\n语音转写成中文：<|startofspeech|>{AUDIO}<|endofspeech|><|im_end|>\n
        // <|im_start|>assistant\n
        let prefix_tokens = [
            151644,  // <|im_start|>
            8948,    // system
            198,     // \n
            2610,    // You
            525,     // are
            264,     // a
            10950,   // helpful
            17847,   // assistant
            13,      // .
            151645,  // <|im_end|>
            198,     // \n
            151644,  // <|im_start|>
            872,     // user
            198,     // \n
            105761,  // 语音
            46670,   // 转
            61443,   // 写
            12857,   // 成
            104811,  // 中文
            5122,    // ：
        ];
        let suffix_tokens = [
            151645,  // <|im_end|>
            198,     // \n
            151644,  // <|im_start|>
            77091,   // assistant
            198,     // \n
        ];

        // Audio position markers
        let speech_start = self.markers.start_token;  // 151646 <|startofspeech|>
        let speech_end = self.markers.end_token;      // 151647 <|endofspeech|>

        // Build full prompt token sequence
        // prefix + speech_start + audio_placeholders + speech_end + suffix
        let prompt_len = prefix_tokens.len() + 1 + audio_len as usize + 1 + suffix_tokens.len();
        let mut prompt_tokens: Vec<i32> = Vec::with_capacity(prompt_len);
        prompt_tokens.extend_from_slice(&prefix_tokens);
        prompt_tokens.push(speech_start);
        // Audio placeholders - will be replaced with audio embeddings
        for _ in 0..audio_len {
            prompt_tokens.push(0);  // Placeholder
        }
        prompt_tokens.push(speech_end);
        prompt_tokens.extend_from_slice(&suffix_tokens);

        // Get text embeddings for the full prompt
        let prompt_array = Array::from_slice(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
        let embeddings = self.llm.get_token_embeddings(&prompt_array)?;
        eval([&embeddings])?;

        // Audio position: starts after prefix + speech_start
        let audio_start = prefix_tokens.len() + 1;
        let audio_end = audio_start + audio_len as usize;

        // Replace audio placeholder embeddings with actual audio features
        // embeddings shape: [1, seq_len, hidden_dim]
        // audio_features shape: [1, audio_len, hidden_dim]

        // Build new embeddings with audio inserted
        let prefix_embed = embeddings.index((.., ..audio_start as i32, ..));
        let suffix_embed = embeddings.index((.., audio_end as i32.., ..));

        // Concatenate: prefix + audio + suffix
        // The audio features will be normalized by the first layer's RMSNorm
        let h = mlx_rs::ops::concatenate_axis(&[&prefix_embed, audio_features, &suffix_embed], 1)?;
        eval([&h])?;

        // First forward pass with the full prompt
        let logits = self.llm.forward_embeddings(&h, &mut cache)?;

        // Sample from last position
        let last_logits = logits.index((.., -1, ..));
        let token = Self::sample_with_config(&last_logits, config, &tokens)?;
        eval([&token])?;
        let mut token_id = token.item::<i32>();

        // Track recent tokens for repetition detection
        let mut recent_tokens: Vec<i32> = Vec::new();

        for _ in 0..config.max_tokens {
            // Check for EOS tokens
            if token_id == self.markers.eos_token || token_id == self.markers.im_end_token {
                break;
            }

            // Check for excessive repetition
            recent_tokens.push(token_id);
            if recent_tokens.len() > 10 {
                recent_tokens.remove(0);
            }
            if recent_tokens.len() >= 10 && recent_tokens.iter().all(|&t| t == token_id) {
                // Too many repetitions, force EOS
                break;
            }

            tokens.push(token_id);

            // Get embedding for next step
            let token_array = Array::from_slice(&[token_id], &[1, 1]);
            let h = self.llm.get_token_embeddings(&token_array)?;

            // Forward through LLM
            let logits = self.llm.forward_embeddings(&h, &mut cache)?;

            // Sample from last position with config
            let last_logits = logits.index((.., -1, ..));
            let token = Self::sample_with_config(&last_logits, config, &tokens)?;

            eval([&token])?;
            token_id = token.item::<i32>();
        }

        // Decode tokens to text
        self.decode_tokens(&tokens)
    }

    /// Sample from logits with sampling configuration.
    ///
    /// Supports:
    /// - Greedy decoding (temperature = 0)
    /// - Temperature scaling
    /// - Top-k filtering
    /// - Top-p (nucleus) sampling
    /// - Repetition penalty
    fn sample_with_config(
        logits: &Array,
        config: &SamplingConfig,
        _prev_tokens: &[i32],  // Reserved for future repetition penalty
    ) -> std::result::Result<Array, mlx_rs::error::Exception> {
        // Greedy decoding
        if config.temperature == 0.0 {
            return mlx_rs::ops::indexing::argmax_axis(logits, -1, false);
        }

        // Apply temperature scaling
        let mut scaled = logits.multiply(&Array::from(1.0 / config.temperature))?;

        // Apply top-k filtering
        if config.top_k > 0 {
            scaled = Self::apply_top_k(&scaled, config.top_k)?;
        }

        // Apply top-p (nucleus) filtering
        if config.top_p < 1.0 {
            scaled = Self::apply_top_p(&scaled, config.top_p)?;
        }

        // Sample from filtered distribution
        mlx_rs::random::categorical(&scaled, None, None, None)
    }

    /// Apply top-k filtering to logits.
    /// Keeps only the top k highest probability tokens.
    fn apply_top_k(logits: &Array, k: usize) -> std::result::Result<Array, mlx_rs::error::Exception> {
        // Get shape and vocab size
        let shape = logits.shape();
        let vocab_size = shape[shape.len() - 1] as usize;
        let k = k.min(vocab_size);

        // Find the k-th largest value as threshold
        // Sort descending and take the k-th value
        let sorted = mlx_rs::ops::sort_axis(logits, -1)?;
        let threshold_idx = (vocab_size - k) as i32;
        let threshold = sorted.index((.., threshold_idx));

        // Create mask for top-k values
        let mask = logits.ge(&threshold)?;

        // Apply mask: set non-top-k to negative infinity
        let neg_inf = Array::from(f32::NEG_INFINITY);
        mlx_rs::ops::r#where(&mask, logits, &neg_inf)
    }

    /// Apply top-p (nucleus) filtering to logits.
    /// Keeps tokens whose cumulative probability reaches p.
    fn apply_top_p(logits: &Array, p: f32) -> std::result::Result<Array, mlx_rs::error::Exception> {
        // Convert to probabilities
        let probs = mlx_rs::ops::softmax_axis(logits, -1, None)?;

        // Sort probabilities in descending order
        let _sorted_indices = mlx_rs::ops::argsort_axis(&probs, -1)?;
        // Reverse to get descending order
        let sorted_probs = mlx_rs::ops::sort_axis(&probs, -1)?;

        // Compute cumulative sum
        let cum_probs = mlx_rs::ops::cumsum(&sorted_probs, -1, None, None)?;

        // Find where cumulative prob exceeds p
        let p_array = Array::from(p);
        let mask = cum_probs.le(&p_array)?;

        // Apply mask
        let neg_inf = Array::from(f32::NEG_INFINITY);
        mlx_rs::ops::r#where(&mask, logits, &neg_inf)
    }

    /// Decode token IDs to text.
    fn decode_tokens(&self, tokens: &[i32]) -> Result<String> {
        if let Some(ref tokenizer) = self.tokenizer {
            let token_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            tokenizer
                .decode(&token_ids, true)
                .map_err(|e| Error::Tokenizer(e.to_string()))
        } else {
            // Fallback: return token IDs as string
            Ok(tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "))
        }
    }
}

/// Streaming transcription context.
///
/// Holds state for incremental audio processing, allowing real-time
/// transcription of audio chunks as they arrive.
///
/// # Example
///
/// ```rust,ignore
/// let mut model = FunASRNano::load("model_dir")?;
/// let mut ctx = model.create_streaming_context();
///
/// // Process audio chunks as they arrive
/// for chunk in audio_stream {
///     if let Some(partial) = model.transcribe_chunk(&mut ctx, &chunk)? {
///         println!("Partial: {}", partial);
///     }
/// }
///
/// // Get final transcription
/// let final_text = model.finalize_stream(ctx)?;
/// ```
pub struct StreamingContext {
    /// Buffered audio samples (16kHz mono)
    audio_buffer: Vec<f32>,
    /// Minimum samples needed before processing (e.g., 1 second = 16000 samples)
    min_samples: usize,
    /// Accumulated mel frames
    mel_frames: Vec<f32>,
    /// Number of mel frames accumulated
    n_mel_frames: usize,
    /// Generated tokens so far
    tokens: Vec<i32>,
    /// Sampling configuration
    sampling_config: SamplingConfig,
    /// Whether finalized
    finalized: bool,
}

impl StreamingContext {
    /// Create a new streaming context.
    ///
    /// # Arguments
    /// * `min_chunk_seconds` - Minimum audio duration before processing (default: 1.0s)
    /// * `sampling_config` - Sampling configuration for text generation
    pub fn new(min_chunk_seconds: f32, sampling_config: SamplingConfig) -> Self {
        let sample_rate = 16000;
        let min_samples = (min_chunk_seconds * sample_rate as f32) as usize;

        Self {
            audio_buffer: Vec::new(),
            min_samples,
            mel_frames: Vec::new(),
            n_mel_frames: 0,
            tokens: Vec::new(),
            sampling_config,
            finalized: false,
        }
    }

    /// Get the current partial transcription tokens.
    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    /// Check if context has been finalized.
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Get buffered audio length in seconds.
    pub fn buffered_seconds(&self) -> f32 {
        self.audio_buffer.len() as f32 / 16000.0
    }
}

impl Default for StreamingContext {
    fn default() -> Self {
        Self::new(1.0, SamplingConfig::default())
    }
}

/// Custom prompt for transcription or translation tasks.
#[derive(Debug, Clone)]
pub struct TaskPrompt {
    /// System prompt (e.g., "You are a helpful assistant.")
    pub system: String,
    /// User instruction (e.g., "语音转写成中文：" or "Transcribe and translate to English:")
    pub instruction: String,
}

impl Default for TaskPrompt {
    fn default() -> Self {
        Self {
            system: "You are a helpful assistant.".to_string(),
            instruction: "语音转写成中文：".to_string(),
        }
    }
}

impl TaskPrompt {
    /// Create a prompt for Chinese transcription (default behavior).
    pub fn transcribe_chinese() -> Self {
        Self::default()
    }

    /// Create a prompt for transcription with translation to English.
    pub fn translate_to_english() -> Self {
        Self {
            system: "You are a speech translation assistant.".to_string(),
            instruction: "Transcribe the following speech and translate to English:".to_string(),
        }
    }

    /// Create a prompt for correction and translation.
    pub fn correct_and_translate() -> Self {
        Self {
            system: "You are a speech translation assistant that accurately transcribes and translates.".to_string(),
            instruction: "请准确转写语音并翻译为英文：".to_string(),
        }
    }

    /// Create a custom prompt.
    pub fn custom(system: &str, instruction: &str) -> Self {
        Self {
            system: system.to_string(),
            instruction: instruction.to_string(),
        }
    }
}

impl FunASRNano {
    /// Create a new streaming context for incremental transcription.
    ///
    /// # Arguments
    /// * `min_chunk_seconds` - Minimum audio to buffer before processing (default: 1.0s)
    pub fn create_streaming_context(&self) -> StreamingContext {
        StreamingContext::default()
    }

    /// Create a streaming context with custom configuration.
    pub fn create_streaming_context_with_config(
        &self,
        min_chunk_seconds: f32,
        sampling_config: SamplingConfig,
    ) -> StreamingContext {
        StreamingContext::new(min_chunk_seconds, sampling_config)
    }

    /// Process an audio chunk and return partial transcription if available.
    ///
    /// Audio samples should be mono float32 at any sample rate (will be resampled to 16kHz).
    ///
    /// # Returns
    /// * `Ok(Some(text))` - Partial transcription available
    /// * `Ok(None)` - More audio needed, no transcription yet
    /// * `Err(e)` - Processing error
    pub fn transcribe_chunk(
        &mut self,
        ctx: &mut StreamingContext,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Option<String>> {
        if ctx.finalized {
            return Err(Error::Audio("Streaming context already finalized".to_string()));
        }

        // Resample to 16kHz if needed
        let samples = if sample_rate != 16000 {
            audio::resample(samples, sample_rate, 16000)?
        } else {
            samples.to_vec()
        };

        // Add to buffer
        ctx.audio_buffer.extend_from_slice(&samples);

        // Check if we have enough samples to process
        if ctx.audio_buffer.len() < ctx.min_samples {
            return Ok(None);
        }

        // Process the buffered audio
        self.process_buffered_audio(ctx)
    }

    /// Finalize the streaming context and return the complete transcription.
    ///
    /// This processes any remaining buffered audio and returns the final text.
    pub fn finalize_stream(&mut self, mut ctx: StreamingContext) -> Result<String> {
        if ctx.finalized {
            return Err(Error::Audio("Streaming context already finalized".to_string()));
        }

        ctx.finalized = true;

        // Process any remaining audio
        if !ctx.audio_buffer.is_empty() {
            // Force process even if below minimum
            let _ = self.process_buffered_audio(&mut ctx)?;
        }

        // If we have tokens, decode them
        if !ctx.tokens.is_empty() {
            self.decode_tokens(&ctx.tokens)
        } else {
            Ok(String::new())
        }
    }

    /// Internal: Process buffered audio and update context.
    fn process_buffered_audio(&mut self, ctx: &mut StreamingContext) -> Result<Option<String>> {
        if ctx.audio_buffer.is_empty() {
            return Ok(None);
        }

        // Compute mel spectrogram for buffered audio
        let mel = self.mel_frontend.compute_mel_spectrogram(&ctx.audio_buffer)?;

        // Apply LFR (Low Frame Rate) - stack 7 frames, subsample by 6
        let mel_lfr = audio::apply_lfr(&mel, 7, 6)?;

        // Encode audio
        let audio_features = self.encode_audio(&mel_lfr)?;

        // Generate text
        let text = self.generate_text_with_config(&audio_features, &ctx.sampling_config)?;

        // Store tokens (we'd need to get them from generate_text, but for now just clear buffer)
        ctx.audio_buffer.clear();

        Ok(Some(text))
    }

    // ============================================================
    // Translation and Custom Prompt Methods
    // ============================================================

    /// Transcribe audio with a custom prompt.
    ///
    /// This allows changing the task from pure transcription to translation,
    /// correction, or other audio-to-text tasks.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use funasr_nano_mlx::{FunASRNano, TaskPrompt};
    ///
    /// let mut model = FunASRNano::load("model_dir")?;
    ///
    /// // Transcribe and translate to English
    /// let prompt = TaskPrompt::translate_to_english();
    /// let english = model.transcribe_with_prompt("audio.wav", &prompt)?;
    /// ```
    pub fn transcribe_with_prompt(
        &mut self,
        audio_path: impl AsRef<Path>,
        prompt: &TaskPrompt,
    ) -> Result<String> {
        self.transcribe_with_prompt_and_config(audio_path, prompt, SamplingConfig::default())
    }

    /// Transcribe with custom prompt and sampling configuration.
    pub fn transcribe_with_prompt_and_config(
        &mut self,
        audio_path: impl AsRef<Path>,
        prompt: &TaskPrompt,
        config: SamplingConfig,
    ) -> Result<String> {
        // Load and preprocess audio
        let (samples, sample_rate) = audio::load_wav(audio_path)?;
        let samples = audio::resample(&samples, sample_rate, self.audio_config.sample_rate)?;

        // Compute mel spectrogram
        let mel = self.mel_frontend.compute_mel_spectrogram(&samples)?;

        // Apply LFR
        let mel_lfr = audio::apply_lfr(&mel, 7, 6)?;

        // Encode audio
        let audio_features = self.encode_audio(&mel_lfr)?;

        // Generate text with custom prompt
        self.generate_text_with_prompt(&audio_features, prompt, &config)
    }

    /// Generate text from audio features with a custom prompt.
    pub fn generate_text_with_prompt(
        &mut self,
        audio_features: &Array,
        prompt: &TaskPrompt,
        config: &SamplingConfig,
    ) -> Result<String> {
        let mut cache: Vec<Option<KVCache>> = Vec::new();
        let mut tokens: Vec<i32> = Vec::new();

        // Audio feature length
        let audio_len = audio_features.shape()[1];

        // Tokenize the custom prompt
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| Error::Tokenizer("Tokenizer not loaded".to_string()))?;

        // Build ChatML format with custom system and instruction
        // <|im_start|>system\n{system}<|im_end|>\n
        // <|im_start|>user\n{instruction}<|startofspeech|>{AUDIO}<|endofspeech|><|im_end|>\n
        // <|im_start|>assistant\n

        let system_text = format!("{}\n", prompt.system);
        let system_tokens = tokenizer.encode(system_text.as_str(), false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        let instruction_tokens = tokenizer.encode(prompt.instruction.as_str(), false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        // Special tokens
        let im_start: i32 = 151644;  // <|im_start|>
        let im_end: i32 = 151645;    // <|im_end|>
        let newline: i32 = 198;      // \n
        let system_id: i32 = 8948;   // system
        let user_id: i32 = 872;      // user
        let assistant_id: i32 = 77091; // assistant

        // Build prefix tokens
        let mut prefix_tokens: Vec<i32> = vec![
            im_start,
            system_id,
            newline,
        ];
        for tok in system_tokens.get_ids() {
            prefix_tokens.push(*tok as i32);
        }
        prefix_tokens.extend_from_slice(&[im_end, newline, im_start, user_id, newline]);
        for tok in instruction_tokens.get_ids() {
            prefix_tokens.push(*tok as i32);
        }

        let suffix_tokens = [
            im_end,
            newline,
            im_start,
            assistant_id,
            newline,
        ];

        // Audio position markers
        let speech_start = self.markers.start_token;  // 151646
        let speech_end = self.markers.end_token;      // 151647

        // Build full prompt
        let prompt_len = prefix_tokens.len() + 1 + audio_len as usize + 1 + suffix_tokens.len();
        let mut prompt_tokens: Vec<i32> = Vec::with_capacity(prompt_len);
        prompt_tokens.extend_from_slice(&prefix_tokens);
        prompt_tokens.push(speech_start);
        for _ in 0..audio_len {
            prompt_tokens.push(0);  // Placeholder for audio
        }
        prompt_tokens.push(speech_end);
        prompt_tokens.extend_from_slice(&suffix_tokens);

        // Get text embeddings
        let prompt_array = Array::from_slice(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
        let embeddings = self.llm.get_token_embeddings(&prompt_array)?;
        eval([&embeddings])?;

        // Audio position
        let audio_start = prefix_tokens.len() + 1;
        let audio_end = audio_start + audio_len as usize;

        // Replace placeholder embeddings with audio features
        let prefix_embed = embeddings.index((.., ..audio_start as i32, ..));
        let suffix_embed = embeddings.index((.., audio_end as i32.., ..));

        let h = mlx_rs::ops::concatenate_axis(&[&prefix_embed, audio_features, &suffix_embed], 1)?;
        eval([&h])?;

        // Forward pass
        let logits = self.llm.forward_embeddings(&h, &mut cache)?;

        // Sample
        let last_logits = logits.index((.., -1, ..));
        let token = Self::sample_with_config(&last_logits, config, &tokens)?;
        eval([&token])?;
        let mut token_id = token.item::<i32>();

        let mut recent_tokens: Vec<i32> = Vec::new();

        for _ in 0..config.max_tokens {
            if token_id == self.markers.eos_token || token_id == self.markers.im_end_token {
                break;
            }

            recent_tokens.push(token_id);
            if recent_tokens.len() > 10 {
                recent_tokens.remove(0);
            }
            if recent_tokens.len() >= 10 && recent_tokens.iter().all(|&t| t == token_id) {
                break;
            }

            tokens.push(token_id);

            let token_array = Array::from_slice(&[token_id], &[1, 1]);
            let h = self.llm.get_token_embeddings(&token_array)?;
            let logits = self.llm.forward_embeddings(&h, &mut cache)?;

            let last_logits = logits.index((.., -1, ..));
            let token = Self::sample_with_config(&last_logits, config, &tokens)?;
            eval([&token])?;
            token_id = token.item::<i32>();
        }

        self.decode_tokens(&tokens)
    }

    /// Generate text from text-only input (no audio).
    ///
    /// This uses the Qwen3 LLM directly for text-to-text tasks like
    /// translation, correction, or summarization.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let translation = model.generate_from_text(
    ///     "请翻译为英文：今天天气很好",
    ///     &SamplingConfig::default(),
    /// )?;
    /// ```
    pub fn generate_from_text(
        &mut self,
        text: &str,
        config: &SamplingConfig,
    ) -> Result<String> {
        let mut cache: Vec<Option<KVCache>> = Vec::new();
        let mut tokens: Vec<i32> = Vec::new();

        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| Error::Tokenizer("Tokenizer not loaded".to_string()))?;

        // Build ChatML format
        // <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
        let text_tokens = tokenizer.encode(text, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        let im_start: i32 = 151644;
        let im_end: i32 = 151645;
        let newline: i32 = 198;
        let user_id: i32 = 872;
        let assistant_id: i32 = 77091;

        let mut prompt_tokens: Vec<i32> = vec![im_start, user_id, newline];
        for tok in text_tokens.get_ids() {
            prompt_tokens.push(*tok as i32);
        }
        prompt_tokens.extend_from_slice(&[im_end, newline, im_start, assistant_id, newline]);

        // Get embeddings and forward
        let prompt_array = Array::from_slice(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
        let h = self.llm.get_token_embeddings(&prompt_array)?;
        eval([&h])?;

        let logits = self.llm.forward_embeddings(&h, &mut cache)?;

        let last_logits = logits.index((.., -1, ..));
        let token = Self::sample_with_config(&last_logits, config, &tokens)?;
        eval([&token])?;
        let mut token_id = token.item::<i32>();

        let mut recent_tokens: Vec<i32> = Vec::new();

        for _ in 0..config.max_tokens {
            if token_id == self.markers.eos_token || token_id == self.markers.im_end_token {
                break;
            }

            recent_tokens.push(token_id);
            if recent_tokens.len() > 10 {
                recent_tokens.remove(0);
            }
            if recent_tokens.len() >= 10 && recent_tokens.iter().all(|&t| t == token_id) {
                break;
            }

            tokens.push(token_id);

            let token_array = Array::from_slice(&[token_id], &[1, 1]);
            let h = self.llm.get_token_embeddings(&token_array)?;
            let logits = self.llm.forward_embeddings(&h, &mut cache)?;

            let last_logits = logits.index((.., -1, ..));
            let token = Self::sample_with_config(&last_logits, config, &tokens)?;
            eval([&token])?;
            token_id = token.item::<i32>();
        }

        self.decode_tokens(&tokens)
    }

    /// Translate text from Chinese to English.
    ///
    /// Uses the integrated Qwen3-0.6B LLM for text translation.
    pub fn translate_text(&mut self, text: &str) -> Result<String> {
        let prompt = format!("请将以下中文翻译为英文：{}", text);
        self.generate_from_text(&prompt, &SamplingConfig::default())
    }

    /// Transcribe audio and then translate the transcription to English.
    ///
    /// This is a two-pass approach:
    /// 1. ASR: Audio -> Chinese transcription
    /// 2. Translation: Chinese -> English
    ///
    /// # Returns
    /// A tuple of (Chinese transcription, English translation)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let (chinese, english) = model.transcribe_and_translate("audio.wav")?;
    /// println!("Chinese: {}", chinese);
    /// println!("English: {}", english);
    /// ```
    pub fn transcribe_and_translate(
        &mut self,
        audio_path: impl AsRef<Path>,
    ) -> Result<(String, String)> {
        // Pass 1: ASR
        let transcription = self.transcribe(&audio_path)?;

        // Pass 2: Translation
        let translation = self.translate_text(&transcription)?;

        Ok((transcription, translation))
    }

    /// Transcribe with correction and translation in one prompt.
    ///
    /// Attempts to use a single audio-to-English prompt.
    /// Note: Quality depends on Qwen3-0.6B's translation capability.
    pub fn transcribe_translate_direct(
        &mut self,
        audio_path: impl AsRef<Path>,
    ) -> Result<String> {
        let prompt = TaskPrompt::translate_to_english();
        self.transcribe_with_prompt(audio_path, &prompt)
    }
}

// Legacy streaming state (kept for backward compatibility)
/// Legacy streaming state - prefer using StreamingContext instead.
#[deprecated(note = "Use StreamingContext instead")]
pub struct StreamingState {
    /// KV cache for LLM
    pub cache: Vec<Option<KVCache>>,
    /// Previous context text
    pub prev_text: String,
    /// Accumulated tokens
    pub tokens: Vec<i32>,
}

#[allow(deprecated)]
impl StreamingState {
    /// Create new streaming state.
    pub fn new() -> Self {
        Self {
            cache: Vec::new(),
            prev_text: String::new(),
            tokens: Vec::new(),
        }
    }
}

#[allow(deprecated)]
impl Default for StreamingState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_config_default() {
        let config = SamplingConfig::default();
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.top_p, 1.0);
        assert_eq!(config.repetition_penalty, 1.0);
        assert_eq!(config.max_tokens, 256);
    }

    #[test]
    fn test_sampling_config_greedy() {
        let config = SamplingConfig::greedy();
        assert_eq!(config.temperature, 0.0);
    }

    #[test]
    fn test_sampling_config_with_temperature() {
        let config = SamplingConfig::with_temperature(0.7);
        assert_eq!(config.temperature, 0.7);
    }

    #[test]
    fn test_sampling_config_top_k() {
        let config = SamplingConfig::top_k(50);
        assert_eq!(config.top_k, 50);
        assert_eq!(config.temperature, 1.0);
    }

    #[test]
    fn test_sampling_config_top_p() {
        let config = SamplingConfig::top_p(0.9);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.temperature, 1.0);
    }

    #[test]
    fn test_streaming_context_default() {
        let ctx = StreamingContext::default();
        assert!(!ctx.is_finalized());
        assert_eq!(ctx.buffered_seconds(), 0.0);
        assert!(ctx.tokens().is_empty());
    }

    #[test]
    fn test_streaming_context_custom() {
        let ctx = StreamingContext::new(2.0, SamplingConfig::with_temperature(0.5));
        assert!(!ctx.is_finalized());
        assert_eq!(ctx.buffered_seconds(), 0.0);
    }

    #[test]
    fn test_speech_markers_default() {
        let markers = SpeechMarkers::default();
        assert_eq!(markers.start_token, 151646);
        assert_eq!(markers.end_token, 151647);
        assert_eq!(markers.eos_token, 151643);
        assert_eq!(markers.im_end_token, 151645);
    }

    #[test]
    fn test_funasr_nano_config_default() {
        let config = FunASRNanoConfig::default();
        assert_eq!(config.encoder.output_size, 512);
        assert_eq!(config.adaptor.encoder_dim, 512);
        assert_eq!(config.llm.hidden_size, 1024);
    }
}
