//! Qwen3-ASR combined model.
//!
//! AudioEncoder + Qwen3 LLM decoder for speech recognition.

use crate::audio::{self, AudioConfig, MelFrontend};
use crate::encoder::{AudioEncoder, AudioEncoderConfig};
use crate::error::{Error, Result};
use crate::qwen::{convert_rope_scaling, QwenAttention, QwenBlock, QwenConfig, QwenMLP, QwenModel};

use mlx_rs_core::{initialize_rope, KVCache};
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::{ModuleParameters as ModuleParametersTrait, Param};
use mlx_rs::nn;
use mlx_rs::ops::indexing::*;
use mlx_rs::quantization::MaybeQuantized;
use mlx_rs::transforms::eval;
use mlx_rs::Array;
use std::collections::HashMap;
use std::path::Path;

/// Quantization configuration.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct QuantizationConfig {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
}

fn default_group_size() -> i32 { 64 }
fn default_bits() -> i32 { 4 }

/// Qwen3-ASR model configuration.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen3ASRConfig {
    #[serde(default)]
    pub audio_config: AudioEncoderConfig,
    #[serde(default)]
    pub text_config: QwenConfig,
    #[serde(default = "default_audio_token_id")]
    pub audio_token_id: i32,
    #[serde(default = "default_audio_start_token_id")]
    pub audio_start_token_id: i32,
    #[serde(default = "default_audio_end_token_id")]
    pub audio_end_token_id: i32,
    #[serde(default)]
    pub support_languages: Vec<String>,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

fn default_audio_token_id() -> i32 { 151676 }
fn default_audio_start_token_id() -> i32 { 151669 }
fn default_audio_end_token_id() -> i32 { 151670 }

impl Default for Qwen3ASRConfig {
    fn default() -> Self {
        Self {
            audio_config: AudioEncoderConfig::default(),
            text_config: QwenConfig::default(),
            audio_token_id: 151676,
            audio_start_token_id: 151669,
            audio_end_token_id: 151670,
            support_languages: vec![
                "Chinese".into(), "English".into(), "Cantonese".into(),
                "Japanese".into(), "Korean".into(), "French".into(),
                "German".into(), "Spanish".into(), "Russian".into(),
            ],
            quantization: None,
        }
    }
}

impl Qwen3ASRConfig {
    /// Parse from config.json, handling the thinker_config nesting.
    pub fn from_config_json(value: &serde_json::Value) -> Result<Self> {
        let thinker = value.get("thinker_config").unwrap_or(value);

        let audio_config: AudioEncoderConfig = if let Some(ac) = thinker.get("audio_config") {
            serde_json::from_value(ac.clone())?
        } else {
            AudioEncoderConfig::default()
        };

        let text_config: QwenConfig = if let Some(tc) = thinker.get("text_config") {
            serde_json::from_value(tc.clone())?
        } else {
            QwenConfig::default()
        };

        let audio_token_id = thinker.get("audio_token_id")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32)
            .unwrap_or(151676);

        let audio_start_token_id = thinker.get("audio_start_token_id")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32)
            .unwrap_or(151669);

        let audio_end_token_id = thinker.get("audio_end_token_id")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32)
            .unwrap_or(151670);

        let support_languages = value.get("support_languages")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        // Quantization is at top level
        let quantization: Option<QuantizationConfig> = value.get("quantization")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        Ok(Self {
            audio_config,
            text_config,
            audio_token_id,
            audio_start_token_id,
            audio_end_token_id,
            support_languages,
            quantization,
        })
    }
}

/// Sampling configuration for text generation.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub max_tokens: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,  // Greedy
            max_tokens: 8192,
        }
    }
}

/// Qwen3-ASR model.
#[derive(ModuleParameters)]
pub struct Qwen3ASR {
    /// Audio encoder (Conv2d + Transformer)
    #[param]
    pub audio_tower: AudioEncoder,

    /// Text decoder (Qwen3)
    #[param]
    pub model: QwenModel,

    /// Model configuration
    pub config: Qwen3ASRConfig,

    /// Audio frontend
    mel_frontend: MelFrontend,

    /// Tokenizer
    tokenizer: Option<tokenizers::Tokenizer>,

    /// EOS token IDs for stopping generation
    eos_token_ids: Vec<i32>,
}

// ============================================================================
// Weight loading helpers
// ============================================================================

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    weights.get(key)
        .cloned()
        .ok_or_else(|| Error::Weight(format!("Weight not found: {}", key)))
}

fn make_quantized_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedLinear> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;

    let inner = nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(None),
    };

    let mut ql = nn::QuantizedLinear {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };
    ql.freeze_parameters(true);
    Ok(ql)
}

fn make_quantized_embedding(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedEmbedding> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;

    let inner = nn::Embedding {
        weight: Param::new(weight),
    };

    let mut qe = nn::QuantizedEmbedding {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };
    qe.freeze_parameters(true);
    Ok(qe)
}

/// Load all safetensors weights from a model directory.
fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>> {
    let single_file = model_dir.join("model.safetensors");
    if single_file.exists() {
        let loaded = Array::load_safetensors(&single_file)
            .map_err(|e| Error::ModelLoad(format!("Failed to load safetensors: {}", e)))?;
        return Ok(loaded);
    }

    // Try sharded files: model-00001-of-NNNNN.safetensors
    let mut all_weights = HashMap::new();
    let mut shard_idx = 1;
    loop {
        let shard_name = format!("model-{:05}-of-", shard_idx);
        let shard_files: Vec<_> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                name.starts_with(&shard_name) && name.ends_with(".safetensors")
            })
            .collect();

        if shard_files.is_empty() {
            break;
        }

        for entry in shard_files {
            let loaded = Array::load_safetensors(&entry.path())
                .map_err(|e| Error::ModelLoad(format!("Failed to load {}: {}", entry.path().display(), e)))?;
            all_weights.extend(loaded);
        }
        shard_idx += 1;
    }

    if all_weights.is_empty() {
        return Err(Error::ModelLoad(format!(
            "No safetensors files found in {}", model_dir.display()
        )));
    }

    Ok(all_weights)
}

impl Qwen3ASR {
    /// Load model from directory.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Load config.json
        let config_path = model_dir.join("config.json");
        if !config_path.exists() {
            return Err(Error::ModelLoad(format!(
                "config.json not found at {}", config_path.display()
            )));
        }
        let config_json: serde_json::Value = {
            let file = std::fs::File::open(&config_path)?;
            serde_json::from_reader(file)?
        };
        let config = Qwen3ASRConfig::from_config_json(&config_json)?;

        eprintln!("Audio encoder: {} layers, d_model={}", config.audio_config.encoder_layers, config.audio_config.d_model);
        eprintln!("Text decoder: {} layers, hidden_size={}", config.text_config.num_hidden_layers, config.text_config.hidden_size);

        if let Some(ref qc) = config.quantization {
            eprintln!("Quantization: {}bit, group_size={}", qc.bits, qc.group_size);
        }

        // Load all weights
        eprintln!("Loading weights...");
        let weights = load_all_weights(model_dir)?;
        eprintln!("Loaded {} weight tensors", weights.len());

        // Build audio encoder (NOT quantized) and load its weights
        let mut audio_tower = AudioEncoder::new(config.audio_config.clone())?;
        {
            let mut params = audio_tower.parameters_mut().flatten();
            let mut loaded = 0;
            for (key, value) in &weights {
                if key.starts_with("audio_tower.") {
                    let param_key = &key["audio_tower.".len()..];
                    if let Some(param) = params.get_mut(param_key) {
                        **param = value.clone();
                        loaded += 1;
                    }
                }
            }
            eprintln!("Audio tower: loaded {} parameters", loaded);
            eval(params.values().map(|v| &**v))?;
        }

        // Build text decoder
        let text_model = if let Some(ref qc) = config.quantization {
            Self::build_quantized_text_model(&config.text_config, &weights, qc.group_size, qc.bits)?
        } else {
            Self::build_text_model(&config.text_config, &weights)?
        };

        // Load tokenizer and EOS tokens
        let tokenizer = Self::load_tokenizer(model_dir)?;
        let eos_token_ids = Self::parse_eos_tokens(model_dir);

        let mel_frontend = MelFrontend::new(AudioConfig::default());

        Ok(Self {
            audio_tower,
            model: text_model,
            config,
            mel_frontend,
            tokenizer,
            eos_token_ids,
        })
    }

    /// Build quantized text decoder from weight HashMap.
    fn build_quantized_text_model(
        config: &QwenConfig,
        weights: &HashMap<String, Array>,
        group_size: i32,
        bits: i32,
    ) -> Result<QwenModel> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);

        let rope_scaling_map = convert_rope_scaling(&config.rope_scaling);

        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);

            let attention = QwenAttention {
                n_heads: config.num_attention_heads,
                n_kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                scale: (config.head_dim as f32).powf(-0.5),
                q_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    weights, &format!("{}.self_attn.q_proj", prefix), group_size, bits
                )?),
                k_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    weights, &format!("{}.self_attn.k_proj", prefix), group_size, bits
                )?),
                v_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    weights, &format!("{}.self_attn.v_proj", prefix), group_size, bits
                )?),
                o_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    weights, &format!("{}.self_attn.o_proj", prefix), group_size, bits
                )?),
                q_norm: nn::RmsNorm {
                    weight: Param::new(get_weight(weights, &format!("{}.self_attn.q_norm.weight", prefix))?),
                    eps: config.rms_norm_eps,
                },
                k_norm: nn::RmsNorm {
                    weight: Param::new(get_weight(weights, &format!("{}.self_attn.k_norm.weight", prefix))?),
                    eps: config.rms_norm_eps,
                },
                rope: initialize_rope(
                    config.head_dim,
                    config.rope_theta,
                    false,
                    &rope_scaling_map,
                    config.max_position_embeddings,
                )?,
            };

            let mlp = QwenMLP {
                gate_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    weights, &format!("{}.mlp.gate_proj", prefix), group_size, bits
                )?),
                down_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    weights, &format!("{}.mlp.down_proj", prefix), group_size, bits
                )?),
                up_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    weights, &format!("{}.mlp.up_proj", prefix), group_size, bits
                )?),
            };

            let block = QwenBlock {
                self_attn: attention,
                mlp,
                input_layernorm: nn::RmsNorm {
                    weight: Param::new(get_weight(weights, &format!("{}.input_layernorm.weight", prefix))?),
                    eps: config.rms_norm_eps,
                },
                post_attention_layernorm: nn::RmsNorm {
                    weight: Param::new(get_weight(weights, &format!("{}.post_attention_layernorm.weight", prefix))?),
                    eps: config.rms_norm_eps,
                },
            };

            layers.push(block);
        }

        let qwen_model = QwenModel {
            embed_tokens: MaybeQuantized::Quantized(make_quantized_embedding(
                weights, "model.embed_tokens", group_size, bits
            )?),
            layers,
            norm: nn::RmsNorm {
                weight: Param::new(get_weight(weights, "model.norm.weight")?),
                eps: config.rms_norm_eps,
            },
            config: config.clone(),
        };

        eprintln!("Text decoder: loaded {} quantized layers", config.num_hidden_layers);

        // Eval all text model params
        let params = qwen_model.parameters().flatten();
        eval(params.values().copied())?;

        Ok(qwen_model)
    }

    /// Build non-quantized text decoder from weight HashMap (fallback).
    fn build_text_model(
        config: &QwenConfig,
        weights: &HashMap<String, Array>,
    ) -> Result<QwenModel> {
        let mut model = QwenModel::new(config.clone())?;
        let mut params = model.parameters_mut().flatten();
        let mut loaded = 0;
        for (key, value) in weights {
            if key.starts_with("model.") {
                // Parameter paths from flatten() are relative to QwenModel,
                // but safetensors keys include "model." prefix.
                // Flatten keys: embed_tokens.weight, layers.0.self_attn.q_proj.weight, ...
                // Safetensors keys: model.embed_tokens.weight, model.layers.0..., ...
                let param_key = &key["model.".len()..];
                if let Some(param) = params.get_mut(param_key) {
                    **param = value.clone();
                    loaded += 1;
                }
            }
        }
        eprintln!("Text decoder: loaded {} parameters (non-quantized)", loaded);
        eval(params.values().map(|v| &**v))?;
        Ok(model)
    }

    /// Load tokenizer from model directory.
    ///
    /// Tries `tokenizer.json` first. If missing, builds from `vocab.json` +
    /// `merges.txt` + special tokens from `tokenizer_config.json`, then caches
    /// the result as `tokenizer.json` for subsequent loads.
    fn load_tokenizer(model_dir: &Path) -> Result<Option<tokenizers::Tokenizer>> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| Error::Tokenizer(e.to_string()))?;
            return Ok(Some(tokenizer));
        }

        // Build from vocab.json + merges.txt
        let vocab_path = model_dir.join("vocab.json");
        let merges_path = model_dir.join("merges.txt");
        if !vocab_path.exists() || !merges_path.exists() {
            eprintln!("Warning: no tokenizer found at {}", model_dir.display());
            return Ok(None);
        }

        use tokenizers::models::bpe::BPE;
        let bpe = BPE::from_file(&vocab_path.to_string_lossy(), &merges_path.to_string_lossy())
            .build()
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut tokenizer = tokenizers::Tokenizer::new(bpe);

        // ByteLevel decoder for proper CJK/Unicode handling
        let byte_level = tokenizers::pre_tokenizers::byte_level::ByteLevel::new(true, true, true);
        tokenizer.with_decoder(Some(byte_level));

        // Load special tokens from tokenizer_config.json
        let config_path = model_dir.join("tokenizer_config.json");
        if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)?;
            let config: serde_json::Value = serde_json::from_str(&config_str)?;
            if let Some(added_tokens) = config.get("added_tokens_decoder").and_then(|v| v.as_object()) {
                let special_tokens: Vec<_> = added_tokens.values()
                    .filter_map(|info| {
                        info.get("content").and_then(|v| v.as_str())
                            .map(|content| tokenizers::AddedToken::from(content, true))
                    })
                    .collect();
                if !special_tokens.is_empty() {
                    eprintln!("Adding {} special tokens from tokenizer_config.json", special_tokens.len());
                    tokenizer.add_special_tokens(&special_tokens);
                }
            }
        }

        // Cache for next time
        if let Err(e) = tokenizer.save(&tokenizer_path, false) {
            eprintln!("Warning: could not cache tokenizer.json: {}", e);
        }

        Ok(Some(tokenizer))
    }

    /// Parse EOS token IDs from tokenizer_config.json.
    ///
    /// Reads `eos_token` and `pad_token` entries, resolves their IDs from
    /// `added_tokens_decoder`. Falls back to Qwen3 defaults (151645, 151643).
    fn parse_eos_tokens(model_dir: &Path) -> Vec<i32> {
        let config_path = model_dir.join("tokenizer_config.json");
        let config: serde_json::Value = (|| {
            let s = std::fs::read_to_string(&config_path).ok()?;
            serde_json::from_str(&s).ok()
        })().unwrap_or_default();

        let added_tokens = config.get("added_tokens_decoder")
            .and_then(|v| v.as_object());

        let mut eos_ids = Vec::new();
        for key in &["eos_token", "pad_token"] {
            if let Some(token_content) = config.get(key).and_then(|v| v.as_str()) {
                if let Some(ref tokens) = added_tokens {
                    for (id_str, info) in tokens.iter() {
                        if info.get("content").and_then(|v| v.as_str()) == Some(token_content) {
                            if let Ok(id) = id_str.parse::<i32>() {
                                if !eos_ids.contains(&id) {
                                    eos_ids.push(id);
                                }
                            }
                        }
                    }
                }
            }
        }

        if eos_ids.is_empty() {
            vec![151645, 151643] // <|im_end|>, <|endoftext|>
        } else {
            eos_ids
        }
    }

    /// Transcribe audio file.
    pub fn transcribe(&mut self, audio_path: impl AsRef<Path>) -> Result<String> {
        self.transcribe_with_language(audio_path, "Chinese")
    }

    /// Transcribe audio file with specified language.
    pub fn transcribe_with_language(
        &mut self,
        audio_path: impl AsRef<Path>,
        language: &str,
    ) -> Result<String> {
        let (samples, sample_rate) = audio::load_wav(audio_path)?;
        let samples = audio::resample(&samples, sample_rate, 16000)?;
        self.transcribe_samples(&samples, language)
    }

    /// Transcribe audio samples (16kHz mono f32).
    /// For audio longer than 30 seconds, automatically uses chunked processing.
    pub fn transcribe_samples(
        &mut self,
        samples: &[f32],
        language: &str,
    ) -> Result<String> {
        let config = SamplingConfig::default();
        let chunk_threshold = 30 * 16000; // 30 seconds at 16kHz
        if samples.len() > chunk_threshold {
            self.transcribe_samples_chunked(samples, language, &config, 30.0)
        } else {
            self.transcribe_samples_with_config(samples, language, &config)
        }
    }

    /// Transcribe long audio by splitting into chunks.
    /// Each chunk is processed independently with its own KV cache.
    pub fn transcribe_samples_chunked(
        &mut self,
        samples: &[f32],
        language: &str,
        config: &SamplingConfig,
        chunk_duration_secs: f32,
    ) -> Result<String> {
        let chunk_size = (chunk_duration_secs * 16000.0) as usize;
        let total_duration = samples.len() as f32 / 16000.0;
        let num_chunks = (samples.len() + chunk_size - 1) / chunk_size;

        eprintln!(
            "Long audio: {:.1}s, splitting into {} chunks of {:.0}s",
            total_duration, num_chunks, chunk_duration_secs,
        );

        let mut transcriptions = Vec::new();

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(samples.len());
            let chunk_samples = &samples[start..end];
            let chunk_duration = chunk_samples.len() as f32 / 16000.0;

            // Skip very short trailing chunks
            if chunk_samples.len() < 1600 {
                continue;
            }

            eprintln!(
                "\n--- Chunk {}/{} ({:.1}s - {:.1}s, {:.1}s) ---",
                chunk_idx + 1,
                num_chunks,
                start as f32 / 16000.0,
                end as f32 / 16000.0,
                chunk_duration,
            );

            let chunk_start = std::time::Instant::now();
            match self.transcribe_samples_with_config(chunk_samples, language, config) {
                Ok(text) => {
                    let elapsed = chunk_start.elapsed().as_secs_f32();
                    let preview: String = text.chars().take(40).collect();
                    eprintln!(
                        "  -> {:.2}s ({:.1}x RT): {}{}",
                        elapsed,
                        chunk_duration / elapsed,
                        preview,
                        if text.chars().count() > 40 { "..." } else { "" }
                    );
                    if !text.is_empty() {
                        transcriptions.push(text);
                    }
                }
                Err(e) => {
                    eprintln!("  -> Error: {}", e);
                }
            }
        }

        Ok(transcriptions.join(""))
    }

    /// Transcribe with full configuration.
    pub fn transcribe_samples_with_config(
        &mut self,
        samples: &[f32],
        language: &str,
        config: &SamplingConfig,
    ) -> Result<String> {
        // 1. Compute mel spectrogram
        let mel = self.mel_frontend.compute_mel_spectrogram(samples)?;
        eval([&mel])?;

        // 2. Encode audio
        let audio_features = self.audio_tower.forward_encoder(&mel)?;
        eval([&audio_features])?;

        let num_audio_tokens = audio_features.shape()[0];
        eprintln!("Audio: {} mel frames -> {} audio tokens", mel.shape()[1], num_audio_tokens);

        // 3. Build prompt
        let input_ids = self.build_prompt(num_audio_tokens, language)?;
        eval([&input_ids])?;

        // 4. Build input embeddings with audio merged in
        let inputs_embeds = self.build_inputs_embeds(&input_ids, &audio_features)?;
        eval([&inputs_embeds])?;

        // 5. Autoregressive generation
        self.generate(&inputs_embeds, config)
    }

    /// Build prompt token IDs.
    fn build_prompt(&self, num_audio_tokens: i32, language: &str) -> Result<Array> {
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| Error::Tokenizer("Tokenizer not loaded".to_string()))?;

        let prompt = format!(
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>{}<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage {}<asr_text>",
            "<|audio_pad|>".repeat(num_audio_tokens as usize),
            language,
        );

        let encoding = tokenizer.encode(prompt.as_str(), false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        let ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        let len = ids.len() as i32;
        Ok(Array::from_slice(&ids, &[1, len]))
    }

    /// Build input embeddings with audio features replacing audio_pad tokens.
    fn build_inputs_embeds(
        &mut self,
        input_ids: &Array,
        audio_features: &Array,
    ) -> Result<Array> {
        // Get text embeddings for all tokens
        let embeddings = self.model.get_token_embeddings(input_ids)?;
        eval([&embeddings])?;

        // Find audio_pad token positions
        let audio_token_id = self.config.audio_token_id;

        let ids_flat = input_ids.reshape(&[-1])?;
        eval([&ids_flat])?;
        let ids_data: &[i32] = ids_flat.try_as_slice::<i32>()
            .map_err(|e| Error::Inference(format!("Failed to read input_ids: {}", e)))?;

        let mut first_audio = None;
        let mut last_audio = None;
        for (i, &id) in ids_data.iter().enumerate() {
            if id == audio_token_id {
                if first_audio.is_none() {
                    first_audio = Some(i);
                }
                last_audio = Some(i);
            }
        }

        if let (Some(first), Some(last)) = (first_audio, last_audio) {
            let audio_start = first as i32;
            let audio_end = (last + 1) as i32;

            let prefix_embed = embeddings.index((.., ..audio_start, ..));
            let suffix_embed = embeddings.index((.., audio_end.., ..));

            let audio_embed = audio_features.reshape(&[
                1, audio_features.shape()[0], audio_features.shape()[1]
            ])?;

            let audio_embed = audio_embed.as_dtype(embeddings.dtype())?;

            let result = mlx_rs::ops::concatenate_axis(
                &[&prefix_embed, &audio_embed, &suffix_embed],
                1,
            )?;
            Ok(result)
        } else {
            Ok(embeddings)
        }
    }

    /// Autoregressive text generation.
    fn generate(
        &mut self,
        inputs_embeds: &Array,
        config: &SamplingConfig,
    ) -> Result<String> {
        let mut cache: Vec<Option<KVCache>> = Vec::new();
        let mut tokens: Vec<i32> = Vec::new();
        let eos_tokens = &self.eos_token_ids;

        // First forward pass with full prompt
        let hidden_states = self.model.forward_embeddings(inputs_embeds, &mut cache)?;

        // Get logits from last position using tied weights
        let last_hidden = hidden_states.index((.., -1, ..));
        let last_hidden = last_hidden.reshape(&[1, 1, self.config.text_config.hidden_size])?;
        let logits = self.model.compute_logits(&last_hidden)?;

        let last_logits = logits.index((.., -1, ..));
        let token = Self::sample(&last_logits, config)?;
        eval([&token])?;
        let mut token_id = token.item::<i32>();

        let mut recent_tokens: Vec<i32> = Vec::new();

        for _ in 0..config.max_tokens {
            if eos_tokens.contains(&token_id) {
                break;
            }

            // Repetition detection
            recent_tokens.push(token_id);
            if recent_tokens.len() > 10 {
                recent_tokens.remove(0);
            }
            if recent_tokens.len() >= 10 && recent_tokens.iter().all(|&t| t == token_id) {
                break;
            }

            tokens.push(token_id);

            // Get embedding for next token
            let token_array = Array::from_slice(&[token_id], &[1, 1]);
            let h = self.model.get_token_embeddings(&token_array)?;

            // Forward through decoder
            let hidden_states = self.model.forward_embeddings(&h, &mut cache)?;
            let last_hidden = hidden_states.index((.., -1, ..));
            let last_hidden = last_hidden.reshape(&[1, 1, self.config.text_config.hidden_size])?;
            let logits = self.model.compute_logits(&last_hidden)?;

            let last_logits = logits.index((.., -1, ..));
            let token = Self::sample(&last_logits, config)?;
            eval([&token])?;
            token_id = token.item::<i32>();
        }

        // Decode tokens
        self.decode_tokens(&tokens)
    }

    /// Sample from logits.
    fn sample(
        logits: &Array,
        config: &SamplingConfig,
    ) -> std::result::Result<Array, mlx_rs::error::Exception> {
        if config.temperature == 0.0 {
            argmax_axis(logits, -1, false)
        } else {
            let scaled = logits.multiply(&Array::from(1.0 / config.temperature))?;
            mlx_rs::random::categorical(&scaled, None, None, None)
        }
    }

    /// Decode token IDs to text.
    fn decode_tokens(&self, tokens: &[i32]) -> Result<String> {
        if let Some(ref tokenizer) = self.tokenizer {
            let token_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            tokenizer
                .decode(&token_ids, true)
                .map_err(|e| Error::Tokenizer(e.to_string()))
        } else {
            Ok(tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "))
        }
    }
}
