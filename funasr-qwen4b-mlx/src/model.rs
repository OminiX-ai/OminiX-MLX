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

/// Speech tokens for multimodal prompts
pub struct SpeechTokens {
    pub start_of_speech: u32,  // <|startofspeech|>
    pub end_of_speech: u32,    // <|endofspeech|>
    pub im_start: u32,         // <|im_start|>
    pub im_end: u32,           // <|im_end|>
    pub eos: u32,              // <|endoftext|>
}

impl Default for SpeechTokens {
    fn default() -> Self {
        Self {
            start_of_speech: 151646,
            end_of_speech: 151647,
            im_start: 151644,
            im_end: 151645,
            eos: 151643,
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
}

impl FunASRQwen4B {
    /// Load model from directory
    ///
    /// Expected directory structure:
    /// ```text
    /// model_dir/
    /// ├── sensevoice/
    /// │   └── encoder.safetensors
    /// ├── adaptor.safetensors
    /// ├── qwen3-4b/
    /// │   ├── model-00001-of-00003.safetensors
    /// │   ├── ...
    /// │   └── tokenizer.json
    /// └── config.yaml (optional)
    /// ```
    pub fn load(model_dir: &str) -> Result<Self> {
        let model_path = Path::new(model_dir);

        // Load SenseVoice encoder
        let mut encoder = SenseVoiceEncoder::new(SenseVoiceEncoderConfig::default())?;

        // Try to load encoder weights from various locations
        let encoder_paths = [
            model_path.join("sensevoice").join("encoder.safetensors"),
            model_path.join("model.safetensors"),  // Combined model file
            std::path::PathBuf::from(std::env::var("SENSEVOICE_WEIGHTS").unwrap_or_default()),
            dirs::home_dir().unwrap_or_default().join(".dora/models/funasr-nano/model.safetensors"),
        ];

        let mut encoder_loaded = false;
        for encoder_path in &encoder_paths {
            if encoder_path.exists() {
                match encoder.load_weights(encoder_path) {
                    Ok(_) => {
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

        // Load adaptor
        let adaptor_path = model_path.join("adaptor.safetensors");
        let mut adaptor = AudioAdaptorQwen4B::new()?;
        if adaptor_path.exists() {
            adaptor.load_weights(adaptor_path.to_str().unwrap())?;
        }

        // Load Qwen3-4B using qwen3-mlx
        let qwen_path = model_path.join("qwen3-4b");
        let llm = load_model(&qwen_path)
            .map_err(|e| Error::ModelLoad(format!("Failed to load Qwen3: {:?}", e)))?;

        // Load tokenizer
        let tokenizer = load_tokenizer(&qwen_path)
            .map_err(|e| Error::Tokenizer(format!("Failed to load tokenizer: {:?}", e)))?;

        // Create audio config
        let audio_config = AudioConfig::default();

        Ok(Self {
            encoder,
            adaptor,
            llm,
            tokenizer,
            audio_config,
            speech_tokens: SpeechTokens::default(),
        })
    }

    /// Transcribe audio file to Chinese text
    pub fn transcribe(&mut self, audio_path: &str) -> Result<String> {
        // Load and preprocess audio
        let (samples, sample_rate) = load_wav(audio_path)?;
        self.transcribe_samples(&samples, sample_rate)
    }

    /// Transcribe raw audio samples to Chinese text
    ///
    /// This is useful for streaming audio or when you already have the samples.
    pub fn transcribe_samples(&mut self, samples: &[f32], sample_rate: u32) -> Result<String> {
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
        let adapted = self.adaptor.forward(&encoder_out)?;

        // Generate text
        let text = self.generate_from_audio_features(&adapted, "语音转写成中文：")?;

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
            "<|im_start|>user\nTranslate to English: {}<|im_end|>\n<|im_start|>assistant\n",
            chinese
        );

        self.generate_text(&prompt, 200)
    }

    /// Generate text from audio features (multimodal)
    ///
    /// Uses ChatML format with audio embedding injection:
    /// `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n`
    /// `<|im_start|>user\n{prompt}<|startofspeech|>{AUDIO}<|endofspeech|><|im_end|>\n`
    /// `<|im_start|>assistant\n`
    ///
    /// Based on funasr-nano-mlx approach: build full prompt with placeholders,
    /// get embeddings, then replace audio placeholder with actual features.
    fn generate_from_audio_features(
        &mut self,
        audio_features: &Array,
        prompt: &str,
    ) -> Result<String> {
        let audio_len = audio_features.shape()[1] as usize;

        // Build prompt tokens using ChatML format (same as funasr-nano-mlx)
        // Hard-coded token IDs for speed
        let prefix_tokens: Vec<i32> = vec![
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
        ];

        // Tokenize the custom prompt instruction
        let prompt_encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| Error::Tokenizer(format!("Tokenization failed: {}", e)))?;

        let suffix_tokens: Vec<i32> = vec![
            151645,  // <|im_end|>
            198,     // \n
            151644,  // <|im_start|>
            77091,   // assistant
            198,     // \n
        ];

        // Audio markers
        let speech_start = self.speech_tokens.start_of_speech as i32;
        let speech_end = self.speech_tokens.end_of_speech as i32;

        // Build full prompt: prefix + prompt_tokens + speech_start + [placeholders] + speech_end + suffix
        let mut prompt_tokens_full: Vec<i32> = Vec::with_capacity(
            prefix_tokens.len() + prompt_encoding.get_ids().len() + 1 + audio_len + 1 + suffix_tokens.len()
        );
        prompt_tokens_full.extend_from_slice(&prefix_tokens);
        for &tok in prompt_encoding.get_ids() {
            prompt_tokens_full.push(tok as i32);
        }
        prompt_tokens_full.push(speech_start);
        // Audio placeholders - will be replaced with audio embeddings
        for _ in 0..audio_len {
            prompt_tokens_full.push(0);  // Placeholder
        }
        prompt_tokens_full.push(speech_end);
        prompt_tokens_full.extend_from_slice(&suffix_tokens);

        // Get text embeddings for the full prompt
        let prompt_array = Array::from_slice(&prompt_tokens_full, &[1, prompt_tokens_full.len() as i32]);
        let embeddings = self.get_token_embeddings(&prompt_array)?;
        mlx_rs::transforms::eval([&embeddings])?;

        // Audio position: starts after prefix + prompt + speech_start
        let audio_start = prefix_tokens.len() + prompt_encoding.get_ids().len() + 1;
        let audio_end = audio_start + audio_len;

        // Replace placeholder embeddings with actual audio features
        let prefix_embed = embeddings.index((.., ..audio_start as i32, ..));
        let suffix_embed = embeddings.index((.., audio_end as i32.., ..));

        // Concatenate: prefix + audio + suffix
        let h = mlx_rs::ops::concatenate_axis(&[&prefix_embed, audio_features, &suffix_embed], 1)
            .map_err(|e| Error::ModelLoad(format!("Concatenate failed: {:?}", e)))?;
        mlx_rs::transforms::eval([&h])?;

        // Generate tokens autoregressively
        self.generate_from_embeddings(&h, 256)
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
    /// Uses Qwen3-recommended sampling (temperature=0.6, top_k=20) with
    /// presence penalty and n-gram repetition detection.
    fn generate_from_embeddings(
        &mut self,
        embeddings: &Array,
        max_tokens: usize,
    ) -> Result<String> {
        let temperature: f32 = 0.6;
        let top_k: usize = 20;
        let presence_penalty: f32 = 1.0;

        let mut cache: Vec<Option<KVCache>> = Vec::new();
        let mut tokens: Vec<i32> = Vec::new();

        // Prefill: forward pass with full embeddings
        let logits = self.forward_embeddings(embeddings, &mut cache)?;

        // Sample from last position
        let last_logits = logits.index((.., -1, ..));
        let token = Self::sample_top_k_p(&last_logits, temperature, top_k, &tokens, presence_penalty)?;
        mlx_rs::transforms::eval([&token])?;
        let mut token_id: i32 = token.item();

        for _ in 0..max_tokens {
            // Check for EOS tokens
            if token_id == self.speech_tokens.eos as i32 || token_id == self.speech_tokens.im_end as i32 {
                break;
            }

            tokens.push(token_id);

            // N-gram repetition detection (patterns 1-64 tokens, keep 1 copy)
            if tokens.len() >= 4 {
                let mut repeat_n: usize = 0;
                let max_n = 64.min(tokens.len() / 2);
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
                    tokens.truncate(pos + repeat_n);
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
        self.tokenizer.decode(&token_ids, true)
            .map_err(|e| Error::Tokenizer(format!("Decoding failed: {}", e)))
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
