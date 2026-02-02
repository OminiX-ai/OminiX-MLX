//! Full ASR pipeline with chunked processing
//!
//! Processes long audio in 10s chunks, transcribes each chunk using
//! proper ChatML multimodal template with ASR-specific system prompt.
//!
//! Run: cargo run --example test_real_audio --release -- <audio_path> [chunk_seconds] [raw]

use funasr_qwen4b_mlx::sensevoice_encoder::{SenseVoiceEncoder, SenseVoiceEncoderConfig};
use funasr_qwen4b_mlx::adaptor::AudioAdaptorQwen4B;
use funasr_qwen4b_mlx::audio::{load_wav, resample, AudioConfig, MelFrontendMLX, apply_lfr};
use funasr_qwen4b_mlx::error::Result;
use mlx_rs::module::Module;
use mlx_rs::quantization::MaybeQuantized;
use mlx_rs::ops::indexing::IndexOp;
use qwen3_mlx::{
    load_model, load_tokenizer, KVCache,
    AttentionInput, sample, create_attention_mask, AttentionMask,
};

const SAMPLE_RATE: usize = 16000;
const MAX_ASR_TOKENS: usize = 100;
const TEMPERATURE: f32 = 0.6;  // Qwen3 recommended (DO NOT use 0.0 — causes endless repetition)
const TOP_K: usize = 20;       // Qwen3 recommended: keep top 20 tokens
const PRESENCE_PENALTY: f32 = 1.0;  // Conservative for ASR (Qwen3 suggests 1.5 general)
const ENTROPY_THRESHOLD: f32 = 0.5;  // Low entropy = degenerate state
const ENTROPY_WINDOW: usize = 5;     // Consecutive low-entropy steps before stopping
const EOS_TOKEN: i32 = 151643;
const IM_END_TOKEN: i32 = 151645;
const SILENCE_THRESHOLD_DB: f32 = -40.0;

#[derive(Clone, Copy, PartialEq)]
enum TemplateMode {
    ChatML,  // Improved ChatML with ASR-specific system prompt
    Raw,     // Training-matched: just audio features, no template
}

/// Check if audio chunk is silence (below energy threshold)
fn is_silent(samples: &[f32], threshold_db: f32) -> bool {
    if samples.is_empty() {
        return true;
    }
    let rms = (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    let db = 20.0 * rms.max(1e-10).log10();
    db < threshold_db
}

/// Clean up transcription output by removing meta-commentary and formatting artifacts.
fn clean_transcription(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    // 1. Strip echoed instruction prefixes
    let mut result = trimmed.to_string();
    for prefix in &[
        "语音转写成中文：",
        "语音转写成中文:",
        "语音转写：",
        "转写内容：",
        "转写内容:",
    ] {
        if let Some(stripped) = result.strip_prefix(prefix) {
            result = stripped.trim().to_string();
            break;
        }
    }

    // 2. Detect meta-commentary (return empty if entire output is commentary)
    let meta_prefixes = [
        "这段文字似乎",
        "这段文本似乎",
        "这段内容似乎",
        "这段音频",
        "我无法理解",
        "我无法确定",
        "我无法识别",
        "我可能不太明白",
        "我认为这可能是由于语音",
        "我认为这可能是由于",
        "您提供的",
        "你提供的",
        "您说的不太清楚",
        "您说的不太",
        "以下是对",
        "以下是您",
        "这是一段",
        "抱歉",
        "很抱歉",
        "对不起",
        "作为人工智能",
        "作为AI",
        "该文本",
        "该语音",
        "无法转写",
        "我将按照",
        "这个文本看起来",
        "这个句子看起来",
        "虽然这个句子",
        "等等，你这段",
        "哎呀，这样的语音",
        "嗯，我可能",
        "嗯，我不太",
    ];
    for pattern in &meta_prefixes {
        if result.starts_with(pattern) {
            return String::new();
        }
    }
    // Also check for meta-commentary keywords anywhere in short outputs
    if result.chars().count() < 80 {
        let meta_keywords = [
            "语音输入过程中出现的错误",
            "建议您重新尝试输入",
            "能再重复一遍吗",
            "能否再详细说明",
            "导致生成的文本内容混乱",
            "缺乏逻辑",
        ];
        for kw in &meta_keywords {
            if result.contains(kw) {
                return String::new();
            }
        }
    }

    // 3. Strip markdown formatting (join lines, remove bullets/numbers/quotes)
    if result.contains('\n') {
        let lines: Vec<String> = result.lines().map(|line| {
            let mut l = line.trim().to_string();
            if l.starts_with("- ") || l.starts_with("* ") {
                l = l[2..].to_string();
            }
            if l.starts_with("> ") {
                l = l[2..].to_string();
            }
            if let Some(dot_pos) = l.find(". ") {
                if dot_pos <= 3 && l[..dot_pos].chars().all(|c| c.is_ascii_digit()) {
                    l = l[dot_pos + 2..].to_string();
                }
            }
            // Skip bold markdown headers
            if l.starts_with("**") && l.ends_with("**") {
                return String::new();
            }
            l
        }).filter(|l| !l.is_empty()).collect();
        result = lines.join("");
    }

    // 4. Remove parenthetical notes at end: （注：...） or (注：...)
    if let Some(paren_start) = result.rfind('（') {
        if result.ends_with('）') {
            let inner = &result[paren_start..];
            if inner.contains("注") || inner.contains("备注") || inner.contains("可能") {
                result = result[..paren_start].trim().to_string();
            }
        }
    }

    // 5. Remove surrounding quotes
    let chars: Vec<char> = result.chars().collect();
    if chars.len() > 2 {
        let (first, last) = (chars[0], chars[chars.len() - 1]);
        if (first == '"' && last == '"')
            || (first == '\u{201C}' && last == '\u{201D}')
            || (first == '\u{300C}' && last == '\u{300D}')
        {
            result = chars[1..chars.len() - 1].iter().collect();
        }
    }

    result
}

/// Build multimodal embeddings with ChatML template (ASR-specific system prompt):
///
/// <|im_start|>system\n你是语音转写系统。直接输出语音内容的中文文字，不要添加任何解释、评论或格式。<|im_end|>\n
/// <|im_start|>user\n<|startofspeech|>[AUDIO]<|endofspeech|><|im_end|>\n
/// <|im_start|>assistant\n<think>\n\n</think>\n\n
fn build_multimodal_embeddings(
    audio_features: &mlx_rs::Array,
    llm: &mut qwen3_mlx::Model,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<mlx_rs::Array> {
    let audio_len = audio_features.shape()[1] as usize;

    // ASR-specific system prompt (Chinese, constrains output to transcription only)
    let system_prompt = "你是语音转写系统。直接输出语音内容的中文文字，不要添加任何解释、评论或格式。";
    let system_encoding = tokenizer.encode(system_prompt, false)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::Tokenizer(format!("{}", e)))?;

    // Build prefix: <|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n
    let mut prefix_tokens: Vec<i32> = vec![
        151644,  // <|im_start|>
        8948,    // system
        198,     // \n
    ];
    for &tok in system_encoding.get_ids() {
        prefix_tokens.push(tok as i32);
    }
    prefix_tokens.extend_from_slice(&[
        151645,  // <|im_end|>
        198,     // \n
        151644,  // <|im_start|>
        872,     // user
        198,     // \n
    ]);

    // Note: removed 151646/151647 (actually <|object_ref_start|>/<|object_ref_end|> in tokenizer,
    // not speech tokens). Training used raw audio+text concat without speech markers.

    // Suffix: im_end + assistant + <think>\n\n</think>\n\n
    let suffix_tokens: Vec<i32> = vec![
        151645,  // <|im_end|>
        198,     // \n
        151644,  // <|im_start|>
        77091,   // assistant
        198,     // \n
        151667,  // <think>
        198,     // \n
        198,     // \n
        151668,  // </think>
        198,     // \n
        198,     // \n
    ];

    // Build full token sequence: prefix + [audio placeholders] + suffix
    // (no instruction text in user turn - it's in the system prompt)
    let mut all_tokens: Vec<i32> = Vec::new();
    all_tokens.extend_from_slice(&prefix_tokens);
    let audio_start_idx = all_tokens.len();
    for _ in 0..audio_len {
        all_tokens.push(0);  // placeholders
    }
    let audio_end_idx = all_tokens.len();
    all_tokens.extend_from_slice(&suffix_tokens);

    // Get embeddings for all tokens
    let token_array = mlx_rs::Array::from_slice(&all_tokens, &[1, all_tokens.len() as i32]);
    let embeddings = embed_tokens(llm, &token_array)?;
    mlx_rs::transforms::eval([&embeddings])?;

    // Splice: replace placeholder region with actual audio features
    let prefix_embed = embeddings.index((.., ..audio_start_idx as i32, ..));
    let suffix_embed = embeddings.index((.., audio_end_idx as i32.., ..));

    let combined = mlx_rs::ops::concatenate_axis(
        &[&prefix_embed, audio_features, &suffix_embed],
        1,
    ).map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
    mlx_rs::transforms::eval([&combined])?;

    Ok(combined)
}

fn get_logits(llm: &mut qwen3_mlx::Model, hidden: &mlx_rs::Array) -> Result<mlx_rs::Array> {
    match &mut llm.lm_head {
        Some(lm_head) => lm_head.forward(hidden)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e))),
        None => match &mut llm.model.embed_tokens {
            MaybeQuantized::Original(embed) => embed.as_linear(hidden)
                .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e))),
            MaybeQuantized::Quantized(embed) => embed.as_linear(hidden)
                .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e))),
        },
    }
}

fn embed_tokens(llm: &mut qwen3_mlx::Model, token_array: &mlx_rs::Array) -> Result<mlx_rs::Array> {
    match &mut llm.model.embed_tokens {
        MaybeQuantized::Original(embed) => embed.forward(token_array)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e))),
        MaybeQuantized::Quantized(embed) => embed.forward(token_array)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e))),
    }
}

/// Sample with top-k filtering and presence penalty.
/// Replaces greedy sampling (temperature=0) with Qwen3-recommended parameters.
fn sample_top_k_p(
    logits: &mlx_rs::Array,
    temperature: f32,
    top_k: usize,
    generated_tokens: &[i32],
    presence_penalty: f32,
) -> Result<mlx_rs::Array> {
    // Greedy fallback when no penalties needed
    if temperature == 0.0 && (presence_penalty == 0.0 || generated_tokens.is_empty()) {
        return sample(logits, 0.0)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)));
    }

    let shape = logits.shape();
    let vocab_size = *shape.last().unwrap() as usize;

    // 1. Apply presence penalty: subtract flat penalty from logits of any previously generated token
    let mut modified = logits.clone();
    if presence_penalty > 0.0 && !generated_tokens.is_empty() {
        let mut penalty_data = vec![0.0f32; vocab_size];
        for &tok in generated_tokens {
            if (tok as usize) < vocab_size {
                penalty_data[tok as usize] = presence_penalty;
            }
        }
        let penalty = mlx_rs::Array::from_slice(&penalty_data, &[1, vocab_size as i32]);
        modified = mlx_rs::ops::subtract(&modified, &penalty)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
    }

    // 2. Scale by temperature
    modified = modified.multiply(mlx_rs::array!(1.0 / temperature))
        .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;

    // 3. Top-k filtering: keep only top_k logits, set rest to -inf
    if top_k > 0 && top_k < vocab_size {
        let topk_vals = mlx_rs::ops::indexing::topk_axis_device(
            &modified, top_k as i32, -1, mlx_rs::StreamOrDevice::default()
        ).map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
        // k-th largest value = last element of topk (sorted descending)
        let threshold = topk_vals.index((.., (top_k as i32 - 1)));
        let threshold = threshold.reshape(&[1, 1])
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
        // Mask: True where logits >= threshold
        let mask = modified.ge(&threshold)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
        let neg_inf = mlx_rs::array!(f32::NEG_INFINITY);
        modified = mlx_rs::ops::r#where(&mask, &modified, &neg_inf)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
    }

    // 4. Sample using categorical (temp=1.0 since we already scaled)
    sample(&modified, 1.0)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))
}

/// Compute Shannon entropy of logits distribution (CPU, fast for single vector)
fn compute_entropy(logits: &mlx_rs::Array) -> f32 {
    if mlx_rs::transforms::eval([logits]).is_err() {
        return f32::MAX;
    }
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
/// Returns true if mean pairwise cosine similarity > 0.95.
fn check_embedding_collapse(embeddings: &mlx_rs::Array) -> bool {
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
    total_cos / count as f32 > 0.95
}

fn format_timestamp(seconds: usize) -> String {
    let m = seconds / 60;
    let s = seconds % 60;
    format!("{:02}:{:02}", m, s)
}

/// Transcribe one audio chunk
fn transcribe_chunk(
    chunk_samples: &[f32],
    mel_frontend: &MelFrontendMLX,
    encoder: &mut SenseVoiceEncoder,
    adaptor: &mut AudioAdaptorQwen4B,
    llm: &mut qwen3_mlx::Model,
    tokenizer: &tokenizers::Tokenizer,
    template_mode: TemplateMode,
) -> Result<(String, Vec<i32>)> {
    // Audio pipeline: mel → LFR → encoder → adaptor
    let mel = mel_frontend.compute_mel_spectrogram(chunk_samples)?;
    let mel_lfr = apply_lfr(&mel, 7, 6)?;
    let encoder_out = encoder.forward(&mel_lfr)?;
    mlx_rs::transforms::eval([&encoder_out])?;
    let audio_features = adaptor.forward(&encoder_out)?;
    mlx_rs::transforms::eval([&audio_features])?;

    // Check for embedding collapse (all audio tokens ≈ same vector → skip)
    if check_embedding_collapse(&audio_features) {
        return Ok(("".to_string(), vec![]));
    }

    // Build embeddings based on template mode
    let embeddings = match template_mode {
        TemplateMode::ChatML => build_multimodal_embeddings(&audio_features, llm, tokenizer)?,
        TemplateMode::Raw => audio_features,  // Training-matched: just audio features
    };

    // Generate
    let mut cache: Vec<Option<KVCache>> = (0..llm.model.layers.len())
        .map(|_| Some(KVCache::default()))
        .collect();

    let mask = match create_attention_mask(&embeddings, &cache, Some(true))
        .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?
    {
        Some(AttentionMask::Array(m)) => Some(m),
        _ => None,
    };

    // Prefill
    let mut hidden = embeddings;
    for (layer, c) in llm.model.layers.iter_mut().zip(cache.iter_mut()) {
        hidden = layer.forward(AttentionInput {
            x: &hidden,
            mask: mask.as_ref(),
            cache: c.as_mut(),
        }).map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
    }
    hidden = llm.model.norm.forward(&hidden)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;

    let last_hidden = hidden.index((.., -1, ..));
    let logits = get_logits(llm, &last_hidden)?;

    let mut tokens: Vec<i32> = Vec::new();
    let mut token = sample_top_k_p(&logits, TEMPERATURE, TOP_K, &tokens, PRESENCE_PENALTY)?;
    mlx_rs::transforms::eval([&token])?;
    let mut token_id: i32 = token.item();
    let mut low_entropy_count: usize = 0;

    for _step in 0..MAX_ASR_TOKENS {
        // Stop conditions depend on template mode
        if token_id == EOS_TOKEN {
            break;
        }
        if template_mode == TemplateMode::ChatML && token_id == IM_END_TOKEN {
            break;
        }
        tokens.push(token_id);

        // Early stop on n-gram repetition (catches multi-token phrase loops)
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
                // Strip trailing repetitions: keep 1 copy, remove the rest
                let pattern: Vec<i32> = tokens[tokens.len() - repeat_n..].to_vec();
                let mut pos = tokens.len();
                while pos >= repeat_n {
                    if tokens[pos - repeat_n..pos] == pattern[..] {
                        pos -= repeat_n;
                    } else {
                        break;
                    }
                }
                // Keep exactly 1 copy of the repeated pattern
                tokens.truncate(pos + repeat_n);
                break;
            }
        }

        let token_array = mlx_rs::Array::from_slice(&[token_id], &[1, 1]);
        let y_embed = embed_tokens(llm, &token_array)?;

        let mut h = y_embed;
        for (layer, c) in llm.model.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(AttentionInput {
                x: &h,
                mask: None,
                cache: c.as_mut(),
            }).map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
        }
        h = llm.model.norm.forward(&h)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;

        let logits = get_logits(llm, &h.index((.., -1, ..)))?;

        // Entropy monitoring: detect degenerate low-entropy states
        let entropy = compute_entropy(&logits);
        if entropy < ENTROPY_THRESHOLD {
            low_entropy_count += 1;
            if low_entropy_count >= ENTROPY_WINDOW {
                break;  // Model stuck in degenerate state
            }
        } else {
            low_entropy_count = 0;
        }

        token = sample_top_k_p(&logits, TEMPERATURE, TOP_K, &tokens, PRESENCE_PENALTY)?;
        mlx_rs::transforms::eval([&token])?;
        token_id = token.item();
    }

    let token_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&token_ids, true)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::Tokenizer(format!("{}", e)))?;
    Ok((text, tokens))
}

fn main() -> Result<()> {
    let audio_path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/rust_talk.wav".to_string());
    let chunk_seconds: usize = std::env::args().nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let template_mode = std::env::args().nth(3)
        .map(|s| if s == "raw" { TemplateMode::Raw } else { TemplateMode::ChatML })
        .unwrap_or(TemplateMode::ChatML);
    let chunk_samples = chunk_seconds * SAMPLE_RATE;

    let mode_name = if template_mode == TemplateMode::Raw { "raw (training-matched)" } else { "ChatML" };

    // Prefer 4-bit quantized model for ~3x faster inference
    let qwen_path = if std::path::Path::new("models/Qwen3-4B-4bit/config.json").exists() {
        eprintln!("Using 4-bit quantized model");
        "models/Qwen3-4B-4bit"
    } else {
        eprintln!("Using BF16 model (quantize with scripts for 3x speedup)");
        "models/Qwen3-4B"
    };
    let sensevoice_path = "sensevoice_iic.safetensors";
    let adaptor_path = "adaptor_phase2_final.safetensors";

    for (name, path) in [("Qwen3-4B", qwen_path), ("SenseVoice", sensevoice_path), ("Adaptor", adaptor_path)] {
        if !std::path::Path::new(path).exists() {
            eprintln!("Missing: {} at {}", name, path);
            return Ok(());
        }
    }
    if !std::path::Path::new(&audio_path).exists() {
        eprintln!("Audio not found: {}", audio_path);
        return Ok(());
    }

    // === 1. Load audio ===
    println!("=== Loading audio ===");
    let (samples, sample_rate) = load_wav(&audio_path)?;
    let total_duration = samples.len() as f32 / sample_rate as f32;
    println!("  File: {}", audio_path);
    println!("  Duration: {:.1}s ({:.1} min)", total_duration, total_duration / 60.0);

    let samples = if sample_rate != 16000 {
        println!("  Resampling {}Hz -> 16kHz...", sample_rate);
        resample(&samples, sample_rate, 16000)?
    } else {
        samples
    };

    let n_chunks = (samples.len() + chunk_samples - 1) / chunk_samples;
    println!("  Chunks: {} x {}s", n_chunks, chunk_seconds);
    println!("  Template: {}", mode_name);

    // === 2. Load models (once) ===
    println!("\n=== Loading models ===");
    let audio_config = AudioConfig::default();
    let mel_frontend = MelFrontendMLX::new(audio_config)?;
    let warmup_samples = vec![0.0f32; SAMPLE_RATE];
    let _ = mel_frontend.compute_mel_spectrogram(&warmup_samples)?;
    println!("  Mel frontend ready (MLX GPU)");

    let mut encoder = SenseVoiceEncoder::new(SenseVoiceEncoderConfig::default())?;
    encoder.load_weights(sensevoice_path)?;
    println!("  SenseVoice encoder loaded");

    let mut adaptor = AudioAdaptorQwen4B::new()?;
    adaptor.load_weights(adaptor_path)?;
    println!("  Audio adaptor loaded");

    let mut llm = load_model(qwen_path)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
    let tokenizer = load_tokenizer(qwen_path)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::Tokenizer(format!("{:?}", e)))?;
    println!("  Qwen3-4B loaded");

    // === 3. Transcribe all chunks ===
    println!("\n=== Transcribing ({} chunks x {}s, {}) ===\n", n_chunks, chunk_seconds, mode_name);
    let asr_start = std::time::Instant::now();
    let mut all_transcriptions: Vec<String> = Vec::new();
    let mut silent_count = 0;
    let mut meta_count = 0;

    for chunk_idx in 0..n_chunks {
        let start_sample = chunk_idx * chunk_samples;
        let end_sample = (start_sample + chunk_samples).min(samples.len());
        let chunk = &samples[start_sample..end_sample];

        let start_sec = chunk_idx * chunk_seconds;
        let end_sec = start_sec + (chunk.len() / SAMPLE_RATE);

        // Skip silent chunks
        if is_silent(chunk, SILENCE_THRESHOLD_DB) {
            silent_count += 1;
            println!("[{}/{}] {}-{} (skip: silence)",
                chunk_idx + 1, n_chunks,
                format_timestamp(start_sec), format_timestamp(end_sec),
            );
            continue;
        }

        let chunk_start = std::time::Instant::now();

        let (raw_text, _tokens) = transcribe_chunk(
            chunk, &mel_frontend, &mut encoder, &mut adaptor, &mut llm, &tokenizer, template_mode
        )?;

        let text = clean_transcription(&raw_text);
        let chunk_time = chunk_start.elapsed();

        if text.is_empty() && !raw_text.trim().is_empty() {
            meta_count += 1;
        }

        // Truncate display for long text
        let display_text = if text.is_empty() {
            if raw_text.trim().is_empty() {
                "(empty)".to_string()
            } else {
                "(filtered)".to_string()
            }
        } else if text.chars().count() > 60 {
            let end = text.char_indices().nth(60).map_or(text.len(), |(i, _)| i);
            format!("{}...", &text[..end])
        } else {
            text.clone()
        };

        println!("[{}/{}] {}-{} ({:.1}s) {}",
            chunk_idx + 1, n_chunks,
            format_timestamp(start_sec), format_timestamp(end_sec),
            chunk_time.as_secs_f64(),
            display_text,
        );

        if !text.is_empty() {
            all_transcriptions.push(text);
        }
    }

    let asr_time = asr_start.elapsed();
    let full_chinese = all_transcriptions.join("");

    // === 4. Print full Chinese transcript ===
    println!("\n=== Chinese Transcript ({:.1}s ASR time) ===\n", asr_time.as_secs_f64());
    println!("{}", full_chinese);

    // Save Chinese transcript
    std::fs::write("/tmp/rust_talk_transcript_zh.txt", &full_chinese)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::Audio(format!("Write failed: {}", e)))?;
    println!("\n  Saved: /tmp/rust_talk_transcript_zh.txt");

    // === Summary ===
    println!("\n=== Summary ===");
    println!("Audio: {:.1}s ({:.1} min)", total_duration, total_duration / 60.0);
    println!("Chunks: {} x {}s (template: {})", n_chunks, chunk_seconds, mode_name);
    println!("Silent chunks skipped: {}", silent_count);
    println!("Meta-commentary filtered: {}", meta_count);
    println!("ASR time: {:.1}s (RTF: {:.2}x)", asr_time.as_secs_f64(), asr_time.as_secs_f64() / total_duration as f64);
    println!("Chinese chars: {}", full_chinese.chars().count());
    println!("File: /tmp/rust_talk_transcript_zh.txt");

    Ok(())
}
