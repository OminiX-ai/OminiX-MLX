//! Benchmark ASR pipeline against Emilia dataset with ground truth comparison.
//!
//! Usage: cargo run --example benchmark_emilia --release -- /tmp/emilia_zh/wavs /tmp/emilia_zh/samples
//!
//! Args:
//!   1. Directory containing .wav files (16kHz mono)
//!   2. Directory containing .json metadata files (ground truth)

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
const TEMPERATURE: f32 = 0.6;
const TOP_K: usize = 20;
const PRESENCE_PENALTY: f32 = 1.0;
const ENTROPY_THRESHOLD: f32 = 0.5;
const ENTROPY_WINDOW: usize = 5;
const EOS_TOKEN: i32 = 151643;
const IM_END_TOKEN: i32 = 151645;

fn build_multimodal_embeddings(
    audio_features: &mlx_rs::Array,
    llm: &mut qwen3_mlx::Model,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<mlx_rs::Array> {
    let audio_len = audio_features.shape()[1] as usize;
    let system_prompt = "你是语音转写系统。直接输出语音内容的中文文字，不要添加任何解释、评论或格式。";
    let system_encoding = tokenizer.encode(system_prompt, false)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::Tokenizer(format!("{}", e)))?;

    let mut prefix_tokens: Vec<i32> = vec![151644, 8948, 198];
    for &tok in system_encoding.get_ids() {
        prefix_tokens.push(tok as i32);
    }
    prefix_tokens.extend_from_slice(&[151645, 198, 151644, 872, 198]);

    let suffix_tokens: Vec<i32> = vec![
        151645, 198, 151644, 77091, 198, 151667, 198, 198, 151668, 198, 198,
    ];

    let mut all_tokens: Vec<i32> = Vec::new();
    all_tokens.extend_from_slice(&prefix_tokens);
    let audio_start_idx = all_tokens.len();
    for _ in 0..audio_len {
        all_tokens.push(0);
    }
    let audio_end_idx = all_tokens.len();
    all_tokens.extend_from_slice(&suffix_tokens);

    let token_array = mlx_rs::Array::from_slice(&all_tokens, &[1, all_tokens.len() as i32]);
    let embeddings = embed_tokens(llm, &token_array)?;
    mlx_rs::transforms::eval([&embeddings])?;

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

fn sample_top_k_p(
    logits: &mlx_rs::Array,
    temperature: f32,
    top_k: usize,
    generated_tokens: &[i32],
    presence_penalty: f32,
) -> Result<mlx_rs::Array> {
    if temperature == 0.0 && (presence_penalty == 0.0 || generated_tokens.is_empty()) {
        return sample(logits, 0.0)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)));
    }

    let shape = logits.shape();
    let vocab_size = *shape.last().unwrap() as usize;

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

    modified = modified.multiply(mlx_rs::array!(1.0 / temperature))
        .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;

    if top_k > 0 && top_k < vocab_size {
        let topk_vals = mlx_rs::ops::indexing::topk_axis_device(
            &modified, top_k as i32, -1, mlx_rs::StreamOrDevice::default()
        ).map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
        let threshold = topk_vals.index((.., (top_k as i32 - 1)));
        let threshold = threshold.reshape(&[1, 1])
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
        let mask = modified.ge(&threshold)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
        let neg_inf = mlx_rs::array!(f32::NEG_INFINITY);
        modified = mlx_rs::ops::r#where(&mask, &modified, &neg_inf)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
    }

    sample(&modified, 1.0)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))
}

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

fn transcribe_single(
    samples: &[f32],
    mel_frontend: &MelFrontendMLX,
    encoder: &mut SenseVoiceEncoder,
    adaptor: &mut AudioAdaptorQwen4B,
    llm: &mut qwen3_mlx::Model,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<String> {
    let mel = mel_frontend.compute_mel_spectrogram(samples)?;
    let mel_lfr = apply_lfr(&mel, 7, 6)?;
    let encoder_out = encoder.forward(&mel_lfr)?;
    mlx_rs::transforms::eval([&encoder_out])?;
    let audio_features = adaptor.forward(&encoder_out)?;
    mlx_rs::transforms::eval([&audio_features])?;

    let embeddings = build_multimodal_embeddings(&audio_features, llm, tokenizer)?;

    let mut cache: Vec<Option<KVCache>> = (0..llm.model.layers.len())
        .map(|_| Some(KVCache::default()))
        .collect();

    let mask = match create_attention_mask(&embeddings, &cache, Some(true))
        .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?
    {
        Some(AttentionMask::Array(m)) => Some(m),
        _ => None,
    };

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
        if token_id == EOS_TOKEN || token_id == IM_END_TOKEN {
            break;
        }
        tokens.push(token_id);

        // N-gram repetition detection
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

        let entropy = compute_entropy(&logits);
        if entropy < ENTROPY_THRESHOLD {
            low_entropy_count += 1;
            if low_entropy_count >= ENTROPY_WINDOW {
                break;
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
    Ok(text)
}

/// Compute Character Error Rate between hypothesis and reference.
/// Simple edit distance on character level.
fn compute_cer(hypothesis: &str, reference: &str) -> f32 {
    let hyp: Vec<char> = hypothesis.chars().filter(|c| !c.is_whitespace()).collect();
    let ref_chars: Vec<char> = reference.chars().filter(|c| !c.is_whitespace()).collect();

    if ref_chars.is_empty() {
        return if hyp.is_empty() { 0.0 } else { 1.0 };
    }

    let m = hyp.len();
    let n = ref_chars.len();
    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if hyp[i - 1] == ref_chars[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[m][n] as f32 / n as f32
}

fn main() -> Result<()> {
    let wav_dir = std::env::args().nth(1).unwrap_or_else(|| "/tmp/emilia_zh/wavs".to_string());
    let json_dir = std::env::args().nth(2).unwrap_or_else(|| "/tmp/emilia_zh/samples".to_string());

    let qwen_path = if std::path::Path::new("models/Qwen3-4B-4bit/config.json").exists() {
        eprintln!("Using 4-bit quantized model");
        "models/Qwen3-4B-4bit"
    } else {
        eprintln!("Using BF16 model");
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

    // Collect WAV files
    let mut wav_files: Vec<String> = std::fs::read_dir(&wav_dir)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::Audio(format!("{}", e)))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "wav"))
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();
    wav_files.sort();

    println!("=== Emilia-ZH Benchmark ===");
    println!("WAV directory: {}", wav_dir);
    println!("JSON directory: {}", json_dir);
    println!("Files: {}", wav_files.len());

    // === Load models ===
    println!("\n=== Loading models ===");
    let audio_config = AudioConfig::default();
    let mel_frontend = MelFrontendMLX::new(audio_config)?;
    let warmup_samples = vec![0.0f32; SAMPLE_RATE];
    let _ = mel_frontend.compute_mel_spectrogram(&warmup_samples)?;
    println!("  Mel frontend ready");

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

    // === Process each file ===
    println!("\n=== Transcribing {} files ===\n", wav_files.len());
    let total_start = std::time::Instant::now();

    let mut total_cer = 0.0f32;
    let mut total_audio_duration = 0.0f32;
    let mut count = 0;
    let mut results: Vec<(String, String, String, f32, f32)> = Vec::new(); // (id, ref, hyp, cer, duration)

    for wav_path in &wav_files {
        let filename = std::path::Path::new(wav_path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();

        // Load ground truth from JSON
        let json_path = format!("{}/{}.json", json_dir, filename);
        let reference = if std::path::Path::new(&json_path).exists() {
            let json_str = std::fs::read_to_string(&json_path)
                .map_err(|e| funasr_qwen4b_mlx::error::Error::Audio(format!("{}", e)))?;
            let json: serde_json::Value = serde_json::from_str(&json_str)
                .map_err(|e| funasr_qwen4b_mlx::error::Error::Audio(format!("{}", e)))?;
            json["text"].as_str().unwrap_or("").to_string()
        } else {
            String::new()
        };

        // Load audio
        let (samples, sample_rate) = load_wav(wav_path)?;
        let audio_duration = samples.len() as f32 / sample_rate as f32;
        total_audio_duration += audio_duration;

        let samples = if sample_rate != 16000 {
            resample(&samples, sample_rate, 16000)?
        } else {
            samples
        };

        let chunk_start = std::time::Instant::now();
        let hypothesis = transcribe_single(
            &samples, &mel_frontend, &mut encoder, &mut adaptor, &mut llm, &tokenizer,
        )?;
        let infer_time = chunk_start.elapsed().as_secs_f64();

        let cer = if !reference.is_empty() {
            compute_cer(&hypothesis, &reference)
        } else {
            -1.0
        };

        let ref_display = if reference.chars().count() > 40 {
            let end = reference.char_indices().nth(40).map_or(reference.len(), |(i, _)| i);
            format!("{}...", &reference[..end])
        } else {
            reference.clone()
        };
        let hyp_display = if hypothesis.chars().count() > 40 {
            let end = hypothesis.char_indices().nth(40).map_or(hypothesis.len(), |(i, _)| i);
            format!("{}...", &hypothesis[..end])
        } else {
            hypothesis.clone()
        };

        let cer_str = if cer >= 0.0 { format!("{:.1}%", cer * 100.0) } else { "N/A".to_string() };
        println!("[{}/{}] {} ({:.1}s audio, {:.1}s infer, CER: {})",
            count + 1, wav_files.len(), filename, audio_duration, infer_time, cer_str);
        println!("  REF: {}", ref_display);
        println!("  HYP: {}", hyp_display);
        println!();

        if cer >= 0.0 {
            total_cer += cer;
        }
        results.push((filename, reference, hypothesis, cer, audio_duration));
        count += 1;
    }

    let total_time = total_start.elapsed();

    // === Summary ===
    let valid_count = results.iter().filter(|(_, _, _, cer, _)| *cer >= 0.0).count();
    let avg_cer = if valid_count > 0 { total_cer / valid_count as f32 } else { 0.0 };

    println!("=== Summary ===");
    println!("Files: {}", count);
    println!("Total audio: {:.1}s", total_audio_duration);
    println!("Total inference: {:.1}s", total_time.as_secs_f64());
    println!("RTF: {:.2}x", total_time.as_secs_f64() / total_audio_duration as f64);
    println!("Average CER: {:.1}% ({} samples with ground truth)", avg_cer * 100.0, valid_count);
    println!();

    // CER distribution
    println!("=== CER Distribution ===");
    let mut cer_buckets = [0usize; 5]; // 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
    for (_, _, _, cer, _) in &results {
        if *cer >= 0.0 {
            let bucket = ((*cer * 5.0).floor() as usize).min(4);
            cer_buckets[bucket] += 1;
        }
    }
    let labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"];
    for (label, &count) in labels.iter().zip(cer_buckets.iter()) {
        println!("  {}: {} samples", label, count);
    }

    Ok(())
}
