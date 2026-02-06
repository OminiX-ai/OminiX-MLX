//! AISHELL-1 Benchmark for funasr-qwen4b-mlx
//!
//! Computes CER (Character Error Rate) on AISHELL-1 test set
//! and compares with reported Fun-ASR-Nano performance.
//!
//! Setup:
//! ```bash
//! # Download AISHELL-1 (15GB)
//! cd /tmp
//! wget https://openslr.trmal.net/resources/33/data_aishell.tgz
//! tar -xzf data_aishell.tgz
//! ```
//!
//! Run:
//! ```bash
//! cargo run --example benchmark_aishell --release -- /tmp/data_aishell [max_samples]
//! ```

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
use std::collections::HashMap;
use std::fs;
use std::path::Path;

const SAMPLE_RATE: usize = 16000;
const MAX_ASR_TOKENS: usize = 100;
const TEMPERATURE: f32 = 0.6;
const TOP_K: usize = 20;
const PRESENCE_PENALTY: f32 = 1.0;
const EOS_TOKEN: i32 = 151643;
const IM_END_TOKEN: i32 = 151645;

/// Compute Character Error Rate using Levenshtein distance
fn compute_cer(reference: &str, hypothesis: &str) -> (usize, usize, usize, usize) {
    let ref_chars: Vec<char> = reference.chars().filter(|c| !c.is_whitespace()).collect();
    let hyp_chars: Vec<char> = hypothesis.chars().filter(|c| !c.is_whitespace()).collect();

    let n = ref_chars.len();
    let m = hyp_chars.len();

    if n == 0 {
        return (m, 0, m, 0); // All insertions
    }
    if m == 0 {
        return (n, n, 0, 0); // All deletions
    }

    // Dynamic programming for edit distance with operation tracking
    let mut dp = vec![vec![0usize; m + 1]; n + 1];

    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }

    for i in 1..=n {
        for j in 1..=m {
            let cost = if ref_chars[i - 1] == hyp_chars[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)          // deletion
                .min(dp[i][j - 1] + 1)              // insertion
                .min(dp[i - 1][j - 1] + cost);      // substitution
        }
    }

    let edit_distance = dp[n][m];

    // Backtrack to count S, D, I
    let mut i = n;
    let mut j = m;
    let mut substitutions = 0;
    let mut deletions = 0;
    let mut insertions = 0;

    while i > 0 || j > 0 {
        if i > 0 && j > 0 && ref_chars[i - 1] == hyp_chars[j - 1] {
            i -= 1;
            j -= 1;
        } else if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1 {
            substitutions += 1;
            i -= 1;
            j -= 1;
        } else if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            deletions += 1;
            i -= 1;
        } else if j > 0 && dp[i][j] == dp[i][j - 1] + 1 {
            insertions += 1;
            j -= 1;
        } else {
            break;
        }
    }

    (edit_distance, substitutions, insertions, deletions)
}

/// Load AISHELL transcript file
fn load_transcripts(transcript_path: &Path) -> Result<HashMap<String, String>> {
    let content = fs::read_to_string(transcript_path)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::Audio(format!("Failed to read transcript: {}", e)))?;

    let mut transcripts = HashMap::new();
    for line in content.lines() {
        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        if parts.len() == 2 {
            let utt_id = parts[0].to_string();
            let text = parts[1].to_string();
            transcripts.insert(utt_id, text);
        }
    }

    Ok(transcripts)
}

/// Find all test WAV files
fn find_test_files(test_dir: &Path) -> Vec<(String, std::path::PathBuf)> {
    let mut files = Vec::new();

    if let Ok(entries) = fs::read_dir(test_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Recurse into speaker directories
                files.extend(find_test_files(&path));
            } else if path.extension().map_or(false, |e| e == "wav") {
                if let Some(stem) = path.file_stem() {
                    files.push((stem.to_string_lossy().to_string(), path));
                }
            }
        }
    }

    files.sort_by(|a, b| a.0.cmp(&b.0));
    files
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

fn sample_top_k(
    logits: &mlx_rs::Array,
    temperature: f32,
    top_k: usize,
    generated_tokens: &[i32],
    presence_penalty: f32,
) -> Result<mlx_rs::Array> {
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

/// Transcribe single utterance (simplified, no ChatML for benchmark)
fn transcribe_utterance(
    samples: &[f32],
    mel_frontend: &MelFrontendMLX,
    encoder: &mut SenseVoiceEncoder,
    adaptor: &mut AudioAdaptorQwen4B,
    llm: &mut qwen3_mlx::Model,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<String> {
    // Audio pipeline
    let mel = mel_frontend.compute_mel_spectrogram(samples)?;
    let mel_lfr = apply_lfr(&mel, 7, 6)?;
    let encoder_out = encoder.forward(&mel_lfr)?;
    mlx_rs::transforms::eval([&encoder_out])?;
    let audio_features = adaptor.forward(&encoder_out)?;
    mlx_rs::transforms::eval([&audio_features])?;

    // Generate (raw mode - just audio features)
    let mut cache: Vec<Option<KVCache>> = (0..llm.model.layers.len())
        .map(|_| Some(KVCache::default()))
        .collect();

    let mask = match create_attention_mask(&audio_features, &cache, Some(true))
        .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?
    {
        Some(AttentionMask::Array(m)) => Some(m),
        _ => None,
    };

    let mut hidden = audio_features;
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
    let mut token = sample_top_k(&logits, TEMPERATURE, TOP_K, &tokens, PRESENCE_PENALTY)?;
    mlx_rs::transforms::eval([&token])?;
    let mut token_id: i32 = token.item();

    for _ in 0..MAX_ASR_TOKENS {
        if token_id == EOS_TOKEN || token_id == IM_END_TOKEN {
            break;
        }
        tokens.push(token_id);

        // N-gram repetition check
        if tokens.len() >= 6 {
            let n = 3;
            if tokens.len() >= n * 2 {
                let tail = &tokens[tokens.len() - n * 2..];
                if tail[..n] == tail[n..] {
                    tokens.truncate(tokens.len() - n);
                    break;
                }
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
        token = sample_top_k(&logits, TEMPERATURE, TOP_K, &tokens, PRESENCE_PENALTY)?;
        mlx_rs::transforms::eval([&token])?;
        token_id = token.item();
    }

    let token_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&token_ids, true)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::Tokenizer(format!("{}", e)))?;

    // Clean up
    let text = text.trim()
        .replace("<think>", "")
        .replace("</think>", "")
        .trim()
        .to_string();

    Ok(text)
}

fn main() -> Result<()> {
    let aishell_dir = std::env::args().nth(1).unwrap_or_else(|| "/tmp/data_aishell".to_string());
    let max_samples: usize = std::env::args().nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100); // Default to 100 for quick test

    let aishell_path = Path::new(&aishell_dir);
    let transcript_path = aishell_path.join("transcript/aishell_transcript_v0.8.txt");
    let test_wav_dir = aishell_path.join("wav/test");

    // Check paths
    if !transcript_path.exists() {
        eprintln!("Transcript not found: {:?}", transcript_path);
        eprintln!("\nDownload AISHELL-1:");
        eprintln!("  cd /tmp");
        eprintln!("  wget https://openslr.trmal.net/resources/33/data_aishell.tgz");
        eprintln!("  tar -xzf data_aishell.tgz");
        return Ok(());
    }
    if !test_wav_dir.exists() {
        eprintln!("Test WAV directory not found: {:?}", test_wav_dir);
        return Ok(());
    }

    println!("=== AISHELL-1 Benchmark ===\n");

    // Load transcripts
    println!("Loading transcripts...");
    let transcripts = load_transcripts(&transcript_path)?;
    println!("  Loaded {} transcripts", transcripts.len());

    // Find test files
    println!("Finding test files...");
    let test_files = find_test_files(&test_wav_dir);
    println!("  Found {} test files", test_files.len());

    let num_to_test = max_samples.min(test_files.len());
    println!("  Testing {} samples\n", num_to_test);

    // Load models
    println!("Loading models...");
    let qwen_path = if Path::new("models/Qwen3-4B-4bit/config.json").exists() {
        "models/Qwen3-4B-4bit"
    } else {
        "models/Qwen3-4B"
    };
    let sensevoice_path = "sensevoice_iic.safetensors";
    let adaptor_path = "adaptor_phase2_final.safetensors";

    for (name, path) in [("Qwen3-4B", qwen_path), ("SenseVoice", sensevoice_path), ("Adaptor", adaptor_path)] {
        if !Path::new(path).exists() {
            eprintln!("Missing: {} at {}", name, path);
            return Ok(());
        }
    }

    let audio_config = AudioConfig::default();
    let mel_frontend = MelFrontendMLX::new(audio_config)?;

    let mut encoder = SenseVoiceEncoder::new(SenseVoiceEncoderConfig::default())?;
    encoder.load_weights(sensevoice_path)?;

    let mut adaptor = AudioAdaptorQwen4B::new()?;
    adaptor.load_weights(adaptor_path)?;

    let mut llm = load_model(qwen_path)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
    let tokenizer = load_tokenizer(qwen_path)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::Tokenizer(format!("{:?}", e)))?;

    println!("  Models loaded\n");

    // Run benchmark
    println!("=== Running Benchmark ===\n");
    let start_time = std::time::Instant::now();

    let mut total_ref_chars = 0usize;
    let mut total_errors = 0usize;
    let mut total_substitutions = 0usize;
    let mut total_insertions = 0usize;
    let mut total_deletions = 0usize;
    let mut processed = 0usize;
    let mut skipped = 0usize;

    for (idx, (utt_id, wav_path)) in test_files.iter().take(num_to_test).enumerate() {
        // Get reference transcript
        let reference = match transcripts.get(utt_id) {
            Some(t) => t.clone(),
            None => {
                skipped += 1;
                continue;
            }
        };

        // Load and resample audio
        let (samples, sample_rate) = match load_wav(&wav_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[{}/{}] {} - Load error: {:?}", idx + 1, num_to_test, utt_id, e);
                skipped += 1;
                continue;
            }
        };

        let samples = if sample_rate != 16000 {
            resample(&samples, sample_rate, 16000)?
        } else {
            samples
        };

        // Transcribe
        let hypothesis = match transcribe_utterance(
            &samples, &mel_frontend, &mut encoder, &mut adaptor, &mut llm, &tokenizer
        ) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("[{}/{}] {} - Transcribe error: {:?}", idx + 1, num_to_test, utt_id, e);
                skipped += 1;
                continue;
            }
        };

        // Compute CER
        let ref_chars = reference.chars().filter(|c| !c.is_whitespace()).count();
        let (errors, subs, ins, dels) = compute_cer(&reference, &hypothesis);

        total_ref_chars += ref_chars;
        total_errors += errors;
        total_substitutions += subs;
        total_insertions += ins;
        total_deletions += dels;
        processed += 1;

        let cer = if ref_chars > 0 { errors as f64 / ref_chars as f64 * 100.0 } else { 0.0 };

        // Progress output
        if (idx + 1) % 10 == 0 || idx + 1 == num_to_test {
            let running_cer = if total_ref_chars > 0 {
                total_errors as f64 / total_ref_chars as f64 * 100.0
            } else {
                0.0
            };
            println!("[{}/{}] CER: {:.2}% | Running CER: {:.2}%",
                idx + 1, num_to_test, cer, running_cer);
        }

        // Show some examples
        if idx < 5 || cer > 50.0 {
            println!("  REF: {}", reference);
            println!("  HYP: {}", hypothesis);
            println!("  CER: {:.2}% (S:{} I:{} D:{})\n", cer, subs, ins, dels);
        }
    }

    let elapsed = start_time.elapsed();

    // Final results
    println!("\n=== AISHELL-1 Benchmark Results ===\n");

    let final_cer = if total_ref_chars > 0 {
        total_errors as f64 / total_ref_chars as f64 * 100.0
    } else {
        0.0
    };

    println!("Samples processed: {}", processed);
    println!("Samples skipped: {}", skipped);
    println!("Total reference chars: {}", total_ref_chars);
    println!("Total errors: {} (S:{} I:{} D:{})",
        total_errors, total_substitutions, total_insertions, total_deletions);
    println!();
    println!("**CER: {:.2}%**", final_cer);
    println!();
    println!("Time: {:.1}s ({:.2} utterances/sec)",
        elapsed.as_secs_f64(),
        processed as f64 / elapsed.as_secs_f64());

    println!("\n=== Comparison with Reported Results ===\n");
    println!("| Model | Opensource WER/CER |");
    println!("|-------|-------------------|");
    println!("| Fun-ASR (7.7B) | 3.38% |");
    println!("| Fun-ASR-Nano (0.8B) | 4.22% |");
    println!("| Paraformer v2 (0.2B) | 6.23% |");
    println!("| **funasr-qwen4b-mlx** | **{:.2}%** |", final_cer);

    Ok(())
}
