//! AISHELL-1 Benchmark for funasr-qwen4b-mlx
//!
//! Computes CER (Character Error Rate) on AISHELL-1 test set.
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
//! cargo run --example benchmark_aishell --release -- /tmp/data_aishell [--max N] [--model-dir DIR]
//! ```

use funasr_qwen4b_mlx::FunASRQwen4B;
use funasr_qwen4b_mlx::audio::{load_wav, resample};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Strip punctuation for fair CER comparison (references are unpunctuated).
fn strip_punctuation(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace() && !is_punctuation(*c))
        .collect()
}

fn is_punctuation(c: char) -> bool {
    matches!(c,
        '。' | '，' | '、' | '？' | '！' | '；' | '：' | '"' | '"' |
        '\u{2018}' | '\u{2019}' | '【' | '】' | '《' | '》' | '（' | '）' | '—' |
        '…' | '·' | '～' | '「' | '」' | '﹑' | '＋' |
        '.' | ',' | '?' | '!' | ';' | ':' | '"' | '\'' | '(' | ')' |
        '[' | ']' | '{' | '}' | '-' | '/' | '\\' | '~'
    )
}

/// Compute Character Error Rate using Levenshtein distance.
/// Returns (edit_distance, substitutions, insertions, deletions).
fn compute_cer(reference: &str, hypothesis: &str) -> (usize, usize, usize, usize) {
    let ref_chars: Vec<char> = strip_punctuation(reference).chars().collect();
    let hyp_chars: Vec<char> = strip_punctuation(hypothesis).chars().collect();

    let n = ref_chars.len();
    let m = hyp_chars.len();

    if n == 0 {
        return (m, 0, m, 0);
    }
    if m == 0 {
        return (n, n, 0, 0);
    }

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
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    let edit_distance = dp[n][m];

    // Backtrack to count S, D, I
    let mut i = n;
    let mut j = m;
    let (mut substitutions, mut deletions, mut insertions) = (0, 0, 0);

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

/// Load AISHELL transcript file (space-separated characters).
fn load_transcripts(transcript_path: &Path) -> HashMap<String, String> {
    let content = fs::read_to_string(transcript_path).expect("Failed to read transcript");
    let mut transcripts = HashMap::new();
    for line in content.lines() {
        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        if parts.len() == 2 {
            transcripts.insert(parts[0].to_string(), parts[1].to_string());
        }
    }
    transcripts
}

/// Recursively find all WAV files under a directory.
fn find_test_files(test_dir: &Path) -> Vec<(String, std::path::PathBuf)> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(test_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
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

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut aishell_dir = "/tmp/data_aishell".to_string();
    let mut max_samples: usize = 200;
    let mut model_dir = ".".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--max" => {
                i += 1;
                max_samples = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(200);
            }
            "--model-dir" => {
                i += 1;
                model_dir = args.get(i).cloned().unwrap_or_else(|| ".".to_string());
            }
            s if !s.starts_with("--") && i == 1 => {
                aishell_dir = s.to_string();
            }
            _ => {}
        }
        i += 1;
    }

    let aishell_path = Path::new(&aishell_dir);
    let transcript_path = aishell_path.join("transcript/aishell_transcript_v0.8.txt");
    let test_wav_dir = aishell_path.join("wav/test");

    if !transcript_path.exists() {
        eprintln!("Transcript not found: {:?}", transcript_path);
        eprintln!("\nDownload AISHELL-1:");
        eprintln!("  cd /tmp && wget https://openslr.trmal.net/resources/33/data_aishell.tgz");
        eprintln!("  tar -xzf data_aishell.tgz");
        return Ok(());
    }
    if !test_wav_dir.exists() {
        eprintln!("Test WAV directory not found: {:?}", test_wav_dir);
        return Ok(());
    }

    println!("=== AISHELL-1 CER Benchmark ===\n");

    let transcripts = load_transcripts(&transcript_path);
    println!("Loaded {} transcripts", transcripts.len());

    let test_files = find_test_files(&test_wav_dir);
    println!("Found {} test files", test_files.len());

    let num_to_test = max_samples.min(test_files.len());
    println!("Testing {} samples\n", num_to_test);

    // Load model using high-level API (handles ChatML, anti-repetition, etc.)
    println!("Loading FunASR-Qwen4B from '{}'...", model_dir);
    let mut model = FunASRQwen4B::load(&model_dir)?;
    println!("Model loaded.\n");

    println!("=== Running Benchmark ===\n");
    let start_time = std::time::Instant::now();

    let mut total_ref_chars = 0usize;
    let mut total_errors = 0usize;
    let mut total_substitutions = 0usize;
    let mut total_insertions = 0usize;
    let mut total_deletions = 0usize;
    let mut processed = 0usize;
    let mut skipped = 0usize;
    let mut total_audio_secs = 0.0f64;

    for (idx, (utt_id, wav_path)) in test_files.iter().take(num_to_test).enumerate() {
        let reference = match transcripts.get(utt_id) {
            Some(t) => t.clone(),
            None => { skipped += 1; continue; }
        };

        let (samples, sample_rate) = match load_wav(wav_path) {
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

        total_audio_secs += samples.len() as f64 / 16000.0;

        let hypothesis = match model.transcribe_samples(&samples, 16000) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("[{}/{}] {} - Transcribe error: {:?}", idx + 1, num_to_test, utt_id, e);
                skipped += 1;
                continue;
            }
        };

        let ref_chars = reference.chars().filter(|c| !c.is_whitespace()).count();
        let (errors, subs, ins, dels) = compute_cer(&reference, &hypothesis);

        total_ref_chars += ref_chars;
        total_errors += errors;
        total_substitutions += subs;
        total_insertions += ins;
        total_deletions += dels;
        processed += 1;

        let cer = if ref_chars > 0 { errors as f64 / ref_chars as f64 * 100.0 } else { 0.0 };

        if (idx + 1) % 10 == 0 || idx + 1 == num_to_test {
            let running_cer = if total_ref_chars > 0 {
                total_errors as f64 / total_ref_chars as f64 * 100.0
            } else { 0.0 };
            println!("[{}/{}] Running CER: {:.2}%", idx + 1, num_to_test, running_cer);
        }

        // Show first 5 examples and any with very high CER
        if idx < 5 || cer > 50.0 {
            println!("  REF: {}", reference);
            println!("  HYP: {}", hypothesis);
            println!("  CER: {:.2}% (S:{} I:{} D:{})\n", cer, subs, ins, dels);
        }
    }

    let elapsed = start_time.elapsed();

    println!("\n=== AISHELL-1 Benchmark Results ===\n");

    let final_cer = if total_ref_chars > 0 {
        total_errors as f64 / total_ref_chars as f64 * 100.0
    } else { 0.0 };

    println!("Samples processed: {}", processed);
    println!("Samples skipped:   {}", skipped);
    println!("Total ref chars:   {}", total_ref_chars);
    println!("Total errors:      {} (S:{} I:{} D:{})",
        total_errors, total_substitutions, total_insertions, total_deletions);
    println!();
    println!("**CER: {:.2}%**", final_cer);
    println!();
    println!("Audio duration: {:.1}s", total_audio_secs);
    println!("Wall time:      {:.1}s ({:.2} utt/s)",
        elapsed.as_secs_f64(), processed as f64 / elapsed.as_secs_f64());
    println!("RTF:            {:.2}x", elapsed.as_secs_f64() / total_audio_secs);

    println!("\n=== Comparison ===\n");
    println!("| Model | CER |");
    println!("|-------|-----|");
    println!("| FunASR (7.7B) | 3.38% |");
    println!("| FunASR-Nano (0.8B) | 4.22% |");
    println!("| Phase 2 adaptor only (8-bit) | 5.10% |");
    println!("| Paraformer v2 (0.2B) | 6.23% |");
    println!("| **Phase 3 + LoRA r=16 (8-bit)** | **{:.2}%** |", final_cer);

    Ok(())
}
