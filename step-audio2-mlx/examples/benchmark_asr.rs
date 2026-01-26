//! Unified ASR Benchmark
//!
//! Compares ASR performance across three MLX implementations:
//! - step-audio2-mlx (Whisper encoder + Qwen2.5-7B)
//! - funasr-mlx (Paraformer)
//! - funasr-nano-mlx (SenseVoice + Qwen)
//!
//! Usage:
//!     cargo run --release --example benchmark_asr -- <audio.wav> [iterations]
//!
//! Each model must be available in the expected locations:
//! - Step-Audio-2-mini in ./Step-Audio-2-mini
//! - Paraformer in ./paraformer
//! - FunASR-Nano in ./Fun-ASR-Nano-2512

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use step_audio2_mlx::audio::{load_wav, resample};
use step_audio2_mlx::{StepAudio2, Result, Error};

/// Benchmark statistics
#[derive(Debug, Clone)]
struct BenchmarkStats {
    name: String,
    audio_duration: f64,
    iterations: usize,
    times_ms: Vec<f64>,
    min_ms: f64,
    max_ms: f64,
    mean_ms: f64,
    median_ms: f64,
    std_dev_ms: f64,
    rtf_mean: f64,
    rtf_best: f64,
    sample_output: String,
}

impl BenchmarkStats {
    fn new(name: &str, audio_duration: f64, times: Vec<f64>, sample_output: String) -> Self {
        let iterations = times.len();
        let mut sorted = times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let mean = sorted.iter().sum::<f64>() / iterations as f64;
        let median = sorted[iterations / 2];

        let variance = sorted.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / iterations as f64;
        let std_dev = variance.sqrt();

        let rtf_mean = (mean / 1000.0) / audio_duration;
        let rtf_best = (min / 1000.0) / audio_duration;

        Self {
            name: name.to_string(),
            audio_duration,
            iterations,
            times_ms: times,
            min_ms: min,
            max_ms: max,
            mean_ms: mean,
            median_ms: median,
            std_dev_ms: std_dev,
            rtf_mean,
            rtf_best,
            sample_output,
        }
    }

    fn print(&self) {
        println!("\n=== {} ===", self.name);
        println!("Audio: {:.2}s", self.audio_duration);
        println!("\nLatency (ms):");
        println!("  Min:    {:.1}", self.min_ms);
        println!("  Max:    {:.1}", self.max_ms);
        println!("  Mean:   {:.1}", self.mean_ms);
        println!("  Median: {:.1}", self.median_ms);
        println!("  Std:    {:.1}", self.std_dev_ms);
        println!("\nReal-Time Factor:");
        println!("  Mean RTF: {:.4}x ({:.1}x real-time)", self.rtf_mean, 1.0 / self.rtf_mean);
        println!("  Best RTF: {:.4}x ({:.1}x real-time)", self.rtf_best, 1.0 / self.rtf_best);
        println!("\nSample output: {}...", &self.sample_output.chars().take(80).collect::<String>());
    }
}

/// Benchmark step-audio2-mlx
fn benchmark_step_audio2(
    samples: &[f32],
    sample_rate: u32,
    audio_duration: f64,
    iterations: usize,
    model_dir: &str,
) -> Option<BenchmarkStats> {
    println!("\n--- Benchmarking step-audio2-mlx ---");

    let model_path = PathBuf::from(model_dir);
    if !model_path.exists() {
        println!("Model not found at {}, skipping", model_dir);
        return None;
    }

    println!("Loading model from {}...", model_dir);
    let load_start = Instant::now();
    let mut model = match StepAudio2::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            println!("Failed to load model: {}", e);
            return None;
        }
    };
    println!("Model loaded in {:.2}s", load_start.elapsed().as_secs_f64());

    // Warmup
    println!("Warmup run...");
    let warmup_result = match model.transcribe_samples(samples, sample_rate) {
        Ok(r) => r,
        Err(e) => {
            println!("Warmup failed: {}", e);
            return None;
        }
    };

    // Benchmark
    println!("Benchmarking {} iterations...", iterations);
    let mut times = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let start = Instant::now();
        let _ = model.transcribe_samples(samples, sample_rate);
        let elapsed = start.elapsed().as_millis() as f64;
        times.push(elapsed);

        if (i + 1) % 5 == 0 || i == iterations - 1 {
            println!("  [{}/{}] {:.1} ms", i + 1, iterations, elapsed);
        }
    }

    Some(BenchmarkStats::new(
        "step-audio2-mlx (Whisper + Qwen2.5-7B)",
        audio_duration,
        times,
        warmup_result,
    ))
}

fn print_comparison(results: &[BenchmarkStats]) {
    if results.is_empty() {
        println!("\nNo benchmark results to compare.");
        return;
    }

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                     ASR BENCHMARK COMPARISON                         ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Model                              │ Mean (ms) │ RTF    │ Speed      ║");
    println!("╠════════════════════════════════════╪═══════════╪════════╪════════════╣");

    for stats in results {
        let speed = 1.0 / stats.rtf_mean;
        let name_truncated: String = stats.name.chars().take(34).collect();
        println!(
            "║ {:34} │ {:>9.1} │ {:>6.4} │ {:>6.1}x RT ║",
            name_truncated,
            stats.mean_ms,
            stats.rtf_mean,
            speed
        );
    }

    println!("╚════════════════════════════════════╧═══════════╧════════╧════════════╝");

    // Find fastest
    if let Some(fastest) = results.iter().min_by(|a, b| {
        a.mean_ms.partial_cmp(&b.mean_ms).unwrap()
    }) {
        println!("\nFastest: {} ({:.1} ms mean)", fastest.name, fastest.mean_ms);

        // Show speedup relative to others
        for stats in results {
            if stats.name != fastest.name {
                let speedup = stats.mean_ms / fastest.mean_ms;
                println!("  vs {}: {:.2}x faster", stats.name, speedup);
            }
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Unified ASR Benchmark");
        eprintln!("");
        eprintln!("Usage: {} <audio.wav> [iterations]", args[0]);
        eprintln!("");
        eprintln!("Arguments:");
        eprintln!("  audio.wav   Path to audio file (WAV format)");
        eprintln!("  iterations  Number of benchmark iterations (default: 10)");
        eprintln!("");
        eprintln!("Model directories (must exist):");
        eprintln!("  ./Step-Audio-2-mini    - Step-Audio 2 model");
        eprintln!("  ./paraformer           - FunASR Paraformer model");
        eprintln!("  ./Fun-ASR-Nano-2512    - FunASR Nano model");
        eprintln!("");
        eprintln!("Example:");
        eprintln!("  {} ./speech.wav 20", args[0]);
        return Err(Error::Config("Invalid arguments".into()));
    }

    let audio_path = PathBuf::from(&args[1]);
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    if !audio_path.exists() {
        return Err(Error::Audio(format!(
            "Audio file not found: {}",
            audio_path.display()
        )));
    }

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              UNIFIED ASR BENCHMARK - step-audio2-mlx                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!("");

    // Load audio
    println!("Loading audio: {}", audio_path.display());
    let (samples, sample_rate) = load_wav(&audio_path)?;
    let audio_duration = samples.len() as f64 / sample_rate as f64;
    println!("Audio: {:.2}s at {}Hz ({} samples)", audio_duration, sample_rate, samples.len());

    // Resample to 16kHz for all models
    let samples_16k = if sample_rate != 16000 {
        println!("Resampling to 16kHz...");
        resample(&samples, sample_rate, 16000)
    } else {
        samples.clone()
    };
    println!("Resampled: {} samples at 16kHz", samples_16k.len());

    // Run benchmarks
    let mut results = Vec::new();

    // Benchmark step-audio2-mlx
    if let Some(stats) = benchmark_step_audio2(
        &samples_16k,
        16000,
        audio_duration,
        iterations,
        "./Step-Audio-2-mini",
    ) {
        stats.print();
        results.push(stats);
    }

    // Note: funasr-mlx and funasr-nano-mlx would need to be benchmarked separately
    // as they are different crates. This benchmark focuses on step-audio2-mlx
    // but the structure supports adding more results.

    println!("\n");
    println!("Note: To compare with funasr-mlx and funasr-nano-mlx, run their");
    println!("respective benchmark examples:");
    println!("  cd ../funasr-mlx && cargo run --release --example benchmark -- <audio.wav>");
    println!("  cd ../funasr-nano-mlx && cargo run --release --example benchmark -- <audio.wav>");

    // Print comparison if we have results
    if !results.is_empty() {
        for stats in &results {
            stats.print();
        }
    }

    Ok(())
}
