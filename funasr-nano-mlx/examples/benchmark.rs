//! Benchmark FunASR-Nano Rust implementation
//!
//! Usage:
//!   cargo run --release --example benchmark -- <audio.wav> [iterations]
//!
//! Model path: Uses FUNASR_NANO_MODEL_PATH env var or ~/.OminiX/models/funasr-nano

use std::env;
use std::time::Instant;

use funasr_nano_mlx::{audio, FunASRNano, default_model_path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let model_dir = default_model_path();
    let default_audio = model_dir.join("example/zh.wav");
    let audio_path = args.get(1).map(|s| s.to_string())
        .unwrap_or_else(|| default_audio.to_string_lossy().to_string());
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    // Load model
    println!("Loading model from {}...", model_dir.display());
    let load_start = Instant::now();
    let mut model = FunASRNano::load(&model_dir)?;
    let load_time = load_start.elapsed();
    println!("Model loaded in {:.2}s", load_time.as_secs_f32());

    // Load audio to get duration
    println!("\nLoading audio: {}", audio_path);
    let (samples, sample_rate) = audio::load_wav(audio_path)?;
    let duration_secs = samples.len() as f32 / sample_rate as f32;
    println!("Audio: {:.2}s at {}Hz", duration_secs, sample_rate);

    // Warmup
    println!("\nWarmup run...");
    let result = model.transcribe(audio_path)?;
    let preview_len = result.chars().take(100).map(|c| c.len_utf8()).sum::<usize>();
    println!("Result: {}...", &result[..preview_len.min(result.len())]);

    // Benchmark
    println!("\nBenchmarking {} iterations...", iterations);
    let mut times = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let start = Instant::now();
        let _result = model.transcribe(audio_path)?;
        let elapsed = start.elapsed().as_millis() as f64;
        times.push(elapsed);

        if (i + 1) % 5 == 0 || i == iterations - 1 {
            println!("  [{}/{}] {:.1} ms", i + 1, iterations, elapsed);
        }
    }

    // Statistics
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times[0];
    let max = times[times.len() - 1];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let median = times[times.len() / 2];

    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let std_dev = variance.sqrt();

    let rtf_mean = (mean / 1000.0) / duration_secs as f64;
    let rtf_min = (min / 1000.0) / duration_secs as f64;

    println!("\n=== Rust FunASR-Nano Results ===");
    println!("Audio: {:.2}s", duration_secs);
    println!();
    println!("Latency (ms):");
    println!("  Min:    {:.1}", min);
    println!("  Max:    {:.1}", max);
    println!("  Mean:   {:.1}", mean);
    println!("  Median: {:.1}", median);
    println!("  Std:    {:.1}", std_dev);
    println!();
    println!("Real-Time Factor:");
    println!("  Mean RTF: {:.4}x ({:.1}x real-time)", rtf_mean, 1.0 / rtf_mean);
    println!("  Best RTF: {:.4}x ({:.1}x real-time)", rtf_min, 1.0 / rtf_min);

    Ok(())
}
