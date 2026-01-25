//! Benchmark Paraformer performance
//!
//! Usage:
//!   cargo run --release --example benchmark -- <audio.wav> <model_dir> [iterations]

use std::env;
use std::time::Instant;

use funasr_mlx::audio::{load_wav, resample};
use funasr_mlx::{load_model, parse_cmvn_file};
use mlx_rs::module::Module;
use mlx_rs::transforms::eval;
use mlx_rs::Array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <audio.wav> <model_dir> [iterations]", args[0]);
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let model_dir = &args[2];
    let iterations: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);

    let weights_path = format!("{}/paraformer.safetensors", model_dir);
    let cmvn_path = format!("{}/am.mvn", model_dir);

    // Load audio
    println!("Loading audio: {}", audio_path);
    let (samples, sample_rate) = load_wav(audio_path)?;
    let duration_secs = samples.len() as f32 / sample_rate as f32;
    println!("  {:.2}s audio", duration_secs);

    let samples = if sample_rate != 16000 {
        resample(&samples, sample_rate, 16000)
    } else {
        samples
    };

    let audio = Array::from_slice(&samples, &[samples.len() as i32]);

    // Load model
    println!("\nLoading model...");
    let mut model = load_model(&weights_path)?;
    model.training_mode(false);

    let (addshift, rescale) = parse_cmvn_file(&cmvn_path)?;
    model.set_cmvn(addshift, rescale);

    // Warmup
    println!("\nWarmup run...");
    let token_ids = model.transcribe(&audio)?;
    eval([&token_ids])?;
    let num_tokens = token_ids.shape()[1];
    println!("  {} tokens generated", num_tokens);

    // Benchmark
    println!("\nBenchmarking {} iterations...", iterations);
    let mut times = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let start = Instant::now();
        let token_ids = model.transcribe(&audio)?;
        eval([&token_ids])?;
        let elapsed = start.elapsed();
        times.push(elapsed.as_millis() as f64);

        if (i + 1) % 5 == 0 || i == iterations - 1 {
            println!("  [{}/{}] {} ms", i + 1, iterations, elapsed.as_millis());
        }
    }

    // Calculate statistics
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times[0];
    let max = times[times.len() - 1];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let median = times[times.len() / 2];

    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let std_dev = variance.sqrt();

    let rtf_mean = (mean / 1000.0) / duration_secs as f64;
    let rtf_min = (min / 1000.0) / duration_secs as f64;

    println!("\n=== Benchmark Results ===");
    println!("Audio: {:.2}s, {} tokens", duration_secs, num_tokens);
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
