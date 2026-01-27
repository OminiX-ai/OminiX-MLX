//! End-to-End ASR Performance Benchmark
//!
//! Measures the full pipeline performance:
//! 1. Audio preprocessing (mel spectrogram)
//! 2. Encoder forward pass
//! 3. LLM generation
//! 4. Total end-to-end latency
//!
//! Usage:
//!     cargo run --release --example benchmark_e2e -- <audio.wav> [iterations]

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use step_audio2_mlx::audio::{load_wav, resample, compute_mel_spectrogram};
use step_audio2_mlx::config::AudioConfig;
use step_audio2_mlx::{StepAudio2, Result, Error};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("End-to-End ASR Performance Benchmark");
        eprintln!();
        eprintln!("Usage: {} <audio.wav> [iterations]", args[0]);
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} ./test.wav 5", args[0]);
        return Err(Error::Config("Missing audio file argument".into()));
    }

    let audio_path = PathBuf::from(&args[1]);
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    // Find model
    let model_dirs = [
        "./Step-Audio-2-mini",
        "../Step-Audio-2-mini",
        "step-audio2-mlx/Step-Audio-2-mini",
    ];

    let model_dir = model_dirs.iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
        .ok_or_else(|| Error::Config("Step-Audio-2-mini model not found".into()))?;

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║          END-TO-END ASR PERFORMANCE BENCHMARK                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Load audio
    println!("Loading audio: {}", audio_path.display());
    let (samples, sample_rate) = load_wav(&audio_path)?;
    let audio_duration = samples.len() as f64 / sample_rate as f64;
    println!("Audio: {:.2}s at {}Hz ({} samples)", audio_duration, sample_rate, samples.len());

    // Resample to 16kHz
    let samples_16k = if sample_rate != 16000 {
        println!("Resampling to 16kHz...");
        resample(&samples, sample_rate, 16000)
    } else {
        samples.clone()
    };
    println!();

    // Load model
    println!("Loading model from {}...", model_dir.display());
    let load_start = Instant::now();
    let mut model = StepAudio2::load(&model_dir)?;
    let load_time = load_start.elapsed().as_secs_f64();
    println!("Model loaded in {:.2}s", load_time);
    println!();

    // Warmup run
    println!("Warmup run...");
    let warmup_start = Instant::now();
    let warmup_result = model.transcribe_samples(&samples_16k, 16000)?;
    let warmup_time = warmup_start.elapsed().as_secs_f64() * 1000.0;
    println!("Warmup completed in {:.1}ms", warmup_time);
    println!("Output: {}", warmup_result);
    println!();

    // Benchmark iterations
    println!("Running {} benchmark iterations...", iterations);
    println!();

    let mut total_times = Vec::with_capacity(iterations);
    let mut audio_times = Vec::with_capacity(iterations);

    for i in 0..iterations {
        // Measure audio preprocessing separately
        let audio_start = Instant::now();
        let config = AudioConfig::default();
        let mel = compute_mel_spectrogram(&samples_16k, &config)?;
        mel.eval()?;
        let audio_time = audio_start.elapsed().as_secs_f64() * 1000.0;
        audio_times.push(audio_time);

        // Measure total E2E time
        let total_start = Instant::now();
        let result = model.transcribe_samples(&samples_16k, 16000)?;
        let total_time = total_start.elapsed().as_secs_f64() * 1000.0;
        total_times.push(total_time);

        println!("  [{}/{}] Audio: {:.1}ms | Total: {:.1}ms | Output: {}...",
                 i + 1, iterations, audio_time, total_time,
                 result.chars().take(40).collect::<String>());
    }
    println!();

    // Calculate statistics
    let calc_stats = |times: &[f64]| -> (f64, f64, f64) {
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        (min, max, mean)
    };

    let (audio_min, audio_max, audio_mean) = calc_stats(&audio_times);
    let (total_min, total_max, total_mean) = calc_stats(&total_times);

    // Infer encoder + LLM time
    let model_mean = total_mean - audio_mean;
    let model_min = total_min - audio_min;

    // RTF calculation
    let total_rtf = (total_mean / 1000.0) / audio_duration;
    let audio_rtf = (audio_mean / 1000.0) / audio_duration;

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                        BENCHMARK RESULTS                             ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Audio Duration: {:.2}s                                                ║", audio_duration);
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Component          │    Min (ms)    │    Mean (ms)   │    Max (ms)   ║");
    println!("╠════════════════════╪════════════════╪════════════════╪═══════════════╣");
    println!("║ Audio Preprocess   │ {:>14.1} │ {:>14.1} │ {:>13.1} ║", audio_min, audio_mean, audio_max);
    println!("║ Encoder + LLM      │ {:>14.1} │ {:>14.1} │ {:>13.1} ║", model_min, model_mean, total_max - audio_max);
    println!("╠════════════════════╪════════════════╪════════════════╪═══════════════╣");
    println!("║ TOTAL E2E          │ {:>14.1} │ {:>14.1} │ {:>13.1} ║", total_min, total_mean, total_max);
    println!("╚════════════════════╧════════════════╧════════════════╧═══════════════╝");
    println!();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                     REAL-TIME FACTOR (RTF)                           ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Audio Preprocessing: RTF = {:.4}x ({:.1}x real-time)                  ║", audio_rtf, 1.0/audio_rtf);
    println!("║ Total E2E:           RTF = {:.4}x ({:.1}x real-time)                  ║", total_rtf, 1.0/total_rtf);
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Performance breakdown
    let audio_pct = (audio_mean / total_mean) * 100.0;
    let model_pct = 100.0 - audio_pct;

    println!("Performance Breakdown:");
    println!("  Audio preprocessing: {:.1}% ({:.1}ms)", audio_pct, audio_mean);
    println!("  Encoder + LLM:       {:.1}% ({:.1}ms)", model_pct, model_mean);
    println!();

    if total_rtf < 1.0 {
        println!("Status: REAL-TIME CAPABLE ({:.1}x faster than real-time)", 1.0/total_rtf);
    } else {
        println!("Status: NOT REAL-TIME (RTF = {:.2}x)", total_rtf);
    }

    Ok(())
}
