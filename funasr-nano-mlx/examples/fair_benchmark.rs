//! Fair benchmark comparing inference time only (no file I/O)

use std::time::Instant;
use funasr_nano_mlx::audio;
use funasr_nano_mlx::model::FunASRNano;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio_path = "/tmp/voice_memo.wav";
    let model_dir = "./Fun-ASR-Nano-2512";
    let iterations = 10;

    // Load model
    println!("Loading model from {}...", model_dir);
    let mut model = FunASRNano::load(model_dir)?;

    // Pre-load audio (outside timing)
    println!("Pre-loading audio: {}", audio_path);
    let (samples, sample_rate) = audio::load_wav(audio_path)?;
    let samples = audio::resample(&samples, sample_rate, 16000)?;
    let duration_secs = samples.len() as f32 / 16000.0;
    println!("Audio: {:.2}s at 16kHz ({} samples)", duration_secs, samples.len());

    // Warmup
    println!("\nWarmup run...");
    let result = model.transcribe_samples(&samples, 16000)?;
    println!("Result: {}...", &result.chars().take(80).collect::<String>());

    // Benchmark (inference only, no file I/O)
    println!("\nBenchmarking {} iterations (inference only)...", iterations);
    let mut times = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let start = Instant::now();
        let _result = model.transcribe_samples(&samples, 16000)?;
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
    let rtf_mean = (mean / 1000.0) / duration_secs as f64;
    let rtf_min = (min / 1000.0) / duration_secs as f64;

    println!("\n=== funasr-nano-mlx (Inference Only) ===");
    println!("Audio: {:.2}s", duration_secs);
    println!("\nLatency (ms):");
    println!("  Min:    {:.1}", min);
    println!("  Max:    {:.1}", max);
    println!("  Mean:   {:.1}", mean);
    println!("  Median: {:.1}", median);
    println!("\nReal-Time Factor:");
    println!("  Mean RTF: {:.4}x ({:.1}x real-time)", rtf_mean, 1.0 / rtf_mean);
    println!("  Best RTF: {:.4}x ({:.1}x real-time)", rtf_min, 1.0 / rtf_min);

    Ok(())
}
