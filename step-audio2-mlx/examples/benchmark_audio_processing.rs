//! Audio Processing Performance Benchmark
//!
//! Measures the performance of GPU-accelerated vs CPU audio processing.
//!
//! Usage:
//!     cargo run --release --example benchmark_audio_processing -- [duration_secs]
//!
//! Default: 5 seconds of synthetic audio, 10 iterations

use std::env;
use std::time::Instant;

use step_audio2_mlx::config::AudioConfig;

/// Generate synthetic audio for benchmarking
fn generate_sine_wave(duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    let freq = 440.0; // A4 note

    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
        })
        .collect()
}

/// CPU-based mel spectrogram (for comparison)
fn compute_mel_cpu(samples: &[f32], config: &AudioConfig) -> Vec<f32> {
    use std::f32::consts::PI;

    let n_fft = config.n_fft as usize;
    let hop_length = config.hop_length as usize;
    let n_mels = config.n_mels as usize;
    let n_freqs = n_fft / 2 + 1;
    let sample_rate = config.sample_rate;
    let fmin = config.fmin;
    let fmax = config.fmax.unwrap_or(sample_rate as f32 / 2.0);

    // Add padding
    let padding = 479;
    let mut padded_samples = samples.to_vec();
    padded_samples.extend(vec![0.0f32; padding]);

    // Create Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|i| {
            let t = i as f32 / (n_fft - 1) as f32;
            0.5 - 0.5 * (2.0 * PI * t).cos()
        })
        .collect();

    // Calculate frames
    let n_frames = if padded_samples.len() >= n_fft {
        (padded_samples.len() - n_fft) / hop_length + 1
    } else {
        0
    };
    let effective_frames = n_frames.saturating_sub(1).max(1);

    // STFT (CPU DFT)
    let mut power = vec![0.0f32; n_freqs * effective_frames];
    for frame in 0..effective_frames {
        let start = frame * hop_length;
        let mut windowed = vec![0.0f32; n_fft];
        for i in 0..n_fft {
            if start + i < padded_samples.len() {
                windowed[i] = padded_samples[start + i] * window[i];
            }
        }

        for k in 0..n_freqs {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            for n in 0..n_fft {
                let angle = 2.0 * PI * k as f32 * n as f32 / n_fft as f32;
                real += windowed[n] * angle.cos();
                imag -= windowed[n] * angle.sin();
            }
            power[k * effective_frames + frame] = real * real + imag * imag;
        }
    }

    // Create mel filterbank
    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..=(n_mels + 1) {
        let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
        mel_points.push(mel_to_hz(mel));
    }

    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
        .collect();

    let mut filterbank = vec![0.0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let f_left = mel_points[m];
        let f_center = mel_points[m + 1];
        let f_right = mel_points[m + 2];

        for k in 0..n_freqs {
            let freq = fft_freqs[k];
            if freq >= f_left && freq <= f_center {
                filterbank[m * n_freqs + k] = (freq - f_left) / (f_center - f_left);
            } else if freq > f_center && freq <= f_right {
                filterbank[m * n_freqs + k] = (f_right - freq) / (f_right - f_center);
            }
        }
    }

    // Apply mel filterbank
    let mut mel_spec = vec![0.0f32; n_mels * effective_frames];
    for m in 0..n_mels {
        for t in 0..effective_frames {
            let mut sum = 0.0f32;
            for f in 0..n_freqs {
                sum += filterbank[m * n_freqs + f] * power[f * effective_frames + t];
            }
            mel_spec[m * effective_frames + t] = sum;
        }
    }

    // Log and normalize
    let epsilon = 1e-10f32;
    for v in &mut mel_spec {
        *v = (*v).max(epsilon).log10();
    }
    let max_val = mel_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for v in &mut mel_spec {
        *v = (*v).max(max_val - 8.0);
        *v = (*v + 4.0) / 4.0;
    }

    mel_spec
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let duration_secs: f32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5.0);
    let iterations = 10;

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║           AUDIO PROCESSING PERFORMANCE BENCHMARK                     ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let config = AudioConfig::default();
    println!("Config: n_fft={}, hop_length={}, n_mels={}, sample_rate={}",
             config.n_fft, config.hop_length, config.n_mels, config.sample_rate);
    println!("Audio duration: {:.1}s", duration_secs);
    println!("Iterations: {}", iterations);
    println!();

    // Generate test audio
    let samples = generate_sine_wave(duration_secs, config.sample_rate as u32);
    let num_samples = samples.len();
    println!("Generated {} samples ({:.2} MB)", num_samples, num_samples as f64 * 4.0 / 1024.0 / 1024.0);
    println!();

    // Warmup
    println!("Warming up...");
    let _ = compute_mel_cpu(&samples, &config);
    let _ = step_audio2_mlx::audio::compute_mel_spectrogram(&samples, &config);
    println!();

    // Benchmark CPU
    println!("--- CPU Benchmark ---");
    let mut cpu_times = Vec::with_capacity(iterations);
    for i in 0..iterations {
        let start = Instant::now();
        let result = compute_mel_cpu(&samples, &config);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        cpu_times.push(elapsed);
        if i == 0 {
            println!("  Output shape: [1, {}, {}]", config.n_mels, result.len() / config.n_mels as usize);
        }
    }

    let cpu_mean = cpu_times.iter().sum::<f64>() / iterations as f64;
    let cpu_min = cpu_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let cpu_max = cpu_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("  Min:  {:.2} ms", cpu_min);
    println!("  Max:  {:.2} ms", cpu_max);
    println!("  Mean: {:.2} ms", cpu_mean);
    println!();

    // Benchmark GPU
    println!("--- GPU Benchmark (MLX) ---");
    let mut gpu_times = Vec::with_capacity(iterations);
    for i in 0..iterations {
        let start = Instant::now();
        let result = step_audio2_mlx::audio::compute_mel_spectrogram(&samples, &config).unwrap();
        // Force full evaluation by reading a value
        result.eval().unwrap();
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        gpu_times.push(elapsed);
        if i == 0 {
            println!("  Output shape: {:?}", result.shape());
        }
    }

    let gpu_mean = gpu_times.iter().sum::<f64>() / iterations as f64;
    let gpu_min = gpu_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let gpu_max = gpu_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("  Min:  {:.2} ms", gpu_min);
    println!("  Max:  {:.2} ms", gpu_max);
    println!("  Mean: {:.2} ms", gpu_mean);
    println!();

    // Summary
    let speedup = cpu_mean / gpu_mean;
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                          RESULTS                                     ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Implementation │    Min (ms)    │    Mean (ms)   │   Speedup        ║");
    println!("╠════════════════╪════════════════╪════════════════╪══════════════════╣");
    println!("║ CPU (baseline) │ {:>14.2} │ {:>14.2} │      1.00x       ║", cpu_min, cpu_mean);
    println!("║ GPU (MLX)      │ {:>14.2} │ {:>14.2} │ {:>10.1}x       ║", gpu_min, gpu_mean, speedup);
    println!("╚════════════════╧════════════════╧════════════════╧══════════════════╝");
    println!();

    if speedup > 1.0 {
        println!("GPU is {:.1}x faster than CPU for {:.1}s audio", speedup, duration_secs);
    } else {
        println!("CPU is {:.1}x faster than GPU for {:.1}s audio (GPU overhead)", 1.0/speedup, duration_secs);
    }

    // RTF calculation
    let cpu_rtf = (cpu_mean / 1000.0) / duration_secs as f64;
    let gpu_rtf = (gpu_mean / 1000.0) / duration_secs as f64;
    println!();
    println!("Real-Time Factor (audio processing only):");
    println!("  CPU: {:.4}x ({:.1}x real-time)", cpu_rtf, 1.0 / cpu_rtf);
    println!("  GPU: {:.4}x ({:.1}x real-time)", gpu_rtf, 1.0 / gpu_rtf);
}
