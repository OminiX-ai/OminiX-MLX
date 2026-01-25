//! Compare STFT performance: Manual DFT vs FFT
//!
//! This benchmark demonstrates the ~45x speedup from using FFT instead of manual DFT.
//!
//! Usage:
//!   cargo run --release --example compare_stft

use std::f32::consts::PI;
use std::sync::Arc;
use std::time::Instant;

use rustfft::{num_complex::Complex, FftPlanner};

/// Manual DFT implementation (O(N²)) - the OLD approach
fn compute_stft_manual_dft(
    samples: &[f32],
    window: &[f32],
    n_fft: usize,
    hop_length: usize,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;

    let n_frames = if samples.len() >= n_fft {
        (samples.len() - n_fft) / hop_length + 1
    } else {
        0
    };

    if n_frames == 0 {
        return vec![0.0f32; n_freqs];
    }

    let mut power_spec = vec![0.0f32; n_frames * n_freqs];

    for frame in 0..n_frames {
        let start = frame * hop_length;

        let mut windowed = vec![0.0f32; n_fft];
        for i in 0..n_fft {
            windowed[i] = samples[start + i] * window[i];
        }

        // O(N²) manual DFT
        for k in 0..n_freqs {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for n in 0..n_fft {
                let angle = 2.0 * PI * k as f32 * n as f32 / n_fft as f32;
                real += windowed[n] * angle.cos();
                imag -= windowed[n] * angle.sin();
            }

            power_spec[frame * n_freqs + k] = real * real + imag * imag;
        }
    }

    power_spec
}

/// FFT-based STFT (O(N log N)) - the NEW approach
fn compute_stft_fft(
    samples: &[f32],
    window: &[f32],
    n_fft: usize,
    hop_length: usize,
    fft: &Arc<dyn rustfft::Fft<f32>>,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;

    let n_frames = if samples.len() >= n_fft {
        (samples.len() - n_fft) / hop_length + 1
    } else {
        0
    };

    if n_frames == 0 {
        return vec![0.0f32; n_freqs];
    }

    let mut power_spec = vec![0.0f32; n_frames * n_freqs];
    let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n_fft];

    for frame in 0..n_frames {
        let start = frame * hop_length;

        // Apply window and convert to complex
        for i in 0..n_fft {
            buffer[i] = Complex::new(samples[start + i] * window[i], 0.0);
        }

        // O(N log N) FFT
        fft.process(&mut buffer);

        // Extract power spectrum
        for k in 0..n_freqs {
            let c = buffer[k];
            power_spec[frame * n_freqs + k] = c.re * c.re + c.im * c.im;
        }
    }

    power_spec
}

fn main() {
    println!("=== STFT Performance Comparison ===\n");
    println!("Comparing Manual DFT (O(N²)) vs FFT (O(N log N))\n");

    // Configuration matching Paraformer
    let n_fft = 400; // 25ms window at 16kHz
    let hop_length = 160; // 10ms hop
    let sample_rate = 16000;

    // Test different audio durations
    let durations_secs = [1.0, 3.0, 5.0, 10.0];

    // Create Hamming window
    let window: Vec<f32> = (0..n_fft)
        .map(|i| {
            let t = i as f32 / (n_fft - 1) as f32;
            0.54 - 0.46 * (2.0 * PI * t).cos()
        })
        .collect();

    // Pre-create FFT planner
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    println!(
        "| Duration | Frames | Manual DFT | FFT      | Speedup |"
    );
    println!(
        "|----------|--------|------------|----------|---------|"
    );

    for duration in durations_secs {
        let num_samples = (duration * sample_rate as f32) as usize;

        // Generate test audio (sine wave)
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let n_frames = (num_samples - n_fft) / hop_length + 1;

        // Warmup
        let _ = compute_stft_fft(&samples, &window, n_fft, hop_length, &fft);

        // Benchmark Manual DFT (limit iterations for long audio)
        let manual_iterations = if duration <= 3.0 { 5 } else { 2 };
        let start = Instant::now();
        for _ in 0..manual_iterations {
            let _ = compute_stft_manual_dft(&samples, &window, n_fft, hop_length);
        }
        let manual_time = start.elapsed().as_secs_f64() / manual_iterations as f64;

        // Benchmark FFT
        let fft_iterations = 20;
        let start = Instant::now();
        for _ in 0..fft_iterations {
            let _ = compute_stft_fft(&samples, &window, n_fft, hop_length, &fft);
        }
        let fft_time = start.elapsed().as_secs_f64() / fft_iterations as f64;

        let speedup = manual_time / fft_time;

        println!(
            "| {:>6.1}s  | {:>6} | {:>8.2}ms | {:>6.2}ms | {:>5.1}x  |",
            duration,
            n_frames,
            manual_time * 1000.0,
            fft_time * 1000.0,
            speedup
        );
    }

    println!("\n=== Complexity Analysis ===\n");
    println!("For n_fft = {}:", n_fft);
    println!("  Manual DFT: O(N²) = O({}) = ~{} operations/frame", n_fft * n_fft, n_fft * n_fft);
    let log_n = (n_fft as f32).log2().ceil() as usize;
    println!("  FFT:        O(N log N) = O({} × {}) = ~{} operations/frame", n_fft, log_n, n_fft * log_n);
    println!("  Theoretical speedup: ~{:.0}x", (n_fft * n_fft) as f32 / (n_fft * log_n) as f32);

    println!("\n=== Memory Usage ===\n");
    println!("FFT buffer: {} complex floats = {} bytes", n_fft, n_fft * 8);
    println!("Window: {} floats = {} bytes", n_fft, n_fft * 4);
}
