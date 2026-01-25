//! Validate correctness of FFT-based STFT and batch CIF
//!
//! Compares outputs of optimized implementations against reference implementations
//! to ensure numerical correctness.
//!
//! Usage:
//!   cargo run --release --example validate_correctness

use std::f32::consts::PI;
use std::sync::Arc;

use rustfft::{num_complex::Complex, FftPlanner};

// ============================================================================
// STFT Validation
// ============================================================================

/// Reference: Manual DFT implementation (known correct, but slow)
fn compute_stft_reference(
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

/// Optimized: FFT-based STFT
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
        for i in 0..n_fft {
            buffer[i] = Complex::new(samples[start + i] * window[i], 0.0);
        }
        fft.process(&mut buffer);
        for k in 0..n_freqs {
            let c = buffer[k];
            power_spec[frame * n_freqs + k] = c.re * c.re + c.im * c.im;
        }
    }

    power_spec
}

// ============================================================================
// CIF Validation
// ============================================================================

/// Reference: Single-item CIF fire (known correct)
fn cif_fire_reference(
    hidden: &[f32],
    alphas: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    threshold: f32,
    tail_threshold: f32,
) -> (Vec<f32>, usize) {
    let mut integrate = 0.0f32;
    let mut frame = vec![0.0f32; hidden_dim];
    let mut list_frames: Vec<Vec<f32>> = Vec::new();

    for t in 0..seq_len {
        let alpha = alphas[t];
        let distribution_completion = 1.0 - integrate;
        integrate += alpha;

        let fire_place = integrate >= threshold;
        if fire_place {
            integrate -= 1.0;
        }

        let cur = if fire_place { distribution_completion } else { alpha };
        let remainds = alpha - cur;

        for d in 0..hidden_dim {
            frame[d] += cur * hidden[t * hidden_dim + d];
        }

        if fire_place {
            list_frames.push(frame.clone());
            for d in 0..hidden_dim {
                frame[d] = remainds * hidden[t * hidden_dim + d];
            }
        }
    }

    if integrate > tail_threshold {
        list_frames.push(frame);
    }

    let num_tokens = list_frames.len();
    let flat: Vec<f32> = list_frames.into_iter().flatten().collect();
    (flat, num_tokens)
}

/// Optimized: Batch CIF fire
fn cif_fire_batch(
    hidden: &[f32],
    alphas: &[f32],
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
    threshold: f32,
    tail_threshold: f32,
) -> (Vec<f32>, Vec<usize>) {
    let mut all_batch_frames: Vec<Vec<Vec<f32>>> = Vec::with_capacity(batch_size);
    let mut token_counts: Vec<usize> = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let mut integrate = 0.0f32;
        let mut frame = vec![0.0f32; hidden_dim];
        let mut list_frames: Vec<Vec<f32>> = Vec::new();

        for t in 0..seq_len {
            let alpha_idx = b * seq_len + t;
            let hidden_offset = b * seq_len * hidden_dim + t * hidden_dim;

            let alpha = alphas[alpha_idx];
            let distribution_completion = 1.0 - integrate;
            integrate += alpha;

            let fire_place = integrate >= threshold;
            if fire_place {
                integrate -= 1.0;
            }

            let cur = if fire_place { distribution_completion } else { alpha };
            let remainds = alpha - cur;

            for d in 0..hidden_dim {
                frame[d] += cur * hidden[hidden_offset + d];
            }

            if fire_place {
                list_frames.push(frame.clone());
                for d in 0..hidden_dim {
                    frame[d] = remainds * hidden[hidden_offset + d];
                }
            }
        }

        if integrate > tail_threshold {
            list_frames.push(frame);
        }

        token_counts.push(list_frames.len());
        all_batch_frames.push(list_frames);
    }

    let max_tokens = token_counts.iter().copied().max().unwrap_or(0);
    let mut flat_embeds = vec![0.0f32; batch_size * max_tokens * hidden_dim];

    for (b, batch_frames) in all_batch_frames.into_iter().enumerate() {
        for (t, frame) in batch_frames.into_iter().enumerate() {
            let offset = b * max_tokens * hidden_dim + t * hidden_dim;
            for (d, &val) in frame.iter().enumerate() {
                flat_embeds[offset + d] = val;
            }
        }
    }

    (flat_embeds, token_counts)
}

// ============================================================================
// Validation Utilities
// ============================================================================

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn mean_abs_error(a: &[f32], b: &[f32]) -> f32 {
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    sum / a.len() as f32
}

fn relative_error(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    let denom: f32 = a.iter().map(|x| x.powi(2)).sum();
    if denom > 0.0 {
        (num / denom).sqrt()
    } else {
        0.0
    }
}

fn main() {
    println!("=== Correctness Validation ===\n");

    // ========================================================================
    // STFT Validation
    // ========================================================================
    println!("1. STFT Validation (Manual DFT vs FFT)\n");

    let n_fft = 400;
    let hop_length = 160;
    let sample_rate = 16000;

    // Create Hamming window
    let window: Vec<f32> = (0..n_fft)
        .map(|i| {
            let t = i as f32 / (n_fft - 1) as f32;
            0.54 - 0.46 * (2.0 * PI * t).cos()
        })
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    // Test with different signal types
    let test_cases = [
        ("Sine wave (440Hz)", generate_sine(sample_rate, 1.0, 440.0)),
        ("White noise", generate_noise(sample_rate, 1.0)),
        ("Mixed frequencies", generate_mixed(sample_rate, 1.0)),
        ("Speech-like", generate_speech_like(sample_rate, 1.0)),
    ];

    println!("| Test Case          | Max Error  | Mean Error | Rel Error  | Status |");
    println!("|--------------------|------------|------------|------------|--------|");

    let mut all_stft_passed = true;
    for (name, samples) in &test_cases {
        let reference = compute_stft_reference(samples, &window, n_fft, hop_length);
        let optimized = compute_stft_fft(samples, &window, n_fft, hop_length, &fft);

        let max_err = max_abs_error(&reference, &optimized);
        let mean_err = mean_abs_error(&reference, &optimized);
        let rel_err = relative_error(&reference, &optimized);

        // Allow small numerical differences (float32 precision)
        let passed = rel_err < 1e-5;
        all_stft_passed &= passed;

        println!(
            "| {:18} | {:10.2e} | {:10.2e} | {:10.2e} | {:6} |",
            name,
            max_err,
            mean_err,
            rel_err,
            if passed { "PASS" } else { "FAIL" }
        );
    }

    println!();
    if all_stft_passed {
        println!("STFT Validation: PASSED - FFT produces identical results to manual DFT\n");
    } else {
        println!("STFT Validation: FAILED - Results differ beyond tolerance\n");
    }

    // ========================================================================
    // CIF Validation
    // ========================================================================
    println!("2. CIF Batch Validation (Single vs Batch Processing)\n");

    let seq_len = 100;
    let hidden_dim = 512;
    let threshold = 1.0f32;
    let tail_threshold = 0.45f32;
    let batch_sizes = [1, 2, 4, 8];

    println!("| Batch Size | Tokens Match | Max Error  | Mean Error | Status |");
    println!("|------------|--------------|------------|------------|--------|");

    let mut all_cif_passed = true;
    for &batch_size in &batch_sizes {
        // Generate test data
        let hidden: Vec<f32> = (0..batch_size * seq_len * hidden_dim)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        let alphas: Vec<f32> = (0..batch_size * seq_len)
            .map(|i| 0.5 + 0.3 * (i as f32 * 0.1).sin())
            .collect();

        // Run batch version
        let (batch_output, batch_counts) = cif_fire_batch(
            &hidden,
            &alphas,
            batch_size,
            seq_len,
            hidden_dim,
            threshold,
            tail_threshold,
        );
        let max_tokens = batch_counts.iter().copied().max().unwrap_or(0);

        // Run reference version for each item and compare
        let mut tokens_match = true;
        let mut max_err = 0.0f32;
        let mut total_err = 0.0f32;
        let mut total_count = 0usize;

        for b in 0..batch_size {
            let h_start = b * seq_len * hidden_dim;
            let h_end = h_start + seq_len * hidden_dim;
            let a_start = b * seq_len;
            let a_end = a_start + seq_len;

            let (ref_output, ref_count) = cif_fire_reference(
                &hidden[h_start..h_end],
                &alphas[a_start..a_end],
                seq_len,
                hidden_dim,
                threshold,
                tail_threshold,
            );

            if ref_count != batch_counts[b] {
                tokens_match = false;
            }

            // Compare outputs (only valid tokens, not padding)
            for t in 0..ref_count {
                for d in 0..hidden_dim {
                    let ref_val = ref_output[t * hidden_dim + d];
                    let batch_val = batch_output[b * max_tokens * hidden_dim + t * hidden_dim + d];
                    let err = (ref_val - batch_val).abs();
                    max_err = max_err.max(err);
                    total_err += err;
                    total_count += 1;
                }
            }
        }

        let mean_err = if total_count > 0 {
            total_err / total_count as f32
        } else {
            0.0
        };

        let passed = tokens_match && max_err < 1e-6;
        all_cif_passed &= passed;

        println!(
            "| {:>10} | {:>12} | {:10.2e} | {:10.2e} | {:6} |",
            batch_size,
            if tokens_match { "Yes" } else { "No" },
            max_err,
            mean_err,
            if passed { "PASS" } else { "FAIL" }
        );
    }

    println!();
    if all_cif_passed {
        println!("CIF Validation: PASSED - Batch processing produces identical results\n");
    } else {
        println!("CIF Validation: FAILED - Results differ\n");
    }

    // ========================================================================
    // Summary
    // ========================================================================
    println!("=== Summary ===\n");
    println!(
        "STFT (FFT):      {}",
        if all_stft_passed {
            "PASSED - Numerically identical to reference"
        } else {
            "FAILED"
        }
    );
    println!(
        "CIF (Batch):     {}",
        if all_cif_passed {
            "PASSED - Numerically identical to reference"
        } else {
            "FAILED"
        }
    );

    if all_stft_passed && all_cif_passed {
        println!("\nAll validations PASSED! Optimizations are correct.");
        std::process::exit(0);
    } else {
        println!("\nSome validations FAILED!");
        std::process::exit(1);
    }
}

// ============================================================================
// Test Signal Generators
// ============================================================================

fn generate_sine(sample_rate: usize, duration: f32, freq: f32) -> Vec<f32> {
    let num_samples = (duration * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
        .collect()
}

fn generate_noise(sample_rate: usize, duration: f32) -> Vec<f32> {
    let num_samples = (duration * sample_rate as f32) as usize;
    // Simple LCG pseudo-random
    let mut seed = 12345u64;
    (0..num_samples)
        .map(|_| {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) as f32 / 32768.0) - 1.0
        })
        .collect()
}

fn generate_mixed(sample_rate: usize, duration: f32) -> Vec<f32> {
    let num_samples = (duration * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * PI * 200.0 * t).sin()
                + 0.3 * (2.0 * PI * 500.0 * t).sin()
                + 0.2 * (2.0 * PI * 1000.0 * t).sin()
        })
        .collect()
}

fn generate_speech_like(sample_rate: usize, duration: f32) -> Vec<f32> {
    let num_samples = (duration * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Simulate speech: fundamental + harmonics with amplitude modulation
            let fundamental = 150.0;
            let envelope = (2.0 * PI * 3.0 * t).sin().abs(); // ~3Hz modulation
            envelope
                * (0.4 * (2.0 * PI * fundamental * t).sin()
                    + 0.3 * (2.0 * PI * 2.0 * fundamental * t).sin()
                    + 0.2 * (2.0 * PI * 3.0 * fundamental * t).sin()
                    + 0.1 * (2.0 * PI * 4.0 * fundamental * t).sin())
        })
        .collect()
}
