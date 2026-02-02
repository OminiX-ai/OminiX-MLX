//! Benchmark full audio preprocessing pipeline
//!
//! This tests the real-world impact of LFR optimization in the full pipeline.

use funasr_nano_mlx::audio::{apply_lfr, AudioConfig, MelFrontend};
use mlx_rs::Array;
use std::time::Instant;

/// Reference CPU-based LFR (old implementation)
fn apply_lfr_cpu(mel: &Array, lfr_m: usize, lfr_n: usize) -> mlx_rs::error::Result<Array> {
    let shape = mel.shape();
    let batch = shape[0] as usize;
    let n_mels = shape[1] as usize;
    let n_frames = shape[2] as usize;

    let n_lfr_frames = (n_frames + lfr_n - 1) / lfr_n;
    let lfr_dim = n_mels * lfr_m;

    mlx_rs::transforms::eval([mel])?;
    let mel_transposed = mel.transpose_axes(&[0, 2, 1])?;
    let mel_contiguous = mlx_rs::ops::contiguous(&mel_transposed)?;
    mlx_rs::transforms::eval([&mel_contiguous])?;
    let mel_data: Vec<f32> = mel_contiguous.as_slice::<f32>().to_vec();

    let mut lfr_data = vec![0.0f32; batch * n_lfr_frames * lfr_dim];

    for b in 0..batch {
        for out_frame in 0..n_lfr_frames {
            let center_frame = out_frame * lfr_n;
            for m in 0..lfr_m {
                let src_frame = if m < lfr_m / 2 {
                    let offset = (lfr_m / 2) - m;
                    if offset > center_frame { 0 } else { center_frame - offset }
                } else {
                    let offset = m - (lfr_m / 2);
                    (center_frame + offset).min(n_frames - 1)
                };

                let src_idx = b * n_frames * n_mels + src_frame * n_mels;
                let dst_idx = b * n_lfr_frames * lfr_dim + out_frame * lfr_dim + m * n_mels;

                for i in 0..n_mels {
                    lfr_data[dst_idx + i] = mel_data[src_idx + i];
                }
            }
        }
    }

    Ok(Array::from_slice(
        &lfr_data,
        &[batch as i32, n_lfr_frames as i32, lfr_dim as i32],
    ))
}

fn generate_audio(duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let n_samples = (duration_secs * sample_rate as f32) as usize;
    (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Mix of frequencies to simulate speech
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
        })
        .collect()
}

fn main() {
    println!("=== Full Pipeline Benchmark ===\n");

    let config = AudioConfig::default();
    let frontend = MelFrontend::new(config);

    let durations = [1.0, 3.0, 5.0, 10.0];
    let iterations = 20;

    for &duration in &durations {
        let audio = generate_audio(duration, 16000);

        // Warm up
        let mel = frontend.compute_mel_spectrogram(&audio).unwrap();
        let _ = apply_lfr(&mel, 7, 6).unwrap();
        let _ = apply_lfr_cpu(&mel, 7, 6).unwrap();

        // Benchmark with pure MLX LFR
        let start = Instant::now();
        for _ in 0..iterations {
            let mel = frontend.compute_mel_spectrogram(&audio).unwrap();
            let lfr = apply_lfr(&mel, 7, 6).unwrap();
            mlx_rs::transforms::eval([&lfr]).unwrap();
        }
        let mlx_time = start.elapsed();

        // Benchmark with CPU LFR
        let start = Instant::now();
        for _ in 0..iterations {
            let mel = frontend.compute_mel_spectrogram(&audio).unwrap();
            let lfr = apply_lfr_cpu(&mel, 7, 6).unwrap();
            mlx_rs::transforms::eval([&lfr]).unwrap();
        }
        let cpu_time = start.elapsed();

        let mlx_avg = mlx_time.as_secs_f64() * 1000.0 / iterations as f64;
        let cpu_avg = cpu_time.as_secs_f64() * 1000.0 / iterations as f64;
        let diff_pct = (cpu_avg - mlx_avg) / cpu_avg * 100.0;

        println!(
            "  {:.0}s audio: MLX={:.2}ms, CPU={:.2}ms, diff={:+.1}%",
            duration, mlx_avg, cpu_avg, diff_pct
        );
    }

    println!("\n=== Memory Transfer Analysis ===\n");

    // Show what happens with consecutive operations
    let audio = generate_audio(5.0, 16000);
    let mel = frontend.compute_mel_spectrogram(&audio).unwrap();

    // With CPU LFR: mel (GPU) → CPU → lfr (GPU) → next op uses lfr on GPU
    // With MLX LFR: mel (GPU) → lfr (GPU) → next op uses lfr on GPU (no CPU detour)

    println!("  CPU LFR: GPU→CPU→GPU transfer for each LFR call");
    println!("  MLX LFR: Data stays on GPU throughout pipeline");
    println!("\n  The benefit is more visible when:");
    println!("  - Processing many audio files in sequence");
    println!("  - The LFR output feeds directly into GPU operations");
    println!("  - Memory bandwidth is constrained");
}
