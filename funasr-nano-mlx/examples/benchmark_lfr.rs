//! Benchmark LFR implementations: pure MLX vs CPU-based
//!
//! This benchmark compares:
//! 1. New pure MLX LFR (GPU-based, no transfers)
//! 2. Reference CPU-based LFR (with GPU→CPU→GPU transfer)

use funasr_nano_mlx::audio::apply_lfr;
use mlx_rs::Array;
use std::time::Instant;

/// Reference CPU-based LFR implementation (the old version)
fn apply_lfr_cpu(mel: &Array, lfr_m: usize, lfr_n: usize) -> mlx_rs::error::Result<Array> {
    let shape = mel.shape();
    let batch = shape[0] as usize;
    let n_mels = shape[1] as usize;
    let n_frames = shape[2] as usize;

    let n_lfr_frames = (n_frames + lfr_n - 1) / lfr_n;
    let lfr_dim = n_mels * lfr_m;

    // GPU→CPU transfer
    mlx_rs::transforms::eval([mel])?;
    let mel_transposed = mel.transpose_axes(&[0, 2, 1])?;
    let mel_contiguous = mlx_rs::ops::contiguous(&mel_transposed)?;
    mlx_rs::transforms::eval([&mel_contiguous])?;
    let mel_data: Vec<f32> = mel_contiguous.as_slice::<f32>().to_vec();

    // CPU-side LFR processing
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

    // CPU→GPU transfer
    Ok(Array::from_slice(
        &lfr_data,
        &[batch as i32, n_lfr_frames as i32, lfr_dim as i32],
    ))
}

fn benchmark_lfr(n_frames: usize, iterations: usize) {
    let n_mels = 80;
    let lfr_m = 7;
    let lfr_n = 6;

    // Create test mel spectrogram [1, 80, n_frames]
    let mel_data: Vec<f32> = (0..n_mels * n_frames)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let mel = Array::from_slice(&mel_data, &[1, n_mels as i32, n_frames as i32]);
    mlx_rs::transforms::eval([&mel]).unwrap();

    // Warm up
    let _ = apply_lfr(&mel, lfr_m, lfr_n).unwrap();
    let _ = apply_lfr_cpu(&mel, lfr_m, lfr_n).unwrap();

    // Benchmark pure MLX LFR
    let start = Instant::now();
    for _ in 0..iterations {
        let result = apply_lfr(&mel, lfr_m, lfr_n).unwrap();
        mlx_rs::transforms::eval([&result]).unwrap();
    }
    let mlx_time = start.elapsed();

    // Benchmark CPU-based LFR
    let start = Instant::now();
    for _ in 0..iterations {
        let result = apply_lfr_cpu(&mel, lfr_m, lfr_n).unwrap();
        mlx_rs::transforms::eval([&result]).unwrap();
    }
    let cpu_time = start.elapsed();

    let mlx_avg = mlx_time.as_secs_f64() * 1000.0 / iterations as f64;
    let cpu_avg = cpu_time.as_secs_f64() * 1000.0 / iterations as f64;
    let speedup = cpu_avg / mlx_avg;

    println!(
        "  n_frames={:4}: MLX={:.3}ms, CPU={:.3}ms, speedup={:.2}x",
        n_frames, mlx_avg, cpu_avg, speedup
    );
}

fn verify_correctness(n_frames: usize) -> bool {
    let n_mels = 80;
    let lfr_m = 7;
    let lfr_n = 6;

    let mel_data: Vec<f32> = (0..n_mels * n_frames)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let mel = Array::from_slice(&mel_data, &[1, n_mels as i32, n_frames as i32]);

    let mlx_result = apply_lfr(&mel, lfr_m, lfr_n).unwrap();
    let cpu_result = apply_lfr_cpu(&mel, lfr_m, lfr_n).unwrap();

    mlx_rs::transforms::eval([&mlx_result, &cpu_result]).unwrap();

    let mlx_data: Vec<f32> = mlx_result.as_slice::<f32>().to_vec();
    let cpu_data: Vec<f32> = cpu_result.as_slice::<f32>().to_vec();

    if mlx_data.len() != cpu_data.len() {
        println!("Length mismatch: MLX={}, CPU={}", mlx_data.len(), cpu_data.len());
        return false;
    }

    let max_diff = mlx_data
        .iter()
        .zip(cpu_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    if max_diff > 1e-5 {
        println!("Max diff: {}", max_diff);
        return false;
    }

    true
}

fn main() {
    println!("=== LFR Implementation Verification ===\n");

    // Verify correctness for various frame counts
    let test_frames = [12, 50, 100, 200, 500, 1000];
    let mut all_correct = true;

    for &n_frames in &test_frames {
        let correct = verify_correctness(n_frames);
        println!(
            "  n_frames={:4}: {}",
            n_frames,
            if correct { "✓ PASS" } else { "✗ FAIL" }
        );
        all_correct &= correct;
    }

    if !all_correct {
        println!("\nVerification FAILED! Skipping benchmark.");
        return;
    }

    println!("\n=== LFR Performance Benchmark ===\n");

    // Benchmark different audio lengths
    // 100 frames ≈ 1s audio, 1000 frames ≈ 10s audio
    let iterations = 100;

    for &n_frames in &test_frames {
        benchmark_lfr(n_frames, iterations);
    }

    println!("\nNote: Pure MLX LFR avoids GPU→CPU→GPU transfers");
}
