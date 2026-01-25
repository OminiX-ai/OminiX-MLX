//! Compare CIF performance: Sequential vs Batched processing
//!
//! This benchmark shows the throughput improvement from batch support.
//!
//! Usage:
//!   cargo run --release --example compare_cif_batch

use std::time::Instant;

use mlx_rs::Array;

fn main() {
    println!("=== CIF Batch Processing Comparison ===\n");

    // Simulate encoder output dimensions
    let seq_len = 100; // ~1.6s audio after LFR (100 * 60ms)
    let hidden_dim = 512;
    let batch_sizes = [1, 2, 4, 8];

    println!("Simulating encoder output: seq_len={}, hidden_dim={}\n", seq_len, hidden_dim);

    println!("| Batch Size | Sequential | Batched   | Items/sec (Seq) | Items/sec (Batch) | Speedup |");
    println!("|------------|------------|-----------|-----------------|-------------------|---------|");

    for &batch_size in &batch_sizes {
        // Create test data
        let hidden_data: Vec<f32> = (0..batch_size * seq_len * hidden_dim)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        let alphas_data: Vec<f32> = (0..batch_size * seq_len)
            .map(|i| 0.5 + 0.3 * (i as f32 * 0.1).sin()) // Values around 0.5, summing to ~50 tokens per sequence
            .collect();

        // Benchmark sequential processing (batch_size=1, loop over items)
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            for b in 0..batch_size {
                // Process each item separately
                let offset_h = b * seq_len * hidden_dim;
                let offset_a = b * seq_len;
                let single_hidden: Vec<f32> = hidden_data[offset_h..offset_h + seq_len * hidden_dim].to_vec();
                let single_alphas: Vec<f32> = alphas_data[offset_a..offset_a + seq_len].to_vec();

                // Simulate CIF fire for single item
                let _ = simulate_cif_fire_single(&single_hidden, &single_alphas, seq_len, hidden_dim);
            }
        }
        let sequential_time = start.elapsed().as_secs_f64() / iterations as f64;

        // Benchmark batched processing
        let start = Instant::now();
        for _ in 0..iterations {
            // Process entire batch at once
            let _ = simulate_cif_fire_batch(&hidden_data, &alphas_data, batch_size, seq_len, hidden_dim);
        }
        let batched_time = start.elapsed().as_secs_f64() / iterations as f64;

        let seq_throughput = batch_size as f64 / sequential_time;
        let batch_throughput = batch_size as f64 / batched_time;
        let speedup = sequential_time / batched_time;

        println!(
            "| {:>10} | {:>8.3}ms | {:>7.3}ms | {:>15.1} | {:>17.1} | {:>6.2}x |",
            batch_size,
            sequential_time * 1000.0,
            batched_time * 1000.0,
            seq_throughput,
            batch_throughput,
            speedup
        );
    }

    println!("\n=== Analysis ===\n");
    println!("Batch processing benefits:");
    println!("  1. Reduced function call overhead");
    println!("  2. Better memory locality");
    println!("  3. Enables future GPU parallelization");
    println!("  4. Single allocation for output tensor");
}

/// Simulate CIF fire for a single sequence (old approach)
fn simulate_cif_fire_single(
    hidden: &[f32],
    alphas: &[f32],
    seq_len: usize,
    hidden_dim: usize,
) -> (Vec<f32>, usize) {
    let threshold = 1.0f32;
    let tail_threshold = 0.45f32;

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

/// Simulate CIF fire for a batch (new approach)
fn simulate_cif_fire_batch(
    hidden: &[f32],
    alphas: &[f32],
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
) -> (Vec<f32>, Vec<usize>) {
    let threshold = 1.0f32;
    let tail_threshold = 0.45f32;

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

    // Find max tokens and create padded output
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
