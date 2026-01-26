//! Z-Image-Turbo Quantized Model Benchmark
//!
//! This example demonstrates loading and running the 4-bit quantized Z-Image model.
//! Memory usage: ~3GB instead of ~12GB with dequantized weights.
//!
//! Usage:
//!   cargo run --example generate_zimage_quantized --release

use flux_klein_mlx::load_safetensors;
use zimage_mlx::{
    load_quantized_zimage_transformer, ZImageConfig,
    create_coordinate_grid,
};

use mlx_rs::{
    array, ops, Array, Dtype,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Z-Image-Turbo Quantized Model Benchmark ===\n");

    // Model path (MLX quantized version)
    let transformer_path = "/tmp/zimage_mlx_test/Z-Image-Turbo-MLX/transformer";

    // ========================================================================
    // Load Quantized Transformer (keeps 4-bit weights!)
    // ========================================================================
    println!("Loading quantized transformer (4-bit)...");
    let trans_start = Instant::now();

    let config = ZImageConfig::default();
    let trans_weights = load_safetensors(&format!("{}/model.safetensors", transformer_path))?;
    println!("  Loaded {} weight tensors", trans_weights.len());

    let mut transformer = load_quantized_zimage_transformer(trans_weights, config.clone())?;
    println!("  Quantized transformer loaded in {:.2}s", trans_start.elapsed().as_secs_f32());

    // ========================================================================
    // Setup Benchmark
    // ========================================================================
    let width = 512;
    let height = 512;
    let h_tok = height / 16;
    let w_tok = width / 16;
    let img_seq_len = h_tok * w_tok;
    let cap_len = 512i32;
    let steps = 9;

    println!("\nBenchmark settings:");
    println!("  Image: {}x{}", width, height);
    println!("  Token grid: {}x{} = {} tokens", h_tok, w_tok, img_seq_len);
    println!("  Caption length: {}", cap_len);
    println!("  Steps: {}", steps);

    // Create position grids
    let x_pos = create_coordinate_grid(
        (1, h_tok, w_tok),
        ((cap_len + 1) as i32, 0, 0),
    )?.reshape(&[1, img_seq_len, 3])?;
    let x_pos = x_pos.as_dtype(Dtype::Bfloat16)?;

    let cap_pos = create_coordinate_grid(
        (cap_len, 1, 1),
        (1, 0, 0),
    )?.reshape(&[1, cap_len, 3])?;
    let cap_pos = cap_pos.as_dtype(Dtype::Bfloat16)?;

    // Pre-compute RoPE
    let (cos_cached, sin_cached) = transformer.compute_rope(&x_pos, &cap_pos)?;
    let cos_cached = cos_cached.as_dtype(Dtype::Bfloat16)?;
    let sin_cached = sin_cached.as_dtype(Dtype::Bfloat16)?;

    // Initialize random inputs
    mlx_rs::random::seed(42)?;
    let mut latents = mlx_rs::random::normal::<f32>(&[1, 16, height / 8, width / 8], None, None, None)?;
    latents = latents.as_dtype(Dtype::Bfloat16)?;

    let cap_feats = mlx_rs::random::normal::<f32>(&[1, cap_len, 2560], None, None, None)?;
    let cap_feats = cap_feats.as_dtype(Dtype::Bfloat16)?;

    // ========================================================================
    // Timestep Schedule
    // ========================================================================
    fn calculate_shift(image_seq_len: i32) -> f32 {
        let base_seq_len = 256.0f32;
        let max_seq_len = 4096.0f32;
        let base_shift = 0.5f32;
        let max_shift = 1.15f32;
        let m = (max_shift - base_shift) / (max_seq_len - base_seq_len);
        let b = base_shift - m * base_seq_len;
        (image_seq_len as f32) * m + b
    }

    let mu = calculate_shift(img_seq_len);
    let timesteps: Vec<f32> = (0..=steps)
        .map(|i| {
            let t = 1.0 - (i as f32) / (steps as f32);
            if t > 0.0 {
                mu.exp() / (mu.exp() + (1.0 / t - 1.0))
            } else {
                0.0
            }
        })
        .collect();

    // ========================================================================
    // Warmup
    // ========================================================================
    println!("\nWarmup...");
    let t_warmup = array!([0.0f32]).as_dtype(Dtype::Bfloat16)?;
    let x_warmup = latents.reshape(&[16, 1, 1, h_tok, 2, w_tok, 2])?;
    let x_warmup = x_warmup.transpose_axes(&[1, 2, 3, 5, 4, 6, 0])?;
    let x_warmup = x_warmup.reshape(&[1, img_seq_len, 64])?;

    let _ = transformer.forward_with_rope(
        &x_warmup,
        &t_warmup,
        &cap_feats,
        &x_pos,
        &cap_pos,
        &cos_cached,
        &sin_cached,
        None,
        None,
    )?;
    latents.eval()?;

    // ========================================================================
    // Denoising Loop
    // ========================================================================
    println!("\nDenoising ({} steps)...", steps);
    let denoise_start = Instant::now();
    let mut step_times = Vec::new();

    for i in 0..steps {
        let step_start = Instant::now();

        let t_curr = timesteps[i as usize];
        let t_prev = timesteps[(i + 1) as usize];
        let t_input = array!([1.0 - t_curr]).as_dtype(Dtype::Bfloat16)?;

        // Reshape latents: [B,C,H,W] -> [B, h_tok*w_tok, C*4]
        let x = latents.reshape(&[16, 1, 1, h_tok, 2, w_tok, 2])?;
        let x = x.transpose_axes(&[1, 2, 3, 5, 4, 6, 0])?;
        let x = x.reshape(&[1, img_seq_len, 64])?;

        // Forward through transformer
        let out = transformer.forward_with_rope(
            &x,
            &t_input,
            &cap_feats,
            &x_pos,
            &cap_pos,
            &cos_cached,
            &sin_cached,
            None,
            None,
        )?;

        // Reshape output back: [B, h_tok*w_tok, C*4] -> [B,C,H,W]
        let noise_pred = out.reshape(&[1, 1, h_tok, w_tok, 2, 2, 16])?;
        let noise_pred = noise_pred.transpose_axes(&[6, 0, 1, 2, 4, 3, 5])?;
        let noise_pred = noise_pred.reshape(&[1, 16, height / 8, width / 8])?;
        let noise_pred = ops::negative(&noise_pred)?;

        // Euler step
        let dt = t_prev - t_curr;
        let dt_array = array!(dt);
        latents = ops::add(&latents, &ops::multiply(&dt_array, &noise_pred)?)?;

        latents.eval()?;
        let step_time = step_start.elapsed().as_secs_f32();
        step_times.push(step_time);
        println!("  Step {}/{}: {:.3}s", i + 1, steps, step_time);
    }

    let total_time = denoise_start.elapsed().as_secs_f32();
    let avg_time: f32 = step_times.iter().sum::<f32>() / step_times.len() as f32;

    println!("\n=== Results ===");
    println!("Total denoising time: {:.2}s", total_time);
    println!("Average step time: {:.3}s", avg_time);
    println!("Final latents sum: {:.4}", latents.sum(None)?.item::<f32>());

    // Compare with non-quantized
    println!("\n=== Performance Comparison ===");
    println!("Quantized (4-bit):   {:.3}s/step", avg_time);
    println!("Dequantized (f32):   ~1.87s/step (from previous benchmark)");
    println!("Memory: ~3GB quantized vs ~12GB dequantized");

    Ok(())
}
