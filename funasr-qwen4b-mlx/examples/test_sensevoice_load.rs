//! Test SenseVoice encoder weight loading
//!
//! Run: cargo run --example test_sensevoice_load --release

use funasr_qwen4b_mlx::sensevoice_encoder::{SenseVoiceEncoder, SenseVoiceEncoderConfig};
use funasr_qwen4b_mlx::error::Result;
use mlx_rs::Array;

fn main() -> Result<()> {
    println!("=== SenseVoice Encoder Weight Loading Test ===\n");

    // Path to funasr-nano model weights
    let weights_path = std::env::var("SENSEVOICE_WEIGHTS").unwrap_or_else(|_| {
        dirs::home_dir().unwrap_or_default()
            .join(".OminiX/models/funasr-nano/model.safetensors")
            .to_string_lossy().to_string()
    });

    if !std::path::Path::new(&weights_path).exists() {
        println!("Weights not found: {}", weights_path);
        println!("Set SENSEVOICE_WEIGHTS environment variable or use default path.");
        return Ok(());
    }

    // Create encoder
    println!("Creating SenseVoice encoder...");
    let config = SenseVoiceEncoderConfig::default();
    println!("Config:");
    println!("  output_size: {}", config.output_size);
    println!("  attention_heads: {}", config.attention_heads);
    println!("  linear_units: {}", config.linear_units);
    println!("  num_blocks: {}", config.num_blocks);
    println!("  tp_blocks: {}", config.tp_blocks);
    println!("  kernel_size: {}", config.kernel_size);
    println!("  lfr_dim: {}", config.lfr_dim);

    let mut encoder = SenseVoiceEncoder::new(config)?;
    println!("\nEncoder created:");
    println!("  encoders0: {} layers", encoder.encoders0.len());
    println!("  encoders: {} layers", encoder.encoders.len());
    println!("  tp_encoders: {} layers", encoder.tp_encoders.len());

    // Load weights
    println!("\nLoading weights from {}...", weights_path);
    encoder.load_weights(&weights_path)?;
    println!("Weights loaded successfully!");

    // Test forward pass with dummy input
    println!("\n=== Testing Forward Pass ===");
    let batch_size = 1;
    let seq_len = 100;  // 100 frames after LFR
    let lfr_dim = 560;  // 80 mels * 7 LFR

    let dummy_input = Array::zeros::<f32>(&[batch_size, seq_len, lfr_dim])?;
    println!("Input shape: {:?}", dummy_input.shape());

    let start = std::time::Instant::now();
    let output = encoder.forward(&dummy_input)?;
    let elapsed = start.elapsed();

    println!("Output shape: {:?}", output.shape());
    println!("Expected: [{}, {}, 512]", batch_size, seq_len);
    println!("Forward pass time: {:.2?}", elapsed);

    // Verify output shape
    let shape = output.shape();
    assert_eq!(shape[0], batch_size);
    assert_eq!(shape[1], seq_len);
    assert_eq!(shape[2], 512);

    // Check output statistics
    let output_sum = mlx_rs::ops::sum(&output, None)?;
    let output_sum_val: f32 = output_sum.item();
    println!("Output sum: {:.4}", output_sum_val);

    // Non-zero output indicates weights are being applied
    if output_sum_val.abs() > 1e-6 {
        println!("\n✓ Output is non-zero - weights are being applied correctly!");
    } else {
        println!("\n⚠ Output is zero - weights may not be loaded correctly");
    }

    println!("\n=== SenseVoice Encoder Test Complete ===");
    Ok(())
}
