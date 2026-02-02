//! Test loading adaptor weights from safetensors.
//!
//! Run with:
//!   cargo run --example test_adaptor_load

use funasr_qwen4b_mlx::adaptor::AudioAdaptorQwen4B;
use funasr_qwen4b_mlx::error::Result;
use mlx_rs::Array;

fn main() -> Result<()> {
    println!("Creating AudioAdaptorQwen4B...");
    let mut adaptor = AudioAdaptorQwen4B::new()?;

    // Check if weights file exists
    let weights_path = "adaptor_phase2_final.safetensors";
    if !std::path::Path::new(weights_path).exists() {
        println!("Weights file not found: {}", weights_path);
        println!("Please run: python3 scripts/convert_adaptor_weights.py adaptor_phase2_final.pt");
        return Ok(());
    }

    println!("Loading weights from {}...", weights_path);
    adaptor.load_weights(weights_path)?;
    println!("Weights loaded successfully!");

    // Test forward pass with dummy input
    println!("\nTesting forward pass...");
    let batch_size = 1;
    let seq_len = 10;
    let input_dim = 512;

    // Create random input [batch, seq, 512]
    let input = Array::zeros::<f32>(&[batch_size, seq_len, input_dim])?;
    println!("Input shape: {:?}", input.shape());

    let output = adaptor.forward(&input)?;
    println!("Output shape: {:?}", output.shape());
    println!("Expected: [{}, {}, 2560]", batch_size, seq_len);

    // Verify output shape
    let output_shape = output.shape();
    assert_eq!(output_shape[0], batch_size);
    assert_eq!(output_shape[1], seq_len);
    assert_eq!(output_shape[2], 2560);

    println!("\nAll tests passed!");
    Ok(())
}
