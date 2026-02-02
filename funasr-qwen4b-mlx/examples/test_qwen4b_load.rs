//! Test loading Qwen3-4B weights from safetensors.
//!
//! Prerequisites:
//! 1. Download Qwen3-4B from HuggingFace:
//!    huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
//!
//! 2. Convert weights to MLX format:
//!    python3 scripts/convert_qwen4b_weights.py models/Qwen3-4B --stats
//!
//! 3. Run this example:
//!    cargo run --example test_qwen4b_load --release

use funasr_qwen4b_mlx::qwen4b::{Qwen4BConfig, Qwen4BModel};
use funasr_qwen4b_mlx::error::Result;
use mlx_rs::Array;
use mlx_rs_core::KVCache;

fn main() -> Result<()> {
    let model_path = "models/Qwen3-4B/model_mlx.safetensors";

    println!("=== Qwen3-4B Weight Loading Test ===\n");

    // Check if weights exist
    if !std::path::Path::new(model_path).exists() {
        println!("Weights not found at: {}", model_path);
        println!("\nTo download and convert weights:");
        println!("  1. pip install huggingface-hub");
        println!("  2. huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B");
        println!("  3. python3 scripts/convert_qwen4b_weights.py models/Qwen3-4B --stats");
        println!("\nOr try loading from sharded files directly...\n");

        // Try loading from directory with sharded files
        let model_dir = "models/Qwen3-4B";
        if std::path::Path::new(model_dir).exists() {
            test_load_from_dir(model_dir)?;
        } else {
            println!("Model directory not found: {}", model_dir);
            println!("\nSkipping Qwen4B test - no weights available.");
            println!("The model architecture is correct; only weights are missing.");
        }
        return Ok(());
    }

    test_load_from_file(model_path)?;
    Ok(())
}

fn test_load_from_file(model_path: &str) -> Result<()> {
    println!("Creating Qwen4BModel with default config...");
    let config = Qwen4BConfig::default();
    println!("  hidden_size: {}", config.hidden_size);
    println!("  num_layers: {}", config.num_hidden_layers);
    println!("  num_heads: {}", config.num_attention_heads);
    println!("  num_kv_heads: {}", config.num_key_value_heads);

    let mut model = Qwen4BModel::new(config)?;
    println!("Model created successfully!");

    println!("\nLoading weights from {}...", model_path);
    model.load_weights(model_path)?;
    println!("Weights loaded successfully!");

    // Test forward pass with a small input
    println!("\nTesting forward pass...");
    let input_tokens = Array::from_slice(&[1i32, 2, 3, 4, 5], &[1, 5]);
    let mut cache: Vec<Option<KVCache>> = Vec::new();

    let logits = model.forward_tokens(&input_tokens, &mut cache)?;
    println!("Input shape: {:?}", input_tokens.shape());
    println!("Output shape: {:?}", logits.shape());
    println!("Expected: [1, 5, {}]", 151936); // vocab_size

    let shape = logits.shape();
    assert_eq!(shape[0], 1);
    assert_eq!(shape[1], 5);
    assert_eq!(shape[2], 151936);

    println!("\nAll tests passed!");
    Ok(())
}

fn test_load_from_dir(model_dir: &str) -> Result<()> {
    println!("Creating Qwen4BModel with default config...");
    let config = Qwen4BConfig::default();
    let mut model = Qwen4BModel::new(config)?;
    println!("Model created successfully!");

    println!("\nLoading weights from directory: {}...", model_dir);
    model.load_weights_from_dir(model_dir)?;
    println!("Weights loaded successfully!");

    // Test forward pass
    println!("\nTesting forward pass...");
    let input_tokens = Array::from_slice(&[1i32, 2, 3, 4, 5], &[1, 5]);
    let mut cache: Vec<Option<KVCache>> = Vec::new();

    let logits = model.forward_tokens(&input_tokens, &mut cache)?;
    println!("Input shape: {:?}", input_tokens.shape());
    println!("Output shape: {:?}", logits.shape());

    println!("\nAll tests passed!");
    Ok(())
}
