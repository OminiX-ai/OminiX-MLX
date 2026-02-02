//! Test Qwen3-4B text generation.
//!
//! Prerequisites:
//! 1. Download Qwen3-4B from HuggingFace:
//!    huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
//!
//! 2. Convert weights to MLX format:
//!    python3 scripts/convert_qwen4b_weights.py models/Qwen3-4B --stats
//!
//! 3. Run this example:
//!    cargo run --example test_qwen4b_generate --release

use funasr_qwen4b_mlx::qwen4b::{Qwen4BConfig, Qwen4BModel};
use funasr_qwen4b_mlx::error::Result;
use mlx_rs::Array;
use mlx_rs::ops::indexing::{IndexOp, argmax_axis};
use mlx_rs_core::KVCache;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let model_dir = "models/Qwen3-4B";

    println!("=== Qwen3-4B Text Generation Test ===\n");

    // Check if model directory exists
    let model_path = std::path::Path::new(model_dir);
    if !model_path.exists() {
        println!("Model directory not found: {}", model_dir);
        println!("\nTo download the model:");
        println!("  pip install huggingface-hub");
        println!("  huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B");
        println!("  python3 scripts/convert_qwen4b_weights.py models/Qwen3-4B --stats");
        return Ok(());
    }

    // Load tokenizer
    let tokenizer_path = model_path.join("tokenizer.json");
    if !tokenizer_path.exists() {
        println!("Tokenizer not found at: {}", tokenizer_path.display());
        return Ok(());
    }

    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::Tokenizer(format!("Failed to load: {}", e)))?;
    println!("Tokenizer loaded: {} tokens", tokenizer.get_vocab_size(false));

    // Load model
    println!("\nCreating Qwen4BModel...");
    let config = Qwen4BConfig::default();
    println!("  hidden_size: {}", config.hidden_size);
    println!("  num_layers: {}", config.num_hidden_layers);
    println!("  num_heads: {}", config.num_attention_heads);
    println!("  num_kv_heads: {}", config.num_key_value_heads);
    println!("  vocab_size: {}", config.vocab_size);

    let mut model = Qwen4BModel::new(config)?;
    println!("Model created successfully!");

    // Load weights
    let mlx_path = model_path.join("model_mlx.safetensors");
    if mlx_path.exists() {
        println!("\nLoading weights from {}...", mlx_path.display());
        model.load_weights(&mlx_path)?;
    } else {
        println!("\nLoading weights from sharded files...");
        model.load_weights_from_dir(model_path)?;
    }
    println!("Weights loaded successfully!");

    // Test generation with a simple prompt
    println!("\n=== Testing Text Generation ===\n");

    // Use Qwen3's chat format
    let prompt = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n";
    println!("Prompt: {}", prompt);

    // Tokenize
    let encoded = tokenizer.encode(prompt, false)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::Tokenizer(format!("Encode failed: {}", e)))?;
    let input_ids: Vec<i32> = encoded.get_ids().iter().map(|&id| id as i32).collect();
    println!("Input tokens: {:?} (len={})", &input_ids[..input_ids.len().min(10)], input_ids.len());

    // Create input array
    let input_array = Array::from_slice(&input_ids, &[1, input_ids.len() as i32]);

    // Generate tokens
    let max_new_tokens = 50;
    let mut cache: Vec<Option<KVCache>> = Vec::new();
    let mut generated_tokens: Vec<i32> = Vec::new();

    // EOS tokens for Qwen3
    let eos_token_id = 151643;  // <|endoftext|>
    let im_end_id = 151645;     // <|im_end|>

    println!("\nGenerating {} tokens...", max_new_tokens);
    let start = std::time::Instant::now();

    // First forward pass with full prompt
    let mut logits = model.forward_tokens(&input_array, &mut cache)?;

    for i in 0..max_new_tokens {
        // Get logits for last position
        let last_logits = logits.index((.., -1, ..));

        // Greedy decoding: argmax
        let next_token_arr = argmax_axis(&last_logits, -1, false)?;
        let next_token: i32 = next_token_arr.item();

        // Check for EOS
        if next_token == eos_token_id || next_token == im_end_id {
            println!("\n[EOS at step {}]", i);
            break;
        }

        generated_tokens.push(next_token);

        // Print token as it's generated
        let decoded = tokenizer.decode(&[next_token as u32], false)
            .unwrap_or_else(|_| format!("<{}>", next_token));
        print!("{}", decoded);
        std::io::Write::flush(&mut std::io::stdout()).ok();

        // Next forward pass with single token
        let next_array = Array::from_slice(&[next_token], &[1, 1]);
        logits = model.forward_tokens(&next_array, &mut cache)?;
    }

    let elapsed = start.elapsed();
    println!("\n\nGeneration complete!");
    println!("Generated {} tokens in {:.2?}", generated_tokens.len(), elapsed);
    println!("Speed: {:.1} tokens/sec", generated_tokens.len() as f64 / elapsed.as_secs_f64());

    // Decode full output
    let generated_u32: Vec<u32> = generated_tokens.iter().map(|&t| t as u32).collect();
    let output = tokenizer.decode(&generated_u32, true)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::Tokenizer(format!("Decode failed: {}", e)))?;

    println!("\n=== Full Output ===");
    println!("{}", output);

    println!("\n=== P1.2 Test Complete ===");
    Ok(())
}
