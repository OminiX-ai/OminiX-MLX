//! Test forward_train pass with mock data
//!
//! This example verifies that the T2SModel forward_train method works correctly.

use gpt_sovits_mlx::models::t2s::{T2SModel, T2SConfig, load_t2s_model};
use mlx_rs::{Array, transforms::eval};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing T2S forward_train...\n");

    // Load pretrained model
    let model_path = "~/.OminiX/models/gpt-sovits-mlx/doubao_mixed_gpt_new.safetensors";
    println!("Loading model from: {}", model_path);

    let mut model = load_t2s_model(model_path)?;
    println!("Model loaded successfully\n");

    // Create mock training batch
    let batch_size = 2;
    let phoneme_len = 32;
    let semantic_len = 64;
    let bert_dim = 1024;

    // Random phoneme IDs [batch, phoneme_len]
    let phoneme_ids = Array::from_slice(
        &vec![1i32; batch_size * phoneme_len],
        &[batch_size as i32, phoneme_len as i32],
    );

    // Random BERT features [batch, bert_dim, phoneme_len]
    let bert_features = Array::zeros::<f32>(&[batch_size as i32, bert_dim, phoneme_len as i32])?;

    // Random semantic IDs (targets) [batch, semantic_len]
    let semantic_ids = Array::from_slice(
        &vec![100i32; batch_size * semantic_len],
        &[batch_size as i32, semantic_len as i32],
    );

    // Sequence lengths
    let phoneme_lens = Array::from_slice(
        &vec![phoneme_len as i32; batch_size],
        &[batch_size as i32],
    );
    let semantic_lens = Array::from_slice(
        &vec![semantic_len as i32; batch_size],
        &[batch_size as i32],
    );

    println!("Input shapes:");
    println!("  phoneme_ids: {:?}", phoneme_ids.shape());
    println!("  bert_features: {:?}", bert_features.shape());
    println!("  semantic_ids: {:?}", semantic_ids.shape());
    println!();

    // Run forward_train
    println!("Running forward_train...");
    let logits = model.forward_train(
        &phoneme_ids,
        &phoneme_lens,
        &bert_features,
        &semantic_ids,
        &semantic_lens,
    )?;

    eval([&logits])?;

    println!("Output logits shape: {:?}", logits.shape());
    println!("Expected shape: [{}, {}, {}]", batch_size, semantic_len, 1025);

    // Verify shape
    let expected_shape = vec![batch_size as i32, semantic_len as i32, 1025];
    assert_eq!(logits.shape(), &expected_shape, "Shape mismatch!");

    // Compute cross-entropy loss manually
    println!("\nComputing loss...");
    let log_probs = mlx_rs::nn::log_softmax(&logits, Some(-1))?;
    let targets_expanded = semantic_ids.reshape(&[batch_size as i32, semantic_len as i32, 1])?;
    let selected_log_probs = log_probs.take_along_axis(&targets_expanded, -1)?;
    let nll = selected_log_probs.negative()?;
    let loss = mlx_rs::ops::mean(&nll, false)?;
    eval([&loss])?;

    let loss_val: f32 = loss.item();
    println!("Loss: {:.4}", loss_val);

    // With random targets, loss should be around log(vocab_size) = log(1025) ≈ 6.93
    println!("Expected loss (random): ~{:.4}", (1025.0f32).ln());

    println!("\n✓ forward_train test passed!");
    Ok(())
}
