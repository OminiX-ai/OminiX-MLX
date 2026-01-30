//! Compare T2S intermediates between Rust and Python

use mlx_rs::{Array, module::Module, transforms::eval};
use gpt_sovits_mlx::{
    inference::preprocess_text,
    models::t2s::{load_t2s_model, T2SConfig, T2SInput},
    cache::ConcatKeyValueCache,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Compare T2S Intermediates ===\n");

    // First, check phoneme preprocessing
    let text = "你好";
    let (phoneme_ids_array, phonemes, word2ph, _normalized) = preprocess_text(text);
    eval([&phoneme_ids_array])?;

    // Convert to Vec for comparison
    let phoneme_ids: Vec<i32> = phoneme_ids_array.flatten(None, None)?.as_slice().to_vec();

    println!("Text: {}", text);
    println!("Phonemes: {:?}", phonemes);
    println!("Phoneme IDs: {:?}", phoneme_ids);
    println!("word2ph: {:?}", word2ph);

    // Python produced: [227, 167, 158, 119] for "你好" (without period)
    println!("\nPython phoneme IDs: [227, 167, 158, 119] (without period)");
    let expected = vec![227, 167, 158, 119];
    let matches = phoneme_ids == expected;
    println!("Match: {}", matches);
    if !matches {
        println!("WARNING: Phoneme IDs don't match! This could cause T2S divergence.");
        println!("  Rust:   {:?}", phoneme_ids);
        println!("  Python: {:?}", expected);
    }

    // Load T2S model
    let t2s_path = "~/.dora/models/primespeech/gpt-sovits-mlx/doubao_mixed_gpt_new.safetensors";
    let mut t2s = load_t2s_model(t2s_path)?;
    let config = T2SConfig::default();
    println!("\nT2S config: hidden_size={}, num_layers={}", config.hidden_size, config.num_layers);

    // Create test input matching Python:
    // x = [227, 167, 158, 119, 3]  (phoneme IDs with period)
    let phoneme_ids_with_period: Vec<i32> = vec![227, 167, 158, 119, 3];
    let phoneme_ids_array = Array::from_slice(&phoneme_ids_with_period, &[1, 5]);

    // Start with semantic token 0 (start token)
    let semantic_ids = Array::zeros::<i32>(&[1, 1])?;

    // Zero BERT features
    let bert_features = Array::zeros::<f32>(&[1, 5, 1024])?;

    // Initialize cache
    let mut caches: Vec<Option<ConcatKeyValueCache>> = (0..config.num_layers as usize)
        .map(|_| None)
        .collect();

    println!("\n=== Running T2S forward pass ===");
    println!("phoneme_ids_array shape: {:?}", phoneme_ids_array.shape());
    println!("semantic_ids shape: {:?}", semantic_ids.shape());
    println!("bert_features shape: {:?}", bert_features.shape());

    // Run forward pass
    let input = T2SInput {
        phoneme_ids: &phoneme_ids_array,
        semantic_ids: &semantic_ids,
        bert_features: &bert_features,
        cache: &mut caches,
    };
    let logits = t2s.forward(input)?;
    eval([&logits])?;
    println!("Logits shape: {:?}", logits.shape());

    // Print top-10 tokens from logits
    let logits_vec: Vec<f32> = logits.flatten(None, None)?.as_slice().to_vec();
    let seq_len = logits.shape()[1] as usize;
    let vocab_size = logits.shape()[2] as usize;

    // Get last position logits
    let last_start = (seq_len - 1) * vocab_size;
    let last_logits: Vec<f32> = logits_vec[last_start..last_start + vocab_size].to_vec();

    let mut indexed: Vec<(usize, f32)> = last_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop-10 tokens at last position (after prefill):");
    for (i, (tok, logit)) in indexed.iter().take(10).enumerate() {
        println!("  {}: token {} = {:.4}", i + 1, tok, logit);
    }

    // Now do another forward pass with the top token to see next prediction
    let top_token = indexed[0].0 as i32;
    println!("\nUsing top token {} for next step...", top_token);

    let semantic_ids_next = Array::from_slice(&[top_token], &[1, 1]);
    let input2 = T2SInput {
        phoneme_ids: &phoneme_ids_array,
        semantic_ids: &semantic_ids_next,
        bert_features: &bert_features,
        cache: &mut caches,
    };
    let logits2 = t2s.forward(input2)?;
    eval([&logits2])?;

    let logits2_vec: Vec<f32> = logits2.flatten(None, None)?.as_slice().to_vec();
    let seq_len2 = logits2.shape()[1] as usize;
    let last_start2 = (seq_len2 - 1) * vocab_size;
    let last_logits2: Vec<f32> = logits2_vec[last_start2..last_start2 + vocab_size].to_vec();

    let mut indexed2: Vec<(usize, f32)> = last_logits2.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop-10 tokens at step 1 (after first generated token):");
    for (i, (tok, logit)) in indexed2.iter().take(10).enumerate() {
        println!("  {}: token {} = {:.4}", i + 1, tok, logit);
    }

    Ok(())
}
