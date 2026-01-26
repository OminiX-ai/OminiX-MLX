//! Trace full T2S generation in zero-shot mode
//!
//! Run with: cargo run --release --example trace_t2s_full

use mlx_rs::{
    argmax_axis,
    module::Module,
    ops::indexing::IndexOp,
    transforms::eval,
    Array,
};
use gpt_sovits_mlx::{
    cache::ConcatKeyValueCache,
    models::t2s::{load_t2s_weights, T2SConfig, T2SInput, T2SModel},
};

fn load_npy(path: &str) -> Array {
    let bytes = std::fs::read(path).expect("Failed to read npy file");
    assert_eq!(&bytes[0..6], b"\x93NUMPY");
    let version = bytes[6];

    let (header_len, header_start) = if version == 1 {
        let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (len, 10)
    } else {
        let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (len, 12)
    };

    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .expect("Invalid header");
    let shape_start = header.find("'shape':").expect("No shape") + 8;
    let shape_str = &header[shape_start..];
    let paren_start = shape_str.find('(').unwrap();
    let paren_end = shape_str.find(')').unwrap();
    let dims_str = &shape_str[paren_start + 1..paren_end];
    let shape: Vec<i32> = dims_str
        .split(',')
        .filter_map(|s| {
            let s = s.trim();
            if s.is_empty() { None } else { s.parse().ok() }
        })
        .collect();

    let data_start = header_start + header_len;
    let data = &bytes[data_start..];
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let floats: Vec<f32> = data[..n_elements * 4]
        .chunks(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Array::from_slice(&floats, &shape)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Full T2S Generation (Zero-Shot) ===");

    // Load T2S model weights
    let weights_path = "/tmp/gpt-sovits-mlx/doubao_mixed_gpt_new.safetensors";
    println!("Loading T2S weights from {}", weights_path);

    let weights = Array::load_safetensors(weights_path)?;
    let config = T2SConfig::default();
    let mut model = T2SModel::new(config.clone())?;
    load_t2s_weights(&mut model, &weights)?;

    // Prepare inputs (same as Python)
    let phone_ids = Array::from_slice(&[318i32, 166, 122, 97, 250, 164, 316, 257, 227, 177], &[1, 10]);
    println!("Phone IDs: {:?}", phone_ids.as_slice::<i32>());

    // Load BERT features from Python
    let bert_features = load_npy("/tmp/tts_comparison/python_bert_expanded.npy");
    let bert_shape = bert_features.shape().to_vec();
    let bert_features = if bert_shape.len() == 2 {
        bert_features.reshape(&[1, bert_shape[0], bert_shape[1]])?
    } else {
        bert_features
    };
    println!("BERT features shape: {:?}", bert_features.shape());

    // Zero-shot: start with token 0
    let mut semantic_ids = Array::from_slice(&[0i32], &[1, 1]);
    println!("Initial prompts (zero-shot): [[0]]");

    // Initialize cache
    let num_layers = config.num_layers as usize;
    let mut caches: Vec<Option<ConcatKeyValueCache>> = (0..num_layers).map(|_| None).collect();

    // Generation parameters
    let temperature = 1.0f32;
    let top_k = 20;
    let max_tokens = 50;
    let eos_token = config.eos_token;

    // Collect generated tokens
    let mut all_tokens: Vec<i32> = vec![0]; // Start with the initial token

    // First forward pass (prefill)
    let input = T2SInput {
        phoneme_ids: &phone_ids,
        semantic_ids: &semantic_ids,
        bert_features: &bert_features,
        cache: &mut caches,
    };
    let logits = model.forward(input)?;
    eval([&logits])?;
    println!("Logits shape after prefill: {:?}", logits.shape());

    let seq_len = logits.shape()[1] as i32;
    let vocab_size = logits.shape()[2] as i32;
    // Get last position logits: [batch, vocab_size]
    let last_logits = logits.index((.., seq_len - 1, ..));
    eval([&last_logits])?;
    println!("Last logits shape: {:?}", last_logits.shape());

    // Squeeze batch dimension for 1D array
    let last_logits_1d = last_logits.squeeze()?;
    eval([&last_logits_1d])?;
    println!("Last logits 1D shape: {:?}", last_logits_1d.shape());

    // Sample first token using simple argmax for now
    let first_token_idx = argmax_axis!(&last_logits_1d, -1)?;
    eval([&first_token_idx])?;
    println!("First token (argmax) shape: {:?}", first_token_idx.shape());
    let token_id: i32 = first_token_idx.item();
    all_tokens.push(token_id);
    semantic_ids = Array::from_slice(&[token_id], &[1, 1]);

    println!("Token 1: {}", token_id);

    // Continue generation
    for step in 2..=max_tokens {
        let input = T2SInput {
            phoneme_ids: &phone_ids,
            semantic_ids: &semantic_ids,
            bert_features: &bert_features,
            cache: &mut caches,
        };
        let logits = model.forward(input)?;
        eval([&logits])?;

        let last_logits = logits.index((.., -1, ..)).squeeze()?;
        eval([&last_logits])?;

        let next_token_idx = argmax_axis!(&last_logits, -1)?;
        eval([&next_token_idx])?;
        let token_id: i32 = next_token_idx.item();

        all_tokens.push(token_id);
        semantic_ids = Array::from_slice(&[token_id], &[1, 1]);

        if token_id == eos_token {
            println!("Token {}: {} (EOS)", step, token_id);
            break;
        }

        if step <= 10 || step % 10 == 0 {
            println!("Token {}: {}", step, token_id);
        }
    }

    println!("\n=== Results ===");
    println!("Generated {} tokens", all_tokens.len());
    println!("First 20 tokens: {:?}", &all_tokens[..20.min(all_tokens.len())]);

    // Compare with Python
    let py_tokens = vec![0, 462, 583, 637, 1016, 190, 372, 219, 387, 387, 928, 197, 132, 396, 644, 301, 888, 1014, 660, 511];
    println!("\nPython first 20: {:?}", py_tokens);

    let matches = all_tokens.iter()
        .zip(py_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();
    println!("Matching prefix length: {}", matches);

    Ok(())
}
