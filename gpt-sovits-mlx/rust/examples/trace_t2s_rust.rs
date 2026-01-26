//! Trace T2S model step by step and compare with Python
//!
//! Run with: cargo run --release --example trace_t2s_rust

use mlx_rs::{
    array,
    module::{Module, Param},
    ops::{concatenate_axis, indexing::IndexOp},
    transforms::eval,
    Array,
};
use gpt_sovits_mlx::{
    cache::ConcatKeyValueCache,
    models::t2s::{load_t2s_weights, T2SConfig, T2SInput, T2SModel},
};

fn load_npy(path: &str) -> Array {
    // Simple NPY loader for f32 arrays
    let bytes = std::fs::read(path).expect("Failed to read npy file");

    // Parse NPY header
    // Magic: 0x93NUMPY
    // Version: 1 byte major, 1 byte minor
    // Header len: 2 bytes (v1) or 4 bytes (v2/3)
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

    // Parse shape from header like "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 10, 512)}"
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

    // Convert to f32
    let n_elements: usize = shape.iter().map(|&d| d as usize).product();
    let floats: Vec<f32> = data[..n_elements * 4]
        .chunks(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Array::from_slice(&floats, &shape)
}

fn compare_arrays(name: &str, rust: &Array, python_path: &str) {
    let python = load_npy(python_path);
    eval([&python]).unwrap();

    let rust_data: Vec<f32> = rust.as_slice().to_vec();
    let python_data: Vec<f32> = python.as_slice().to_vec();

    let max_diff = rust_data.iter()
        .zip(python_data.iter())
        .map(|(r, p)| (r - p).abs())
        .fold(0.0f32, f32::max);

    println!("=== {} ===", name);
    println!("Rust shape: {:?}", rust.shape());
    println!("Python shape: {:?}", python.shape());
    println!("Rust[0,:5]: {:?}", &rust_data[..5.min(rust_data.len())]);
    println!("Python[0,:5]: {:?}", &python_data[..5.min(python_data.len())]);
    println!("Max diff: {}", max_diff);

    if max_diff > 1e-3 {
        println!("*** MISMATCH! ***");
    } else {
        println!("MATCH ✓");
    }
    println!();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load T2S model weights
    let weights_path = "/tmp/gpt-sovits-mlx/doubao_mixed_gpt_new.safetensors";
    println!("Loading T2S weights from {}", weights_path);

    let weights = Array::load_safetensors(weights_path)?;

    // Create model
    let config = T2SConfig::default();
    let mut model = T2SModel::new(config)?;
    load_t2s_weights(&mut model, &weights)?;

    // Prepare inputs (same as Python)
    let phone_ids = Array::from_slice(&[318i32, 166, 122, 97, 250, 164, 316, 257, 227, 177], &[1, 10]);
    println!("Phone IDs: {:?}", phone_ids.as_slice::<i32>());

    // Load BERT features from Python
    let bert_features = load_npy("/tmp/tts_comparison/python_bert_expanded.npy");
    // Add batch dimension [10, 1024] -> [1, 10, 1024]
    let bert_shape = bert_features.shape().to_vec();
    let bert_features = if bert_shape.len() == 2 {
        bert_features.reshape(&[1, bert_shape[0], bert_shape[1]])?
    } else {
        bert_features
    };
    println!("BERT features shape: {:?}", bert_features.shape());

    // Step 1: Phoneme embedding
    let phone_emb = model.phoneme_embedding.forward(&phone_ids)?;
    eval([&phone_emb])?;
    compare_arrays("Step 1: Phoneme embedding", &phone_emb, "/tmp/tts_comparison/py_step1_phone_emb.npy");

    // Step 2: BERT projection
    let bert_proj = model.bert_proj.forward(&bert_features)?;
    eval([&bert_proj])?;
    compare_arrays("Step 2: BERT projection", &bert_proj, "/tmp/tts_comparison/py_step2_bert_proj.npy");

    // Step 3: Combined (phoneme + BERT)
    let combined = phone_emb.add(&bert_proj)?;
    eval([&combined])?;
    compare_arrays("Step 3: Phoneme + BERT", &combined, "/tmp/tts_comparison/py_step3_combined.npy");

    // Step 4: Text position encoding
    let text_with_pos = model.text_position.apply(&combined, 0)?;
    eval([&text_with_pos])?;
    compare_arrays("Step 4: After text position", &text_with_pos, "/tmp/tts_comparison/py_step4_text_pos.npy");

    // Step 5: Semantic embedding (start token = 0)
    let semantic_ids = Array::from_slice(&[0i32], &[1, 1]);
    let sem_emb = model.semantic_embedding.forward(&semantic_ids)?;
    eval([&sem_emb])?;
    compare_arrays("Step 5: Semantic embedding", &sem_emb, "/tmp/tts_comparison/py_step5_sem_emb.npy");

    // Step 6: Audio position encoding
    let audio_with_pos = model.audio_position.apply(&sem_emb, 0)?;
    eval([&audio_with_pos])?;
    compare_arrays("Step 6: After audio position", &audio_with_pos, "/tmp/tts_comparison/py_step6_audio_pos.npy");

    // Step 7: Concatenate text + audio
    let xy = concatenate_axis(&[&text_with_pos, &audio_with_pos], 1)?;
    eval([&xy])?;
    compare_arrays("Step 7: Concatenated xy", &xy, "/tmp/tts_comparison/py_step7_xy.npy");

    // Step 8: Attention mask
    let mask = model.create_t2s_mask(10, 1)?;
    eval([&mask])?;
    compare_arrays("Step 8: Attention mask", &mask, "/tmp/tts_comparison/py_step8_mask.npy");

    // Step 9: Layer 0 attention QKV
    // Compute QKV manually for debugging
    let qkv = model.layers[0].self_attn.in_proj.forward(&xy)?;
    eval([&qkv])?;

    // Print QKV values instead of slicing to avoid contiguity issues
    println!("=== Step 9: QKV ===");
    println!("QKV shape: {:?}", qkv.shape());
    let qkv_data: Vec<f32> = qkv.as_slice().to_vec();
    println!("Rust QKV[0,0,:5]: {:?}", &qkv_data[..5.min(qkv_data.len())]);

    // Load Python Q and compare manually
    let py_q = load_npy("/tmp/tts_comparison/py_step9_q.npy");
    eval([&py_q])?;
    let py_q_data: Vec<f32> = py_q.as_slice().to_vec();
    println!("Python Q[0,0,:5]: {:?}", &py_q_data[..5.min(py_q_data.len())]);

    // Q is first 512 elements of each row
    let max_diff_q: f32 = qkv_data.chunks(1536) // Each row is 1536 elements (3*512)
        .flat_map(|row| &row[0..512]) // Q section
        .zip(py_q_data.iter())
        .map(|(r, p)| (r - p).abs())
        .fold(0.0, f32::max);
    println!("Max diff Q: {}", max_diff_q);
    if max_diff_q > 1e-3 {
        println!("*** Q MISMATCH! ***");
    } else {
        println!("Q MATCH ✓");
    }
    println!();

    // Full forward pass
    let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
    let input = T2SInput {
        phoneme_ids: &phone_ids,
        semantic_ids: &semantic_ids,
        bert_features: &bert_features,
        cache: &mut cache,
    };

    let logits = model.forward(input)?;
    eval([&logits])?;

    // Get last position logits
    let last_logits = logits.index((.., -1, ..));
    eval([&last_logits])?;

    println!("=== Step 11: Final logits ===");
    println!("Logits shape: {:?}", logits.shape());
    let logits_data: Vec<f32> = last_logits.as_slice().to_vec();
    println!("Rust[0,:10]: {:?}", &logits_data[..10.min(logits_data.len())]);

    // Load Python logits
    let py_logits = load_npy("/tmp/tts_comparison/py_step11_logits.npy");
    let py_logits_data: Vec<f32> = py_logits.as_slice().to_vec();
    println!("Python[0,:10]: {:?}", &py_logits_data[..10.min(py_logits_data.len())]);

    let max_diff = logits_data.iter()
        .zip(py_logits_data.iter())
        .map(|(r, p)| (r - p).abs())
        .fold(0.0f32, f32::max);
    println!("Max diff: {}", max_diff);

    // Argmax
    let argmax_rust = logits_data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let argmax_python = py_logits_data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    println!("\nRust argmax: {}", argmax_rust);
    println!("Python argmax: {}", argmax_python);

    if argmax_rust == argmax_python {
        println!("ARGMAX MATCH ✓");
    } else {
        println!("*** ARGMAX MISMATCH! ***");
    }

    Ok(())
}
