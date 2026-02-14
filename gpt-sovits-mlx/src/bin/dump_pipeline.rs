//! Dump all intermediate pipeline outputs for verification against Python
//!
//! Usage: cargo run --features jieba -p gpt-sovits-mlx --bin dump_pipeline -- "text" [output_dir]

use gpt_sovits_mlx::text::preprocessor::{chinese_g2p, normalize_chinese};
use gpt_sovits_mlx::text::symbols::{symbol_to_id, has_symbol};
use gpt_sovits_mlx::text::bert_features::BertFeatureExtractor;
use mlx_rs::transforms::eval;
use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;

/// Write a 2D f32 array as .npy (NumPy format)
fn save_npy_f32(path: &Path, data: &[f32], shape: &[usize]) {
    let mut f = fs::File::create(path).unwrap();
    // NumPy .npy format v1.0
    let magic = b"\x93NUMPY";
    f.write_all(magic).unwrap();
    f.write_all(&[1u8, 0]).unwrap(); // version 1.0

    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "))
    };
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}",
        shape_str
    );
    // Pad header to align to 64 bytes (including magic + version + header_len)
    let prefix_len = 10; // magic(6) + version(2) + header_len(2)
    let padding_needed = 64 - ((prefix_len + header.len() + 1) % 64); // +1 for \n
    let padded_header = format!("{}{}\n", header, " ".repeat(padding_needed));
    let header_len = padded_header.len() as u16;
    f.write_all(&header_len.to_le_bytes()).unwrap();
    f.write_all(padded_header.as_bytes()).unwrap();

    // Write data as little-endian f32
    for &val in data {
        f.write_all(&val.to_le_bytes()).unwrap();
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <text> [output_dir]", args[0]);
        std::process::exit(1);
    }

    let text = &args[1];
    let output_dir = if args.len() > 2 {
        args[2].clone()
    } else {
        "/tmp/rust_pipeline".to_string()
    };
    fs::create_dir_all(&output_dir).unwrap();

    eprintln!("=== Rust Pipeline Dump ===");
    eprintln!("Input: {}...", &text.chars().take(80).collect::<String>());
    eprintln!();

    // Stage 1: Normalization
    let normalized = normalize_chinese(text);
    eprintln!("[Stage 1] Normalized ({} chars):", normalized.chars().count());
    eprintln!("  {}...", &normalized.chars().take(120).collect::<String>());
    eprintln!();

    // Stage 2: Phones + word2ph
    let (phones, word2ph) = chinese_g2p(&normalized);
    eprintln!("[Stage 2] Phones ({}):", phones.len());
    eprintln!("  {:?}...", &phones[..phones.len().min(40)]);
    eprintln!("[Stage 2] Word2Ph ({}):", word2ph.len());
    eprintln!("  {:?}...", &word2ph[..word2ph.len().min(40)]);
    let w2p_sum: i32 = word2ph.iter().sum();
    eprintln!("  Sum of word2ph: {}", w2p_sum);
    eprintln!("  Phones == sum(word2ph): {}", phones.len() as i32 == w2p_sum);
    eprintln!();

    // Stage 3: Phone IDs
    let phone_ids: Vec<i32> = phones.iter().map(|p| symbol_to_id(p)).collect();
    let unknown: Vec<(usize, &String)> = phones.iter().enumerate()
        .filter(|(_, p)| !has_symbol(p))
        .collect();
    eprintln!("[Stage 3] Phone IDs ({}):", phone_ids.len());
    eprintln!("  {:?}...", &phone_ids[..phone_ids.len().min(40)]);
    if !unknown.is_empty() {
        eprintln!("  WARNING: {} unknown symbols:", unknown.len());
        for (i, s) in &unknown {
            eprintln!("    [{}] {:?}", i, s);
        }
    }
    eprintln!();

    // Stage 4: BERT features
    let bert_model_dir = "~/.OminiX/models/gpt-sovits-mlx";
    let tokenizer_path = format!("{}/chinese-roberta-tokenizer/tokenizer.json", bert_model_dir);
    eprintln!("[Stage 4] BERT Features:");
    let bert_features = if Path::new(&tokenizer_path).exists() {
        match BertFeatureExtractor::new(
            &tokenizer_path,
            format!("{}/bert.safetensors", bert_model_dir),
            -3,
        ) {
            Ok(mut extractor) => {
                match extractor.extract_features(&normalized, &word2ph) {
                    Ok(features) => {
                        eval([&features]).unwrap();
                        let shape = features.shape().to_vec();
                        eprintln!("  Shape: {:?}", shape);
                        // Get data as f32 slice
                        let data: Vec<f32> = features.as_slice::<f32>().to_vec();
                        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
                        eprintln!("  Range: [{:.4}, {:.4}]", min, max);
                        eprintln!("  Mean: {:.6}", mean);
                        Some((data, shape))
                    }
                    Err(e) => {
                        eprintln!("  ERROR extracting features: {}", e);
                        None
                    }
                }
            }
            Err(e) => {
                eprintln!("  ERROR loading BERT: {}", e);
                None
            }
        }
    } else {
        eprintln!("  SKIP: tokenizer not found at {}", tokenizer_path);
        None
    };
    eprintln!();

    // Output JSON to stdout (text stages only)
    let phones_json: Vec<String> = phones.iter().map(|p| format!("\"{}\"", p)).collect();
    println!(
        r#"{{"input": "{}", "normalized": "{}", "phones": [{}], "word2ph": {:?}, "phone_ids": {:?}}}"#,
        text.replace('"', "\\\""),
        normalized.replace('"', "\\\""),
        phones_json.join(", "),
        word2ph,
        phone_ids,
    );

    // Save files
    let dir = Path::new(&output_dir);
    fs::write(dir.join("phones.txt"), phones.join("\n") + "\n").unwrap();
    fs::write(
        dir.join("phone_ids.txt"),
        phone_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join("\n") + "\n",
    ).unwrap();
    fs::write(
        dir.join("word2ph.txt"),
        word2ph.iter().map(|w| w.to_string()).collect::<Vec<_>>().join("\n") + "\n",
    ).unwrap();

    // Save BERT features as .npy
    if let Some((data, shape)) = bert_features {
        // Rust shape is [1, total_phones, 1024], Python expects [1024, total_phones]
        // Save as [1024, total_phones] for comparison: transpose
        let total_phones = shape[1] as usize;
        let hidden_dim = shape[2] as usize;
        let mut transposed = vec![0.0f32; hidden_dim * total_phones];
        for i in 0..total_phones {
            for j in 0..hidden_dim {
                transposed[j * total_phones + i] = data[i * hidden_dim + j];
            }
        }
        save_npy_f32(
            &dir.join("bert_features.npy"),
            &transposed,
            &[hidden_dim, total_phones],
        );
        eprintln!("  Saved bert_features.npy ({}, {})", hidden_dim, total_phones);
    }

    eprintln!("Saved to {}/", output_dir);
}
