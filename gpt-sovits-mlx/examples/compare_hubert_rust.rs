//! Compare HuBERT outputs between Rust and Python

use mlx_rs::{Array, transforms::eval, module::Module};
use std::fs::File;
use std::io::Read;

fn load_npy_f32(path: &str) -> Vec<f32> {
    let mut file = File::open(path).expect("Failed to open npy file");
    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("Failed to read npy file");

    // Skip NPY header (find the newline after header dict)
    let mut header_end = 10;
    while header_end < data.len() && data[header_end] != b'\n' {
        header_end += 1;
    }
    header_end += 1;

    // Read as f32
    data[header_end..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn load_npy_i32(path: &str) -> Vec<i32> {
    let mut file = File::open(path).expect("Failed to open npy file");
    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("Failed to read npy file");

    // Skip NPY header
    let mut header_end = 10;
    while header_end < data.len() && data[header_end] != b'\n' {
        header_end += 1;
    }
    header_end += 1;

    // Read as i64 (numpy default int type)
    data[header_end..]
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as i32)
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use gpt_sovits_mlx::audio::load_audio_for_hubert;
    use gpt_sovits_mlx::models::hubert::load_hubert_model;
    use gpt_sovits_mlx::models::vits::load_vits_model;

    println!("Loading models...");
    let mut hubert = load_hubert_model("~/.dora/models/primespeech/gpt-sovits-mlx/hubert.safetensors")?;
    let vits = load_vits_model("~/.dora/models/primespeech/gpt-sovits-mlx/doubao_mixed_sovits_new.safetensors")?;

    // Load reference audio
    let ref_path = "/Users/yuechen/.dora/models/primespeech/moyoyo/ref_audios/doubao_ref_mix_new.wav";
    let audio_16k = load_audio_for_hubert(ref_path)?;
    eval([&audio_16k])?;

    // Add 0.3s padding (like Python)
    let audio_data: Vec<f32> = audio_16k.as_slice().to_vec();
    let pad_samples = (0.3 * 16000.0) as usize;
    let mut audio_padded = audio_data;
    audio_padded.extend(vec![0.0f32; pad_samples]);
    let audio_16k = Array::from_slice(&audio_padded, &[1, audio_padded.len() as i32]);

    println!("Audio samples: {}", audio_padded.len());

    // Step 1: HuBERT
    println!("\n=== Step 1: HuBERT ===");
    let hubert_out = hubert.forward(&audio_16k)?;
    eval([&hubert_out])?;
    println!("HuBERT output shape: {:?}", hubert_out.shape());

    // Load Python's HuBERT output for comparison
    let python_hubert = load_npy_f32("/tmp/python_hubert_features_nlc.npy");
    println!("Python HuBERT features: {} values", python_hubert.len());

    // Compare first 10 values
    let rust_hubert: Vec<f32> = hubert_out.as_slice().to_vec();
    println!("Rust first 10: {:?}", &rust_hubert[..10]);
    println!("Python first 10: {:?}", &python_hubert[..10]);

    // Compute max difference
    let max_diff: f32 = rust_hubert.iter()
        .zip(python_hubert.iter())
        .map(|(r, p)| (r - p).abs())
        .fold(0.0, f32::max);
    println!("Max difference in HuBERT features: {:.6}", max_diff);

    // Step 2: ssl_proj
    println!("\n=== Step 2: ssl_proj ===");
    // Convert to NCL for ssl_proj
    let hubert_ncl = hubert_out.transpose_axes(&[0, 2, 1])?;
    eval([&hubert_ncl])?;
    println!("HuBERT NCL shape: {:?}", hubert_ncl.shape());

    // Apply ssl_proj (MLX Conv1d expects NLC format)
    // NCL -> NLC -> Conv1d -> NLC -> NCL
    let hubert_nlc = mlx_rs::ops::swap_axes(&hubert_ncl, 1, 2)?;
    let mut ssl_proj = vits.ssl_proj.clone();
    let ssl_nlc = ssl_proj.forward(&hubert_nlc)?;
    let ssl_ncl = mlx_rs::ops::swap_axes(&ssl_nlc, 1, 2)?;
    eval([&ssl_ncl])?;
    println!("ssl_proj output shape (NCL): {:?}", ssl_ncl.shape());

    // Load Python's ssl_proj output
    let python_ssl = load_npy_f32("/tmp/python_ssl_proj_output_ncl.npy");
    println!("Python ssl_proj output: {} values", python_ssl.len());

    // Compare first 10 values at position 0 (channel 0, time 0)
    let rust_ssl: Vec<f32> = ssl_ncl.as_slice().to_vec();
    println!("Rust ssl_proj first 10 (pos 0): {:?}", &rust_ssl[..10]);
    println!("Python ssl_proj first 10 (pos 0): {:?}", &python_ssl[..10]);

    let max_ssl_diff: f32 = rust_ssl.iter()
        .zip(python_ssl.iter())
        .map(|(r, p)| (r - p).abs())
        .fold(0.0, f32::max);
    println!("Max difference in ssl_proj output: {:.6}", max_ssl_diff);

    // Step 3: Quantizer encode
    println!("\n=== Step 3: Quantizer encode ===");
    let codes = vits.quantizer.encode(&ssl_ncl)?;
    eval([&codes])?;
    println!("Codes shape: {:?}", codes.shape());

    let rust_codes: Vec<i32> = codes.as_slice().to_vec();
    println!("Rust first 20 codes: {:?}", &rust_codes[..20.min(rust_codes.len())]);

    // Load Python's codes
    let python_codes = load_npy_i32("/tmp/python_quantizer_codes.npy");
    println!("Python first 20 codes: {:?}", &python_codes[..20.min(python_codes.len())]);

    // Count differences
    let diff_count = rust_codes.iter()
        .zip(python_codes.iter())
        .filter(|(r, p)| r != p)
        .count();
    println!("\nCode differences: {} out of {} ({:.1}%)",
        diff_count, rust_codes.len(), (diff_count as f32 / rust_codes.len() as f32) * 100.0);

    // Show first 10 differences
    let diffs: Vec<_> = rust_codes.iter()
        .zip(python_codes.iter())
        .enumerate()
        .filter(|(_, (r, p))| r != p)
        .take(10)
        .collect();
    println!("First differences (pos, rust, python):");
    for (pos, (r, p)) in diffs {
        println!("  {} : {} vs {}", pos, r, p);
    }

    Ok(())
}
