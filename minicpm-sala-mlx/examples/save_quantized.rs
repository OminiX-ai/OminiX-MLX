//! Save 8-bit quantized MiniCPM-SALA weights to safetensors.
//!
//! Usage:
//!   cargo run --release -p minicpm-sala-mlx --example save_quantized -- \
//!     --model ./models/MiniCPM-SALA \
//!     --output ./models/MiniCPM-SALA-8bit \
//!     --bits 8

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use clap::Parser;
use mlx_rs::module::ModuleParameters;
use mlx_rs::Array;

use minicpm_sala_mlx::{load_model, Model};

#[derive(Parser)]
struct Args {
    /// Path to BF16 model directory
    #[arg(long)]
    model: String,
    /// Output directory for quantized model
    #[arg(long)]
    output: String,
    /// Quantization bits (default: 8)
    #[arg(long, default_value = "8")]
    bits: i32,
    /// Quantization group size (default: 64)
    #[arg(long, default_value = "64")]
    group_size: i32,
}

/// Collect all parameters from a model with flattened key names.
fn collect_all_params(model: &Model) -> HashMap<String, Array> {
    let mut out = HashMap::new();
    for (key, value) in model.parameters().flatten() {
        out.insert(key.to_string(), value.clone());
    }
    out
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load BF16 model
    eprintln!("Loading BF16 model from {}...", args.model);
    let model = load_model(&args.model)?;
    eprintln!("Model loaded.");

    // Quantize — we'll quantize all linear layers by re-saving with quantized weights
    eprintln!("Quantizing to {} bits (group_size={})...", args.bits, args.group_size);

    // Collect all parameters
    let all_params = collect_all_params(&model);
    eprintln!("Collected {} parameter tensors.", all_params.len());

    // Quantize linear weight tensors: for each key ending in ".weight" that has 2D shape
    // and a corresponding projection layer, quantize it
    let mut quantized_params: HashMap<String, Array> = HashMap::new();

    for (key, value) in &all_params {
        let shape = value.shape();

        // Quantize 2D weight tensors (linear layers) that are NOT embeddings or norms
        if key.ends_with(".weight")
            && shape.len() == 2
            && !key.contains("embed_tokens")
            && !key.contains("layernorm")
            && !key.contains("norm.weight")
            && !key.contains("_norm.")
        {
            // Quantize this weight
            let (quantized, scales, biases) = mlx_rs::ops::quantize(
                value,
                args.group_size,
                args.bits,
                None,
            )?;
            quantized_params.insert(key.clone(), quantized);
            let scales_key = key.replace(".weight", ".scales");
            let biases_key = key.replace(".weight", ".biases");
            quantized_params.insert(scales_key, scales);
            quantized_params.insert(biases_key, biases);
        } else {
            // Keep as-is (norms, embeddings, etc.)
            quantized_params.insert(key.clone(), value.clone());
        }
    }

    eprintln!("Quantized to {} tensors.", quantized_params.len());

    // Eval all arrays before saving
    let refs: Vec<&Array> = quantized_params.values().collect();
    mlx_rs::transforms::eval(refs)?;

    // Create output directory
    let out_dir = Path::new(&args.output);
    std::fs::create_dir_all(out_dir)?;

    // Save as safetensors
    let out_path = out_dir.join("model.safetensors");
    eprintln!("Saving to {:?}...", out_path);
    Array::save_safetensors(&quantized_params, None, &out_path)?;

    // Copy config and tokenizer files, injecting quantization config
    let src = Path::new(&args.model);
    let config_path = src.join("config.json");
    if config_path.exists() {
        let config_str = std::fs::read_to_string(&config_path)?;
        let mut config: serde_json::Value = serde_json::from_str(&config_str)?;
        // Inject quantization config
        config["quantization"] = serde_json::json!({
            "group_size": args.group_size,
            "bits": args.bits,
        });
        let config_out = serde_json::to_string_pretty(&config)?;
        std::fs::write(out_dir.join("config.json"), config_out)?;
        eprintln!("Wrote config.json with quantization settings.");
    }

    for fname in &[
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "generation_config.json",
    ] {
        let src_file = src.join(fname);
        if src_file.exists() {
            std::fs::copy(&src_file, out_dir.join(fname))?;
            eprintln!("Copied {}", fname);
        }
    }

    let size = std::fs::metadata(&out_path)?.len();
    eprintln!("Done! Quantized model size: {:.1} GB", size as f64 / 1e9);

    Ok(())
}
