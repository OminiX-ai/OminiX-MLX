//! Moxin-7B VLM inference example.
//!
//! Usage:
//!   cargo run --release -p moxin-vlm-mlx --example generate -- \
//!     --model ./models/Moxin-7B-VLM-hf \
//!     --image ./test.jpg \
//!     --prompt "What is in this image?"

use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use image::imageops::FilterType;
use mlx_rs::Array;

use moxin_vlm_mlx::{load_model, load_tokenizer, normalize_dino, normalize_siglip, Generate, KVCache};

#[derive(Parser)]
#[command(name = "moxin-vlm-generate")]
struct Args {
    /// Path to Moxin-7B-VLM model directory
    #[arg(long)]
    model: String,

    /// Path to input image
    #[arg(long)]
    image: String,

    /// Text prompt (will be formatted as "In: <prompt>\nOut:")
    #[arg(long, default_value = "Describe this image.")]
    prompt: String,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value = "0.0")]
    temp: f32,

    /// Maximum tokens to generate
    #[arg(long, default_value = "256")]
    max_tokens: usize,

    /// Quantize to N bits (e.g. 8 or 4). 0 = no quantization (default).
    #[arg(long, default_value = "0")]
    quantize: i32,
}

fn load_and_preprocess_image(path: &str) -> Result<(Array, Array)> {
    let img = image::open(path)?;
    let img = img.resize_exact(224, 224, FilterType::CatmullRom);
    let rgb = img.to_rgb8();

    // Convert to [1, 224, 224, 3] float32 in [0, 1] (NHWC for MLX)
    let pixels: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| p.0.iter().map(|&v| v as f32 / 255.0))
        .collect();
    let tensor = Array::from_slice(&pixels, &[1, 224, 224, 3]);

    let dino_img = normalize_dino(&tensor)?;
    let siglip_img = normalize_siglip(&tensor)?;

    Ok((dino_img, siglip_img))
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load model and tokenizer
    let vlm = load_model(&args.model)?;
    let mut vlm = if args.quantize > 0 {
        eprintln!("Quantizing to {} bits...", args.quantize);
        let t = Instant::now();
        let q = vlm.quantize(64, args.quantize)?;
        eprintln!("Quantized in {:.1}s", t.elapsed().as_secs_f64());
        q
    } else {
        vlm
    };
    let tokenizer = load_tokenizer(&args.model)?;

    // Load and preprocess image
    eprintln!("Loading image: {}", args.image);
    let (dino_img, siglip_img) = load_and_preprocess_image(&args.image)?;

    // Format prompt (Prismatic "Pure" format)
    let prompt_text = format!("In: {}\nOut:", args.prompt);
    eprintln!("Prompt: {}", prompt_text);

    // Tokenize (with BOS)
    let encoding = tokenizer
        .encode(prompt_text.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;
    let input_ids = Array::from_iter(
        encoding.get_ids().iter().map(|&id| id as i32),
        &[1, encoding.get_ids().len() as i32],
    );

    eprintln!(
        "Input: {} text tokens + 256 visual tokens",
        encoding.get_ids().len()
    );

    // Generate
    let mut cache: Vec<KVCache> = Vec::new();
    let generator = Generate::new(
        &mut vlm,
        &mut cache,
        args.temp,
        dino_img,
        siglip_img,
        input_ids,
    );

    let eos_token_id = 2u32; // </s>
    let mut generated = Vec::new();
    let mut prev_text_len = 0;
    let mut prefill_time = None;
    let t0 = Instant::now();

    for token_result in generator.take(args.max_tokens) {
        let token = token_result?;

        // Record prefill time after first token
        if prefill_time.is_none() {
            prefill_time = Some(t0.elapsed());
        }

        let token_id = token.item::<u32>();

        if token_id == eos_token_id {
            break;
        }

        generated.push(token_id);

        // Decode all tokens to get correct spacing, print only new chars
        let text = tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;
        if text.len() > prev_text_len {
            print!("{}", &text[prev_text_len..]);
        }
        prev_text_len = text.len();
    }

    let total_time = t0.elapsed();
    println!();

    // Performance stats
    let n = generated.len();
    let prefill_ms = prefill_time.map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0);
    let decode_ms = total_time.as_secs_f64() * 1000.0 - prefill_ms;
    let decode_tps = if n > 1 { (n - 1) as f64 / (decode_ms / 1000.0) } else { 0.0 };

    eprintln!("\n--- Performance ---");
    eprintln!("Prefill:  {:.0} ms (vision + {} text tokens + 256 visual tokens)", prefill_ms, encoding.get_ids().len());
    eprintln!("Decode:   {} tokens in {:.0} ms ({:.1} tokens/s)", n, decode_ms, decode_tps);
    eprintln!("Total:    {:.1} s", total_time.as_secs_f64());

    Ok(())
}
