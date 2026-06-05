//! End-to-end chat example: encode a text prompt, generate a reply, decode it.
//!
//! Usage:
//!   cargo run -p gemma4-mlx --example chat_gemma4 --release -- \
//!     <MODEL_DIR> "<your prompt>"
//!
//! Example:
//!   cargo run -p gemma4-mlx --example chat_gemma4 --release -- \
//!     /Users/alan0x/models/gemma-4-12B-it-4bit "Explain what MLX is in one sentence."

use std::path::Path;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .expect("usage: chat_gemma4 <MODEL_DIR> \"<prompt>\"");
    let prompt = args
        .next()
        .expect("usage: chat_gemma4 <MODEL_DIR> \"<prompt>\"");

    let dir = Path::new(&model_dir);

    // Load tokenizer
    eprintln!("Loading tokenizer ...");
    let tok = gemma4_mlx::load_tokenizer(dir)?;

    // Encode the chat prompt
    let prompt_ids = gemma4_mlx::encode_chat(&tok, &prompt)?;
    eprintln!("Prompt: {:?}", prompt);
    eprintln!("Prompt token count: {}", prompt_ids.len());

    // Load model
    eprintln!("Loading model from {model_dir} ...");
    let mut model = gemma4_mlx::load_model(dir)?;

    // Generate
    let eos = gemma4_mlx::eos_ids();
    eprintln!("Generating (max 128 tokens) ...");
    let t0 = Instant::now();
    let gen_ids = gemma4_mlx::generate_greedy(&mut model, &prompt_ids, 128, &eos)?;
    let elapsed = t0.elapsed().as_secs_f64();

    let n_gen = gen_ids.len();
    let toks_per_sec = n_gen as f64 / elapsed.max(1e-9);

    // Decode generated ids to text
    let gen_u32: Vec<u32> = gen_ids.iter().map(|&id| id as u32).collect();
    let reply = tok
        .decode(&gen_u32, true)
        .map_err(|e| anyhow::anyhow!("decode error: {e}"))?;

    println!("{reply}");
    eprintln!("---");
    eprintln!("Generated {n_gen} tokens in {elapsed:.2}s ({toks_per_sec:.1} tok/s)");

    Ok(())
}
