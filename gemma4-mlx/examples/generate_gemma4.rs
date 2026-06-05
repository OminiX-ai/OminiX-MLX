//! Token-level generation example (no tokenizer — raw ids in/out).
//!
//! Useful for parity testing: supply explicit prompt ids, get back generated ids.
//!
//! Usage:
//!   cargo run -p gemma4-mlx --example generate_gemma4 --release -- \
//!     <MODEL_DIR> "<space-separated prompt ids>" <max_new>
//!
//! Example:
//!   cargo run -p gemma4-mlx --example generate_gemma4 --release -- \
//!     /path/to/gemma-4-12B-it-4bit "2 105 2364 107 9259 106 107 105 4368 107" 20

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .expect("usage: generate_gemma4 <MODEL_DIR> \"<ids>\" <max_new>");
    let ids_str = args
        .next()
        .expect("usage: generate_gemma4 <MODEL_DIR> \"<ids>\" <max_new>");
    let max_new: usize = args
        .next()
        .expect("usage: generate_gemma4 <MODEL_DIR> \"<ids>\" <max_new>")
        .parse()
        .expect("max_new must be a positive integer");

    let prompt_ids: Vec<i32> = ids_str
        .split_whitespace()
        .map(|s| s.parse::<i32>().expect("each id must be an integer"))
        .collect();

    eprintln!("Loading model from {model_dir} ...");
    let mut model = gemma4_mlx::load_model(&model_dir)?;

    eprintln!("Prompt ids ({} tokens): {:?}", prompt_ids.len(), &prompt_ids);
    eprintln!("Generating up to {max_new} tokens ...");

    let eos = gemma4_mlx::eos_ids();
    let generated = gemma4_mlx::generate_greedy(&mut model, &prompt_ids, max_new, &eos)?;

    // Print space-separated generated ids on stdout (for easy scripted parity checks)
    let out: Vec<String> = generated.iter().map(|id| id.to_string()).collect();
    println!("{}", out.join(" "));

    eprintln!("Generated {} tokens.", generated.len());
    Ok(())
}
