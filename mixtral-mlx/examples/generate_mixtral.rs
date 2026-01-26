//! Simple text generation example with Mixtral

use std::env;
use std::time::Instant;
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::transforms::eval;
use mixtral_mlx::{load_model, load_tokenizer, Generate, KVCache, Error};

fn main() -> Result<(), Error> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_dir> [prompt]", args[0]);
        eprintln!("Note: Mixtral requires 4-bit quantized model. Use mlx_lm to quantize first.");
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let prompt = args.get(2).map(|s| s.as_str()).unwrap_or("Hello, I am a");

    println!("Loading Mixtral MoE model from: {}", model_dir);
    let start = Instant::now();

    let tokenizer = load_tokenizer(model_dir)?;
    let mut model = load_model(model_dir)?;

    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f32());

    let encoding = tokenizer.encode(prompt, true)?;
    let prompt_tokens = mlx_rs::Array::from(encoding.get_ids()).index(NewAxis);
    let prompt_len = encoding.get_ids().len();

    println!("Prompt ({} tokens): {}", prompt_len, prompt);
    println!("---");

    let mut cache = Vec::new();
    let temperature = 0.7;
    let max_tokens = 100;

    let generate_start = Instant::now();
    let generator = Generate::<KVCache>::new(&mut model, &mut cache, temperature, &prompt_tokens);

    let mut tokens = Vec::new();
    for (i, token) in generator.enumerate() {
        let token = token?;
        tokens.push(token.clone());

        if tokens.len() % 10 == 0 {
            eval(&tokens)?;
            let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
            print!("{}", tokenizer.decode(&slice, true)?);
        }

        if i >= max_tokens - 1 { break; }
    }

    if !tokens.is_empty() {
        eval(&tokens)?;
        let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
        print!("{}", tokenizer.decode(&slice, true)?);
    }
    println!();

    let gen_time = generate_start.elapsed().as_secs_f32();
    println!("---");
    println!("Generated {} tokens in {:.2}s ({:.1} tok/s)", max_tokens, gen_time, max_tokens as f32 / gen_time);

    Ok(())
}
