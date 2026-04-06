use std::{env, error::Error, path::PathBuf};

use gemma4_mlx::{load_model, load_tokenizer, Generate, KVCache, EOS_TOKEN_IDS};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    Array,
};

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    let model_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-4-26B-A4B-it"));
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "Write a short haiku about Gemma 4.".to_string());

    let mut model = load_model(&model_dir)?;
    let tokenizer = load_tokenizer(&model_dir)?;

    let encoding = tokenizer.encode(prompt, true)?;
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);

    let mut cache = Vec::<KVCache>::new();
    let generator = Generate::new(&mut model, &mut cache, 0.0, &prompt_tokens);

    for token in generator.take(2048) {
        let token = token?;
        let token_id = token.item::<u32>();
        if EOS_TOKEN_IDS.contains(&token_id) {
            break;
        }
        print!("{}", tokenizer.decode(&[token_id], true)?);
    }
    println!();

    Ok(())
}
