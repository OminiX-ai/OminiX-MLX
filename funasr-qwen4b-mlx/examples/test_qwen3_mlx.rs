//! Test Qwen3-4B using qwen3-mlx crate directly
//!
//! cargo run --example test_qwen3_mlx --release

use qwen3_mlx::{load_model, load_tokenizer, Generate, KVCache};
use mlx_rs::ops::indexing::{IndexOp, NewAxis};

fn main() -> anyhow::Result<()> {
    let model_dir = "models/Qwen3-4B";

    println!("=== Qwen3-4B Test (using qwen3-mlx) ===\n");

    if !std::path::Path::new(model_dir).exists() {
        println!("Model not found at: {}", model_dir);
        return Ok(());
    }

    println!("Loading tokenizer...");
    let tokenizer = load_tokenizer(model_dir)?;
    println!("Tokenizer loaded: {} tokens", tokenizer.get_vocab_size(false));

    println!("\nLoading model...");
    let mut model = load_model(model_dir)?;
    println!("Model loaded successfully!");
    println!("Model type: {}", model.model_type());

    // Test generation
    let prompt = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n";
    println!("\nPrompt: {}", prompt);

    let encoding = tokenizer.encode(prompt, false).map_err(|e| anyhow::anyhow!("{}", e))?;
    let prompt_tokens = mlx_rs::Array::from(encoding.get_ids()).index(NewAxis);
    println!("Prompt tokens: {} tokens", encoding.get_ids().len());

    let mut cache: Vec<Option<KVCache>> = Vec::new();

    println!("\nGenerating...");
    let start = std::time::Instant::now();

    let generator = Generate::<KVCache>::new(&mut model, &mut cache, 0.0, &prompt_tokens);

    let mut generated_tokens = Vec::new();
    let im_end_id = 151645u32;
    let eos_id = 151643u32;

    for token_result in generator.take(100) {
        let token = token_result?;
        let token_id: u32 = token.item();

        if token_id == im_end_id || token_id == eos_id {
            println!("\n[EOS]");
            break;
        }

        generated_tokens.push(token_id);

        let text = tokenizer.decode(&[token_id], false).map_err(|e| anyhow::anyhow!("{}", e))?;
        print!("{}", text);
        std::io::Write::flush(&mut std::io::stdout())?;
    }

    let elapsed = start.elapsed();
    println!("\n\nGenerated {} tokens in {:.2?}", generated_tokens.len(), elapsed);
    println!("Speed: {:.1} tokens/sec", generated_tokens.len() as f64 / elapsed.as_secs_f64());

    let full_output = tokenizer.decode(&generated_tokens, true).map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("\n=== Full Output ===\n{}", full_output);

    Ok(())
}
