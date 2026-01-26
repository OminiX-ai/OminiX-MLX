//! Interactive chat example with Qwen3

use std::env;
use std::io::{self, Write};
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::transforms::eval;
use qwen3_mlx::{load_model, load_tokenizer, Generate, KVCache, Error};

fn main() -> Result<(), Error> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_dir>", args[0]);
        std::process::exit(1);
    }

    let model_dir = &args[1];

    println!("Loading model from: {}", model_dir);
    let tokenizer = load_tokenizer(model_dir)?;
    let mut model = load_model(model_dir)?;
    println!("Model loaded. Type 'quit' to exit.\n");

    let system_prompt = "You are a helpful assistant.";
    let temperature = 0.7;
    let max_tokens = 200;

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        let user_input = user_input.trim();

        if user_input.is_empty() {
            continue;
        }
        if user_input == "quit" || user_input == "exit" {
            break;
        }

        // Format as chat (simplified - adjust based on model's chat template)
        let prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system_prompt, user_input
        );

        let encoding = tokenizer.encode(prompt.as_str(), true)?;
        let prompt_tokens = mlx_rs::Array::from(encoding.get_ids()).index(NewAxis);

        let mut cache = Vec::new();
        let generator = Generate::<KVCache>::new(
            &mut model,
            &mut cache,
            temperature,
            &prompt_tokens,
        );

        print!("Assistant: ");
        io::stdout().flush()?;

        let mut tokens = Vec::new();
        for (i, token) in generator.enumerate() {
            let token = token?;
            let token_id = token.item::<u32>();

            // Check for EOS tokens
            if token_id == 151643 || token_id == 151645 { // Qwen3 EOS tokens
                break;
            }

            tokens.push(token);

            if tokens.len() % 5 == 0 {
                eval(&tokens)?;
                let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
                let text = tokenizer.decode(&slice, true)?;
                print!("{}", text);
                io::stdout().flush()?;
            }

            if i >= max_tokens - 1 {
                break;
            }
        }

        if !tokens.is_empty() {
            eval(&tokens)?;
            let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
            let text = tokenizer.decode(&slice, true)?;
            print!("{}", text);
        }
        println!("\n");
    }

    println!("Goodbye!");
    Ok(())
}
