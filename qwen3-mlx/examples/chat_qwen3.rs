//! Interactive chat example with Qwen3 using proper chat templates

use std::env;
use std::io::{self, Write};
use std::path::Path;

use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::transforms::eval;
use mlx_rs::Array;
use mlx_lm_utils::tokenizer::{
    load_model_chat_template_from_file, ApplyChatTemplateArgs, Conversation, Role, Tokenizer,
};
use qwen3_mlx::{load_model, Generate, KVCache};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_dir>", args[0]);
        eprintln!("Example: {} ./Qwen3-4B-bf16", args[0]);
        std::process::exit(1);
    }

    let model_dir = Path::new(&args[1]);
    let model_id = model_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("qwen3")
        .to_string();

    println!("Loading model from: {}", model_dir.display());

    // Load tokenizer with chat template
    let tokenizer_file = model_dir.join("tokenizer.json");
    let tokenizer_config_file = model_dir.join("tokenizer_config.json");
    let mut tokenizer = Tokenizer::from_file(&tokenizer_file)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {:?}", e))?;
    let chat_template = load_model_chat_template_from_file(&tokenizer_config_file)?
        .expect("Chat template not found in tokenizer_config.json");

    // Load model
    let mut model = load_model(model_dir)?;
    println!("Model loaded. Type 'quit' to exit.\n");

    let temperature = 0.7;
    let max_tokens = 256;

    // Store conversation as owned strings
    let mut history: Vec<(String, String)> = Vec::new(); // (role, content)

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        let user_input = user_input.trim().to_string();

        if user_input.is_empty() {
            continue;
        }
        if user_input == "quit" || user_input == "exit" {
            break;
        }
        if user_input == "clear" {
            history.clear();
            println!("Conversation cleared.\n");
            continue;
        }

        // Add user message
        history.push(("user".to_string(), user_input.clone()));

        // Build conversations from history
        let conversations: Vec<Conversation<Role, &str>> = history
            .iter()
            .map(|(role, content)| Conversation {
                role: if role == "user" { Role::User } else { Role::Assistant },
                content: content.as_str(),
            })
            .collect();

        // Apply chat template
        let args = ApplyChatTemplateArgs {
            conversations: vec![conversations.into()],
            documents: None,
            model_id: &model_id,
            chat_template_id: None,
            add_generation_prompt: None,
            continue_final_message: None,
        };
        let encodings = tokenizer.apply_chat_template_and_encode(chat_template.clone(), args)?;
        let prompt: Vec<u32> = encodings
            .iter()
            .flat_map(|encoding| encoding.get_ids())
            .copied()
            .collect();
        let prompt_tokens = Array::from(&prompt[..]).index(NewAxis);

        // Generate response
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
        let mut response_text = String::new();

        for (i, token) in generator.enumerate() {
            let token = token?;
            let token_id = token.item::<u32>();

            // Check for EOS tokens (Qwen3: 151643, 151645)
            if token_id == 151643 || token_id == 151645 {
                break;
            }

            tokens.push(token);

            // Stream output every 5 tokens
            if tokens.len() % 5 == 0 {
                eval(&tokens)?;
                let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
                let text = tokenizer.decode(&slice, true)
                    .map_err(|e| anyhow::anyhow!("Decode error: {:?}", e))?;
                print!("{}", text);
                io::stdout().flush()?;
                response_text.push_str(&text);
            }

            if i >= max_tokens - 1 {
                break;
            }
        }

        // Flush remaining tokens
        if !tokens.is_empty() {
            eval(&tokens)?;
            let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
            let text = tokenizer.decode(&slice, true)
                .map_err(|e| anyhow::anyhow!("Decode error: {:?}", e))?;
            print!("{}", text);
            response_text.push_str(&text);
        }
        println!("\n");

        // Add assistant response to history
        history.push(("assistant".to_string(), response_text));
    }

    println!("Goodbye!");
    Ok(())
}
