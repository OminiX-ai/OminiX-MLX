use std::io::{self, BufRead, Write};
use std::time::Instant;

use clap::Parser;
use minicpm_sala_mlx::{
    create_layer_caches, format_chat_prompt, get_model_args, load_model, load_tokenizer, sample,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::eval;

#[derive(Parser)]
#[command(name = "chat", about = "Interactive chat with MiniCPM-SALA")]
struct Args {
    /// Path to model directory
    model_dir: String,

    /// Maximum tokens per response
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// System prompt
    #[arg(long, default_value = "You are a helpful assistant.")]
    system: String,

    /// Hide <think>...</think> reasoning
    #[arg(long)]
    no_think: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Load model
    let load_start = Instant::now();
    eprintln!("Loading model from {}...", args.model_dir);
    let model_args = get_model_args(&args.model_dir)?;
    eprintln!(
        "  {} layers ({} sparse, {} lightning)",
        model_args.num_hidden_layers,
        model_args
            .mixer_types
            .iter()
            .filter(|t| *t == "minicpm4")
            .count(),
        model_args
            .mixer_types
            .iter()
            .filter(|t| *t == "lightning-attn")
            .count(),
    );
    if let Some(q) = &model_args.quantization {
        eprintln!("  Quantized: {} bits, group_size={}", q.bits, q.group_size);
    }

    let tokenizer = load_tokenizer(&args.model_dir)?;
    let mut model = load_model(&args.model_dir)?;
    let mut caches = create_layer_caches(&model.args);

    eprintln!("Model loaded in {:.2}s", load_start.elapsed().as_secs_f32());
    eprintln!(
        "Settings: temp={}, max_tokens={}, no_think={}",
        args.temperature, args.max_tokens, args.no_think
    );
    eprintln!("Type your message and press Enter. Type 'quit' or Ctrl-D to exit.\n");

    let stdin = io::stdin();
    let mut turn_num = 0usize;
    let mut last_response_text = String::new();

    loop {
        // Prompt
        eprint!("> ");
        io::stderr().flush()?;

        let mut user_input = String::new();
        if stdin.lock().read_line(&mut user_input)? == 0 {
            eprintln!("\nBye!");
            break;
        }
        let user_input = user_input.trim();
        if user_input.is_empty() {
            continue;
        }
        if user_input == "quit" || user_input == "exit" {
            eprintln!("Bye!");
            break;
        }

        // Build prompt tokens for this turn
        let (prompt_tokens, prompt_label) = if turn_num == 0 {
            // First turn: full prompt with system message, BOS added by tokenizer
            let full_prompt = format_chat_prompt(&args.system, user_input);
            let encoding = tokenizer
                .encode(full_prompt.as_str(), true)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            (encoding.get_ids().to_vec(), "full prompt")
        } else {
            // Continuation: previous response + im_end + new user turn + assistant prefix
            // Include prev response so tokenizer sees correct boundary context
            // Caches already hold everything BEFORE the response, so we feed:
            //   {prev_response}<|im_end|>\n<|im_start|>user\n{new_msg}<|im_end|>\n<|im_start|>assistant\n
            let continuation = format!(
                "{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                last_response_text, user_input
            );
            let encoding = tokenizer
                .encode(continuation.as_str(), false) // false = no BOS
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            (encoding.get_ids().to_vec(), "continuation")
        };

        let prompt_len = prompt_tokens.len();
        let input = mlx_rs::Array::from_slice(
            &prompt_tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
            &[1, prompt_len as i32],
        );

        // Prefill (reusing caches from previous turns)
        let gen_start = Instant::now();
        let logits = model.forward(&input, &mut caches)?;
        let last_logits = logits.index((.., -1, ..));
        let mut token = sample(&last_logits, args.temperature)?;
        eval([&token])?;
        let prefill_time = gen_start.elapsed().as_secs_f32();

        // Decode
        let mut generated_ids: Vec<u32> = Vec::new();
        let mut prev_text_len = 0;
        let mut think_done = false;

        let decode_start = Instant::now();
        let mut num_decode_tokens = 0;

        for _ in 0..args.max_tokens {
            let token_id = token.item::<u32>();
            if token_id == 2 || token_id == 73440 {
                break;
            }

            generated_ids.push(token_id);
            num_decode_tokens += 1;

            if let Ok(full_text) = tokenizer.decode(&generated_ids, true) {
                if args.no_think {
                    if !think_done {
                        if let Some(end) = full_text.find("</think>") {
                            think_done = true;
                            let after = &full_text[end + "</think>".len()..];
                            let trimmed = after.trim_start_matches('\n');
                            if !trimmed.is_empty() {
                                print!("{}", trimmed);
                                io::stdout().flush()?;
                            }
                            prev_text_len = full_text.len();
                        }
                    } else if full_text.len() > prev_text_len {
                        print!("{}", &full_text[prev_text_len..]);
                        io::stdout().flush()?;
                        prev_text_len = full_text.len();
                    }
                } else {
                    if full_text.len() > prev_text_len {
                        print!("{}", &full_text[prev_text_len..]);
                        io::stdout().flush()?;
                    }
                    prev_text_len = full_text.len();
                }
            }

            let input = token.reshape(&[1, 1])?;
            let logits = model.forward(&input, &mut caches)?;
            let last_logits = logits.index((.., -1, ..));
            token = sample(&last_logits, args.temperature)?;
        }
        eval([&token])?;
        let decode_time = decode_start.elapsed().as_secs_f32();
        let total_time = gen_start.elapsed().as_secs_f32();

        println!();

        // Store response for next turn's continuation
        last_response_text = if !generated_ids.is_empty() {
            tokenizer.decode(&generated_ids, true).unwrap_or_default()
        } else {
            String::new()
        };
        turn_num += 1;

        eprintln!(
            "[{} {} tokens, prefill {:.2}s ({:.0} tok/s) + {} decoded in {:.2}s ({:.1} tok/s), total {:.2}s]",
            prompt_label,
            prompt_len,
            prefill_time,
            prompt_len as f32 / prefill_time,
            num_decode_tokens,
            decode_time,
            num_decode_tokens as f32 / decode_time,
            total_time,
        );
        eprintln!();
    }

    Ok(())
}
