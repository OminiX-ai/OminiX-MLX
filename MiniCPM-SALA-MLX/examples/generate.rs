use std::io::Write;
use std::time::Instant;

use clap::Parser;
use minicpm_sala_mlx::{
    create_layer_caches, format_chat_prompt, get_model_args, load_model, load_tokenizer, sample,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::eval;

#[derive(Parser)]
#[command(name = "generate", about = "MiniCPM-SALA text generation")]
struct Args {
    /// Path to model directory
    model_dir: String,

    /// Prompt text
    prompt: String,

    /// Maximum tokens to generate
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Raw completion mode (no chat template)
    #[arg(long)]
    raw: bool,

    /// System prompt for chat mode
    #[arg(long, default_value = "You are a helpful assistant.")]
    system: String,

    /// Hide <think>...</think> reasoning and only show final answer
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
        model_args.mixer_types.iter().filter(|t| *t == "minicpm4").count(),
        model_args.mixer_types.iter().filter(|t| *t == "lightning-attn").count(),
    );
    if let Some(q) = &model_args.quantization {
        eprintln!("  Quantized: {} bits, group_size={}", q.bits, q.group_size);
    }

    let tokenizer = load_tokenizer(&args.model_dir)?;
    let mut model = load_model(&args.model_dir)?;
    let mut caches = create_layer_caches(&model.args);

    let load_time = load_start.elapsed().as_secs_f32();
    eprintln!("Model loaded in {:.2}s", load_time);

    // Format prompt
    let full_prompt = if args.raw {
        args.prompt.clone()
    } else {
        format_chat_prompt(&args.system, &args.prompt)
    };

    eprintln!(
        "Generating with temp={}, max_tokens={}, mode={}...\n",
        args.temperature,
        args.max_tokens,
        if args.raw { "raw" } else { "chat" }
    );

    let encoding = tokenizer
        .encode(full_prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let prompt_tokens = encoding.get_ids();
    let prompt_len = prompt_tokens.len();
    let input = mlx_rs::Array::from_slice(
        &prompt_tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
        &[1, prompt_len as i32],
    );

    // Prefill
    let prefill_start = Instant::now();
    let logits = model.forward(&input, &mut caches)?;
    let last_logits = logits.index((.., -1, ..));
    let mut token = sample(&last_logits, args.temperature)?;
    eval([&token])?;
    let prefill_time = prefill_start.elapsed().as_secs_f32();

    // Decode
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut prev_text_len = 0;
    let mut think_done = false; // for --no-think: have we passed </think>?

    let decode_start = Instant::now();
    let mut num_decode_tokens = 0;

    for _ in 0..args.max_tokens {
        let token_id = token.item::<u32>();
        if token_id == 2 || token_id == 73440 {
            break; // EOS or <|im_end|>
        }

        generated_ids.push(token_id);
        num_decode_tokens += 1;

        // Print incrementally
        if let Ok(full_text) = tokenizer.decode(&generated_ids, true) {
            if args.no_think {
                if !think_done {
                    if let Some(end) = full_text.find("</think>") {
                        think_done = true;
                        let after = &full_text[end + "</think>".len()..];
                        let trimmed = after.trim_start_matches('\n');
                        if !trimmed.is_empty() {
                            print!("{}", trimmed);
                            std::io::stdout().flush()?;
                        }
                        prev_text_len = full_text.len();
                    }
                    // Still in <think> block — don't print anything
                } else {
                    // Past </think>, print new chars
                    if full_text.len() > prev_text_len {
                        print!("{}", &full_text[prev_text_len..]);
                        std::io::stdout().flush()?;
                    }
                    prev_text_len = full_text.len();
                }
            } else {
                if full_text.len() > prev_text_len {
                    print!("{}", &full_text[prev_text_len..]);
                    std::io::stdout().flush()?;
                }
                prev_text_len = full_text.len();
            }
        }

        // Decode step
        let input = token.reshape(&[1, 1])?;
        let logits = model.forward(&input, &mut caches)?;
        let last_logits = logits.index((.., -1, ..));
        token = sample(&last_logits, args.temperature)?;
    }
    eval([&token])?;
    let decode_time = decode_start.elapsed().as_secs_f32();

    println!();
    eprintln!();
    eprintln!("--- Stats ---");
    eprintln!("Prompt tokens:  {}", prompt_len);
    eprintln!(
        "Prefill:        {:.2}s ({:.1} tok/s)",
        prefill_time,
        prompt_len as f32 / prefill_time
    );
    eprintln!("Decode tokens:  {}", num_decode_tokens);
    eprintln!(
        "Decode:         {:.2}s ({:.1} tok/s)",
        decode_time,
        num_decode_tokens as f32 / decode_time
    );
    eprintln!("Total:          {:.2}s", prefill_time + decode_time);

    Ok(())
}
