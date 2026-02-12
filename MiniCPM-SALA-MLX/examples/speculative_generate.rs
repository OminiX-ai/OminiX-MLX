use std::io::Write;
use std::time::Instant;

use clap::Parser;
use minicpm_sala_mlx::{
    create_layer_caches, format_chat_prompt, get_model_args, load_model, load_tokenizer, sample,
    SpeculativeDecoder,
};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::eval;

#[derive(Parser)]
#[command(
    name = "speculative_generate",
    about = "MiniCPM-SALA with self-speculative decoding"
)]
struct Args {
    /// Path to model directory
    model_dir: String,

    /// Prompt text
    prompt: String,

    /// Maximum tokens to generate
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// Number of layers for draft model
    #[arg(long, default_value_t = 8)]
    draft_layers: usize,

    /// Number of draft tokens per speculation round
    #[arg(long, default_value_t = 4)]
    num_draft: usize,

    /// Raw completion mode (no chat template)
    #[arg(long)]
    raw: bool,

    /// System prompt for chat mode
    #[arg(long, default_value = "You are a helpful assistant.")]
    system: String,
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

    let load_time = load_start.elapsed().as_secs_f32();
    eprintln!("Model loaded in {:.2}s", load_time);

    let speculator = SpeculativeDecoder::new(args.draft_layers, args.num_draft, args.temperature);
    eprintln!(
        "Speculative: draft_layers={}, num_draft={}, temp={}",
        args.draft_layers, args.num_draft, args.temperature
    );

    // Format prompt
    let full_prompt = if args.raw {
        args.prompt.clone()
    } else {
        format_chat_prompt(&args.system, &args.prompt)
    };

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
    eprintln!("Prefilling {} tokens...", prompt_len);
    let prefill_start = Instant::now();
    let logits = model.forward(&input, &mut caches)?;
    let last_logits = logits.index((.., -1, ..));
    let first_token = sample(&last_logits, args.temperature)?;
    eval([&first_token])?;
    let prefill_time = prefill_start.elapsed().as_secs_f32();
    eprintln!(
        "Prefill: {:.2}s ({:.1} tok/s)\n",
        prefill_time,
        prompt_len as f32 / prefill_time
    );

    // Speculative decode loop
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut prev_text_len = 0;
    let mut total_spec_rounds = 0;
    let mut total_accepted = 0;
    let mut total_draft_proposed = 0;

    let first_id = first_token.item::<u32>();
    if first_id != 2 && first_id != 73440 {
        generated_ids.push(first_id);
    }

    let decode_start = Instant::now();
    let mut last_token = first_token;

    while generated_ids.len() < args.max_tokens {
        let last_id = last_token.item::<u32>();
        if last_id == 2 || last_id == 73440 {
            break;
        }

        let result = speculator.step(&mut model, &mut caches, &last_token)?;

        total_spec_rounds += 1;
        total_accepted += result.num_accepted;
        total_draft_proposed += args.num_draft;

        for &tid in &result.tokens {
            if tid == 2 || tid == 73440 {
                break;
            }
            generated_ids.push(tid);
        }

        // Set last_token for next round
        if let Some(&last_id) = result.tokens.last() {
            last_token = mlx_rs::Array::from_slice(&[last_id as i32], &[1]);
        } else {
            break;
        }

        // Print incrementally
        if let Ok(full_text) = tokenizer.decode(&generated_ids, true) {
            if full_text.len() > prev_text_len {
                print!("{}", &full_text[prev_text_len..]);
                std::io::stdout().flush()?;
            }
            prev_text_len = full_text.len();
        }

        // Check for EOS in result
        if result.tokens.iter().any(|&t| t == 2 || t == 73440) {
            break;
        }
    }

    let decode_time = decode_start.elapsed().as_secs_f32();

    println!();
    eprintln!();
    eprintln!("--- Speculative Decode Stats ---");
    eprintln!("Prompt tokens:       {}", prompt_len);
    eprintln!(
        "Prefill:             {:.2}s ({:.1} tok/s)",
        prefill_time,
        prompt_len as f32 / prefill_time
    );
    eprintln!("Generated tokens:    {}", generated_ids.len());
    eprintln!("Speculation rounds:  {}", total_spec_rounds);
    eprintln!(
        "Acceptance rate:     {:.1}% ({}/{})",
        if total_draft_proposed > 0 {
            total_accepted as f32 / total_draft_proposed as f32 * 100.0
        } else {
            0.0
        },
        total_accepted,
        total_draft_proposed,
    );
    let avg_tokens_per_round = if total_spec_rounds > 0 {
        generated_ids.len() as f32 / total_spec_rounds as f32
    } else {
        0.0
    };
    eprintln!(
        "Avg tokens/round:    {:.2} (vs 1.0 standard)",
        avg_tokens_per_round
    );
    eprintln!(
        "Decode:              {:.2}s ({:.1} tok/s)",
        decode_time,
        generated_ids.len() as f32 / decode_time
    );
    eprintln!(
        "Total:               {:.2}s",
        prefill_time + decode_time
    );

    Ok(())
}
