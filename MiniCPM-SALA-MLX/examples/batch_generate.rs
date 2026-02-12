use std::io::Write;
use std::time::Instant;

use minicpm_sala_mlx::{create_layer_caches, get_model_args, load_model, load_tokenizer, sample};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::eval;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <model_dir> <prompt> [batch_size] [max_tokens] [temperature]",
            args[0]
        );
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let prompt = &args[2];
    let batch_size: i32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let max_tokens: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(100);
    let temperature: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.0);

    // Load model
    let load_start = Instant::now();
    eprintln!("Loading model from {}...", model_dir);
    let model_args = get_model_args(model_dir)?;
    eprintln!(
        "  {} layers ({} sparse, {} lightning)",
        model_args.num_hidden_layers,
        model_args.mixer_types.iter().filter(|t| *t == "minicpm4").count(),
        model_args
            .mixer_types
            .iter()
            .filter(|t| *t == "lightning-attn")
            .count(),
    );
    if let Some(q) = &model_args.quantization {
        eprintln!("  Quantized: {} bits, group_size={}", q.bits, q.group_size);
    }

    let tokenizer = load_tokenizer(model_dir)?;
    let mut model = load_model(model_dir)?;
    let mut caches = create_layer_caches(&model.args);

    let load_time = load_start.elapsed().as_secs_f32();
    eprintln!("Model loaded in {:.2}s", load_time);
    eprintln!(
        "Generating with batch_size={}, temp={}, max_tokens={}...\n",
        batch_size, temperature, max_tokens
    );

    // Tokenize prompt
    let encoding = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let prompt_tokens = encoding.get_ids();
    let prompt_len = prompt_tokens.len() as i32;

    // Create batched input: [B, prompt_len] by repeating prompt B times
    let single_tokens: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
    let batched_tokens: Vec<i32> = single_tokens
        .iter()
        .cycle()
        .take(single_tokens.len() * batch_size as usize)
        .cloned()
        .collect();
    let input = mlx_rs::Array::from_slice(&batched_tokens, &[batch_size, prompt_len]);

    // Prefill
    let prefill_start = Instant::now();
    let logits = model.forward(&input, &mut caches)?;
    eval([&logits])?;

    let last_logits = logits.index((.., -1, ..)); // [B, vocab_size]
    let mut tokens = sample(&last_logits, temperature)?; // [B]
    eval([&tokens])?;
    let prefill_time = prefill_start.elapsed().as_secs_f32();

    // Per-sequence tracking
    let mut all_generated: Vec<Vec<u32>> = vec![Vec::new(); batch_size as usize];
    let mut finished: Vec<bool> = vec![false; batch_size as usize];
    let mut prev_text_len = 0;

    // Print prompt
    print!("{}", prompt);
    std::io::stdout().flush()?;

    let decode_start = Instant::now();
    let mut num_decode_steps = 0;

    for _step in 0..max_tokens {
        // Extract per-sequence tokens
        eval([&tokens])?;
        let mut all_done = true;

        for b in 0..batch_size as usize {
            if finished[b] {
                continue;
            }
            let token_id = if batch_size == 1 {
                tokens.item::<u32>()
            } else {
                tokens.index(b as i32).item::<u32>()
            };
            if token_id == 2 || token_id == 73440 {
                finished[b] = true;
                continue;
            }
            all_generated[b].push(token_id);
            all_done = false;
        }

        if all_done {
            break;
        }

        num_decode_steps += 1;

        // Print sequence 0 incrementally (new chars only)
        if !finished[0] && !all_generated[0].is_empty() {
            if let Ok(full_text) = tokenizer.decode(&all_generated[0], true) {
                let new_text = &full_text[prev_text_len..];
                if !new_text.is_empty() {
                    print!("{}", new_text);
                    std::io::stdout().flush()?;
                }
                prev_text_len = full_text.len();
            }
        }

        // Decode step: [B, 1]
        let input = tokens.reshape(&[batch_size, 1])?;
        let logits = model.forward(&input, &mut caches)?;

        let last_logits = logits.index((.., -1, ..)); // [B, vocab_size]
        tokens = sample(&last_logits, temperature)?;
    }
    eval([&tokens])?;
    let decode_time = decode_start.elapsed().as_secs_f32();

    println!();

    // Count total generated tokens across all sequences
    let total_generated: usize = all_generated.iter().map(|g| g.len()).sum();

    eprintln!();
    eprintln!("--- Stats ---");
    eprintln!("Batch size:     {}", batch_size);
    eprintln!(
        "Prompt tokens:  {} × {} = {}",
        prompt_len,
        batch_size,
        prompt_len as i32 * batch_size
    );
    eprintln!(
        "Prefill:        {:.2}s ({:.1} tok/s total)",
        prefill_time,
        prompt_len as f32 * batch_size as f32 / prefill_time
    );
    eprintln!(
        "Decode steps:   {} (generated {} tokens total)",
        num_decode_steps, total_generated
    );
    let total_tok_s = total_generated as f32 / decode_time;
    let per_seq_tok_s = total_tok_s / batch_size as f32;
    eprintln!(
        "Decode:         {:.2}s ({:.1} tok/s total, {:.1} tok/s/seq)",
        decode_time, total_tok_s, per_seq_tok_s
    );
    eprintln!(
        "Total:          {:.2}s",
        prefill_time + decode_time
    );

    // Print all B sequences if batch_size > 1
    if batch_size > 1 {
        eprintln!();
        eprintln!("--- All sequences ---");
        for (b, gen) in all_generated.iter().enumerate() {
            if let Ok(text) = tokenizer.decode(gen, true) {
                eprintln!("[seq {}] {}{}", b, prompt, text);
            }
        }
    }

    Ok(())
}
