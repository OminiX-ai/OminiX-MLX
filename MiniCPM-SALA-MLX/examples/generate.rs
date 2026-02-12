use std::io::Write;
use std::time::Instant;

use minicpm_sala_mlx::{create_layer_caches, get_model_args, load_model, load_tokenizer, sample};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::eval;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model_dir> <prompt> [max_tokens] [temperature]", args[0]);
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let prompt = &args[2];
    let max_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let temperature: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.9);

    // Load model
    let load_start = Instant::now();
    eprintln!("Loading model from {}...", model_dir);
    let model_args = get_model_args(model_dir)?;
    eprintln!(
        "  {} layers ({} sparse, {} lightning)",
        model_args.num_hidden_layers,
        model_args.mixer_types.iter().filter(|t| *t == "minicpm4").count(),
        model_args.mixer_types.iter().filter(|t| *t == "lightning-attn").count(),
    );
    if let Some(q) = &model_args.quantization {
        eprintln!("  Quantized: {} bits, group_size={}", q.bits, q.group_size);
    }

    let tokenizer = load_tokenizer(model_dir)?;
    let mut model = load_model(model_dir)?;
    let mut caches = create_layer_caches(&model.args);

    let load_time = load_start.elapsed().as_secs_f32();
    eprintln!("Model loaded in {:.2}s", load_time);
    eprintln!("Generating with temp={}, max_tokens={}...\n", temperature, max_tokens);

    let encoding = tokenizer.encode(prompt.as_str(), true).map_err(|e| anyhow::anyhow!("{e}"))?;
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
    let mut token = sample(&last_logits, temperature)?;
    eval([&token])?;
    let prefill_time = prefill_start.elapsed().as_secs_f32();

    // Incremental decode: accumulate all tokens and decode the full
    // sequence each time, printing only new characters for correct spacing.
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut prev_text_len = 0;

    print!("{}", prompt);

    let decode_start = Instant::now();
    let mut num_decode_tokens = 0;

    for _ in 0..max_tokens {
        let token_id = token.item::<u32>();
        if token_id == 2 || token_id == 73440 {
            break; // EOS
        }

        generated_ids.push(token_id);
        num_decode_tokens += 1;

        if let Ok(full_text) = tokenizer.decode(&generated_ids, true) {
            let new_text = &full_text[prev_text_len..];
            if !new_text.is_empty() {
                print!("{}", new_text);
                std::io::stdout().flush()?;
            }
            prev_text_len = full_text.len();
        }

        // Decode step
        let input = token.reshape(&[1, 1])?;
        let logits = model.forward(&input, &mut caches)?;
        let last_logits = logits.index((.., -1, ..));
        token = sample(&last_logits, temperature)?;
    }
    eval([&token])?;
    let decode_time = decode_start.elapsed().as_secs_f32();

    println!();
    eprintln!();
    eprintln!("--- Stats ---");
    eprintln!("Prompt tokens:  {}", prompt_len);
    eprintln!("Prefill:        {:.2}s ({:.1} tok/s)", prefill_time, prompt_len as f32 / prefill_time);
    eprintln!("Decode tokens:  {}", num_decode_tokens);
    eprintln!("Decode:         {:.2}s ({:.1} tok/s)", decode_time, num_decode_tokens as f32 / decode_time);
    eprintln!("Total:          {:.2}s", prefill_time + decode_time);

    Ok(())
}
