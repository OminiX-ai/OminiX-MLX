use std::io::Write;

use minicpm_sala_mlx::{create_layer_caches, get_model_args, load_model, load_tokenizer, sample};
use mlx_rs::ops::indexing::IndexOp;

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

    eprintln!("Loading model from {}...", model_dir);
    let model_args = get_model_args(model_dir)?;
    eprintln!(
        "  {} layers ({} sparse, {} lightning), {} params",
        model_args.num_hidden_layers,
        model_args.mixer_types.iter().filter(|t| *t == "minicpm4").count(),
        model_args.mixer_types.iter().filter(|t| *t == "lightning-attn").count(),
        "9B",
    );

    let tokenizer = load_tokenizer(model_dir)?;
    let mut model = load_model(model_dir)?;
    let mut caches = create_layer_caches(&model.args);

    eprintln!("Generating with temp={}, max_tokens={}...\n", temperature, max_tokens);

    let encoding = tokenizer.encode(prompt.as_str(), true).map_err(|e| anyhow::anyhow!("{e}"))?;
    let prompt_tokens = encoding.get_ids();
    let input = mlx_rs::Array::from_slice(
        &prompt_tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
        &[1, prompt_tokens.len() as i32],
    );

    // Prefill
    let logits = model.forward(&input, &mut caches)?;
    let last_logits = logits.index((.., -1, ..));
    let mut token = sample(&last_logits, temperature)?;

    // Incremental decode: accumulate all tokens and decode the full
    // sequence each time, printing only new characters for correct spacing.
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut prev_text_len = 0;

    print!("{}", prompt);

    for _ in 0..max_tokens {
        let token_id = token.item::<u32>();
        if token_id == 2 || token_id == 73440 {
            break; // EOS
        }

        generated_ids.push(token_id);
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
    println!();

    Ok(())
}
