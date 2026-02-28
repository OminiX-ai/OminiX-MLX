use std::collections::HashSet;
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use qwen3_5_35b_mlx::{load_model, load_tokenizer, Generate};

fn main() -> anyhow::Result<()> {
    // Check for --cpu flag
    let use_cpu = std::env::args().any(|a| a == "--cpu");
    if use_cpu {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        eprintln!("Using CPU device");
    } else {
        eprintln!("Using GPU device");
    }

    let args: Vec<String> = std::env::args().filter(|a| a != "--cpu").collect();
    let model_dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "./models/Qwen3.5-27B-4bit".to_string());

    eprintln!("Loading model from {}...", model_dir);
    let start = std::time::Instant::now();
    let tokenizer = load_tokenizer(&model_dir)?;
    let mut model = load_model(&model_dir)?;
    eprintln!("Model loaded in {:.1}s", start.elapsed().as_secs_f64());

    let prompt = args.get(2).cloned().unwrap_or_else(|| {
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
            .to_string()
    });

    let encoding = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let prompt_tokens = mlx_rs::Array::from(encoding.get_ids());
    let prompt_tokens = prompt_tokens.index(NewAxis); // [1, L]
    eprintln!(
        "Prompt: {} tokens",
        encoding.get_ids().len()
    );

    let max_tokens: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    let temp: f32 = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.7);

    // Load EOS token IDs from config.json
    let eos_tokens: HashSet<u32> = {
        let config_path = std::path::Path::new(&model_dir).join("config.json");
        let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        match &config["eos_token_id"] {
            serde_json::Value::Array(ids) => {
                ids.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect()
            }
            serde_json::Value::Number(n) => {
                let mut s = HashSet::new();
                s.insert(n.as_u64().unwrap_or(248044) as u32);
                s
            }
            _ => {
                let mut s = HashSet::new();
                s.insert(248044u32);
                s
            }
        }
    };
    let gen = Generate::new(&mut model, temp, &prompt_tokens);

    let start = std::time::Instant::now();
    let mut token_count = 0;
    let mut ttft: Option<f64> = None;

    for token_result in gen.take(max_tokens) {
        let token = token_result?;
        if ttft.is_none() {
            ttft = Some(start.elapsed().as_secs_f64());
        }
        let token_id = token.item::<u32>();
        if eos_tokens.contains(&token_id) {
            break;
        }

        let text = tokenizer
            .decode(&[token_id], true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        print!("{}", text);
        token_count += 1;
    }
    println!();

    let elapsed = start.elapsed();
    let decode_time = elapsed.as_secs_f64() - ttft.unwrap_or(0.0);
    let decode_tokens = if token_count > 1 { token_count - 1 } else { 0 };
    let decode_tps = if decode_tokens > 0 {
        decode_tokens as f64 / decode_time
    } else {
        0.0
    };
    eprintln!(
        "TTFT: {:.2}s | Decode: {} tok in {:.2}s ({:.1} tok/s) | Total: {} tok in {:.2}s",
        ttft.unwrap_or(0.0),
        decode_tokens,
        decode_time,
        decode_tps,
        token_count,
        elapsed.as_secs_f64(),
    );

    Ok(())
}
