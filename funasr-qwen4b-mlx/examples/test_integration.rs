//! Test FunASR-Qwen4B integration with qwen3-mlx
//!
//! Run: cargo run --example test_integration --release

use funasr_qwen4b_mlx::{load_qwen3, load_qwen3_tokenizer, Generate, KVCache};
use mlx_rs::ops::indexing::{IndexOp, NewAxis};

fn main() -> anyhow::Result<()> {
    let model_dir = "models/Qwen3-4B";

    println!("=== FunASR-Qwen4B Integration Test ===\n");

    if !std::path::Path::new(model_dir).exists() {
        println!("Model not found: {}", model_dir);
        return Ok(());
    }

    // Load model and tokenizer using re-exported functions
    println!("Loading model via funasr_qwen4b_mlx re-exports...");
    let mut model = load_qwen3(model_dir)?;
    let tokenizer = load_qwen3_tokenizer(model_dir)?;
    println!("Model loaded: {}", model.model_type());

    // Test translation prompt
    let prompt = "<|im_start|>user\n翻译成英文：今天天气很好<|im_end|>\n<|im_start|>assistant\n";
    println!("\nPrompt: {}", prompt);

    let encoding = tokenizer.encode(prompt, false).map_err(|e| anyhow::anyhow!("{}", e))?;
    let prompt_tokens = mlx_rs::Array::from(encoding.get_ids()).index(NewAxis);

    let mut cache: Vec<Option<KVCache>> = Vec::new();
    let mut generated_tokens = Vec::new();

    let start = std::time::Instant::now();
    let generator = Generate::<KVCache>::new(&mut model, &mut cache, 0.0, &prompt_tokens);

    print!("Output: ");
    for token_result in generator.take(100) {
        let token = token_result?;
        let token_id: u32 = token.item();

        // EOS tokens
        if token_id == 151645 || token_id == 151643 {
            break;
        }

        generated_tokens.push(token_id);
        let text = tokenizer.decode(&[token_id], false).map_err(|e| anyhow::anyhow!("{}", e))?;
        print!("{}", text);
        std::io::Write::flush(&mut std::io::stdout())?;
    }

    let elapsed = start.elapsed();
    println!("\n\nGenerated {} tokens in {:.2?} ({:.1} tok/s)",
             generated_tokens.len(), elapsed,
             generated_tokens.len() as f64 / elapsed.as_secs_f64());

    // Test adaptor loading
    println!("\n=== Testing Adaptor ===");
    let adaptor_path = "adaptor_phase2_final.safetensors";
    if std::path::Path::new(adaptor_path).exists() {
        use funasr_qwen4b_mlx::adaptor::AudioAdaptorQwen4B;
        let mut adaptor = AudioAdaptorQwen4B::new()?;
        adaptor.load_weights(adaptor_path)?;
        println!("Adaptor loaded successfully!");

        // Test forward pass
        let dummy_input = mlx_rs::Array::zeros::<f32>(&[1, 10, 512])?;
        let output = adaptor.forward(&dummy_input)?;
        println!("Adaptor forward: {:?} -> {:?}", dummy_input.shape(), output.shape());
    } else {
        println!("Adaptor weights not found: {}", adaptor_path);
    }

    println!("\n=== Integration Test Complete ===");
    Ok(())
}
