//! Debug weight loading

use funasr_qwen4b_mlx::qwen4b::{Qwen4BConfig, Qwen4BModel};
use funasr_qwen4b_mlx::error::Result;
use mlx_rs::module::ModuleParameters;
use mlx_rs::Array;

fn main() -> Result<()> {
    let config = Qwen4BConfig::default();
    let mut model = Qwen4BModel::new(config)?;

    // Load weights
    model.load_weights("models/Qwen3-4B/model_mlx.safetensors")?;

    // Check some weight statistics
    println!("\n=== Weight Statistics ===");

    let params = model.parameters().flatten();
    let mut total_nonzero = 0;
    let mut checked = 0;

    for (key, arr) in params.iter() {
        let shape = arr.shape();
        let arr_f32 = arr.as_dtype(mlx_rs::Dtype::Float32);
        let arr_f32 = arr_f32.as_ref().unwrap_or(arr);

        // Sum absolute values to check if weights are non-zero
        let sum = mlx_rs::ops::sum(arr_f32, None).unwrap();
        let sum_val: f32 = sum.item();

        if checked < 10 {
            println!("  {}: shape={:?}, sum={:.4}", key, shape, sum_val);
            checked += 1;
        }

        if sum_val.abs() > 1e-6 {
            total_nonzero += 1;
        }
    }

    println!("\nNon-zero weights: {} / {}", total_nonzero, params.len());

    // Check embed_tokens specifically
    println!("\n=== Embedding Check ===");
    let embed = &model.embed_tokens.weight;
    let embed_sum = mlx_rs::ops::sum(embed, None)?;
    let embed_sum_val: f32 = embed_sum.item();
    println!("embed_tokens.weight sum: {:.4}", embed_sum_val);
    println!("embed_tokens.weight shape: {:?}", embed.shape());

    Ok(())
}
