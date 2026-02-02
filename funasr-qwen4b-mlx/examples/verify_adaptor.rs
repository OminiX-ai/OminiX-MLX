//! Verify adaptor output matches PyTorch
//!
//! Run: cargo run --example verify_adaptor --release

use funasr_qwen4b_mlx::adaptor::AudioAdaptorQwen4B;
use funasr_qwen4b_mlx::error::Result;
use mlx_rs::Array;

fn main() -> Result<()> {
    println!("=== Adaptor Verification ===\n");

    let mut adaptor = AudioAdaptorQwen4B::new()?;
    adaptor.load_weights("adaptor_phase2_final.safetensors")?;
    println!("Adaptor loaded");

    // Test with zeros (same as PyTorch test)
    let zeros = Array::zeros::<f32>(&[1, 94, 512])?;
    let out = adaptor.forward(&zeros)?;
    mlx_rs::transforms::eval([&out])?;

    let mean = mlx_rs::ops::mean(&out, None)?;
    let var = mlx_rs::ops::var(&out, None, None)?;
    let min_val = mlx_rs::ops::min(&out, None)?;
    let max_val = mlx_rs::ops::max(&out, None)?;

    mlx_rs::transforms::eval([&mean, &var, &min_val, &max_val])?;

    let std_val = var.item::<f32>().sqrt();

    println!("\nRust adaptor output (zeros input):");
    println!("  Shape: {:?}", out.shape());
    println!("  Mean: {:.6}", mean.item::<f32>());
    println!("  Std:  {:.6}", std_val);
    println!("  Min:  {:.6}", min_val.item::<f32>());
    println!("  Max:  {:.6}", max_val.item::<f32>());

    println!("\nPyTorch reference (zeros input):");
    println!("  Mean: -0.000251");
    println!("  Std:  0.990634");
    println!("  Min:  -5.845397");
    println!("  Max:  3.721543");

    // Check if within tolerance
    let mean_diff = (mean.item::<f32>() + 0.000251).abs();
    let std_diff = (std_val - 0.990634).abs();

    println!("\nDifferences:");
    println!("  Mean diff: {:.6}", mean_diff);
    println!("  Std diff:  {:.6}", std_diff);

    if mean_diff < 0.01 && std_diff < 0.1 {
        println!("\n✓ Adaptor output matches PyTorch!");
    } else {
        println!("\n✗ Adaptor output differs from PyTorch");
    }

    Ok(())
}
