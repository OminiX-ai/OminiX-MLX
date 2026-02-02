//! Debug parameter names

use funasr_qwen4b_mlx::qwen4b::{Qwen4BConfig, Qwen4BModel};
use funasr_qwen4b_mlx::error::Result;
use mlx_rs::module::ModuleParameters;

fn main() -> Result<()> {
    let config = Qwen4BConfig::default();
    let model = Qwen4BModel::new(config)?;

    // Print parameter names
    let params = model.parameters().flatten();
    println!("Rust model parameter keys ({} total):", params.len());
    for (i, key) in params.keys().enumerate() {
        if i < 15 || i > params.len() - 5 {
            println!("  {}", key);
        } else if i == 15 {
            println!("  ...");
        }
    }
    Ok(())
}
