//! Convert FunASR Paraformer model from PyTorch to MLX-compatible safetensors format.
//!
//! Pure Rust implementation using mlx-rs-core conversion utilities.
//!
//! # Download Model
//!
//! ```bash
//! git lfs install
//! git clone https://modelscope.cn/models/damo/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git ./paraformer-src
//! ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --features convert --example convert_model -- ./paraformer-src ./models/paraformer
//! ```

use std::fs;
use std::path::Path;

use mlx_rs_core::convert::{convert_paraformer, load_pytorch_model, paraformer_weight_mapping};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: {} <input_dir> <output_dir>", args[0]);
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  input_dir   Path to FunASR model directory (containing model.pt)");
        eprintln!("  output_dir  Output directory for converted model");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} ./paraformer-src ./models/paraformer", args[0]);
        eprintln!();
        eprintln!("Download model first:");
        eprintln!("  git lfs install");
        eprintln!("  git clone https://modelscope.cn/models/damo/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git ./paraformer-src");
        std::process::exit(1);
    }

    let input_dir = Path::new(&args[1]);
    let output_dir = Path::new(&args[2]);

    // Find model file
    let model_path = if input_dir.join("model.pt").exists() {
        input_dir.join("model.pt")
    } else if input_dir.join("model.pb").exists() {
        input_dir.join("model.pb")
    } else {
        eprintln!("Error: model.pt or model.pb not found in {:?}", input_dir);
        eprintln!("Available files:");
        for entry in fs::read_dir(input_dir)? {
            if let Ok(entry) = entry {
                eprintln!("  {:?}", entry.path());
            }
        }
        std::process::exit(1);
    };

    println!("Loading model from {:?}...", model_path);

    // Load PyTorch model to show tensor count
    let tensors = load_pytorch_model(&model_path)?;
    println!("Found {} tensors in checkpoint", tensors.len());

    // Get mapping for statistics
    let mapping = paraformer_weight_mapping();

    // Convert using the shared conversion function
    println!("\nConverting model...");
    let (converted_count, unmapped_count) = convert_paraformer(input_dir, output_dir)?;

    println!("\nConversion summary:");
    println!("  Converted: {} tensors", converted_count);
    println!("  Unmapped: {} tensors (SEACo components not used by base Paraformer)", unmapped_count);
    println!("  Mapping entries: {}", mapping.len());

    // Print output file info
    let output_path = output_dir.join("paraformer.safetensors");
    if output_path.exists() {
        let size_mb = fs::metadata(&output_path)?.len() as f64 / (1024.0 * 1024.0);
        println!("\nOutput:");
        println!("  {:?}", output_path);
        println!("  Size: {:.1} MB", size_mb);
    }

    println!("\nDone! Model saved to {:?}", output_dir);
    println!();
    println!("Usage in funasr-mlx:");
    println!("  let model = load_model(\"{}/paraformer.safetensors\")?;", output_dir.display());
    println!("  let (addshift, rescale) = parse_cmvn_file(\"{}/am.mvn\")?;", output_dir.display());
    println!("  model.set_cmvn(addshift, rescale);");

    Ok(())
}
