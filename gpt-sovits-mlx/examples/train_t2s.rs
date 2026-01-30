//! T2S Model Training Example
//!
//! This example demonstrates how to finetune a T2S model for voice cloning.
//!
//! # Usage
//!
//! First, preprocess your audio data using the Python script:
//! ```bash
//! python scripts/preprocess_voice.py \
//!     --input-dir data/audio \
//!     --output-dir data/training \
//!     --base-model-dir /path/to/base/model
//! ```
//!
//! Then run training:
//! ```bash
//! cargo run --release --example train_t2s -- \
//!     --dataset data/training \
//!     --base-model /path/to/pretrained/t2s.safetensors \
//!     --output checkpoints \
//!     --max-steps 1000 \
//!     --batch-size 4
//! ```

use std::path::PathBuf;

use gpt_sovits_mlx::{
    training::{T2STrainer, TrainingConfig, TrainingDataset},
    Error,
};

fn main() -> Result<(), Error> {
    // Parse command-line arguments
    let args: Vec<String> = std::env::args().collect();

    let mut dataset_path = PathBuf::from("data/training");
    let mut base_model_path = PathBuf::from("checkpoints/base_t2s.safetensors");
    let mut output_dir = PathBuf::from("checkpoints");
    let mut max_steps: usize = 1000;
    let mut batch_size: usize = 4;
    let mut learning_rate: f32 = 1e-4;
    let mut warmup_steps: usize = 100;
    let mut log_interval: usize = 1;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dataset" => {
                i += 1;
                dataset_path = PathBuf::from(&args[i]);
            }
            "--base-model" => {
                i += 1;
                base_model_path = PathBuf::from(&args[i]);
            }
            "--output" => {
                i += 1;
                output_dir = PathBuf::from(&args[i]);
            }
            "--max-steps" => {
                i += 1;
                max_steps = args[i].parse().expect("Invalid max-steps");
            }
            "--batch-size" => {
                i += 1;
                batch_size = args[i].parse().expect("Invalid batch-size");
            }
            "--learning-rate" | "--lr" => {
                i += 1;
                learning_rate = args[i].parse().expect("Invalid learning-rate");
            }
            "--warmup-steps" => {
                i += 1;
                warmup_steps = args[i].parse().expect("Invalid warmup-steps");
            }
            "--log-interval" => {
                i += 1;
                log_interval = args[i].parse().expect("Invalid log-interval");
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    println!("T2S Model Training");
    println!("==================");
    println!("Dataset: {:?}", dataset_path);
    println!("Base model: {:?}", base_model_path);
    println!("Output directory: {:?}", output_dir);
    println!("Max steps: {}", max_steps);
    println!("Batch size: {}", batch_size);
    println!("Learning rate: {}", learning_rate);
    println!("Warmup steps: {}", warmup_steps);
    println!();

    // Create training configuration
    let config = TrainingConfig::new()
        .with_learning_rate(learning_rate)
        .with_batch_size(batch_size)
        .with_max_steps(max_steps)
        .with_warmup_steps(warmup_steps)
        .with_checkpoint_dir(output_dir)
        .with_log_interval(log_interval);

    // Create trainer
    let mut trainer = T2STrainer::new(config)?;

    // Load pretrained model
    println!("Loading pretrained model...");
    trainer.load_pretrained(&base_model_path)?;

    // Load dataset
    println!("Loading dataset...");
    let dataset = TrainingDataset::load(&dataset_path)?;
    println!("Dataset loaded: {} samples", dataset.len());

    // Run training
    trainer.train(&dataset)?;

    println!("\nTraining completed!");
    Ok(())
}

fn print_help() {
    println!("T2S Model Training Example");
    println!();
    println!("USAGE:");
    println!("    cargo run --release --example train_t2s -- [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    --dataset <PATH>       Path to preprocessed training dataset");
    println!("    --base-model <PATH>    Path to pretrained T2S model weights (.safetensors)");
    println!("    --output <DIR>         Output directory for checkpoints");
    println!("    --max-steps <N>        Maximum training steps (default: 1000)");
    println!("    --batch-size <N>       Batch size (default: 4)");
    println!("    --learning-rate <F>    Learning rate (default: 1e-4)");
    println!("    --warmup-steps <N>     Number of warmup steps (default: 100)");
    println!("    --help, -h             Print this help message");
    println!();
    println!("EXAMPLE:");
    println!("    cargo run --release --example train_t2s -- \\");
    println!("        --dataset data/training \\");
    println!("        --base-model checkpoints/base_t2s.safetensors \\");
    println!("        --output checkpoints/finetuned \\");
    println!("        --max-steps 2000");
}
