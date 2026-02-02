//! VITS Training CLI for Fewshot Voice Cloning
//!
//! Train VITS (SoVITS) on preprocessed audio data for voice cloning.
//!
//! # Usage
//!
//! ```bash
//! # 1. Preprocess audio with Python GPT-SoVITS
//! # 2. Convert to training format
//! python scripts/convert_vits_training_data.py \
//!   --input /tmp/fewshot_1min \
//!   --output /tmp/fewshot_1min_vits
//!
//! # 3. Train VITS
//! cargo run --release --example train_vits -- \
//!   --data-dir /tmp/fewshot_1min_vits \
//!   --pretrained ~/.dora/models/primespeech/gpt-sovits-mlx/vits_pretrained_v2.safetensors \
//!   --output /tmp/vits_finetuned.safetensors \
//!   --epochs 4
//! ```

use std::path::PathBuf;

use clap::Parser;
use gpt_sovits_mlx::{
    error::Error,
    training::{VITSTrainer, VITSTrainingConfig, VITSDataset},
};

/// VITS Training CLI for Fewshot Voice Cloning
#[derive(Parser, Debug)]
#[command(name = "train_vits")]
#[command(about = "Train VITS model for voice cloning")]
struct Args {
    /// Training data directory (converted from GPT-SoVITS format)
    #[arg(long)]
    data_dir: PathBuf,

    /// Pretrained VITS model weights
    #[arg(long)]
    pretrained: Option<PathBuf>,

    /// Output path for finetuned model
    #[arg(long)]
    output: PathBuf,

    /// Number of training epochs
    #[arg(long, default_value = "4")]
    epochs: usize,

    /// Batch size
    #[arg(long, default_value = "2")]
    batch_size: usize,

    /// Generator learning rate (1e-5 recommended for finetuning without weight normalization)
    #[arg(long, default_value = "0.00001")]
    lr_g: f32,

    /// Discriminator learning rate (1e-5 recommended for finetuning)
    #[arg(long, default_value = "0.00001")]
    lr_d: f32,

    /// L2 regularization strength towards pretrained weights (prevents drift)
    #[arg(long, default_value = "0.001")]
    pretrained_reg: f32,

    /// Audio segment size for training (samples at 32kHz)
    #[arg(long, default_value = "20480")]
    segment_size: i32,

    /// Log every N steps
    #[arg(long, default_value = "10")]
    log_every: usize,

    /// Save checkpoint every N steps
    #[arg(long, default_value = "100")]
    save_every: usize,
}

fn main() -> Result<(), Error> {
    let args = Args::parse();

    println!("VITS Fewshot Training");
    println!("=====================");
    println!("Data directory: {:?}", args.data_dir);
    println!("Pretrained: {:?}", args.pretrained);
    println!("Output: {:?}", args.output);
    println!("Epochs: {}", args.epochs);
    println!("Batch size: {}", args.batch_size);
    println!("Learning rate (G/D): {}/{}", args.lr_g, args.lr_d);
    println!("Segment size: {} samples (~{:.1}ms)", args.segment_size, args.segment_size as f32 / 32.0);
    println!();

    // Load dataset
    println!("Loading dataset...");
    let mut dataset = VITSDataset::load(&args.data_dir)?;
    println!("  Loaded {} samples", dataset.len());
    println!("  Sample rate: {} Hz", dataset.sample_rate());
    println!();

    // Create training config
    let config = VITSTrainingConfig {
        learning_rate_g: args.lr_g,
        learning_rate_d: args.lr_d,
        batch_size: args.batch_size,
        segment_size: args.segment_size,
        log_every: args.log_every,
        save_every: args.save_every,
        pretrained_reg_strength: args.pretrained_reg,
        ..Default::default()
    };
    println!("Pretrained regularization: {}", args.pretrained_reg);

    // Create trainer
    println!("Creating trainer...");
    let mut trainer = VITSTrainer::new(config)?;

    // Load pretrained weights if specified
    if let Some(pretrained_path) = &args.pretrained {
        println!("Loading pretrained weights from {:?}...", pretrained_path);
        // Use regularization-aware loading to prevent weight drift during finetuning
        // This stores a copy of pretrained weights for L2 regularization
        trainer.load_generator_weights_with_regularization(pretrained_path)?;

        // For fewshot training: freeze encoder/flow, only train decoder
        // This preserves the learned phoneme representations and only adapts the voice timbre
        println!("Freezing encoder/flow layers (fewshot mode)...");
        trainer.freeze_non_decoder_layers();
    }

    // Compute total steps
    let steps_per_epoch = (dataset.len() + args.batch_size - 1) / args.batch_size;
    let total_steps = steps_per_epoch * args.epochs;
    println!("  Steps per epoch: {}", steps_per_epoch);
    println!("  Total steps: {}", total_steps);
    println!();

    // Training loop
    println!("Starting training...");
    println!("-------------------");

    let hop_length = 640; // Default for 32kHz audio

    for epoch in 0..args.epochs {
        println!("\nEpoch {}/{}", epoch + 1, args.epochs);

        // Shuffle at start of each epoch
        dataset.shuffle(Some(epoch as u64 * 42));

        let mut epoch_loss_d = 0.0;
        let mut epoch_loss_g = 0.0;
        let mut epoch_loss_mel = 0.0;
        let mut epoch_steps = 0;

        // Iterate over batches
        for (batch_idx, batch_result) in dataset
            .iter_batches(args.batch_size, args.segment_size, hop_length)
            .enumerate()
        {
            let batch = batch_result?;

            // Training step
            let losses = trainer.train_step(&batch)?;

            epoch_loss_d += losses.loss_d;
            epoch_loss_g += losses.loss_gen;
            epoch_loss_mel += losses.loss_mel;
            epoch_steps += 1;

            // Log progress
            let global_step = epoch * steps_per_epoch + batch_idx;
            if global_step % args.log_every == 0 || batch_idx == steps_per_epoch - 1 {
                println!(
                    "  Step {:4} | D: {:.4} | G: {:.4} | FM: {:.4} | Mel: {:.4} | KL: {:.4} | Reg: {:.4}",
                    global_step,
                    losses.loss_d,
                    losses.loss_gen,
                    losses.loss_fm,
                    losses.loss_mel,
                    losses.loss_kl,
                    losses.loss_reg,
                );
            }

            // Save checkpoint
            if global_step > 0 && global_step % args.save_every == 0 {
                let checkpoint_path = args.output.with_extension(format!("step{}.safetensors", global_step));
                trainer.save_checkpoint(&checkpoint_path)?;
                println!("  Saved checkpoint: {:?}", checkpoint_path);
            }
        }

        // Print epoch summary
        if epoch_steps > 0 {
            println!(
                "\n  Epoch {} Summary: D={:.4}, G={:.4}, Mel={:.4}",
                epoch + 1,
                epoch_loss_d / epoch_steps as f32,
                epoch_loss_g / epoch_steps as f32,
                epoch_loss_mel / epoch_steps as f32,
            );
        }
    }

    // Save final model (full checkpoint with discriminator)
    println!("\nSaving final checkpoint to {:?}...", args.output);
    trainer.save_checkpoint(&args.output)?;

    // Save generator-only for inference
    let gen_output = args.output.with_extension("generator.safetensors");
    println!("Saving generator weights to {:?}...", gen_output);
    trainer.save_generator(&gen_output)?;

    println!("Training complete!");

    Ok(())
}
