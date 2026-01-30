//! VITS Training CLI
//!
//! This example demonstrates how to fine-tune VITS (SoVITS) models on custom voice data.
//!
//! Usage:
//! ```bash
//! cargo run --release --example train_vits -- \
//!     --data-dir /path/to/training_data \
//!     --pretrained /path/to/pretrained.safetensors \
//!     --output /path/to/finetuned.safetensors \
//!     --lr 0.0002 \
//!     --batch-size 4 \
//!     --max-steps 1000
//! ```
//!
//! Expected data format:
//! ```
//! training_data/
//! ├── ssl_features/    # HuBERT features [ssl_dim, seq_len] .npy files
//! ├── spec/            # Linear spectrogram [n_fft/2+1, frames] .npy files
//! ├── audio/           # Audio waveforms [samples] .npy files
//! ├── phonemes/        # Phoneme indices [text_len] .npy files
//! ├── refer_mel/       # Reference mel [mel_channels, time] .npy files
//! └── metadata.json    # Sample list
//! ```

use std::path::PathBuf;

use clap::Parser;
use gpt_sovits_mlx::{
    training::{VITSTrainer, VITSTrainingConfig, VITSBatch},
    error::Error,
};
use mlx_rs::Array;

/// VITS Training CLI for voice cloning
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to training data directory
    #[arg(long)]
    data_dir: PathBuf,

    /// Path to pretrained model weights (safetensors)
    #[arg(long)]
    pretrained: Option<PathBuf>,

    /// Path to save finetuned model
    #[arg(long, default_value = "vits_finetuned.safetensors")]
    output: PathBuf,

    /// Generator learning rate
    #[arg(long, default_value = "0.0002")]
    lr_g: f32,

    /// Discriminator learning rate
    #[arg(long, default_value = "0.0002")]
    lr_d: f32,

    /// Batch size
    #[arg(long, default_value = "4")]
    batch_size: usize,

    /// Maximum training steps
    #[arg(long, default_value = "1000")]
    max_steps: usize,

    /// Mel loss weight
    #[arg(long, default_value = "45.0")]
    c_mel: f32,

    /// KL loss weight
    #[arg(long, default_value = "1.0")]
    c_kl: f32,

    /// Feature matching loss weight
    #[arg(long, default_value = "2.0")]
    c_fm: f32,

    /// Save checkpoint every N steps
    #[arg(long, default_value = "500")]
    save_every: usize,

    /// Log every N steps
    #[arg(long, default_value = "10")]
    log_every: usize,
}

/// Load a single training sample from files
fn load_sample(
    data_dir: &PathBuf,
    sample_id: &str,
) -> Result<VITSBatch, Error> {
    // Load SSL features
    let ssl_path = data_dir.join("ssl_features").join(format!("{}.npy", sample_id));
    let ssl_features = Array::load_numpy(&ssl_path)
        .map_err(|e| Error::Message(format!("Failed to load SSL features: {}", e)))?;
    // Add batch dimension: [dim, seq] -> [1, dim, seq]
    let ssl_features = ssl_features.reshape(&[1, ssl_features.dim(0) as i32, ssl_features.dim(1) as i32])
        .map_err(|e| Error::Message(e.to_string()))?;

    // Load spectrogram
    let spec_path = data_dir.join("spec").join(format!("{}.npy", sample_id));
    let spec = Array::load_numpy(&spec_path)
        .map_err(|e| Error::Message(format!("Failed to load spec: {}", e)))?;
    let spec = spec.reshape(&[1, spec.dim(0) as i32, spec.dim(1) as i32])
        .map_err(|e| Error::Message(e.to_string()))?;

    // Load audio
    let audio_path = data_dir.join("audio").join(format!("{}.npy", sample_id));
    let audio = Array::load_numpy(&audio_path)
        .map_err(|e| Error::Message(format!("Failed to load audio: {}", e)))?;
    // Add batch and channel dims: [samples] -> [1, 1, samples]
    let audio = audio.reshape(&[1, 1, audio.dim(0) as i32])
        .map_err(|e| Error::Message(e.to_string()))?;

    // Load phonemes
    let phoneme_path = data_dir.join("phonemes").join(format!("{}.npy", sample_id));
    let text = Array::load_numpy(&phoneme_path)
        .map_err(|e| Error::Message(format!("Failed to load phonemes: {}", e)))?;
    let text = text.reshape(&[1, text.dim(0) as i32])
        .map_err(|e| Error::Message(e.to_string()))?;

    // Load reference mel
    let mel_path = data_dir.join("refer_mel").join(format!("{}.npy", sample_id));
    let refer_mel = Array::load_numpy(&mel_path)
        .map_err(|e| Error::Message(format!("Failed to load refer_mel: {}", e)))?;
    let refer_mel = refer_mel.reshape(&[1, refer_mel.dim(0) as i32, refer_mel.dim(1) as i32])
        .map_err(|e| Error::Message(e.to_string()))?;

    // Spec lengths (full length for single sample)
    let spec_lengths = Array::from_slice(&[spec.dim(2) as i32], &[1]);
    let text_lengths = Array::from_slice(&[text.dim(1) as i32], &[1]);

    Ok(VITSBatch {
        ssl_features,
        spec,
        spec_lengths,
        text,
        text_lengths,
        audio,
        refer_mel,
    })
}

/// Load metadata and create batch iterator
fn load_dataset(
    data_dir: &PathBuf,
) -> Result<Vec<String>, Error> {
    let metadata_path = data_dir.join("metadata.json");
    let metadata_str = std::fs::read_to_string(&metadata_path)
        .map_err(|e| Error::Message(format!("Failed to read metadata: {}", e)))?;

    // Parse JSON to get sample IDs
    let metadata: serde_json::Value = serde_json::from_str(&metadata_str)
        .map_err(|e| Error::Message(format!("Failed to parse metadata: {}", e)))?;

    let samples = metadata["samples"]
        .as_array()
        .ok_or_else(|| Error::Message("metadata.json missing 'samples' array".to_string()))?;

    let sample_ids: Vec<String> = samples
        .iter()
        .filter_map(|s| s["id"].as_str().map(|id| id.to_string()))
        .collect();

    Ok(sample_ids)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("VITS Training");
    println!("=============");
    println!("Data dir: {:?}", args.data_dir);
    println!("Pretrained: {:?}", args.pretrained);
    println!("Output: {:?}", args.output);
    println!("LR (G/D): {}/{}", args.lr_g, args.lr_d);
    println!("Batch size: {}", args.batch_size);
    println!("Max steps: {}", args.max_steps);
    println!();

    // Create training config
    let config = VITSTrainingConfig {
        learning_rate_g: args.lr_g,
        learning_rate_d: args.lr_d,
        batch_size: args.batch_size,
        c_mel: args.c_mel,
        c_kl: args.c_kl,
        c_fm: args.c_fm,
        max_steps: args.max_steps,
        save_every: args.save_every,
        log_every: args.log_every,
        ..Default::default()
    };

    // Create trainer
    println!("Creating VITS trainer...");
    let mut trainer = VITSTrainer::new(config)?;

    // Load pretrained weights if provided
    if let Some(ref pretrained_path) = args.pretrained {
        println!("Loading pretrained weights from {:?}", pretrained_path);
        trainer.load_generator_weights(pretrained_path)?;
    }

    // Load dataset
    println!("Loading dataset...");
    let sample_ids = load_dataset(&args.data_dir)?;
    println!("Found {} samples", sample_ids.len());

    // Create batch iterator
    let batches = sample_ids
        .iter()
        .cycle()  // Repeat dataset
        .take(args.max_steps)
        .filter_map(|id| load_sample(&args.data_dir, id).ok());

    // Train
    println!("\nStarting training...\n");
    trainer.train(batches)?;

    // Save final checkpoint
    println!("\nSaving final model to {:?}", args.output);
    trainer.save_checkpoint(&args.output)?;

    println!("\nTraining complete!");
    Ok(())
}
