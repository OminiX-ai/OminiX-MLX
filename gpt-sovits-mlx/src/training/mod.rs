//! Training support for GPT-SoVITS voice cloning
//!
//! This module provides training infrastructure for fine-tuning the T2S model
//! on custom voice data.
//!
//! # Overview
//!
//! The training pipeline consists of:
//! 1. Data loading from preprocessed .npy files
//! 2. T2S model forward pass with teacher forcing
//! 3. CrossEntropy loss computation
//! 4. Gradient computation and parameter updates
//!
//! # Example
//!
//! ```rust,no_run
//! use gpt_sovits_mlx::training::{T2STrainer, TrainingConfig};
//!
//! let config = TrainingConfig::default()
//!     .with_learning_rate(1e-4)
//!     .with_batch_size(4);
//!
//! let mut trainer = T2STrainer::new(config)?;
//! trainer.load_pretrained("path/to/base_model.safetensors")?;
//!
//! let dataset = trainer.load_dataset("path/to/dataset")?;
//! trainer.train(&dataset)?;
//!
//! trainer.save("path/to/finetuned_model.safetensors")?;
//! ```

mod config;
mod dataset;
mod lr_scheduler;
mod trainer;

pub use config::TrainingConfig;
pub use dataset::{TrainingDataset, TrainingBatch};
pub use lr_scheduler::{LRScheduler, CosineScheduler, WarmupScheduler};
pub use trainer::T2STrainer;
