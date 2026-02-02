//! Training support for GPT-SoVITS voice cloning
//!
//! This module provides training infrastructure for fine-tuning the T2S and VITS models
//! on custom voice data.
//!
//! # Overview
//!
//! The training pipeline consists of:
//! - **T2S Training**: CrossEntropy loss for text-to-semantic prediction
//! - **VITS Training**: GAN training with discriminator for audio synthesis
//!
//! # T2S Training Example
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
//!
//! # VITS Training Example
//!
//! ```rust,no_run
//! use gpt_sovits_mlx::training::{VITSTrainer, VITSTrainingConfig};
//!
//! let config = VITSTrainingConfig::default();
//! let mut trainer = VITSTrainer::new(config)?;
//! trainer.load_generator_weights("path/to/pretrained.safetensors")?;
//! // ... training loop
//! ```

// T2S Training
mod config;
mod dataset;
mod lr_scheduler;
mod trainer;

// VITS Training
mod vits_dataset;
mod vits_loss;
mod vits_trainer;

// T2S exports
pub use config::TrainingConfig;
pub use dataset::{TrainingDataset, TrainingBatch};
pub use lr_scheduler::{LRScheduler, CosineScheduler, WarmupScheduler};
pub use trainer::T2STrainer;

// VITS exports
pub use vits_loss::{
    generator_loss, discriminator_loss, feature_matching_loss,
    kl_loss, mel_reconstruction_loss,
};
pub use vits_dataset::{VITSDataset, VITSDatasetMetadata, VITSSampleMetadata};
pub use vits_trainer::{VITSTrainer, VITSTrainingConfig, VITSLosses, VITSBatch};
