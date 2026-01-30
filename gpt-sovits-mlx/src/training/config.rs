//! Training configuration

use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Configuration for T2S model training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    // === Optimization ===
    /// Base learning rate (default: 1e-4)
    pub learning_rate: f32,

    /// Weight decay for AdamW (default: 0.01)
    pub weight_decay: f32,

    /// Adam beta1 (default: 0.9)
    pub beta1: f32,

    /// Adam beta2 (default: 0.999)
    pub beta2: f32,

    /// Epsilon for numerical stability (default: 1e-8)
    pub epsilon: f32,

    /// Gradient clipping max norm (default: 1.0)
    pub max_grad_norm: f32,

    // === Schedule ===
    /// Number of warmup steps (default: 1000)
    pub warmup_steps: usize,

    /// Maximum training steps (default: 100000)
    pub max_steps: usize,

    /// Batch size (default: 4)
    pub batch_size: usize,

    /// Gradient accumulation steps (default: 1)
    pub gradient_accumulation_steps: usize,

    // === Checkpointing ===
    /// Directory for saving checkpoints
    pub checkpoint_dir: PathBuf,

    /// Save checkpoint every N steps (default: 1000)
    pub save_every_n_steps: usize,

    /// Keep only last N checkpoints (default: 5, 0 = keep all)
    pub keep_last_n_checkpoints: usize,

    // === Logging ===
    /// Log metrics every N steps (default: 100)
    pub log_every_n_steps: usize,

    // === Model ===
    /// Path to pretrained T2S model weights
    pub base_model_path: Option<PathBuf>,

    /// Freeze BERT projection layer (default: true)
    pub freeze_bert_proj: bool,

    /// Freeze text embedding layer (default: false)
    pub freeze_text_embedding: bool,

    // === Data ===
    /// Maximum sequence length for phonemes (default: 512)
    pub max_phoneme_len: usize,

    /// Maximum sequence length for semantic tokens (default: 1024)
    pub max_semantic_len: usize,

    /// Shuffle training data (default: true)
    pub shuffle: bool,

    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            // Optimization
            learning_rate: 1e-4,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            max_grad_norm: 1.0,

            // Schedule
            warmup_steps: 1000,
            max_steps: 100000,
            batch_size: 4,
            gradient_accumulation_steps: 1,

            // Checkpointing
            checkpoint_dir: PathBuf::from("checkpoints"),
            save_every_n_steps: 1000,
            keep_last_n_checkpoints: 5,

            // Logging
            log_every_n_steps: 100,

            // Model
            base_model_path: None,
            freeze_bert_proj: true,
            freeze_text_embedding: false,

            // Data
            max_phoneme_len: 512,
            max_semantic_len: 1024,
            shuffle: true,
            seed: 42,
        }
    }
}

impl TrainingConfig {
    /// Create a new config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set warmup steps
    pub fn with_warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Set max training steps
    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps = steps;
        self
    }

    /// Set checkpoint directory
    pub fn with_checkpoint_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.checkpoint_dir = dir.into();
        self
    }

    /// Set base model path
    pub fn with_base_model(mut self, path: impl Into<PathBuf>) -> Self {
        self.base_model_path = Some(path.into());
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set log interval
    pub fn with_log_interval(mut self, steps: usize) -> Self {
        self.log_every_n_steps = steps;
        self
    }

    /// Load config from YAML file
    pub fn from_yaml(path: impl AsRef<std::path::Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Save config to YAML file
    pub fn to_yaml(&self, path: impl AsRef<std::path::Path>) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_yaml::to_string(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be positive".to_string());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be at least 1".to_string());
        }
        if self.max_steps == 0 {
            return Err("max_steps must be at least 1".to_string());
        }
        if self.warmup_steps >= self.max_steps {
            return Err("warmup_steps must be less than max_steps".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TrainingConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder_pattern() {
        let config = TrainingConfig::new()
            .with_learning_rate(5e-5)
            .with_batch_size(8)
            .with_max_steps(50000);

        assert_eq!(config.learning_rate, 5e-5);
        assert_eq!(config.batch_size, 8);
        assert_eq!(config.max_steps, 50000);
    }
}
