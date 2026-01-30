//! T2S Model Trainer

use std::path::Path;
use std::time::Instant;

use mlx_rs::{
    Array,
    array,
    module::ModuleParameters,
    nn,
    optimizers::{AdamW, Optimizer, clip_grad_norm},
    transforms::eval,
};

use crate::error::Error;
use crate::models::t2s::{T2SModel, T2SConfig, load_t2s_model};

use super::config::TrainingConfig;
use super::dataset::{TrainingDataset, TrainingBatch};
use super::lr_scheduler::{LRScheduler, CosineScheduler};

/// Training state that can be checkpointed
#[derive(Debug)]
pub struct TrainingState {
    /// Current training step
    pub step: usize,
    /// Best validation loss seen
    pub best_loss: f32,
    /// Total training time in seconds
    pub total_time_secs: f64,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            step: 0,
            best_loss: f32::MAX,
            total_time_secs: 0.0,
        }
    }
}

/// T2S Model Trainer
pub struct T2STrainer {
    /// Training configuration
    config: TrainingConfig,
    /// T2S model
    model: Option<T2SModel>,
    /// T2S model configuration
    model_config: T2SConfig,
    /// Optimizer
    optimizer: Option<AdamW>,
    /// Learning rate scheduler
    scheduler: CosineScheduler,
    /// Training state
    state: TrainingState,
}

impl T2STrainer {
    /// Create a new trainer
    pub fn new(config: TrainingConfig) -> Result<Self, Error> {
        config.validate().map_err(|e| Error::Message(e))?;

        // Create scheduler
        let scheduler = CosineScheduler::new(
            config.learning_rate,
            config.warmup_steps,
            config.max_steps,
        );

        Ok(Self {
            config,
            model: None,
            model_config: T2SConfig::default(),
            optimizer: None,
            scheduler,
            state: TrainingState::default(),
        })
    }

    /// Load pretrained model weights
    pub fn load_pretrained(&mut self, path: impl AsRef<Path>) -> Result<(), Error> {
        let path = path.as_ref();
        println!("Loading pretrained model from {:?}", path);

        // Load model using existing function
        let model = load_t2s_model(path)?;

        self.model = Some(model);
        self.model_config = T2SConfig::default();

        // Initialize optimizer with model parameters
        self.init_optimizer()?;

        println!("  Model loaded successfully");
        Ok(())
    }

    /// Initialize a new model (for training from scratch)
    pub fn init_model(&mut self, config: T2SConfig) -> Result<(), Error> {
        println!("Initializing new T2S model");

        let model = T2SModel::new(config.clone())?;

        self.model = Some(model);
        self.model_config = config;

        // Initialize optimizer
        self.init_optimizer()?;

        println!("  Model initialized");
        Ok(())
    }

    /// Initialize optimizer with current model
    fn init_optimizer(&mut self) -> Result<(), Error> {
        let _model = self.model.as_ref()
            .ok_or_else(|| Error::Message("Model not loaded".to_string()))?;

        // Note: Using default betas, eps, and weight_decay
        // AdamW defaults: betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        let optimizer = AdamW::new(self.config.learning_rate);

        self.optimizer = Some(optimizer);
        Ok(())
    }

    /// Update optimizer learning rate from scheduler
    fn update_lr(&mut self) {
        if let Some(ref mut optimizer) = self.optimizer {
            let lr = self.scheduler.get_lr();
            optimizer.lr = array!(lr);
        }
    }

    /// Run a single training step with gradient computation and parameter update
    pub fn train_step(&mut self, batch: &TrainingBatch) -> Result<f32, Error> {
        // Update optimizer learning rate from scheduler
        self.update_lr();

        // Take ownership temporarily for the closure
        let mut model = self.model.take()
            .ok_or_else(|| Error::Message("Model not loaded".to_string()))?;
        let mut optimizer = self.optimizer.take()
            .ok_or_else(|| Error::Message("Optimizer not initialized".to_string()))?;

        // Clone batch data for the closure (Arrays are reference-counted, cheap to clone)
        let phoneme_ids = batch.phoneme_ids.clone();
        let phoneme_lens = batch.phoneme_lens.clone();
        let bert_features = batch.bert_features.clone();
        let semantic_ids = batch.semantic_ids.clone();
        let semantic_lens = batch.semantic_lens.clone();

        // Define loss function closure
        // Takes model and batch data, returns loss scalar
        let loss_fn = |model: &mut T2SModel,
                       (ph_ids, ph_lens, bert, sem_ids, sem_lens): (&Array, &Array, &Array, &Array, &Array)|
                       -> Result<Array, mlx_rs::error::Exception> {
            // Forward pass
            let logits = model.forward_train(ph_ids, ph_lens, bert, sem_ids, sem_lens)?;
            // Compute cross-entropy loss
            compute_cross_entropy_loss_inner(&logits, sem_ids, sem_lens)
        };

        // Create value_and_grad function
        let mut value_and_grad = nn::value_and_grad(loss_fn);

        // Compute loss and gradients
        let batch_args = (&phoneme_ids, &phoneme_lens, &bert_features, &semantic_ids, &semantic_lens);
        let (loss, gradients) = value_and_grad(&mut model, batch_args)
            .map_err(|e| Error::Message(format!("Gradient computation failed: {}", e)))?;

        // Evaluate to materialize loss value
        eval([&loss]).map_err(|e| Error::Message(e.to_string()))?;
        let loss_value = loss.item::<f32>();

        // Clip gradients
        let max_grad_norm = self.config.max_grad_norm;
        let (clipped_gradients, _grad_norm) = clip_grad_norm(&gradients, max_grad_norm)
            .map_err(|e| Error::Message(format!("Gradient clipping failed: {}", e)))?;

        // Convert clipped gradients to owned arrays for optimizer
        let owned_gradients: mlx_rs::module::FlattenedModuleParam = clipped_gradients
            .into_iter()
            .map(|(k, v)| (k, v.into_owned()))
            .collect();

        // Apply gradients to model parameters
        optimizer.update(&mut model, &owned_gradients)
            .map_err(|e| Error::Message(format!("Optimizer update failed: {}", e)))?;

        // Evaluate updated parameters
        let params: Vec<_> = model.trainable_parameters().flatten().into_iter().map(|(_, v)| v.clone()).collect();
        eval(params.iter()).map_err(|e| Error::Message(e.to_string()))?;

        // Put model and optimizer back
        self.model = Some(model);
        self.optimizer = Some(optimizer);

        // Update scheduler and step counter
        self.scheduler.step();
        self.state.step += 1;

        Ok(loss_value)
    }

    /// Train on dataset
    pub fn train(&mut self, dataset: &TrainingDataset) -> Result<(), Error> {
        if self.model.is_none() {
            return Err(Error::Message("Model not loaded. Call load_pretrained() first.".to_string()));
        }

        println!("Starting training...");
        println!("  Dataset size: {} samples", dataset.len());
        println!("  Batch size: {}", self.config.batch_size);
        println!("  Max steps: {}", self.config.max_steps);
        println!("  Learning rate: {}", self.config.learning_rate);

        // Create checkpoint directory
        std::fs::create_dir_all(&self.config.checkpoint_dir)?;

        let start_time = Instant::now();
        let mut epoch = 0;
        let mut running_loss = 0.0;
        let mut loss_count = 0;

        while self.state.step < self.config.max_steps {
            epoch += 1;
            println!("\nEpoch {}", epoch);

            // Shuffle dataset at start of each epoch
            // Note: We'd need mutable access to dataset for shuffling
            // For now, iterate in order

            for batch_result in dataset.iter_batches(self.config.batch_size) {
                let batch = batch_result?;

                // Training step
                let loss = self.train_step(&batch)?;
                running_loss += loss;
                loss_count += 1;

                // Logging
                if self.state.step % self.config.log_every_n_steps == 0 {
                    let avg_loss = running_loss / loss_count as f32;
                    let lr = self.scheduler.get_lr();
                    let elapsed = start_time.elapsed().as_secs_f64();

                    println!(
                        "  Step {}/{}: loss={:.4}, lr={:.2e}, time={:.1}s",
                        self.state.step, self.config.max_steps, avg_loss, lr, elapsed
                    );

                    running_loss = 0.0;
                    loss_count = 0;
                }

                // Checkpointing
                if self.state.step % self.config.save_every_n_steps == 0 {
                    let checkpoint_path = self.config.checkpoint_dir
                        .join(format!("checkpoint-{}.safetensors", self.state.step));
                    self.save(&checkpoint_path)?;
                    println!("  Saved checkpoint: {:?}", checkpoint_path);

                    // Clean up old checkpoints
                    self.cleanup_old_checkpoints()?;
                }

                if self.state.step >= self.config.max_steps {
                    break;
                }
            }
        }

        // Save final checkpoint
        let final_path = self.config.checkpoint_dir.join("final.safetensors");
        self.save(&final_path)?;
        println!("\nTraining complete! Final model saved to {:?}", final_path);

        self.state.total_time_secs = start_time.elapsed().as_secs_f64();
        Ok(())
    }

    /// Save model checkpoint
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let model = self.model.as_ref()
            .ok_or_else(|| Error::Message("Model not loaded".to_string()))?;

        model.save_weights(path)?;
        Ok(())
    }

    /// Load model checkpoint
    pub fn load(&mut self, path: impl AsRef<Path>) -> Result<(), Error> {
        let model = self.model.as_mut()
            .ok_or_else(|| Error::Message("Model not initialized".to_string()))?;

        model.load_weights(path)?;
        Ok(())
    }

    /// Clean up old checkpoints, keeping only the most recent N
    fn cleanup_old_checkpoints(&self) -> Result<(), Error> {
        if self.config.keep_last_n_checkpoints == 0 {
            return Ok(()); // Keep all checkpoints
        }

        let mut checkpoints: Vec<_> = std::fs::read_dir(&self.config.checkpoint_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("checkpoint-") && n.ends_with(".safetensors"))
                    .unwrap_or(false)
            })
            .collect();

        // Sort by modification time (newest first)
        checkpoints.sort_by(|a, b| {
            b.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                .cmp(
                    &a.metadata()
                        .and_then(|m| m.modified())
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                )
        });

        // Remove old checkpoints
        for checkpoint in checkpoints.iter().skip(self.config.keep_last_n_checkpoints) {
            std::fs::remove_file(checkpoint.path())?;
        }

        Ok(())
    }

    /// Get current training state
    pub fn state(&self) -> &TrainingState {
        &self.state
    }

    /// Get current learning rate
    pub fn current_lr(&self) -> f32 {
        self.scheduler.get_lr()
    }
}

/// Compute cross-entropy loss with padding mask (inner version for value_and_grad)
fn compute_cross_entropy_loss_inner(
    logits: &Array,
    targets: &Array,
    _target_lens: &Array,
) -> Result<Array, mlx_rs::error::Exception> {
    use mlx_rs::nn::log_softmax;
    use mlx_rs::ops::mean;

    // logits: [batch, seq_len, vocab_size]
    // targets: [batch, seq_len]
    // target_lens: [batch]

    let batch_size = logits.shape()[0];
    let seq_len = logits.shape()[1];

    // Apply log_softmax along vocab dimension
    let log_probs = log_softmax(logits, Some(-1))?;

    // Use take_along_axis to gather log probabilities for target tokens
    // Expand targets to [batch, seq_len, 1] for take_along_axis
    let targets_expanded = targets.reshape(&[batch_size, seq_len, 1])?;

    // Gather: select log_probs[b, t, targets[b, t]] for each position
    let selected_log_probs = log_probs.take_along_axis(&targets_expanded, -1)?;

    // Shape: [batch, seq_len, 1] -> [batch, seq_len]
    let selected_log_probs = selected_log_probs.reshape(&[batch_size, seq_len])?;

    // Compute negative log likelihood
    let nll = selected_log_probs.negative()?;

    // Create padding mask from target lengths
    // For now, just take mean over all positions
    // TODO: Proper masking based on target_lens
    let loss = mean(&nll, false)?;

    Ok(loss)
}

/// Compute cross-entropy loss with padding mask (public version)
fn compute_cross_entropy_loss(
    logits: &Array,
    targets: &Array,
    target_lens: &Array,
) -> Result<Array, Error> {
    compute_cross_entropy_loss_inner(logits, targets, target_lens)
        .map_err(|e| Error::Message(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let config = TrainingConfig::default();
        let trainer = T2STrainer::new(config);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_training_state_default() {
        let state = TrainingState::default();
        assert_eq!(state.step, 0);
        assert_eq!(state.best_loss, f32::MAX);
    }
}
