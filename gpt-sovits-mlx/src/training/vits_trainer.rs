//! VITS Training Loop with GAN Training
//!
//! This module implements the training loop for VITS (SoVITS) models using
//! alternating Generator/Discriminator updates following the HiFi-GAN training
//! procedure.

use std::path::Path;

use mlx_rs::{
    module::ModuleParameters,
    ops::indexing::IndexOp,
    optimizers::AdamW,
    transforms::eval,
    Array,
};

use crate::{
    error::Error,
    models::{
        discriminator::{MultiPeriodDiscriminator, MPDConfig, losses as disc_losses},
        vits::{SynthesizerTrn, VITSConfig, load_vits_model},
    },
    audio::{MelConfig, mel_spectrogram_mlx},
};

use super::vits_loss::{kl_loss, mel_reconstruction_loss};

/// Configuration for VITS training
#[derive(Debug, Clone)]
pub struct VITSTrainingConfig {
    /// Generator learning rate
    pub learning_rate_g: f32,
    /// Discriminator learning rate
    pub learning_rate_d: f32,
    /// Batch size
    pub batch_size: usize,
    /// Segment size in samples for training
    pub segment_size: i32,
    /// Mel loss weight
    pub c_mel: f32,
    /// KL loss weight
    pub c_kl: f32,
    /// Feature matching loss weight
    pub c_fm: f32,
    /// Gradient clipping threshold
    pub grad_clip: f32,
    /// Maximum training steps
    pub max_steps: usize,
    /// Save checkpoint every N steps
    pub save_every: usize,
    /// Log every N steps
    pub log_every: usize,
}

impl Default for VITSTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate_g: 2e-4,
            learning_rate_d: 2e-4,
            batch_size: 4,
            segment_size: 8192,
            c_mel: 45.0,
            c_kl: 1.0,
            c_fm: 2.0,
            grad_clip: 5.0,
            max_steps: 10000,
            save_every: 1000,
            log_every: 10,
        }
    }
}

/// Loss values from a single training step
#[derive(Debug, Clone)]
pub struct VITSLosses {
    pub loss_d: f32,
    pub loss_gen: f32,
    pub loss_fm: f32,
    pub loss_mel: f32,
    pub loss_kl: f32,
    pub loss_total: f32,
}

/// Training batch for VITS
pub struct VITSBatch {
    /// SSL features from HuBERT [batch, ssl_dim, ssl_len]
    pub ssl_features: Array,
    /// Target spectrogram [batch, n_fft/2+1, spec_len]
    pub spec: Array,
    /// Spectrogram lengths [batch]
    pub spec_lengths: Array,
    /// Phoneme indices [batch, text_len]
    pub text: Array,
    /// Text lengths [batch]
    #[allow(dead_code)]
    pub text_lengths: Array,
    /// Target audio [batch, 1, samples]
    pub audio: Array,
    /// Reference mel spectrogram [batch, mel_channels, time]
    pub refer_mel: Array,
}

/// VITS Trainer with GAN training loop
pub struct VITSTrainer {
    /// Generator (SynthesizerTrn)
    pub generator: SynthesizerTrn,
    /// Discriminator (MultiPeriodDiscriminator)
    pub discriminator: MultiPeriodDiscriminator,
    /// Generator optimizer (for future use with gradient updates)
    #[allow(dead_code)]
    optim_g: AdamW,
    /// Discriminator optimizer (for future use with gradient updates)
    #[allow(dead_code)]
    optim_d: AdamW,
    /// Training configuration
    pub config: VITSTrainingConfig,
    /// Mel spectrogram configuration
    pub mel_config: MelConfig,
    /// Current training step
    pub step: usize,
}

impl VITSTrainer {
    /// Create a new VITS trainer
    pub fn new(config: VITSTrainingConfig) -> Result<Self, Error> {
        // Create generator with default config
        let vits_config = VITSConfig::default();
        let generator = SynthesizerTrn::new(vits_config)
            .map_err(|e| Error::Message(e.to_string()))?;

        // Create discriminator
        let mpd_config = MPDConfig::default();
        let discriminator = MultiPeriodDiscriminator::new(mpd_config)?;

        // Create optimizers (for future gradient-based training)
        let optim_g = AdamW::new(config.learning_rate_g);
        let optim_d = AdamW::new(config.learning_rate_d);

        let mel_config = MelConfig::default();

        Ok(Self {
            generator,
            discriminator,
            optim_g,
            optim_d,
            config,
            mel_config,
            step: 0,
        })
    }

    /// Load pretrained generator weights
    pub fn load_generator_weights(&mut self, path: impl AsRef<Path>) -> Result<(), Error> {
        self.generator = load_vits_model(path)?;
        Ok(())
    }

    /// Single training step
    ///
    /// Performs forward pass and loss computation for both D and G.
    /// Currently computes losses without gradient updates.
    ///
    /// TODO: Add gradient-based parameter updates using value_and_grad
    pub fn train_step(&mut self, batch: &VITSBatch) -> Result<VITSLosses, Error> {
        // ======================
        // Step 1: Forward pass through generator
        // ======================

        let (y_hat, z_p, m_p, logs_p, _z, _m_q, logs_q, y_mask) = self.generator.forward_train(
            &batch.ssl_features,
            &batch.spec,
            &batch.spec_lengths,
            &batch.text,
            &batch.refer_mel,
        ).map_err(|e| Error::Message(e.to_string()))?;

        // Force evaluation
        eval([&y_hat, &z_p, &m_p, &logs_p, &logs_q, &y_mask])?;

        // Match audio lengths for discriminator
        let gen_len = y_hat.dim(2) as i32;
        let real_len = batch.audio.dim(2) as i32;
        let min_len = gen_len.min(real_len);

        let y_hat_sliced = y_hat.index((.., .., 0..min_len));
        let y_real_sliced = batch.audio.index((.., .., 0..min_len));

        // ======================
        // Step 2: Discriminator forward and loss
        // ======================

        let (d_real, d_fake, fmap_real, fmap_fake) =
            self.discriminator.forward(&y_real_sliced, &y_hat_sliced)?;

        // Discriminator loss: wants real=1, fake=0
        let loss_d = disc_losses::discriminator_loss(&d_real, &d_fake)?;

        // ======================
        // Step 3: Generator losses
        // ======================

        // Generator adversarial loss: wants fake=1
        let loss_gen = disc_losses::generator_loss(&d_fake)?;

        // Feature matching loss
        let loss_fm = disc_losses::feature_matching_loss(&fmap_real, &fmap_fake)?;

        // Mel spectrogram reconstruction loss
        let mel_real = mel_spectrogram_mlx(&y_real_sliced.squeeze_axes(&[1])?, &self.mel_config)
            .map_err(|e| Error::Message(e.to_string()))?;
        let mel_fake = mel_spectrogram_mlx(&y_hat_sliced.squeeze_axes(&[1])?, &self.mel_config)
            .map_err(|e| Error::Message(e.to_string()))?;

        let loss_mel = mel_reconstruction_loss(&mel_real, &mel_fake)
            .map_err(|e| Error::Message(e.to_string()))?;

        // KL divergence loss
        let loss_kl = kl_loss(&z_p, &logs_q, &m_p, &logs_p, &y_mask)
            .map_err(|e| Error::Message(e.to_string()))?;

        // Evaluate all losses
        eval([&loss_d, &loss_gen, &loss_fm, &loss_mel, &loss_kl])?;

        let loss_d_val: f32 = loss_d.item();
        let loss_gen_val: f32 = loss_gen.item();
        let loss_fm_val: f32 = loss_fm.item();
        let loss_mel_val: f32 = loss_mel.item();
        let loss_kl_val: f32 = loss_kl.item();

        // Total generator loss (weighted sum)
        let loss_total = loss_gen_val
            + loss_fm_val * self.config.c_fm
            + loss_mel_val * self.config.c_mel
            + loss_kl_val * self.config.c_kl;

        self.step += 1;

        Ok(VITSLosses {
            loss_d: loss_d_val,
            loss_gen: loss_gen_val,
            loss_fm: loss_fm_val,
            loss_mel: loss_mel_val,
            loss_kl: loss_kl_val,
            loss_total,
        })
    }

    /// Save checkpoint to safetensors file
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let path = path.as_ref();

        // Get generator trainable parameters
        let g_params = self.generator.trainable_parameters().flatten();
        let g_arrays: std::collections::HashMap<String, &Array> = g_params
            .iter()
            .map(|(k, v)| (format!("generator.{}", k), v.as_ref()))
            .collect();

        // Get discriminator trainable parameters
        let d_params = self.discriminator.trainable_parameters().flatten();
        let d_arrays: std::collections::HashMap<String, &Array> = d_params
            .iter()
            .map(|(k, v)| (format!("discriminator.{}", k), v.as_ref()))
            .collect();

        // Combine all parameters
        let mut all_params: std::collections::HashMap<String, &Array> = g_arrays;
        all_params.extend(d_arrays);

        // Save to safetensors (with None metadata)
        Array::save_safetensors(all_params, None, path)?;

        Ok(())
    }

    /// Training loop
    pub fn train(&mut self, batches: impl Iterator<Item = VITSBatch>) -> Result<(), Error> {
        for batch in batches {
            if self.step >= self.config.max_steps {
                break;
            }

            let losses = self.train_step(&batch)?;

            if self.step % self.config.log_every == 0 {
                println!(
                    "Step {}: D={:.4}, G={:.4}, FM={:.4}, Mel={:.4}, KL={:.4}, Total={:.4}",
                    self.step,
                    losses.loss_d,
                    losses.loss_gen,
                    losses.loss_fm,
                    losses.loss_mel,
                    losses.loss_kl,
                    losses.loss_total,
                );
            }

            if self.step % self.config.save_every == 0 && self.step > 0 {
                let ckpt_path = format!("checkpoint_{}.safetensors", self.step);
                self.save_checkpoint(&ckpt_path)?;
                println!("Saved checkpoint to {}", ckpt_path);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = VITSTrainingConfig::default();
        assert_eq!(config.batch_size, 4);
        assert_eq!(config.segment_size, 8192);
        assert!((config.learning_rate_g - 2e-4).abs() < 1e-6);
    }
}
