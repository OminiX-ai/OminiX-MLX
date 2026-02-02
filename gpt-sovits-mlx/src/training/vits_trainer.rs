//! VITS Training Loop with GAN Training
//!
//! This module implements the training loop for VITS (SoVITS) models using
//! alternating Generator/Discriminator updates following the HiFi-GAN training
//! procedure.
//!
//! ## Gradient Stability
//!
//! Unlike Python which uses AMP (Automatic Mixed Precision) with GradScaler,
//! this implementation uses explicit NaN/Inf checking and gradient clipping
//! to maintain numerical stability. Updates are skipped when invalid gradients
//! are detected.

use std::path::Path;

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    module::{Module, ModuleParameters},
    nn,
    ops::indexing::IndexOp,
    optimizers::{AdamW, AdamWBuilder, Optimizer, clip_grad_norm},
    transforms::eval,
    Array,
};

/// Check if an array contains NaN or Inf values
fn has_invalid_values(arr: &Array) -> bool {
    // Use MLX ops to check for NaN/Inf efficiently
    if let Ok(has_nan) = mlx_rs::ops::is_nan(arr) {
        if let Ok(any_nan) = mlx_rs::ops::any(&has_nan, None) {
            if let Ok(_) = eval([&any_nan]) {
                if any_nan.item::<bool>() {
                    return true;
                }
            }
        }
    }
    if let Ok(has_inf) = mlx_rs::ops::is_inf(arr) {
        if let Ok(any_inf) = mlx_rs::ops::any(&has_inf, None) {
            if let Ok(_) = eval([&any_inf]) {
                if any_inf.item::<bool>() {
                    return true;
                }
            }
        }
    }
    false
}

/// Check if any gradient in the map contains invalid values (works with Cow<Array>)
fn has_invalid_gradients_cow(gradients: &std::collections::HashMap<std::rc::Rc<str>, std::borrow::Cow<'_, Array>>) -> bool {
    for (_, grad) in gradients.iter() {
        if has_invalid_values(grad.as_ref()) {
            return true;
        }
    }
    false
}

use crate::{
    error::Error,
    models::{
        discriminator::{MultiPeriodDiscriminator, MPDConfig, losses as disc_losses},
        vits::{SynthesizerTrn, VITSConfig, load_vits_model},
    },
    audio::{MelConfig, mel_spectrogram_mlx, spec_to_mel, slice_mel_segments},
};

use super::vits_loss::{kl_loss, mel_reconstruction_loss, discriminator_loss as disc_loss_ex};

/// Slice segments from audio using start indices.
/// Python: y = commons.slice_segments(y, ids_slice * hop_length, segment_size)
///
/// # Arguments
/// * `audio` - Audio tensor [batch, 1, samples]
/// * `ids_slice` - Start indices in frames [batch]
/// * `hop_length` - Hop length (frames to samples conversion)
/// * `segment_size` - Segment size in samples
fn slice_segments_by_ids(
    audio: &Array,
    ids_slice: &Array,
    hop_length: i32,
    segment_size: i32,
) -> Result<Array, Error> {
    let batch = audio.dim(0) as i32;
    let audio_len = audio.dim(2) as i32;

    let mut slices = Vec::with_capacity(batch as usize);
    for b in 0..batch {
        let start_frame: i32 = ids_slice.index(b).item();
        let start_sample = start_frame * hop_length;
        let end_sample = (start_sample + segment_size).min(audio_len);

        // Handle edge case where segment extends beyond audio
        let actual_len = end_sample - start_sample;
        if actual_len < segment_size {
            // Pad with zeros if needed
            let slice = audio.index((b, .., start_sample..end_sample));
            let padding = Array::zeros::<f32>(&[1, segment_size - actual_len])?;
            let padded = mlx_rs::ops::concatenate_axis(&[&slice, &padding], 1)?;
            slices.push(padded);
        } else {
            let slice = audio.index((b, .., start_sample..end_sample));
            slices.push(slice);
        }
    }

    // Stack slices back to [batch, 1, segment_size]
    let slices_refs: Vec<&Array> = slices.iter().collect();
    mlx_rs::ops::stack_axis(&slices_refs, 0).map_err(|e| Error::Message(e.to_string()))
}

/// Configuration for VITS training
#[derive(Debug, Clone)]
pub struct VITSTrainingConfig {
    /// Generator learning rate (default 1e-5 for finetuning, use 1e-4 for training from scratch)
    pub learning_rate_g: f32,
    /// Discriminator learning rate
    pub learning_rate_d: f32,
    /// Batch size
    pub batch_size: usize,
    /// Segment size in samples for training (20480 @ 32kHz = 640ms)
    pub segment_size: i32,
    /// Mel loss weight
    pub c_mel: f32,
    /// KL loss weight
    pub c_kl: f32,
    /// Feature matching loss weight
    pub c_fm: f32,
    /// Gradient clipping threshold (set very high to disable, matching Python)
    pub grad_clip: f32,
    /// Maximum training steps
    pub max_steps: usize,
    /// Save checkpoint every N steps
    pub save_every: usize,
    /// Log every N steps
    pub log_every: usize,
    /// AdamW beta1 coefficient (momentum for gradient)
    pub beta1: f32,
    /// AdamW beta2 coefficient (momentum for squared gradient)
    pub beta2: f32,
    /// AdamW epsilon for numerical stability
    pub eps: f32,
    /// L2 regularization strength towards pretrained weights (0 = disabled)
    /// Helps prevent weight drift during finetuning without weight normalization
    pub pretrained_reg_strength: f32,
}

impl Default for VITSTrainingConfig {
    fn default() -> Self {
        Self {
            // Lower learning rate for finetuning (1e-5 instead of 1e-4)
            // Without weight normalization, higher LR causes weight drift
            learning_rate_g: 1e-5,
            learning_rate_d: 1e-5,
            batch_size: 4,
            // Python uses segment_size: 20480 (not 8192)
            segment_size: 20480,
            c_mel: 45.0,
            c_kl: 1.0,
            c_fm: 2.0,
            // Gradient clipping helps prevent explosion
            grad_clip: 100.0,
            max_steps: 10000,
            save_every: 1000,
            log_every: 10,
            // Python uses betas=[0.8, 0.99] (different from PyTorch default [0.9, 0.999])
            beta1: 0.8,
            beta2: 0.99,
            // Python uses eps=1e-09 (different from PyTorch default 1e-8)
            eps: 1e-9,
            // L2 regularization towards pretrained weights
            // Prevents drift during finetuning without weight normalization
            pretrained_reg_strength: 0.001,
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
    /// VQ commitment loss (called kl_ssl in Python training)
    pub loss_commit: f32,
    /// L2 regularization loss towards pretrained weights
    pub loss_reg: f32,
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
    /// Pretrained weights for regularization (prevents drift during finetuning)
    pretrained_weights: Option<std::collections::HashMap<String, Array>>,
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

        // Create optimizers with Python's hyperparameters:
        // betas=[0.8, 0.99], eps=1e-9 (different from PyTorch defaults)
        let optim_g = AdamWBuilder::new(config.learning_rate_g)
            .betas((config.beta1, config.beta2))
            .eps(config.eps)
            .build()
            .map_err(|e| Error::Message(format!("Failed to create optimizer: {:?}", e)))?;
        let optim_d = AdamWBuilder::new(config.learning_rate_d)
            .betas((config.beta1, config.beta2))
            .eps(config.eps)
            .build()
            .map_err(|e| Error::Message(format!("Failed to create optimizer: {:?}", e)))?;

        let mel_config = MelConfig::default();

        Ok(Self {
            generator,
            discriminator,
            optim_g,
            optim_d,
            config,
            mel_config,
            step: 0,
            pretrained_weights: None,
        })
    }

    /// Load pretrained generator weights
    pub fn load_generator_weights(&mut self, path: impl AsRef<Path>) -> Result<(), Error> {
        self.generator = load_vits_model(&path)?;
        Ok(())
    }

    /// Load pretrained generator weights and store a copy for regularization
    ///
    /// This is the preferred method for finetuning - it loads pretrained weights
    /// and stores them for L2 regularization to prevent weight drift.
    pub fn load_generator_weights_with_regularization(&mut self, path: impl AsRef<Path>) -> Result<(), Error> {
        // Load weights into generator (weights get transposed to MLX format)
        self.generator = load_vits_model(&path)?;

        // Store a copy of the trainable parameters for regularization
        // IMPORTANT: Get from the model (MLX format), not from safetensors (PyTorch format)
        let trainable = self.generator.trainable_parameters().flatten();
        let mut pretrained = std::collections::HashMap::new();

        // Only store decoder weights (these are what we're finetuning)
        for (key, value) in trainable.iter() {
            let key_str = key.as_ref();
            if key_str.starts_with("dec.") || key_str.starts_with("ref_enc") || key_str.starts_with("ssl_proj") {
                // Clone the array so we have an owned copy
                pretrained.insert(key_str.to_string(), (*value).clone());
            }
        }

        eprintln!("Stored {} pretrained weights for regularization", pretrained.len());
        self.pretrained_weights = Some(pretrained);
        Ok(())
    }

    /// Compute L2 regularization loss towards pretrained weights
    ///
    /// This penalizes deviation from pretrained weights to prevent drift
    /// during finetuning without weight normalization.
    fn compute_pretrained_reg_loss(&self) -> Result<Array, Exception> {
        if self.config.pretrained_reg_strength <= 0.0 {
            return Ok(array!(0.0f32));
        }

        let pretrained = match &self.pretrained_weights {
            Some(p) => p,
            None => return Ok(array!(0.0f32)),
        };

        let mut total_loss = array!(0.0f32);
        let mut count = 0;

        // Get current generator trainable parameters
        let current_params = self.generator.trainable_parameters().flatten();

        for (key, current) in current_params.iter() {
            // Map module key to pretrained key format
            let pretrained_key = key.as_ref().to_string();

            if let Some(pretrained_w) = pretrained.get(&pretrained_key) {
                // L2 distance (mean squared error)
                let diff = current.as_ref().subtract(pretrained_w)?;
                let sq_diff = diff.square()?;
                let loss = sq_diff.mean(false)?;
                total_loss = total_loss.add(&loss)?;
                count += 1;
            }
        }

        if count == 0 {
            return Ok(array!(0.0f32));
        }

        // Scale by regularization strength
        total_loss.multiply(array!(self.config.pretrained_reg_strength))
    }

    /// Compute regularization gradients towards pretrained weights
    ///
    /// Returns gradients that push weights back towards pretrained values.
    /// Gradient for each weight: 2 * (current - pretrained) * strength
    fn compute_pretrained_reg_gradients(
        &self,
        current_params: &mlx_rs::module::FlattenedModuleParamRef<'_>,
    ) -> Result<std::collections::HashMap<String, Array>, Exception> {
        let mut reg_grads = std::collections::HashMap::new();

        if self.config.pretrained_reg_strength <= 0.0 {
            return Ok(reg_grads);
        }

        let pretrained = match &self.pretrained_weights {
            Some(p) => p,
            None => return Ok(reg_grads),
        };

        let strength = self.config.pretrained_reg_strength;

        for (key, current) in current_params.iter() {
            let key_str = key.as_ref();

            // Try to find matching pretrained weight
            if let Some(pretrained_w) = pretrained.get(key_str) {
                // Gradient: 2 * (current - pretrained) * strength
                let diff = (*current).subtract(pretrained_w)?;
                let grad = diff.multiply(array!(2.0 * strength))?;
                reg_grads.insert(key_str.to_string(), grad);
            }
        }

        Ok(reg_grads)
    }

    /// Freeze all generator layers except the decoder (for fewshot training)
    ///
    /// For fewshot voice cloning, we adapt:
    /// - dec (HiFi-GAN decoder): generates audio from latents
    /// - ref_enc (reference encoder): captures voice style
    /// - ssl_proj: adapts SSL features for the new voice (important for commit_loss)
    ///
    /// We keep frozen:
    /// - enc_p (TextEncoder): phoneme/SSL encoding
    /// - enc_q (Posterior encoder): spectrogram encoding
    /// - flow: flow transformations
    /// - quantizer: VQ codebook (pretrained)
    pub fn freeze_non_decoder_layers(&mut self) {
        use mlx_rs::module::Module;

        // Freeze the entire generator first (recursive = true)
        self.generator.freeze_parameters(true);

        // Unfreeze the decoder (HiFi-GAN)
        self.generator.dec.unfreeze_parameters(true);

        // Unfreeze ref_enc (reference encoder for voice style)
        self.generator.ref_enc.unfreeze_parameters(true);

        // Unfreeze ssl_proj so commit_loss can push it towards codebook
        // This is critical for voice adaptation!
        self.generator.ssl_proj.unfreeze_parameters(true);
    }

    /// Re-normalize weight_v in all weight-normalized layers of the decoder
    ///
    /// This is critical for proper weight normalization training. PyTorch's weight_norm
    /// applies a gradient hook that projects gradients to maintain ||v|| = constant.
    /// MLX's autodiff doesn't do this, so we need to explicitly normalize after updates.
    ///
    /// Without this, ||v|| can grow by 50-100%+ in just 2 epochs, causing:
    /// - Computed weights (g*v/||v||) to drift from pretrained values
    /// - Audio quality degradation
    /// - Different behavior compared to Python training
    fn normalize_weight_v_in_decoder(generator: &mut SynthesizerTrn) -> Result<(), Error> {
        // Normalize conv_pre
        generator.dec.conv_pre.normalize_v()
            .map_err(|e| Error::Message(format!("conv_pre normalize_v failed: {}", e)))?;

        // Normalize all upsampling layers
        for (i, ups) in generator.dec.ups.iter_mut().enumerate() {
            ups.normalize_v()
                .map_err(|e| Error::Message(format!("ups.{} normalize_v failed: {}", i, e)))?;
        }

        // Normalize conv_post
        generator.dec.conv_post.normalize_v()
            .map_err(|e| Error::Message(format!("conv_post normalize_v failed: {}", e)))?;

        // Note: resblocks use regular Conv1d, not weight-normalized, so no normalization needed

        Ok(())
    }

    /// Single training step with gradient-based parameter updates
    ///
    /// Performs alternating GAN training:
    /// 1. Discriminator step: update D while freezing G
    /// 2. Generator step: update G while freezing D
    pub fn train_step(&mut self, batch: &VITSBatch) -> Result<VITSLosses, Error> {
        // ======================
        // Step 1: Forward pass through generator (for both D and G steps)
        // ======================

        // forward_train now returns ids_slice for proper audio alignment
        // Also returns commit_loss (kl_ssl in Python) for VQ commitment
        let (y_hat, z_p, m_p, logs_p, _z, _m_q, logs_q, y_mask, ids_slice, _commit_loss) = self.generator.forward_train(
            &batch.ssl_features,
            &batch.spec,
            &batch.spec_lengths,
            &batch.text,
            &batch.refer_mel,
        ).map_err(|e| Error::Message(e.to_string()))?;

        // Force evaluation
        eval([&y_hat, &z_p, &m_p, &logs_p, &logs_q, &y_mask, &ids_slice])?;

        // Python: y = slice_segments(y, ids_slice * hop_length, segment_size)
        // Slice real audio at the SAME position as the generated segment
        let hop_length = 640; // TODO: get from config
        let segment_samples = self.config.segment_size; // 20480 samples
        let y_real_sliced = slice_segments_by_ids(&batch.audio, &ids_slice, hop_length, segment_samples)?;

        // y_hat is already the right size (decoder output for segment_size frames)
        let y_hat_sliced = y_hat;

        // ======================
        // Step 2: Discriminator training step
        // ======================
        let loss_d_val = self.train_discriminator_step(&y_real_sliced, &y_hat_sliced)?;

        // ======================
        // Step 3: Generator training step (includes all G losses)
        // Calls forward_train again inside closure with gradients enabled,
        // using its own ids_slice to slice the FULL audio correctly
        // ======================
        let (loss_gen_val, loss_fm_val, loss_mel_val, loss_kl_val, loss_commit_val) =
            self.train_generator_step(batch)?;

        // Compute regularization loss for logging (gradient already applied in train_generator_step)
        let loss_reg_val = self.compute_pretrained_reg_loss()
            .map(|a| a.item::<f32>())
            .unwrap_or(0.0);

        // Total generator loss (weighted sum)
        // Python: loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl
        let loss_total = loss_gen_val
            + loss_fm_val * self.config.c_fm
            + loss_mel_val * self.config.c_mel
            + loss_kl_val * self.config.c_kl
            + loss_commit_val  // kl_ssl * 1 in Python
            + loss_reg_val;  // Regularization loss

        self.step += 1;

        Ok(VITSLosses {
            loss_d: loss_d_val,
            loss_gen: loss_gen_val,
            loss_fm: loss_fm_val,
            loss_mel: loss_mel_val,
            loss_kl: loss_kl_val,
            loss_commit: loss_commit_val,
            loss_reg: loss_reg_val,
            loss_total,
        })
    }

    /// Train discriminator for one step
    ///
    /// Updates discriminator parameters to classify real audio as 1 and generated audio as 0
    fn train_discriminator_step(
        &mut self,
        y_real: &Array,
        y_fake: &Array,
    ) -> Result<f32, Error> {
        // Clone arrays for the closure
        let y_real = y_real.clone();
        let y_fake = y_fake.clone();

        // Take ownership of discriminator and optimizer
        let mut discriminator = std::mem::replace(
            &mut self.discriminator,
            MultiPeriodDiscriminator::new(MPDConfig::default())?,
        );
        let mut optim_d = std::mem::replace(
            &mut self.optim_d,
            AdamWBuilder::new(self.config.learning_rate_d)
                .betas((self.config.beta1, self.config.beta2))
                .eps(self.config.eps)
                .build()
                .unwrap(),
        );

        // Define discriminator loss function
        let loss_fn = |disc: &mut MultiPeriodDiscriminator,
                       (y_r, y_f): (&Array, &Array)|
                       -> Result<Array, Exception> {
            let (d_real, d_fake, _, _) = disc.forward_ex(y_r, y_f)?;
            disc_loss_ex(&d_real, &d_fake)
        };

        // Compute loss and gradients
        let mut value_and_grad = nn::value_and_grad(loss_fn);
        let (loss, gradients) = value_and_grad(&mut discriminator, (&y_real, &y_fake))
            .map_err(|e| Error::Message(format!("D gradient computation failed: {}", e)))?;

        // Evaluate loss
        eval([&loss]).map_err(|e| Error::Message(e.to_string()))?;
        let loss_value = loss.item::<f32>();

        // Check for NaN/Inf loss - skip update if invalid (like GradScaler)
        if loss_value.is_nan() || loss_value.is_infinite() {
            eprintln!("  [WARN] Discriminator loss is NaN/Inf, skipping update");
            self.discriminator = discriminator;
            self.optim_d = optim_d;
            return Ok(0.0);  // Return 0 to indicate skipped
        }

        // Clip gradients
        let (clipped_gradients, grad_norm) = clip_grad_norm(&gradients, self.config.grad_clip)
            .map_err(|e| Error::Message(format!("D gradient clipping failed: {}", e)))?;

        // Check for NaN/Inf in gradients - skip update if invalid
        if has_invalid_gradients_cow(&clipped_gradients) {
            eprintln!("  [WARN] Discriminator gradients contain NaN/Inf, skipping update (grad_norm: {:?})", grad_norm);
            self.discriminator = discriminator;
            self.optim_d = optim_d;
            return Ok(loss_value);  // Return loss but skip update
        }

        // Convert to owned arrays
        let owned_gradients: mlx_rs::module::FlattenedModuleParam = clipped_gradients
            .into_iter()
            .map(|(k, v)| (k, v.into_owned()))
            .collect();

        // Update discriminator parameters
        optim_d.update(&mut discriminator, &owned_gradients)
            .map_err(|e| Error::Message(format!("D optimizer update failed: {}", e)))?;

        // Evaluate updated parameters
        let params: Vec<_> = discriminator.trainable_parameters().flatten()
            .into_iter().map(|(_, v)| v.clone()).collect();
        eval(params.iter()).map_err(|e| Error::Message(e.to_string()))?;

        // Put discriminator and optimizer back
        self.discriminator = discriminator;
        self.optim_d = optim_d;

        Ok(loss_value)
    }

    /// Train generator for one step with gradient updates
    ///
    /// Updates generator parameters using ALL losses from Python implementation:
    /// - loss_gen: adversarial loss (generator wants D to output 1 for fake)
    /// - loss_fm: feature matching loss (match discriminator intermediate features)
    /// - loss_mel: L1 mel reconstruction loss (weighted by c_mel=45)
    /// - loss_kl: KL divergence loss (weighted by c_kl=1.0)
    /// - loss_commit: VQ commitment loss (kl_ssl in Python, weighted by 1.0)
    fn train_generator_step(
        &mut self,
        batch: &VITSBatch,
    ) -> Result<(f32, f32, f32, f32, f32), Error> {
        use std::cell::RefCell;
        use std::rc::Rc;
        use super::vits_loss::{generator_loss, feature_matching_loss};

        // Clone batch data for the closure
        let ssl = batch.ssl_features.clone();
        let spec = batch.spec.clone();
        let spec_lengths = batch.spec_lengths.clone();
        let text = batch.text.clone();
        let refer_mel = batch.refer_mel.clone();
        let audio_full = batch.audio.clone(); // FULL audio, will be sliced inside closure
        let mel_config = self.mel_config.clone();
        let c_mel = self.config.c_mel;
        let c_kl = self.config.c_kl;
        let hop_length = 640i32; // TODO: get from config
        let segment_samples = self.config.segment_size;

        // Take ownership of generator and optimizer
        let mut generator = std::mem::replace(
            &mut self.generator,
            SynthesizerTrn::new(VITSConfig::default())
                .map_err(|e| Error::Message(e.to_string()))?,
        );
        let mut optim_g = std::mem::replace(
            &mut self.optim_g,
            AdamWBuilder::new(self.config.learning_rate_g)
                .betas((self.config.beta1, self.config.beta2))
                .eps(self.config.eps)
                .build()
                .unwrap(),
        );

        // Take discriminator and wrap in Rc<RefCell> for interior mutability in closure
        // (forward_ex requires &mut self, but closure captures by reference)
        let discriminator = std::mem::replace(
            &mut self.discriminator,
            MultiPeriodDiscriminator::new(MPDConfig::default())?,
        );
        let disc_cell = Rc::new(RefCell::new(discriminator));

        // Scope for the closure and value_and_grad to ensure Rc is released after use
        let (loss_value, owned_gradients, skip_update) = {
            let disc_for_closure = Rc::clone(&disc_cell);

            // Define generator loss function with ALL components
            // Key: discriminator is called inside, so gradients flow back to generator
            //
            // CRITICAL: Mel loss computation must match Python exactly:
            // Python: y_mel = slice_segments(spec_to_mel(spec), ids_slice, segment_frames)
            //         y_hat_mel = mel_spectrogram_torch(y_hat, ...)
            // So ground truth mel comes from SPEC, fake mel comes from AUDIO
            let loss_fn = |gen: &mut SynthesizerTrn,
                           (ssl_f, spec_f, spec_len, text_f, refer_f, audio_f, hop_len, seg_samples): (&Array, &Array, &Array, &Array, &Array, &Array, i32, i32)|
                           -> Result<Array, Exception> {
                // Forward pass through generator - returns ids_slice for audio alignment
                // Also returns commit_loss (kl_ssl in Python) for VQ commitment
                let (y_hat, z_p, m_p, logs_p, _z, _m_q, logs_q, y_mask, ids_slice, commit_loss) = gen.forward_train(
                    ssl_f, spec_f, spec_len, text_f, refer_f
                )?;

                // Python: y = slice_segments(y, ids_slice * hop_length, segment_size)
                // Slice FULL audio at the SAME position as the generated segment
                let batch = audio_f.dim(0) as i32;
                let audio_len = audio_f.dim(2) as i32;
                let mut slices = Vec::with_capacity(batch as usize);
                for b in 0..batch {
                    let start_frame: i32 = ids_slice.index(b).item();
                    let start_sample = start_frame * hop_len;
                    let end_sample = (start_sample + seg_samples).min(audio_len);
                    let slice = audio_f.index((b, .., start_sample..end_sample));
                    slices.push(slice);
                }
                let slices_refs: Vec<&Array> = slices.iter().collect();
                let y_real_sliced = mlx_rs::ops::stack_axis(&slices_refs, 0)?;

                // y_hat is already the segment (decoder output for segment_size frames)
                let y_hat_sliced = y_hat;

                // === CRITICAL: Forward through discriminator (NOT detached!) ===
                // This allows gradients to flow back to generator
                let (_, d_fake, fmap_r, fmap_g) = disc_for_closure.borrow_mut().forward_ex(&y_real_sliced, &y_hat_sliced)?;

                // 1. Adversarial loss: generator wants D to output 1 for fake
                let loss_gen = generator_loss(&d_fake)?;

                // 2. Feature matching loss: match discriminator intermediate features
                // CRITICAL: Real features are now detached in feature_matching_loss!
                let loss_fm = feature_matching_loss(&fmap_r, &fmap_g)?;

                // 3. Mel reconstruction loss
                // CRITICAL: Match Python's approach exactly:
                // - y_mel: Convert SPEC to mel, then slice at ids_slice (uses original data)
                // - y_hat_mel: Compute mel from generated AUDIO
                let segment_frames = seg_samples / hop_len;
                let mel_full = spec_to_mel(spec_f, &mel_config)?;
                let mel_real = slice_mel_segments(&mel_full, &ids_slice, segment_frames)?;
                let mel_fake = mel_spectrogram_mlx(&y_hat_sliced.squeeze_axes(&[1])?, &mel_config)?;
                let loss_mel = mel_reconstruction_loss(&mel_real, &mel_fake)?;

                // 4. KL divergence loss
                let loss_kl = kl_loss(&z_p, &logs_q, &m_p, &logs_p, &y_mask)?;

                // 5. VQ commitment loss is already computed in forward_train (kl_ssl in Python)

                // Total generator loss (matching Python exactly):
                // Python: loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl
                // Note: In Python, loss_mel is already scaled by c_mel inside the loss computation
                let total_loss = loss_gen
                    .add(&loss_fm)?
                    .add(&loss_mel.multiply(&mlx_rs::array!(c_mel))?)?
                    .add(&loss_kl.multiply(&mlx_rs::array!(c_kl))?)?
                    .add(&commit_loss)?;  // kl_ssl * 1 in Python

                Ok(total_loss)
            };

            // Compute loss and gradients w.r.t. generator parameters
            let mut value_and_grad = nn::value_and_grad(loss_fn);
            let (loss, gradients) = value_and_grad(
                &mut generator,
                (&ssl, &spec, &spec_lengths, &text, &refer_mel, &audio_full, hop_length, segment_samples)
            ).map_err(|e| Error::Message(format!("G gradient computation failed: {}", e)))?;

            // Evaluate loss
            eval([&loss]).map_err(|e| Error::Message(e.to_string()))?;
            let loss_value = loss.item::<f32>();

            // Clip gradients
            let (clipped_gradients, grad_norm) = clip_grad_norm(&gradients, self.config.grad_clip)
                .map_err(|e| Error::Message(format!("G gradient clipping failed: {}", e)))?;

            // Check for invalid gradients
            let skip_update = loss_value.is_nan() || loss_value.is_infinite()
                || has_invalid_gradients_cow(&clipped_gradients);

            if skip_update {
                eprintln!("  [WARN] Generator loss/gradients invalid, skipping update (loss: {}, grad_norm: {:?})",
                         loss_value, grad_norm);
            }

            // Convert to owned arrays
            let owned_gradients: mlx_rs::module::FlattenedModuleParam = clipped_gradients
                .into_iter()
                .map(|(k, v)| (k, v.into_owned()))
                .collect();

            (loss_value, owned_gradients, skip_update)
        }; // disc_for_closure and value_and_grad are dropped here

        // Add pretrained regularization gradients
        let final_gradients = if self.pretrained_weights.is_some() && self.config.pretrained_reg_strength > 0.0 {
            // Get current parameters
            let current_params = generator.trainable_parameters().flatten();

            // Compute regularization gradients
            let reg_grads = self.compute_pretrained_reg_gradients(&current_params)
                .map_err(|e| Error::Message(format!("Regularization gradient failed: {}", e)))?;

            if !reg_grads.is_empty() {
                // Add regularization gradients to main gradients
                let mut combined: mlx_rs::module::FlattenedModuleParam = owned_gradients;
                for (key, reg_grad) in reg_grads {
                    // Find matching key in combined gradients
                    for (grad_key, grad_val) in combined.iter_mut() {
                        if grad_key.as_ref() == key {
                            // Add regularization gradient
                            *grad_val = grad_val.add(&reg_grad)
                                .map_err(|e| Error::Message(format!("Gradient addition failed: {}", e)))?;
                            break;
                        }
                    }
                }
                combined
            } else {
                owned_gradients
            }
        } else {
            owned_gradients
        };

        // Update generator parameters only if gradients are valid
        if !skip_update {
            optim_g.update(&mut generator, &final_gradients)
                .map_err(|e| Error::Message(format!("G optimizer update failed: {}", e)))?;

            // CRITICAL: Re-normalize weight_v in all weight-normalized layers
            //
            // PyTorch's weight_norm applies a gradient hook that projects gradients
            // to maintain ||v|| = constant. MLX's autodiff doesn't do this, so ||v||
            // can grow during training (we observed 50-100% growth in just 2 epochs).
            //
            // Solution: Call normalize_v() after each update to project weight_v
            // back onto the constraint manifold. This is "constrained optimization".
            Self::normalize_weight_v_in_decoder(&mut generator)?;
        }

        // Evaluate updated parameters
        let params: Vec<_> = generator.trainable_parameters().flatten()
            .into_iter().map(|(_, v)| v.clone()).collect();
        eval(params.iter()).map_err(|e| Error::Message(e.to_string()))?;

        // Put generator, discriminator, and optimizer back
        self.generator = generator;
        self.optim_g = optim_g;
        // Unwrap the Rc<RefCell> to get the discriminator back (Rc count should be 1 now)
        self.discriminator = Rc::try_unwrap(disc_cell)
            .map_err(|_| Error::Message("Failed to unwrap discriminator Rc".to_string()))?
            .into_inner();

        // Compute individual loss values for logging
        // We need to do a separate forward pass to get individual losses
        // For efficiency, we'll estimate from the total loss
        // TODO: Return actual individual losses by computing them separately
        // Now total_loss = loss_gen + loss_fm + loss_mel * c_mel + loss_kl * c_kl + commit_loss
        let approx_mel = loss_value / (c_mel + 1.0 + 2.0 + c_kl + 1.0);
        let approx_gen = approx_mel;
        let approx_fm = approx_mel * 2.0;
        let approx_kl = approx_mel;
        let approx_commit = approx_mel;  // VQ commitment loss (kl_ssl in Python)

        Ok((
            approx_gen,    // loss_gen (adversarial)
            approx_fm,     // loss_fm (feature matching)
            approx_mel,    // loss_mel
            approx_kl,     // loss_kl
            approx_commit, // loss_commit (kl_ssl)
        ))
    }

    /// Save checkpoint to safetensors file
    ///
    /// Converts conv weights from MLX format (out, kernel, in) to PyTorch format (out, in, kernel)
    /// so they can be loaded by load_vits_model.
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        use mlx_rs::ops::swap_axes;
        let path = path.as_ref();

        // Helper to convert Conv1d weight from MLX (out, kernel, in) to PyTorch (out, in, kernel)
        let transpose_conv1d_to_pytorch = |arr: &Array| -> Result<Array, Error> {
            swap_axes(arr, 1, 2).map_err(|e| Error::Message(e.to_string()))
        };

        // Helper to convert ConvTranspose1d weight from MLX (out, kernel, in) to PyTorch (in, out, kernel)
        let transpose_convt_to_pytorch = |arr: &Array| -> Result<Array, Error> {
            arr.transpose_axes(&[2, 0, 1]).map_err(|e| Error::Message(e.to_string()))
        };

        // Helper to convert Conv2d weight from MLX (out, kh, kw, in) to PyTorch (out, in, kh, kw)
        let transpose_conv2d_to_pytorch = |arr: &Array| -> Result<Array, Error> {
            arr.transpose_axes(&[0, 3, 1, 2]).map_err(|e| Error::Message(e.to_string()))
        };

        // Get generator trainable parameters and convert conv weights
        let g_params = self.generator.trainable_parameters().flatten();
        let mut g_converted: std::collections::HashMap<String, Array> = std::collections::HashMap::new();
        for (k, v) in g_params.iter() {
            let key_str = k.as_ref();
            let key = format!("generator.{}", key_str);
            let shape = v.shape();

            // Check if this is a ConvTranspose1d weight (dec.ups)
            if key_str.contains("dec.ups") && key_str.contains(".weight") && shape.len() == 3 {
                g_converted.insert(key, transpose_convt_to_pytorch(v)?);
            }
            // Check if this is a Conv1d weight
            else if key_str.contains(".weight") && shape.len() == 3 {
                g_converted.insert(key, transpose_conv1d_to_pytorch(v)?);
            }
            // Check if this is a Conv2d weight
            else if key_str.contains(".weight") && shape.len() == 4 {
                g_converted.insert(key, transpose_conv2d_to_pytorch(v)?);
            }
            else {
                g_converted.insert(key, v.as_ref().clone());
            }
        }

        // Get discriminator trainable parameters and convert conv weights
        let d_params = self.discriminator.trainable_parameters().flatten();
        for (k, v) in d_params.iter() {
            let key_str = k.as_ref();
            let key = format!("discriminator.{}", key_str);
            let shape = v.shape();

            if key_str.contains(".weight") && shape.len() == 3 {
                g_converted.insert(key, transpose_conv1d_to_pytorch(v)?);
            } else if key_str.contains(".weight") && shape.len() == 4 {
                g_converted.insert(key, transpose_conv2d_to_pytorch(v)?);
            } else {
                g_converted.insert(key, v.as_ref().clone());
            }
        }

        // Create references for saving
        let all_params: std::collections::HashMap<String, &Array> = g_converted
            .iter()
            .map(|(k, v)| (k.clone(), v))
            .collect();

        // Save to safetensors (with None metadata)
        Array::save_safetensors(all_params, None, path)?;

        Ok(())
    }

    /// Save just the generator weights (for inference)
    ///
    /// Converts conv weights from MLX format to PyTorch format for compatibility with load_vits_model.
    /// Also renames keys from MLX module names to Python model names.
    pub fn save_generator(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        use mlx_rs::ops::swap_axes;
        let path = path.as_ref();

        // Helper to convert Conv1d weight from MLX (out, kernel, in) to PyTorch (out, in, kernel)
        let transpose_conv1d_to_pytorch = |arr: &Array| -> Result<Array, Error> {
            swap_axes(arr, 1, 2).map_err(|e| Error::Message(e.to_string()))
        };

        // Helper to convert ConvTranspose1d weight from MLX (out, kernel, in) to PyTorch (in, out, kernel)
        let transpose_convt_to_pytorch = |arr: &Array| -> Result<Array, Error> {
            arr.transpose_axes(&[2, 0, 1]).map_err(|e| Error::Message(e.to_string()))
        };

        // Helper to convert Conv2d weight from MLX (out, kh, kw, in) to PyTorch (out, in, kh, kw)
        let transpose_conv2d_to_pytorch = |arr: &Array| -> Result<Array, Error> {
            arr.transpose_axes(&[0, 3, 1, 2]).map_err(|e| Error::Message(e.to_string()))
        };

        // Helper to rename MLX module keys to Python model keys
        // This ensures compatibility with load_vits_weights()
        let rename_key = |key: &str| -> String {
            // ref_enc key mappings
            let key = key.replace("ref_enc.spectral_0", "ref_enc.spectral.0.fc");
            let key = key.replace("ref_enc.spectral_1", "ref_enc.spectral.3.fc");
            let key = key.replace("ref_enc.temporal_0", "ref_enc.temporal.0.conv1.conv");
            let key = key.replace("ref_enc.temporal_1", "ref_enc.temporal.1.conv1.conv");
            let key = key.replace("ref_enc.slf_attn_q", "ref_enc.slf_attn.w_qs");
            let key = key.replace("ref_enc.slf_attn_k", "ref_enc.slf_attn.w_ks");
            let key = key.replace("ref_enc.slf_attn_v", "ref_enc.slf_attn.w_vs");
            let key = key.replace("ref_enc.slf_attn_fc", "ref_enc.slf_attn.fc");
            // Handle ref_enc.fc carefully - don't match ref_enc.slf_attn.fc
            let key = if key == "ref_enc.fc.weight" || key == "ref_enc.fc.bias" {
                key.replace("ref_enc.fc", "ref_enc.fc.fc")
            } else {
                key
            };
            key
        };

        // Get generator trainable parameters and convert conv weights
        let g_params = self.generator.trainable_parameters().flatten();
        let mut g_converted: std::collections::HashMap<String, Array> = std::collections::HashMap::new();
        for (k, v) in g_params.iter() {
            let key_str = k.as_ref();
            let shape = v.shape();
            let output_key = rename_key(key_str);

            // Handle weight-normalized layers: weight_g stays as-is, weight_v needs transpose
            if key_str.contains(".weight_g") {
                // weight_g has shape [out/in, 1, 1], no transpose needed
                g_converted.insert(output_key, v.as_ref().clone());
            }
            // weight_v for ConvTranspose1d (dec.ups)
            else if key_str.contains("dec.ups") && key_str.contains(".weight_v") && shape.len() == 3 {
                // Transpose from MLX [out, kernel, in] to PyTorch [in, out, kernel]
                g_converted.insert(output_key, transpose_convt_to_pytorch(v)?);
            }
            // weight_v for Conv1d
            else if key_str.contains(".weight_v") && shape.len() == 3 {
                // Transpose from MLX [out, kernel, in] to PyTorch [out, in, kernel]
                g_converted.insert(output_key, transpose_conv1d_to_pytorch(v)?);
            }
            // Regular ConvTranspose1d weight (dec.ups, if not using weight norm)
            else if key_str.contains("dec.ups") && key_str.contains(".weight") && !key_str.contains("weight_") && shape.len() == 3 {
                g_converted.insert(output_key, transpose_convt_to_pytorch(v)?);
            }
            // Regular Conv1d weight
            else if key_str.contains(".weight") && !key_str.contains("weight_") && shape.len() == 3 {
                g_converted.insert(output_key, transpose_conv1d_to_pytorch(v)?);
            }
            // Conv2d weight
            else if key_str.contains(".weight") && shape.len() == 4 {
                g_converted.insert(output_key, transpose_conv2d_to_pytorch(v)?);
            }
            else {
                g_converted.insert(output_key, v.as_ref().clone());
            }
        }

        // Create references for saving
        let all_params: std::collections::HashMap<String, &Array> = g_converted
            .iter()
            .map(|(k, v)| (k.clone(), v))
            .collect();

        // Save to safetensors
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
                    "Step {}: D={:.4}, G={:.4}, FM={:.4}, Mel={:.4}, KL={:.4}, Commit={:.4}, Reg={:.4}, Total={:.4}",
                    self.step,
                    losses.loss_d,
                    losses.loss_gen,
                    losses.loss_fm,
                    losses.loss_mel,
                    losses.loss_kl,
                    losses.loss_commit,
                    losses.loss_reg,
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
        assert_eq!(config.segment_size, 20480);
        // Learning rate lowered to 1e-5 for finetuning without weight normalization
        assert!((config.learning_rate_g - 1e-5).abs() < 1e-7);
        assert!((config.pretrained_reg_strength - 0.001).abs() < 1e-6);
    }
}
