//! Qwen-Image generation pipeline
//!
//! Reference: diffusers QwenImagePipeline
//! "End-to-end text-to-image generation pipeline"

use mlx_rs::error::Exception;
use mlx_rs::ops;
use mlx_rs::Dtype;
use mlx_rs::Array;

use crate::transformer::QwenTransformer;
use crate::vae::QwenVAE;

/// Flow-matching Euler scheduler
/// Reference: diffusers FlowMatchEulerDiscreteScheduler
#[derive(Debug, Clone)]
pub struct FlowMatchEulerScheduler {
    pub num_inference_steps: i32,
    pub shift: f32,
    timesteps: Vec<f32>,
    sigmas: Vec<f32>,
}

impl FlowMatchEulerScheduler {
    pub fn new(num_inference_steps: i32, shift: f32) -> Self {
        // Linear timesteps from 1.0 to 0.0
        let timesteps: Vec<f32> = (0..=num_inference_steps)
            .map(|i| 1.0 - (i as f32 / num_inference_steps as f32))
            .collect();

        // Sigmas with time shift
        let sigmas: Vec<f32> = timesteps
            .iter()
            .map(|&t| {
                let shifted = shift * t / (1.0 + (shift - 1.0) * t);
                shifted
            })
            .collect();

        Self {
            num_inference_steps,
            shift,
            timesteps,
            sigmas,
        }
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps[..self.timesteps.len() - 1]
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// Perform one Euler step
    /// x_{t-dt} = x_t + (sigma_{t-dt} - sigma_t) * v_pred
    pub fn step(
        &self,
        model_output: &Array,
        timestep_idx: usize,
        sample: &Array,
    ) -> Result<Array, Exception> {
        let sigma = self.sigmas[timestep_idx];
        let sigma_next = self.sigmas[timestep_idx + 1];
        let dt = sigma_next - sigma;

        let dt_arr = Array::from_f32(dt);
        let delta = ops::multiply(model_output, &dt_arr)?;
        ops::add(sample, &delta)
    }

    /// Scale noise for initial latent
    pub fn scale_noise(&self, noise: &Array) -> Result<Array, Exception> {
        let sigma_max = self.sigmas[0];
        let scale = Array::from_f32(sigma_max);
        ops::multiply(noise, &scale)
    }
}

/// Qwen-Image generation pipeline
pub struct QwenImagePipeline {
    pub transformer: QwenTransformer,
    pub vae: QwenVAE,
    pub scheduler: FlowMatchEulerScheduler,
}

impl QwenImagePipeline {
    pub fn new(
        transformer: QwenTransformer,
        vae: QwenVAE,
        num_inference_steps: i32,
        shift: f32,
    ) -> Self {
        Self {
            transformer,
            vae,
            scheduler: FlowMatchEulerScheduler::new(num_inference_steps, shift),
        }
    }

    /// Generate image from text embeddings
    /// - encoder_hidden_states: [batch, seq_len, dim] text embeddings
    /// - height, width: output image dimensions (must be divisible by 16)
    /// - num_frames: number of frames (1 for image)
    /// - seed: random seed for reproducibility
    pub fn generate(
        &mut self,
        encoder_hidden_states: &Array,
        height: i32,
        width: i32,
        num_frames: i32,
        seed: Option<u64>,
    ) -> Result<Array, Exception> {
        let batch = encoder_hidden_states.dim(0);

        // Compute latent dimensions
        let latent_h = height / 8; // VAE downsamples 8x
        let latent_w = width / 8;
        let latent_channels = 16;

        // Initialize noise
        let latents = if let Some(s) = seed {
            let key = mlx_rs::random::key(s)?;
            mlx_rs::random::normal::<f32>(
                &[batch, latent_channels, num_frames, latent_h, latent_w],
                None,
                None,
                &key,
            )?
        } else {
            mlx_rs::random::normal::<f32>(
                &[batch, latent_channels, num_frames, latent_h, latent_w],
                None,
                None,
                None,
            )?
        };

        // Scale initial noise
        let mut latents = self.scheduler.scale_noise(&latents)?;

        // Denoising loop
        for (idx, &t) in self.scheduler.timesteps().iter().enumerate() {
            let timestep = Array::from_slice(&[t], &[batch]);
            let timestep = timestep.as_dtype(Dtype::Float32)?;

            // Predict velocity
            let v_pred = self.transformer.forward(
                &latents,
                encoder_hidden_states,
                &timestep,
            )?;

            // Euler step
            latents = self.scheduler.step(&v_pred, idx, &latents)?;
        }

        // Decode latents to image
        self.vae.decode(&latents)
    }

    /// Generate with classifier-free guidance
    pub fn generate_cfg(
        &mut self,
        encoder_hidden_states: &Array,      // Conditional embeddings
        null_encoder_hidden_states: &Array, // Unconditional embeddings
        height: i32,
        width: i32,
        num_frames: i32,
        guidance_scale: f32,
        seed: Option<u64>,
    ) -> Result<Array, Exception> {
        let batch = encoder_hidden_states.dim(0);

        // Compute latent dimensions
        let latent_h = height / 8;
        let latent_w = width / 8;
        let latent_channels = 16;

        // Initialize noise
        let latents = if let Some(s) = seed {
            let key = mlx_rs::random::key(s)?;
            mlx_rs::random::normal::<f32>(
                &[batch, latent_channels, num_frames, latent_h, latent_w],
                None,
                None,
                &key,
            )?
        } else {
            mlx_rs::random::normal::<f32>(
                &[batch, latent_channels, num_frames, latent_h, latent_w],
                None,
                None,
                None,
            )?
        };

        let mut latents = self.scheduler.scale_noise(&latents)?;

        // Denoising loop with CFG
        for (idx, &t) in self.scheduler.timesteps().iter().enumerate() {
            let timestep = Array::from_slice(&[t], &[batch]);
            let timestep = timestep.as_dtype(Dtype::Float32)?;

            // Predict conditional velocity
            let v_cond = self.transformer.forward(
                &latents,
                encoder_hidden_states,
                &timestep,
            )?;

            // Predict unconditional velocity
            let v_uncond = self.transformer.forward(
                &latents,
                null_encoder_hidden_states,
                &timestep,
            )?;

            // CFG: v = v_uncond + guidance_scale * (v_cond - v_uncond)
            let guidance = Array::from_f32(guidance_scale);
            let diff = ops::subtract(&v_cond, &v_uncond)?;
            let scaled_diff = ops::multiply(&diff, &guidance)?;
            let v_pred = ops::add(&v_uncond, &scaled_diff)?;

            // Euler step
            latents = self.scheduler.step(&v_pred, idx, &latents)?;
        }

        // Decode latents to image
        self.vae.decode(&latents)
    }
}

/// Attention mask builder for variable-length sequences
pub fn build_attention_mask(
    image_seq_len: i32,
    text_seq_len: i32,
    batch_size: i32,
) -> Result<Array, Exception> {
    // For now, return None (full attention)
    // Full mask would be [batch, 1, total_seq, total_seq]
    let total_seq = image_seq_len + text_seq_len;
    let zeros = Array::zeros::<f32>(&[batch_size, 1, total_seq, total_seq])?;
    Ok(zeros)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler() {
        let scheduler = FlowMatchEulerScheduler::new(20, 3.0);
        assert_eq!(scheduler.timesteps().len(), 20);
        assert!(scheduler.sigmas()[0] > scheduler.sigmas()[scheduler.sigmas().len() - 1]);
    }

    #[test]
    fn test_scheduler_step() {
        let scheduler = FlowMatchEulerScheduler::new(10, 3.0);
        let sample = Array::ones::<f32>(&[1, 4, 1, 8, 8]).unwrap();
        let v_pred = Array::ones::<f32>(&[1, 4, 1, 8, 8]).unwrap();
        let result = scheduler.step(&v_pred, 0, &sample).unwrap();
        assert_eq!(result.shape(), &[1, 4, 1, 8, 8]);
    }
}
