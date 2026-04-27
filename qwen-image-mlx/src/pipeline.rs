//! Qwen-Image generation pipeline
//!
//! Reference: diffusers QwenImagePipeline
//! "End-to-end text-to-image generation pipeline"

use mlx_rs::error::Exception;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
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

    /// Generate with classifier-free guidance (batched: cond+uncond in a single forward).
    /// P0 optimization: halves weight reads per step by running batch=2.
    pub fn generate_cfg(
        &mut self,
        encoder_hidden_states: &Array,      // Conditional embeddings [1, T_pos, D]
        null_encoder_hidden_states: &Array, // Unconditional embeddings [1, T_neg, D]
        height: i32,
        width: i32,
        num_frames: i32,
        guidance_scale: f32,
        seed: Option<u64>,
    ) -> Result<Array, Exception> {
        let latent_h = height / 8;
        let latent_w = width / 8;
        let latent_channels = 16;

        // Initialize noise [1, C, F, H, W]
        let latents = if let Some(s) = seed {
            let key = mlx_rs::random::key(s)?;
            mlx_rs::random::normal::<f32>(
                &[1, latent_channels, num_frames, latent_h, latent_w],
                None,
                None,
                &key,
            )?
        } else {
            mlx_rs::random::normal::<f32>(
                &[1, latent_channels, num_frames, latent_h, latent_w],
                None,
                None,
                None,
            )?
        };

        let mut latents = self.scheduler.scale_noise(&latents)?;

        // P0: Pad cond/uncond to same text length, stack as batch=2
        let t_pos = encoder_hidden_states.dim(1);
        let t_neg = null_encoder_hidden_states.dim(1);
        let t_max = t_pos.max(t_neg);
        let d = encoder_hidden_states.dim(2);

        let pos_padded = if t_pos < t_max {
            let pad = Array::zeros::<f32>(&[1, t_max - t_pos, d])?;
            ops::concatenate_axis(&[encoder_hidden_states, &pad], 1)?
        } else {
            encoder_hidden_states.clone()
        };
        let neg_padded = if t_neg < t_max {
            let pad = Array::zeros::<f32>(&[1, t_max - t_neg, d])?;
            ops::concatenate_axis(&[null_encoder_hidden_states, &pad], 1)?
        } else {
            null_encoder_hidden_states.clone()
        };
        let batched_prompt = ops::concatenate_axis(&[&pos_padded, &neg_padded], 0)?; // [2, T, D]

        // Denoising loop — single batched forward per step
        for (idx, &t) in self.scheduler.timesteps().iter().enumerate() {
            let timestep = Array::from_slice(&[t], &[1]);
            let timestep = timestep.as_dtype(Dtype::Float32)?;

            // Broadcast latents to batch=2
            let latents_b = ops::broadcast_to(
                &latents,
                &[2, latent_channels, num_frames, latent_h, latent_w],
            )?;

            let both_pred = self.transformer.forward(
                &latents_b,
                &batched_prompt,
                &timestep,
            )?;

            // Split: cond = [0:1], uncond = [1:2]
            let v_cond = both_pred.index((0..1, .., .., .., ..));
            let v_uncond = both_pred.index((1..2, .., .., .., ..));

            // CFG: v = v_uncond + guidance_scale * (v_cond - v_uncond)
            let guidance = Array::from_f32(guidance_scale);
            let diff = ops::subtract(&v_cond, &v_uncond)?;
            let scaled_diff = ops::multiply(&diff, &guidance)?;
            let v_pred = ops::add(&v_uncond, &scaled_diff)?;

            latents = self.scheduler.step(&v_pred, idx, &latents)?;
        }

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

// ─── Latent packing/unpacking (patchify for DiT) ────────────────────────────

/// Pack latents: [B, C, 1, H, W] -> [B, (H/2)*(W/2), C*4]
/// Rearranges spatial dims into patch tokens for the DiT transformer.
pub fn pack_latents(latents: &Array) -> Result<Array, Exception> {
    let batch = latents.dim(0);
    let channels = latents.dim(1);
    let height = latents.dim(3);
    let width = latents.dim(4);

    // [B, C, H//2, 2, W//2, 2]
    let x = latents.reshape(&[batch, channels, height / 2, 2, width / 2, 2])?;
    // [B, H//2, W//2, C, 2, 2]
    let x = x.transpose_axes(&[0, 2, 4, 1, 3, 5])?;
    // [B, (H//2)*(W//2), C*4]
    x.reshape(&[batch, (height / 2) * (width / 2), channels * 4])
}

/// Unpack latents: [B, seq, C*4] -> [B, C, 1, H, W]
/// height/width are original image dimensions (in pixels).
pub fn unpack_latents(latents: &Array, height: i32, width: i32) -> Result<Array, Exception> {
    let batch = latents.dim(0);
    let channels = latents.dim(2); // packed channels (e.g., 64)
    let vae_scale_factor = 8;
    let latent_h = 2 * (height / (vae_scale_factor * 2));
    let latent_w = 2 * (width / (vae_scale_factor * 2));
    let out_channels = channels / 4;

    // [B, H/2, W/2, C/4, 2, 2]
    let x = latents.reshape(&[batch, latent_h / 2, latent_w / 2, out_channels, 2, 2])?;
    // [B, C/4, H/2, 2, W/2, 2]
    let x = x.transpose_axes(&[0, 3, 1, 4, 2, 5])?;
    // [B, C/4, 1, H, W]
    x.reshape(&[batch, out_channels, 1, latent_h, latent_w])
}

/// Encode a reference image through the VAE and pack into DiT patches.
/// image: [B, 4, H, W] (RGBA, values in [-1, 1])
/// Returns packed patches [B, (H/16)*(W/16), 64] ready for forward_edit.
pub fn encode_reference_latent(vae: &mut QwenVAE, image: &Array) -> Result<Array, Exception> {
    // VAE encode returns normalized [B, 16, H/8, W/8]
    let normalized = vae.encode(image)?;

    // Reshape to 5D for packing: [B, 16, 1, H/8, W/8]
    let b = normalized.dim(0);
    let c = normalized.dim(1);
    let h = normalized.dim(2);
    let w = normalized.dim(3);
    let latent_5d = normalized.reshape(&[b, c, 1, h, w])?;

    pack_latents(&latent_5d)
}

/// Compute the ref_shape in patchified space for a reference image.
/// latent_h, latent_w: VAE latent dimensions (image_dim / 8)
/// Returns (frame=1, patch_h, patch_w)
pub fn ref_shape_from_latent(latent_h: i32, latent_w: i32) -> (i32, i32, i32) {
    (1, latent_h / 2, latent_w / 2)
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

    #[test]
    fn test_pack_unpack_latents() {
        // [1, 16, 1, 8, 8] -> pack -> unpack should preserve shape
        let latents = Array::zeros::<f32>(&[1, 16, 1, 8, 8]).unwrap();
        let packed = pack_latents(&latents).unwrap();
        assert_eq!(packed.shape(), &[1, 16, 64]); // (8/2)*(8/2)=16 patches, 16*4=64 dim

        // Unpack: original image was 64x64 pixels (latent 8x8)
        let unpacked = unpack_latents(&packed, 64, 64).unwrap();
        assert_eq!(unpacked.shape(), &[1, 16, 1, 8, 8]);
    }

    #[test]
    fn test_pack_latents_larger() {
        // 512x512 image -> latent 64x64 -> packed 32*32=1024 patches, 64 dim
        let latents = Array::zeros::<f32>(&[1, 16, 1, 64, 64]).unwrap();
        let packed = pack_latents(&latents).unwrap();
        assert_eq!(packed.shape(), &[1, 1024, 64]);
    }
}
