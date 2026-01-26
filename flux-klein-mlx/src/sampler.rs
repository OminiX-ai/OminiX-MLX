//! FLUX sampler - Rectified Flow implementation
//!
//! Ported from MLX Python implementation:
//! https://github.com/ml-explore/mlx-examples/blob/main/flux/flux/sampler.py

use mlx_rs::{
    array,
    error::Exception,
    ops,
    Array,
};

// ============================================================================
// Flux Sampler
// ============================================================================

/// Sampler configuration
#[derive(Debug, Clone)]
pub struct FluxSamplerConfig {
    /// Number of inference steps
    pub num_steps: i32,
    /// Whether this is a "schnell" (fast) model
    pub is_schnell: bool,
    /// Time shift parameter for non-schnell models
    pub shift: f32,
}

impl Default for FluxSamplerConfig {
    fn default() -> Self {
        Self {
            num_steps: 4,
            is_schnell: true,
            shift: 1.0,
        }
    }
}

impl FluxSamplerConfig {
    /// Config for FLUX.1-schnell (fast, 4 steps)
    pub fn schnell() -> Self {
        Self {
            num_steps: 4,
            is_schnell: true,
            shift: 1.0,
        }
    }

    /// Config for FLUX.1-dev (quality, 50 steps)
    pub fn dev() -> Self {
        Self {
            num_steps: 50,
            is_schnell: false,
            shift: 1.0,
        }
    }

    /// Config for FLUX.2-klein (distilled, 4 steps, linear schedule)
    pub fn klein() -> Self {
        Self {
            num_steps: 4,
            is_schnell: true,  // Uses linear schedule like schnell
            shift: 1.0,
        }
    }

    /// Set number of steps
    pub fn with_steps(mut self, steps: i32) -> Self {
        self.num_steps = steps;
        self
    }

    /// Set time shift
    pub fn with_shift(mut self, shift: f32) -> Self {
        self.shift = shift;
        self
    }
}

/// FLUX sampler implementing rectified flow
#[derive(Debug, Clone)]
pub struct FluxSampler {
    pub config: FluxSamplerConfig,
}

impl FluxSampler {
    /// Create a new sampler with the given configuration
    pub fn new(config: FluxSamplerConfig) -> Self {
        Self { config }
    }

    /// Create a schnell sampler
    pub fn schnell() -> Self {
        Self::new(FluxSamplerConfig::schnell())
    }

    /// Create a dev sampler
    pub fn dev() -> Self {
        Self::new(FluxSamplerConfig::dev())
    }

    /// Generate timestep schedule
    ///
    /// Returns timesteps from 1.0 to 0.0 (or near 0)
    pub fn timesteps(&self, num_steps: Option<i32>) -> Result<Vec<f32>, Exception> {
        let steps = num_steps.unwrap_or(self.config.num_steps);

        if self.config.is_schnell {
            // Schnell uses quantized timesteps
            let timesteps: Vec<f32> = (0..=steps)
                .map(|i| 1.0 - (i as f32 / steps as f32))
                .collect();
            Ok(timesteps)
        } else {
            // Dev uses shifted timesteps
            let timesteps: Vec<f32> = (0..=steps)
                .map(|i| {
                    let t = 1.0 - (i as f32 / steps as f32);
                    self.time_shift(t)
                })
                .collect();
            Ok(timesteps)
        }
    }

    /// Apply time shifting for non-schnell models
    ///
    /// This warps the timestep schedule to spend more time at higher noise levels
    fn time_shift(&self, t: f32) -> f32 {
        let shift = self.config.shift;
        // Exponential shift: t' = exp(shift) * t / (1 + (exp(shift) - 1) * t)
        let exp_shift = shift.exp();
        exp_shift * t / (1.0 + (exp_shift - 1.0) * t)
    }

    /// Sample from prior (standard Gaussian)
    ///
    /// # Arguments
    /// * `shape` - Shape of the latent tensor [batch, seq, channels]
    pub fn sample_prior(&self, shape: &[i32]) -> Result<Array, Exception> {
        mlx_rs::random::normal::<f32>(shape, None, None, None)
    }

    /// Add noise to data (forward diffusion)
    ///
    /// Implements: x_t = t * noise + (1 - t) * data
    ///
    /// # Arguments
    /// * `data` - Clean data
    /// * `noise` - Gaussian noise
    /// * `t` - Timestep value (scalar or [batch])
    pub fn add_noise(&self, data: &Array, noise: &Array, t: &Array) -> Result<Array, Exception> {
        // Expand t for broadcasting: [batch] -> [batch, 1, 1]
        let t = t.reshape(&[-1, 1, 1])?;
        let one_minus_t = ops::subtract(&array!(1.0), &t)?;

        // x_t = t * noise + (1 - t) * data
        let noisy = ops::add(
            &ops::multiply(&t, noise)?,
            &ops::multiply(&one_minus_t, data)?,
        )?;

        Ok(noisy)
    }

    /// Single denoising step (rectified flow)
    ///
    /// Implements: x_{t-1} = x_t + (t_prev - t) * v_pred
    ///
    /// # Arguments
    /// * `x_t` - Current noisy sample
    /// * `v_pred` - Model velocity prediction
    /// * `t` - Current timestep
    /// * `t_prev` - Previous (target) timestep
    pub fn step(
        &self,
        x_t: &Array,
        v_pred: &Array,
        t: f32,
        t_prev: f32,
    ) -> Result<Array, Exception> {
        let dt = t_prev - t;
        // x_{t-1} = x_t + dt * v_pred
        let update = ops::multiply(&array!(dt), v_pred)?;
        ops::add(x_t, &update)
    }

    /// Run the full denoising loop
    ///
    /// # Arguments
    /// * `model` - The FLUX model (passed as a closure for flexibility)
    /// * `latents` - Initial noisy latents from prior
    /// * `txt` - Text embeddings
    /// * `y` - Pooled text embeddings
    /// * `img_ids` - Image position IDs
    /// * `txt_ids` - Text position IDs
    /// * `guidance` - Optional guidance scale
    /// * `num_steps` - Override number of steps
    pub fn denoise_loop<F>(
        &self,
        mut model_fn: F,
        latents: Array,
        num_steps: Option<i32>,
    ) -> Result<Array, Exception>
    where
        F: FnMut(&Array, f32) -> Result<Array, Exception>,
    {
        let timesteps = self.timesteps(num_steps)?;
        let mut x = latents;

        // Iterate through timestep pairs (t, t_prev)
        for i in 0..timesteps.len() - 1 {
            let t = timesteps[i];
            let t_prev = timesteps[i + 1];

            // Get model prediction
            let v_pred = model_fn(&x, t)?;

            // Update latents
            x = self.step(&x, &v_pred, t, t_prev)?;
        }

        Ok(x)
    }
}

// ============================================================================
// Classifier-Free Guidance
// ============================================================================

/// Apply classifier-free guidance to model predictions
///
/// Implements: v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
///
/// # Arguments
/// * `v_cond` - Conditional prediction
/// * `v_uncond` - Unconditional prediction
/// * `guidance_scale` - Guidance scale (typically 3.5-7.0)
pub fn apply_cfg(
    v_cond: &Array,
    v_uncond: &Array,
    guidance_scale: f32,
) -> Result<Array, Exception> {
    let diff = ops::subtract(v_cond, v_uncond)?;
    let scaled = ops::multiply(&array!(guidance_scale), &diff)?;
    ops::add(v_uncond, &scaled)
}

// ============================================================================
// Official FLUX.2 Schedule
// ============================================================================

/// Compute empirical mu for the official FLUX.2 schedule
///
/// Matches Python's get_schedule() function from official flux2 code.
fn compute_empirical_mu(image_seq_len: i32, num_steps: i32) -> f32 {
    const A1: f32 = 8.73809524e-05;
    const B1: f32 = 1.89833333;
    const A2: f32 = 0.00016927;
    const B2: f32 = 0.45666666;

    if image_seq_len > 4300 {
        return A2 * image_seq_len as f32 + B2;
    }

    let m_200 = A2 * image_seq_len as f32 + B2;
    let m_10 = A1 * image_seq_len as f32 + B1;

    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * num_steps as f32 + b
}

/// Generalized time SNR shift used in official FLUX.2 schedule
fn generalized_time_snr_shift(t: f32, mu: f32, sigma: f32) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    mu.exp() / (mu.exp() + (1.0 / t - 1.0).powf(sigma))
}

/// Generate official FLUX.2 timestep schedule
///
/// Uses resolution-dependent mu calculation for optimal denoising trajectory.
///
/// # Arguments
/// * `num_steps` - Number of inference steps
/// * `image_seq_len` - Number of image tokens (patch_h * patch_w)
pub fn official_schedule(num_steps: i32, image_seq_len: i32) -> Vec<f32> {
    let mu = compute_empirical_mu(image_seq_len, num_steps);

    (0..=num_steps)
        .map(|i| {
            let t = 1.0 - (i as f32 / num_steps as f32);
            generalized_time_snr_shift(t, mu, 1.0)
        })
        .collect()
}

// ============================================================================
// Training Utilities
// ============================================================================

/// Sample random timesteps for training
///
/// # Arguments
/// * `batch_size` - Number of samples
/// * `is_schnell` - Whether to use quantized timesteps
/// * `num_steps` - Number of discrete steps (for schnell quantization)
pub fn sample_timesteps(
    batch_size: i32,
    is_schnell: bool,
    num_steps: Option<i32>,
) -> Result<Array, Exception> {
    if is_schnell {
        // Quantized timesteps for schnell
        let steps = num_steps.unwrap_or(4);
        let indices = mlx_rs::random::randint::<i32, i32>(0, steps, &[batch_size], None)?;
        let t = ops::divide(&indices.as_type::<f32>()?, &array!(steps as f32))?;
        Ok(t)
    } else {
        // Uniform timesteps for dev
        mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[batch_size], None)
    }
}

/// Compute training loss
///
/// For rectified flow, the loss is MSE between predicted velocity and target velocity.
/// Target velocity: v_target = noise - data
///
/// # Arguments
/// * `v_pred` - Model velocity prediction
/// * `noise` - Noise added to data
/// * `data` - Clean data
pub fn compute_loss(v_pred: &Array, noise: &Array, data: &Array) -> Result<Array, Exception> {
    // Target velocity
    let v_target = ops::subtract(noise, data)?;

    // MSE loss
    let diff = ops::subtract(v_pred, &v_target)?;
    let squared = ops::multiply(&diff, &diff)?;
    squared.mean(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampler_config() {
        let schnell = FluxSamplerConfig::schnell();
        assert!(schnell.is_schnell);
        assert_eq!(schnell.num_steps, 4);

        let dev = FluxSamplerConfig::dev();
        assert!(!dev.is_schnell);
        assert_eq!(dev.num_steps, 50);
    }

    #[test]
    fn test_timesteps() {
        let sampler = FluxSampler::schnell();
        let ts = sampler.timesteps(Some(4)).unwrap();

        assert_eq!(ts.len(), 5);
        assert!((ts[0] - 1.0).abs() < 1e-6);
        assert!((ts[4] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_noise() {
        let sampler = FluxSampler::schnell();

        let data = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let noise = Array::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[1, 2, 2]);
        let t = Array::from_slice(&[0.0f32], &[1]);

        let noisy = sampler.add_noise(&data, &noise, &t).unwrap();

        // At t=0, should return data unchanged
        let diff = ops::subtract(&noisy, &data).unwrap();
        let max_diff = ops::abs(&diff).unwrap().max(None).unwrap().item::<f32>();
        assert!(max_diff < 1e-6);
    }

    #[test]
    fn test_step() {
        let sampler = FluxSampler::schnell();

        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2]);
        let v = Array::from_slice(&[0.5f32, 0.5], &[1, 1, 2]);

        // Step from t=1.0 to t=0.75 with velocity [0.5, 0.5]
        let result = sampler.step(&x, &v, 1.0, 0.75).unwrap();

        // Expected: x + (0.75 - 1.0) * v = x - 0.25 * v
        // = [1.0, 2.0] + (-0.25) * [0.5, 0.5]
        // = [1.0 - 0.125, 2.0 - 0.125]
        // = [0.875, 1.875]
        let expected = Array::from_slice(&[0.875f32, 1.875], &[1, 1, 2]);
        let diff = ops::subtract(&result, &expected).unwrap();
        let max_diff = ops::abs(&diff).unwrap().max(None).unwrap().item::<f32>();
        assert!(max_diff < 1e-5);
    }
}
