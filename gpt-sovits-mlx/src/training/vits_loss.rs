//! VITS Loss Functions for GAN Training
//!
//! This module provides loss functions used in VITS/HiFi-GAN training:
//! - Generator adversarial loss (LSGAN)
//! - Discriminator adversarial loss (LSGAN)
//! - Feature matching loss
//! - KL divergence loss
//! - Mel spectrogram reconstruction loss

use mlx_rs::{
    array,
    error::Exception,
    ops::{abs as array_abs, exp, ones_like, square},
    Array,
};

/// Generator adversarial loss (Least Squares GAN)
///
/// For each discriminator output, computes: mean((output - 1)^2)
/// This encourages the generator to produce outputs that the discriminator
/// classifies as real (output close to 1).
pub fn generator_loss(disc_outputs: &[Array]) -> Result<Array, Exception> {
    let mut total_loss = array!(0.0f32);

    for output in disc_outputs {
        // (output - 1)^2
        let ones = ones_like(output)?;
        let diff = output.subtract(&ones)?;
        let squared = square(&diff)?;
        let loss = squared.mean(false)?;

        total_loss = total_loss.add(&loss)?;
    }

    Ok(total_loss)
}

/// Discriminator adversarial loss (Least Squares GAN)
///
/// For each discriminator:
/// - Real loss: mean((real_output - 1)^2) - wants real to be classified as 1
/// - Fake loss: mean(fake_output^2) - wants fake to be classified as 0
pub fn discriminator_loss(
    real_outputs: &[Array],
    fake_outputs: &[Array],
) -> Result<Array, Exception> {
    let mut total_loss = array!(0.0f32);

    for (real, fake) in real_outputs.iter().zip(fake_outputs.iter()) {
        // Real: (real - 1)^2
        let ones = ones_like(real)?;
        let diff_real = real.subtract(&ones)?;
        let loss_real = square(&diff_real)?.mean(false)?;

        // Fake: fake^2
        let loss_fake = square(fake)?.mean(false)?;

        let loss = loss_real.add(&loss_fake)?;
        total_loss = total_loss.add(&loss)?;
    }

    Ok(total_loss)
}

/// Feature matching loss
///
/// Computes L1 distance between real and fake feature maps from discriminator.
/// This helps stabilize GAN training by providing additional gradients.
///
/// Loss = 2 * sum(mean(|real_fmap - fake_fmap|))
pub fn feature_matching_loss(
    real_fmaps: &[Vec<Array>],
    fake_fmaps: &[Vec<Array>],
) -> Result<Array, Exception> {
    let mut total_loss = array!(0.0f32);

    for (real_fmap, fake_fmap) in real_fmaps.iter().zip(fake_fmaps.iter()) {
        // Detach real features (stop gradient)
        for (real, fake) in real_fmap.iter().zip(fake_fmap.iter()) {
            let diff = real.subtract(fake)?;
            let abs_diff = array_abs(&diff)?;
            let loss = abs_diff.mean(false)?;
            total_loss = total_loss.add(&loss)?;
        }
    }

    // Multiply by 2 as in original implementation
    total_loss.multiply(array!(2.0f32))
}

/// KL divergence loss for VAE posterior
///
/// Computes KL(q(z|x) || p(z|c)) where:
/// - q: posterior (from spectrogram encoder)
/// - p: prior (from text encoder)
///
/// KL = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p)^2) * exp(-2 * logs_p)
pub fn kl_loss(
    z_p: &Array,      // Flow-transformed latent
    logs_q: &Array,   // Posterior log-variance
    m_p: &Array,      // Prior mean
    logs_p: &Array,   // Prior log-variance
    z_mask: &Array,   // Mask for valid positions
) -> Result<Array, Exception> {
    // KL = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p)^2) * exp(-2 * logs_p)
    let diff = z_p.subtract(m_p)?;
    let diff_sq = square(&diff)?;
    let neg_2_logs_p = logs_p.multiply(array!(-2.0f32))?;
    let exp_term = exp(&neg_2_logs_p)?;

    let kl_term = logs_p
        .subtract(logs_q)?
        .subtract(array!(0.5f32))?
        .add(&diff_sq.multiply(&exp_term)?.multiply(array!(0.5f32))?)?;

    // Mask and sum
    let kl_masked = kl_term.multiply(z_mask)?;
    let kl_sum = kl_masked.sum(false)?;
    let mask_sum = z_mask.sum(false)?;

    // Average over valid positions
    kl_sum.divide(&mask_sum)
}

/// Mel spectrogram L1 loss
///
/// Simple L1 reconstruction loss between real and generated mel spectrograms.
pub fn mel_reconstruction_loss(
    mel_real: &Array,
    mel_fake: &Array,
) -> Result<Array, Exception> {
    let diff = mel_real.subtract(mel_fake)?;
    let abs_diff = array_abs(&diff)?;
    abs_diff.mean(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_loss() {
        // Discriminator output = 0.5 (midway between real and fake)
        let output = Array::from_slice(&[0.5f32; 10], &[1, 10]);
        let loss = generator_loss(&[output]).unwrap();

        // Expected: mean((0.5 - 1)^2) = mean(0.25) = 0.25
        let loss_val: f32 = loss.item();
        assert!((loss_val - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_discriminator_loss() {
        // Real output = 0.8 (close to 1), Fake output = 0.2 (close to 0)
        let real = Array::from_slice(&[0.8f32; 10], &[1, 10]);
        let fake = Array::from_slice(&[0.2f32; 10], &[1, 10]);

        let loss = discriminator_loss(&[real], &[fake]).unwrap();

        // Expected: mean((0.8-1)^2) + mean(0.2^2) = 0.04 + 0.04 = 0.08
        let loss_val: f32 = loss.item();
        assert!((loss_val - 0.08).abs() < 0.01);
    }
}
