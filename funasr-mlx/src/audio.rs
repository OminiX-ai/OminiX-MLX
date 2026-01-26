//! Audio loading and processing utilities
//!
//! Re-exports audio functions from mlx-rs-core with ASR-specific defaults.

// Re-export core audio functions
pub use mlx_rs_core::audio::{load_wav, resample, save_wav};

use crate::error::Result;

/// Resample audio for Paraformer (16kHz target)
///
/// Convenience wrapper around mlx_rs_core::audio::resample with Paraformer's
/// required sample rate.
pub fn resample_to_16k(samples: &[f32], src_rate: u32) -> Vec<f32> {
    resample(samples, src_rate, 16000)
}

/// Load and preprocess audio for Paraformer
///
/// Loads a WAV file and resamples to 16kHz if needed.
pub fn load_audio_for_paraformer(path: impl AsRef<std::path::Path>) -> Result<(Vec<f32>, u32)> {
    let (samples, sample_rate) = load_wav(&path)?;

    let samples = if sample_rate != 16000 {
        resample(&samples, sample_rate, 16000)
    } else {
        samples
    };

    Ok((samples, 16000))
}
