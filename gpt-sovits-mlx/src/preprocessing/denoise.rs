//! Audio Denoising using spectral subtraction
//!
//! Provides basic noise reduction for audio preprocessing

use std::path::Path;
use super::{PreprocessingError, Result};
use tracing::{debug, info};

/// Configuration for audio denoising
#[derive(Debug, Clone)]
pub struct DenoiseConfig {
    /// Sample rate for processing
    pub sample_rate: u32,
    /// FFT size for STFT
    pub n_fft: usize,
    /// Hop length for STFT
    pub hop_length: usize,
    /// Noise estimation frames (from beginning of audio)
    pub noise_frames: usize,
    /// Spectral floor to prevent musical noise
    pub spectral_floor: f32,
    /// Over-subtraction factor (1.0 = normal, >1.0 = aggressive)
    pub over_subtraction: f32,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            sample_rate: 32000,
            n_fft: 2048,
            hop_length: 512,
            noise_frames: 10,
            spectral_floor: 0.01,
            over_subtraction: 1.0,
        }
    }
}

/// Audio denoiser using spectral subtraction
pub struct Denoiser {
    config: DenoiseConfig,
    window: Vec<f32>,
}

impl Denoiser {
    /// Create a new denoiser
    pub fn new(config: DenoiseConfig) -> Result<Self> {
        // Create Hann window
        let window: Vec<f32> = (0..config.n_fft)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / config.n_fft as f32).cos())
            })
            .collect();

        Ok(Self { config, window })
    }

    /// Compute STFT magnitude and phase
    fn stft(&self, samples: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;
        let n_bins = n_fft / 2 + 1;

        // Pad signal
        let pad_len = n_fft / 2;
        let mut padded = vec![0.0f32; pad_len];
        padded.extend_from_slice(samples);
        padded.extend(vec![0.0f32; pad_len]);

        // Number of frames
        let n_frames = (padded.len() - n_fft) / hop + 1;

        let mut magnitudes = Vec::with_capacity(n_frames);
        let mut phases = Vec::with_capacity(n_frames);

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            let frame: Vec<f32> = padded[start..start + n_fft]
                .iter()
                .zip(&self.window)
                .map(|(s, w)| s * w)
                .collect();

            // Simple DFT (for production, use FFT library)
            let (mag, phase) = self.dft(&frame, n_bins);
            magnitudes.push(mag);
            phases.push(phase);
        }

        (magnitudes, phases)
    }

    /// Simple DFT implementation
    fn dft(&self, frame: &[f32], n_bins: usize) -> (Vec<f32>, Vec<f32>) {
        let n = frame.len();
        let mut mag = Vec::with_capacity(n_bins);
        let mut phase = Vec::with_capacity(n_bins);

        for k in 0..n_bins {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for (i, &x) in frame.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * i as f32 / n as f32;
                real += x * angle.cos();
                imag += x * angle.sin();
            }

            mag.push((real * real + imag * imag).sqrt());
            phase.push(imag.atan2(real));
        }

        (mag, phase)
    }

    /// Inverse DFT
    fn idft(&self, mag: &[f32], phase: &[f32]) -> Vec<f32> {
        let n_bins = mag.len();
        let n = self.config.n_fft;
        let mut output = vec![0.0f32; n];

        // Reconstruct full spectrum (mirror for real signal)
        for i in 0..n {
            let mut sum = 0.0f32;

            for k in 0..n_bins {
                let angle = 2.0 * std::f32::consts::PI * k as f32 * i as f32 / n as f32;
                sum += mag[k] * (angle + phase[k]).cos();
            }

            // Mirror contribution
            for k in 1..n_bins - 1 {
                let angle = 2.0 * std::f32::consts::PI * (n - k) as f32 * i as f32 / n as f32;
                sum += mag[k] * (angle - phase[k]).cos();
            }

            output[i] = sum / n as f32;
        }

        output
    }

    /// Overlap-add synthesis
    fn istft(&self, magnitudes: &[Vec<f32>], phases: &[Vec<f32>]) -> Vec<f32> {
        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;
        let n_frames = magnitudes.len();

        // Output length
        let output_len = (n_frames - 1) * hop + n_fft;
        let mut output = vec![0.0f32; output_len];
        let mut window_sum = vec![0.0f32; output_len];

        for (frame_idx, (mag, phase)) in magnitudes.iter().zip(phases.iter()).enumerate() {
            let frame = self.idft(mag, phase);
            let start = frame_idx * hop;

            for (i, (&sample, &w)) in frame.iter().zip(&self.window).enumerate() {
                if start + i < output_len {
                    output[start + i] += sample * w;
                    window_sum[start + i] += w * w;
                }
            }
        }

        // Normalize by window sum
        for (out, &w_sum) in output.iter_mut().zip(&window_sum) {
            if w_sum > 1e-8 {
                *out /= w_sum;
            }
        }

        // Remove padding
        let pad = n_fft / 2;
        if output.len() > 2 * pad {
            output[pad..output.len() - pad].to_vec()
        } else {
            output
        }
    }

    /// Estimate noise profile from initial frames
    fn estimate_noise(&self, magnitudes: &[Vec<f32>]) -> Vec<f32> {
        let n_frames = magnitudes.len().min(self.config.noise_frames);
        let n_bins = magnitudes[0].len();
        let mut noise_profile = vec![0.0f32; n_bins];

        for frame in &magnitudes[..n_frames] {
            for (i, &mag) in frame.iter().enumerate() {
                noise_profile[i] += mag;
            }
        }

        for val in &mut noise_profile {
            *val /= n_frames as f32;
        }

        noise_profile
    }

    /// Apply spectral subtraction
    fn spectral_subtract(
        &self,
        magnitudes: &[Vec<f32>],
        noise_profile: &[f32],
    ) -> Vec<Vec<f32>> {
        let floor = self.config.spectral_floor;
        let alpha = self.config.over_subtraction;

        magnitudes
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .zip(noise_profile)
                    .map(|(&mag, &noise)| {
                        let subtracted = mag - alpha * noise;
                        subtracted.max(floor * mag) // Spectral floor
                    })
                    .collect()
            })
            .collect()
    }

    /// Denoise audio samples
    pub fn denoise(&self, samples: &[f32]) -> Vec<f32> {
        debug!(samples = samples.len(), "Denoising audio");

        // STFT
        let (magnitudes, phases) = self.stft(samples);

        // Estimate noise from first few frames
        let noise_profile = self.estimate_noise(&magnitudes);

        // Spectral subtraction
        let denoised_mag = self.spectral_subtract(&magnitudes, &noise_profile);

        // ISTFT
        let output = self.istft(&denoised_mag, &phases);

        // Trim to original length
        let output_len = samples.len().min(output.len());
        output[..output_len].to_vec()
    }

    /// Process an audio file
    pub fn process_file<P1: AsRef<Path>, P2: AsRef<Path>>(&self, input_path: P1, output_path: P2) -> Result<()> {
        let input_path = input_path.as_ref();
        let output_path = output_path.as_ref();

        info!(input = %input_path.display(), output = %output_path.display(), "Denoising file");

        // Load audio
        let (samples, sr) = mlx_rs_core::audio::load_wav(input_path)
            .map_err(|e| PreprocessingError::Audio(format!("Failed to load: {}", e)))?;

        // Resample if needed
        let samples = if sr != self.config.sample_rate {
            mlx_rs_core::audio::resample(&samples, sr, self.config.sample_rate)
        } else {
            samples
        };

        // Denoise
        let denoised = self.denoise(&samples);

        // Save
        mlx_rs_core::audio::save_wav(&denoised, self.config.sample_rate, output_path)
            .map_err(|e| PreprocessingError::Audio(format!("Failed to save: {}", e)))?;

        Ok(())
    }

    /// Process a directory of audio files
    pub fn process_directory<P1: AsRef<Path>, P2: AsRef<Path>>(&self, input_dir: P1, output_dir: P2) -> Result<()> {
        let input_dir = input_dir.as_ref();
        let output_dir = output_dir.as_ref();

        std::fs::create_dir_all(output_dir)?;

        for entry in std::fs::read_dir(input_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if ext == "wav" {
                        let filename = path.file_name().unwrap();
                        let output_path = output_dir.join(filename);
                        self.process_file(&path, &output_path)?;
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_denoise_config_default() {
        let config = DenoiseConfig::default();
        assert_eq!(config.n_fft, 2048);
        assert_eq!(config.hop_length, 512);
    }

    #[test]
    fn test_hann_window() {
        let config = DenoiseConfig {
            n_fft: 8,
            ..Default::default()
        };
        let denoiser = Denoiser::new(config).unwrap();

        // Hann window should be symmetric and peak at center
        assert!(denoiser.window[0] < 0.01); // Near zero at edges
        assert!(denoiser.window[4] > 0.99); // Peak at center
    }

    #[test]
    fn test_denoise_passthrough() {
        let config = DenoiseConfig {
            n_fft: 256,
            hop_length: 64,
            noise_frames: 2,
            spectral_floor: 0.0,
            over_subtraction: 0.0, // No subtraction = passthrough
            ..Default::default()
        };
        let denoiser = Denoiser::new(config).unwrap();

        // Create test signal
        let samples: Vec<f32> = (0..1000)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();

        let output = denoiser.denoise(&samples);

        // Output should be similar length
        assert!(output.len() >= samples.len() - 256);
    }
}
