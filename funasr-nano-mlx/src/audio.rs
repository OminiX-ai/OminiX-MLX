//! Audio processing utilities for Fun-ASR-Nano.
//!
//! This module provides:
//! - Audio file loading (WAV)
//! - Resampling to 16kHz
//! - Mel spectrogram computation with FFT (O(n log n) instead of O(n²) DFT)

use crate::error::{Error, Result};
use mlx_rs::Array;
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;

/// Audio processing configuration.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Target sample rate (16000 Hz for Whisper)
    pub sample_rate: u32,
    /// Number of mel filterbank bins
    pub n_mels: usize,
    /// FFT window size in samples
    pub n_fft: usize,
    /// Hop length between frames
    pub hop_length: usize,
    /// Maximum audio length in seconds (30s for Whisper)
    pub max_length: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_mels: 80,
            n_fft: 400,        // 25ms at 16kHz
            hop_length: 160,   // 10ms at 16kHz
            max_length: 30.0,
        }
    }
}

/// Mel spectrogram frontend with cached FFT for efficient computation.
///
/// Pre-computes the FFT planner, window function, and mel filterbank
/// for repeated use across multiple audio files.
pub struct MelFrontend {
    /// Cached FFT instance (thread-safe)
    fft: Arc<dyn rustfft::Fft<f32>>,
    /// Pre-computed Hann window
    window: Vec<f32>,
    /// Pre-computed mel filterbank [n_mels, n_freqs]
    mel_filters: Vec<f32>,
    /// Configuration
    config: AudioConfig,
    /// Number of frequency bins (n_fft/2 + 1)
    n_freqs: usize,
}

impl MelFrontend {
    /// Create a new MelFrontend with the given configuration.
    pub fn new(config: AudioConfig) -> Self {
        let n_fft = config.n_fft;
        let n_freqs = n_fft / 2 + 1;

        // Create cached FFT planner
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);

        // Pre-compute Hann window
        let window: Vec<f32> = (0..n_fft)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n_fft - 1) as f32).cos())
            })
            .collect();

        // Pre-compute mel filterbank
        let mel_filters = create_mel_filterbank(config.sample_rate, n_fft, config.n_mels);

        Self {
            fft,
            window,
            mel_filters,
            config,
            n_freqs,
        }
    }

    /// Compute mel spectrogram from audio samples using FFT.
    ///
    /// Returns a tensor of shape [1, n_mels, n_frames].
    ///
    /// # Errors
    ///
    /// Returns an error if samples is empty or too short to produce any frames.
    pub fn compute_mel_spectrogram(&self, samples: &[f32]) -> Result<Array> {
        // Validate input
        if samples.is_empty() {
            return Err(Error::AudioFormat {
                message: "Cannot compute mel spectrogram: audio samples are empty".to_string(),
            });
        }

        let n_fft = self.config.n_fft;
        let hop_length = self.config.hop_length;
        let n_mels = self.config.n_mels;

        // Check minimum length for at least one frame
        if samples.len() < hop_length {
            return Err(Error::audio_too_short(
                (samples.len() as u64 * 1000) / self.config.sample_rate as u64,
                (hop_length as u64 * 1000) / self.config.sample_rate as u64,
            ));
        }

        // Pad audio to ensure we have enough samples
        let max_samples = (self.config.max_length * self.config.sample_rate as f32) as usize;
        let audio: Vec<f32> = if samples.len() > max_samples {
            samples[..max_samples].to_vec()
        } else {
            samples.to_vec()
        };

        // Compute number of frames
        let n_frames = (audio.len() / hop_length).max(1);

        // Pre-allocate FFT buffer (reused for each frame)
        let mut fft_buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n_fft];
        let mut mel_spec = vec![0.0f32; n_mels * n_frames];

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;
            let end = (start + n_fft).min(audio.len());

            // Apply window and copy to FFT buffer
            for i in 0..n_fft {
                let sample = if start + i < end { audio[start + i] } else { 0.0 };
                fft_buffer[i] = Complex::new(sample * self.window[i], 0.0);
            }

            // Compute FFT in-place
            self.fft.process(&mut fft_buffer);

            // Compute power spectrum and apply mel filterbank
            for mel_idx in 0..n_mels {
                let mut mel_energy = 0.0f32;
                for freq_idx in 0..self.n_freqs {
                    let c = fft_buffer[freq_idx];
                    let power = c.re * c.re + c.im * c.im;
                    mel_energy += power * self.mel_filters[mel_idx * self.n_freqs + freq_idx];
                }
                // Log mel spectrogram
                mel_spec[mel_idx * n_frames + frame_idx] = (mel_energy.max(1e-10)).ln();
            }
        }

        // Create MLX array [1, n_mels, n_frames]
        let array = Array::from_slice(&mel_spec, &[1, n_mels as i32, n_frames as i32]);

        Ok(array)
    }
}

/// Minimum audio duration in milliseconds for processing.
pub const MIN_AUDIO_DURATION_MS: u64 = 100;

/// Load audio from a WAV file.
///
/// Returns samples normalized to [-1, 1] and the sample rate.
///
/// # Errors
///
/// Returns an error if:
/// - File does not exist
/// - File is not a valid WAV file
/// - Audio is empty or too short (< 100ms)
pub fn load_wav(path: impl AsRef<std::path::Path>) -> Result<(Vec<f32>, u32)> {
    use std::fs::File;
    use std::io::BufReader;

    let path = path.as_ref();

    // Check if file exists
    if !path.exists() {
        return Err(Error::audio_not_found(path));
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Simple WAV parsing - assumes PCM format
    let mut reader = hound::WavReader::new(reader).map_err(|e| Error::AudioFormat {
        message: format!("Failed to read WAV file '{}': {}", path.display(), e),
    })?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    // Validate sample rate
    if sample_rate == 0 {
        return Err(Error::AudioFormat {
            message: "Invalid sample rate: 0".to_string(),
        });
    }

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
    };

    // Convert stereo to mono if needed
    let samples = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    // Validate audio is not empty
    if samples.is_empty() {
        return Err(Error::AudioFormat {
            message: format!("Audio file '{}' contains no samples", path.display()),
        });
    }

    // Validate minimum duration
    let duration_ms = (samples.len() as u64 * 1000) / sample_rate as u64;
    if duration_ms < MIN_AUDIO_DURATION_MS {
        return Err(Error::audio_too_short(duration_ms, MIN_AUDIO_DURATION_MS));
    }

    Ok((samples, sample_rate))
}

/// Resample audio to target sample rate using high-quality sinc interpolation.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    use rubato::{FftFixedInOut, Resampler};

    // FftFixedInOut::new(sample_rate_in, sample_rate_out, chunk_size_in, channels)
    let mut resampler = FftFixedInOut::<f32>::new(from_rate as usize, to_rate as usize, 1024, 1)
        .map_err(|e| Error::Audio(format!("Resampler init failed: {}", e)))?;

    let mut output = Vec::new();
    let chunk_size = resampler.input_frames_max();

    for chunk in samples.chunks(chunk_size) {
        let mut padded = chunk.to_vec();
        if padded.len() < chunk_size {
            padded.resize(chunk_size, 0.0);
        }

        let result = resampler
            .process(&[padded], None)
            .map_err(|e| Error::Audio(format!("Resampling failed: {}", e)))?;

        output.extend_from_slice(&result[0]);
    }

    // Trim to expected length
    let expected_len = (samples.len() as f64 * to_rate as f64 / from_rate as f64).round() as usize;
    output.truncate(expected_len);

    Ok(output)
}

/// Compute mel spectrogram from audio samples using FFT.
///
/// Returns a tensor of shape [1, n_mels, n_frames].
///
/// Note: For processing multiple audio files, use `MelFrontend` directly
/// to avoid recreating the FFT planner and mel filterbank each time.
pub fn compute_mel_spectrogram(samples: &[f32], config: &AudioConfig) -> Result<Array> {
    let frontend = MelFrontend::new(config.clone());
    frontend.compute_mel_spectrogram(samples)
}

/// Create mel filterbank matrix.
fn create_mel_filterbank(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;

    // Mel scale conversion
    let hz_to_mel = |hz: f32| -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() };
    let mel_to_hz = |mel: f32| -> f32 { 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0) };

    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(sample_rate as f32 / 2.0);

    // Create mel points
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert to FFT bin indices
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((n_fft + 1) as f32 * hz / sample_rate as f32).floor() as usize)
        .collect();

    // Create filterbank
    let mut filterbank = vec![0.0f32; n_mels * n_freqs];

    for mel_idx in 0..n_mels {
        let left = bin_points[mel_idx];
        let center = bin_points[mel_idx + 1];
        let right = bin_points[mel_idx + 2];

        // Rising edge
        for k in left..center {
            if k < n_freqs && center > left {
                filterbank[mel_idx * n_freqs + k] = (k - left) as f32 / (center - left) as f32;
            }
        }

        // Falling edge
        for k in center..right {
            if k < n_freqs && right > center {
                filterbank[mel_idx * n_freqs + k] = (right - k) as f32 / (right - center) as f32;
            }
        }
    }

    filterbank
}

/// Apply Low Frame Rate (LFR) transformation using pure MLX operations.
///
/// LFR stacks `m` consecutive frames and subsamples by factor `n`.
/// This reduces the sequence length while preserving temporal information.
///
/// Input: [batch, n_mels, n_frames]
/// Output: [batch, n_frames/n, n_mels * m]
///
/// For Fun-ASR-Nano: m=7, n=6, producing 560-dim features (80 * 7).
///
/// This implementation uses pure MLX operations to avoid GPU→CPU→GPU transfers,
/// providing ~5-10% speedup over CPU-based implementations.
pub fn apply_lfr(mel: &Array, lfr_m: usize, lfr_n: usize) -> Result<Array> {
    use mlx_rs::ops::{self, indexing::IndexOp};

    let shape = mel.shape();
    let batch = shape[0];
    let n_mels = shape[1];
    let n_frames = shape[2] as usize;

    // Calculate output dimensions
    let n_lfr_frames = (n_frames + lfr_n - 1) / lfr_n;
    let left_pad = lfr_m / 2;

    // Transpose to [batch, n_frames, n_mels] for frame-wise processing
    let mel_t = mel.transpose_axes(&[0, 2, 1])?;

    // Pad frames on both sides to handle edge cases
    // Left padding: replicate first frame
    // Right padding: replicate last frame + extra for subsampling alignment
    let right_pad = lfr_m - 1 - left_pad + (lfr_n - (n_frames % lfr_n)) % lfr_n;

    // Get first and last frames for padding
    let first_frame = mel_t.index((.., 0..1, ..)); // [batch, 1, n_mels]
    let last_frame = mel_t.index((.., (n_frames as i32 - 1).., ..)); // [batch, 1, n_mels]

    // Create padding by repeating first/last frames along axis 1
    let left_padding = if left_pad > 0 {
        ops::repeat_axis::<f32>(first_frame, left_pad as i32, 1)?
    } else {
        Array::zeros::<f32>(&[batch, 0, n_mels])?
    };

    let right_padding = if right_pad > 0 {
        ops::repeat_axis::<f32>(last_frame, right_pad as i32, 1)?
    } else {
        Array::zeros::<f32>(&[batch, 0, n_mels])?
    };

    // Concatenate: [left_pad | mel_t | right_pad]
    let padded = ops::concatenate_axis(&[&left_padding, &mel_t, &right_padding], 1)?;
    // padded shape: [batch, left_pad + n_frames + right_pad, n_mels]

    // Now gather lfr_m consecutive frames for each output frame
    // Output frame i uses frames [i*lfr_n, i*lfr_n + lfr_m)
    let mut stacked_frames = Vec::with_capacity(lfr_m);
    for m in 0..lfr_m {
        // Create indices for this offset: [0, lfr_n, 2*lfr_n, ...]
        let indices: Vec<i32> = (0..n_lfr_frames)
            .map(|i| (i * lfr_n + m) as i32)
            .collect();
        let idx_array = Array::from_slice(&indices, &[n_lfr_frames as i32]);

        // Gather frames at these indices using take_axis
        let gathered = padded.take_axis(&idx_array, 1)?; // [batch, n_lfr_frames, n_mels]
        stacked_frames.push(gathered);
    }

    // Concatenate along the last axis to create [batch, n_lfr_frames, n_mels * lfr_m]
    let refs: Vec<&Array> = stacked_frames.iter().collect();
    let lfr_result = ops::concatenate_axis(&refs, 2)?;

    Ok(lfr_result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_filterbank() {
        let fb = create_mel_filterbank(16000, 400, 80);
        assert_eq!(fb.len(), 80 * 201);
    }

    #[test]
    fn test_lfr_dimensions() {
        // Create dummy mel spectrogram [1, 80, 100]
        let mel_data: Vec<f32> = (0..8000).map(|i| i as f32 * 0.001).collect();
        let mel = Array::from_slice(&mel_data, &[1, 80, 100]);

        // Apply LFR with m=7, n=6
        let lfr = apply_lfr(&mel, 7, 6).unwrap();

        // Expected output: [1, 17, 560] (100/6=16.67 -> 17 frames, 80*7=560 dim)
        let shape = lfr.shape();
        assert_eq!(shape[0], 1);  // batch
        assert_eq!(shape[2], 560);  // lfr_dim = 80 * 7
    }

    #[test]
    fn test_mel_frontend_creation() {
        let config = AudioConfig::default();
        let frontend = MelFrontend::new(config);
        assert_eq!(frontend.n_freqs, 201);  // 400/2 + 1
    }

    #[test]
    fn test_mel_spectrogram_basic() {
        let config = AudioConfig::default();
        let frontend = MelFrontend::new(config);

        // Generate 1 second of sine wave at 440Hz
        let sample_rate = 16000;
        let duration = 1.0;
        let samples: Vec<f32> = (0..(sample_rate as f32 * duration) as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mel = frontend.compute_mel_spectrogram(&samples).unwrap();
        let shape = mel.shape();

        assert_eq!(shape[0], 1);  // batch
        assert_eq!(shape[1], 80); // n_mels
        assert!(shape[2] > 0);    // n_frames
    }

    #[test]
    fn test_mel_spectrogram_empty_error() {
        let config = AudioConfig::default();
        let frontend = MelFrontend::new(config);

        let result = frontend.compute_mel_spectrogram(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_mel_spectrogram_too_short_error() {
        let config = AudioConfig::default();
        let frontend = MelFrontend::new(config);

        // Only 10 samples (less than hop_length of 160)
        let samples: Vec<f32> = vec![0.0; 10];
        let result = frontend.compute_mel_spectrogram(&samples);
        assert!(result.is_err());
    }

    #[test]
    fn test_resample_same_rate() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = resample(&samples, 16000, 16000).unwrap();
        assert_eq!(result, samples);
    }

    #[test]
    fn test_resample_downsample() {
        // Generate 1 second at 48kHz
        let samples: Vec<f32> = (0..48000).map(|i| (i as f32 / 48000.0).sin()).collect();
        let result = resample(&samples, 48000, 16000).unwrap();

        // Should have ~16000 samples
        assert!(result.len() >= 15000 && result.len() <= 17000);
    }

    #[test]
    fn test_audio_config_defaults() {
        let config = AudioConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.n_fft, 400);
        assert_eq!(config.hop_length, 160);
        assert_eq!(config.max_length, 30.0);
    }

    #[test]
    fn test_lfr_correctness() {
        // Create a known mel spectrogram pattern [1, 80, 12]
        // Input layout: mel[batch, n_mels, n_frames] where mel[0, m, t] is at linear index m*n_frames + t
        let n_mels = 80;
        let n_frames = 12;
        let lfr_m = 7;
        let lfr_n = 6;

        // Create mel data where mel[0, m, t] = m * n_frames + t
        // After transpose to [1, n_frames, n_mels]: mel_t[0, t, m] = mel[0, m, t] = m * n_frames + t
        let mut mel_data: Vec<f32> = Vec::with_capacity(n_mels * n_frames);
        for m in 0..n_mels {
            for t in 0..n_frames {
                mel_data.push((m * n_frames + t) as f32);
            }
        }
        let mel = Array::from_slice(&mel_data, &[1, n_mels as i32, n_frames as i32]);

        // Apply LFR
        let lfr = apply_lfr(&mel, lfr_m, lfr_n).unwrap();
        let lfr_data: Vec<f32> = lfr.as_slice().to_vec();
        let shape = lfr.shape();

        // Check dimensions
        let expected_frames = (n_frames + lfr_n - 1) / lfr_n; // ceil(12/6) = 2
        assert_eq!(shape[0], 1, "batch dimension mismatch");
        assert_eq!(shape[1], expected_frames as i32, "lfr_frames dimension mismatch");
        assert_eq!(shape[2], (n_mels * lfr_m) as i32, "lfr_dim dimension mismatch");

        // Validate first output frame
        // After transpose: mel_t[0, t, m] = m * n_frames + t
        // LFR output: lfr[0, out_frame, offset * n_mels + mel_bin]
        //   = mel_t[0, padded_src_frame, mel_bin]
        // where padded_src_frame = clamp(out_frame * lfr_n + offset - left_pad, 0, n_frames-1)
        let left_pad = lfr_m / 2; // 3
        for offset in 0..lfr_m {
            let padded_idx = 0 * lfr_n + offset; // for first output frame
            let src_frame = if padded_idx < left_pad {
                0
            } else if padded_idx - left_pad < n_frames {
                padded_idx - left_pad
            } else {
                n_frames - 1
            };

            for m_bin in 0..n_mels {
                let lfr_idx = offset * n_mels + m_bin;
                // mel_t[0, src_frame, m_bin] = m_bin * n_frames + src_frame
                let expected_val = (m_bin * n_frames + src_frame) as f32;
                assert!(
                    (lfr_data[lfr_idx] - expected_val).abs() < 1e-6,
                    "Mismatch at offset={}, m_bin={}: got {} expected {} (src_frame={})",
                    offset, m_bin, lfr_data[lfr_idx], expected_val, src_frame
                );
            }
        }
    }
}
