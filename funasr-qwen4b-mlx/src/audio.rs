//! Audio processing utilities for FunASR-Qwen4B.
//!
//! This module provides:
//! - Audio file loading (WAV)
//! - Resampling to 16kHz
//! - Mel spectrogram computation with FFT (O(n log n) instead of O(n²) DFT)
//! - MLX GPU-accelerated mel spectrogram (MelFrontendMLX)

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
    /// Window size in samples (25ms at 16kHz = 400)
    pub window_size: usize,
    /// FFT size (next power of 2 from window_size, Kaldi convention)
    pub n_fft: usize,
    /// Hop length between frames
    pub hop_length: usize,
    /// Maximum audio length in seconds
    pub max_length: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_mels: 80,
            window_size: 400,  // 25ms at 16kHz
            n_fft: 512,        // next power of 2 (Kaldi convention)
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
        let window_size = config.window_size;
        let n_freqs = n_fft / 2 + 1;

        // Create cached FFT planner
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);

        // Pre-compute Hamming window for window_size samples (matches FunASR SenseVoice)
        // Kaldi: window applied to window_size samples, zero-padded to n_fft
        let window: Vec<f32> = (0..window_size)
            .map(|i| {
                0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (window_size - 1) as f32).cos()
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
        let audio_slice = if samples.len() > max_samples {
            &samples[..max_samples]
        } else {
            samples
        };

        // Audio is already normalized to [-1, 1], use as-is
        // NO 32768 scaling (FunASR SenseVoice doesn't use this)
        // NO dithering (training didn't use it)
        // NO pre-emphasis (FunASR SenseVoice doesn't use this for inference)
        let audio: Vec<f32> = audio_slice.to_vec();

        let window_size = self.window.len();

        // Compute number of frames (Kaldi snip_edges=True)
        let n_frames = if audio.len() >= window_size {
            1 + (audio.len() - window_size) / hop_length
        } else {
            1
        };

        // Pre-allocate FFT buffer (reused for each frame)
        let mut fft_buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n_fft];
        let mut mel_spec = vec![0.0f32; n_mels * n_frames];

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;

            // Zero the FFT buffer (for zero-padding beyond window_size)
            for b in fft_buffer.iter_mut() {
                *b = Complex::new(0.0, 0.0);
            }

            // Apply window to window_size samples, zero-pad rest to n_fft
            for i in 0..window_size {
                let sample = if start + i < audio.len() { audio[start + i] } else { 0.0 };
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
                // Log mel spectrogram (Kaldi uses FLT_EPSILON as floor)
                mel_spec[mel_idx * n_frames + frame_idx] = (mel_energy.max(f32::EPSILON)).ln();
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

// ============================================================================
// MLX GPU-Accelerated Mel Spectrogram
// ============================================================================

/// MLX GPU-accelerated mel spectrogram frontend.
///
/// Uses MLX's rfft for GPU-parallel FFT computation instead of CPU-based rustfft.
/// This provides significant speedup for longer audio or batch processing.
pub struct MelFrontendMLX {
    /// Pre-computed Hann window as MLX array
    window: Array,
    /// Pre-computed mel filterbank [n_mels, n_freqs] as MLX array
    mel_filters: Array,
    /// Configuration
    config: AudioConfig,
    /// Number of frequency bins (n_fft/2 + 1)
    n_freqs: usize,
}

impl MelFrontendMLX {
    /// Create a new MLX-based MelFrontend with the given configuration.
    pub fn new(config: AudioConfig) -> Result<Self> {
        let n_fft = config.n_fft;
        let n_freqs = n_fft / 2 + 1;
        let n_mels = config.n_mels;

        let window_size = config.window_size;

        // Pre-compute Hamming window for window_size samples (matches FunASR/Kaldi)
        // Window is applied to window_size samples, then zero-padded to n_fft
        let mut window_data = vec![0.0f32; n_fft];
        for i in 0..window_size {
            window_data[i] = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (window_size - 1) as f32).cos();
        }
        // Remaining n_fft - window_size entries stay 0 (implicit zero-padding)
        let window = Array::from_slice(&window_data, &[n_fft as i32]);

        // Pre-compute mel filterbank as MLX array [n_mels, n_freqs]
        let mel_filters_data = create_mel_filterbank(config.sample_rate, n_fft, n_mels);
        let mel_filters = Array::from_slice(&mel_filters_data, &[n_mels as i32, n_freqs as i32]);

        Ok(Self {
            window,
            mel_filters,
            config,
            n_freqs,
        })
    }

    /// Compute mel spectrogram from audio samples using MLX GPU-accelerated FFT.
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

        // Truncate or use all samples
        let max_samples = (self.config.max_length * self.config.sample_rate as f32) as usize;
        let audio_len = samples.len().min(max_samples);

        // FunASR SenseVoice preprocessing:
        // - Audio is already normalized to [-1, 1], use as-is
        // - NO 32768 scaling (FunASR SenseVoice doesn't use this)
        // - NO dithering (training didn't use it)
        // - NO pre-emphasis (FunASR SenseVoice doesn't use this for inference)
        let audio: Vec<f32> = samples[..audio_len].to_vec();

        let window_size = self.config.window_size;

        // Compute number of frames (Kaldi snip_edges=True)
        // n_frames = 1 + (signal_length - frame_length) / frame_shift
        let n_frames = if audio.len() >= window_size {
            1 + (audio.len() - window_size) / hop_length
        } else {
            1
        };

        // Create frames matrix [n_frames, n_fft] - batch all frames together
        // Each frame: window_size samples extracted, zero-padded to n_fft (Kaldi convention)
        let mut frames_data = vec![0.0f32; n_frames * n_fft];
        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;
            for i in 0..window_size {
                let sample_idx = start + i;
                if sample_idx < audio.len() {
                    frames_data[frame_idx * n_fft + i] = audio[sample_idx];
                }
            }
            // Positions window_size..n_fft remain zero (zero-padding)
        }

        // Create MLX array for frames [n_frames, n_fft]
        let frames = Array::from_slice(&frames_data, &[n_frames as i32, n_fft as i32]);

        // Apply window: frames * window (broadcasting)
        let windowed = mlx_rs::ops::multiply(&frames, &self.window)?;

        // GPU-accelerated FFT using MLX rfft
        // rfft returns [n_frames, n_fft/2 + 1] complex values
        let spectrum = mlx_rs::fft::rfft(&windowed, n_fft as i32, -1)?;

        // Compute power spectrum: |fft|^2 = re^2 + im^2
        // MLX complex arrays: use abs() then square
        let magnitude = mlx_rs::ops::abs(&spectrum)?;
        let power = mlx_rs::ops::square(&magnitude)?;

        // Apply mel filterbank: [n_mels, n_freqs] @ [n_frames, n_freqs].T
        // = [n_mels, n_freqs] @ [n_freqs, n_frames] = [n_mels, n_frames]
        let power_t = power.transpose_axes(&[1, 0])?;  // [n_freqs, n_frames]
        let mel_spec = mlx_rs::ops::matmul(&self.mel_filters, &power_t)?;  // [n_mels, n_frames]

        // Apply log: log(max(mel_spec, FLT_EPSILON))
        // Kaldi uses FLT_EPSILON (~1.19e-7) as the energy floor
        let floor = Array::from_slice(&[f32::EPSILON], &[1]);
        let mel_spec_floored = mlx_rs::ops::maximum(&mel_spec, &floor)?;
        let log_mel = mlx_rs::ops::log(&mel_spec_floored)?;

        // Reshape to [1, n_mels, n_frames]
        let output = log_mel.reshape(&[1, n_mels as i32, n_frames as i32])?;

        // Evaluate to ensure computation is done
        mlx_rs::transforms::eval([&output])?;

        Ok(output)
    }
}

/// Compute mel spectrogram using MLX GPU-accelerated FFT.
///
/// Returns a tensor of shape [1, n_mels, n_frames].
///
/// This is faster than the CPU version for longer audio or batch processing.
pub fn compute_mel_spectrogram_mlx(samples: &[f32], config: &AudioConfig) -> Result<Array> {
    let frontend = MelFrontendMLX::new(config.clone())?;
    frontend.compute_mel_spectrogram(samples)
}

/// Create mel filterbank matrix.
fn create_mel_filterbank(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;

    // Mel scale conversion
    let hz_to_mel = |hz: f32| -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() };
    let mel_to_hz = |mel: f32| -> f32 { 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0) };

    // Kaldi defaults: low_freq=20, high_freq=Nyquist
    let mel_low = hz_to_mel(20.0);
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

/// Apply CMVN (Cepstral Mean Variance Normalization) to LFR features.
///
/// Loads normalization stats from safetensors file and applies:
/// normalized = (features + shift) * rescale
///
/// Input: [batch, n_frames, 560] (LFR features)
/// Output: [batch, n_frames, 560] (normalized features)
pub fn apply_cmvn(features: &Array, cmvn_path: &str) -> Result<Array> {
    // Load CMVN stats
    let cmvn = Array::load_safetensors(cmvn_path)
        .map_err(|e| Error::Audio(format!("Failed to load CMVN: {:?}", e)))?;

    let cmvn: std::collections::HashMap<String, Array> = cmvn.into_iter().collect();

    let shift = cmvn.get("shift")
        .ok_or_else(|| Error::Audio("CMVN missing 'shift'".to_string()))?;
    let rescale = cmvn.get("rescale")
        .ok_or_else(|| Error::Audio("CMVN missing 'rescale'".to_string()))?;

    // Apply: normalized = (features + shift) * rescale
    let shifted = mlx_rs::ops::add(features, shift)?;
    let normalized = mlx_rs::ops::multiply(&shifted, rescale)?;

    Ok(normalized)
}

/// Apply Low Frame Rate (LFR) transformation.
///
/// LFR stacks `m` consecutive frames and subsamples by factor `n`.
/// This reduces the sequence length while preserving temporal information.
///
/// Input: [batch, n_mels, n_frames]
/// Output: [batch, n_frames/n, n_mels * m]
///
/// For FunASR-Qwen4B: m=7, n=6, producing 560-dim features (80 * 7).
pub fn apply_lfr(mel: &Array, lfr_m: usize, lfr_n: usize) -> Result<Array> {
    let shape = mel.shape();
    let batch = shape[0] as usize;
    let n_mels = shape[1] as usize;
    let n_frames = shape[2] as usize;

    // Calculate output dimensions
    let n_lfr_frames = (n_frames + lfr_n - 1) / lfr_n;
    let lfr_dim = n_mels * lfr_m;

    // Get data from mel array
    mlx_rs::transforms::eval([mel])?;

    // Transpose to [batch, n_frames, n_mels] for easier processing
    let mel_transposed = mel.transpose_axes(&[0, 2, 1])?;
    // Make contiguous for as_slice to work (transpose creates strided view)
    let mel_contiguous = mlx_rs::ops::contiguous(&mel_transposed)?;
    mlx_rs::transforms::eval([&mel_contiguous])?;

    // Get the mel data as a slice
    let mel_data: Vec<f32> = mel_contiguous.try_as_slice::<f32>()
        .map_err(|e| Error::Audio(format!("Failed to get mel data: {}", e)))?
        .to_vec();

    // Create output buffer
    let mut lfr_data = vec![0.0f32; batch * n_lfr_frames * lfr_dim];

    // Process each batch
    for b in 0..batch {
        for out_frame in 0..n_lfr_frames {
            let center_frame = out_frame * lfr_n;

            // Stack lfr_m frames centered around center_frame
            for m in 0..lfr_m {
                // Calculate source frame index with padding
                let src_frame = if m < lfr_m / 2 {
                    // Left padding: use first frame
                    let offset = (lfr_m / 2) - m;
                    if offset > center_frame {
                        0
                    } else {
                        center_frame - offset
                    }
                } else {
                    // Right: use frame or pad with last
                    let offset = m - (lfr_m / 2);
                    (center_frame + offset).min(n_frames - 1)
                };

                // Copy mel features for this frame
                let src_idx = b * n_frames * n_mels + src_frame * n_mels;
                let dst_idx = b * n_lfr_frames * lfr_dim + out_frame * lfr_dim + m * n_mels;

                for i in 0..n_mels {
                    if src_idx + i < mel_data.len() && dst_idx + i < lfr_data.len() {
                        lfr_data[dst_idx + i] = mel_data[src_idx + i];
                    }
                }
            }
        }
    }

    // Create output array [batch, n_lfr_frames, lfr_dim]
    let lfr_array = Array::from_slice(&lfr_data, &[batch as i32, n_lfr_frames as i32, lfr_dim as i32]);

    Ok(lfr_array)
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
        assert_eq!(frontend.n_freqs, 257);  // 512/2 + 1
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
        assert_eq!(config.n_fft, 512);
        assert_eq!(config.hop_length, 160);
        assert_eq!(config.max_length, 30.0);
    }
}
