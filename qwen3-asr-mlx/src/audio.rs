//! Audio processing for Qwen3-ASR.
//!
//! WhisperFeatureExtractor compatible: 128 mel bins, n_fft=400, hop=160, 16kHz.
//! Uses Slaney mel scale with Slaney normalization and log10 + Whisper normalization.

use crate::error::{Error, Result};
use mlx_rs::Array;
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;

/// Audio processing configuration (WhisperFeatureExtractor defaults).
#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub n_mels: usize,
    pub n_fft: usize,
    pub hop_length: usize,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_mels: 128,
            n_fft: 400,
            hop_length: 160,
        }
    }
}

/// Mel spectrogram frontend with cached FFT.
pub struct MelFrontend {
    fft: Arc<dyn rustfft::Fft<f32>>,
    window: Vec<f32>,
    mel_filters: Vec<f32>, // [n_mels, n_freqs]
    config: AudioConfig,
    n_freqs: usize,
}

impl MelFrontend {
    pub fn new(config: AudioConfig) -> Self {
        let n_fft = config.n_fft;
        let n_freqs = n_fft / 2 + 1;

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);

        // Hann window (matching np.hanning)
        let window: Vec<f32> = (0..n_fft)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos())
            })
            .collect();

        let mel_filters = create_whisper_mel_filterbank(
            config.sample_rate, n_fft, config.n_mels,
        );

        Self {
            fft,
            window,
            mel_filters,
            config,
            n_freqs,
        }
    }

    /// Compute log mel spectrogram matching WhisperFeatureExtractor.
    /// Returns [n_mels, n_frames] Array.
    pub fn compute_mel_spectrogram(&self, samples: &[f32]) -> Result<Array> {
        if samples.is_empty() {
            return Err(Error::AudioFormat {
                message: "Audio samples are empty".to_string(),
            });
        }

        let n_fft = self.config.n_fft;
        let hop_length = self.config.hop_length;
        let n_mels = self.config.n_mels;

        if samples.len() < hop_length {
            return Err(Error::audio_too_short(
                (samples.len() as u64 * 1000) / self.config.sample_rate as u64,
                (hop_length as u64 * 1000) / self.config.sample_rate as u64,
            ));
        }

        let n_frames = 1 + (samples.len() - n_fft) / hop_length;
        if n_frames == 0 {
            return Err(Error::AudioFormat {
                message: format!("Audio too short for FFT: {} samples < n_fft={}", samples.len(), n_fft),
            });
        }

        let mut fft_buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n_fft];
        let mut mel_spec = vec![0.0f32; n_mels * n_frames];

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;

            for i in 0..n_fft {
                let sample = if start + i < samples.len() { samples[start + i] } else { 0.0 };
                fft_buffer[i] = Complex::new(sample * self.window[i], 0.0);
            }

            self.fft.process(&mut fft_buffer);

            // Compute mel energies: mel_filters[n_mels, n_freqs] @ magnitudes[n_freqs]
            for mel_idx in 0..n_mels {
                let mut mel_energy = 0.0f32;
                for freq_idx in 0..self.n_freqs {
                    let c = fft_buffer[freq_idx];
                    let power = c.re * c.re + c.im * c.im;
                    mel_energy += power * self.mel_filters[mel_idx * self.n_freqs + freq_idx];
                }
                // log10 (matching WhisperFeatureExtractor)
                mel_spec[mel_idx * n_frames + frame_idx] = mel_energy.max(1e-10).log10();
            }
        }

        // Whisper normalization:
        // 1. Clip to max - 8.0
        let max_val = mel_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = max_val - 8.0;
        for v in mel_spec.iter_mut() {
            *v = v.max(min_val);
        }
        // 2. Normalize: (x + 4.0) / 4.0
        for v in mel_spec.iter_mut() {
            *v = (*v + 4.0) / 4.0;
        }

        let array = Array::from_slice(&mel_spec, &[n_mels as i32, n_frames as i32]);
        Ok(array)
    }
}

/// Minimum audio duration in milliseconds.
pub const MIN_AUDIO_DURATION_MS: u64 = 100;

/// Load audio from WAV file. Returns (samples, sample_rate).
pub fn load_wav(path: impl AsRef<std::path::Path>) -> Result<(Vec<f32>, u32)> {
    use std::fs::File;
    use std::io::BufReader;

    let path = path.as_ref();
    if !path.exists() {
        return Err(Error::audio_not_found(path));
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut reader = hound::WavReader::new(reader).map_err(|e| Error::AudioFormat {
        message: format!("Failed to read WAV '{}': {}", path.display(), e),
    })?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

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

    // Stereo to mono
    let samples = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    if samples.is_empty() {
        return Err(Error::AudioFormat {
            message: format!("Audio '{}' contains no samples", path.display()),
        });
    }

    let duration_ms = (samples.len() as u64 * 1000) / sample_rate as u64;
    if duration_ms < MIN_AUDIO_DURATION_MS {
        return Err(Error::audio_too_short(duration_ms, MIN_AUDIO_DURATION_MS));
    }

    Ok((samples, sample_rate))
}

/// Resample audio to target sample rate.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    use rubato::{FftFixedInOut, Resampler};

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

    let expected_len = (samples.len() as f64 * to_rate as f64 / from_rate as f64).round() as usize;
    output.truncate(expected_len);

    Ok(output)
}

// ============================================================================
// Slaney mel scale (matching WhisperFeatureExtractor / transformers)
// ============================================================================

fn hz_to_slaney_mel(freq: f32) -> f32 {
    let f_sp: f32 = 200.0 / 3.0;
    let min_log_hz: f32 = 1000.0;
    let min_log_mel: f32 = min_log_hz / f_sp;
    let logstep: f32 = (6.4f32).ln() / 27.0;

    if freq < min_log_hz {
        freq / f_sp
    } else {
        min_log_mel + (freq / min_log_hz).ln() / logstep
    }
}

fn slaney_mel_to_hz(mel: f32) -> f32 {
    let f_sp: f32 = 200.0 / 3.0;
    let min_log_hz: f32 = 1000.0;
    let min_log_mel: f32 = min_log_hz / f_sp;
    let logstep: f32 = (6.4f32).ln() / 27.0;

    if mel < min_log_mel {
        f_sp * mel
    } else {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}

/// Create Whisper-compatible mel filterbank using Slaney mel scale + Slaney normalization.
/// Returns [n_mels, n_freqs] layout.
fn create_whisper_mel_filterbank(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let fmin = 0.0f32;
    let fmax = sample_rate as f32 / 2.0;

    // n_mels + 2 equally spaced mel points
    let mel_min = hz_to_slaney_mel(fmin);
    let mel_max = hz_to_slaney_mel(fmax);
    let mel_points: Vec<f32> = (0..=(n_mels + 1))
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let filter_freqs: Vec<f32> = mel_points.iter().map(|&m| slaney_mel_to_hz(m)).collect();

    // FFT bin frequencies (linearly spaced from 0 to fmax)
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * fmax / (n_freqs - 1) as f32)
        .collect();

    // Create triangular filters [n_freqs, n_mels]
    let mut filter_bank_transposed = vec![0.0f32; n_freqs * n_mels];

    for mel_idx in 0..n_mels {
        let lower = filter_freqs[mel_idx];
        let center = filter_freqs[mel_idx + 1];
        let upper = filter_freqs[mel_idx + 2];

        for freq_idx in 0..n_freqs {
            let freq = fft_freqs[freq_idx];
            if freq >= lower && freq <= center && center > lower {
                filter_bank_transposed[freq_idx * n_mels + mel_idx] =
                    (freq - lower) / (center - lower);
            } else if freq > center && freq <= upper && upper > center {
                filter_bank_transposed[freq_idx * n_mels + mel_idx] =
                    (upper - freq) / (upper - center);
            }
        }
    }

    // Slaney normalization: divide each filter by its bandwidth
    for mel_idx in 0..n_mels {
        let bandwidth = filter_freqs[mel_idx + 2] - filter_freqs[mel_idx];
        if bandwidth > 0.0 {
            let norm = 2.0 / bandwidth;
            for freq_idx in 0..n_freqs {
                filter_bank_transposed[freq_idx * n_mels + mel_idx] *= norm;
            }
        }
    }

    // Transpose to [n_mels, n_freqs] for efficient mel computation
    let mut mel_filters = vec![0.0f32; n_mels * n_freqs];
    for mel_idx in 0..n_mels {
        for freq_idx in 0..n_freqs {
            mel_filters[mel_idx * n_freqs + freq_idx] =
                filter_bank_transposed[freq_idx * n_mels + mel_idx];
        }
    }

    mel_filters
}
