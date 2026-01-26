//! Audio processing utilities for Step-Audio 2
//!
//! Provides mel spectrogram computation and audio I/O.
//!
//! Key differences from GPT-SoVITS:
//! - 128-mel filterbank (vs 704 for GPT-SoVITS v2)
//! - 16kHz sample rate
//! - True mel-scale filterbank (not raw STFT)
//!
//! Adapted from `mlx-rs-core/src/audio.rs`.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;

use mlx_rs::Array;
use mlx_rs::error::Exception;

use crate::config::AudioConfig;
use crate::error::{Error, Result};

// ============================================================================
// Audio I/O
// ============================================================================

/// Load WAV file and return samples as f32 in range [-1, 1]
pub fn load_wav(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    let file = File::open(&path)?;
    let mut reader = BufReader::new(file);

    // Read RIFF header
    let mut header = [0u8; 4];
    reader.read_exact(&mut header)?;
    if &header != b"RIFF" {
        return Err(Error::Audio("Not a RIFF file".into()));
    }

    // Skip file size
    reader.seek(SeekFrom::Current(4))?;

    // Read WAVE header
    reader.read_exact(&mut header)?;
    if &header != b"WAVE" {
        return Err(Error::Audio("Not a WAVE file".into()));
    }

    let mut sample_rate = 0u32;
    let mut bits_per_sample = 16u16;
    let mut num_channels = 1u16;
    let mut audio_data: Vec<u8> = Vec::new();

    // Read chunks
    loop {
        let mut chunk_id = [0u8; 4];
        if reader.read_exact(&mut chunk_id).is_err() {
            break;
        }

        let mut chunk_size_bytes = [0u8; 4];
        reader.read_exact(&mut chunk_size_bytes)?;
        let chunk_size = u32::from_le_bytes(chunk_size_bytes);

        match &chunk_id {
            b"fmt " => {
                let mut fmt_data = vec![0u8; chunk_size as usize];
                reader.read_exact(&mut fmt_data)?;

                // Audio format (should be 1 for PCM)
                let _audio_format = u16::from_le_bytes([fmt_data[0], fmt_data[1]]);
                num_channels = u16::from_le_bytes([fmt_data[2], fmt_data[3]]);
                sample_rate = u32::from_le_bytes([
                    fmt_data[4],
                    fmt_data[5],
                    fmt_data[6],
                    fmt_data[7],
                ]);
                bits_per_sample = u16::from_le_bytes([fmt_data[14], fmt_data[15]]);
            }
            b"data" => {
                audio_data = vec![0u8; chunk_size as usize];
                reader.read_exact(&mut audio_data)?;
                break;
            }
            _ => {
                // Skip unknown chunk
                reader.seek(SeekFrom::Current(chunk_size as i64))?;
            }
        }
    }

    // Convert to f32 samples
    let samples: Vec<f32> = match bits_per_sample {
        16 => {
            let mut samples = Vec::with_capacity(audio_data.len() / 2);
            for chunk in audio_data.chunks_exact(2) {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                samples.push(sample as f32 / 32768.0);
            }
            samples
        }
        24 => {
            let mut samples = Vec::with_capacity(audio_data.len() / 3);
            for chunk in audio_data.chunks_exact(3) {
                let sample = i32::from_le_bytes([0, chunk[0], chunk[1], chunk[2]]) >> 8;
                samples.push(sample as f32 / 8388608.0);
            }
            samples
        }
        32 => {
            let mut samples = Vec::with_capacity(audio_data.len() / 4);
            for chunk in audio_data.chunks_exact(4) {
                let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                samples.push(sample);
            }
            samples
        }
        _ => {
            return Err(Error::Audio(format!(
                "Unsupported bits per sample: {}",
                bits_per_sample
            )));
        }
    };

    // Mix to mono if stereo
    let samples = if num_channels > 1 {
        samples
            .chunks_exact(num_channels as usize)
            .map(|ch| ch.iter().sum::<f32>() / num_channels as f32)
            .collect()
    } else {
        samples
    };

    Ok((samples, sample_rate))
}

/// Save audio samples to a WAV file (16-bit PCM, mono)
pub fn save_wav(samples: &[f32], sample_rate: u32, path: impl AsRef<Path>) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);

    let num_samples = samples.len() as u32;
    let bytes_per_sample = 2u16;
    let num_channels = 1u16;
    let byte_rate = sample_rate * num_channels as u32 * bytes_per_sample as u32;
    let block_align = num_channels * bytes_per_sample;
    let data_size = num_samples * bytes_per_sample as u32;
    let file_size = 36 + data_size;

    // RIFF header
    writer.write_all(b"RIFF")?;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // fmt chunk
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&num_channels.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&(bytes_per_sample * 8).to_le_bytes())?;

    // data chunk
    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;

    // Write samples as 16-bit PCM
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let pcm = (clamped * 32767.0) as i16;
        writer.write_all(&pcm.to_le_bytes())?;
    }

    Ok(())
}

// ============================================================================
// Resampling
// ============================================================================

/// Resample audio using high-quality sinc interpolation
pub fn resample(samples: &[f32], src_rate: u32, target_rate: u32) -> Vec<f32> {
    if src_rate == target_rate || samples.is_empty() {
        return samples.to_vec();
    }

    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    // High-quality sinc interpolation parameters
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Cubic,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = target_rate as f64 / src_rate as f64;
    let chunk_size = 4096.min(samples.len());

    // Create resampler
    let mut resampler = match SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, 1) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Warning: Failed to create sinc resampler: {}. Falling back to linear.", e);
            return resample_linear(samples, src_rate, target_rate);
        }
    };

    let mut output = Vec::with_capacity((samples.len() as f64 * ratio).ceil() as usize + 100);
    let mut pos = 0;

    // Process full chunks
    while pos + chunk_size <= samples.len() {
        let chunk = vec![samples[pos..pos + chunk_size].to_vec()];
        match resampler.process(&chunk, None) {
            Ok(out) => {
                if !out.is_empty() {
                    output.extend_from_slice(&out[0]);
                }
            }
            Err(e) => {
                eprintln!("Warning: Sinc resampling failed: {}. Falling back to linear.", e);
                return resample_linear(samples, src_rate, target_rate);
            }
        }
        pos += chunk_size;
    }

    // Process remaining samples
    if pos < samples.len() {
        let remaining = samples.len() - pos;
        let mut padded = vec![0.0f32; chunk_size];
        padded[..remaining].copy_from_slice(&samples[pos..]);

        let chunk = vec![padded];
        if let Ok(out) = resampler.process_partial(Some(&chunk), None) {
            if !out.is_empty() {
                let expected_out = (remaining as f64 * ratio).ceil() as usize;
                let take = expected_out.min(out[0].len());
                output.extend_from_slice(&out[0][..take]);
            }
        }
    }

    // Trim to expected length
    let expected_len = (samples.len() as f64 * ratio).round() as usize;
    if output.len() > expected_len {
        output.truncate(expected_len);
    }

    output
}

/// Fallback linear interpolation resampler
fn resample_linear(samples: &[f32], src_rate: u32, target_rate: u32) -> Vec<f32> {
    if src_rate == target_rate {
        return samples.to_vec();
    }

    let ratio = src_rate as f64 / target_rate as f64;
    let out_len = (samples.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_idx = i as f64 * ratio;
        let idx_floor = src_idx.floor() as usize;
        let idx_ceil = (idx_floor + 1).min(samples.len() - 1);
        let frac = (src_idx - idx_floor as f64) as f32;

        let sample = samples[idx_floor] * (1.0 - frac) + samples[idx_ceil] * frac;
        output.push(sample);
    }

    output
}

// ============================================================================
// Mel Spectrogram
// ============================================================================

/// Create Hann window
fn hann_window(size: usize) -> Vec<f32> {
    let mut window = Vec::with_capacity(size);
    for i in 0..size {
        let t = i as f32 / (size - 1) as f32;
        window.push(0.5 - 0.5 * (2.0 * std::f32::consts::PI * t).cos());
    }
    window
}

/// Convert frequency to mel scale
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale to frequency
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Create mel filterbank matrix
/// Returns [n_mels, n_freqs] matrix
fn mel_filterbank(n_fft: i32, n_mels: i32, sample_rate: i32, fmin: f32, fmax: f32) -> Vec<f32> {
    let n_freqs = (n_fft / 2 + 1) as usize;

    // Mel points
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let mut mel_points = Vec::with_capacity(n_mels as usize + 2);
    for i in 0..=(n_mels + 1) as usize {
        let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
        mel_points.push(mel_to_hz(mel));
    }

    // Convert to FFT bins
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
        .collect();

    // Create filterbank [n_mels, n_freqs]
    let mut filterbank = vec![0.0f32; n_mels as usize * n_freqs];

    for m in 0..n_mels as usize {
        let f_left = mel_points[m];
        let f_center = mel_points[m + 1];
        let f_right = mel_points[m + 2];

        for k in 0..n_freqs {
            let freq = fft_freqs[k];

            if freq >= f_left && freq <= f_center {
                filterbank[m * n_freqs + k] = (freq - f_left) / (f_center - f_left);
            } else if freq > f_center && freq <= f_right {
                filterbank[m * n_freqs + k] = (f_right - freq) / (f_right - f_center);
            }
        }
    }

    filterbank
}

// stft_magnitude function removed - replaced by stft_power_spectrum for Step-Audio 2

/// Compute 128-mel spectrogram from audio samples (Step-Audio 2 format)
///
/// Returns Array with shape [1, n_mels, n_frames] (NCL format for Conv1d)
///
/// This matches the Python implementation:
/// - Uses squared magnitudes (power spectrum)
/// - Uses log10 (not ln)
/// - Applies Step-Audio specific normalization: (log + 4.0) / 4.0
/// - Adds 479 samples padding at end
pub fn compute_mel_spectrogram(samples: &[f32], config: &AudioConfig) -> std::result::Result<Array, Exception> {
    let n_fft = config.n_fft;
    let hop_length = config.hop_length;
    let n_mels = config.n_mels;
    let sample_rate = config.sample_rate;
    let fmin = config.fmin;
    let fmax = config.fmax.unwrap_or(sample_rate as f32 / 2.0);

    let n_freqs = (n_fft / 2 + 1) as usize;

    // Add Step-Audio specific padding (479 samples at end)
    let padding = 479;
    let mut padded_samples = samples.to_vec();
    padded_samples.extend(vec![0.0f32; padding]);

    // Compute STFT squared magnitude (power spectrum) [n_freqs, n_frames]
    let stft_power = stft_power_spectrum(&padded_samples, n_fft, hop_length);
    let n_frames = stft_power.len() / n_freqs;

    if n_frames == 0 {
        return Ok(Array::zeros::<f32>(&[1, n_mels, 1])?);
    }

    // Create mel filterbank [n_mels, n_freqs]
    let filterbank = mel_filterbank(n_fft, n_mels, sample_rate, fmin, fmax);

    // Apply mel filterbank: [n_mels, n_freqs] @ [n_freqs, n_frames] -> [n_mels, n_frames]
    let mut mel_spec = vec![0.0f32; n_mels as usize * n_frames];

    for m in 0..n_mels as usize {
        for t in 0..n_frames {
            let mut sum = 0.0f32;
            for f in 0..n_freqs {
                sum += filterbank[m * n_freqs + f] * stft_power[f * n_frames + t];
            }
            mel_spec[m * n_frames + t] = sum;
        }
    }

    // Apply log10 compression with clamping (Step-Audio specific)
    let epsilon = 1e-10f32;
    for v in &mut mel_spec {
        *v = (*v).max(epsilon).log10();
    }

    // Find max for normalization
    let max_val = mel_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Apply Step-Audio normalization: clamp to max-8, then (x + 4) / 4
    for v in &mut mel_spec {
        *v = (*v).max(max_val - 8.0);
        *v = (*v + 4.0) / 4.0;
    }

    // Create Array [1, n_mels, n_frames]
    let mel_array = Array::from_slice(&mel_spec, &[1, n_mels, n_frames as i32]);

    Ok(mel_array)
}

/// Compute STFT power spectrum (squared magnitude)
/// Returns [n_freqs, n_frames] where n_freqs = n_fft/2 + 1
fn stft_power_spectrum(samples: &[f32], n_fft: i32, hop_length: i32) -> Vec<f32> {
    use std::f32::consts::PI;

    let n_fft = n_fft as usize;
    let hop_length = hop_length as usize;
    let n_freqs = n_fft / 2 + 1;

    // Create Hann window
    let window = hann_window(n_fft);

    // For Step-Audio, we don't use center padding (unlike librosa default)
    // Just process the signal as-is
    let n_frames = if samples.len() >= n_fft {
        (samples.len() - n_fft) / hop_length + 1
    } else {
        0
    };

    if n_frames == 0 {
        return vec![0.0f32; n_freqs];
    }

    // Output power spectrogram [n_freqs, n_frames]
    // Note: Python STFT does [:-1] which removes the last frame, so we do the same
    let effective_frames = n_frames.saturating_sub(1).max(1);
    let mut power = vec![0.0f32; n_freqs * effective_frames];

    for frame in 0..effective_frames {
        let start = frame * hop_length;

        // Apply window
        let mut windowed = vec![0.0f32; n_fft];
        for i in 0..n_fft {
            if start + i < samples.len() {
                windowed[i] = samples[start + i] * window[i];
            }
        }

        // DFT to compute power spectrum (squared magnitude)
        for k in 0..n_freqs {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for n in 0..n_fft {
                let angle = 2.0 * PI * k as f32 * n as f32 / n_fft as f32;
                real += windowed[n] * angle.cos();
                imag -= windowed[n] * angle.sin();
            }

            // Squared magnitude (power spectrum)
            power[k * effective_frames + frame] = real * real + imag * imag;
        }
    }

    power
}

/// Maximum audio duration in seconds for Step-Audio 2
/// Based on max context length of 1500 mel frames with hop_length=160 at 16kHz
pub const MAX_AUDIO_DURATION_SECS: f32 = 15.0;

/// Load audio and compute mel spectrogram for Step-Audio 2
///
/// Returns Array with shape [1, 128, n_frames]
/// Audio is truncated to MAX_AUDIO_DURATION_SECS if longer
pub fn load_audio_mel(path: impl AsRef<Path>, config: &AudioConfig) -> Result<Array> {
    // Load WAV
    let (samples, src_rate) = load_wav(&path)?;

    // Resample to 16kHz if needed
    let samples = if src_rate != config.sample_rate as u32 {
        resample(&samples, src_rate, config.sample_rate as u32)
    } else {
        samples
    };

    // Truncate to max duration (Step-Audio 2 has 1500 frame max context)
    let max_samples = (MAX_AUDIO_DURATION_SECS * config.sample_rate as f32) as usize;
    let samples = if samples.len() > max_samples {
        eprintln!(
            "Warning: Audio truncated from {:.2}s to {:.2}s (max context limit)",
            samples.len() as f32 / config.sample_rate as f32,
            MAX_AUDIO_DURATION_SECS
        );
        samples[..max_samples].to_vec()
    } else {
        samples
    };

    // Compute mel spectrogram
    let mel = compute_mel_spectrogram(&samples, config)?;

    Ok(mel)
}

/// Load audio samples for processing
///
/// Returns (samples, sample_rate) resampled to target rate
pub fn load_audio_samples(path: impl AsRef<Path>, target_rate: u32) -> Result<Vec<f32>> {
    let (samples, src_rate) = load_wav(&path)?;

    let samples = if src_rate != target_rate {
        resample(&samples, src_rate, target_rate)
    } else {
        samples
    };

    Ok(samples)
}

/// Convert raw audio samples to mel spectrogram
///
/// Resamples to 16kHz if needed and computes 128-mel spectrogram.
pub fn samples_to_mel(samples: &[f32], sample_rate: u32, config: &AudioConfig) -> Result<Array> {
    // Resample if needed
    let samples = if sample_rate != config.sample_rate as u32 {
        resample(samples, sample_rate, config.sample_rate as u32)
    } else {
        samples.to_vec()
    };

    // Compute mel spectrogram
    let mel = compute_mel_spectrogram(&samples, config)?;

    Ok(mel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let window = hann_window(256);
        assert_eq!(window.len(), 256);
        assert!((window[0]).abs() < 1e-6); // Start at 0
        assert!((window[127] - 1.0).abs() < 0.01); // Peak near middle
    }

    #[test]
    fn test_hz_to_mel() {
        assert!((hz_to_mel(0.0)).abs() < 1e-6);
        assert!((hz_to_mel(1000.0) - 1000.0).abs() < 50.0);
    }

    #[test]
    fn test_mel_filterbank() {
        let fb = mel_filterbank(400, 128, 16000, 0.0, 8000.0);
        assert_eq!(fb.len(), 128 * 201); // n_mels * n_freqs
    }

    #[test]
    fn test_compute_mel_spectrogram() {
        // Create a simple sine wave
        let sample_rate = 16000;
        let duration = 1.0; // 1 second
        let samples: Vec<f32> = (0..(sample_rate as f32 * duration) as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let config = AudioConfig::default();
        let mel = compute_mel_spectrogram(&samples, &config);
        assert!(mel.is_ok());

        let mel = mel.unwrap();
        assert_eq!(mel.shape()[0], 1);
        assert_eq!(mel.shape()[1], 128);
        assert!(mel.shape()[2] > 0);
    }

    #[test]
    fn test_resample() {
        // Use a larger sample to ensure the resampler can work properly
        let samples: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.1).sin()).collect();
        let resampled = resample(&samples, 16000, 32000);
        // Output should be roughly 2x length (within some tolerance due to resampler behavior)
        let expected_len = (samples.len() as f64 * 2.0).round() as usize;
        // Allow for some variance in output length (within 5%)
        assert!(resampled.len() > samples.len() || (resampled.len() as f64 / expected_len as f64 > 0.95));
    }
}
