//! MLX-native mel spectrogram computation for VITS training
//!
//! This module provides GPU-accelerated STFT and mel spectrogram computation
//! using MLX arrays, enabling efficient loss computation during training.

use mlx_rs::{
    array,
    error::Exception,
    ops::{concatenate_axis, indexing::IndexOp, matmul, maximum, sqrt, zeros},
    Array,
};
use std::f32::consts::PI;

/// Configuration for mel spectrogram computation
#[derive(Debug, Clone)]
pub struct MelConfig {
    /// FFT size (default: 2048)
    pub n_fft: i32,
    /// Hop length in samples (default: 640)
    pub hop_length: i32,
    /// Window length (default: 2048)
    pub win_length: i32,
    /// Sample rate (default: 32000)
    pub sample_rate: i32,
    /// Number of mel bins (default: 128)
    pub n_mels: i32,
    /// Minimum frequency for mel filterbank (default: 0.0)
    pub fmin: f32,
    /// Maximum frequency for mel filterbank (default: None = sr/2)
    pub fmax: Option<f32>,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            hop_length: 640,
            win_length: 2048,
            sample_rate: 32000,
            n_mels: 128,
            fmin: 0.0,
            fmax: None, // Will use sr/2
        }
    }
}

/// Create Hann window for STFT
fn hann_window(size: i32) -> Array {
    let mut window = vec![0.0f32; size as usize];
    for i in 0..size as usize {
        window[i] = 0.5 * (1.0 - (2.0 * PI * i as f32 / (size as f32 - 1.0)).cos());
    }
    Array::from_slice(&window, &[size])
}

/// Convert frequency to mel scale
fn hz_to_mel(freq: f32) -> f32 {
    2595.0 * (1.0 + freq / 700.0).log10()
}

/// Convert mel to frequency
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Create mel filterbank matrix
///
/// Returns Array of shape [n_mels, n_fft/2 + 1]
pub fn create_mel_filterbank(config: &MelConfig) -> Array {
    let n_freqs = (config.n_fft / 2 + 1) as usize;
    let fmax = config.fmax.unwrap_or(config.sample_rate as f32 / 2.0);

    // Mel scale edges
    let mel_min = hz_to_mel(config.fmin);
    let mel_max = hz_to_mel(fmax);

    // Create mel points (n_mels + 2 for edges)
    let n_points = config.n_mels as usize + 2;
    let mut mel_points = vec![0.0f32; n_points];
    for i in 0..n_points {
        mel_points[i] = mel_min + (mel_max - mel_min) * i as f32 / (n_points - 1) as f32;
    }

    // Convert to Hz
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert to FFT bin indices
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * config.sample_rate as f32 / config.n_fft as f32)
        .collect();

    // Create filterbank
    let mut filterbank = vec![0.0f32; config.n_mels as usize * n_freqs];

    for m in 0..config.n_mels as usize {
        let f_m_minus = hz_points[m];
        let f_m = hz_points[m + 1];
        let f_m_plus = hz_points[m + 2];

        for k in 0..n_freqs {
            let freq = fft_freqs[k];
            if freq >= f_m_minus && freq <= f_m {
                // Rising edge
                filterbank[m * n_freqs + k] = (freq - f_m_minus) / (f_m - f_m_minus);
            } else if freq >= f_m && freq <= f_m_plus {
                // Falling edge
                filterbank[m * n_freqs + k] = (f_m_plus - freq) / (f_m_plus - f_m);
            }
        }
    }

    Array::from_slice(&filterbank, &[config.n_mels, n_freqs as i32])
}

/// Compute STFT magnitude using MLX
///
/// Input: audio [batch, samples] or [samples]
/// Output: magnitude [batch, n_fft/2+1, frames] or [n_fft/2+1, frames]
///
/// Note: This is a simplified STFT that uses real FFT approximation.
/// For exact matching with torch.stft, you may need rfft support in MLX.
pub fn stft_mlx(
    audio: &Array,
    n_fft: i32,
    hop_length: i32,
    win_length: i32,
) -> Result<Array, Exception> {
    let shape = audio.shape();
    let is_batched = shape.len() == 2;

    // Flatten to 1D for processing
    let audio_1d = if is_batched {
        audio.reshape(&[-1])?
    } else {
        audio.clone()
    };

    let batch_size = if is_batched { shape[0] } else { 1 };
    let num_samples = if is_batched { shape[1] } else { shape[0] };

    // Pad audio
    let pad_left = (n_fft - hop_length) / 2;
    let pad_right = (n_fft - hop_length) / 2;
    let padded_len = num_samples + pad_left + pad_right;

    // Reflect padding (simplified: use zero padding for now)
    let padded = if pad_left > 0 || pad_right > 0 {
        let left_pad = zeros::<f32>(&[batch_size, pad_left])?;
        let right_pad = zeros::<f32>(&[batch_size, pad_right])?;
        let audio_2d = if is_batched {
            audio.clone()
        } else {
            audio.reshape(&[1, num_samples])?
        };
        concatenate_axis(&[&left_pad, &audio_2d, &right_pad], 1)?
    } else if is_batched {
        audio.clone()
    } else {
        audio.reshape(&[1, num_samples])?
    };

    // Number of frames
    let n_frames = (padded_len - n_fft) / hop_length + 1;
    let n_freqs = n_fft / 2 + 1;

    // Create window
    let window = hann_window(win_length);

    // Compute magnitude spectrum using DFT matrix approach
    // This is less efficient than FFT but works in MLX without rfft
    // For production, consider using fft.rfft when available

    // Create DFT basis (real part only for magnitude approximation)
    // This is a simplified approach - for exact matching use proper FFT
    let mut dft_real = vec![0.0f32; (n_freqs * win_length) as usize];
    let mut dft_imag = vec![0.0f32; (n_freqs * win_length) as usize];

    for k in 0..n_freqs as usize {
        for n in 0..win_length as usize {
            let angle = 2.0 * PI * k as f32 * n as f32 / n_fft as f32;
            dft_real[k * win_length as usize + n] = angle.cos();
            dft_imag[k * win_length as usize + n] = -angle.sin();
        }
    }

    let dft_real = Array::from_slice(&dft_real, &[n_freqs, win_length]);
    let dft_imag = Array::from_slice(&dft_imag, &[n_freqs, win_length]);

    // Extract frames and compute STFT
    let mut magnitudes = Vec::new();

    // Get padded audio as slice for frame extraction
    let padded_flat = padded.flatten(None, None)?;
    let padded_data: Vec<f32> = padded_flat.as_slice().to_vec();

    for b in 0..batch_size as usize {
        let mut batch_mags = Vec::new();

        for frame_idx in 0..n_frames as usize {
            let start = b * padded_len as usize + frame_idx * hop_length as usize;
            let end = start + win_length as usize;

            if end <= padded_data.len() {
                let frame_data: Vec<f32> = padded_data[start..end]
                    .iter()
                    .zip(hann_window(win_length).as_slice::<f32>().iter())
                    .map(|(x, w)| x * w)
                    .collect();

                let frame = Array::from_slice(&frame_data, &[1, win_length]);

                // DFT: X = frame @ dft^T
                let real_part = matmul(&frame, &dft_real.transpose()?)?;
                let imag_part = matmul(&frame, &dft_imag.transpose()?)?;

                // Magnitude: sqrt(real^2 + imag^2)
                let mag = sqrt(&real_part.square()?.add(&imag_part.square()?)?)?;
                batch_mags.push(mag);
            }
        }

        if !batch_mags.is_empty() {
            // Stack frames: [n_frames, 1, n_freqs] -> [n_freqs, n_frames]
            let refs: Vec<&Array> = batch_mags.iter().collect();
            let stacked = concatenate_axis(&refs, 0)?; // [n_frames, 1, n_freqs]
            let squeezed = stacked.squeeze_axes(&[1])?; // [n_frames, n_freqs]
            let transposed = squeezed.transpose()?; // [n_freqs, n_frames]
            magnitudes.push(transposed);
        }
    }

    if magnitudes.is_empty() {
        return Err(Exception::from("No frames computed in STFT"));
    }

    // Stack batches
    if batch_size == 1 {
        Ok(magnitudes.into_iter().next().unwrap())
    } else {
        let refs: Vec<&Array> = magnitudes.iter().collect();
        let stacked = concatenate_axis(&refs, 0)?;
        // Reshape to [batch, n_freqs, n_frames]
        let n_frames_actual = stacked.shape()[1];
        stacked.reshape(&[batch_size, n_freqs, n_frames_actual])
    }
}

/// Compute mel spectrogram from audio using MLX
///
/// Input: audio [batch, samples] or [samples]
/// Output: mel [batch, n_mels, frames] or [n_mels, frames]
///
/// Applies log compression: log(clamp(mel, 1e-5))
pub fn mel_spectrogram_mlx(
    audio: &Array,
    config: &MelConfig,
) -> Result<Array, Exception> {
    // Compute STFT magnitude
    let stft_mag = stft_mlx(audio, config.n_fft, config.hop_length, config.win_length)?;

    // Create mel filterbank
    let mel_basis = create_mel_filterbank(config);

    let shape = stft_mag.shape();
    let is_batched = shape.len() == 3;

    // Apply mel filterbank: [n_mels, n_freqs] @ [n_freqs, frames] = [n_mels, frames]
    let mel = if is_batched {
        // For batched: [batch, n_freqs, frames] -> process each batch
        let batch_size = shape[0];
        let n_frames = shape[2];

        let mut mels = Vec::new();
        for b in 0..batch_size as usize {
            // Extract batch: [n_freqs, frames]
            let batch_stft = stft_mag.index((b as i32, .., ..));
            let batch_mel = matmul(&mel_basis, &batch_stft)?;
            mels.push(batch_mel);
        }

        let refs: Vec<&Array> = mels.iter().collect();
        let stacked = concatenate_axis(&refs, 0)?;
        stacked.reshape(&[batch_size, config.n_mels, n_frames])?
    } else {
        // [n_freqs, frames] -> [n_mels, frames]
        matmul(&mel_basis, &stft_mag)?
    };

    // Log compression
    let mel_log = maximum(&mel, &array!(1e-5f32))?.log()?;

    Ok(mel_log)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_filterbank() {
        let config = MelConfig::default();
        let fb = create_mel_filterbank(&config);
        assert_eq!(fb.shape(), &[128, 1025]);
    }

    #[test]
    fn test_hann_window() {
        let win = hann_window(1024);
        assert_eq!(win.shape(), &[1024]);
        // Hann window should be 0 at edges, 1 at center
        let data: Vec<f32> = win.as_slice().to_vec();
        assert!(data[0].abs() < 0.01);
        assert!((data[512] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_stft_basic() {
        // Create a simple sine wave
        let sr = 32000;
        let freq = 440.0;
        let duration = 0.1;
        let n_samples = (sr as f32 * duration) as usize;

        let audio: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect();

        let audio_arr = Array::from_slice(&audio, &[n_samples as i32]);
        let config = MelConfig::default();

        let result = stft_mlx(&audio_arr, config.n_fft, config.hop_length, config.win_length);
        assert!(result.is_ok());

        let stft_mag = result.unwrap();
        // Should have shape [n_fft/2+1, frames]
        assert_eq!(stft_mag.shape()[0], config.n_fft / 2 + 1);
    }
}
