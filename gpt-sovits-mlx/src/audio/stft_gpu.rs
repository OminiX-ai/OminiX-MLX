//! GPU-accelerated STFT using MLX's rfft
//!
//! This provides O(N log N) STFT computation on GPU, replacing the
//! O(N²) DFT matrix approach.

use mlx_rs::{
    error::Exception,
    fft::rfft,
    ops::{abs, concatenate_axis, indexing::IndexOp, zeros},
    Array,
};
use std::f32::consts::PI;

/// Create Hann window as MLX Array
fn hann_window_mlx(size: i32) -> Array {
    let mut window = vec![0.0f32; size as usize];
    for i in 0..size as usize {
        window[i] = 0.5 * (1.0 - (2.0 * PI * i as f32 / (size as f32 - 1.0)).cos());
    }
    Array::from_slice(&window, &[size])
}

/// GPU-accelerated STFT using MLX rfft
///
/// This uses O(N log N) FFT instead of O(N²) DFT, providing ~100x speedup
/// for typical audio lengths.
///
/// Input: audio [samples] (1D)
/// Output: magnitude [n_fft/2+1, frames]
pub fn stft_rfft(
    audio: &Array,
    n_fft: i32,
    hop_length: i32,
    win_length: i32,
) -> Result<Array, Exception> {
    let num_samples = audio.dim(0) as i32;

    // Center padding (like librosa center=True)
    let pad_length = n_fft / 2;
    let left_pad = zeros::<f32>(&[pad_length])?;
    let right_pad = zeros::<f32>(&[pad_length])?;
    let padded = concatenate_axis(&[&left_pad, audio, &right_pad], 0)?;
    let padded_len = padded.dim(0) as i32;

    // Number of frames
    let n_frames = (padded_len - n_fft) / hop_length + 1;
    let n_freqs = n_fft / 2 + 1;

    // Create window
    let window = hann_window_mlx(win_length);

    // Extract all frames at once using unfold-like operation
    // For efficiency, we'll process frames in a batch

    // Build frame tensor: [n_frames, n_fft]
    let padded_data: Vec<f32> = padded.as_slice().to_vec();
    let mut frames_data = vec![0.0f32; (n_frames * n_fft) as usize];

    for frame_idx in 0..n_frames as usize {
        let start = frame_idx * hop_length as usize;
        for i in 0..win_length as usize {
            if start + i < padded_data.len() {
                // Apply window
                let win_val: f32 = window.as_slice::<f32>()[i];
                frames_data[frame_idx * n_fft as usize + i] = padded_data[start + i] * win_val;
            }
        }
        // Zero-pad if win_length < n_fft (already zero-initialized)
    }

    // Create frames tensor [n_frames, n_fft]
    let frames = Array::from_slice(&frames_data, &[n_frames, n_fft]);

    // Apply rfft to all frames at once (GPU accelerated!)
    // rfft on axis=-1 (last axis): [n_frames, n_fft] -> [n_frames, n_fft/2+1] complex
    let fft_result = rfft(&frames, n_fft, -1)?;

    // Compute magnitude: |complex| = sqrt(real² + imag²)
    // MLX abs on complex returns magnitude
    let magnitude = abs(&fft_result)?;

    // Transpose to [n_freqs, n_frames]
    let magnitude_t = magnitude.transpose()?;

    Ok(magnitude_t)
}

/// GPU-accelerated STFT for reference mel computation
///
/// Returns [1, n_bins, frames] where n_bins = min(704, n_fft/2+1)
/// This is the format expected by GPT-SoVITS v2 ref_enc.
pub fn stft_rfft_for_reference(
    audio: &Array,
    n_fft: i32,
    hop_length: i32,
    win_length: i32,
    n_bins: i32,  // Usually 704 for v2
) -> Result<Array, Exception> {
    // Compute full STFT
    let stft_mag = stft_rfft(audio, n_fft, hop_length, win_length)?;

    // stft_mag shape: [n_freqs, frames]
    let n_freqs = stft_mag.dim(0) as i32;
    let n_frames = stft_mag.dim(1) as i32;

    // Take first n_bins frequency bins
    let n_bins_actual = n_bins.min(n_freqs);
    let spec = stft_mag.index((..n_bins_actual, ..));

    // Reshape to [1, n_bins, frames]
    spec.reshape(&[1, n_bins_actual, n_frames])
}

/// Load reference audio and compute spectrogram using GPU FFT
///
/// This is a drop-in replacement for `mlx_rs_core::audio::load_reference_mel`
/// that uses GPU-accelerated FFT instead of naive CPU DFT.
///
/// Returns Array with shape [1, n_bins, n_frames] (NCL format)
pub fn load_reference_mel_gpu(
    path: impl AsRef<std::path::Path>,
    n_fft: i32,
    hop_length: i32,
    win_length: i32,
    n_bins: i32,  // Usually 704 for GPT-SoVITS v2
    target_sample_rate: u32,
) -> Result<Array, Box<dyn std::error::Error>> {
    use mlx_rs_core::audio::{load_wav, resample};

    // Load WAV
    let (samples, src_rate) = load_wav(&path)?;

    // Resample if needed
    let samples = if src_rate != target_sample_rate {
        resample(&samples, src_rate, target_sample_rate)
    } else {
        samples
    };

    // Normalize (match Python: if maxx > 1: audio /= min(2, maxx))
    let max_val = samples.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let samples: Vec<f32> = if max_val > 1.0 {
        let scale = max_val.min(2.0);
        samples.iter().map(|x| x / scale).collect()
    } else {
        samples
    };

    // Convert to MLX Array
    let audio = Array::from_slice(&samples, &[samples.len() as i32]);

    // Compute STFT using GPU FFT
    let spec = stft_rfft_for_reference(&audio, n_fft, hop_length, win_length, n_bins)?;

    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stft_rfft_basic() {
        // Create a simple sine wave
        let sr = 32000;
        let freq = 440.0f32;
        let duration = 0.5;  // 0.5 seconds
        let n_samples = (sr as f32 * duration) as usize;

        let audio: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sr as f32).sin())
            .collect();

        let audio_arr = Array::from_slice(&audio, &[n_samples as i32]);

        let n_fft = 2048;
        let hop_length = 640;
        let win_length = 2048;

        let result = stft_rfft(&audio_arr, n_fft, hop_length, win_length);
        assert!(result.is_ok(), "STFT should succeed");

        let stft_mag = result.unwrap();

        // Should have shape [n_fft/2+1, frames]
        assert_eq!(stft_mag.dim(0) as i32, n_fft / 2 + 1);
        println!("STFT shape: {:?}", stft_mag.shape());

        // The peak should be near bin 440 * 2048 / 32000 ≈ 28
        let expected_bin = (freq * n_fft as f32 / sr as f32).round() as usize;
        println!("Expected peak at bin ~{}", expected_bin);
    }

    #[test]
    fn test_stft_rfft_performance() {
        // Test with ~10 seconds of audio (typical reference)
        let sr = 32000;
        let duration = 10.0;
        let n_samples = (sr as f32 * duration) as usize;

        // Random-ish audio
        let audio: Vec<f32> = (0..n_samples)
            .map(|i| ((i as f32 * 0.01).sin() + (i as f32 * 0.023).cos()) * 0.5)
            .collect();

        let audio_arr = Array::from_slice(&audio, &[n_samples as i32]);

        let start = std::time::Instant::now();
        let result = stft_rfft(&audio_arr, 2048, 640, 2048);
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        println!("STFT on {} samples took {:?}", n_samples, elapsed);

        // Should be much faster than 3.7s (the current DFT time)
        // Expect < 100ms with GPU FFT
    }
}
