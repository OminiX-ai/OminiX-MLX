//! Audio loading and processing utilities

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::{Error, Result};

/// Load WAV file and return samples as f32 in range [-1, 1]
///
/// Supports 16-bit, 24-bit, and 32-bit float WAV files.
/// Stereo files are automatically mixed to mono.
///
/// # Arguments
/// * `path` - Path to the WAV file
///
/// # Returns
/// Tuple of (samples, sample_rate)
pub fn load_wav(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    let file = File::open(path)?;
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

/// Resample audio using high-quality sinc interpolation
///
/// Uses windowed sinc interpolation which is mathematically optimal for
/// bandlimited signals. Falls back to linear interpolation if sinc fails.
///
/// # Arguments
/// * `samples` - Input audio samples
/// * `src_rate` - Source sample rate in Hz
/// * `target_rate` - Target sample rate in Hz (must be 16000 for Paraformer)
///
/// # Returns
/// Resampled audio samples at the target rate
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

    // Create resampler
    let mut resampler = match SincFixedIn::<f32>::new(ratio, 2.0, params, samples.len(), 1) {
        Ok(r) => r,
        Err(e) => {
            eprintln!(
                "Warning: Failed to create sinc resampler: {}. Falling back to linear.",
                e
            );
            return resample_linear(samples, src_rate, target_rate);
        }
    };

    // Process the entire input as one chunk
    let input = vec![samples.to_vec()];
    match resampler.process(&input, None) {
        Ok(output) => {
            if !output.is_empty() {
                output[0].clone()
            } else {
                resample_linear(samples, src_rate, target_rate)
            }
        }
        Err(e) => {
            eprintln!(
                "Warning: Sinc resampling failed: {}. Falling back to linear.",
                e
            );
            resample_linear(samples, src_rate, target_rate)
        }
    }
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

/// Save audio samples to a WAV file (16-bit, mono)
///
/// # Arguments
/// * `samples` - Audio samples in range [-1, 1]
/// * `sample_rate` - Sample rate in Hz
/// * `path` - Output file path
pub fn save_wav(samples: &[f32], sample_rate: u32, path: impl AsRef<Path>) -> Result<()> {
    use std::io::Write;

    let file = File::create(path)?;
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
    writer.write_all(&16u32.to_le_bytes())?; // Chunk size
    writer.write_all(&1u16.to_le_bytes())?; // PCM format
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
