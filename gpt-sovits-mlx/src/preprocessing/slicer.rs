//! Audio Slicer - Split long audio into shorter segments based on silence detection
//!
//! Port of GPT-SoVITS slicer2.py to Rust

use std::path::{Path, PathBuf};
use super::{PreprocessingError, Result};
use tracing::{debug, info};

/// Configuration for audio slicing
#[derive(Debug, Clone)]
pub struct SlicerConfig {
    /// Sample rate for processing (default: 32000)
    pub sample_rate: u32,
    /// Volume threshold in dB for silence detection (default: -40.0)
    pub threshold_db: f32,
    /// Minimum length of each chunk in milliseconds (default: 5000)
    pub min_length_ms: u32,
    /// Minimum interval of silence to trigger a slice in milliseconds (default: 300)
    pub min_interval_ms: u32,
    /// Hop size for RMS calculation in milliseconds (default: 20)
    pub hop_size_ms: u32,
    /// Maximum silence to keep around sliced clips in milliseconds (default: 1000)
    pub max_sil_kept_ms: u32,
    /// Maximum amplitude for normalization (default: 0.9)
    pub max_amplitude: f32,
    /// Alpha for amplitude mixing (default: 0.25)
    pub alpha: f32,
}

impl Default for SlicerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 32000,
            threshold_db: -40.0,
            min_length_ms: 5000,
            min_interval_ms: 300,
            hop_size_ms: 20,
            max_sil_kept_ms: 1000,
            max_amplitude: 0.9,
            alpha: 0.25,
        }
    }
}

/// A sliced audio chunk
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// The audio samples
    pub samples: Vec<f32>,
    /// Start time in milliseconds (from original file)
    pub start_ms: u64,
    /// End time in milliseconds (from original file)
    pub end_ms: u64,
    /// Output file path (if saved)
    pub output_path: PathBuf,
}

/// Audio slicer based on silence detection
pub struct AudioSlicer {
    config: SlicerConfig,
    /// Threshold as linear amplitude
    threshold: f32,
    /// Hop size in samples
    hop_size: usize,
    /// Window size in samples
    win_size: usize,
    /// Minimum length in frames
    min_length: usize,
    /// Minimum interval in frames
    min_interval: usize,
    /// Maximum silence kept in frames
    max_sil_kept: usize,
}

impl AudioSlicer {
    /// Create a new audio slicer with the given configuration
    pub fn new(config: SlicerConfig) -> Self {
        let sr = config.sample_rate as f32;

        // Convert dB threshold to linear amplitude
        let threshold = 10f32.powf(config.threshold_db / 20.0);

        // Convert milliseconds to samples
        let hop_size = (sr * config.hop_size_ms as f32 / 1000.0).round() as usize;
        let min_interval_samples = (sr * config.min_interval_ms as f32 / 1000.0).round() as usize;
        let win_size = min_interval_samples.min(4 * hop_size);

        // Convert to frames
        let min_length = (sr * config.min_length_ms as f32 / 1000.0 / hop_size as f32).round() as usize;
        let min_interval = (min_interval_samples as f32 / hop_size as f32).round() as usize;
        let max_sil_kept = (sr * config.max_sil_kept_ms as f32 / 1000.0 / hop_size as f32).round() as usize;

        debug!(
            threshold = threshold,
            hop_size = hop_size,
            win_size = win_size,
            min_length = min_length,
            min_interval = min_interval,
            max_sil_kept = max_sil_kept,
            "Slicer initialized"
        );

        Self {
            config,
            threshold,
            hop_size,
            win_size,
            min_length,
            min_interval,
            max_sil_kept,
        }
    }

    /// Calculate RMS (Root Mean Square) energy for each frame
    fn get_rms(&self, samples: &[f32]) -> Vec<f32> {
        let frame_length = self.win_size;
        let hop_length = self.hop_size;

        // Pad the signal
        let pad_size = frame_length / 2;
        let mut padded = vec![0.0f32; pad_size];
        padded.extend_from_slice(samples);
        padded.extend(vec![0.0f32; pad_size]);

        // Calculate number of frames
        let n_frames = (padded.len() - frame_length) / hop_length + 1;
        let mut rms = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let start = i * hop_length;
            let end = start + frame_length;
            let frame = &padded[start..end];

            // Calculate power (mean of squared values)
            let power: f32 = frame.iter().map(|x| x * x).sum::<f32>() / frame_length as f32;
            rms.push(power.sqrt());
        }

        rms
    }

    /// Find the index of minimum value in a slice
    fn argmin(slice: &[f32]) -> usize {
        slice
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Slice audio samples into chunks based on silence detection
    pub fn slice(&self, samples: &[f32]) -> Vec<(Vec<f32>, u64, u64)> {
        let total_samples = samples.len();

        // If audio is too short, return as single chunk
        if total_samples <= self.min_length * self.hop_size {
            let end_ms = (total_samples as f64 / self.config.sample_rate as f64 * 1000.0) as u64;
            return vec![(samples.to_vec(), 0, end_ms)];
        }

        // Calculate RMS for each frame
        let rms_list = self.get_rms(samples);
        let total_frames = rms_list.len();

        // Find silence regions
        let mut sil_tags: Vec<(usize, usize)> = Vec::new();
        let mut silence_start: Option<usize> = None;
        let mut clip_start = 0usize;

        for (i, &rms) in rms_list.iter().enumerate() {
            // Frame is silent
            if rms < self.threshold {
                if silence_start.is_none() {
                    silence_start = Some(i);
                }
                continue;
            }

            // Frame is not silent
            if silence_start.is_none() {
                continue;
            }

            let sil_start = silence_start.unwrap();

            // Check if we should slice
            let is_leading_silence = sil_start == 0 && i > self.max_sil_kept;
            let need_slice = (i - sil_start >= self.min_interval) && (i - clip_start >= self.min_length);

            if !is_leading_silence && !need_slice {
                silence_start = None;
                continue;
            }

            // Determine slice position (use exclusive ranges with +1)
            if i - sil_start <= self.max_sil_kept {
                let end_idx = (i + 1).min(rms_list.len());
                let pos = Self::argmin(&rms_list[sil_start..end_idx]) + sil_start;
                if sil_start == 0 {
                    sil_tags.push((0, pos));
                } else {
                    sil_tags.push((pos, pos));
                }
                clip_start = pos;
            } else if i - sil_start <= self.max_sil_kept * 2 {
                let search_start = if i >= self.max_sil_kept { i - self.max_sil_kept } else { 0 };
                let search_end = (sil_start + self.max_sil_kept + 1).min(rms_list.len());
                let pos = Self::argmin(&rms_list[search_start..search_end]) + search_start;

                let pos_l = Self::argmin(&rms_list[sil_start..search_end]) + sil_start;
                let pos_r_end = (i + 1).min(rms_list.len());
                let pos_r = Self::argmin(&rms_list[search_start..pos_r_end]) + search_start;

                if sil_start == 0 {
                    sil_tags.push((0, pos_r));
                    clip_start = pos_r;
                } else {
                    sil_tags.push((pos_l.min(pos), pos_r.max(pos)));
                    clip_start = pos_r.max(pos);
                }
            } else {
                let pos_l = Self::argmin(&rms_list[sil_start..(sil_start + self.max_sil_kept + 1).min(rms_list.len())]) + sil_start;
                let search_start = if i >= self.max_sil_kept { i - self.max_sil_kept } else { 0 };
                let pos_r_end = (i + 1).min(rms_list.len());
                let pos_r = Self::argmin(&rms_list[search_start..pos_r_end]) + search_start;

                if sil_start == 0 {
                    sil_tags.push((0, pos_r));
                } else {
                    sil_tags.push((pos_l, pos_r));
                }
                clip_start = pos_r;
            }

            silence_start = None;
        }

        // Handle trailing silence
        if let Some(sil_start) = silence_start {
            if total_frames - sil_start >= self.min_interval {
                let silence_end = (sil_start + self.max_sil_kept + 1).min(total_frames);
                let pos = Self::argmin(&rms_list[sil_start..silence_end]) + sil_start;
                sil_tags.push((pos, total_frames));
            }
        }

        // Convert silence tags to chunks
        let mut chunks = Vec::new();
        let sr = self.config.sample_rate as f64;

        if sil_tags.is_empty() {
            let end_ms = (total_samples as f64 / sr * 1000.0) as u64;
            return vec![(samples.to_vec(), 0, end_ms)];
        }

        // First chunk (before first silence)
        if sil_tags[0].0 > 0 {
            let end_sample = (sil_tags[0].0 * self.hop_size).min(total_samples);
            let start_ms = 0u64;
            let end_ms = (end_sample as f64 / sr * 1000.0) as u64;
            chunks.push((samples[..end_sample].to_vec(), start_ms, end_ms));
        }

        // Middle chunks (between silences)
        for i in 0..sil_tags.len() - 1 {
            let start_sample = sil_tags[i].1 * self.hop_size;
            let end_sample = (sil_tags[i + 1].0 * self.hop_size).min(total_samples);

            if start_sample < end_sample && end_sample <= total_samples {
                let start_ms = (start_sample as f64 / sr * 1000.0) as u64;
                let end_ms = (end_sample as f64 / sr * 1000.0) as u64;
                chunks.push((samples[start_sample..end_sample].to_vec(), start_ms, end_ms));
            }
        }

        // Last chunk (after last silence)
        if let Some(last_tag) = sil_tags.last() {
            if last_tag.1 < total_frames {
                let start_sample = last_tag.1 * self.hop_size;
                if start_sample < total_samples {
                    let start_ms = (start_sample as f64 / sr * 1000.0) as u64;
                    let end_ms = (total_samples as f64 / sr * 1000.0) as u64;
                    chunks.push((samples[start_sample..].to_vec(), start_ms, end_ms));
                }
            }
        }

        chunks
    }

    /// Normalize audio chunk
    fn normalize(&self, samples: &mut [f32]) {
        let max_val = samples.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

        if max_val > 1.0 {
            for s in samples.iter_mut() {
                *s /= max_val;
            }
        }

        // Apply amplitude scaling with alpha mixing
        let target = self.config.max_amplitude * self.config.alpha;
        let mix = 1.0 - self.config.alpha;

        for s in samples.iter_mut() {
            let scaled = *s / max_val.max(1.0) * target;
            *s = scaled + mix * *s;
        }
    }

    /// Slice an audio file and save chunks to output directory
    pub fn slice_file<P1: AsRef<Path>, P2: AsRef<Path>>(&self, input_path: P1, output_dir: P2) -> Result<Vec<AudioChunk>> {
        let input_path = input_path.as_ref();
        let output_dir = output_dir.as_ref();

        // Load audio file
        let (samples, sr) = mlx_rs_core::audio::load_wav(input_path)
            .map_err(|e| PreprocessingError::Audio(format!("Failed to load audio: {}", e)))?;

        // Resample if needed
        let samples = if sr != self.config.sample_rate {
            info!(from = sr, to = self.config.sample_rate, "Resampling audio");
            mlx_rs_core::audio::resample(&samples, sr, self.config.sample_rate)
        } else {
            samples
        };

        // Slice the audio
        let raw_chunks = self.slice(&samples);
        info!(chunks = raw_chunks.len(), "Sliced audio into chunks");

        // Create output directory
        std::fs::create_dir_all(output_dir)?;

        // Save each chunk
        let stem = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("audio");

        let mut chunks = Vec::new();
        for (i, (mut chunk_samples, start_ms, end_ms)) in raw_chunks.into_iter().enumerate() {
            // Normalize
            self.normalize(&mut chunk_samples);

            // Generate output filename
            let filename = format!("{}_{:010}_{:010}.wav", stem, start_ms, end_ms);
            let output_path = output_dir.join(&filename);

            // Save as WAV
            mlx_rs_core::audio::save_wav(&chunk_samples, self.config.sample_rate, &output_path)
                .map_err(|e| PreprocessingError::Audio(format!("Failed to save chunk: {}", e)))?;

            debug!(path = %output_path.display(), start_ms, end_ms, "Saved chunk");

            chunks.push(AudioChunk {
                samples: chunk_samples,
                start_ms,
                end_ms,
                output_path,
            });
        }

        Ok(chunks)
    }

    /// Slice multiple audio files from a directory
    pub fn slice_directory<P1: AsRef<Path>, P2: AsRef<Path>>(&self, input_dir: P1, output_dir: P2) -> Result<Vec<AudioChunk>> {
        let input_dir = input_dir.as_ref();
        let output_dir = output_dir.as_ref();

        let mut all_chunks = Vec::new();

        let mut entries: Vec<_> = std::fs::read_dir(input_dir)?
            .filter_map(|e| e.ok())
            .collect();
        entries.sort_by_key(|e| e.path());

        for entry in entries {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if ["wav", "mp3", "flac", "ogg", "m4a"].contains(&ext.as_str()) {
                        info!(file = %path.display(), "Processing audio file");
                        match self.slice_file(&path, output_dir) {
                            Ok(chunks) => all_chunks.extend(chunks),
                            Err(e) => {
                                tracing::warn!(file = %path.display(), error = %e, "Failed to process file");
                            }
                        }
                    }
                }
            }
        }

        Ok(all_chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slicer_config_default() {
        let config = SlicerConfig::default();
        assert_eq!(config.sample_rate, 32000);
        assert_eq!(config.threshold_db, -40.0);
        assert_eq!(config.min_length_ms, 5000);
    }

    #[test]
    fn test_rms_calculation() {
        let config = SlicerConfig {
            sample_rate: 16000,
            hop_size_ms: 10,
            ..Default::default()
        };
        let slicer = AudioSlicer::new(config);

        // Create a simple test signal
        let samples: Vec<f32> = (0..16000).map(|i| (i as f32 / 16000.0 * 440.0 * std::f32::consts::TAU).sin() * 0.5).collect();
        let rms = slicer.get_rms(&samples);

        // RMS should be approximately 0.5 / sqrt(2) ≈ 0.35 for a sine wave
        assert!(!rms.is_empty());
        let avg_rms: f32 = rms.iter().sum::<f32>() / rms.len() as f32;
        assert!(avg_rms > 0.3 && avg_rms < 0.4);
    }

    #[test]
    fn test_slice_short_audio() {
        let config = SlicerConfig {
            sample_rate: 16000,
            min_length_ms: 1000,
            ..Default::default()
        };
        let slicer = AudioSlicer::new(config);

        // Audio shorter than min_length should return single chunk
        let samples = vec![0.0f32; 8000]; // 0.5 seconds
        let chunks = slicer.slice(&samples);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0.len(), 8000);
    }
}
