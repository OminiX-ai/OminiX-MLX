//! VITS Training Dataset
//!
//! Loads preprocessed data for VITS fewshot training:
//! - SSL features (HuBERT) [768, T]
//! - Audio waveforms [samples]
//! - Phoneme IDs [T]

use std::path::{Path, PathBuf};

use mlx_rs::Array;
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::error::Error;
use super::vits_trainer::VITSBatch;

/// Metadata for a single VITS training sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VITSSampleMetadata {
    pub id: String,
    pub ssl_len: usize,
    pub audio_len: usize,
    pub phoneme_len: usize,
}

/// Dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VITSDatasetMetadata {
    pub num_samples: usize,
    pub sample_rate: u32,
    pub ssl_dim: usize,
    pub samples: Vec<VITSSampleMetadata>,
}

/// VITS Training Dataset
pub struct VITSDataset {
    /// Dataset root directory
    root_dir: PathBuf,
    /// Sample metadata
    metadata: VITSDatasetMetadata,
    /// Current shuffle order
    indices: Vec<usize>,
    /// Random number generator
    rng: StdRng,
}

impl VITSDataset {
    /// Load dataset from directory
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let root_dir = path.as_ref().to_path_buf();

        // Load metadata
        let metadata_path = root_dir.join("metadata.json");
        if !metadata_path.exists() {
            return Err(Error::Message(format!(
                "Dataset metadata not found: {:?}",
                metadata_path
            )));
        }

        let metadata_str = std::fs::read_to_string(&metadata_path)?;
        let metadata: VITSDatasetMetadata = serde_json::from_str(&metadata_str)
            .map_err(|e| Error::Message(format!("Failed to parse metadata: {}", e)))?;

        // Verify directories exist
        for dir_name in ["ssl", "audio", "phonemes"] {
            let dir = root_dir.join(dir_name);
            if !dir.exists() {
                return Err(Error::Message(format!(
                    "Dataset directory not found: {:?}",
                    dir
                )));
            }
        }

        // Initialize indices
        let indices: Vec<usize> = (0..metadata.num_samples).collect();
        let rng = StdRng::seed_from_u64(42);

        Ok(Self {
            root_dir,
            metadata,
            indices,
            rng,
        })
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.metadata.num_samples
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.metadata.sample_rate
    }

    /// Shuffle dataset
    pub fn shuffle(&mut self, seed: Option<u64>) {
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        }
        self.indices.shuffle(&mut self.rng);
    }

    /// Reset indices to original order
    pub fn reset(&mut self) {
        self.indices = (0..self.metadata.num_samples).collect();
    }

    /// Load a single sample
    fn load_sample(&self, idx: usize) -> Result<VITSSampleData, Error> {
        let sample = &self.metadata.samples[idx];

        // Load SSL features [768, T]
        let ssl_path = self.root_dir.join("ssl").join(format!("{}.npy", sample.id));
        let (ssl_data, ssl_shape) = load_npy_f32_with_shape(&ssl_path)?;

        // Load audio [samples]
        let audio_path = self.root_dir.join("audio").join(format!("{}.npy", sample.id));
        let audio_data = load_npy_f32(&audio_path)?;

        // Load phonemes [T]
        let phoneme_path = self.root_dir.join("phonemes").join(format!("{}.npy", sample.id));
        let phoneme_data = load_npy_i32(&phoneme_path)?;

        Ok(VITSSampleData {
            id: sample.id.clone(),
            ssl: ssl_data,
            ssl_shape,
            audio: audio_data,
            phonemes: phoneme_data,
        })
    }

    /// Get a batch of samples
    ///
    /// This loads samples, computes spectrograms, and pads to max lengths.
    /// For fewshot training, segment extraction is done in the trainer.
    pub fn get_batch(
        &self,
        batch_indices: &[usize],
        segment_size: i32,
        hop_length: i32,
    ) -> Result<VITSBatch, Error> {
        use crate::audio::{spectrogram_mlx, SpectrogramConfig};
        use mlx_rs::ops::indexing::IndexOp;
        use mlx_rs::transforms::eval;

        let batch_size = batch_indices.len();
        if batch_size == 0 {
            return Err(Error::Message("Empty batch".to_string()));
        }

        // Load all samples
        let mut samples = Vec::with_capacity(batch_size);
        for &idx in batch_indices {
            let actual_idx = self.indices[idx];
            samples.push(self.load_sample(actual_idx)?);
        }

        // Extract random segments from audio
        // For VITS training, we need aligned SSL and audio segments
        let ssl_segment_size = segment_size / (hop_length * 2); // SSL is downsampled by hop_length * 2 relative to audio (320 hop * 2)
        let spec_segment_size = segment_size / hop_length;

        let mut ssl_list = Vec::with_capacity(batch_size);
        let mut audio_list = Vec::with_capacity(batch_size);
        let mut phoneme_list = Vec::with_capacity(batch_size);

        for sample in &samples {
            // Random start position (in audio samples)
            let max_audio_start = sample.audio.len() as i32 - segment_size;
            let audio_start = if max_audio_start > 0 {
                (rand::random::<f32>() * max_audio_start as f32) as usize
            } else {
                0
            };

            // Compute aligned SSL start
            let ssl_start = audio_start / (hop_length as usize * 2);
            let ssl_end = (ssl_start + ssl_segment_size as usize).min(sample.ssl_shape[1]);

            // Extract SSL segment [768, ssl_segment_size]
            let ssl_dim = sample.ssl_shape[0];
            let ssl_len = sample.ssl_shape[1];
            let mut ssl_segment = vec![0.0f32; ssl_dim * ssl_segment_size as usize];
            let actual_ssl_len = ssl_end - ssl_start;
            for c in 0..ssl_dim {
                for t in 0..actual_ssl_len {
                    let src_idx = c * ssl_len + ssl_start + t;
                    let dst_idx = c * ssl_segment_size as usize + t;
                    if src_idx < sample.ssl.len() && dst_idx < ssl_segment.len() {
                        ssl_segment[dst_idx] = sample.ssl[src_idx];
                    }
                }
            }
            ssl_list.push((ssl_segment, ssl_dim, ssl_segment_size as usize));

            // Extract audio segment [segment_size]
            let audio_end = (audio_start + segment_size as usize).min(sample.audio.len());
            let mut audio_segment = vec![0.0f32; segment_size as usize];
            let actual_audio_len = audio_end - audio_start;
            audio_segment[..actual_audio_len].copy_from_slice(&sample.audio[audio_start..audio_end]);
            audio_list.push(audio_segment);

            // Phonemes (use full sequence for text conditioning)
            phoneme_list.push(sample.phonemes.clone());
        }

        // Find max phoneme length for padding
        let max_phoneme_len = phoneme_list.iter().map(|p| p.len()).max().unwrap_or(1);

        // Create SSL tensor [batch, ssl_dim, ssl_segment_size]
        let ssl_dim = 768;
        let mut ssl_flat = vec![0.0f32; batch_size * ssl_dim * ssl_segment_size as usize];
        for (i, (ssl, _, _)) in ssl_list.iter().enumerate() {
            for (j, &v) in ssl.iter().enumerate() {
                ssl_flat[i * ssl_dim * ssl_segment_size as usize + j] = v;
            }
        }
        let ssl_features = Array::from_slice(
            &ssl_flat,
            &[batch_size as i32, ssl_dim as i32, ssl_segment_size]
        );

        // Create audio tensor [batch, 1, segment_size]
        let mut audio_flat = vec![0.0f32; batch_size * segment_size as usize];
        for (i, audio) in audio_list.iter().enumerate() {
            for (j, &v) in audio.iter().enumerate() {
                audio_flat[i * segment_size as usize + j] = v;
            }
        }
        let audio = Array::from_slice(
            &audio_flat,
            &[batch_size as i32, 1, segment_size]
        );

        // Create phoneme tensor [batch, max_phoneme_len]
        let mut phoneme_flat = vec![0i32; batch_size * max_phoneme_len];
        let mut text_lengths = vec![0i32; batch_size];
        for (i, phonemes) in phoneme_list.iter().enumerate() {
            text_lengths[i] = phonemes.len() as i32;
            for (j, &p) in phonemes.iter().enumerate() {
                phoneme_flat[i * max_phoneme_len + j] = p;
            }
        }
        let text = Array::from_slice(
            &phoneme_flat,
            &[batch_size as i32, max_phoneme_len as i32]
        );
        let text_lengths_arr = Array::from_slice(&text_lengths, &[batch_size as i32]);

        // Compute spectrogram from audio for enc_q input [batch, n_fft/2+1, spec_len]
        let spec_config = SpectrogramConfig::default();
        let audio_squeezed = audio.squeeze_axes(&[1])
            .map_err(|e| Error::Message(format!("squeeze audio failed: {}", e)))?;
        let spec = spectrogram_mlx(&audio_squeezed, &spec_config)
            .map_err(|e| Error::Message(format!("spectrogram failed: {}", e)))?;

        // Spec lengths (all same since we extracted fixed segments)
        let spec_lengths = Array::from_slice(
            &vec![spec_segment_size; batch_size],
            &[batch_size as i32]
        );

        // For reference encoder, use linear spectrogram sliced to 704 frequency bins
        // (ref_enc expects [batch, 704, time], not mel spectrogram)
        // We can reuse the spec which is [batch, 1025, time] and slice it
        let refer_mel = spec.index((.., ..704, ..));

        // Evaluate all arrays
        eval([&ssl_features, &audio, &text, &spec, &refer_mel])
            .map_err(|e| Error::Message(e.to_string()))?;

        Ok(VITSBatch {
            ssl_features,
            spec,
            spec_lengths,
            text,
            text_lengths: text_lengths_arr,
            audio,
            refer_mel,
        })
    }

    /// Iterate over batches
    pub fn iter_batches(&self, batch_size: usize, segment_size: i32, hop_length: i32) -> VITSBatchIterator<'_> {
        VITSBatchIterator {
            dataset: self,
            batch_size,
            segment_size,
            hop_length,
            current_idx: 0,
        }
    }
}

/// Raw sample data before batching
struct VITSSampleData {
    id: String,
    ssl: Vec<f32>,
    ssl_shape: Vec<usize>,
    audio: Vec<f32>,
    phonemes: Vec<i32>,
}

/// Iterator over VITS training batches
pub struct VITSBatchIterator<'a> {
    dataset: &'a VITSDataset,
    batch_size: usize,
    segment_size: i32,
    hop_length: i32,
    current_idx: usize,
}

impl<'a> Iterator for VITSBatchIterator<'a> {
    type Item = Result<VITSBatch, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.dataset.len());
        let batch_indices: Vec<usize> = (self.current_idx..end_idx).collect();
        self.current_idx = end_idx;

        Some(self.dataset.get_batch(&batch_indices, self.segment_size, self.hop_length))
    }
}

// ============================================================================
// NPY file loading utilities
// ============================================================================

/// Load i32 array from .npy file
fn load_npy_i32(path: &Path) -> Result<Vec<i32>, Error> {
    let data = std::fs::read(path)?;

    const NPY_MAGIC: &[u8] = b"\x93NUMPY";
    if data.len() < 10 || &data[..6] != NPY_MAGIC {
        return Err(Error::Message(format!("Invalid NPY file: {:?}", path)));
    }

    // Find header end
    let header_end = data[10..]
        .iter()
        .position(|&b| b == b'\n')
        .map(|p| 10 + p + 1)
        .ok_or_else(|| Error::Message("Invalid NPY header".to_string()))?;

    // Parse data as i32
    let data_slice = &data[header_end..];
    if data_slice.len() % 4 != 0 {
        return Err(Error::Message("NPY data not aligned to 4 bytes".to_string()));
    }

    let values: Vec<i32> = data_slice
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok(values)
}

/// Load f32 array from .npy file
fn load_npy_f32(path: &Path) -> Result<Vec<f32>, Error> {
    let data = std::fs::read(path)?;

    const NPY_MAGIC: &[u8] = b"\x93NUMPY";
    if data.len() < 10 || &data[..6] != NPY_MAGIC {
        return Err(Error::Message(format!("Invalid NPY file: {:?}", path)));
    }

    let header_end = data[10..]
        .iter()
        .position(|&b| b == b'\n')
        .map(|p| 10 + p + 1)
        .ok_or_else(|| Error::Message("Invalid NPY header".to_string()))?;

    let data_slice = &data[header_end..];
    if data_slice.len() % 4 != 0 {
        return Err(Error::Message("NPY data not aligned to 4 bytes".to_string()));
    }

    let values: Vec<f32> = data_slice
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok(values)
}

/// Load f32 array from .npy file with shape
fn load_npy_f32_with_shape(path: &Path) -> Result<(Vec<f32>, Vec<usize>), Error> {
    let data = std::fs::read(path)?;

    const NPY_MAGIC: &[u8] = b"\x93NUMPY";
    if data.len() < 10 || &data[..6] != NPY_MAGIC {
        return Err(Error::Message(format!("Invalid NPY file: {:?}", path)));
    }

    let header_end = data[10..]
        .iter()
        .position(|&b| b == b'\n')
        .map(|p| 10 + p + 1)
        .ok_or_else(|| Error::Message("Invalid NPY header".to_string()))?;

    // Parse header for shape
    let header_str = String::from_utf8_lossy(&data[10..header_end]);
    let shape = parse_npy_shape(&header_str)?;

    let data_slice = &data[header_end..];
    if data_slice.len() % 4 != 0 {
        return Err(Error::Message("NPY data not aligned to 4 bytes".to_string()));
    }

    let values: Vec<f32> = data_slice
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok((values, shape))
}

/// Parse shape from NPY header
fn parse_npy_shape(header: &str) -> Result<Vec<usize>, Error> {
    if let Some(start) = header.find("'shape':") {
        let rest = &header[start + 8..];
        if let Some(paren_start) = rest.find('(') {
            if let Some(paren_end) = rest.find(')') {
                let shape_str = &rest[paren_start + 1..paren_end];
                let shape: Vec<usize> = shape_str
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                if !shape.is_empty() {
                    return Ok(shape);
                }
            }
        }
    }
    Ok(vec![1])
}
