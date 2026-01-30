//! Training dataset and data loading

use std::path::{Path, PathBuf};

use mlx_rs::Array;
use serde::{Deserialize, Serialize};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::error::Error;

/// Metadata for a single training sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleMetadata {
    pub id: String,
    pub audio_path: String,
    pub transcript: String,
    pub phoneme_len: usize,
    pub semantic_len: usize,
}

/// Dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub num_samples: usize,
    pub samples: Vec<SampleMetadata>,
}

/// A batch of training data
#[derive(Debug)]
pub struct TrainingBatch {
    /// Phoneme IDs [batch, max_phoneme_len]
    pub phoneme_ids: Array,
    /// Phoneme sequence lengths [batch]
    pub phoneme_lens: Array,
    /// BERT features [batch, 1024, max_phoneme_len]
    pub bert_features: Array,
    /// Target semantic IDs [batch, max_semantic_len]
    pub semantic_ids: Array,
    /// Semantic sequence lengths [batch]
    pub semantic_lens: Array,
}

/// Training dataset loaded from preprocessed files
pub struct TrainingDataset {
    /// Dataset root directory
    #[allow(dead_code)]
    root_dir: PathBuf,
    /// Sample metadata
    metadata: DatasetMetadata,
    /// Phoneme IDs directory
    phoneme_dir: PathBuf,
    /// BERT features directory
    bert_dir: PathBuf,
    /// Semantic IDs directory
    semantic_dir: PathBuf,
    /// Current shuffle order
    indices: Vec<usize>,
    /// Random number generator for shuffling
    rng: StdRng,
}

impl TrainingDataset {
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
        let metadata: DatasetMetadata = serde_json::from_str(&metadata_str)
            .map_err(|e| Error::Message(format!("Failed to parse metadata: {}", e)))?;

        // Verify directories exist
        let phoneme_dir = root_dir.join("phoneme_ids");
        let bert_dir = root_dir.join("bert_features");
        let semantic_dir = root_dir.join("semantic_ids");

        for dir in [&phoneme_dir, &bert_dir, &semantic_dir] {
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
            phoneme_dir,
            bert_dir,
            semantic_dir,
            indices,
            rng,
        })
    }

    /// Get number of samples in dataset
    pub fn len(&self) -> usize {
        self.metadata.num_samples
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Shuffle dataset indices
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

    /// Load a single sample by index
    fn load_sample(&self, idx: usize) -> Result<(Vec<i32>, Vec<f32>, Vec<usize>, Vec<i32>), Error> {
        let sample = &self.metadata.samples[idx];

        // Load phoneme IDs
        let phoneme_path = self.phoneme_dir.join(format!("{}.npy", sample.id));
        let phoneme_ids = load_npy_i32(&phoneme_path)?;

        // Load BERT features
        let bert_path = self.bert_dir.join(format!("{}.npy", sample.id));
        let (bert_features, bert_shape) = load_npy_f32_with_shape(&bert_path)?;

        // Load semantic IDs
        let semantic_path = self.semantic_dir.join(format!("{}.npy", sample.id));
        let semantic_ids = load_npy_i32(&semantic_path)?;

        Ok((phoneme_ids, bert_features, bert_shape, semantic_ids))
    }

    /// Get a batch of samples
    pub fn get_batch(&self, batch_indices: &[usize]) -> Result<TrainingBatch, Error> {
        let batch_size = batch_indices.len();
        if batch_size == 0 {
            return Err(Error::Message("Empty batch".to_string()));
        }

        // Load all samples
        let mut phoneme_ids_list = Vec::with_capacity(batch_size);
        let mut bert_features_list = Vec::with_capacity(batch_size);
        let mut bert_shapes_list = Vec::with_capacity(batch_size);
        let mut semantic_ids_list = Vec::with_capacity(batch_size);

        for &idx in batch_indices {
            let actual_idx = self.indices[idx];
            let (phonemes, bert, bert_shape, semantics) = self.load_sample(actual_idx)?;
            phoneme_ids_list.push(phonemes);
            bert_features_list.push(bert);
            bert_shapes_list.push(bert_shape);
            semantic_ids_list.push(semantics);
        }

        // Find max lengths for padding
        let max_phoneme_len = phoneme_ids_list.iter().map(|p| p.len()).max().unwrap_or(0);
        let max_semantic_len = semantic_ids_list.iter().map(|s| s.len()).max().unwrap_or(0);

        // Pad and stack phoneme IDs
        let mut phoneme_ids_padded = vec![0i32; batch_size * max_phoneme_len];
        let mut phoneme_lens = vec![0i32; batch_size];
        for (i, phonemes) in phoneme_ids_list.iter().enumerate() {
            phoneme_lens[i] = phonemes.len() as i32;
            for (j, &p) in phonemes.iter().enumerate() {
                phoneme_ids_padded[i * max_phoneme_len + j] = p;
            }
        }

        // Pad and stack BERT features [batch, 1024, max_phoneme_len]
        let bert_dim = 1024;
        let mut bert_features_padded = vec![0.0f32; batch_size * bert_dim * max_phoneme_len];
        for (i, (bert, shape)) in bert_features_list.iter().zip(bert_shapes_list.iter()).enumerate() {
            let seq_len = if shape.len() == 2 { shape[1] } else { shape[0] };
            for c in 0..bert_dim {
                for t in 0..seq_len.min(max_phoneme_len) {
                    let src_idx = c * seq_len + t;
                    let dst_idx = i * bert_dim * max_phoneme_len + c * max_phoneme_len + t;
                    if src_idx < bert.len() {
                        bert_features_padded[dst_idx] = bert[src_idx];
                    }
                }
            }
        }

        // Pad and stack semantic IDs
        let mut semantic_ids_padded = vec![0i32; batch_size * max_semantic_len];
        let mut semantic_lens = vec![0i32; batch_size];
        for (i, semantics) in semantic_ids_list.iter().enumerate() {
            semantic_lens[i] = semantics.len() as i32;
            for (j, &s) in semantics.iter().enumerate() {
                semantic_ids_padded[i * max_semantic_len + j] = s;
            }
        }

        // Create MLX arrays
        let phoneme_ids = Array::from_slice(
            &phoneme_ids_padded,
            &[batch_size as i32, max_phoneme_len as i32]
        );
        let phoneme_lens_arr = Array::from_slice(&phoneme_lens, &[batch_size as i32]);
        let bert_features = Array::from_slice(
            &bert_features_padded,
            &[batch_size as i32, bert_dim as i32, max_phoneme_len as i32]
        );
        let semantic_ids = Array::from_slice(
            &semantic_ids_padded,
            &[batch_size as i32, max_semantic_len as i32]
        );
        let semantic_lens_arr = Array::from_slice(&semantic_lens, &[batch_size as i32]);

        Ok(TrainingBatch {
            phoneme_ids,
            phoneme_lens: phoneme_lens_arr,
            bert_features,
            semantic_ids,
            semantic_lens: semantic_lens_arr,
        })
    }

    /// Iterate over batches
    pub fn iter_batches(&self, batch_size: usize) -> BatchIterator<'_> {
        BatchIterator {
            dataset: self,
            batch_size,
            current_idx: 0,
        }
    }
}

/// Iterator over training batches
pub struct BatchIterator<'a> {
    dataset: &'a TrainingDataset,
    batch_size: usize,
    current_idx: usize,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = Result<TrainingBatch, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.dataset.len());
        let batch_indices: Vec<usize> = (self.current_idx..end_idx).collect();
        self.current_idx = end_idx;

        Some(self.dataset.get_batch(&batch_indices))
    }
}

/// Load i32 array from .npy file
fn load_npy_i32(path: &Path) -> Result<Vec<i32>, Error> {
    let data = std::fs::read(path)?;

    // Simple NPY parser for 1D int32 arrays
    const NPY_MAGIC: &[u8] = b"\x93NUMPY";
    if data.len() < 10 || &data[..6] != NPY_MAGIC {
        return Err(Error::Message(format!("Invalid NPY file: {:?}", path)));
    }

    // Find header end (newline after dict)
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

/// Load f32 array from .npy file with shape
fn load_npy_f32_with_shape(path: &Path) -> Result<(Vec<f32>, Vec<usize>), Error> {
    let data = std::fs::read(path)?;

    // Simple NPY parser
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

    // Parse header for shape (simplified - assumes shape is in header)
    let header_str = String::from_utf8_lossy(&data[10..header_end]);
    let shape = parse_npy_shape(&header_str)?;

    // Parse data as f32
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

/// Parse shape from NPY header string
fn parse_npy_shape(header: &str) -> Result<Vec<usize>, Error> {
    // Look for 'shape': (N, M) or 'shape': (N,)
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
    // Default to 1D if shape not found
    Ok(vec![1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_npy_shape() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1024, 42), }";
        let shape = parse_npy_shape(header).unwrap();
        assert_eq!(shape, vec![1024, 42]);

        let header2 = "{'descr': '<i4', 'fortran_order': False, 'shape': (100,), }";
        let shape2 = parse_npy_shape(header2).unwrap();
        assert_eq!(shape2, vec![100]);
    }
}
