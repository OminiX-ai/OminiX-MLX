//! Model conversion utilities for converting PyTorch models to MLX-compatible safetensors format.
//!
//! This module provides pure Rust utilities for loading PyTorch pickle files and converting
//! them to safetensors format that can be used with MLX models.
//!
//! # Features
//!
//! Enable the `convert` feature to use this module:
//!
//! ```toml
//! [dependencies]
//! mlx-rs-core = { path = "../mlx-rs-core", features = ["convert"] }
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use mlx_rs_core::convert::{load_pytorch_model, save_safetensors, WeightMapping};
//! use std::path::Path;
//!
//! // Load PyTorch model
//! let tensors = load_pytorch_model(Path::new("model.pt"))?;
//!
//! // Create weight mapping
//! let mapping = WeightMapping::new();
//! mapping.add("old.name", "new.name");
//!
//! // Convert and save
//! let converted = mapping.apply(&tensors);
//! save_safetensors(&converted, Path::new("model.safetensors"))?;
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use candle_core::{pickle, DType, Tensor as CandleTensor};
use safetensors::tensor::{Dtype, TensorView};

/// Error type for conversion operations
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),

    #[error("Conversion error: {0}")]
    Conversion(String),
}

/// Result type for conversion operations
pub type Result<T> = std::result::Result<T, ConvertError>;

/// Weight name mapping for model conversion
#[derive(Debug, Clone, Default)]
pub struct WeightMapping {
    mappings: HashMap<String, String>,
}

impl WeightMapping {
    /// Create a new empty weight mapping
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mapping from source name to destination name
    pub fn add(&mut self, src: impl Into<String>, dst: impl Into<String>) -> &mut Self {
        self.mappings.insert(src.into(), dst.into());
        self
    }

    /// Add multiple mappings at once
    pub fn add_many(&mut self, mappings: &[(&str, &str)]) -> &mut Self {
        for (src, dst) in mappings {
            self.mappings.insert((*src).to_string(), (*dst).to_string());
        }
        self
    }

    /// Get the destination name for a source name
    pub fn get(&self, src: &str) -> Option<&String> {
        self.mappings.get(src)
    }

    /// Check if mapping contains a source name
    pub fn contains(&self, src: &str) -> bool {
        self.mappings.contains_key(src)
    }

    /// Get the number of mappings
    pub fn len(&self) -> usize {
        self.mappings.len()
    }

    /// Check if mapping is empty
    pub fn is_empty(&self) -> bool {
        self.mappings.is_empty()
    }

    /// Apply mapping to a list of tensors, returning converted tensors and unmapped names
    pub fn apply(
        &self,
        tensors: &[(String, CandleTensor)],
    ) -> (Vec<(String, CandleTensor)>, Vec<String>) {
        let mut converted = Vec::new();
        let mut unmapped = Vec::new();

        for (src_name, tensor) in tensors {
            if let Some(dst_name) = self.mappings.get(src_name) {
                converted.push((dst_name.clone(), tensor.clone()));
            } else {
                unmapped.push(src_name.clone());
            }
        }

        (converted, unmapped)
    }
}

/// Converted tensor data ready for safetensors serialization
pub struct ConvertedTensor {
    /// Raw bytes (little-endian f32)
    pub data: Vec<u8>,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: Dtype,
}

/// Load a PyTorch model file (.pt or .pth) and return named tensors
pub fn load_pytorch_model(path: &Path) -> Result<Vec<(String, CandleTensor)>> {
    let tensors = pickle::read_all(path)?;
    Ok(tensors)
}

/// Convert a candle tensor to safetensors format
pub fn tensor_to_safetensor(tensor: &CandleTensor) -> Result<ConvertedTensor> {
    // Convert to f32 and get shape
    let tensor = tensor.to_dtype(DType::F32)?;
    let shape: Vec<usize> = tensor.dims().to_vec();

    // Flatten and get data
    let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

    // Convert to bytes (little-endian)
    let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();

    Ok(ConvertedTensor {
        data: bytes,
        shape,
        dtype: Dtype::F32,
    })
}

/// Save tensors to a safetensors file
pub fn save_safetensors(
    tensors: &[(String, CandleTensor)],
    output_path: &Path,
) -> Result<()> {
    // Convert all tensors
    let mut converted: HashMap<String, ConvertedTensor> = HashMap::new();

    for (name, tensor) in tensors {
        let data = tensor_to_safetensor(tensor)?;
        converted.insert(name.clone(), data);
    }

    // Build tensor views
    let tensor_views: HashMap<String, TensorView<'_>> = converted
        .iter()
        .map(|(name, ct)| {
            (
                name.clone(),
                TensorView::new(ct.dtype, ct.shape.clone(), &ct.data).unwrap(),
            )
        })
        .collect();

    // Create parent directory if needed
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Serialize to file
    safetensors::tensor::serialize_to_file(&tensor_views, &None, output_path)?;

    Ok(())
}

/// High-level model conversion function
///
/// Loads a PyTorch model, applies weight mapping, and saves as safetensors.
///
/// # Arguments
///
/// * `input_path` - Path to the PyTorch model file (.pt)
/// * `output_path` - Path for the output safetensors file
/// * `mapping` - Weight name mapping
///
/// # Returns
///
/// A tuple of (converted_count, unmapped_names)
pub fn convert_model(
    input_path: &Path,
    output_path: &Path,
    mapping: &WeightMapping,
) -> Result<(usize, Vec<String>)> {
    // Load PyTorch model
    let tensors = load_pytorch_model(input_path)?;

    // Apply mapping
    let (converted, unmapped) = mapping.apply(&tensors);
    let converted_count = converted.len();

    // Save as safetensors
    save_safetensors(&converted, output_path)?;

    Ok((converted_count, unmapped))
}

/// Copy auxiliary files from source to destination directory
pub fn copy_auxiliary_files(
    src_dir: &Path,
    dst_dir: &Path,
    files: &[&str],
) -> Result<Vec<String>> {
    let mut copied = Vec::new();

    fs::create_dir_all(dst_dir)?;

    for file in files {
        let src = src_dir.join(file);
        if src.exists() {
            let dst = dst_dir.join(file);
            fs::copy(&src, &dst)?;
            copied.push(file.to_string());
        }
    }

    Ok(copied)
}

// ============================================================================
// FunASR Paraformer-specific conversion
// ============================================================================

/// Create weight mapping for FunASR Paraformer model
pub fn paraformer_weight_mapping() -> WeightMapping {
    let mut mapping = WeightMapping::new();

    // Encoder first layer (encoders0)
    let first_layer_mappings = [
        ("encoder.encoders0.0.self_attn.linear_q_k_v.weight", "encoder.encoders0.0.self_attn.linear_q_k_v.weight"),
        ("encoder.encoders0.0.self_attn.linear_q_k_v.bias", "encoder.encoders0.0.self_attn.linear_q_k_v.bias"),
        ("encoder.encoders0.0.self_attn.linear_out.weight", "encoder.encoders0.0.self_attn.out_proj.weight"),
        ("encoder.encoders0.0.self_attn.linear_out.bias", "encoder.encoders0.0.self_attn.out_proj.bias"),
        ("encoder.encoders0.0.self_attn.fsmn_block.weight", "encoder.encoders0.0.self_attn.fsmn_block.weight"),
        ("encoder.encoders0.0.feed_forward.w_1.weight", "encoder.encoders0.0.ffn.up_proj.weight"),
        ("encoder.encoders0.0.feed_forward.w_1.bias", "encoder.encoders0.0.ffn.up_proj.bias"),
        ("encoder.encoders0.0.feed_forward.w_2.weight", "encoder.encoders0.0.ffn.down_proj.weight"),
        ("encoder.encoders0.0.feed_forward.w_2.bias", "encoder.encoders0.0.ffn.down_proj.bias"),
        ("encoder.encoders0.0.norm1.weight", "encoder.encoders0.0.norm1.weight"),
        ("encoder.encoders0.0.norm1.bias", "encoder.encoders0.0.norm1.bias"),
        ("encoder.encoders0.0.norm2.weight", "encoder.encoders0.0.norm2.weight"),
        ("encoder.encoders0.0.norm2.bias", "encoder.encoders0.0.norm2.bias"),
    ];
    mapping.add_many(&first_layer_mappings);

    // Encoder after_norm
    mapping.add("encoder.after_norm.weight", "encoder.after_norm.weight");
    mapping.add("encoder.after_norm.bias", "encoder.after_norm.bias");

    // Encoder layers 0-48 (49 regular layers)
    for i in 0..49 {
        let src_prefix = format!("encoder.encoders.{}", i);
        let dst_prefix = format!("encoder.layers.{}", i);

        let layer_mappings = [
            ("self_attn.linear_q_k_v.weight", "self_attn.linear_q_k_v.weight"),
            ("self_attn.linear_q_k_v.bias", "self_attn.linear_q_k_v.bias"),
            ("self_attn.linear_out.weight", "self_attn.out_proj.weight"),
            ("self_attn.linear_out.bias", "self_attn.out_proj.bias"),
            ("self_attn.fsmn_block.weight", "self_attn.fsmn_block.weight"),
            ("feed_forward.w_1.weight", "ffn.up_proj.weight"),
            ("feed_forward.w_1.bias", "ffn.up_proj.bias"),
            ("feed_forward.w_2.weight", "ffn.down_proj.weight"),
            ("feed_forward.w_2.bias", "ffn.down_proj.bias"),
            ("norm1.weight", "norm1.weight"),
            ("norm1.bias", "norm1.bias"),
            ("norm2.weight", "norm2.weight"),
            ("norm2.bias", "norm2.bias"),
        ];

        for (src_suffix, dst_suffix) in layer_mappings {
            mapping.add(
                format!("{}.{}", src_prefix, src_suffix),
                format!("{}.{}", dst_prefix, dst_suffix),
            );
        }
    }

    // Predictor (CIF)
    mapping.add("predictor.cif_conv1d.weight", "predictor.conv.weight");
    mapping.add("predictor.cif_conv1d.bias", "predictor.conv.bias");
    mapping.add("predictor.cif_output.weight", "predictor.output_proj.weight");
    mapping.add("predictor.cif_output.bias", "predictor.output_proj.bias");

    // Decoder embed
    mapping.add("decoder.embed.0.weight", "decoder.embed.0.weight");

    // Decoder layers 0-15
    for i in 0..16 {
        let src_prefix = format!("decoder.decoders.{}", i);
        let dst_prefix = format!("decoder.layers.{}", i);

        let layer_mappings = [
            ("self_attn.fsmn_block.weight", "self_attn.fsmn_block.weight"),
            ("src_attn.linear_q.weight", "src_attn.q_proj.weight"),
            ("src_attn.linear_q.bias", "src_attn.q_proj.bias"),
            ("src_attn.linear_k_v.weight", "src_attn.linear_k_v.weight"),
            ("src_attn.linear_k_v.bias", "src_attn.linear_k_v.bias"),
            ("src_attn.linear_out.weight", "src_attn.out_proj.weight"),
            ("src_attn.linear_out.bias", "src_attn.out_proj.bias"),
            ("feed_forward.w_1.weight", "ffn.up_proj.weight"),
            ("feed_forward.w_1.bias", "ffn.up_proj.bias"),
            ("feed_forward.w_2.weight", "ffn.down_proj.weight"),
            ("feed_forward.norm.weight", "feed_forward.norm.weight"),
            ("feed_forward.norm.bias", "feed_forward.norm.bias"),
            ("norm1.weight", "norm1.weight"),
            ("norm1.bias", "norm1.bias"),
            ("norm2.weight", "norm2.weight"),
            ("norm2.bias", "norm2.bias"),
            ("norm3.weight", "norm3.weight"),
            ("norm3.bias", "norm3.bias"),
        ];

        for (src_suffix, dst_suffix) in layer_mappings {
            mapping.add(
                format!("{}.{}", src_prefix, src_suffix),
                format!("{}.{}", dst_prefix, dst_suffix),
            );
        }
    }

    // Final FFN layer (decoders3)
    let final_ffn_mappings = [
        ("decoder.decoders3.0.norm1.weight", "decoder.decoders3.0.norm1.weight"),
        ("decoder.decoders3.0.norm1.bias", "decoder.decoders3.0.norm1.bias"),
        ("decoder.decoders3.0.feed_forward.w_1.weight", "decoder.decoders3.0.ffn.up_proj.weight"),
        ("decoder.decoders3.0.feed_forward.w_1.bias", "decoder.decoders3.0.ffn.up_proj.bias"),
        ("decoder.decoders3.0.feed_forward.norm.weight", "decoder.decoders3.0.feed_forward.norm.weight"),
        ("decoder.decoders3.0.feed_forward.norm.bias", "decoder.decoders3.0.feed_forward.norm.bias"),
        ("decoder.decoders3.0.feed_forward.w_2.weight", "decoder.decoders3.0.ffn.down_proj.weight"),
    ];
    mapping.add_many(&final_ffn_mappings);

    // Decoder after_norm and output
    mapping.add("decoder.after_norm.weight", "decoder.after_norm.weight");
    mapping.add("decoder.after_norm.bias", "decoder.after_norm.bias");
    mapping.add("decoder.output_layer.weight", "decoder.output_proj.weight");
    mapping.add("decoder.output_layer.bias", "decoder.output_proj.bias");

    mapping
}

/// Convert FunASR Paraformer model from PyTorch to MLX format
///
/// # Arguments
///
/// * `input_dir` - Directory containing model.pt and auxiliary files (am.mvn, tokens.txt)
/// * `output_dir` - Output directory for converted model
///
/// # Returns
///
/// A tuple of (converted_count, unmapped_count)
pub fn convert_paraformer(input_dir: &Path, output_dir: &Path) -> Result<(usize, usize)> {
    // Find model file
    let model_path = if input_dir.join("model.pt").exists() {
        input_dir.join("model.pt")
    } else if input_dir.join("model.pb").exists() {
        input_dir.join("model.pb")
    } else {
        return Err(ConvertError::Conversion(format!(
            "model.pt or model.pb not found in {:?}",
            input_dir
        )));
    };

    // Load and convert
    let mapping = paraformer_weight_mapping();
    let output_path = output_dir.join("paraformer.safetensors");

    let (converted_count, unmapped) = convert_model(&model_path, &output_path, &mapping)?;

    // Copy auxiliary files
    copy_auxiliary_files(input_dir, output_dir, &["am.mvn", "tokens.txt", "vocab.txt"])?;

    Ok((converted_count, unmapped.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_mapping() {
        let mut mapping = WeightMapping::new();
        mapping.add("old.name", "new.name");
        mapping.add("another.old", "another.new");

        assert_eq!(mapping.len(), 2);
        assert_eq!(mapping.get("old.name"), Some(&"new.name".to_string()));
        assert!(mapping.contains("old.name"));
        assert!(!mapping.contains("nonexistent"));
    }

    #[test]
    fn test_paraformer_mapping() {
        let mapping = paraformer_weight_mapping();

        // Check some expected mappings
        assert!(mapping.contains("encoder.after_norm.weight"));
        assert!(mapping.contains("decoder.output_layer.weight"));
        assert!(mapping.contains("predictor.cif_conv1d.weight"));

        // Check mapping count (should have all encoder, decoder, predictor weights)
        assert!(mapping.len() > 900); // Paraformer has ~956 weights
    }
}
