//! Weight loading helpers for Gemma 4.
//!
//! Mirrors the private helpers in qwen3-mlx's model.rs (which are not exported
//! there), adapted for the gemma4-mlx crate's `Error`/`Result` types.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::{
    module::{ModuleParameters as ModuleParametersTrait, Param},
    nn,
    Array,
};
use serde::Deserialize;

use crate::error::{Error, Result};

// ---------------------------------------------------------------------------
// Internal: index-file shape
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct WeightMap {
    weight_map: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

/// Load every tensor from all shard files listed in `model.safetensors.index.json`.
///
/// This is a blocking, eager load — suitable for inspection and single-inference
/// usage; not for streaming.
pub fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let json = std::fs::read_to_string(index_path)?;
    let wmap: WeightMap = serde_json::from_str(&json)?;

    // Deduplicate shard file names (many keys → same shard).
    let shard_files: HashSet<&String> = wmap.weight_map.values().collect();

    let mut all: HashMap<String, Array> = HashMap::new();
    for shard in shard_files {
        let shard_path = model_dir.join(shard);
        let loaded = Array::load_safetensors(&shard_path)?;
        all.extend(loaded);
    }
    Ok(all)
}

/// Return all weight-map **keys** from the index file, sorted.
///
/// Does NOT load any tensor data — reads only the ~KB index JSON.
/// Useful for structural inspection (R3/R6 verification) without paying the
/// cost of loading multi-GB shard files.
pub fn weight_keys(model_dir: &Path) -> Result<Vec<String>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let json = std::fs::read_to_string(index_path)?;
    let wmap: WeightMap = serde_json::from_str(&json)?;
    let mut keys: Vec<String> = wmap.weight_map.into_keys().collect();
    keys.sort();
    Ok(keys)
}

/// Retrieve a single weight by key, returning `Error::WeightNotFound` on miss.
pub fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

/// Build a frozen `nn::QuantizedEmbedding` from pre-loaded weight tensors.
///
/// Expects the following keys in `weights`:
/// - `{prefix}.weight`
/// - `{prefix}.scales`
/// - `{prefix}.biases`
///
/// The `group_size` and `bits` should come from `QuantConfig::quant_for(prefix)`.
/// The returned embedding is frozen and can also serve as the output-projection
/// (`lm_head`) via `as_linear`.
pub fn make_quantized_embedding(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedEmbedding> {
    let weight = get_weight(weights, &format!("{prefix}.weight"))?;
    let scales = get_weight(weights, &format!("{prefix}.scales"))?;
    let biases = get_weight(weights, &format!("{prefix}.biases"))?;

    let inner = nn::Embedding {
        weight: Param::new(weight),
    };

    let mut qe = nn::QuantizedEmbedding {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };
    qe.freeze_parameters(true);

    Ok(qe)
}

/// Build a frozen `nn::QuantizedLinear` from pre-loaded weight tensors.
///
/// Expects the following keys in `weights`:
/// - `{prefix}.weight`
/// - `{prefix}.scales`
/// - `{prefix}.biases`
///
/// The `group_size` and `bits` should come from `QuantConfig::quant_for(prefix)`.
pub fn make_quantized_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedLinear> {
    let weight = get_weight(weights, &format!("{prefix}.weight"))?;
    let scales = get_weight(weights, &format!("{prefix}.scales"))?;
    let biases = get_weight(weights, &format!("{prefix}.biases"))?;

    let inner = nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(None),
    };

    let mut ql = nn::QuantizedLinear {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };
    ql.freeze_parameters(true);

    Ok(ql)
}
