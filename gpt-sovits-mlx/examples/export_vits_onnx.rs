//! ONNX Weight Inspector and Experimental Patcher
//!
//! **WARNING**: The patching feature is EXPERIMENTAL and may produce silent output.
//! Shape-based matching cannot reliably map weights to ONNX's anonymous layer names.
//!
//! **Recommended**: Use `scripts/export_finetuned_onnx.py` for reliable ONNX export.
//!
//! This tool is useful for:
//! - Listing/inspecting weights in an ONNX model
//! - Understanding ONNX model structure
//!
//! # Usage
//!
//! ```bash
//! # List weights in an ONNX model (RECOMMENDED USE)
//! cargo run --release --example export_vits_onnx -- \
//!     --base ~/.OminiX/models/gpt-sovits-mlx/vits.onnx \
//!     --list
//!
//! # EXPERIMENTAL: Attempt to patch weights (may not work correctly!)
//! cargo run --release --example export_vits_onnx -- \
//!     --base ~/.OminiX/models/gpt-sovits-mlx/vits.onnx \
//!     --weights /tmp/vits_finetuned.generator.safetensors \
//!     --output /tmp/finetuned_vits.onnx
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use onnx_pb::{ModelProto, TensorProto};
use prost_old::Message;
use safetensors::SafeTensors;

#[derive(Parser)]
#[command(name = "export_vits_onnx")]
#[command(about = "Patch ONNX VITS model with finetuned weights")]
struct Args {
    /// Base ONNX model to patch
    #[arg(long)]
    base: PathBuf,

    /// Finetuned weights (safetensors format)
    #[arg(long)]
    weights: Option<PathBuf>,

    /// Output ONNX model path
    #[arg(long)]
    output: Option<PathBuf>,

    /// List weights in the ONNX model
    #[arg(long)]
    list: bool,

    /// Verbose output
    #[arg(long, short)]
    verbose: bool,
}

/// Load ONNX model from file
fn load_onnx_model(path: &PathBuf) -> Result<ModelProto> {
    let file = File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer)?;

    // prost 0.6 API
    Message::decode(&*buffer).context("Failed to decode ONNX model")
}

/// Save ONNX model to file
fn save_onnx_model(model: &ModelProto, path: &PathBuf) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);

    // prost 0.6 API
    let mut buffer = Vec::new();
    model.encode(&mut buffer)?;
    writer.write_all(&buffer)?;
    Ok(())
}

/// Get shape from ONNX tensor
fn get_tensor_shape(tensor: &TensorProto) -> Vec<i64> {
    tensor.dims.clone()
}

/// Get tensor data as f32 array
#[allow(dead_code)]
fn get_tensor_data_f32(tensor: &TensorProto) -> Vec<f32> {
    // ONNX tensors can store data in different ways
    if !tensor.float_data.is_empty() {
        return tensor.float_data.clone();
    }

    // Raw data format
    if !tensor.raw_data.is_empty() {
        let data: Vec<f32> = tensor
            .raw_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        return data;
    }

    Vec::new()
}

/// Set tensor data from f32 array
fn set_tensor_data_f32(tensor: &mut TensorProto, data: &[f32]) {
    // Clear existing data
    tensor.float_data.clear();
    tensor.raw_data.clear();

    // Store as raw data (more compact)
    let mut raw_data = Vec::with_capacity(data.len() * 4);
    for &val in data {
        raw_data.extend_from_slice(&val.to_le_bytes());
    }
    tensor.raw_data = raw_data;
}

/// Compute weight from weight_g and weight_v (weight normalization)
/// weight = g * v / ||v||
fn compute_weight_norm(weight_g: &[f32], weight_v: &[f32], shape: &[i64]) -> Vec<f32> {
    // For Conv1d: shape = [out_channels, in_channels, kernel_size]
    // Norm is computed over axes (1, 2), keeping axis 0
    let out_channels = shape[0] as usize;
    let inner_size = (shape[1] * shape[2]) as usize;

    let mut result = vec![0.0f32; weight_v.len()];

    for out_ch in 0..out_channels {
        // Compute ||v|| for this output channel
        let start = out_ch * inner_size;
        let end = start + inner_size;
        let v_slice = &weight_v[start..end];

        let v_norm: f32 = v_slice.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-12;
        let g = weight_g[out_ch];
        let scale = g / v_norm;

        // Apply: weight = g * v / ||v||
        for i in 0..inner_size {
            result[start + i] = weight_v[start + i] * scale;
        }
    }

    result
}

/// Map safetensors key to ONNX initializer name (named keys)
fn map_safetensors_to_onnx(key: &str) -> String {
    // Safetensors: "dec.ups.0.weight_g" -> ONNX: "vits.dec.ups.0.weight"
    format!("vits.{}", key.replace(".weight_g", ".weight").replace(".weight_v", ".weight"))
}

/// Convert ONNX node path to safetensors key format
/// "/dec/resblocks.0/convs1.0/Conv" -> "dec.resblocks.0.convs1.0.weight"
/// "/dec/ups.0/ConvTranspose" -> "dec.ups.0.weight"
fn onnx_node_to_safetensors_key(node_name: &str) -> Option<String> {
    // Remove leading slash and trailing op type
    let path = node_name.trim_start_matches('/');

    // Split by '/' and filter out the op type (Conv, ConvTranspose, etc)
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() < 2 {
        return None;
    }

    // Remove last part (op type like "Conv" or "ConvTranspose")
    let layer_parts = &parts[..parts.len() - 1];

    // Join with dots and add .weight suffix
    let key = format!("{}.weight", layer_parts.join("."));
    Some(key)
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load base ONNX model
    println!("Loading base ONNX model from {:?}", args.base);
    let mut model = load_onnx_model(&args.base)?;

    let graph = model
        .graph
        .as_mut()
        .context("ONNX model has no graph")?;

    // Build map of ONNX initializers by name
    let mut onnx_weights: HashMap<String, usize> = HashMap::new();

    for (idx, init) in graph.initializer.iter().enumerate() {
        onnx_weights.insert(init.name.clone(), idx);
    }

    println!("ONNX model has {} initializers", onnx_weights.len());

    // List mode
    if args.list {
        println!("\nONNX Initializers:");
        println!("{:-<80}", "");
        for init in &graph.initializer {
            let shape = get_tensor_shape(init);
            let size: i64 = shape.iter().product();
            println!(
                "  {:60} shape={:?} ({} params)",
                init.name,
                shape,
                size
            );
        }

        // Analyze graph structure
        println!("\n\n=== Graph Analysis ===");
        println!("Total nodes: {}", graph.node.len());

        // Find Conv nodes and their weight inputs
        println!("\n=== Conv Nodes → Weight Mapping ===\n");
        let init_names: std::collections::HashSet<_> = graph.initializer.iter()
            .map(|i| i.name.as_str())
            .collect();

        let mut conv_with_init = Vec::new();
        let mut conv_without_init = Vec::new();

        for node in &graph.node {
            if node.op_type == "Conv" && node.input.len() >= 2 {
                let weight_name = &node.input[1];
                if init_names.contains(weight_name.as_str()) {
                    conv_with_init.push((node.name.clone(), weight_name.clone()));
                } else {
                    conv_without_init.push((node.name.clone(), weight_name.clone()));
                }
            }
        }

        println!("Conv nodes with initializer weights: {}", conv_with_init.len());
        println!("Conv nodes with computed weights: {}", conv_without_init.len());

        println!("\n--- Conv with computed (non-initializer) weights ---");
        for (i, (node_name, weight_name)) in conv_without_init.iter().enumerate() {
            if i < 50 {
                println!("  {:3}: node={:60} weight_input={}", i, node_name, weight_name);
            }
        }

        // Check what creates the onnx::Conv_XXXXX initializers
        println!("\n--- Anonymous Conv initializers ---");
        let mut anon_convs: Vec<_> = graph.initializer.iter()
            .filter(|i| i.name.starts_with("onnx::Conv"))
            .collect();
        anon_convs.sort_by_key(|i| {
            i.name.trim_start_matches("onnx::Conv_").parse::<i32>().unwrap_or(0)
        });
        println!("Total: {}", anon_convs.len());
        for (i, init) in anon_convs.iter().enumerate() {
            if i < 20 {
                println!("  {}: {} {:?}", i, init.name, init.dims);
            }
        }

        // Build full mapping: anonymous initializer -> node path
        println!("\n--- Full Anonymous Initializer → Node Mapping ---");
        let mut anon_to_node: HashMap<String, String> = HashMap::new();
        for init in &graph.initializer {
            if init.name.starts_with("onnx::") {
                for node in &graph.node {
                    if node.input.contains(&init.name) {
                        anon_to_node.insert(init.name.clone(), node.name.clone());
                        break;
                    }
                }
            }
        }

        // Group by module (dec, flow, etc)
        let mut dec_mappings = Vec::new();
        let mut flow_mappings = Vec::new();
        let mut other_mappings = Vec::new();

        for (init_name, node_name) in &anon_to_node {
            if node_name.contains("/dec/") {
                dec_mappings.push((init_name.clone(), node_name.clone()));
            } else if node_name.contains("/flow/") {
                flow_mappings.push((init_name.clone(), node_name.clone()));
            } else {
                other_mappings.push((init_name.clone(), node_name.clone()));
            }
        }

        dec_mappings.sort_by(|a, b| a.1.cmp(&b.1));
        flow_mappings.sort_by(|a, b| a.1.cmp(&b.1));

        println!("\nDecoder mappings ({}):", dec_mappings.len());
        for (init, node) in &dec_mappings {
            println!("  {} -> {}", init, node);
        }

        println!("\nFlow mappings ({}):", flow_mappings.len());
        for (i, (init, node)) in flow_mappings.iter().enumerate() {
            if i < 20 {
                println!("  {} -> {}", init, node);
            }
        }

        return Ok(());
    }

    // Need weights and output for patching
    let weights_path = args.weights.context("--weights required for patching")?;
    let output_path = args.output.context("--output required for patching")?;

    // Load finetuned weights from safetensors
    println!("Loading finetuned weights from {:?}", weights_path);
    let weights_data = std::fs::read(&weights_path)?;
    let safetensors = SafeTensors::deserialize(&weights_data)?;

    // =========================================================================
    // Build graph-based mapping: ONNX anonymous initializer -> safetensors key
    // =========================================================================
    println!("\nBuilding graph-based weight mapping...");

    // Map: safetensors key -> ONNX anonymous initializer name
    let mut safetensors_to_anon_onnx: HashMap<String, String> = HashMap::new();

    // Scan graph nodes to find which initializer each node uses
    for node in &graph.node {
        // Handle Conv and ConvTranspose nodes
        if (node.op_type == "Conv" || node.op_type == "ConvTranspose") && node.input.len() >= 2 {
            let weight_init_name = &node.input[1];

            // Only process anonymous keys
            if weight_init_name.starts_with("onnx::") {
                // Convert node name to safetensors key format
                if let Some(safetensors_key) = onnx_node_to_safetensors_key(&node.name) {
                    safetensors_to_anon_onnx.insert(safetensors_key, weight_init_name.clone());
                }
            }
        }
    }

    println!("Found {} anonymous weight mappings from graph", safetensors_to_anon_onnx.len());
    if args.verbose {
        for (st_key, onnx_key) in safetensors_to_anon_onnx.iter().take(10) {
            println!("  {} -> {}", st_key, onnx_key);
        }
    }

    // Group weight_g/weight_v pairs (for weights with weight normalization)
    let mut weight_pairs: HashMap<String, (Option<Vec<f32>>, Option<Vec<f32>>, Vec<i64>)> =
        HashMap::new();

    for (name, tensor) in safetensors.tensors() {
        if name.ends_with(".weight_g") {
            let base = name.trim_end_matches(".weight_g");
            let data: Vec<f32> = tensor
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let entry = weight_pairs.entry(base.to_string()).or_insert((None, None, Vec::new()));
            entry.0 = Some(data);
        } else if name.ends_with(".weight_v") {
            let base = name.trim_end_matches(".weight_v");
            let data: Vec<f32> = tensor
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let shape: Vec<i64> = tensor.shape().iter().map(|&x| x as i64).collect();
            let entry = weight_pairs.entry(base.to_string()).or_insert((None, None, Vec::new()));
            entry.1 = Some(data);
            entry.2 = shape;
        }
    }

    let mut patched = 0;
    let mut skipped = 0;
    let mut used_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // Patch weight-normalized layers (if weight_g/weight_v pairs exist)
    println!("\nPatching weight-normalized layers:");
    for (base, (g, v, shape)) in &weight_pairs {
        if let (Some(weight_g), Some(weight_v)) = (g, v) {
            let weight = compute_weight_norm(weight_g, weight_v, shape);
            let safetensors_key = format!("{}.weight", base);
            let onnx_name = map_safetensors_to_onnx(&format!("{}.weight_g", base));

            // Try named ONNX key first
            if let Some(&idx) = onnx_weights.get(&onnx_name) {
                let init = &mut graph.initializer[idx];
                let onnx_shape = get_tensor_shape(init);

                if onnx_shape == *shape {
                    set_tensor_data_f32(init, &weight);
                    patched += 1;
                    used_indices.insert(idx);
                    if args.verbose {
                        println!("  PATCHED (named): {} -> {} {:?}", base, onnx_name, shape);
                    }
                    continue;
                }
            }

            // Try graph-based mapping for anonymous keys
            if let Some(anon_onnx_name) = safetensors_to_anon_onnx.get(&safetensors_key) {
                if let Some(&idx) = onnx_weights.get(anon_onnx_name) {
                    if !used_indices.contains(&idx) {
                        let init = &mut graph.initializer[idx];
                        set_tensor_data_f32(init, &weight);
                        patched += 1;
                        used_indices.insert(idx);
                        if args.verbose {
                            println!("  PATCHED (graph): {} -> {} {:?}", base, anon_onnx_name, shape);
                        }
                        continue;
                    }
                }
            }

            if args.verbose {
                println!("  SKIP (no match): {} shape={:?}", base, shape);
            }
            skipped += 1;
        }
    }

    // Patch regular weights (including merged weights from Rust training)
    println!("\nPatching regular weights:");
    for (name, tensor) in safetensors.tensors() {
        // Skip weight_g/weight_v (processed above)
        if name.ends_with(".weight_g") || name.ends_with(".weight_v") {
            continue;
        }

        // Only patch decoder and ref_enc weights
        if !name.starts_with("dec.") && !name.starts_with("ref_enc.") {
            continue;
        }

        // Skip if this is a weight that was handled by weight normalization
        let base_name = name.trim_end_matches(".weight");
        if weight_pairs.contains_key(base_name) {
            continue;
        }

        let data: Vec<f32> = tensor
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let shape: Vec<i64> = tensor.shape().iter().map(|&x| x as i64).collect();

        let onnx_name = format!("vits.{}", name);

        // Try named ONNX key first
        if let Some(&idx) = onnx_weights.get(&onnx_name) {
            if !used_indices.contains(&idx) {
                let init = &mut graph.initializer[idx];
                let onnx_shape = get_tensor_shape(init);

                if onnx_shape == shape {
                    set_tensor_data_f32(init, &data);
                    patched += 1;
                    used_indices.insert(idx);
                    if args.verbose {
                        println!("  PATCHED (named): {} -> {} {:?}", name, onnx_name, shape);
                    }
                    continue;
                } else {
                    if args.verbose {
                        println!(
                            "  SHAPE MISMATCH: {} ft={:?} vs onnx={:?}",
                            name, shape, onnx_shape
                        );
                    }
                }
            }
        }

        // Try graph-based mapping for anonymous keys
        if let Some(anon_onnx_name) = safetensors_to_anon_onnx.get(&name.to_string()) {
            if let Some(&idx) = onnx_weights.get(anon_onnx_name) {
                if !used_indices.contains(&idx) {
                    let init = &mut graph.initializer[idx];
                    let onnx_shape = get_tensor_shape(init);

                    if onnx_shape == shape {
                        set_tensor_data_f32(init, &data);
                        patched += 1;
                        used_indices.insert(idx);
                        if args.verbose {
                            println!("  PATCHED (graph): {} -> {} {:?}", name, anon_onnx_name, shape);
                        }
                        continue;
                    }
                }
            }
        }

        if args.verbose {
            println!("  SKIP (no match): {} shape={:?}", name, shape);
        }
        skipped += 1;
    }

    println!("\n{}", "=".repeat(60));
    println!("Summary:");
    println!("  Patched: {}", patched);
    println!("  Skipped: {}", skipped);

    // Save patched model
    println!("\nSaving patched model to {:?}", output_path);
    save_onnx_model(&model, &output_path)?;

    let size = std::fs::metadata(&output_path)?.len();
    println!("Output size: {:.1} MB", size as f64 / 1024.0 / 1024.0);
    println!("Done!");

    Ok(())
}
