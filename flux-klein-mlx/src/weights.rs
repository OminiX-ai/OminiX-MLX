//! Weight loading utilities for FLUX models
//!
//! Handles loading weights from safetensors files and mapping them to the model structure.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use safetensors::SafeTensors;

use crate::Result;

/// Load weights from a safetensors file
pub fn load_safetensors(path: impl AsRef<Path>) -> Result<HashMap<String, Array>> {
    let path = path.as_ref();
    let data = std::fs::read(path).map_err(|e| {
        crate::FluxError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Failed to read safetensors file: {}", e),
        ))
    })?;

    let tensors = SafeTensors::deserialize(&data).map_err(|e| {
        crate::FluxError::WeightLoading(format!("Failed to parse safetensors: {}", e))
    })?;

    let mut weights = HashMap::new();
    for (name, tensor) in tensors.tensors() {
        let array = Array::try_from(tensor).map_err(|e| {
            crate::FluxError::WeightLoading(format!("Failed to convert tensor '{}': {:?}", name, e))
        })?;
        weights.insert(name.to_string(), array);
    }

    Ok(weights)
}

/// Load weights from multiple safetensors files (sharded weights)
pub fn load_sharded_safetensors(paths: &[impl AsRef<Path>]) -> Result<HashMap<String, Array>> {
    let mut all_weights = HashMap::new();

    for path in paths {
        let weights = load_safetensors(path)?;
        all_weights.extend(weights);
    }

    Ok(all_weights)
}

/// Sanitize FLUX weight keys to match our model structure
///
/// Transforms weight keys from the original FLUX checkpoint format to our naming convention.
pub fn sanitize_flux_weights(weights: HashMap<String, Array>) -> HashMap<String, Array> {
    let mut sanitized = HashMap::new();

    for (key, value) in weights {
        // Strip common prefixes
        let mut new_key = key.clone();

        // Remove "model.diffusion_model." prefix if present
        if new_key.starts_with("model.diffusion_model.") {
            new_key = new_key[22..].to_string();
        }

        // Rename .scale to .weight for RMSNorm layers
        new_key = new_key.replace(".scale", ".weight");

        // Convert MLP layer naming
        // ".img_mlp.0." -> ".img_mlp.fc1."
        // ".img_mlp.2." -> ".img_mlp.fc2."
        new_key = new_key.replace(".img_mlp.0.", ".img_mlp.fc1.");
        new_key = new_key.replace(".img_mlp.2.", ".img_mlp.fc2.");
        new_key = new_key.replace(".txt_mlp.0.", ".txt_mlp.fc1.");
        new_key = new_key.replace(".txt_mlp.2.", ".txt_mlp.fc2.");

        // Convert modulation layer naming
        new_key = new_key.replace(".adaLN_modulation.1.", ".mod_layer.linear.");
        new_key = new_key.replace(".img_mod.1.", ".img_mod.linear.");
        new_key = new_key.replace(".txt_mod.1.", ".txt_mod.linear.");
        // Some checkpoints use "lin" instead of "linear"
        new_key = new_key.replace(".img_mod.lin.", ".img_mod.linear.");
        new_key = new_key.replace(".txt_mod.lin.", ".txt_mod.linear.");
        new_key = new_key.replace(".mod_layer.lin.", ".mod_layer.linear.");

        // Convert attention naming
        new_key = new_key.replace(".img_attn.qkv.", ".img_attn.qkv.");
        new_key = new_key.replace(".img_attn.proj.", ".img_attn.proj.");
        new_key = new_key.replace(".txt_attn.qkv.", ".txt_attn.qkv.");
        new_key = new_key.replace(".txt_attn.proj.", ".txt_attn.proj.");

        // Attention norm naming - checkpoint already has correct format (norm.query_norm, norm.key_norm)
        // No conversion needed for double blocks

        // Convert time/vector embedder naming (handle both with and without leading dot)
        new_key = new_key.replace("time_in.in_layer.", "time_in.linear1.");
        new_key = new_key.replace("time_in.out_layer.", "time_in.linear2.");
        new_key = new_key.replace("vector_in.in_layer.", "vector_in.linear1.");
        new_key = new_key.replace("vector_in.out_layer.", "vector_in.linear2.");
        new_key = new_key.replace("guidance_in.in_layer.", "guidance_in.linear1.");
        new_key = new_key.replace("guidance_in.out_layer.", "guidance_in.linear2.");

        // Convert final layer (handle both with and without leading dot)
        new_key = new_key.replace("final_layer.adaLN_modulation.1.", "final_layer.ada_linear.");
        // Also handle mod_layer naming in final layer
        new_key = new_key.replace("final_layer.mod_layer.linear.", "final_layer.ada_linear.");

        // Single stream block naming differences
        // Checkpoint: single_blocks.X.modulation.lin -> Model: single_blocks.X.mod_layer.linear
        new_key = new_key.replace(".modulation.lin.", ".mod_layer.linear.");
        new_key = new_key.replace(".modulation.1.", ".mod_layer.linear.");

        // Single stream blocks have different norm naming:
        // Checkpoint: single_blocks.X.norm.query_norm -> Model: single_blocks.X.norm_q
        // Checkpoint: single_blocks.X.norm.key_norm -> Model: single_blocks.X.norm_k
        // But we must NOT change double_blocks.X.img_attn.norm.query_norm which is correct
        if new_key.starts_with("single_blocks.") && !new_key.contains("_attn.") {
            new_key = new_key.replace(".norm.query_norm.", ".norm_q.");
            new_key = new_key.replace(".norm.key_norm.", ".norm_k.");
        }

        sanitized.insert(new_key, value);
    }

    sanitized
}

/// Print weight keys for debugging
pub fn print_weight_keys(weights: &HashMap<String, Array>) {
    let mut keys: Vec<_> = weights.keys().collect();
    keys.sort();

    println!("Loaded {} weights:", keys.len());
    for key in keys.iter().take(50) {
        let shape = weights.get(*key).map(|a| format!("{:?}", a.shape())).unwrap_or_default();
        println!("  {} -> {}", key, shape);
    }
    if keys.len() > 50 {
        println!("  ... and {} more", keys.len() - 50);
    }
}

/// Map weights to model parameters
///
/// Returns a vector of (parameter_path, array) tuples suitable for load_weights
pub fn map_weights_to_params(
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Vec<(String, Array)> {
    let mut params = Vec::new();

    for (key, value) in weights {
        if key.starts_with(prefix) || prefix.is_empty() {
            params.push((key.clone(), value.clone()));
        }
    }

    params
}

/// Sanitize VAE weight keys to match our decoder model structure
///
/// Transforms weight keys from the HuggingFace diffusers VAE format to our naming convention.
/// Also transposes conv weights from OIHW (PyTorch) to OHWI (MLX) format.
pub fn sanitize_vae_weights(weights: HashMap<String, Array>) -> HashMap<String, Array> {
    let mut sanitized = HashMap::new();

    for (key, value) in weights {
        // Handle post_quant_conv (top-level, not under decoder.)
        if key.starts_with("post_quant_conv.") {
            // Keep as-is, just transpose if conv weight
            let final_value = if key.ends_with(".weight") && value.ndim() == 4 {
                value.transpose_axes(&[0, 2, 3, 1]).unwrap_or(value)
            } else {
                value
            };
            sanitized.insert(key.clone(), final_value);
            continue;
        }

        // Only keep decoder weights for the rest
        if !key.starts_with("decoder.") {
            continue;
        }

        // Remove "decoder." prefix
        let new_key = key[8..].to_string();

        // Convert diffusers naming to our naming convention
        let new_key = new_key
            // Middle block attention: mid_block.attentions.0. -> mid_block_attentions_0.
            .replace("mid_block.attentions.0.", "mid_block_attentions_0.")
            // Middle block resnets: mid_block.resnets.0. -> mid_block_resnets_0.
            .replace("mid_block.resnets.0.", "mid_block_resnets_0.")
            .replace("mid_block.resnets.1.", "mid_block_resnets_1.")
            // Attention to_out.0 -> to_out (remove the extra .0)
            .replace(".to_out.0.", ".to_out.")
            // Up blocks: upsamplers.0.conv -> upsamplers_0_conv
            .replace(".upsamplers.0.conv.", ".upsamplers_0_conv.");

        // Determine if this is a Linear or Conv2d weight
        // Linear weights have 2 dimensions, Conv2d have 4
        let final_value = if new_key.ends_with(".weight") {
            if value.ndim() == 4 {
                // Conv2d: OIHW [out_ch, in_ch, H, W] -> OHWI [out_ch, H, W, in_ch]
                value.transpose_axes(&[0, 2, 3, 1]).unwrap_or(value)
            } else {
                // Linear: no transpose needed
                value
            }
        } else {
            value
        };

        sanitized.insert(new_key, final_value);
    }

    sanitized
}

/// Sanitize VAE encoder weights from diffusers format to match our model structure
pub fn sanitize_vae_encoder_weights(weights: HashMap<String, Array>) -> HashMap<String, Array> {
    let mut sanitized = HashMap::new();

    for (key, value) in weights {
        // Handle quant_conv (top-level, not under encoder.)
        if key.starts_with("quant_conv.") {
            let final_value = if key.ends_with(".weight") && value.ndim() == 4 {
                value.transpose_axes(&[0, 2, 3, 1]).unwrap_or(value)
            } else {
                value
            };
            sanitized.insert(key.clone(), final_value);
            continue;
        }

        // Only keep encoder weights
        if !key.starts_with("encoder.") {
            continue;
        }

        // Remove "encoder." prefix
        let new_key = key[8..].to_string();

        // Convert diffusers naming to our naming convention
        let new_key = new_key
            // Middle block attention: mid_block.attentions.0. -> mid_block_attentions_0.
            .replace("mid_block.attentions.0.", "mid_block_attentions_0.")
            // Middle block resnets: mid_block.resnets.0. -> mid_block_resnets_0.
            .replace("mid_block.resnets.0.", "mid_block_resnets_0.")
            .replace("mid_block.resnets.1.", "mid_block_resnets_1.")
            // Attention to_out.0 -> to_out (remove the extra .0)
            .replace(".to_out.0.", ".to_out.")
            // Down blocks: downsamplers.0.conv -> downsamplers_0_conv
            .replace(".downsamplers.0.conv.", ".downsamplers_0_conv.");

        // Transpose conv weights: OIHW -> OHWI
        let final_value = if new_key.ends_with(".weight") {
            if value.ndim() == 4 {
                value.transpose_axes(&[0, 2, 3, 1]).unwrap_or(value)
            } else {
                value
            }
        } else {
            value
        };

        sanitized.insert(new_key, final_value);
    }

    sanitized
}

/// Sanitize FLUX.2-klein diffusers format weights to match our model structure
///
/// FLUX.2-klein from HuggingFace uses diffusers naming convention which differs
/// significantly from the MLX FLUX.1 format.
///
/// Key differences:
/// - time_guidance_embed (not time_text_embed)
/// - No biases in the checkpoint
/// - SwiGLU MLP: fc1 = gate + up fused [2 * hidden, dim], fc2 = [dim, hidden]
/// - Single blocks: linear1 = QKV + gate + up fused [27648, 3072]
/// - Shared modulation weights across all blocks
pub fn sanitize_flux2_klein_weights(weights: HashMap<String, Array>) -> HashMap<String, Array> {
    use mlx_rs::ops;

    let mut sanitized = HashMap::new();

    // Collect weights for combining Q, K, V -> QKV
    let mut double_block_weights: HashMap<usize, HashMap<String, Array>> = HashMap::new();

    for (key, value) in &weights {
        // Context embedder -> txt_in (no bias in checkpoint)
        if key == "context_embedder.weight" {
            sanitized.insert("txt_in.weight".to_string(), value.clone());
            continue;
        }

        // x_embedder -> img_in (no bias in checkpoint)
        if key == "x_embedder.weight" {
            sanitized.insert("img_in.weight".to_string(), value.clone());
            continue;
        }

        // Time embedder - FLUX.2-klein uses time_guidance_embed
        if key.starts_with("time_guidance_embed.timestep_embedder.") {
            let new_key = key.replace("time_guidance_embed.timestep_embedder.", "time_in.");
            let new_key = new_key.replace("linear_1", "linear1").replace("linear_2", "linear2");
            sanitized.insert(new_key, value.clone());
            continue;
        }

        // Double stream modulation (shared across all 5 double blocks)
        if key == "double_stream_modulation_img.linear.weight" {
            for i in 0..5 {
                sanitized.insert(format!("double_blocks.{}.img_mod.linear.weight", i), value.clone());
            }
            continue;
        }
        if key == "double_stream_modulation_txt.linear.weight" {
            for i in 0..5 {
                sanitized.insert(format!("double_blocks.{}.txt_mod.linear.weight", i), value.clone());
            }
            continue;
        }

        // Single stream modulation (shared across all 20 single blocks)
        if key == "single_stream_modulation.linear.weight" {
            for i in 0..20 {
                sanitized.insert(format!("single_blocks.{}.mod_layer.linear.weight", i), value.clone());
            }
            continue;
        }

        // Norm out -> final_layer ada_linear
        if key == "norm_out.linear.weight" {
            sanitized.insert("final_layer.ada_linear.weight".to_string(), value.clone());
            continue;
        }

        // Proj out -> final_layer linear
        if key == "proj_out.weight" {
            sanitized.insert("final_layer.linear.weight".to_string(), value.clone());
            continue;
        }

        // Double stream transformer blocks
        if key.starts_with("transformer_blocks.") {
            let parts: Vec<&str> = key.split('.').collect();
            if parts.len() >= 2 {
                if let Ok(block_idx) = parts[1].parse::<usize>() {
                    let block_weights = double_block_weights.entry(block_idx).or_insert_with(HashMap::new);
                    block_weights.insert(key.clone(), value.clone());
                }
            }
            continue;
        }

        // Single transformer blocks
        if key.starts_with("single_transformer_blocks.") {
            let parts: Vec<&str> = key.split('.').collect();
            if parts.len() >= 2 {
                if let Ok(block_idx) = parts[1].parse::<usize>() {
                    let rest = parts[2..].join(".");

                    // attn.norm_q -> norm_q
                    if rest == "attn.norm_q.weight" {
                        sanitized.insert(format!("single_blocks.{}.norm_q.weight", block_idx), value.clone());
                    }
                    // attn.norm_k -> norm_k
                    else if rest == "attn.norm_k.weight" {
                        sanitized.insert(format!("single_blocks.{}.norm_k.weight", block_idx), value.clone());
                    }
                    // attn.to_qkv_mlp_proj -> linear1
                    // Shape is [27648, 3072] = QKV(9216) + gate(9216) + up(9216)
                    // Our model expects this full weight for linear1
                    else if rest == "attn.to_qkv_mlp_proj.weight" {
                        sanitized.insert(format!("single_blocks.{}.linear1.weight", block_idx), value.clone());
                    }
                    // attn.to_out -> linear2
                    // Shape is [3072, 12288] = [hidden, attn_out(3072) + mlp_out(9216)]
                    else if rest == "attn.to_out.weight" {
                        sanitized.insert(format!("single_blocks.{}.linear2.weight", block_idx), value.clone());
                    }
                }
            }
            continue;
        }
    }

    // Process double block weights - combine Q, K, V into QKV
    for (block_idx, block_weights) in double_block_weights {
        let prefix = format!("transformer_blocks.{}.", block_idx);

        // Image attention Q, K, V -> img_attn.qkv (combined)
        let img_q = block_weights.get(&format!("{}attn.to_q.weight", prefix));
        let img_k = block_weights.get(&format!("{}attn.to_k.weight", prefix));
        let img_v = block_weights.get(&format!("{}attn.to_v.weight", prefix));

        if let (Some(q), Some(k), Some(v)) = (img_q, img_k, img_v) {
            if let Ok(qkv) = ops::concatenate_axis(&[q.clone(), k.clone(), v.clone()], 0) {
                sanitized.insert(format!("double_blocks.{}.img_attn.qkv.weight", block_idx), qkv);
            }
        }

        // Image attention output projection
        if let Some(proj) = block_weights.get(&format!("{}attn.to_out.0.weight", prefix)) {
            sanitized.insert(format!("double_blocks.{}.img_attn.proj.weight", block_idx), proj.clone());
        }

        // Image attention norms
        if let Some(norm_q) = block_weights.get(&format!("{}attn.norm_q.weight", prefix)) {
            sanitized.insert(format!("double_blocks.{}.img_attn.norm.query_norm.weight", block_idx), norm_q.clone());
        }
        if let Some(norm_k) = block_weights.get(&format!("{}attn.norm_k.weight", prefix)) {
            sanitized.insert(format!("double_blocks.{}.img_attn.norm.key_norm.weight", block_idx), norm_k.clone());
        }

        // Text (added) attention Q, K, V -> txt_attn.qkv (combined)
        let txt_q = block_weights.get(&format!("{}attn.add_q_proj.weight", prefix));
        let txt_k = block_weights.get(&format!("{}attn.add_k_proj.weight", prefix));
        let txt_v = block_weights.get(&format!("{}attn.add_v_proj.weight", prefix));

        if let (Some(q), Some(k), Some(v)) = (txt_q, txt_k, txt_v) {
            if let Ok(qkv) = ops::concatenate_axis(&[q.clone(), k.clone(), v.clone()], 0) {
                sanitized.insert(format!("double_blocks.{}.txt_attn.qkv.weight", block_idx), qkv);
            }
        }

        // Text attention output projection
        if let Some(proj) = block_weights.get(&format!("{}attn.to_add_out.weight", prefix)) {
            sanitized.insert(format!("double_blocks.{}.txt_attn.proj.weight", block_idx), proj.clone());
        }

        // Text attention norms
        if let Some(norm_q) = block_weights.get(&format!("{}attn.norm_added_q.weight", prefix)) {
            sanitized.insert(format!("double_blocks.{}.txt_attn.norm.query_norm.weight", block_idx), norm_q.clone());
        }
        if let Some(norm_k) = block_weights.get(&format!("{}attn.norm_added_k.weight", prefix)) {
            sanitized.insert(format!("double_blocks.{}.txt_attn.norm.key_norm.weight", block_idx), norm_k.clone());
        }

        // Image MLP (ff) - SwiGLU format
        // ff.linear_in.weight [18432, 3072] = gate(9216) + up(9216) fused -> fc1
        // ff.linear_out.weight [3072, 9216] -> fc2
        if let Some(fc1) = block_weights.get(&format!("{}ff.linear_in.weight", prefix)) {
            sanitized.insert(format!("double_blocks.{}.img_mlp.fc1.weight", block_idx), fc1.clone());
        }
        if let Some(fc2) = block_weights.get(&format!("{}ff.linear_out.weight", prefix)) {
            sanitized.insert(format!("double_blocks.{}.img_mlp.fc2.weight", block_idx), fc2.clone());
        }

        // Text MLP (ff_context) - SwiGLU format
        if let Some(fc1) = block_weights.get(&format!("{}ff_context.linear_in.weight", prefix)) {
            sanitized.insert(format!("double_blocks.{}.txt_mlp.fc1.weight", block_idx), fc1.clone());
        }
        if let Some(fc2) = block_weights.get(&format!("{}ff_context.linear_out.weight", prefix)) {
            sanitized.insert(format!("double_blocks.{}.txt_mlp.fc2.weight", block_idx), fc2.clone());
        }

        // Note: FLUX.2-klein checkpoint doesn't have separate norm1/norm2 weights
        // The modulation is applied directly without pre-norm
    }

    sanitized
}

/// Sanitize FLUX.2-klein weights for the new klein_model architecture
///
/// Maps weights to klein_model.rs which uses:
/// - Shared modulation layers stored at top level
/// - Separate Q/K/V projections
/// - Different naming convention
pub fn sanitize_klein_model_weights(weights: HashMap<String, Array>) -> HashMap<String, Array> {
    let mut sanitized = HashMap::new();

    for (key, value) in &weights {
        // Input embeddings
        if key == "x_embedder.weight" {
            sanitized.insert("x_embedder.weight".to_string(), value.clone());
            continue;
        }
        if key == "context_embedder.weight" {
            sanitized.insert("context_embedder.weight".to_string(), value.clone());
            continue;
        }

        // Time embedder
        if key == "time_guidance_embed.timestep_embedder.linear_1.weight" {
            sanitized.insert("time_embed_1.weight".to_string(), value.clone());
            continue;
        }
        if key == "time_guidance_embed.timestep_embedder.linear_2.weight" {
            sanitized.insert("time_embed_2.weight".to_string(), value.clone());
            continue;
        }

        // Shared modulation layers
        if key == "double_stream_modulation_img.linear.weight" {
            sanitized.insert("double_mod_img.linear.weight".to_string(), value.clone());
            continue;
        }
        if key == "double_stream_modulation_txt.linear.weight" {
            sanitized.insert("double_mod_txt.linear.weight".to_string(), value.clone());
            continue;
        }
        if key == "single_stream_modulation.linear.weight" {
            sanitized.insert("single_mod.linear.weight".to_string(), value.clone());
            continue;
        }

        // Final layer
        if key == "norm_out.linear.weight" {
            sanitized.insert("norm_out.weight".to_string(), value.clone());
            continue;
        }
        if key == "proj_out.weight" {
            sanitized.insert("proj_out.weight".to_string(), value.clone());
            continue;
        }

        // Double stream transformer blocks
        if key.starts_with("transformer_blocks.") {
            let parts: Vec<&str> = key.split('.').collect();
            if parts.len() >= 3 {
                if let Ok(block_idx) = parts[1].parse::<usize>() {
                    let rest = parts[2..].join(".");

                    // Image attention
                    if rest == "attn.to_q.weight" {
                        sanitized.insert(format!("double_blocks.{}.img_to_q.weight", block_idx), value.clone());
                    } else if rest == "attn.to_k.weight" {
                        sanitized.insert(format!("double_blocks.{}.img_to_k.weight", block_idx), value.clone());
                    } else if rest == "attn.to_v.weight" {
                        sanitized.insert(format!("double_blocks.{}.img_to_v.weight", block_idx), value.clone());
                    } else if rest == "attn.norm_q.weight" {
                        sanitized.insert(format!("double_blocks.{}.img_norm_q.weight", block_idx), value.clone());
                    } else if rest == "attn.norm_k.weight" {
                        sanitized.insert(format!("double_blocks.{}.img_norm_k.weight", block_idx), value.clone());
                    } else if rest == "attn.to_out.0.weight" {
                        sanitized.insert(format!("double_blocks.{}.img_to_out.weight", block_idx), value.clone());
                    }
                    // Text attention
                    else if rest == "attn.add_q_proj.weight" {
                        sanitized.insert(format!("double_blocks.{}.txt_to_q.weight", block_idx), value.clone());
                    } else if rest == "attn.add_k_proj.weight" {
                        sanitized.insert(format!("double_blocks.{}.txt_to_k.weight", block_idx), value.clone());
                    } else if rest == "attn.add_v_proj.weight" {
                        sanitized.insert(format!("double_blocks.{}.txt_to_v.weight", block_idx), value.clone());
                    } else if rest == "attn.norm_added_q.weight" {
                        sanitized.insert(format!("double_blocks.{}.txt_norm_q.weight", block_idx), value.clone());
                    } else if rest == "attn.norm_added_k.weight" {
                        sanitized.insert(format!("double_blocks.{}.txt_norm_k.weight", block_idx), value.clone());
                    } else if rest == "attn.to_add_out.weight" {
                        sanitized.insert(format!("double_blocks.{}.txt_to_out.weight", block_idx), value.clone());
                    }
                    // Image MLP
                    else if rest == "ff.linear_in.weight" {
                        sanitized.insert(format!("double_blocks.{}.img_mlp_in.weight", block_idx), value.clone());
                    } else if rest == "ff.linear_out.weight" {
                        sanitized.insert(format!("double_blocks.{}.img_mlp_out.weight", block_idx), value.clone());
                    }
                    // Text MLP
                    else if rest == "ff_context.linear_in.weight" {
                        sanitized.insert(format!("double_blocks.{}.txt_mlp_in.weight", block_idx), value.clone());
                    } else if rest == "ff_context.linear_out.weight" {
                        sanitized.insert(format!("double_blocks.{}.txt_mlp_out.weight", block_idx), value.clone());
                    }
                }
            }
            continue;
        }

        // Single transformer blocks
        if key.starts_with("single_transformer_blocks.") {
            let parts: Vec<&str> = key.split('.').collect();
            if parts.len() >= 3 {
                if let Ok(block_idx) = parts[1].parse::<usize>() {
                    let rest = parts[2..].join(".");

                    if rest == "attn.norm_q.weight" {
                        sanitized.insert(format!("single_blocks.{}.norm_q.weight", block_idx), value.clone());
                    } else if rest == "attn.norm_k.weight" {
                        sanitized.insert(format!("single_blocks.{}.norm_k.weight", block_idx), value.clone());
                    } else if rest == "attn.to_qkv_mlp_proj.weight" {
                        sanitized.insert(format!("single_blocks.{}.to_qkv_mlp.weight", block_idx), value.clone());
                    } else if rest == "attn.to_out.weight" {
                        sanitized.insert(format!("single_blocks.{}.to_out.weight", block_idx), value.clone());
                    }
                }
            }
            continue;
        }
    }

    sanitized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_keys() {
        let mut weights = HashMap::new();
        weights.insert("model.diffusion_model.img_in.weight".to_string(), Array::zeros::<f32>(&[1]).unwrap());
        weights.insert("double_blocks.0.img_mlp.0.weight".to_string(), Array::zeros::<f32>(&[1]).unwrap());

        let sanitized = sanitize_flux_weights(weights);

        assert!(sanitized.contains_key("img_in.weight"));
        assert!(sanitized.contains_key("double_blocks.0.img_mlp.fc1.weight"));
    }
}
