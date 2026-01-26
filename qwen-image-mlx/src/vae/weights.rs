//! VAE weight loading
//!
//! Loads weights from safetensors into the VAE decoder.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{Array, module::Param};
use safetensors::SafeTensors;

use super::{QwenVAE, QwenImageDecoder3D};

/// Load VAE weights from safetensors file
pub fn load_vae_weights(
    vae: &mut QwenVAE,
    weights: &HashMap<String, Array>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load post_quant_conv
    load_conv3d_weights(&mut vae.post_quant_conv, weights, "post_quant_conv")?;

    // Load decoder
    load_decoder_weights(&mut vae.decoder, weights)?;

    Ok(())
}

fn load_conv3d_weights(
    conv: &mut super::QwenImageCausalConv3D,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Weight: safetensor [out, kT, kH, kW, in] -> our format [out, in, kT, kH, kW]
    if let Some(w) = weights.get(&format!("{}.conv3d.weight", prefix)) {
        // Transpose from [out, kT, kH, kW, in] to [out, in, kT, kH, kW]
        let w_transposed = w.transpose_axes(&[0, 4, 1, 2, 3])?;
        *conv.weight = w_transposed;
    }
    if let Some(b) = weights.get(&format!("{}.conv3d.bias", prefix)) {
        *conv.bias = Some(b.clone());
    }
    Ok(())
}

fn load_rms_norm_weights(
    norm: &mut super::QwenImageRMSNorm,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(w) = weights.get(&format!("{}.weight", prefix)) {
        *norm.weight = w.clone();
    }
    Ok(())
}

fn load_resblock_weights(
    block: &mut super::QwenImageResBlock3D,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    load_rms_norm_weights(&mut block.norm1, weights, &format!("{}.norm1", prefix))?;
    load_conv3d_weights(&mut block.conv1, weights, &format!("{}.conv1", prefix))?;
    load_rms_norm_weights(&mut block.norm2, weights, &format!("{}.norm2", prefix))?;
    load_conv3d_weights(&mut block.conv2, weights, &format!("{}.conv2", prefix))?;

    if let Some(ref mut skip) = block.skip {
        load_conv3d_weights(skip, weights, &format!("{}.skip_conv", prefix))?;
    }
    Ok(())
}

fn load_attention_weights(
    attn: &mut super::QwenImageAttentionBlock3D,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    load_rms_norm_weights(&mut attn.norm, weights, &format!("{}.norm", prefix))?;

    // to_qkv: 2D conv weight [out, kH, kW, in] - MLX Conv2d expects same format
    if let Some(w) = weights.get(&format!("{}.to_qkv.weight", prefix)) {
        *attn.to_qkv.weight = w.clone();
    }
    if let Some(b) = weights.get(&format!("{}.to_qkv.bias", prefix)) {
        attn.to_qkv.bias = Param::new(Some(b.clone()));
    }

    // proj: 2D conv weight [out, kH, kW, in]
    if let Some(w) = weights.get(&format!("{}.proj.weight", prefix)) {
        *attn.proj.weight = w.clone();
    }
    if let Some(b) = weights.get(&format!("{}.proj.bias", prefix)) {
        attn.proj.bias = Param::new(Some(b.clone()));
    }

    Ok(())
}

fn load_midblock_weights(
    mid: &mut super::QwenImageMidBlock3D,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    for (i, resnet) in mid.resnets.iter_mut().enumerate() {
        load_resblock_weights(resnet, weights, &format!("{}.resnets.{}", prefix, i))?;
    }
    for (i, attn) in mid.attentions.iter_mut().enumerate() {
        load_attention_weights(attn, weights, &format!("{}.attentions.{}", prefix, i))?;
    }
    Ok(())
}

fn load_resample_weights(
    resample: &mut super::QwenImageResample3D,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // resample_conv is a 2D conv: weight [out, kH, kW, in]
    let key = format!("{}.resample_conv.weight", prefix);
    if let Some(w) = weights.get(&key) {
        // Conv2d in MLX expects [out, kH, kW, in] - same as safetensor format
        eprintln!("[DEBUG] Loading {}: shape={:?}", key, w.shape());
        *resample.resample_conv.weight = w.clone();
    } else {
        eprintln!("[WARNING] Missing weight: {}", key);
    }
    let bias_key = format!("{}.resample_conv.bias", prefix);
    if let Some(b) = weights.get(&bias_key) {
        eprintln!("[DEBUG] Loading {}: shape={:?}", bias_key, b.shape());
        resample.resample_conv.bias = Param::new(Some(b.clone()));
    }

    // time_conv is a 3D causal conv
    if let Some(ref mut time_conv) = resample.time_conv {
        load_conv3d_weights(time_conv, weights, &format!("{}.time_conv", prefix))?;
    }

    Ok(())
}

fn load_upblock_weights(
    block: &mut super::QwenImageUpBlock3D,
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    for (i, resnet) in block.resnets.iter_mut().enumerate() {
        load_resblock_weights(resnet, weights, &format!("{}.resnets.{}", prefix, i))?;
    }
    for (i, upsampler) in block.upsamplers.iter_mut().enumerate() {
        load_resample_weights(upsampler, weights, &format!("{}.upsamplers.{}", prefix, i))?;
    }
    Ok(())
}

fn load_decoder_weights(
    decoder: &mut QwenImageDecoder3D,
    weights: &HashMap<String, Array>,
) -> Result<(), Box<dyn std::error::Error>> {
    // conv_in
    load_conv3d_weights(&mut decoder.conv_in, weights, "decoder.conv_in")?;

    // mid_block
    load_midblock_weights(&mut decoder.mid_block, weights, "decoder.mid_block")?;

    // up_blocks
    for (i, block) in decoder.up_blocks.iter_mut().enumerate() {
        load_upblock_weights(block, weights, &format!("decoder.up_block{}", i))?;
    }

    // norm_out
    load_rms_norm_weights(&mut decoder.norm_out, weights, "decoder.norm_out")?;

    // conv_out
    load_conv3d_weights(&mut decoder.conv_out, weights, "decoder.conv_out")?;

    Ok(())
}

/// Load VAE from model directory
pub fn load_vae_from_dir(model_dir: impl AsRef<Path>) -> Result<QwenVAE, Box<dyn std::error::Error>> {
    let vae_path = model_dir.as_ref().join("vae/0.safetensors");

    println!("  Loading VAE weights from: {}", vae_path.display());

    let data = std::fs::read(&vae_path)?;
    let tensors = SafeTensors::deserialize(&data)?;

    let mut weights = HashMap::new();
    for (name, tensor) in tensors.tensors() {
        let array = Array::try_from(tensor)?;
        weights.insert(name.to_string(), array);
    }

    println!("  Loaded {} VAE tensors", weights.len());

    let mut vae = QwenVAE::new()?;
    load_vae_weights(&mut vae, &weights)?;

    println!("  VAE weights loaded successfully!");

    Ok(vae)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vae_creation() {
        let vae = QwenVAE::new().unwrap();
        assert_eq!(vae.decoder.up_blocks.len(), 4);
    }
}
