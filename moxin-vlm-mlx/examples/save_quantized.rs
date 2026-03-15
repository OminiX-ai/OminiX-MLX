//! Save 8-bit quantized Moxin-7B VLM weights to safetensors.
//!
//! Usage:
//!   cargo run --release -p moxin-vlm-mlx --example save_quantized -- \
//!     --model ./models/Moxin-7B-VLM-hf \
//!     --output ./models/Moxin-7B-VLM-8bit-mlx \
//!     --bits 8

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use clap::Parser;
use mlx_rs::module::ModuleParameters;
use mlx_rs::Array;

use moxin_vlm_mlx::load_model;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: String,
    #[arg(long)]
    output: String,
    #[arg(long, default_value = "8")]
    bits: i32,
    #[arg(long, default_value = "64")]
    group_size: i32,
}

/// Collect parameters from a ModuleParameters impl with a key prefix.
fn collect_params(
    module: &impl ModuleParameters,
    prefix: &str,
    out: &mut HashMap<String, Array>,
) {
    for (key, value) in module.parameters().flatten() {
        let full_key = if prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}.{}", prefix, key)
        };
        out.insert(full_key, value.clone());
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load BF16 model
    let vlm = load_model(&args.model)?;

    // Quantize LLM decoder
    eprintln!("Quantizing to {} bits (group_size={})...", args.bits, args.group_size);
    let vlm = vlm.quantize(args.group_size, args.bits)?;
    eprintln!("Quantization complete.");

    // Collect all parameters
    let mut all_params: HashMap<String, Array> = HashMap::new();

    // Vision encoders (BF16)
    collect_params(&vlm.dino.norm, "vision_backbone.featurizer.norm", &mut all_params);
    // DINOv2 patch_embed conv
    all_params.insert(
        "vision_backbone.featurizer.patch_embed.proj.weight".to_string(),
        vlm.dino.patch_embed.weight.as_ref().transpose_axes(&[0, 3, 1, 2])?,  // MLX [O,H,W,I] -> PyTorch [O,I,H,W]
    );
    if let Some(b) = vlm.dino.patch_embed.bias.as_ref() {
        all_params.insert(
            "vision_backbone.featurizer.patch_embed.proj.bias".to_string(),
            b.clone(),
        );
    }
    if let Some(cls) = &vlm.dino.cls_token {
        all_params.insert("vision_backbone.featurizer.cls_token".to_string(), cls.as_ref().clone());
    }
    if let Some(reg) = &vlm.dino.reg_tokens {
        all_params.insert("vision_backbone.featurizer.reg_token".to_string(), reg.as_ref().clone());
    }
    all_params.insert("vision_backbone.featurizer.pos_embed".to_string(), vlm.dino.pos_embed.as_ref().clone());
    for (i, block) in vlm.dino.blocks.iter().enumerate() {
        collect_params(block, &format!("vision_backbone.featurizer.blocks.{}", i), &mut all_params);
    }

    // SigLIP
    collect_params(&vlm.siglip.norm, "vision_backbone.fused_featurizer.norm", &mut all_params);
    all_params.insert(
        "vision_backbone.fused_featurizer.patch_embed.proj.weight".to_string(),
        vlm.siglip.patch_embed.weight.as_ref().transpose_axes(&[0, 3, 1, 2])?,
    );
    if let Some(b) = vlm.siglip.patch_embed.bias.as_ref() {
        all_params.insert(
            "vision_backbone.fused_featurizer.patch_embed.proj.bias".to_string(),
            b.clone(),
        );
    }
    all_params.insert("vision_backbone.fused_featurizer.pos_embed".to_string(), vlm.siglip.pos_embed.as_ref().clone());
    for (i, block) in vlm.siglip.blocks.iter().enumerate() {
        collect_params(block, &format!("vision_backbone.fused_featurizer.blocks.{}", i), &mut all_params);
    }

    // Projector
    collect_params(&vlm.projector, "projector", &mut all_params);

    // LLM decoder (quantized)
    all_params.insert(
        "language_model.model.embed_tokens.weight".to_string(),
        vlm.embed_tokens.weight.as_ref().clone(),
    );
    for (i, layer) in vlm.layers.iter().enumerate() {
        let lp = format!("language_model.model.layers.{}", i);
        collect_params(&layer.self_attn.q_proj, &format!("{}.self_attn.q_proj", lp), &mut all_params);
        collect_params(&layer.self_attn.k_proj, &format!("{}.self_attn.k_proj", lp), &mut all_params);
        collect_params(&layer.self_attn.v_proj, &format!("{}.self_attn.v_proj", lp), &mut all_params);
        collect_params(&layer.self_attn.o_proj, &format!("{}.self_attn.o_proj", lp), &mut all_params);
        collect_params(&layer.mlp.gate_proj, &format!("{}.mlp.gate_proj", lp), &mut all_params);
        collect_params(&layer.mlp.up_proj, &format!("{}.mlp.up_proj", lp), &mut all_params);
        collect_params(&layer.mlp.down_proj, &format!("{}.mlp.down_proj", lp), &mut all_params);
        all_params.insert(
            format!("{}.input_layernorm.weight", lp),
            layer.input_layernorm.weight.as_ref().clone(),
        );
        all_params.insert(
            format!("{}.post_attention_layernorm.weight", lp),
            layer.post_attention_layernorm.weight.as_ref().clone(),
        );
    }
    all_params.insert(
        "language_model.model.norm.weight".to_string(),
        vlm.norm.weight.as_ref().clone(),
    );
    collect_params(&vlm.lm_head, "language_model.lm_head", &mut all_params);

    // Eval all arrays before saving
    let refs: Vec<&Array> = all_params.values().collect();
    mlx_rs::transforms::eval(refs)?;

    // Create output directory
    let out_dir = Path::new(&args.output);
    std::fs::create_dir_all(out_dir)?;

    // Save as safetensors
    let out_path = out_dir.join("model.safetensors");
    eprintln!("Saving {} parameters to {:?}...", all_params.len(), out_path);
    Array::save_safetensors(&all_params, None, &out_path)?;
    eprintln!("Saved.");

    // Copy config, tokenizer files
    let src = Path::new(&args.model);
    for fname in &[
        "config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "generation_config.json",
        "preprocessor_config.json",
    ] {
        let src_file = src.join(fname);
        if src_file.exists() {
            std::fs::copy(&src_file, out_dir.join(fname))?;
            eprintln!("Copied {}", fname);
        }
    }

    // Write quantization config
    let quant_config = format!(
        r#"{{"quantization": {{"group_size": {}, "bits": {}}}}}"#,
        args.group_size, args.bits
    );
    std::fs::write(out_dir.join("quantize_config.json"), quant_config)?;
    eprintln!("Wrote quantize_config.json");

    let size = std::fs::metadata(&out_path)?.len();
    eprintln!("Done! Model size: {:.1} GB", size as f64 / 1e9);

    Ok(())
}
