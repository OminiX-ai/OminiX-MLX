//! M0 single decoder-layer forward parity vs the mlx-vlm golden dump.
//!
//! Builds Gemma 4 12B decoder layer 0 (sliding, standard RoPE) and layer 5
//! (full/global, ProportionalRoPE), runs each on the recorded `hidden_in.npy`
//! with a plain causal mask, and compares to `layer{0,5}_out.npy`.
//!
//! Usage:
//!   cargo run -p gemma4-mlx --example layer_parity --release -- <MODEL_DIR> <GOLDEN_DIR>
//!
//! NOTE: run single-threaded (MLX is not thread-safe). cargo examples are
//! single-threaded by default.

use std::path::Path;

use gemma4_mlx::block::TransformerBlock;
use gemma4_mlx::config::{ModelArgs, QuantConfig};
use gemma4_mlx::mask::full_causal_mask;
use gemma4_mlx::weights::load_all_weights;

use mlx_rs::Array;

/// Read a float32 .npy into (data, shape).
fn read_npy_f32(path: &Path) -> anyhow::Result<(Vec<f32>, Vec<i32>)> {
    let bytes = std::fs::read(path)?;
    let npy = npyz::NpyFile::new(&bytes[..])?;
    let shape: Vec<i32> = npy.shape().iter().map(|&d| d as i32).collect();
    let data: Vec<f32> = npy.into_vec::<f32>()?;
    Ok((data, shape))
}

/// max-abs and mean-abs difference between two equal-length f32 slices.
fn diffs(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    let mut max = 0.0f32;
    let mut sum = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max {
            max = d;
        }
        sum += d as f64;
    }
    (max, (sum / a.len() as f64) as f32)
}

fn stats(a: &[f32]) -> (f32, f32) {
    let mean = a.iter().map(|&v| v as f64).sum::<f64>() / a.len() as f64;
    let var = a.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / a.len() as f64;
    let absmax = a.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    (var.sqrt() as f32, absmax)
}

fn main() -> anyhow::Result<()> {
    let mut args_it = std::env::args().skip(1);
    let model_dir = args_it.next().expect("usage: layer_parity <MODEL_DIR> <GOLDEN_DIR>");
    let golden_dir = args_it.next().expect("usage: layer_parity <MODEL_DIR> <GOLDEN_DIR>");
    let model_dir = Path::new(&model_dir);
    let golden_dir = Path::new(&golden_dir);

    let cfg = std::fs::read_to_string(model_dir.join("config.json"))?;
    let args = ModelArgs::from_config_str(&cfg)?;
    let quant = QuantConfig::from_config_str(&cfg)?
        .ok_or_else(|| anyhow::anyhow!("no quantization config in config.json"))?;

    println!("loading weights from {} ...", model_dir.display());
    let weights = load_all_weights(model_dir)?;
    println!("loaded {} tensors", weights.len());

    // Input
    let (h_data, h_shape) = read_npy_f32(&golden_dir.join("hidden_in.npy"))?;
    println!("hidden_in shape {:?}", h_shape);
    let l = h_shape[1];
    let hidden_in = Array::from_slice(&h_data, &h_shape);

    // Mask. NOTE on the golden reference: the dump script does
    //   mask = create_causal_mask(L, 0).astype(mx.float32)
    // mlx_lm's create_causal_mask returns a BOOL mask (true=visible); casting it
    // to float32 yields 1.0/0.0 values, and mx.fast.scaled_dot_product_attention
    // treats a *float* mask as ADDITIVE. So the reference effectively adds +1.0 to
    // visible-position scores and 0.0 (NOT -inf) to "masked" ones — i.e. for L<window
    // it does not actually causally mask, it just biases. To match the golden bytes
    // exactly we must replicate that float 1.0/0.0 additive mask.
    //
    // (For real inference the bool mask from full_causal_mask is the correct causal
    // mask — see the `_bool_mask` below, kept to document the intended semantics.)
    let bool_mask = full_causal_mask(l, 0)?;
    let _bool_mask = &bool_mask;
    let mask = bool_mask.as_type::<f32>()?;

    let mut all_pass = true;
    for layer_idx in [0i32, 5i32] {
        let kind = args.layer_types[layer_idx as usize];
        let (gold, gshape) = read_npy_f32(&golden_dir.join(format!("layer{layer_idx}_out.npy")))?;

        let mut block = TransformerBlock::from_weights(&weights, &args, &quant, layer_idx)?;
        let out = block.forward(&hidden_in, &mask)?;
        out.eval()?;
        let out_data: Vec<f32> = out.as_slice::<f32>().to_vec();

        let (max_d, mean_d) = diffs(&out_data, &gold);
        let (gstd, gabsmax) = stats(&gold);

        println!(
            "\n=== layer {layer_idx} ({kind:?}) ===\n  out shape {:?} (golden {:?})\n  golden std={gstd:.4} absmax={gabsmax:.4}\n  max-abs-diff = {max_d:.6e}\n  mean-abs-diff = {mean_d:.6e}",
            out.shape(),
            gshape,
        );

        // Tolerance for 4-bit quant slack.
        let pass = max_d < 1e-2;
        println!("  PASS (max-abs-diff < 1e-2): {pass}");
        all_pass &= pass;
    }

    println!("\nOVERALL: {}", if all_pass { "PASS" } else { "FAIL" });
    Ok(())
}
