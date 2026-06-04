//! M1-Task4 full-model forward parity vs mlx-vlm golden dump.
//!
//! Runs the complete 48-layer Gemma 4 12B model on the recorded `tokens.npy`
//! and compares our logits at the last position to `logits_last.npy`.
//!
//! Usage:
//!   cargo run -p gemma4-mlx --example model_parity --release -- <MODEL_DIR> <GOLDEN_DIR>
//!
//! Defaults:
//!   MODEL_DIR  = /Users/alan0x/models/gemma-4-12B-it-4bit
//!   GOLDEN_DIR = /tmp/gemma4_logits
//!
//! NOTE: run single-threaded (MLX is not thread-safe).

use std::path::Path;

use mlx_rs::Array;

/// Read a float32 .npy into (data, shape).
fn read_npy_f32(path: &Path) -> anyhow::Result<(Vec<f32>, Vec<i32>)> {
    let bytes = std::fs::read(path)?;
    let npy = npyz::NpyFile::new(&bytes[..])?;
    let shape: Vec<i32> = npy.shape().iter().map(|&d| d as i32).collect();
    let data: Vec<f32> = npy.into_vec::<f32>()?;
    Ok((data, shape))
}

/// Read an int32 .npy into (data, shape).
fn read_npy_i32(path: &Path) -> anyhow::Result<(Vec<i32>, Vec<i32>)> {
    let bytes = std::fs::read(path)?;
    let npy = npyz::NpyFile::new(&bytes[..])?;
    let shape: Vec<i32> = npy.shape().iter().map(|&d| d as i32).collect();
    let data: Vec<i32> = npy.into_vec::<i32>()?;
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

/// Return the indices of the top-k values (largest first).
fn top_k(values: &[f32], k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_unstable_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap());
    indices.truncate(k);
    indices
}

fn main() -> anyhow::Result<()> {
    let mut args_it = std::env::args().skip(1);
    let model_dir = args_it
        .next()
        .unwrap_or_else(|| "/Users/alan0x/models/gemma-4-12B-it-4bit".to_string());
    let golden_dir = args_it
        .next()
        .unwrap_or_else(|| "/tmp/gemma4_logits".to_string());
    let model_dir = Path::new(&model_dir);
    let golden_dir = Path::new(&golden_dir);

    println!("model_dir  : {}", model_dir.display());
    println!("golden_dir : {}", golden_dir.display());

    // ── 1. Read tokens ─────────────────────────────────────────────────────────
    let (tok_data, tok_shape) = read_npy_i32(&golden_dir.join("tokens.npy"))?;
    println!("\ntokens shape {:?}  values {:?}", tok_shape, &tok_data);
    let seq_len = tok_shape[1] as usize;
    let tokens = Array::from_slice(&tok_data, &tok_shape);

    // ── 2. Read reference logits ────────────────────────────────────────────────
    let (ref_logits, ref_shape) = read_npy_f32(&golden_dir.join("logits_last.npy"))?;
    println!("ref logits shape {:?}  len {}", ref_shape, ref_logits.len());
    let vocab = ref_logits.len(); // 262144

    let ref_argmax = ref_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let ref_top5 = top_k(&ref_logits, 5);
    println!("ref argmax : {ref_argmax}  top5 : {ref_top5:?}");

    // ── 3. Load model ───────────────────────────────────────────────────────────
    println!("\nloading model from {} ...", model_dir.display());
    let mut model = gemma4_mlx::load_model(model_dir)?;
    println!("model loaded  (seq_len={seq_len}, vocab={vocab})");

    // ── 4. Forward ─────────────────────────────────────────────────────────────
    println!("running forward (last_only=true) ...");
    let logits_raw = model.forward(&tokens, true)?; // [1, 1, vocab]
    logits_raw.eval()?;
    println!("logits shape : {:?}", logits_raw.shape());

    // Cast to f32 in case the model outputs bfloat16.
    let logits_f32 = logits_raw.as_type::<f32>()?;
    logits_f32.eval()?;
    let our_logits: Vec<f32> = logits_f32.as_slice::<f32>().to_vec();
    println!("our logits len : {}", our_logits.len());

    // ── 5. Compare ─────────────────────────────────────────────────────────────
    let rust_argmax = our_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let our_top5 = top_k(&our_logits, 5);

    // Both slices cover exactly `vocab` elements; ref_logits is [1, vocab] flattened.
    let (max_abs_diff, mean_abs_diff) = diffs(&our_logits, &ref_logits);

    let our_absmax = our_logits.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let ref_absmax = ref_logits.iter().fold(0.0f32, |m, &v| m.max(v.abs()));

    println!("\n========== MODEL PARITY RESULTS ==========");
    println!("  seq_len        : {seq_len}");
    println!("  vocab          : {vocab}");
    println!("  rust_argmax    : {rust_argmax}");
    println!("  ref_argmax     : {ref_argmax}  (expected 2818)");
    println!("  MATCH          : {}", rust_argmax == ref_argmax);
    println!("  our top5       : {our_top5:?}");
    println!("  ref top5       : {ref_top5:?}  (expected [2818, 714, 107, 1562, 236747])");
    println!("  our absmax     : {our_absmax:.4}");
    println!("  ref absmax     : {ref_absmax:.4}  (expected ~29.5)");
    println!("  max-abs-diff   : {max_abs_diff:.6e}");
    println!("  mean-abs-diff  : {mean_abs_diff:.6e}");
    println!("==========================================");

    if rust_argmax == ref_argmax {
        println!("\nPASS  — rust_argmax ({rust_argmax}) == ref_argmax ({ref_argmax})");
    } else {
        println!(
            "\nFAIL  — rust_argmax ({rust_argmax}) != ref_argmax ({ref_argmax}); \
             our top5 {our_top5:?} vs ref top5 {ref_top5:?}"
        );
        std::process::exit(1);
    }

    Ok(())
}
