//! Decode-correctness validation: our KV-cache greedy decode must be token-for-token
//! identical to a no-cache re-forward of the growing sequence (both deterministic greedy).
//! This proves the KV-cache + RoPE-offset + per-layer decode-mask machinery is correct
//! WITHOUT depending on bit-matching an external bf16 reference.
//!
//! (Separately, vs mlx-vlm greedy the first token matches; later tokens may diverge at
//! close-call argmax positions due to bf16 accumulation across 48 layers — see the M1
//! finding. That is benign and does not indicate a decode bug; this example isolates the
//! decode machinery from that dtype effect.)
//!
//! Usage: cargo run -p gemma4-mlx --example decode_consistency --release -- <MODEL_DIR> [K]
use mlx_rs::Array;
use gemma4_mlx::{load_model, generate_greedy};

fn argmax(logits: &Array) -> i32 {
    let v: Vec<f32> = logits.as_type::<f32>().unwrap().as_slice::<f32>().to_vec();
    let mut bi = 0i32;
    let mut bv = f32::MIN;
    for (i, &x) in v.iter().enumerate() {
        if x > bv { bv = x; bi = i as i32; }
    }
    bi
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).expect("usage: decode_consistency <MODEL_DIR> [K]");
    let k: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let prompt = [2i32, 1024, 2048, 4096, 8192, 16384];

    let mut model = load_model(&dir)?;

    // A) KV-cache greedy decode.
    let cache_ids = generate_greedy(&mut model, &prompt, k, &[1, 106])?;

    // B) No-cache greedy: re-forward the growing sequence each step.
    let mut seq = prompt.to_vec();
    let mut nocache_ids = Vec::with_capacity(k);
    for _ in 0..k {
        let toks = Array::from_slice(&seq, &[1, seq.len() as i32]);
        let logits = model.forward(&toks, true)?;
        let next = argmax(&logits);
        if next == 1 || next == 106 { break; }
        nocache_ids.push(next);
        seq.push(next);
    }

    println!("cache   greedy: {cache_ids:?}");
    println!("nocache greedy: {nocache_ids:?}");
    let matched = cache_ids == nocache_ids;
    println!("TOKEN-FOR-TOKEN MATCH (cache == no-cache): {matched}");
    if !matched {
        eprintln!("DECODE INCONSISTENT — KV-cache path diverges from no-cache re-forward");
        std::process::exit(1);
    }
    println!("PASS — KV-cache decode is correct (matches no-cache re-forward).");
    Ok(())
}
