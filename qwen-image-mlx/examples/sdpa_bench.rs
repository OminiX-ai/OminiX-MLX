//! Synthetic A/B benchmark: fused SDPA (fast::scaled_dot_product_attention)
//! vs eager matmul->scale->softmax->matmul attention, at image-scale and
//! video-scale sequence lengths. Correctness (max abs diff) + timing.
//!
//! Run: cargo run --release --package qwen-image-mlx --example sdpa_bench

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{ops, Array};

fn make_qkv(batch: usize, heads: usize, seq: usize, head_dim: usize) -> (Array, Array, Array) {
    let shape: [i32; 4] = [batch as i32, heads as i32, seq as i32, head_dim as i32];
    let q = mlx_rs::random::normal::<f32>(&shape, None, None, None).unwrap();
    let k = mlx_rs::random::normal::<f32>(&shape, None, None, None).unwrap();
    let v = mlx_rs::random::normal::<f32>(&shape, None, None, None).unwrap();
    (q, k, v)
}

fn eager_attention(q: &Array, k: &Array, v: &Array, scale: f32) -> Array {
    // mirrors the ORIGINAL qwen-image attention.rs: matmul -> multiply -> softmax -> matmul
    let k_t = k.transpose_axes(&[0, 1, 3, 2]).unwrap();
    let mut attn = ops::matmul(q, &k_t).unwrap();
    attn = ops::multiply(&attn, &Array::from_f32(scale)).unwrap();
    attn = ops::softmax_axis(&attn, -1, None).unwrap();
    ops::matmul(&attn, v).unwrap()
}

fn bench(label: &str, batch: usize, heads: usize, seq: usize, head_dim: usize, iters: usize) {
    let (q, k, v) = make_qkv(batch, heads, seq, head_dim);
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // ---- correctness ----
    let fused = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, scale, None).unwrap();
    let eager = eager_attention(&q, &k, &v, scale);
    fused.eval().unwrap();
    eager.eval().unwrap();
    let diff = ops::abs(&ops::subtract(&fused, &eager).unwrap()).unwrap();
    diff.eval().unwrap();
    let max_diff = ops::max(&diff, None).unwrap();
    max_diff.eval().unwrap();
    let max_diff = max_diff.item::<f32>();
    // sample an output value to prove it's not garbage
    let sample = fused.index((0_i32, 0_i32, 0_i32, 0_i32));
    sample.eval().unwrap();
    let sample_val = sample.item::<f32>();

    // ---- timing: eager ----
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let out = eager_attention(&q, &k, &v, scale);
        out.eval().unwrap();
    }
    let t_eager = t0.elapsed().as_secs_f64() / iters as f64;

    // ---- timing: fused ----
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let out = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, scale, None).unwrap();
        out.eval().unwrap();
    }
    let t_fused = t0.elapsed().as_secs_f64() / iters as f64;

    let speedup = t_eager / t_fused;
    println!(
        "{label:<38} seq={seq:<6} eager={t_eager:>8.3}ms fused={t_fused:>8.3}ms speedup={speedup:>5.2}x  max_diff={max_diff:.3e} sample={sample_val:.4}",
    );
}

fn main() {
    println!("=== SDPA fused vs eager — correctness + timing (b=1, heads=24, head_dim=128, fp32) ===");
    println!();
    // image-scale sequences (typical 1024x1024 latent ~ 1024-2048 tokens)
    bench("image  512 tok", 1, 24, 512, 128, 20);
    bench("image 1024 tok", 1, 24, 1024, 128, 20);
    bench("image 2048 tok", 1, 24, 2048, 128, 10);
    // video-scale sequences (MiniMax-style: hundreds of latent frames)
    bench("video 4096 tok", 1, 24, 4096, 128, 5);
    bench("video 8192 tok", 1, 24, 8192, 128, 3);
    bench("video 16384 tok", 1, 24, 16384, 128, 2);
}
