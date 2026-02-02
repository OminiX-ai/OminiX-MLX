# Qwen-Image MLX Optimization Findings

*Last updated: 2026-01-29*

## Executive Summary

This document captures all optimization work done on the Qwen-Image MLX full-precision implementation, including successful optimizations, failed attempts, and critical lessons learned about MLX performance.

## Performance Summary

### Diffusion Time Only (excluding model loading)

| Steps | Diffusion Time | Per Step | Quality | vs Baseline |
|-------|----------------|----------|---------|-------------|
| 20 | ~80s | ~4.0s | Excellent | 1.8x faster |
| **10** | **~40s** | **~4.0s** | **Excellent** | **3.6x faster** |
| 8 | ~32s | ~4.0s | Very Good | 4.5x faster |

*Baseline: 145s for 20 steps before optimization*

### Total Time (including ~25s overhead for loading/encoding/VAE)

| Steps | Total Time | Quality | Recommendation |
|-------|------------|---------|----------------|
| 20 | ~105s | Excellent | Best quality |
| **10** | **~65s** | **Excellent** | **Recommended** |
| 8 | ~57s | Very Good | Fast mode |

*Note: Per-step time (~4s) is consistent. Thermal throttling can add variance.*

See [STEP_REDUCTION_RESULTS.md](./STEP_REDUCTION_RESULTS.md) for detailed analysis.

---

## Successful Optimizations

### 1. RoPE Pre-computation
**Impact: HIGH**

- Pre-compute rotary position embeddings once at initialization
- Eliminates redundant computation across all 60 transformer blocks × 20 steps
- Frequencies cached in `TimestepEmbedder.cached_freqs`

```rust
// In TimestepEmbedder::new()
let freqs: Vec<f32> = (0..half_dim)
    .map(|i| (-(i as f32) * (10000.0f32.ln()) / half_dim as f32).exp())
    .collect();
let cached_freqs = Array::from_slice(&freqs, &[1, half_dim]);
```

### 2. Fast SDPA (Scaled Dot-Product Attention)
**Impact: HIGH**

- Use MLX's optimized `mlx_rs::fast::scaled_dot_product_attention`
- Hardware-accelerated with Flash Attention-style memory efficiency
- Replaces manual attention computation

```rust
// Before (slow)
let scores = ops::matmul(&q, &k.transpose(&[-1, -2])?)?;
let attn = ops::softmax(&scores, -1)?;
let out = ops::matmul(&attn, &v)?;

// After (fast)
let out = fast::scaled_dot_product_attention(&q, &k, &v, scale, None)?;
```

### 3. Fast RMS Norm
**Impact: MEDIUM**

- Use MLX's `mlx_rs::fast::rms_norm` instead of manual implementation
- Fused kernel eliminates intermediate allocations

### 4. Lazy Evaluation Preservation
**Impact: CRITICAL**

The single most important optimization is **not breaking MLX's lazy evaluation**:

- **Do NOT call `eval()` on every step** - forces GPU sync, breaks batching
- **Do NOT call `mlx_clear_cache()` frequently** - adds overhead
- Only eval when necessary (progress reporting every 5 steps, final output)

```rust
// BAD - eval every step
for step in 0..num_steps {
    latents = transformer.forward(&latents, ...)?;
    mlx_rs::transforms::eval([&latents])?;  // DON'T DO THIS
}

// GOOD - eval only when needed
for step in 0..num_steps {
    latents = transformer.forward(&latents, ...)?;
    if (step + 1) % 5 == 0 {
        mlx_rs::transforms::eval([&latents])?;
        println!("Step {}", step + 1);
    }
}
```

---

## Failed/Rejected Optimizations

### 1. Fused Modulate Metal Kernel
**Status**: Working but slower than MLX ops

| Metric | Manual Implementation | Fused Kernel |
|--------|----------------------|--------------|
| Time per step | **4.31s** | 4.52s |

**Implementation**: Custom Metal kernel with parallel reduction for LayerNorm + modulation

```metal
// Kernel performs: (1 + scale) * LayerNorm(x) + shift
// Uses parallel reduction for mean/variance computation
```

**Why it's slower**: MLX's built-in ops with lazy evaluation are already highly optimized. The custom kernel adds overhead that outweighs the fusion benefit.

**Code location**: `mlx-rs-core/src/metal_kernels.rs`

**Configuration**:
```rust
// qwen_full_precision.rs
const USE_FUSED_MODULATE: bool = false;  // Keep disabled for best performance
```

#### Metal Kernel Bug Fix History

The kernel initially produced NaN values due to incorrect grid/threadgroup sizing:

**Original Bug**:
```rust
// WRONG: Only gives ceil(num_rows/256) threadgroups
mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, num_rows, 1, 1);
```

**Fix**:
```rust
// CORRECT: Gives exactly num_rows threadgroups
let total_threads = num_rows * 256;
mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, total_threads, 1, 1);
```

**Explanation**: In MLX's Metal kernel API, `set_grid` specifies total threads, not threadgroups. To get `num_rows` threadgroups (one per row), we need `num_rows * threads_per_group` total threads.

### 2. Cache Clearing Optimizations
**Status**: Rejected - adds overhead

| Attempted | Result |
|-----------|--------|
| `mlx_clear_cache()` every 5 steps | Slowed from 76s to 86s |
| `mlx_clear_cache()` every 15 blocks | Additional slowdown |
| `eval()` on every step | Slowed to ~110s |

**Why it failed**: Forces GPU synchronization, prevents MLX's lazy evaluation from batching operations.

### 3. Constant Caching with OnceLock
**Status**: Build failed

```rust
// ATTEMPTED (doesn't work)
static MODULATE_EPS: std::sync::OnceLock<Array> = std::sync::OnceLock::new();
```

**Error**: `Array` doesn't implement `Sync`, so it can't be stored in `OnceLock`.

**Lesson**: MLX Array is not thread-safe. Even if it worked, it would break lazy evaluation by creating Arrays outside the computation graph.

### 4. Moving Constants Outside Loop
**Status**: Slower

```rust
// ATTEMPTED (slower)
let cfg_arr = Array::from_f32(cfg_scale);  // Outside loop
for step in 0..num_steps {
    let scaled = ops::multiply(&diff, &cfg_arr)?;
    ...
}

// BETTER (faster)
for step in 0..num_steps {
    let cfg_arr = Array::from_f32(cfg_scale);  // Inside loop
    let scaled = ops::multiply(&diff, &cfg_arr)?;
    ...
}
```

**Why**: Creating Arrays inside the loop allows MLX to incorporate them into the computation graph for better fusion. Arrays outside the graph fragment optimization.

### 5. ANE (Apple Neural Engine) Offloading
**Status**: Not possible with MLX

- MLX only supports CPU and GPU (Metal)
- ANE requires CoreML framework
- Would need significant rewrite to use ANE

---

## Deep Code Review Findings

### Performance Issues Identified

| Issue | Location | Severity | Status |
|-------|----------|----------|--------|
| Redundant Array constants | `modulate_manual` | Medium | Won't fix (MLX handles) |
| Clone in AdaLayerNormMod | `forward()` | Low | Won't fix (refcounted) |
| Scalar constants in loop | diffusion loop | Low | Won't fix (MLX handles) |
| Inefficient RoPE reshape | `apply_rope` | Medium | Future optimization |

### Numerical Stability Notes

1. **Epsilon values**: Use `1e-6` consistently for LayerNorm, `1e-12` for rescaling
2. **DiT value explosion**: Hidden states reaching [-51M, +57M] is **normal** - LayerNorm handles it
3. **RoPE precision**: Interleaved format may accumulate errors; monitor for long sequences

### Code Quality Improvements Made

1. Removed unused import (`fused_modulate`)
2. Cleaned up comments
3. Simplified modulate function

---

## Architecture Reference

### DiT (Diffusion Transformer) Value Ranges

| Stage | Value Range | Notes |
|-------|-------------|-------|
| After img_in | [-15, +16] | Normal |
| After 60 blocks | [-51M, +57M] | Expected explosion |
| After norm_out | [-16, +14] | LayerNorm normalizes |
| After proj_out | [-4.4, +4.4] | Final output |

### Modulation Formula

```
output = (1 + scale) * LayerNorm(x) + shift
```

Where:
- `LayerNorm` has no learnable parameters (`elementwise_affine=False`)
- `scale` and `shift` come from timestep embedding projection
- Called 4× per block × 60 blocks = 240 times per forward pass

### Timestep Convention

- Pass sigma (in [0, 1] range) to transformer
- `get_timestep_embedding` internally scales by 1000
- Identical to mflux and diffusers conventions

---

## Memory Usage

| Component | Memory |
|-----------|--------|
| Full precision model | ~13GB |
| Peak during generation | ~15-16GB |
| After text encoder release | ~12-13GB |

**Tip**: Release text encoder after encoding to save ~2-3GB:
```rust
drop(text_encoder);
// Note: Don't call mlx_clear_cache() - it adds overhead
```

---

## Configuration Reference

### `qwen_full_precision.rs`

```rust
/// Use fused Metal kernel for modulation
/// Set to false for better performance (MLX's lazy evaluation is already efficient)
const USE_FUSED_MODULATE: bool = false;
```

---

## Benchmarking Commands

```bash
# Full precision, 20 steps, 512x512
cargo run --release --example generate_fp32 -- \
  --prompt "a fluffy cat" \
  --height 512 --width 512 \
  --steps 20 \
  --output output.png

# Quick test (3 steps)
cargo run --release --example generate_fp32 -- \
  --prompt "a fluffy cat" \
  --height 512 --width 512 \
  --steps 3 \
  --output output_test.png
```

---

## Key Lessons Learned

### 1. Don't Fight MLX's Lazy Evaluation

MLX builds a computation graph and optimizes it holistically. Attempting to "optimize" by:
- Moving constants outside loops
- Caching Arrays in static variables
- Forcing early evaluation

...often **hurts** performance because it fragments the graph.

### 2. Custom Kernels Have Overhead

Custom Metal kernels are only worth it for operations MLX doesn't optimize well. For common operations (LayerNorm, elementwise ops), MLX's built-in ops are already highly optimized.

### 3. Profile Before Optimizing

What seems like an optimization can actually slow things down. Always benchmark before and after.

### 4. Parallel Reduction is Tricky

Metal grid/threadgroup sizing requires careful attention:
- `set_grid` specifies total threads, not threadgroups
- Always use power-of-2 threadgroup sizes for reductions
- Initialize shared memory explicitly

---

## Files Modified

| File | Changes |
|------|---------|
| `mlx-rs-core/src/metal_kernels.rs` | Fixed fused_modulate kernel (grid size bug) |
| `qwen-image-mlx/src/qwen_full_precision.rs` | Removed cache clearing, cleaned up code |
| `qwen-image-mlx/examples/generate_fp32.rs` | Reduced eval frequency |
| `qwen-image-mlx/Cargo.toml` | Added mlx-sys dependency |

---

## Future Optimization Ideas

1. **Quantized Model**: Use 4-bit/8-bit weights for 2-3x faster inference
2. **Batch Processing**: Process multiple images in parallel
3. **Smaller Resolutions**: 256x256 for faster iteration
4. **Fewer Steps**: 10-15 steps with better scheduler (DDIM, DPM++)
5. **Model Distillation**: Train smaller student model
6. **Profile with Instruments**: Use Apple's profiler to identify actual bottlenecks
7. **Async Text Encoding**: Overlap text encoding with latent preparation

---

## References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [mflux Implementation](https://github.com/filipstrand/mflux)
- [flux2.c Metal Implementation](https://github.com/antirez/flux2.c)
- [Diffusers Qwen-Image](https://huggingface.co/docs/diffusers/api/pipelines/flux)
