# Qwen-Image MLX Optimization Techniques

This document details all optimization techniques applied to achieve 45% faster inference.

---

## Performance Summary

| Version | Time (20 steps) | Per Step | vs Baseline |
|---------|-----------------|----------|-------------|
| Baseline (unoptimized) | ~145s | ~7.2s | - |
| **Optimized** | **79.8s** | **3.99s** | **45% faster** |
| 4-bit Quantized | ~78s | ~3.9s | Reference |

---

## 1. RoPE Pre-computation

**Impact:** HIGH (20-40% faster)
**Location:** `examples/generate_fp32.rs:322-411`

### Problem
RoPE (Rotary Position Embedding) was computed inside the transformer forward pass, recalculated 60 blocks × 20 steps × 2 (CFG) = 2400 times per generation.

### Solution
Pre-compute RoPE embeddings once before the diffusion loop:

```rust
// BEFORE: Computed every forward pass
fn forward(&mut self, img: &Array, txt: &Array, ...) {
    let (cos, sin) = compute_rope(...);  // Computed 2400x
    ...
}

// AFTER: Computed ONCE, passed as parameter
let (img_rope_cos, img_rope_sin) = compute_image_rope(&img_ids)?;
let (txt_rope_cos, txt_rope_sin) = compute_text_rope(&txt_ids)?;

for step in 0..num_steps {
    // Reuse pre-computed RoPE
    transformer.forward(img, txt, timestep,
        Some((&img_rope_cos, &img_rope_sin)),
        Some((&txt_rope_cos, &txt_rope_sin)))?;
}
```

### Key Implementation Details
- Image RoPE: 3D positional encoding for patches (frame, height, width)
- Text RoPE: 1D positional encoding for token sequence
- Both use `theta=10000.0` base frequency

---

## 2. Timestep Frequency Caching

**Impact:** MEDIUM (contributes to overall 45% improvement)
**Location:** `src/qwen_full_precision.rs:425-456`

### Problem
`get_timestep_embedding()` recomputed sinusoidal frequencies every call (40× per generation: 20 steps × 2 CFG passes).

```rust
// BEFORE: Frequencies recomputed every call
fn get_timestep_embedding(t: &Array, dim: i32) -> Result<Array, Exception> {
    let half_dim = dim / 2;
    // RECOMPUTED 40x PER GENERATION!
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (-(i as f32) * (10000.0f32.ln()) / half_dim as f32).exp())
        .collect();
    let freqs = Array::from_slice(&freqs, &[1, half_dim]);
    ...
}
```

### Solution
Cache frequencies in the `TimestepEmbedder` struct:

```rust
#[derive(Debug, Clone, ModuleParameters)]
pub struct TimestepEmbedder {
    /// Pre-computed sinusoidal frequencies (cached for performance)
    pub cached_freqs: Array,

    #[param]
    pub linear_1: Linear,
    #[param]
    pub linear_2: Linear,
}

impl TimestepEmbedder {
    pub fn new(dim: i32) -> Result<Self, Exception> {
        // Pre-compute frequencies ONCE at initialization
        let half_dim = 128; // 256 / 2
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| (-(i as f32) * (10000.0f32.ln()) / half_dim as f32).exp())
            .collect();
        let cached_freqs = Array::from_slice(&freqs, &[1, half_dim]);

        Ok(Self {
            cached_freqs,
            linear_1: LinearBuilder::new(256, dim).bias(true).build()?,
            linear_2: LinearBuilder::new(dim, dim).bias(true).build()?,
        })
    }

    pub fn forward(&mut self, t: &Array) -> Result<Array, Exception> {
        // Use cached frequencies - no recomputation!
        let t_scaled = ops::multiply(t, &Array::from_f32(1000.0))?;
        let t_expanded = t_scaled.reshape(&[-1, 1])?;
        let args = ops::multiply(&t_expanded, &self.cached_freqs)?;

        let cos = ops::cos(&args)?;
        let sin = ops::sin(&args)?;
        let emb = ops::concatenate_axis(&[&cos, &sin], -1)?;

        let h = silu(&self.linear_1.forward(&emb)?)?;
        self.linear_2.forward(&h)
    }
}
```

**Source:** Adapted from `zimage-mlx/src/zimage_model.rs:402-450`

---

## 3. Hardware-Accelerated Attention (SDPA)

**Impact:** HIGH
**Location:** `src/qwen_full_precision.rs:237`

### Problem
Manual attention implementation has multiple separate operations:
```rust
// SLOW: Multiple separate operations
let scores = ops::matmul(&q, &k.transpose(&[-1, -2])?)?;
let scores = ops::multiply(&scores, &Array::from_f32(scale))?;
let weights = ops::softmax(&scores, -1)?;
let output = ops::matmul(&weights, &v)?;
```

### Solution
Use MLX's hardware-optimized `fast::scaled_dot_product_attention`:

```rust
use mlx_rs::fast;

// FAST: Single fused Metal kernel
let output = fast::scaled_dot_product_attention(
    queries,  // [batch, heads, seq, head_dim]
    keys,
    values,
    self.scale,
    None,  // No mask needed for joint attention
)?;
```

### Why It's Faster
- Single Metal kernel instead of 4+ operations
- Optimized memory access patterns
- Hardware-specific optimizations for Apple Silicon

---

## 4. Fast RMS Normalization

**Impact:** MEDIUM
**Location:** `src/qwen_full_precision.rs:104`

### Problem
Manual RMSNorm has multiple operations:
```rust
// SLOW: Multiple operations
let var = ops::mean_axis(&ops::multiply(x, x)?, -1, true)?;
let normalized = ops::divide(x, &ops::sqrt(&ops::add(&var, &eps)?)?)?;
ops::multiply(&normalized, &self.weight)
```

### Solution
Use MLX's `fast::rms_norm`:

```rust
pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
    fast::rms_norm(x, &*self.weight, self.eps)
}
```

---

## 5. Optimized GELU Activation

**Impact:** LOW-MEDIUM
**Location:** `src/qwen_full_precision.rs:74`

### Solution
Use MLX's pre-compiled `gelu_approximate`:

```rust
use mlx_rs::nn::gelu_approximate;

pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
    let hidden = self.proj_in.forward(x)?;
    let activated = gelu_approximate(&hidden)?;  // Optimized!
    self.proj_out.forward(&activated)
}
```

### Note on SwiGLU
Qwen-Image uses GELU, not SwiGLU. The fused SwiGLU kernel from `mixtral-mlx` is not applicable here.

---

## 6. Minimal eval() Strategy

**Impact:** HIGH (wrong placement causes 20%+ slowdown)
**Location:** `examples/generate_fp32.rs`

### Problem
MLX uses lazy evaluation. Calling `eval()` too frequently forces synchronization and kills performance.

### Solution
Only call `eval()` at critical synchronization points:

```rust
// CORRECT: Minimal eval() calls
txt_embed.eval()?;        // After text encoding (once)

for step in 0..num_steps {
    latent = denoise_step(...)?;
    latent.eval()?;       // After EACH step (20 times)
}

image.eval()?;            // After VAE decode (once)
```

### What NOT to Do
```rust
// WRONG: eval() inside transformer blocks - 20% SLOWER!
for block in &mut self.blocks {
    let output = block.forward(...)?;
    mlx_rs::transforms::eval([&output])?;  // DON'T DO THIS
}
```

---

## 7. Techniques That Did NOT Help

These were tested and found to hurt performance:

| Technique | Result | Notes |
|-----------|--------|-------|
| `fast::layer_norm` with None weights | **50% SLOWER** | Only use with actual weights |
| `eval()` every 10 blocks | **20% SLOWER** | Breaks lazy evaluation benefits |
| Batched CFG | No improvement | GPU already saturated |

---

## Architecture-Specific Notes

### Why qwen3-mlx Optimizations Don't Apply

Qwen-Image is a **diffusion transformer**, not an **autoregressive LLM**:

| qwen3-mlx Technique | Applicable? | Reason |
|---------------------|-------------|--------|
| KV Cache | No | Diffusion doesn't cache KV |
| Token-by-token generation | No | Diffusion processes all patches at once |
| Async token prefetching | Limited | Each step depends on previous |
| Fused SwiGLU | No | Qwen-Image uses GELU |

### What Transfers Between Models

| Technique | Transfers? | Source |
|-----------|------------|--------|
| RoPE pre-computation | Yes | flux-klein-mlx |
| Timestep freq caching | Yes | zimage-mlx |
| fast::SDPA | Yes | All MLX models |
| Minimal eval() | Yes | All MLX models |
| Custom Metal kernels | Yes (if applicable) | mixtral-mlx |

---

## Code References

| Optimization | File | Lines |
|--------------|------|-------|
| RoPE pre-computation | `examples/generate_fp32.rs` | 322-411 |
| Timestep caching | `src/qwen_full_precision.rs` | 425-456 |
| Fast SDPA | `src/qwen_full_precision.rs` | 237 |
| Fast RMSNorm | `src/qwen_full_precision.rs` | 104 |
| GELU approximate | `src/qwen_full_precision.rs` | 74 |

---

## Future Optimization Opportunities

| Optimization | Expected Gain | Effort |
|--------------|---------------|--------|
| Custom Metal kernel for modulation | 5-15% | High |
| Fused LayerNorm + scale + shift | 5-10% | High |
| Memory layout optimization | Unknown | Medium |

The modulation function (`src/qwen_full_precision.rs:406-418`) has 7+ operations that could be fused into a single Metal kernel:

```rust
fn modulate(x: &Array, shift: &Array, scale: &Array) -> Result<Array, Exception> {
    // These 7+ operations could be fused:
    let eps = Array::from_f32(1e-6);
    let mean = ops::mean_axis(x, -1, true)?;
    let x_centered = ops::subtract(x, &mean)?;
    let var = ops::mean_axis(&ops::multiply(&x_centered, &x_centered)?, -1, true)?;
    let normalized = ops::divide(&x_centered, &ops::sqrt(&ops::add(&var, &eps)?)?)?;
    let one = Array::from_f32(1.0);
    let scaled = ops::multiply(&normalized, &ops::add(scale, &one)?)?;
    ops::add(&scaled, shift)
}
```

---

## Benchmarking Notes

Performance varies with:
- **Thermal state**: ±30% variation on Apple Silicon
- **Background processes**: Close other apps for consistent results
- **Cold vs warm**: First run slower due to model loading and JIT compilation

Recommended benchmarking:
```bash
# Run twice, use second result
cargo run --release --example generate_fp32 -- --prompt "test" --steps 20
cargo run --release --example generate_fp32 -- --prompt "test" --steps 20  # Use this time
```
