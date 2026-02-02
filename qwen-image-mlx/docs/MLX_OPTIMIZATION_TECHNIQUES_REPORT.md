# MLX Optimization Techniques Report

**Analysis of OminiX-MLX implementations to identify optimization strategies for Qwen-Image**

---

## Executive Summary

Reviewed 4 MLX implementations:
- **flux-klein-mlx** - Image generation (most similar to Qwen-Image)
- **gpt-sovits-mlx** - Text-to-speech
- **mixtral-mlx** - MoE language model
- **zimage-mlx** - Image generation

Key findings: **7 high-impact optimizations** applicable to Qwen-Image.

---

## 1. RoPE Pre-computation (HIGH IMPACT)

### Found In: flux-klein-mlx, zimage-mlx

**Current Qwen-Image**: RoPE computed inside transformer forward pass.

**Optimization Strategy**:
```rust
// BEFORE: RoPE computed every forward pass (60 blocks × 20 steps = 1200 times)
fn forward(&mut self, ...) {
    let (cos, sin) = compute_rope(...);  // Computed repeatedly
    ...
}

// AFTER: RoPE computed ONCE before denoising loop
let (rope_cos, rope_sin) = compute_rope(&img_ids, &txt_ids)?;  // ONCE

for step in 0..num_steps {
    transformer.forward_with_rope(..., &rope_cos, &rope_sin)?;  // Reuse
}
```

**Impact**: flux-klein achieves **2.6x speedup** with this optimization alone.

**Implementation for Qwen-Image**:
- Move RoPE computation out of `QwenFullTransformer::forward()`
- Add `forward_with_rope()` method
- Pre-compute in `generate_fp32.rs` before diffusion loop

---

## 2. Custom Metal Kernels (HIGH IMPACT)

### Found In: mixtral-mlx

**Technique**: Fused SwiGLU kernel combining `silu(gate) * x` in single operation.

```rust
// Custom Metal kernel: 10-12x faster than separate operations
const SWIGLU_KERNEL_SOURCE: &str = r#"
    T gate_val = gate[elem];
    T x_val = x[elem];
    T silu_gate = gate_val / (T(1) + metal::exp(-gate_val));
    out[elem] = silu_gate * x_val;
"#;
```

**Impact**: 10-12x speedup for SwiGLU operations.

**Implementation for Qwen-Image**:
- Create fused kernel for modulation: `(1 + scale) * LayerNorm(x) + shift`
- Combine 5+ operations into single kernel
- File: `/mlx-rs-core/src/metal_kernels.rs`

---

## 3. Timestep Embedding Caching (MEDIUM IMPACT)

### Found In: zimage-mlx

**Technique**: Pre-compute sinusoidal frequencies at initialization.

```rust
pub struct TimestepEmbedder {
    cached_freqs: Array,  // Pre-computed at init
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedder {
    pub fn new(dim: i32) -> Self {
        // Compute frequencies ONCE
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| (-10000.0f32.ln() * i / half_dim).exp())
            .collect();
        let cached_freqs = Array::from_slice(&freqs, &[1, half_dim]);
        ...
    }

    pub fn forward(&self, t: &Array) -> Array {
        // Reuse cached frequencies
        let sinusoid = ops::multiply(&t, &self.cached_freqs)?;
        ...
    }
}
```

**Impact**: ~2-5% speedup, eliminates redundant computation.

**Implementation for Qwen-Image**:
- Add `cached_freqs: Array` field to `TimestepEmbedder`
- Pre-compute in `new()`, reuse in `forward()`

---

## 4. KV Cache with Step-Based Pre-allocation (FOR LLM TEXT ENCODER)

### Found In: gpt-sovits-mlx, mixtral-mlx

**Technique**: Pre-allocate buffers in chunks, use in-place updates.

```rust
pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
    step: i32,  // Default: 256 tokens
}

// In-place update instead of concatenation
k.index_mut((Ellipsis, prev..self.offset, ..), &keys);
```

**Impact**: ~50-70% overhead reduction for autoregressive generation.

**Applicability to Qwen-Image**:
- Not directly applicable (diffusion doesn't use KV cache)
- May help with Qwen3 text encoder if doing long prompts

---

## 5. Minimal eval() Strategy (HIGH IMPACT)

### Found In: flux-klein-mlx

**Current Qwen-Image**: No eval() inside transformer, only at end.

**Optimal Strategy** (confirmed by flux-klein):
```rust
// CORRECT: Only 4 eval() calls in entire generation
txt_embed.eval()?;        // After text encoding
latent.eval()?;           // After EACH denoising step (20 times)
image.eval()?;            // After VAE decode

// WRONG: eval() inside transformer blocks (tested, makes it slower)
for block in &mut self.blocks {
    ...
    mlx_rs::transforms::eval([&img, &txt])?;  // DON'T DO THIS
}
```

**Finding**: Qwen-Image's current approach is correct. Adding eval() inside blocks slows it down.

---

## 6. Quantized Linear with gather_qmm (MEDIUM IMPACT)

### Found In: mixtral-mlx

**Technique**: Single fused kernel for quantized matmul with expert routing.

```rust
ops::gather_qmm(
    x, &weight, &scales, &biases,
    None, Some(indices), true,
    group_size, bits, None, sorted_indices
)
```

**Impact**: Reduces memory bandwidth, avoids separate dequantization.

**Implementation for Qwen-Image**:
- Already using `QuantizedLinear` for 4-bit version
- Ensure using MLX's optimized quantized matmul path

---

## 7. Async Evaluation and Prefetching (MEDIUM IMPACT)

### Found In: mixtral-mlx, gpt-sovits-mlx

**Technique**: Compute next iteration while processing current.

```rust
// Queue next computation asynchronously
let _ = async_eval([&next_y]);

// Process current result
let _ = eval([&y]);

// Prefetch is ready for next iteration
self.prefetched = Some(next_y);
```

**Impact**: Overlaps computation and data transfer.

**Implementation for Qwen-Image**:
- Could prefetch next step's noise prediction while current step finalizes
- Limited benefit for diffusion (each step depends on previous)

---

## 8. Periodic Memory Cache Clearing (MEDIUM IMPACT)

### Found In: mixtral-mlx, gpt-sovits-mlx

**Technique**: Clear MLX cache periodically to prevent OOM.

```rust
if self.generated % 256 == 0 {
    unsafe { mlx_sys::mlx_clear_cache(); }
}
```

**Impact**: Prevents memory fragmentation during long inference.

**Implementation for Qwen-Image**:
- Add after every N diffusion steps (e.g., every 5 steps)
- Useful for high-resolution images or many steps

---

## Optimization Priority Matrix

| Optimization | Impact | Effort | Priority |
|-------------|--------|--------|----------|
| RoPE Pre-computation | HIGH (2.6x) | Low | **1** |
| Custom Metal Kernels | HIGH (10x for specific ops) | High | 2 |
| Timestep Freq Caching | MEDIUM (2-5%) | Low | **3** |
| Minimal eval() Strategy | HIGH | Already done | ✓ |
| fast::layer_norm | NEGATIVE | Tested | ✗ |
| eval() inside blocks | NEGATIVE | Tested | ✗ |
| Async prefetch | LOW | Medium | 5 |
| Cache clearing | LOW | Low | 6 |

---

## Recommended Implementation Plan

### Phase 1: RoPE Pre-computation (Highest Priority)
1. Add `forward_with_rope()` to `QwenFullTransformer`
2. Move RoPE computation to example before diffusion loop
3. Expected gain: **20-40% faster**

### Phase 2: Timestep Frequency Caching
1. Add `cached_freqs` field to `TimestepEmbedder`
2. Pre-compute in `new()`, use in `forward()`
3. Expected gain: **2-5% faster**

### Phase 3: Custom Metal Kernel for Modulation (Advanced)
1. Create fused kernel in `mlx-rs-core/src/metal_kernels.rs`
2. Combine LayerNorm + scale + shift into single kernel
3. Expected gain: **5-15% faster**

---

## Performance Comparison: flux-klein-mlx vs qwen-image-mlx

| Metric | flux-klein | qwen-image (current) | Gap |
|--------|------------|---------------------|-----|
| RoPE | Pre-computed | Computed each pass | ❌ |
| Timestep cache | Yes | No | ❌ |
| eval() strategy | Minimal (4 calls) | Minimal | ✓ |
| fast:: functions | SDPA | SDPA + RMS Norm | ✓ |
| Total time | 7.73s (4 steps) | ~145s (20 steps) | - |
| Per step | 1.09s | 7.2s | 6.6x |

Note: flux-klein uses 4 steps, qwen-image uses 20 steps. Per-step comparison shows significant room for improvement.

---

## Conclusion

The **RoPE pre-computation** is the single most impactful optimization missing from Qwen-Image. flux-klein-mlx demonstrates a **2.6x speedup** from this alone.

The current Qwen-Image implementation already uses:
- ✓ fast::scaled_dot_product_attention
- ✓ fast::rms_norm
- ✓ Minimal eval() strategy

To achieve performance parity with flux-klein on a per-step basis:
1. Implement RoPE pre-computation
2. Add timestep frequency caching
3. Consider custom Metal kernels for modulation

---

## References

- `/Users/yuechen/home/OminiX-MLX/flux-klein-mlx/` - RoPE caching, eval() strategy
- `/Users/yuechen/home/OminiX-MLX/mixtral-mlx/` - Custom kernels, KV cache
- `/Users/yuechen/home/OminiX-MLX/gpt-sovits-mlx/` - Async eval, cache clearing
- `/Users/yuechen/home/OminiX-MLX/zimage-mlx/` - Timestep caching, RoPE caching
- `/Users/yuechen/home/OminiX-MLX/OPTIMIZATION_DETAILS.md` - MLX version issues
