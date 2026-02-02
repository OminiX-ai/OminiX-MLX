# Remaining Optimization Potential

## Current State

| Metric | Current |
|--------|---------|
| Total time (20 steps, 512x512) | **76-84s** |
| Per step | **3.8-4.2s** |
| Model precision | fp32 |
| Memory usage | ~15GB peak |

## Optimization Opportunities

### 1. Quantization (HIGH IMPACT)
**Expected improvement: 40-50% faster**

| Precision | Model Size | Expected Time | Speedup |
|-----------|------------|---------------|---------|
| fp32 (current) | ~13GB | 76-84s | 1x |
| fp16/bf16 | ~6.5GB | ~50-60s | 1.3-1.5x |
| int8 | ~3.3GB | ~40-50s | 1.6-2x |
| int4 | ~1.6GB | ~30-40s | 2-2.5x |

**Why it helps:**
- Reduced memory bandwidth (main bottleneck on Apple Silicon)
- Smaller model fits better in cache
- MLX has good quantization support

**Implementation effort:** Medium (2-3 days)
- Need to quantize weights or use pre-quantized model
- May need calibration for best quality

### 2. Fewer Diffusion Steps (MEDIUM IMPACT)
**Expected improvement: 25-50% faster**

| Steps | Time | Quality |
|-------|------|---------|
| 20 (current) | 76-84s | Best |
| 15 | ~60s | Very good |
| 10 | ~42s | Good |
| 8 | ~35s | Acceptable |

**Better schedulers:**
- DPM++ 2M: Good quality at 15-20 steps
- DDIM: Good quality at 20-50 steps
- Euler (current): Needs 20+ steps

**Implementation effort:** Low (1 day)
- Just change scheduler parameters

### 3. CFG Optimization (MEDIUM IMPACT)
**Expected improvement: 10-30% faster**

Currently running 2 transformer passes per step (conditional + unconditional). Options:

| Approach | Speedup | Quality Impact |
|----------|---------|----------------|
| Batch CFG | 10-20% | None (same result) |
| CFG rescale only | 0% | Already implemented |
| Lower CFG scale | 0% | May affect quality |
| Skip uncond at low sigma | 5-10% | Minimal |

**Batch CFG:** Run both conditional and unconditional in a single batched forward pass.

**Implementation effort:** Medium (1-2 days)

### 4. Attention Optimization (LOW-MEDIUM IMPACT)
**Expected improvement: 5-15% faster**

Already using `fast::scaled_dot_product_attention`. Additional options:

| Optimization | Speedup | Status |
|--------------|---------|--------|
| Flash Attention | 10-20% | Already using via SDPA |
| KV Cache | N/A | Not applicable (no autoregressive) |
| Sparse Attention | 5-10% | Would need architecture change |

**Implementation effort:** High (1+ week)

### 5. Lower Resolution (HIGH IMPACT, QUALITY TRADEOFF)
**Expected improvement: 4x faster at 256x256**

| Resolution | Latent Size | Time | Quality |
|------------|-------------|------|---------|
| 512x512 | 32x32 | 76-84s | Best |
| 384x384 | 24x24 | ~45s | Good |
| 256x256 | 16x16 | ~20s | Acceptable |

**Implementation effort:** None (just change parameters)

### 6. Model Distillation (HIGH IMPACT, HIGH EFFORT)
**Expected improvement: 2-4x faster**

Train a smaller student model:
- Fewer transformer blocks (30 instead of 60)
- Smaller hidden dimension
- Knowledge distillation from full model

**Implementation effort:** Very High (weeks-months)

## Comparison with Other Implementations

| Implementation | Time (20 steps) | Notes |
|----------------|-----------------|-------|
| **Our Rust/MLX (fp32)** | **76-84s** | Current |
| mflux (Python/MLX, 4-bit) | ~40-50s | Quantized |
| flux2.c (Pure Metal) | ~35-40s | No MLX overhead |
| Theoretical minimum | ~15-20s | Memory bandwidth limited |

## Realistic Target

With practical optimizations:

| Optimization | Cumulative Time |
|--------------|-----------------|
| Current (fp32) | 76-84s |
| + int8 quantization | ~45-50s |
| + Batch CFG | ~40-45s |
| + 15 steps | ~32-38s |
| + Better scheduler | ~28-35s |

**Realistic target: 30-40s** (2-2.5x improvement)

## Theoretical Limits

### Memory Bandwidth Analysis

```
M2 Max specs:
- Memory bandwidth: 400 GB/s
- Model size (fp32): 13GB
- Model size (int4): 1.6GB

Per forward pass:
- fp32: 13GB / 400GB/s = 32.5ms minimum
- int4: 1.6GB / 400GB/s = 4ms minimum

Per step (2 forwards for CFG):
- fp32: ~65ms minimum
- int4: ~8ms minimum

20 steps:
- fp32: ~1.3s minimum (just weight reads)
- int4: ~160ms minimum
```

**Reality check:** We're at 3.8s/step, theoretical minimum is ~65ms. That's 58x gap, which accounts for:
- Compute (attention, matmul, etc.)
- Activation memory reads/writes
- Kernel launch overhead
- Memory allocation
- Graph compilation

A 2-3x improvement to ~30-40s is realistic. Getting below 20s would require significant architectural changes.

## Recommended Priority

| Priority | Optimization | Effort | Impact |
|----------|--------------|--------|--------|
| 1 | int8/int4 quantization | Medium | High |
| 2 | Reduce to 15 steps | Low | Medium |
| 3 | Batch CFG | Medium | Medium |
| 4 | Better scheduler (DPM++) | Low | Low-Medium |
| 5 | Lower resolution option | None | High (with tradeoff) |

## Summary

| Scenario | Time | Improvement |
|----------|------|-------------|
| Current | 76-84s | Baseline |
| With quantization | 40-50s | ~1.7x |
| With all practical opts | 30-40s | ~2-2.5x |
| Theoretical limit | 15-20s | ~4-5x |

**Bottom line:** There's roughly **2-2.5x improvement** available through practical optimizations, primarily from quantization and step reduction. Getting beyond that requires architectural changes or moving away from MLX to pure Metal.
