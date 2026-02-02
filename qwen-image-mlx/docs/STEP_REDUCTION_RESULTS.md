# Step Reduction Results

*Tested: 2026-01-29*

## Key Finding

**The current Euler scheduler with FlowMatch time-shifting works well at lower step counts.** No scheduler changes needed - just reduce steps from 20 to 8-10 for significant speedup with minimal quality loss.

## Timing Breakdown

### One-Time Costs (not affected by step count)
| Phase | Time |
|-------|------|
| Model loading | ~15-20s |
| Text encoding | ~2-3s |
| VAE decoding | ~5-6s |
| **Total overhead** | **~22-29s** |

### Diffusion Time (scales linearly with steps)
| Steps | Diffusion Time | Per Step | Quality |
|-------|----------------|----------|---------|
| 20 | ~80s | ~4.0s | Excellent |
| 15 | ~60s | ~4.0s | Excellent |
| **10** | **~40s** | **~4.0s** | **Excellent** |
| **8** | **~32s** | **~4.0s** | **Very Good** |
| 6 | ~24s | ~4.0s | Good |
| 4 | ~16s | ~4.0s | Poor (blurry) |

*Note: Per-step time is consistent at ~4s. Total time = steps × 4s. Thermal throttling can add variance.*

### Total Generation Time (including overhead)
| Steps | Diffusion | Overhead | Total | Speedup vs 20 |
|-------|-----------|----------|-------|---------------|
| 20 | 80s | 25s | **105s** | 1.0x |
| 15 | 60s | 25s | **85s** | 1.2x |
| **10** | 40s | 25s | **65s** | **1.6x** |
| **8** | 32s | 25s | **57s** | **1.8x** |
| 6 | 24s | 25s | **49s** | 2.1x |
| 4 | 16s | 25s | **41s** | 2.6x |

## Quality Comparison

### 20 Steps (Baseline)
- Excellent detail
- Sharp edges
- Full texture in fur

### 10 Steps (Recommended)
- Virtually identical to 20 steps
- No visible quality loss
- **Best quality/speed tradeoff**

### 8 Steps (Fast Mode)
- Very good quality
- Slightly less fine detail
- Still production-ready

### 6 Steps
- Good quality
- Some loss of fine detail
- Acceptable for previews

### 4 Steps
- Noticeable blur
- Missing fine details
- Not recommended

## Recommendations

### For Best Quality
```bash
cargo run --release --example generate_fp32 -- \
  --prompt "your prompt" --steps 20 --output output.png
```
Time: ~105s total (~80s diffusion)

### For Quality + Speed Balance (Recommended)
```bash
cargo run --release --example generate_fp32 -- \
  --prompt "your prompt" --steps 10 --output output.png
```
Time: ~65s total (~40s diffusion) - **1.6x faster**

### For Fast Iteration
```bash
cargo run --release --example generate_fp32 -- \
  --prompt "your prompt" --steps 8 --output output.png
```
Time: ~57s total (~32s diffusion) - **1.8x faster**

### For Quick Preview
```bash
cargo run --release --example generate_fp32 -- \
  --prompt "your prompt" --steps 6 --output output.png
```
Time: ~49s total (~24s diffusion) - **2.1x faster**

## Why This Works

The FlowMatch Euler scheduler with exponential time shifting already provides good sigma spacing:

1. **More time at high noise** (σ ≈ 1.0) where large changes happen
2. **Less time at low noise** (σ ≈ 0.02) where changes are small
3. **Smooth interpolation** via the time shift formula

The scheduler formula:
```
σ_shifted = exp(μ) / (exp(μ) + (1/t - 1)^σ)
```

Where μ is computed based on image resolution, providing resolution-adaptive scheduling.

## No Need for DPM++ or Other Schedulers

Testing showed that:
- The current Euler + FlowMatch scheduler works well at 8-10 steps
- Implementing DPM++ would add complexity without significant benefit
- The simple approach (just reduce steps) gives the best results

## Summary

| Mode | Steps | Diffusion Time | Total Time | Quality |
|------|-------|----------------|------------|---------|
| Quality | 20 | 80s | 105s | ★★★★★ |
| **Balanced** | **10** | **40s** | **65s** | **★★★★★** |
| Fast | 8 | 32s | 57s | ★★★★☆ |
| Preview | 6 | 24s | 49s | ★★★☆☆ |

**Recommended default: 10 steps** - Same quality as 20 steps, 1.6x faster.
