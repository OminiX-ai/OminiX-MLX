# FunASR Performance Enhancement Plan

**Date:** 2026-01-30
**Scope:** funasr-mlx (Paraformer) and funasr-nano-mlx (SenseVoice + Qwen3)
**Status:** Active

---

## Executive Summary

This document outlines validated performance optimizations for the FunASR MLX implementations. All recommendations are based on empirical testing and profiling, not theoretical assumptions.

**Key Finding:** Not all GPU optimizations are beneficial. Small operations (like LFR) can be faster on CPU due to MLX kernel overhead.

---

## Current Performance Baselines

| Model | Architecture | Speed | Memory |
|-------|--------------|-------|--------|
| funasr-mlx (Paraformer) | Non-autoregressive | 18-56x real-time | ~600MB |
| funasr-nano-mlx | Encoder + LLM | 3x real-time | ~2.5GB |

---

## Phase 1: Quick Wins (Low Effort, Validated Impact)

### 1.1 Implement Repetition Penalty (funasr-nano-mlx)

**Status:** Not Started
**Effort:** Low (1-2 hours)
**Impact:** Quality improvement for long-form audio

The `repetition_penalty` parameter exists in `SamplingConfig` but is unused.

**Location:** `src/model.rs:673-698`

**Implementation:**
```rust
fn sample_with_config(
    logits: &Array,
    config: &SamplingConfig,
    prev_tokens: &[i32],  // Currently unused
) -> Result<i32> {
    let mut logits = logits.clone();

    // Apply repetition penalty
    if config.repetition_penalty != 1.0 && !prev_tokens.is_empty() {
        for &token in prev_tokens {
            let idx = token as usize;
            let current = logits.index((idx,)).item::<f32>();
            let penalized = if current > 0.0 {
                current / config.repetition_penalty
            } else {
                current * config.repetition_penalty
            };
            // Update logits at index
        }
    }
    // ... rest of sampling
}
```

**Validation:** Test with long audio (>30s) and check for repetitive output.

---

### 1.2 Load Token IDs from Tokenizer (funasr-nano-mlx)

**Status:** Not Started
**Effort:** Low (1-2 hours)
**Impact:** Maintainability, future tokenizer compatibility

**Location:** `src/model.rs:547-568`

**Current (hardcoded):**
```rust
let prefix_tokens = [
    151644,  // <|im_start|>
    8948,    // system
    198,     // \n
    // ... 20+ more hardcoded IDs
];
```

**Proposed:**
```rust
fn get_special_token_id(tokenizer: &Tokenizer, token: &str, fallback: u32) -> u32 {
    tokenizer.token_to_id(token).unwrap_or(fallback)
}

let im_start = get_special_token_id(&tokenizer, "<|im_start|>", 151644);
let system = get_special_token_id(&tokenizer, "system", 8948);
// ...
```

---

### 1.3 Remove Deprecated whisper_encoder Module

**Status:** Not Started
**Effort:** Minimal
**Impact:** Cleaner codebase, faster compilation

**Action:** Remove `src/whisper_encoder.rs` and update `src/lib.rs`.

---

## Phase 2: Medium Effort Optimizations

### 2.1 INT8 Quantization for Encoder

**Status:** Not Started
**Effort:** Medium (1-2 days)
**Impact:** 50% memory reduction, 20-30% speedup on memory-bound systems

**Approach:**
1. Use MLX's built-in quantization for encoder weights
2. Keep LLM in FP16/BF16 for accuracy
3. Validate WER (Word Error Rate) doesn't degrade significantly

**Components to Quantize:**
- SenseVoice Encoder (221M params → ~110MB INT8)
- Audio Adaptor (12.6M params → ~6MB INT8)

**Skip Quantization:**
- Qwen3 LLM (quality-critical for text generation)
- Embedding layers (sparse access patterns)

**MLX Quantization API:**
```rust
// Pseudo-code for MLX quantization
let quantized_weights = mlx_rs::quantize::quantize_weights(
    &encoder_weights,
    bits: 8,
    group_size: 64,
)?;
```

**Validation:**
- Compare WER on test set before/after quantization
- Acceptable degradation: <2% relative WER increase

---

### 2.2 KV Cache Optimization (funasr-nano-mlx)

**Status:** Partially Implemented
**Effort:** Medium
**Impact:** 10-20% generation speedup

**Current Issue:** KV cache may be reallocated during generation.

**Optimization:**
1. Pre-allocate KV cache for maximum expected sequence length
2. Use in-place updates instead of concatenation
3. Consider sliding window attention for very long sequences

---

## Phase 3: High Effort Optimizations

### 3.1 Speculative Decoding (funasr-nano-mlx)

**Status:** Not Started
**Effort:** High (1-2 weeks)
**Impact:** Potential 2x generation speedup

**Concept:** Use a smaller draft model to propose tokens, then verify with the main model in parallel.

**Requirements:**
- Draft model: Qwen3-0.1B or distilled version
- Verification logic for token acceptance
- Fallback to standard decoding when speculation fails

**Architecture:**
```
Draft Model (fast) → Propose N tokens → Main Model (verify) → Accept/Reject
```

**Estimated Speedup:** 1.5-2x for generation phase

**Challenges:**
- Need to train or obtain draft model
- Complexity in verification logic
- May not help for short outputs

---

### 3.2 Streaming Encoder (funasr-nano-mlx)

**Status:** Partially Implemented (streaming context exists)
**Effort:** High
**Impact:** Lower first-token latency

**Current:** Full audio must be encoded before generation starts.

**Proposed:** Chunk-based encoding with incremental processing.

**Challenges:**
- SenseVoice encoder uses bidirectional attention
- Would require architectural changes or chunked attention

---

## Disproven Optimizations (Do NOT Implement)

### ❌ LFR GPU Optimization

**Original Claim:** Move LFR to pure MLX ops for 5-10% speedup.

**Test Results:**
| Audio | MLX LFR | CPU LFR | Winner |
|-------|---------|---------|--------|
| 1s | 2.06ms | 1.34ms | CPU (54% faster) |
| 10s | 10.55ms | 10.50ms | Equal |

**Conclusion:** MLX kernel overhead exceeds GPU→CPU→GPU transfer cost for this small operation. **Keep CPU implementation.**

---

### ❌ CIF GPU Optimization (funasr-mlx)

**Original Claim:** Use scatter/gather for GPU-based CIF.

**Reality:** CIF has data-dependent control flow (fire when threshold exceeded). This doesn't vectorize well on GPU.

**Conclusion:** CPU implementation is appropriate for this algorithm.

---

## Validation Framework

### Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Latency (1s audio) | <500ms | `Instant::now()` |
| Throughput | >10x real-time | audio_duration / processing_time |
| Memory | <3GB peak | Activity Monitor / Instruments |
| WER | <5% degradation | Test on standard dataset |

### Test Commands

```bash
# Benchmark latency
cargo run --release --example benchmark -- audio.wav 20

# Memory profiling (macOS)
instruments -t "Allocations" target/release/examples/transcribe audio.wav

# Correctness validation
cargo test --release
```

---

## Implementation Timeline

| Phase | Items | Est. Effort | Priority |
|-------|-------|-------------|----------|
| Phase 1 | Repetition penalty, token IDs, cleanup | 1-2 days | High |
| Phase 2 | INT8 quantization, KV cache | 3-5 days | Medium |
| Phase 3 | Speculative decoding, streaming | 2-3 weeks | Low |

---

## Appendix: Lessons Learned

### 1. Benchmark Before Optimizing

The LFR GPU optimization seemed obvious but was wrong. Always measure:
- Actual transfer overhead
- Kernel launch latency
- CPU baseline performance

### 2. Small Operations Favor CPU

For operations taking <1ms, CPU is often faster because:
- No kernel launch overhead
- No memory transfer
- CPU SIMD is highly optimized

### 3. MLX Kernel Overhead

MLX operations have fixed overhead (~0.1-0.3ms per operation). Chaining many small operations can be slower than a single CPU loop.

---

*Document created: 2026-01-30*
*Last updated: 2026-01-30*
