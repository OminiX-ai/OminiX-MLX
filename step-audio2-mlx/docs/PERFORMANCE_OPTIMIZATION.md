# Step-Audio 2 MLX Performance Optimization

## Overview

This document outlines the performance optimization strategy for the Step-Audio 2 MLX implementation. The primary goal is to leverage Apple Silicon's GPU and ANE (Apple Neural Engine) for maximum throughput.

## Current Bottlenecks

### 1. CPU-Bound Audio Processing (Critical)

The entire audio preprocessing pipeline runs on CPU:

| Operation | Location | Complexity | Issue |
|-----------|----------|------------|-------|
| STFT | `audio.rs:420-477` | O(n² × frames) | Naive DFT loop |
| Mel Filterbank | `audio.rs:389-397` | O(mels × freqs × frames) | CPU matmul |
| Normalization | `audio.rs:399-412` | O(mels × frames) | CPU loops |

**Impact**: For 15s audio at 16kHz, STFT alone requires ~12M multiply-adds per frame.

### 2. Missing MLX Compilation

No usage of `mlx_rs::compile()` for graph optimization:
- Encoder: 32 transformer layers
- LLM: 28 transformer layers
- Each layer incurs kernel launch overhead

### 3. Sequential Token Generation

Token generation processes one token at a time without:
- Strategic `eval()` placement for lazy evaluation
- Speculative decoding capabilities
- Continuous batching support

## Optimization Strategy

### Phase 1: GPU Audio Processing

Replace CPU audio processing with MLX operations:

```rust
// Before: CPU DFT
for k in 0..n_freqs {
    for n in 0..n_fft {
        real += windowed[n] * (angle).cos();
        imag -= windowed[n] * (angle).sin();
    }
}

// After: GPU FFT
let spectrum = mlx_rs::ops::fft::rfft(&windowed_frames, n_fft)?;
let power = spectrum.abs()?.square()?;
```

### Phase 2: MLX Graph Compilation

Wrap critical forward passes:

```rust
// Compile encoder + adaptor
let encode_audio = mlx_rs::compile(|mel| {
    let encoded = encoder.forward(mel)?;
    adaptor.forward(&encoded)
})?;

// Compile LLM step
let llm_step = mlx_rs::compile(|embeddings, cache| {
    llm.forward_embeddings(embeddings, cache)
})?;
```

### Phase 3: Generation Optimization

1. **Lazy Evaluation**: Only `eval()` at synchronization points
2. **Float16 Inference**: Reduce memory bandwidth
3. **Batch Support**: Process multiple sequences

## Expected Performance Gains

| Optimization | Component | Expected Speedup |
|--------------|-----------|------------------|
| GPU STFT | Audio | 50-100x |
| GPU Mel Filterbank | Audio | 10-50x |
| MLX compile() | Encoder/LLM | 20-40% |
| Strategic eval() | Generation | 10-20% |
| Float16 | All | 10-30% + memory |

## Implementation Status

### Completed

1. **GPU STFT** (`src/audio.rs`)
   - Replaced O(n²) CPU DFT with `mlx_rs::fft::rfft()`
   - Batch processing of all frames in parallel
   - Automatic fallback to CPU if GPU fails

2. **GPU Mel Filterbank** (`src/audio.rs`)
   - Replaced CPU triple-nested loop with `mlx_rs::ops::matmul()`
   - GPU log10 and normalization via element-wise ops

3. **Strategic eval() Documentation** (`src/model.rs`)
   - Documented synchronization points
   - MLX lazy evaluation handles batching automatically
   - `item()` calls are the synchronization points

### Pending

4. **MLX compile() Wrappers**
   - Requires refactoring Module pattern to avoid captures
   - Deferred to future iteration

## Implementation Files

- `src/audio.rs` - GPU-accelerated STFT and mel spectrogram
- `src/model.rs` - Generation loop with documented sync points
- `src/llm.rs` - LLM forward pass (uses SDPA)
- `src/encoder.rs` - Encoder layers (uses MLX ops)

## ANE Utilization

MLX automatically routes operations to ANE when beneficial. To maximize ANE usage:

1. **Use Float16**: ANE excels at fp16 operations
2. **Batch Operations**: ANE throughput improves with batching
3. **Minimize Transfers**: Keep data on device

## Benchmarking

After optimization, measure:
- Audio preprocessing time (target: <100ms for 15s audio)
- Encoder forward pass (target: <200ms)
- Token generation speed (target: >50 tokens/sec)
- Memory usage (target: <8GB for full model)

## References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Apple Silicon GPU Architecture](https://developer.apple.com/metal/)
- [Whisper Audio Processing](https://github.com/openai/whisper)
