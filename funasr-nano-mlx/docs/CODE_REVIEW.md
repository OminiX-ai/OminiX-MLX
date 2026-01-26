# Code Review: funasr-nano-mlx

**Date**: 2026-01-25
**Reviewer**: Claude
**Version**: 0.1.0

## Overview

funasr-nano-mlx is a Rust implementation of FunASR-Nano speech recognition using MLX for Apple Silicon GPU acceleration. The model architecture consists of:

- **SenseVoice Encoder**: 50 SAN-M layers + 20 temporal-parallel layers
- **Audio Adaptor**: Projects encoder output (512-dim) to LLM dimension (1024-dim)
- **Qwen3-0.6B LLM**: Autoregressive text generation with 28 transformer layers

### Current Performance

| Audio Length | Latency | RTF | Speed |
|--------------|---------|-----|-------|
| 5.6s | 442 ms | 0.079x | 12.7x real-time |
| 41.3s | 2,537 ms | 0.061x | 16.3x real-time |

---

## Issue Summary

| Category | Severity | Count | Priority |
|----------|----------|-------|----------|
| Performance | HIGH | 6 | P0 |
| Missing Features | HIGH | 5 | P1 |
| Memory Efficiency | MEDIUM | 3 | P1 |
| Error Handling | MEDIUM | 3 | P2 |
| Code Duplication | MEDIUM | 3 | P2 |
| Edge Cases/Bugs | MEDIUM | 7 | P2 |
| Documentation | HIGH | 4 | P3 |

---

## 1. Performance Bottlenecks

### 1.1 DFT O(n²) Complexity
- **File**: `src/audio.rs:154-163`
- **Severity**: HIGH
- **Impact**: 10-100x slower than FFT for long audio

```rust
// Current: Manual DFT - O(n²)
for k in 0..n_fft / 2 + 1 {
    for n in 0..n_fft {
        let angle = -2.0 * PI * k as f32 * n as f32 / n_fft as f32;
        real_sum += frame[n] * angle.cos();
        imag_sum += frame[n] * angle.sin();
    }
}
```

**Recommendation**: Use `rustfft` crate or MLX's FFT operations.

### 1.2 Manual Attention in Encoder
- **File**: `src/sensevoice_encoder.rs:232-240`
- **Severity**: HIGH
- **Impact**: 2-3x slower than optimized SDPA

```rust
// Current: Manual attention
let scores = q.matmul(k.transpose_axes(&[0, 1, 3, 2])?)?;
let scores = scores.multiply(mlx_rs::array!(self.scale))?;
let attn = mlx_rs::ops::softmax_axis(&scores, -1, None)?;
let out = attn.matmul(&v)?;
```

**Recommendation**: Use `mlx_rs::fast::scaled_dot_product_attention` (already used in adaptor.rs).

### 1.3 Excessive Array Cloning
- **Files**: `sensevoice_encoder.rs:388`, `adaptor.rs:178,183`, `qwen.rs:263,269`
- **Severity**: MEDIUM
- **Impact**: 118 unnecessary clones per inference (70 encoder + 20 tp + 28 LLM layers)

```rust
// Current: Clone in every layer
let residual = x.clone();
let h = self.norm.forward(&x)?;
```

**Recommendation**: Restructure to avoid clones where possible.

### 1.4 Inefficient LFR Processing
- **File**: `src/audio.rs:241-307`
- **Severity**: MEDIUM
- **Impact**: Memory duplication, CPU-bound

```rust
// Current: Copies to CPU memory
let mel_data: Vec<f32> = mel_contiguous.try_as_slice()?.to_vec();
```

**Recommendation**: Use MLX operations directly without CPU roundtrip.

### 1.5 Single-Token Generation Loop
- **File**: `src/model.rs:438-472`
- **Severity**: MEDIUM
- **Impact**: No parallelism in decoding

**Recommendation**: Implement speculative decoding or batch hypothesis generation.

### 1.6 Redundant Audio Resampling
- **File**: `src/model.rs:313-314`
- **Severity**: LOW
- **Impact**: Repeated work for same audio files

**Recommendation**: Add caching or preprocess method.

---

## 2. Missing Features

### 2.1 Streaming Transcription API
- **Priority**: P1
- **Impact**: Cannot process real-time audio or long files incrementally

```rust
// Missing API:
pub fn create_streaming_context(&mut self) -> StreamingContext;
pub fn transcribe_chunk(&mut self, ctx: &mut StreamingContext, chunk: &[f32]) -> Result<String>;
pub fn finalize_stream(&mut self, ctx: StreamingContext) -> Result<String>;
```

### 2.2 Batch Processing
- **Priority**: P1
- **Impact**: GPU underutilization for multiple files

```rust
// Missing API:
pub fn transcribe_batch(&mut self, paths: &[impl AsRef<Path>]) -> Result<Vec<String>>;
```

### 2.3 Advanced Sampling Strategies
- **Priority**: P2
- **Impact**: Limited output quality control

```rust
// Missing:
pub enum SamplingStrategy {
    Greedy,
    TopK { k: usize },
    TopP { p: f32 },
    Temperature { temp: f32 },
    BeamSearch { beam_width: usize },
}
```

### 2.4 Multiple Audio Format Support
- **Priority**: P3
- **Impact**: Only WAV supported, no MP3/FLAC/OGG

### 2.5 Customizable Prompt Templates
- **Priority**: P3
- **Impact**: Hardcoded Chinese prompt limits flexibility

---

## 3. Memory Efficiency

### 3.1 Large Embedding Matrix
- **File**: `src/qwen.rs:300-301`
- **Size**: 151936 × 1024 = ~620MB (f32)
- **Recommendation**: Consider INT8 quantization

### 3.2 Audio Processing Buffers
- **File**: `src/audio.rs:94-105`
- **Issue**: Multiple intermediate allocations in resampling
- **Recommendation**: Pre-allocate and reuse buffers

### 3.3 KV Cache Recreation
- **File**: `src/model.rs:341-343`
- **Issue**: Cache recreated on each call
- **Recommendation**: Reuse cache for streaming scenarios

---

## 4. Error Handling

### 4.1 Generic Error Messages
- **Files**: `audio.rs:50,89,102,262`
- **Issue**: Original error context lost

```rust
// Current:
Error::Audio(format!("Failed to read WAV: {}", e))

// Better:
#[error("Failed to read WAV file: {path}")]
WavReadError { path: PathBuf, #[source] source: hound::Error }
```

### 4.2 Missing Config Validation
- **File**: `src/model.rs:156-187`
- **Issue**: No dimension compatibility checks

```rust
// Missing:
fn validate_config(config: &FunASRNanoConfig) -> Result<()> {
    ensure!(config.adaptor.encoder_dim == config.encoder.output_size);
    ensure!(config.adaptor.llm_dim == config.llm.hidden_size);
    Ok(())
}
```

### 4.3 Silent Tokenizer Failure
- **File**: `src/model.rs:533-541`
- **Issue**: Returns token IDs as string if tokenizer missing

---

## 5. Code Duplication

### 5.1 Attention Implementations
- **Files**: `sensevoice_encoder.rs`, `adaptor.rs`, `qwen.rs`
- **Issue**: ~60+ lines duplicated with minor variations
- **Recommendation**: Create shared attention module

### 5.2 Linear Layer Patterns
- **Issue**: Repeated `nn::LinearBuilder::new().bias().build()?`
- **Recommendation**: Helper function

### 5.3 Reshape Patterns
- **Issue**: Same reshape for attention heads repeated 8+ times
- **Recommendation**: `reshape_for_attention()` helper

---

## 6. Potential Bugs & Edge Cases

### 6.1 Integer Overflow Risk
- **File**: `src/audio.rs:203`
- **Issue**: Mel filterbank calculation for large sample rates

### 6.2 Empty Audio Handling
- **File**: `src/audio.rs:140-141`
- **Issue**: No validation for empty or too-short audio

### 6.3 LFR Padding Edge Cases
- **File**: `src/audio.rs:273-288`
- **Issue**: Complex padding logic, potential off-by-one errors

### 6.4 Shape Mismatch Risk
- **File**: `src/model.rs:415-420`
- **Issue**: No validation of audio_features shape before indexing

### 6.5 Token ID Bounds
- **File**: `src/model.rs:494`
- **Issue**: Missing dimension validation before indexing

### 6.6 EOS Detection Fragility
- **File**: `src/model.rs:442-455`
- **Issue**: Only checks two specific token IDs

### 6.7 Safetensors Key Mapping
- **File**: `src/model.rs:283-307`
- **Issue**: String replacement chain could cause unintended matches

---

## 7. Documentation Gaps

### 7.1 Architecture Documentation
- **Missing**: Data flow through pipeline with tensor shapes
- **Missing**: FSMN memory block explanation
- **Missing**: SAN-M attention description

### 7.2 API Documentation
- **Missing**: Examples for common tasks
- **Missing**: Streaming usage (not implemented)
- **Missing**: Error handling guide

### 7.3 Configuration Documentation
- **Missing**: All config options explained
- **Missing**: Model conversion instructions

---

## 8. Code Quality

### 8.1 Magic Numbers
```rust
// Current (model.rs:352-380):
151644,  // <|im_start|>
151646,  // <|startofspeech|>

// Should be:
const QWEN_IM_START: i32 = 151644;
const QWEN_START_OF_SPEECH: i32 = 151646;
```

### 8.2 Test Coverage
- **Current**: Only 2 tests total
- **Missing**: Encoder, LLM, model loading, error cases

### 8.3 Unused Code
- **File**: `src/model.rs:479-517`
- **Issue**: `sample_with_penalty` function never used

---

## Appendix: File Structure

```
src/
├── lib.rs              # Public API exports
├── model.rs            # FunASRNano model (transcribe, generate_text)
├── audio.rs            # Audio loading, mel spectrogram, LFR
├── sensevoice_encoder.rs  # SenseVoice encoder (SAN-M + FSMN)
├── adaptor.rs          # Audio-to-LLM projection
├── qwen.rs             # Qwen3 LLM (attention, MLP, generation)
├── whisper_encoder.rs  # (Unused) Whisper encoder reference
└── error.rs            # Error types
```
