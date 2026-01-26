# LLM Projects Review & Fun-ASR Integration Analysis

## Executive Summary

This document reviews three MLX-based LLM implementations and analyzes their potential for Fun-ASR-Nano-2512 integration.

**Key Finding:** Fun-ASR-Nano uses **Whisper encoder + Qwen LLM**. The existing `qwen3-mlx` implementation can be leveraged as the LLM backbone, significantly reducing development effort.

---

## 1. Project Overview

| Project | Model | Parameters | Architecture | Lines of Code |
|---------|-------|------------|--------------|---------------|
| **glm4-mlx** | GLM-4 | 9B | Decoder-only + Partial RoPE | ~855 |
| **mixtral-mlx** | Mixtral | 8x7B | MoE (8 experts, top-2) | ~743 |
| **qwen3-mlx** | Qwen3 | 0.6B-32B | Decoder-only + GQA | ~845 |

### Shared Infrastructure (mlx-rs-core)

All three projects share:
- `cache.rs` - KVCache implementations (183 lines)
- `utils.rs` - RoPE, attention masks (199 lines)
- `metal_kernels.rs` - Fused SwiGLU kernel (126 lines)
- `error.rs` - Error handling (47 lines)

---

## 2. Architecture Comparison

### 2.1 Attention Mechanisms

| Feature | GLM-4 | Mixtral | Qwen3 |
|---------|-------|---------|-------|
| Attention Type | Standard MHA | Standard MHA | **GQA** |
| KV Heads | Same as Q | Same as Q | **Reduced** (4:1) |
| RoPE | **Partial (50%)** | Full | Full + Scaling |
| Q/K Normalization | No | No | **Yes** |
| Fused QKV | No | No | No |

**Winner for ASR:** Qwen3 - GQA reduces memory for long audio sequences

### 2.2 MLP/FFN

| Feature | GLM-4 | Mixtral | Qwen3 |
|---------|-------|---------|-------|
| Type | **Fused gate_up** | SwiGLU + MoE | SwiGLU |
| Intermediate | 2x hidden | 14336 | Model-dependent |
| Activation | SiLU | SiLU | SiLU |
| Experts | 1 | **8 (top-2)** | 1 |

**Winner for ASR:** Qwen3 - Simpler architecture, easier to debug

### 2.3 Normalization

| Feature | GLM-4 | Mixtral | Qwen3 |
|---------|-------|---------|-------|
| Norm Type | RMSNorm | RMSNorm | RMSNorm |
| Norm Count/Layer | **4** | 2 | 2 |
| Position | Pre + Post | Pre-LN | Pre-LN |

### 2.4 KV Cache

All three use the same `KVCache` from mlx-rs-core:
- Step-based pre-allocation (256 tokens)
- In-place slice updates
- Offset tracking for RoPE

---

## 3. Code Quality Assessment

| Metric | GLM-4 | Mixtral | Qwen3 |
|--------|-------|---------|-------|
| Code Organization | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Documentation | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| Error Handling | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Quantization | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Examples | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Tests | ⭐ | ⭐ | ⭐ |

**Overall Winner:** qwen3-mlx - Best organized, most complete

---

## 4. Fun-ASR-Nano Architecture (Confirmed)

Based on the arXiv paper (2509.12508):

```
Audio (16kHz)
    ↓
[Whisper Encoder] ← FROZEN
    ↓
[Linear Adaptor] ← Trainable
    ↓
[Qwen LLM] ← Trainable or Frozen
    ↓
Autoregressive Text Generation
```

**Components:**
1. **Audio Encoder:** Whisper-based mel spectrogram + transformer encoder
2. **Adaptor:** Linear projection (encoder_dim → llm_dim)
3. **LLM:** Qwen architecture (causal decoder)

---

## 5. Reusability Analysis for Fun-ASR

### 5.1 What Can Be Reused

| Component | Source | Reusable? | Notes |
|-----------|--------|-----------|-------|
| **Qwen LLM** | qwen3-mlx | ✅ **YES** | Core backbone |
| **KV Cache** | mlx-rs-core | ✅ **YES** | Same mechanism |
| **RoPE** | mlx-rs-core | ✅ **YES** | Same implementation |
| **Sampling** | qwen3-mlx | ✅ **YES** | Temperature sampling |
| **Tokenizer** | HF tokenizers | ✅ **YES** | Same library |
| **Generation** | qwen3-mlx | ⚠️ **PARTIAL** | Needs audio injection |
| **Weight Loading** | qwen3-mlx | ⚠️ **PARTIAL** | Need adaptor weights |

### 5.2 What's Missing

| Component | Effort | Lines Est. |
|-----------|--------|------------|
| **Whisper Encoder** | HIGH | 1000-1500 |
| **Audio Adaptor** | LOW | 50-100 |
| **Multimodal Injection** | MEDIUM | 150-250 |
| **Chat Template Mod** | LOW | 30-50 |

### 5.3 Comparison: Build from Scratch vs Reuse

**Option A: New Implementation (from scratch)**
- Total effort: 2500-4000 lines
- Timeline: 3-5 weeks
- Risk: High (unknown architecture details)

**Option B: Reuse qwen3-mlx (recommended)**
- New code: 1200-1900 lines
- Timeline: 1.5-2.5 weeks
- Risk: Lower (proven LLM backbone)

**Savings: ~50% code, ~50% time**

---

## 6. Recommended Architecture

### 6.1 Project Structure

```
fun-asr-mlx/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public API
│   ├── audio/
│   │   ├── mod.rs
│   │   ├── mel.rs                # Mel spectrogram (from funasr-mlx)
│   │   └── whisper_encoder.rs    # Whisper encoder (NEW)
│   ├── adaptor.rs                # Linear adaptor (NEW, ~100 lines)
│   ├── model.rs                  # Multimodal model (EXTEND qwen3)
│   └── generation.rs             # Audio-aware generation (EXTEND)
└── examples/
    └── transcribe.rs
```

### 6.2 Component Reuse Map

```
┌─────────────────────────────────────────────────────────────────┐
│                        fun-asr-mlx                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Audio Processing │  │  Whisper Encoder │  │  Audio Adaptor  │ │
│  │   (funasr-mlx)  │  │      (NEW)       │  │     (NEW)       │ │
│  └────────┬────────┘  └────────┬─────────┘  └────────┬────────┘ │
│           │                    │                      │          │
│           └──────────┬─────────┴──────────────────────┘          │
│                      ▼                                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Qwen LLM (qwen3-mlx)                    │  │
│  │  • Attention (GQA)  • MLP (SwiGLU)  • Generation          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                      │                                           │
│  ┌───────────────────┴───────────────────────────────────────┐  │
│  │              Shared Infrastructure (mlx-rs-core)           │  │
│  │    • KVCache  • RoPE  • SwiGLU kernel  • Error handling   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Key Implementation Details

#### Audio Injection Point

```rust
// In model.rs - extend Qwen3 forward
pub fn forward_with_audio(
    &mut self,
    text_tokens: &Array,      // [B, L_text]
    audio_features: &Array,   // [B, L_audio, hidden_dim]
    speech_markers: (usize, usize),  // (start, end) positions
    cache: &mut Vec<Option<KVCache>>,
) -> Result<Array> {
    // 1. Embed text tokens
    let text_embeds = self.embed_tokens.forward(text_tokens)?;

    // 2. Inject audio features at marker positions
    let combined = inject_audio(text_embeds, audio_features, speech_markers)?;

    // 3. Forward through transformer layers
    self.forward_embeddings(combined, cache)
}
```

#### Whisper Encoder (Simplified)

```rust
pub struct WhisperEncoder {
    conv1: nn::Conv1d,      // Initial convolution
    conv2: nn::Conv1d,      // Downsampling
    layers: Vec<WhisperEncoderLayer>,
    ln_post: nn::LayerNorm,
}

impl WhisperEncoder {
    pub fn forward(&mut self, mel: &Array) -> Result<Array> {
        // mel: [B, 80, T] -> [B, T/4, encoder_dim]
        let x = self.conv1.forward(mel)?;
        let x = nn::gelu(&x)?;
        let x = self.conv2.forward(&x)?;
        let x = nn::gelu(&x)?;
        let x = x.transpose_axes(&[0, 2, 1])?;  // [B, T, C]

        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }

        self.ln_post.forward(&x)
    }
}
```

---

## 7. Effort Estimation

### 7.1 Development Tasks

| Task | Hours | Dependencies |
|------|-------|--------------|
| Whisper encoder implementation | 20-30 | None |
| Audio adaptor | 4-6 | Whisper encoder |
| Multimodal injection | 8-12 | qwen3-mlx |
| Generation extension | 6-10 | Multimodal injection |
| Weight loading for Fun-ASR | 8-12 | All components |
| Testing & debugging | 15-20 | All components |
| Examples & documentation | 6-8 | All components |
| **Total** | **67-98 hours** | ~2 weeks |

### 7.2 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Unknown Whisper variant | Medium | High | Start with standard Whisper |
| Weight format mismatch | Medium | Medium | Inspect Fun-ASR weights |
| Performance issues | Low | Medium | Profile and optimize |
| Qwen version mismatch | Low | Low | Check Fun-ASR config |

---

## 8. Alternative: Use Existing Whisper MLX

If reverse-engineering Whisper encoder is too complex:

**Option: Use mlx-whisper (if available)**

```rust
// External dependency approach
[dependencies]
mlx-whisper = { path = "../mlx-whisper" }  // If exists

// Or FFI to Python mlx-whisper
```

**Pros:**
- Proven Whisper implementation
- Faster development

**Cons:**
- Additional dependency
- May not match Fun-ASR's exact Whisper variant

---

## 9. Quick Start Implementation Plan

### Week 1: Foundation
1. [ ] Set up fun-asr-mlx crate structure
2. [ ] Import qwen3-mlx as dependency or copy core modules
3. [ ] Implement basic audio adaptor
4. [ ] Test with dummy audio features

### Week 2: Whisper Integration
1. [ ] Implement Whisper encoder (or integrate existing)
2. [ ] Connect encoder → adaptor → Qwen
3. [ ] Load Fun-ASR weights (if available)
4. [ ] End-to-end inference test

### Week 3: Polish
1. [ ] Optimize performance
2. [ ] Add streaming support
3. [ ] Write examples and documentation
4. [ ] Benchmark against Python Fun-ASR

---

## 10. Conclusion

### Recommended Path

1. **Use qwen3-mlx as LLM backbone** - It's the most complete and matches Fun-ASR's architecture
2. **Reuse funasr-mlx audio utilities** - Mel spectrogram, resampling
3. **Implement Whisper encoder** - ~1500 lines, can be simplified
4. **Create thin adaptor layer** - ~100 lines

### Expected Outcome

| Metric | Estimate |
|--------|----------|
| Development time | 2 weeks |
| New code | ~1500 lines |
| Reused code | ~2000 lines |
| Performance | 20-40 tokens/sec (decode) |
| Memory | 2-4 GB (4-bit quantized) |

### Code Reuse Summary

```
fun-asr-mlx total: ~3500 lines
├── From qwen3-mlx: ~800 lines (23%)
├── From funasr-mlx: ~300 lines (9%)
├── From mlx-rs-core: ~500 lines (14%)
└── New implementation: ~1900 lines (54%)
```

**Bottom Line:** By leveraging existing implementations, especially qwen3-mlx, you can build Fun-ASR-Nano support with roughly half the effort of starting from scratch.

---

## Appendix: File Locations

### qwen3-mlx (Primary Reuse Target)
- `/Users/yuechen/home/OminiX-MLX/qwen3-mlx/src/model.rs` - Core model
- `/Users/yuechen/home/OminiX-MLX/qwen3-mlx/src/lib.rs` - Public API

### glm4-mlx
- `/Users/yuechen/home/OminiX-MLX/glm4-mlx/src/model.rs` - GLM-4 implementation

### mixtral-mlx
- `/Users/yuechen/home/OminiX-MLX/mixtral-mlx/src/model.rs` - MoE implementation

### Shared Infrastructure
- `/Users/yuechen/home/OminiX-MLX/mlx-rs-core/src/cache.rs` - KVCache
- `/Users/yuechen/home/OminiX-MLX/mlx-rs-core/src/utils.rs` - RoPE utilities
- `/Users/yuechen/home/OminiX-MLX/mlx-rs-core/src/metal_kernels.rs` - Fused kernels

### Audio Utilities (Reusable)
- `/Users/yuechen/home/OminiX-MLX/funasr-mlx/src/audio.rs` - Mel, resampling
