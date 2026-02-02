# GPT-SoVITS MLX Migration: Architecture Document

**Version:** 1.0
**Date:** 2026-01-20
**Status:** Draft

---

## 1. Executive Summary

This document defines the architecture for migrating dora-primespeech (GPT-SoVITS TTS) from Python/PyTorch to a high-performance hybrid architecture using:

- **CoreML** (Apple Neural Engine) for encoder models (CNHubert, RoBERTa)
- **MLX/mlx-rs** (GPU) for autoregressive generation and vocoding

**Target Performance:** 8-10x speedup over CPU PyTorch (~120ms total latency on M2 Pro)

---

## 2. Current Architecture (Baseline)

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Current: Python/PyTorch Architecture                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Dora Framework (Rust)                                                   │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Python Runtime (GIL)                          │    │
│  │                                                                  │    │
│  │  Input Text ──► Text Processing ──► Phoneme IDs                 │    │
│  │                      │                                           │    │
│  │                      ▼                                           │    │
│  │  Reference Audio ──► CNHubert ──► Audio Features                │    │
│  │                      │ (PyTorch)    768-dim @ 50Hz              │    │
│  │                      │                                           │    │
│  │                      ▼                                           │    │
│  │  Phonemes + Audio ──► GPT Model ──► Semantic Tokens             │    │
│  │                      │ (PyTorch)    ~100-500 tokens             │    │
│  │                      │ 12 layers, autoregressive                │    │
│  │                      │                                           │    │
│  │                      ▼                                           │    │
│  │  Semantic Tokens ──► SoVITS ──► Waveform (32kHz)                │    │
│  │                      (PyTorch)                                   │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Performance: ~1000ms per utterance (CPU), ~270ms (CUDA)                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Performance Breakdown (Current)

| Component | CPU Time | CUDA Time | % Total (CPU) |
|-----------|----------|-----------|---------------|
| Text Processing | 30ms | 30ms | 3% |
| CNHubert Encoder | 150ms | 50ms | 15% |
| RoBERTa (optional) | 100ms | 35ms | 10% |
| GPT Generation | 600ms | 100ms | 57% |
| SoVITS Vocoder | 200ms | 70ms | 19% |
| **Total** | **~1080ms** | **~285ms** | 100% |

### 2.3 Bottleneck Analysis

```
                    Compute Intensity Map

Component        Memory-Bound ◄──────────────► Compute-Bound
─────────────────────────────────────────────────────────────
Text Processing  ████░░░░░░░░░░░░░░░░  (I/O bound)
CNHubert         ░░░░░░░░░░████████░░  (encoder, parallelizable)
RoBERTa          ░░░░░░░░░░████████░░  (encoder, parallelizable)
GPT Stage        ████████████░░░░░░░░  (memory-bound, sequential)
SoVITS Vocoder   ░░░░░░████████░░░░░░  (mixed, parallelizable)

Legend: █ = Primary characteristic
```

**Key Insight:**
- Encoders (CNHubert, RoBERTa) are **compute-bound** → ANE excels
- GPT stage is **memory-bound + sequential** → GPU with optimized KV cache
- Vocoder is **mixed** → GPU with fused kernels

---

## 3. Target Architecture (Hybrid)

### 3.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                Target: Hybrid CoreML + MLX Architecture                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Dora Framework (Rust)                                                   │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                         Orchestrator                             │    │
│  │                    (Python or Rust FFI)                          │    │
│  └───────┬─────────────────────┬─────────────────────┬─────────────┘    │
│          │                     │                     │                   │
│          ▼                     ▼                     ▼                   │
│  ┌───────────────┐    ┌───────────────┐    ┌────────────────────┐       │
│  │ Text Process  │    │   CoreML      │    │      MLX/mlx-rs    │       │
│  │   (Python)    │    │   (ANE)       │    │       (GPU)        │       │
│  │               │    │               │    │                    │       │
│  │ • G2P         │    │ • CNHubert    │    │ • GPT Generator    │       │
│  │ • Phonemizer  │    │ • RoBERTa     │    │ • SoVITS Vocoder   │       │
│  │ • Normalizer  │    │               │    │ • KV Cache         │       │
│  │               │    │ ~10ms total   │    │ ~110ms total       │       │
│  │ ~30ms         │    │               │    │                    │       │
│  └───────────────┘    └───────────────┘    └────────────────────┘       │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Unified Memory (Apple Silicon)                │    │
│  │         Zero-copy data sharing between all components            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Target Performance: ~120ms per utterance (8-10x speedup)               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Data Flow Architecture                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│    Input                                                                  │
│      │                                                                    │
│      ├─► Text ─────────────────────────────────────────────┐             │
│      │         │                                           │             │
│      │         ▼                                           │             │
│      │   ┌──────────┐    phoneme_ids                       │             │
│      │   │ Text     │────────────────────────────┐         │             │
│      │   │ Process  │    [batch, seq_len]        │         │             │
│      │   └──────────┘                            │         │             │
│      │                                           │         │             │
│      └─► Audio ────────────────────────┐         │         │             │
│                │                       │         │         │             │
│                ▼                       │         │         │             │
│          ┌──────────┐                  │         │         │             │
│          │ Resample │ 16kHz            │         │         │             │
│          └────┬─────┘                  │         │         │             │
│               │                        │         │         │             │
│               ▼                        ▼         ▼         │             │
│    ┌─────────────────────────────────────────────────┐     │             │
│    │              CoreML Runtime (ANE)               │     │             │
│    │  ┌─────────────┐      ┌─────────────────────┐  │     │             │
│    │  │  CNHubert   │      │  RoBERTa (optional) │  │     │             │
│    │  │  Encoder    │      │  Text Encoder       │  │     │             │
│    │  └──────┬──────┘      └──────────┬──────────┘  │     │             │
│    │         │                        │             │     │             │
│    │         ▼                        ▼             │     │             │
│    │   audio_features          text_features       │     │             │
│    │   [1, 768, T/320]         [1, seq, 1024]      │     │             │
│    └─────────┬────────────────────────┬────────────┘     │             │
│              │                        │                   │             │
│              │    Zero-copy via       │                   │             │
│              │    Unified Memory      │                   │             │
│              ▼                        ▼                   ▼             │
│    ┌───────────────────────────────────────────────────────────┐       │
│    │                    MLX Runtime (GPU)                       │       │
│    │                                                            │       │
│    │  ┌──────────────────────────────────────────────────────┐ │       │
│    │  │                   GPT Generator                       │ │       │
│    │  │  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │ │       │
│    │  │  │  Phoneme   │  │ Transformer│  │   Sampling     │  │ │       │
│    │  │  │  Embedding │─►│  12 Layers │─►│   (top-k)      │  │ │       │
│    │  │  └────────────┘  │ + KV Cache │  └───────┬────────┘  │ │       │
│    │  │                  └────────────┘          │           │ │       │
│    │  │                                          ▼           │ │       │
│    │  │                              semantic_tokens         │ │       │
│    │  │                              [1, ~100-500]           │ │       │
│    │  └──────────────────────────────────────────────────────┘ │       │
│    │                         │                                  │       │
│    │                         ▼                                  │       │
│    │  ┌──────────────────────────────────────────────────────┐ │       │
│    │  │                   SoVITS Vocoder                      │ │       │
│    │  │  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │ │       │
│    │  │  │  Duration  │  │    RVQ     │  │   Upsampler    │  │ │       │
│    │  │  │  Predictor │─►│   Decoder  │─►│   (ConvT)      │  │ │       │
│    │  │  └────────────┘  └────────────┘  └───────┬────────┘  │ │       │
│    │  │                                          │           │ │       │
│    │  └──────────────────────────────────────────────────────┘ │       │
│    │                         │                                  │       │
│    └─────────────────────────┼──────────────────────────────────┘       │
│                              │                                          │
│                              ▼                                          │
│                         Output Audio                                    │
│                         [samples] @ 32kHz                               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Target Performance

| Component | Target Time | Accelerator | Speedup vs CPU |
|-----------|-------------|-------------|----------------|
| Text Processing | 30ms | CPU | 1x |
| CNHubert Encoder | 5ms | ANE | 30x |
| RoBERTa (optional) | 5ms | ANE | 20x |
| GPT Generation | 60ms | GPU (MLX) | 10x |
| SoVITS Vocoder | 50ms | GPU (MLX) | 4x |
| **Total** | **~120ms** | Hybrid | **~9x** |

---

## 4. Detailed Component Design

### 4.1 CoreML Encoders Module

#### 4.1.1 CNHubert Encoder

```
┌─────────────────────────────────────────────────────────────┐
│                    CNHubert CoreML Model                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: audio_waveform [1, samples] @ 16kHz                 │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Feature Extractor (7 Conv layers)                      │ │
│  │ ├── Conv1d(1, 512, k=10, s=5)                         │ │
│  │ ├── Conv1d(512, 512, k=3, s=2) x 4                    │ │
│  │ ├── Conv1d(512, 512, k=2, s=2) x 2                    │ │
│  │ └── GroupNorm + GELU                                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                         │                                    │
│                         ▼                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Transformer Encoder (12 layers)                        │ │
│  │ ├── Self-Attention (12 heads, 768 dim)                │ │
│  │ ├── Feed-Forward (768 → 3072 → 768)                   │ │
│  │ └── LayerNorm + Residual                              │ │
│  └────────────────────────────────────────────────────────┘ │
│                         │                                    │
│                         ▼                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Feature Projection                                     │ │
│  │ └── Linear(768, 768)                                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Output: audio_features [1, 768, T/320]                     │
│                                                              │
│  ANE Optimizations:                                          │
│  • Conv2d format (B, C, 1, T) instead of Conv1d             │
│  • Chunked attention heads                                   │
│  • FP16 precision                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 4.1.2 RoBERTa Text Encoder

```
┌─────────────────────────────────────────────────────────────┐
│                   RoBERTa CoreML Model                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: token_ids [1, seq_len]                              │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Embedding Layer                                        │ │
│  │ ├── Token Embedding (vocab=21128, dim=1024)           │ │
│  │ ├── Position Embedding (max_len=512)                  │ │
│  │ └── LayerNorm                                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                         │                                    │
│                         ▼                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Transformer Encoder (24 layers for large)              │ │
│  │ ├── Self-Attention (16 heads, 1024 dim)               │ │
│  │ ├── Feed-Forward (1024 → 4096 → 1024)                 │ │
│  │ └── LayerNorm + Residual                              │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Output: text_features [1, seq_len, 1024]                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 MLX GPT Generator

#### 4.2.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GPT-SoVITS GPT (MLX)                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Config:                                                     │
│  ├── hidden_size: 512                                       │
│  ├── num_layers: 12                                         │
│  ├── num_heads: 8                                           │
│  ├── head_dim: 64                                           │
│  ├── phoneme_vocab: 512                                     │
│  ├── semantic_vocab: 1025                                   │
│  └── max_seq_len: 1024                                      │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Input Processing                                       │ │
│  │ ├── phoneme_embed: Embedding(512, 512)                │ │
│  │ ├── semantic_embed: Embedding(1025, 512)              │ │
│  │ ├── audio_proj: Linear(768, 512)  # from CNHubert     │ │
│  │ └── pos_embed: Sinusoidal(1024, 512)                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                         │                                    │
│                         ▼                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Transformer Decoder (12 layers)                        │ │
│  │                                                        │ │
│  │  For each layer:                                       │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │ TransformerBlock                                 │ │ │
│  │  │ ├── ln1: RMSNorm(512)                           │ │ │
│  │  │ ├── self_attn: MultiHeadAttention               │ │ │
│  │  │ │   ├── q_proj: Linear(512, 512)               │ │ │
│  │  │ │   ├── k_proj: Linear(512, 512)               │ │ │
│  │  │ │   ├── v_proj: Linear(512, 512)               │ │ │
│  │  │ │   ├── o_proj: Linear(512, 512)               │ │ │
│  │  │ │   └── KVCache (step=256)                     │ │ │
│  │  │ ├── ln2: RMSNorm(512)                           │ │ │
│  │  │ └── mlp: MLP                                    │ │ │
│  │  │     ├── gate_proj: Linear(512, 2048)           │ │ │
│  │  │     ├── up_proj: Linear(512, 2048)             │ │ │
│  │  │     ├── down_proj: Linear(2048, 512)           │ │ │
│  │  │     └── activation: SiLU (fused SwiGLU)        │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  │                                                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                         │                                    │
│                         ▼                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Output Head                                            │ │
│  │ ├── ln_f: RMSNorm(512)                                │ │
│  │ └── lm_head: Linear(512, 1025)                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Output: logits [batch, seq, 1025]                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 4.2.2 KV Cache Design

```
┌─────────────────────────────────────────────────────────────┐
│                   Step-Allocated KV Cache                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  struct KVCache {                                            │
│      keys: Array,      // [batch, heads, max_len, head_dim] │
│      values: Array,    // [batch, heads, max_len, head_dim] │
│      offset: usize,    // Current position                  │
│      step: usize,      // Allocation step (256)             │
│  }                                                           │
│                                                              │
│  Memory Layout (example: 512 tokens generated):              │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Allocated: 512 tokens (2 steps × 256)               │    │
│  │                                                      │    │
│  │ Keys:   [████████████████████████████████████████]  │    │
│  │          0        128       256       384      512  │    │
│  │                                                      │    │
│  │ Values: [████████████████████████████████████████]  │    │
│  │          0        128       256       384      512  │    │
│  │                                                      │    │
│  │ offset = 512                                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Update Operation (O(1) per token):                          │
│                                                              │
│  fn update(&mut self, new_k, new_v):                        │
│      // Resize only when needed                              │
│      if self.offset + 1 > self.keys.dim(2):                 │
│          self.resize(self.offset + self.step)               │
│                                                              │
│      // In-place slice update (no copy!)                    │
│      self.keys[..., self.offset, :] = new_k                 │
│      self.values[..., self.offset, :] = new_v               │
│      self.offset += 1                                        │
│                                                              │
│  Performance: 0.02-0.05ms per token (vs 1-2ms for concat)   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 MLX SoVITS Vocoder

```
┌─────────────────────────────────────────────────────────────┐
│                   SoVITS Vocoder (MLX)                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: semantic_tokens [batch, seq_len]                    │
│         audio_features [batch, 768, T]  (from CNHubert)     │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Duration Predictor (Flow-based)                        │ │
│  │ ├── Pre-net: Conv1d(256, 256, k=3) × 2                │ │
│  │ ├── Flow layers: 4 × AffineCouplingLayer              │ │
│  │ └── Output: durations [batch, seq_len]                │ │
│  └────────────────────────────────────────────────────────┘ │
│                         │                                    │
│                         ▼                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Length Regulator                                       │ │
│  │ └── Expand semantic tokens by predicted durations     │ │
│  └────────────────────────────────────────────────────────┘ │
│                         │                                    │
│                         ▼                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ RVQ Decoder (8 codebooks)                              │ │
│  │ ├── Codebook 0: Embedding(1024, 256)                  │ │
│  │ ├── Codebook 1-7: Residual refinement                 │ │
│  │ └── Sum all codebook outputs                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                         │                                    │
│                         ▼                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ MRTE Block (Multi-Resolution Temporal Encoding)        │ │
│  │ ├── Conv1d at multiple dilations (1, 2, 4, 8)         │ │
│  │ └── Attention over temporal features                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                         │                                    │
│                         ▼                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Upsampler (ConvTranspose stack)                        │ │
│  │ ├── ConvT1d(256, 256, k=16, s=8)  # 8x upsample      │ │
│  │ ├── ResBlock(256) × 3                                 │ │
│  │ ├── ConvT1d(256, 128, k=8, s=4)   # 4x upsample      │ │
│  │ ├── ResBlock(128) × 3                                 │ │
│  │ ├── ConvT1d(128, 64, k=4, s=2)    # 2x upsample      │ │
│  │ ├── ResBlock(64) × 3                                  │ │
│  │ └── Conv1d(64, 1, k=7)            # Output           │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Output: waveform [batch, samples] @ 32kHz                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Interface Definitions

### 5.1 Python API

```python
# gpt_sovits_mlx/api.py

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class SynthesisConfig:
    """Configuration for TTS synthesis."""
    voice_name: str = "Doubao"
    language: str = "zh"
    top_k: int = 3
    top_p: float = 0.95
    temperature: float = 0.8
    speed_factor: float = 1.0
    max_tokens: int = 500
    sample_rate: int = 32000

@dataclass
class SynthesisResult:
    """Result of TTS synthesis."""
    audio: np.ndarray          # Float32 waveform
    sample_rate: int           # 32000
    duration: float            # Seconds
    semantic_tokens: np.ndarray # For debugging
    timing: dict               # Component timings

class GPTSoVITSEngine:
    """High-level TTS engine using hybrid CoreML + MLX."""

    def __init__(
        self,
        model_dir: str,
        device: str = "auto",  # "auto", "ane", "gpu", "cpu"
        use_ane_encoders: bool = True,
    ):
        """Initialize the TTS engine."""
        ...

    def load_voice(self, voice_name: str) -> None:
        """Load a voice model (GPT + SoVITS weights)."""
        ...

    def synthesize(
        self,
        text: str,
        config: Optional[SynthesisConfig] = None,
    ) -> SynthesisResult:
        """Synthesize speech from text."""
        ...

    def synthesize_streaming(
        self,
        text: str,
        config: Optional[SynthesisConfig] = None,
        chunk_size: int = 4096,
    ) -> Iterator[np.ndarray]:
        """Streaming synthesis yielding audio chunks."""
        ...

    def warmup(self) -> None:
        """Warmup models for optimal first-inference latency."""
        ...
```

### 5.2 Rust FFI (mlx-rs)

```rust
// mlx-rs-lm/src/gpt_sovits/ffi.rs

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

/// Generate semantic tokens from phoneme sequence.
#[pyfunction]
fn generate_semantic_tokens(
    py: Python,
    phoneme_ids: PyReadonlyArray1<i32>,
    audio_features: PyReadonlyArray2<f32>,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> PyResult<Py<PyArray1<i32>>> {
    // Implementation
}

/// Run SoVITS vocoder on semantic tokens.
#[pyfunction]
fn vocode(
    py: Python,
    semantic_tokens: PyReadonlyArray1<i32>,
    audio_features: PyReadonlyArray2<f32>,
    speed_factor: f32,
) -> PyResult<Py<PyArray1<f32>>> {
    // Implementation
}

/// Initialize models and warmup.
#[pyfunction]
fn init_models(
    py: Python,
    gpt_path: &str,
    sovits_path: &str,
) -> PyResult<()> {
    // Implementation
}

#[pymodule]
fn gpt_sovits_mlx_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_semantic_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(vocode, m)?)?;
    m.add_function(wrap_pyfunction!(init_models, m)?)?;
    Ok(())
}
```

### 5.3 Dora Node Interface

```yaml
# dataflow.yaml
nodes:
  - id: primespeech-mlx
    operator:
      python: dora_primespeech_mlx/main.py
    inputs:
      text: orchestrator/tts_text
      control: orchestrator/control
    outputs:
      - audio
      - segment_complete
      - status
      - log

# Input schemas
inputs:
  text:
    type: string
    metadata:
      session_id: string
      request_id: string
      question_id: string
      segment_index: int

  control:
    type: string
    enum: [reset, stats, list_voices, "change_voice:*", cleanup]

# Output schemas
outputs:
  audio:
    type: float32[]
    metadata:
      sample_rate: 32000
      duration: float
      voice: string

  segment_complete:
    type: string
    enum: [completed, error, skipped]
    metadata:
      question_id: string
```

---

## 6. Memory Architecture

### 6.1 Unified Memory Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Apple Silicon Unified Memory                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Total Available: 16GB (M2 Pro) / 32GB (M2 Max) / 64GB (M2 Ultra)       │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Model Weights (Persistent)                          ~2.5GB      │    │
│  │ ├── CNHubert CoreML:          ~400MB                           │    │
│  │ ├── RoBERTa CoreML:           ~500MB                           │    │
│  │ ├── GPT MLX:                  ~500MB                           │    │
│  │ └── SoVITS MLX:               ~1.0GB                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ KV Cache (Dynamic, per-inference)                  ~100-500MB   │    │
│  │ ├── 12 layers × 2 (K,V)                                        │    │
│  │ ├── Shape: [1, 8, max_seq, 64]                                 │    │
│  │ └── Step-allocated in 256-token blocks                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Intermediate Activations (Transient)               ~200-500MB   │    │
│  │ ├── Audio features: [1, 768, T]                                │    │
│  │ ├── Hidden states: [1, seq, 512]                               │    │
│  │ └── Vocoder features: [1, 256, T×64]                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Audio Buffers                                      ~10-50MB     │    │
│  │ ├── Input reference: [samples] @ 16kHz                         │    │
│  │ └── Output waveform: [samples] @ 32kHz                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Peak Usage: ~3-4GB                                                      │
│  Recommended System RAM: 16GB+                                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Zero-Copy Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Zero-Copy Pipeline                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CoreML Output                    MLX Input                              │
│       │                               │                                  │
│       ▼                               ▼                                  │
│  ┌─────────────┐    memoryview    ┌─────────────┐                       │
│  │ MLMultiArray│ ──────────────► │  mx.array   │                       │
│  │ (ANE output)│   (no copy!)    │ (GPU input) │                       │
│  └─────────────┘                  └─────────────┘                       │
│                                                                          │
│  Implementation:                                                         │
│                                                                          │
│  # CoreML output                                                         │
│  coreml_output = model.predict(inputs)                                  │
│  ml_array = coreml_output["features"]  # MLMultiArray                   │
│                                                                          │
│  # Zero-copy to MLX                                                      │
│  import mlx.core as mx                                                   │
│  mlx_array = mx.array(                                                   │
│      np.asarray(ml_array),  # Uses buffer protocol, no copy             │
│      copy=False              # Explicit no-copy                          │
│  )                                                                       │
│                                                                          │
│  # Verify same memory                                                    │
│  assert mlx_array.ctypes.data == ml_array.__array_interface__['data'][0]│
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Error Handling & Recovery

### 7.1 Error Categories

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Error Handling Strategy                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Category 1: Initialization Errors (Fatal)                               │
│  ├── Model file not found                                               │
│  ├── CoreML compilation failure                                          │
│  ├── Insufficient memory                                                 │
│  └── Action: Log error, return failure to Dora orchestrator             │
│                                                                          │
│  Category 2: Inference Errors (Recoverable)                              │
│  ├── Text too long → Truncate with warning                              │
│  ├── Invalid phonemes → Skip unknown, continue                          │
│  ├── GPU memory pressure → Retry with smaller batch                     │
│  └── Action: Return partial result or retry                             │
│                                                                          │
│  Category 3: Runtime Errors (Graceful Degradation)                       │
│  ├── ANE unavailable → Fallback to GPU                                  │
│  ├── GPU unavailable → Fallback to CPU                                  │
│  ├── Timeout → Return partial audio                                     │
│  └── Action: Continue with degraded performance                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Fallback Chain

```python
def synthesize_with_fallback(text: str) -> SynthesisResult:
    """Synthesis with automatic fallback."""

    # Tier 1: Full hybrid (ANE + GPU)
    try:
        return synthesize_hybrid(text)
    except ANEUnavailable:
        logger.warning("ANE unavailable, falling back to GPU-only")

    # Tier 2: GPU-only (MLX)
    try:
        return synthesize_mlx_only(text)
    except GPUMemoryError:
        logger.warning("GPU memory pressure, falling back to CPU")

    # Tier 3: CPU fallback
    try:
        return synthesize_cpu(text)
    except Exception as e:
        logger.error(f"All backends failed: {e}")
        raise SynthesisError("Unable to synthesize audio")
```

---

## 8. Deployment Configuration

### 8.1 Environment Variables

```bash
# Model paths
GPTSVITS_MODEL_DIR=~/.dora/models/gpt-sovits-mlx
GPTSVITS_VOICE=Doubao

# Hardware configuration
GPTSVITS_USE_ANE=true           # Enable ANE for encoders
GPTSVITS_USE_GPU=true           # Enable GPU for GPT/vocoder
GPTSVITS_GPU_MEMORY_LIMIT=4096  # MB, for memory management

# Inference parameters
GPTSVITS_MAX_TOKENS=500
GPTSVITS_TEMPERATURE=0.8
GPTSVITS_TOP_K=3
GPTSVITS_TOP_P=0.95

# Performance tuning
GPTSVITS_KV_CACHE_STEP=256      # KV cache allocation step
GPTSVITS_WARMUP=true            # Warmup on startup
GPTSVITS_COMPILE=true           # Use mx.compile for fusion

# Streaming
GPTSVITS_STREAMING=false
GPTSVITS_CHUNK_SIZE=4096        # Samples per streaming chunk

# Logging
GPTSVITS_LOG_LEVEL=INFO
GPTSVITS_LOG_TIMING=true        # Log component timings
```

### 8.2 Model File Structure

```
~/.dora/models/gpt-sovits-mlx/
├── encoders/
│   ├── cnhubert_ane.mlpackage/     # CoreML CNHubert
│   │   ├── Data/
│   │   │   └── com.apple.CoreML/
│   │   └── Manifest.json
│   └── roberta_ane.mlpackage/      # CoreML RoBERTa
│
├── voices/
│   ├── Doubao/
│   │   ├── gpt.safetensors         # GPT weights
│   │   ├── sovits.safetensors      # SoVITS weights
│   │   ├── config.json             # Voice config
│   │   └── reference.wav           # Reference audio
│   ├── Trump/
│   │   └── ...
│   └── ...
│
├── base/
│   ├── phoneme_vocab.json          # Phoneme vocabulary
│   └── semantic_vocab.json         # Semantic vocabulary
│
└── config.yaml                     # Global configuration
```

---

## 9. Testing Strategy

### 9.1 Test Categories

```
┌─────────────────────────────────────────────────────────────┐
│                      Testing Pyramid                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                     ┌─────────┐                             │
│                     │  E2E    │  Integration with Dora      │
│                     │  Tests  │  ~5 tests, slow             │
│                    ─┴─────────┴─                            │
│                   ┌─────────────┐                           │
│                   │ Integration │  CoreML↔MLX handoff       │
│                   │   Tests     │  ~20 tests, medium        │
│                  ─┴─────────────┴─                          │
│                 ┌─────────────────┐                         │
│                 │   Component     │  Individual models      │
│                 │     Tests       │  ~50 tests, fast        │
│                ─┴─────────────────┴─                        │
│               ┌─────────────────────┐                       │
│               │     Unit Tests      │  Functions, utils     │
│               │                     │  ~100 tests, instant  │
│              ─┴─────────────────────┴─                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Benchmark Suite

```python
# tests/benchmark_suite.py

BENCHMARK_CONFIGS = [
    # (text_length, expected_latency_ms, tolerance_ms)
    ("short", "你好", 80, 20),
    ("medium", "今天天气真不错，我们一起去公园散步吧。", 120, 30),
    ("long", "..." * 100, 500, 100),
]

def benchmark_e2e_latency():
    """Benchmark end-to-end synthesis latency."""
    for name, text, expected, tolerance in BENCHMARK_CONFIGS:
        result = engine.synthesize(text)
        assert result.timing["total"] < expected + tolerance, \
            f"{name}: {result.timing['total']}ms > {expected + tolerance}ms"

def benchmark_component_latency():
    """Benchmark individual component latency."""
    # CNHubert: < 10ms
    # RoBERTa: < 10ms
    # GPT: < 100ms
    # Vocoder: < 80ms
    ...

def benchmark_memory_usage():
    """Benchmark peak memory usage."""
    # Peak should be < 4GB on M2 Pro
    ...

def benchmark_throughput():
    """Benchmark synthesis throughput."""
    # Target: > 5 utterances/second for short text
    ...
```

---

## 10. Appendix

### 10.1 Glossary

| Term | Definition |
|------|------------|
| ANE | Apple Neural Engine - dedicated ML accelerator |
| CoreML | Apple's ML framework for ANE deployment |
| GPT | Generative Pre-trained Transformer (semantic generation) |
| KV Cache | Key-Value cache for efficient autoregressive generation |
| MLX | Apple's array framework for ML on Apple Silicon |
| MRTE | Multi-Resolution Temporal Encoding |
| RVQ | Residual Vector Quantization |
| SoVITS | Variational Inference Text-to-Speech |
| TTS | Text-to-Speech |

### 10.2 References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [CoreML Documentation](https://developer.apple.com/documentation/coreml)
- [GPT-SoVITS Paper](https://arxiv.org/abs/2401.13193)
- [ane_transformers](https://machinelearning.apple.com/research/neural-engine-transformers)
- [mlx-examples](https://github.com/ml-explore/mlx-examples)
