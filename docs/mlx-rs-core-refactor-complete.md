# MLX-RS Core Refactoring - Complete

## Summary

Refactored the monorepo to centralize shared infrastructure in `mlx-rs-core` and move model-specific code to dedicated `xxx-mlx` crates.

## Changes Made

### 1. Created mlx-rs-core (Shared Infrastructure)

**Location:** `/mlx-rs-core/`

**Contains:**
- `cache.rs` - KV cache implementations (KeyValueCache, ConcatKeyValueCache, KVCache)
- `utils.rs` - RoPE, attention masks, scaled dot-product attention, try_unwrap macro
- `metal_kernels.rs` - Custom Metal kernels (fused_swiglu for MoE)
- `error.rs` - Common error types
- `audio.rs` - Audio processing (WAV loading, mel spectrograms, resampling)
- `sampler.rs` - Token sampling (DefaultSampler trait)
- `generate/` - Generic token generation infrastructure
- `speculative.rs` - Speculative decoding support

**Exports:**
```rust
pub use cache::{ConcatKeyValueCache, KVCache, KeyValueCache};
pub use error::{Error, Result};
pub use metal_kernels::fused_swiglu;
pub use sampler::{DefaultSampler, Sampler};
pub use utils::{create_attention_mask, initialize_rope, scaled_dot_product_attention, ...};
pub use tokenizers::Tokenizer;

// Generic model traits
pub trait ModelInput<'a, C, T> { ... }
pub trait ModelOutput { ... }
pub struct ModelInputBuilder<'a, C, T> { ... }
```

### 2. LLM Model Crates

| Crate | Models | Source |
|-------|--------|--------|
| `qwen3-mlx` | Qwen2, Qwen3, Qwen3-MoE | Moved from mlx-rs-lm |
| `glm4-mlx` | GLM4 | Moved from mlx-rs-lm |
| `glm4-moe-mlx` | GLM4-MoE | Moved from mlx-rs-lm |
| `mixtral-mlx` | Mixtral | Moved from mlx-rs-lm |

All LLM crates now depend on `mlx-rs-core` for shared components.

### 3. Removed mlx-rs-lm

The `mlx-rs-lm` crate was removed entirely:
- LLM models → moved to respective `xxx-mlx` crates
- TTS models → already exist in `gpt-sovits-mlx/rust/`
- Shared infrastructure → moved to `mlx-rs-core`

### 4. Updated Dependencies

**Root Cargo.toml:**
- Removed `mlx-rs-lm` from workspace members
- Added `mlx-rs-core` to workspace dependencies

**examples/lm:**
- Updated to use `qwen3-mlx` instead of `mlx-rs-lm`

## Final Crate Structure

```
OminiX-MLX/
├── mlx-rs/                 # Core MLX Rust bindings
├── mlx-sys/                # MLX C++ FFI bindings
├── mlx-rs-core/            # Shared inference infrastructure
│   ├── src/
│   │   ├── cache.rs        # KV cache
│   │   ├── utils.rs        # RoPE, attention, SDPA
│   │   ├── metal_kernels.rs# Custom Metal kernels
│   │   ├── error.rs        # Error types
│   │   ├── audio.rs        # Audio processing
│   │   ├── sampler.rs      # Token sampling
│   │   ├── generate/       # Generic generation
│   │   └── speculative.rs  # Speculative decoding
│   └── Cargo.toml
│
├── qwen3-mlx/              # Qwen model family
├── glm4-mlx/               # GLM4
├── glm4-moe-mlx/           # GLM4-MoE
├── mixtral-mlx/            # Mixtral
├── funasr-mlx/             # FunASR (Python-based)
├── funasr-nano-mlx/        # FunASR Paraformer (Rust)
├── gpt-sovits-mlx/         # GPT-SoVITS TTS
│   └── rust/               # Rust implementation
├── flux-klein-mlx/         # FLUX image generation
├── zimage-mlx/             # Z-Image generation
└── qwen-image-mlx/         # Qwen image generation
```

## Usage Example

```rust
use qwen3_mlx::{load_model, Generate, ConcatKeyValueCache};
use mlx_rs::ops::indexing::NewAxis;

let mut model = load_model("path/to/Qwen3-4B")?;
let mut cache = Vec::new();
let prompt = Array::from(&[1, 2, 3]).index(NewAxis);

let generator = Generate::<ConcatKeyValueCache>::new(
    &mut model, &mut cache, 0.7, &prompt
);

for token in generator.take(100) {
    // process tokens
}
```

## ASR Component Sharing

Both `funasr-mlx` and `funasr-nano-mlx` now share components via `mlx-rs-core`:

| Component | mlx-rs-core | funasr-mlx | funasr-nano-mlx |
|-----------|-------------|------------|-----------------|
| `load_wav()` | ✓ (canonical) | re-exports | has own (hound) |
| `resample()` | ✓ (high-quality sinc) | re-exports | has own (FFT-based) |
| `save_wav()` | ✓ | re-exports | ✗ |
| `AudioConfig` | ✓ (TTS defaults) | ✗ | ✓ (ASR defaults) |
| `MelFrontend` | ✗ | ✓ (in paraformer.rs) | ✓ |
| `apply_lfr()` | ✗ | ✗ | ✓ |
| KVCache | ✓ | ✗ | re-exports |

**funasr-mlx** now depends on `mlx-rs-core` and re-exports audio functions.

**funasr-nano-mlx** keeps its own MelFrontend and audio processing for ASR-specific optimizations (FFT-based resampling, LFR transformation).

## TTS (GPT-SoVITS) Integration

**gpt-sovits-mlx/rust/** now uses `mlx-rs-core` for shared components:

| Component | Before | After |
|-----------|--------|-------|
| `audio.rs` | 665 lines (duplicate) | 21 lines (re-exports) |
| `cache.rs` | 200+ lines (duplicate) | 9 lines (re-exports) |
| Dependencies | hound, rubato | mlx-rs-core (includes both) |

**Changes:**
- Added `mlx-rs-core` dependency
- Removed `hound` and `rubato` direct dependencies (now via mlx-rs-core)
- `audio.rs` re-exports: `load_wav`, `save_wav`, `resample`, `AudioConfig`, `compute_mel_spectrogram`, `load_audio_for_hubert`, `load_reference_mel`
- `cache.rs` re-exports: `KeyValueCache`, `ConcatKeyValueCache`, `KVCache`

## Summary of Shared Components

All model crates now use `mlx-rs-core` for:

| Component | LLM | ASR | TTS |
|-----------|-----|-----|-----|
| KVCache | ✓ | ✓ | ✓ |
| Audio I/O | - | ✓ | ✓ |
| Resample | - | ✓ | ✓ |
| RoPE | ✓ | - | - |
| Attention Masks | ✓ | - | - |
| SDPA | ✓ | - | - |
| Mel Spectrogram | - | - | ✓ |

## Next Steps

1. **Vision-Language**: Create `qwen-vl-mlx` for Qwen VL models
