# mlx-rs-core

Shared inference infrastructure for MLX Rust model crates.

## Overview

`mlx-rs-core` provides common components used across all model-specific crates:

- **KV Cache**: Efficient key-value caching for autoregressive generation
- **Attention**: Scaled dot-product attention with mask support
- **RoPE**: Rotary position embeddings
- **Audio**: WAV I/O, resampling, mel spectrograms
- **Sampling**: Token sampling strategies
- **Metal Kernels**: Fused operations (SwiGLU for MoE)

## Installation

```toml
[dependencies]
mlx-rs-core = { path = "../mlx-rs-core" }
```

## Components

### KV Cache

```rust
use mlx_rs_core::{KeyValueCache, ConcatKeyValueCache, KVCache};

// Simple concatenating cache
let mut cache: Vec<Option<ConcatKeyValueCache>> = vec![None; num_layers];

// Use in attention
let (keys, values) = cache[layer].update_and_fetch(k, v)?;
```

### Attention Utilities

```rust
use mlx_rs_core::{
    create_attention_mask,
    scaled_dot_product_attention,
    initialize_rope,
    AttentionMask,
    SdpaMask,
};

// Create causal mask for prefill
let mask = create_attention_mask(&hidden_states, &cache, None)?;

// Initialize RoPE
let rope = initialize_rope(head_dim, base, traditional, &scaling_config, max_pos)?;

// Scaled dot-product attention
let output = scaled_dot_product_attention(
    queries, keys, values, None, scale, Some(SdpaMask::Causal)
)?;
```

### Audio Processing

```rust
use mlx_rs_core::audio::{load_wav, save_wav, resample, AudioConfig};

// Load WAV file
let (samples, sample_rate) = load_wav("input.wav")?;

// Resample to 16kHz
let samples_16k = resample(&samples, sample_rate, 16000);

// Save output
save_wav(&output_samples, 16000, "output.wav")?;

// Compute mel spectrogram (for TTS)
let config = AudioConfig::default();
let mel = compute_mel_spectrogram(&samples, &config)?;
```

### Token Sampling

```rust
use mlx_rs_core::{Sampler, DefaultSampler};

let mut sampler = DefaultSampler;
let token = sampler.sample(&logits, temperature)?;
```

### Metal Kernels

```rust
use mlx_rs_core::fused_swiglu;

// Fused SwiGLU for MoE models (45x faster than separate ops)
let output = fused_swiglu(&x, &gate)?;
```

## Model Traits

For building custom generators:

```rust
use mlx_rs_core::{ModelInput, ModelOutput, ModelInputBuilder};

// Implement for your model's input type
impl<'a, C> ModelInput<'a, C, MyState> for MyModelInput<'a, C> {
    fn from_model_input_builder(builder: ModelInputBuilder<'a, C, MyState>) -> Self {
        // ...
    }
}
```

## Crates Using mlx-rs-core

| Crate | Components Used |
|-------|-----------------|
| qwen3-mlx | KVCache, RoPE, AttentionMask, SDPA |
| glm4-mlx | KVCache, RoPE, AttentionMask, SDPA |
| glm4-moe-mlx | KVCache, RoPE, SDPA, fused_swiglu |
| mixtral-mlx | KVCache, RoPE, AttentionMask, SDPA |
| funasr-mlx | load_wav, resample, save_wav |
| funasr-nano-mlx | KVCache |
| gpt-sovits-mlx | KVCache, Audio (all) |

## API Reference

### Cache Module

- `KeyValueCache` - Trait for KV cache implementations
- `ConcatKeyValueCache` - Simple concatenating cache
- `KVCache` - Alias for ConcatKeyValueCache

### Utils Module

- `initialize_rope()` - Create RoPE with scaling support
- `create_attention_mask()` - Generate causal/window masks
- `scaled_dot_product_attention()` - SDPA with mask support
- `AttentionMask` - Enum for mask types
- `SdpaMask` - SDPA-specific mask wrapper
- `FloatOrString` - Config value type

### Audio Module

- `load_wav()` - Load WAV file (16/24/32-bit)
- `save_wav()` - Save 16-bit PCM WAV
- `resample()` - High-quality sinc resampling
- `compute_mel_spectrogram()` - STFT-based mel features
- `load_audio_for_hubert()` - HuBERT preprocessing
- `AudioConfig` - Spectrogram configuration

### Error Module

- `Error` - Common error type
- `Result<T>` - Result alias

## License

MIT OR Apache-2.0
