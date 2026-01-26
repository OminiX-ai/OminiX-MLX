# gpt-sovits-mlx (Rust)

Pure Rust implementation of GPT-SoVITS voice cloning with MLX acceleration.

## Features

- **Few-shot voice cloning**: Clone any voice with just a few seconds of reference audio
- **Mixed Chinese-English**: Natural handling of mixed language text with G2PW
- **High performance**: 4x real-time synthesis on Apple Silicon
- **Pure Rust**: No Python dependencies at inference time
- **GPU accelerated**: Metal GPU via MLX for all operations

## Installation

```toml
[dependencies]
gpt-sovits-mlx = { path = "../gpt-sovits-mlx/rust" }
```

## Quick Start

```rust
use gpt_sovits_mlx::VoiceCloner;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create voice cloner with default models
    let mut cloner = VoiceCloner::with_defaults()?;

    // Set reference audio for voice cloning
    cloner.set_reference_audio("reference.wav")?;

    // Synthesize speech
    let audio = cloner.synthesize("Hello, world!")?;

    // Save output
    cloner.save_wav(&audio, "output.wav")?;

    Ok(())
}
```

## CLI Usage

### Voice Cloning

```bash
# Basic voice cloning
cargo run --release --example voice_clone -- \
    --reference ./audio/reference.wav \
    --text "Hello, this is a voice clone test."

# With custom model paths
cargo run --release --example voice_clone -- \
    --model-dir ./models/gpt-sovits \
    --reference ./audio/speaker.wav \
    --text "你好，世界！"
```

### Model Download

Download pre-converted MLX models:

```bash
# GPT-SoVITS model pack (includes all required files)
huggingface-cli download lj1995/GPT-SoVITS-MLX --local-dir ./models/gpt-sovits

# Or download individual components:
# CNHubert encoder
huggingface-cli download TencentGameMate/chinese-hubert-base --local-dir ./models/hubert

# Chinese BERT
huggingface-cli download hfl/chinese-roberta-wwm-ext --local-dir ./models/bert
```

### Model Files

A complete model setup requires:

```
models/gpt-sovits/
├── gpt.safetensors          # GPT-SoVITS T2S model
├── sovits.safetensors       # SoVITS VITS decoder
├── cnhubert.safetensors     # CNHubert audio encoder
├── bert.safetensors         # Chinese BERT
├── g2pw.onnx                # G2PW polyphone model
└── tokenizer.json           # BERT tokenizer
```

## API Reference

### VoiceCloner

```rust
use gpt_sovits_mlx::{VoiceCloner, VoiceClonerConfig};

// Create with custom config
let config = VoiceClonerConfig {
    gpt_path: "./models/gpt.safetensors".into(),
    sovits_path: "./models/sovits.safetensors".into(),
    ..Default::default()
};
let mut cloner = VoiceCloner::new(config)?;

// Set reference audio (required for voice cloning)
cloner.set_reference_audio("reference.wav")?;

// Synthesize text to speech
let audio = cloner.synthesize("Text to synthesize")?;

// Get audio output
let samples: Vec<f32> = audio.samples();
let sample_rate = audio.sample_rate();
```

### Text Processing

```rust
use gpt_sovits_mlx::text::{preprocess_text, Language};

// Preprocess text with language detection
let text = "Hello 你好 world!";
let (phonemes, language) = preprocess_text(text)?;

// Or specify language explicitly
let phonemes = preprocess_text_with_language(text, Language::Chinese)?;
```

### Audio I/O

```rust
use gpt_sovits_mlx::audio::{load_wav, save_wav, resample};

// Load audio file
let (samples, sample_rate) = load_wav("input.wav")?;

// Resample to target rate
let samples_16k = resample(&samples, sample_rate, 16000);

// Save audio
save_wav(&samples, 24000, "output.wav")?;
```

## Architecture

```
                    GPT-SoVITS Pipeline

Text Input          Reference Audio
    │                    │
    ▼                    ▼
┌─────────┐        ┌─────────────┐
│  G2PW   │        │  CNHubert   │
│ (ONNX)  │        │   Encoder   │
└────┬────┘        └──────┬──────┘
     │                    │
     ▼                    ▼
┌─────────┐        ┌─────────────┐
│  BERT   │        │ Quantizer   │
│Embedding│        │  (Codes)    │
└────┬────┘        └──────┬──────┘
     │                    │
     └────────┬───────────┘
              │
              ▼
       ┌─────────────┐
       │  GPT T2S    │  (Text-to-Semantic)
       │  Decoder    │
       └──────┬──────┘
              │
              ▼
       ┌─────────────┐
       │   SoVITS    │  (VITS Vocoder)
       │   Decoder   │
       └──────┬──────┘
              │
              ▼
         Audio Output
```

## Components

| Module | Description |
|--------|-------------|
| `audio` | WAV I/O, resampling, mel spectrogram (re-exports from mlx-rs-core) |
| `cache` | KV cache for autoregressive generation (re-exports from mlx-rs-core) |
| `text` | G2PW, pinyin, language detection, phoneme processing |
| `models/t2s` | GPT text-to-semantic transformer |
| `models/vits` | SoVITS VITS vocoder |
| `models/hubert` | CNHubert audio encoder |
| `models/bert` | Chinese BERT embeddings |
| `inference` | T2S generation with cache |
| `voice_clone` | High-level voice cloning API |

## Performance

Benchmarks on Apple M3 Max:

| Stage | Time | Notes |
|-------|------|-------|
| Reference processing | ~50ms | CNHubert + quantization |
| BERT embedding | ~20ms | Text encoding |
| T2S generation | ~100ms | GPT decoding (variable) |
| VITS synthesis | ~50ms | Audio generation |
| **Total** | ~220ms | For 2s audio output |

Real-time factor: **~4x** (generates 2s audio in 500ms)

## Shared Components

This crate uses shared infrastructure from `mlx-rs-core`:

| Component | Source |
|-----------|--------|
| `load_wav`, `save_wav`, `resample` | mlx-rs-core::audio |
| `KVCache`, `ConcatKeyValueCache` | mlx-rs-core::cache |
| `compute_mel_spectrogram` | mlx-rs-core::audio |

## Development

```bash
# Build
cargo build --release -p gpt-sovits-mlx

# Run tests
cargo test -p gpt-sovits-mlx

# Run with debug output
cargo run --release --example voice_clone --features debug-attn
```

## License

MIT
