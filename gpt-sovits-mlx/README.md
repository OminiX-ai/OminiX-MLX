# gpt-sovits-mlx (Rust)

Pure Rust implementation of GPT-SoVITS voice cloning with MLX acceleration.

## Features

- **Few-shot voice cloning**: Clone any voice with just a few seconds of reference audio
- **Mixed Chinese-English**: Natural handling of mixed language text with G2PW
- **High performance**: 4x real-time synthesis on Apple Silicon
- **Pure Rust**: No Python dependencies at inference time
- **GPU accelerated**: Metal GPU via MLX for all operations

## Prerequisites

- **Rust** (stable, 1.75+)
- **Python 3.10+** (for one-time model setup only)
- **macOS** with Apple Silicon (M1/M2/M3/M4)

## First-Time Setup

Download and convert all required model weights (~2GB download):

```bash
python scripts/setup_models.py
```

This automatically:
1. Installs Python dependencies (torch CPU, safetensors, transformers, huggingface_hub)
2. Downloads pretrained checkpoints from HuggingFace
3. Converts everything to MLX-compatible safetensors format
4. Places output in `~/.dora/models/primespeech/gpt-sovits-mlx/`

After setup, the model directory will contain:

```
~/.dora/models/primespeech/gpt-sovits-mlx/
├── doubao_mixed_gpt_new.safetensors         # GPT T2S model
├── doubao_mixed_sovits_new.safetensors      # SoVITS VITS decoder
├── hubert.safetensors                       # CNHubert audio encoder
├── bert.safetensors                         # Chinese BERT
└── chinese-roberta-tokenizer/
    └── tokenizer.json                       # BERT tokenizer
```

> **Note:** The ONNX VITS model (`vits.onnx`) is not included in setup.
> The Rust code automatically falls back to MLX VITS when ONNX is not available.
> You can also pass `--mlx-vits` explicitly.

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
# Basic voice cloning (uses MLX VITS automatically when ONNX is absent)
cargo run --release --example voice_clone -- \
    --mlx-vits --ref ./audio/reference.wav "Hello, this is a voice clone test."

# With custom model directory
cargo run --release --example voice_clone -- \
    --model-dir ./models/gpt-sovits \
    --mlx-vits --ref ./audio/speaker.wav "你好，世界！"
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
