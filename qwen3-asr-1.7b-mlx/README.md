# qwen3-asr-mlx

Qwen3-ASR speech recognition on Apple Silicon, written in Rust with [MLX](https://github.com/ml-explore/mlx).

Supports all Qwen3-ASR model sizes (0.6B, 1.7B) with 8-bit quantization. Architecture is fully config-driven — the same binary runs any variant.

## Features

- **30+ languages** — Chinese, English, Japanese, Korean, French, German, Spanish, and 23 more
- **30x–50x realtime** on Apple Silicon (M-series) with 8-bit quantized models
- **Long-form audio** — automatic 30-second chunking for files of any length
- **Config-driven** — one crate supports 0.6B and 1.7B; dimensions parsed from `config.json`
- **Tokenizer auto-build** — generates `tokenizer.json` from `vocab.json` + `merges.txt` if missing
- **Zero Python** — pure Rust, no Python runtime needed

## Quick Start

### 1. Download a model

```bash
# 1.7B 8-bit (recommended, 2.46 GB)
huggingface-cli download mlx-community/Qwen3-ASR-1.7B-8bit \
    --local-dir ~/.OminiX/models/qwen3-asr-1.7b

# 0.6B 8-bit (faster download, 1.01 GB)
huggingface-cli download mlx-community/Qwen3-ASR-0.6B-8bit \
    --local-dir ~/.OminiX/models/qwen3-asr-0.6b
```

### 2. Transcribe

```bash
# Using default model (1.7B)
cargo run --release --example transcribe -- audio.wav

# Using 0.6B model
cargo run --release --example transcribe -- ~/.OminiX/models/qwen3-asr-0.6b audio.wav

# English
cargo run --release --example transcribe -- audio.wav --language English

# Non-WAV formats (requires ffmpeg)
cargo run --release --example transcribe -- meeting.m4a
```

### 3. As a library

```rust
use qwen3_asr_mlx::{Qwen3ASR, default_model_path};

let mut model = Qwen3ASR::load(default_model_path())?;

// Simple
let text = model.transcribe("audio.wav")?;

// With language
let text = model.transcribe_with_language("audio.wav", "English")?;

// From raw samples (16kHz mono f32)
let text = model.transcribe_samples(&samples, "Chinese")?;
```

## Supported Models

| Model | HuggingFace Repo | Size | Speed |
|-------|-----------------|------|-------|
| **Qwen3-ASR-1.7B-8bit** | `mlx-community/Qwen3-ASR-1.7B-8bit` | 2.46 GB | ~30x RT |
| Qwen3-ASR-0.6B-8bit | `mlx-community/Qwen3-ASR-0.6B-8bit` | 1.01 GB | ~22x RT |

Speed measured on Apple M4 Max with 37-minute Chinese business meeting audio.

## Architecture

```
Audio (16kHz) → 128-mel Spectrogram → Conv2d×3 (8× downsample)
             → Transformer Encoder → Linear Projector → Qwen3 Decoder → Text
```

| Component | 0.6B | 1.7B |
|-----------|------|------|
| Encoder layers | 18 | 24 |
| Encoder d_model | 896 | 1024 |
| Encoder heads | 14 | 16 |
| Encoder FFN dim | 3584 | 4096 |
| Decoder layers | 28 | 28 |
| Decoder hidden | 1024 | 2048 |
| Decoder heads (Q/KV) | 16/8 | 16/8 |

- **Audio Frontend**: WhisperFeatureExtractor compatible (128 mels, n_fft=400, hop=160)
- **Audio Encoder**: 3× Conv2d (stride 2, GELU) + sinusoidal position embeddings + Transformer with windowed block attention
- **Projector**: Linear(d_model → d_model, GELU) + Linear(d_model → decoder_hidden)
- **Text Decoder**: Qwen3 with GQA, Q/K RMSNorm, SwiGLU MLP, RoPE (theta=1M), tied embeddings

## Benchmarks vs Whisper-large-v3

Qwen3-ASR-1.7B outperforms Whisper-large-v3 on nearly every benchmark:

### Chinese Mandarin (CER ↓)

| Dataset | Whisper-large-v3 | Qwen3-ASR-0.6B | **Qwen3-ASR-1.7B** |
|---------|-----------------|----------------|---------------------|
| WenetSpeech (meeting) | 19.11 | 6.88 | **5.88** |
| AISHELL-2 | 5.06 | 3.15 | **2.71** |
| SpeechIO | 7.56 | 3.44 | **2.88** |

### English (WER ↓)

| Dataset | Whisper-large-v3 | Qwen3-ASR-0.6B | **Qwen3-ASR-1.7B** |
|---------|-----------------|----------------|---------------------|
| LibriSpeech (other) | 3.97 | 4.55 | **3.38** |
| GigaSpeech | 9.76 | 8.88 | **8.45** |
| CommonVoice-en | 9.90 | 9.92 | **7.39** |

### Multilingual (WER ↓, averaged)

| Dataset | Whisper-large-v3 | Qwen3-ASR-0.6B | **Qwen3-ASR-1.7B** |
|---------|-----------------|----------------|---------------------|
| MLS | 8.62 | 13.19 | **8.55** |
| CommonVoice | 10.77 | 12.75 | **9.18** |
| Fleurs | 5.27 | 7.57 | **4.90** |

## Supported Languages

Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian

Plus 22 Chinese dialects (Sichuan, Cantonese, Wu, Minnan, etc.)

## Project Structure

```
qwen3-asr-1.7b-mlx/
├── Cargo.toml
├── src/
│   ├── lib.rs         # Public API, model path resolution
│   ├── error.rs       # Error types
│   ├── audio.rs       # Mel spectrogram (128 mels, Slaney scale, Whisper normalization)
│   ├── encoder.rs     # Audio encoder (Conv2d + Transformer + windowed attention)
│   ├── qwen.rs        # Qwen3 text decoder (GQA, Q/K RMSNorm, SwiGLU, RoPE)
│   └── model.rs       # Combined model, prompt building, generation, weight loading
└── examples/
    └── transcribe.rs  # CLI transcription example
```

## API Reference

### `Qwen3ASR`

```rust
// Load model from directory
let mut model = Qwen3ASR::load("~/.OminiX/models/qwen3-asr-1.7b")?;

// Transcribe WAV file (default: Chinese)
let text = model.transcribe("audio.wav")?;

// Transcribe with language
let text = model.transcribe_with_language("audio.wav", "English")?;

// Transcribe raw 16kHz f32 samples
let text = model.transcribe_samples(&samples, "Japanese")?;

// Full control: custom sampling config + chunked processing
let config = SamplingConfig { temperature: 0.0, max_tokens: 8192 };
let text = model.transcribe_samples_with_config(&samples, "Chinese", &config)?;
let text = model.transcribe_samples_chunked(&samples, "Chinese", &config, 30.0)?;
```

### Model Path Resolution

1. Explicit path passed to `Qwen3ASR::load()`
2. `QWEN3_ASR_MODEL_PATH` environment variable
3. `~/.OminiX/models/qwen3-asr-1.7b` (default)

### Audio Input

- **WAV**: Native support (any sample rate, mono/stereo, 16/24/32-bit int or float)
- **M4A/MP3/FLAC/OGG/AAC**: Automatic conversion via ffmpeg (example only)
- **Raw samples**: `transcribe_samples()` accepts `&[f32]` at 16kHz mono

## API Server

Qwen3-ASR is available via the unified OminiX-API server:

```bash
# Start API server
cargo run --release -p ominix-api -- \
    --asr-model ~/.OminiX/models/qwen3-asr-1.7b --port 8080

# Transcribe (OpenAI Whisper-compatible multipart)
curl http://localhost:8080/v1/audio/transcriptions \
    -F file=@audio.wav -F language=Chinese

# Transcribe (JSON)
curl http://localhost:8080/v1/audio/transcriptions \
    -H "Content-Type: application/json" \
    -d '{"file_path": "audio.wav", "language": "English", "response_format": "verbose_json"}'
```

See the [OminiX-API](../ominix-api/) crate for full documentation.

## Building

Requires macOS with Apple Silicon (M1/M2/M3/M4).

```bash
# Build library
cargo build --release

# Build and run example
cargo build --release --example transcribe
```

### Dependencies

| Crate | Purpose |
|-------|---------|
| `mlx-rs` | MLX framework bindings (Metal + Accelerate) |
| `mlx-rs-core` | Shared LLM infrastructure (KV cache, RoPE, sampling) |
| `tokenizers` | HuggingFace tokenizer (BPE) |
| `rustfft` | FFT for mel spectrogram computation |
| `hound` | WAV file reading |
| `rubato` | High-quality audio resampling |

## Weight Format

Models use safetensors format with keys:
- `audio_tower.*` — Audio encoder (full precision)
- `model.*` — Text decoder (8-bit affine quantized, group_size=64)

The audio encoder is **not** quantized — only the text decoder uses 8-bit quantization. This preserves audio feature quality while reducing memory for the larger LLM component.

## License

Apache-2.0 (same as Qwen3-ASR models)

## Credits

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) by the Qwen team at Alibaba Cloud
- [mlx-community](https://huggingface.co/mlx-community) for quantized MLX model conversions
- [mlx-rs](https://github.com/oxideai/mlx-rs) for Rust MLX bindings
