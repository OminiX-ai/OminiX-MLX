<div align="center">
<h1><b>OminiX-MLX</b></h1>

High-performance ML inference on Apple Silicon: LLMs, ASR, TTS, and Image Generation in pure Rust.

[![Rust Version](https://img.shields.io/badge/Rust-1.82.0+-blue)](https://releases.rs/docs/1.82.0)
![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)

</div>

---

## Overview

OminiX-MLX is a comprehensive Rust ecosystem for running machine learning models on Apple Silicon using [MLX](https://github.com/ml-explore/mlx). It provides:

- **mlx-rs**: Safe Rust bindings to Apple's MLX framework
- **mlx-rs-core**: Shared inference infrastructure (KV cache, RoPE, attention)
- **Model Crates**: Dedicated crates for each model family

Built for production use with zero Python dependencies at inference time.

## Features

| Feature | Description |
|---------|-------------|
| **GPU Acceleration** | Metal-optimized inference on M1/M2/M3/M4 chips |
| **Unified Memory** | Zero-copy data sharing between CPU and GPU |
| **Lazy Evaluation** | Automatic kernel fusion and memory optimization |
| **Pure Rust** | No Python runtime required for inference |
| **Modular Design** | Use only what you need |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Application                                │
│                    (OminiX-API / Custom Rust Application)                   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   LLM Crates  │         │   Audio Crates  │         │  Image Crates   │
├───────────────┤         ├─────────────────┤         ├─────────────────┤
│ qwen3-mlx     │         │ funasr-mlx      │         │ flux-klein-mlx  │
│ glm4-mlx      │         │ funasr-nano-mlx │         │ zimage-mlx      │
│ glm4-moe-mlx  │         │ gpt-sovits-mlx  │         │ qwen-image-mlx  │
│ mixtral-mlx   │         │                 │         │                 │
│ mistral-mlx   │         │                 │         │                 │
└───────┬───────┘         └────────┬────────┘         └────────┬────────┘
        │                          │                           │
        └──────────────────────────┼───────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │       mlx-rs-core        │
                    ├──────────────────────────┤
                    │ • KV Cache Management    │
                    │ • RoPE Embeddings        │
                    │ • Attention (SDPA)       │
                    │ • Audio Processing       │
                    │ • Metal Kernels          │
                    │ • Speculative Decoding   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │         mlx-rs           │
                    ├──────────────────────────┤
                    │ • Safe Rust API          │
                    │ • Array Operations       │
                    │ • Neural Network Layers  │
                    │ • Transforms (eval, jit) │
                    │ • Random/Ops/Indexing    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │         mlx-sys          │
                    ├──────────────────────────┤
                    │ • FFI Bindings (bindgen) │
                    │ • mlx-c Submodule        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      Apple MLX (C++)     │
                    ├──────────────────────────┤
                    │ • Metal GPU Backend      │
                    │ • Accelerate Framework   │
                    │ • Unified Memory         │
                    │ • Lazy Evaluation        │
                    └──────────────────────────┘
```

### Data Flow

```
                         ┌─────────────────────────────────────┐
                         │            Input Data               │
                         │  (Text / Audio / Image)             │
                         └──────────────┬──────────────────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         │                              │                              │
         ▼                              ▼                              ▼
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│    Tokenizer    │          │  Audio Frontend │          │   VAE Encoder   │
│  (tokenizers)   │          │  (Mel/STFT)     │          │  (img→latent)   │
└────────┬────────┘          └────────┬────────┘          └────────┬────────┘
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│   Transformer   │          │   Encoder       │          │   Transformer   │
│   (Attention +  │          │   (SAN-M /      │          │   (DiT /        │
│    MLP layers)  │          │    Whisper)     │          │    MMDiT)       │
└────────┬────────┘          └────────┬────────┘          └────────┬────────┘
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│   LM Head       │          │   CTC / CIF     │          │   VAE Decoder   │
│  (logits→token) │          │   Decoder       │          │  (latent→img)   │
└────────┬────────┘          └────────┬────────┘          └────────┬────────┘
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│  Detokenizer    │          │  Vocabulary     │          │   PNG/JPEG      │
│  (token→text)   │          │  (idx→text)     │          │   Encoder       │
└────────┬────────┘          └────────┬────────┘          └────────┬────────┘
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Output Data                                     │
│                       (Generated Text / Transcript / Image)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Crate Structure

```
OminiX-MLX/
├── mlx-rs/              # Core MLX Rust bindings
├── mlx-rs-core/         # Shared inference infrastructure
│
├── qwen3-mlx/           # Qwen2, Qwen3, Qwen3-MoE
├── glm4-mlx/            # GLM4
├── glm4-moe-mlx/        # GLM4-MoE (45 experts)
├── mixtral-mlx/         # Mixtral 8x7B/8x22B
├── mistral-mlx/         # Mistral 7B
│
├── gpt-sovits-mlx/      # GPT-SoVITS voice cloning
├── funasr-mlx/          # FunASR Paraformer ASR
├── funasr-nano-mlx/     # FunASR-Nano (SenseVoice + Qwen)
│
├── flux-klein-mlx/      # FLUX.2-klein image generation
├── zimage-mlx/          # Z-Image generation
└── qwen-image-mlx/      # Qwen image generation
```

## Supported Models

### Language Models (LLMs)

| Model | Crate | Sizes | Notes |
|-------|-------|-------|-------|
| Qwen2 | `qwen3-mlx` | 0.5B - 72B | Full range supported |
| Qwen3 | `qwen3-mlx` | 0.6B - 235B | Including MoE variants |
| GLM-4 | `glm4-mlx` | 9B | Chat and base models |
| GLM-4-MoE | `glm4-moe-mlx` | 9B | 45 expert MoE |
| Mixtral | `mixtral-mlx` | 8x7B, 8x22B | MoE architecture |
| Mistral | `mistral-mlx` | 7B | Sliding window attention |

### Speech Recognition (ASR)

| Model | Crate | Languages | Performance |
|-------|-------|-----------|-------------|
| Paraformer-large | `funasr-mlx` | Chinese, English | 18x real-time |
| FunASR-Nano | `funasr-nano-mlx` | Chinese, English | SenseVoice + Qwen |

### Text-to-Speech (TTS)

| Model | Crate | Features | Performance |
|-------|-------|----------|-------------|
| GPT-SoVITS | `gpt-sovits-mlx` | Few-shot voice cloning | 4x real-time |

### Image Generation

| Model | Crate | Notes |
|-------|-------|-------|
| FLUX.2-klein | `flux-klein-mlx` | Qwen3 text encoder |
| Z-Image | `zimage-mlx` | Fast generation |

## Quick Start

### Prerequisites

- macOS 14.0+ (Sonoma)
- Apple Silicon (M1/M2/M3/M4)
- Rust 1.82+
- Xcode Command Line Tools

### Build

```bash
# Clone repository
git clone https://github.com/anthropics/OminiX-MLX.git
cd OminiX-MLX

# Build all crates
cargo build --release

# Build specific crate
cargo build --release -p qwen3-mlx
```

### LLM Generation

```bash
# Download model
huggingface-cli download mlx-community/Qwen3-4B-bf16 --local-dir ./models/Qwen3-4B

# Run text generation
cargo run --release -p qwen3-mlx --example generate_qwen3 -- ./models/Qwen3-4B "Hello, how are you?"

# Run interactive chat
cargo run --release -p qwen3-mlx --example chat_qwen3 -- ./models/Qwen3-4B
```

```rust
use qwen3_mlx::{load_model, Generate, ConcatKeyValueCache};

let mut model = load_model("./models/Qwen3-4B")?;
let mut cache = Vec::new();

let generator = Generate::<ConcatKeyValueCache>::new(
    &mut model, &mut cache, 0.7, &prompt_tokens
);

for token in generator.take(100) {
    let token = token?;
    print!("{}", tokenizer.decode(&[token.item::<u32>()], true)?);
}
```

### Speech Recognition

```bash
cd funasr-mlx

# Run transcription
cargo run --release --example transcribe -- \
    --model ./models/paraformer \
    --audio ./audio/test.wav
```

```rust
use funasr_mlx::{load_model, transcribe, Vocabulary};
use funasr_mlx::audio::{load_wav, resample};

// Load audio
let (samples, rate) = load_wav("audio.wav")?;
let samples = resample(&samples, rate, 16000);

// Load model and transcribe
let mut model = load_model("paraformer.safetensors")?;
let vocab = Vocabulary::load("tokens.txt")?;
let text = transcribe(&mut model, &samples, &vocab)?;
```

### Voice Cloning

```bash
cd gpt-sovits-mlx/rust

cargo run --release --example voice_clone -- \
    --reference ./audio/reference.wav \
    --text "Hello, this is a voice clone."
```

```rust
use gpt_sovits_mlx::VoiceCloner;

let mut cloner = VoiceCloner::with_defaults()?;
cloner.set_reference_audio("reference.wav")?;

let audio = cloner.synthesize("Hello, world!")?;
cloner.save_wav(&audio, "output.wav")?;
```

### Image Generation

```bash
# Download Z-Image model
huggingface-cli download uqer1244/MLX-z-image --local-dir ./models/zimage-turbo-mlx

# Generate image with Z-Image
cargo run --release -p zimage-mlx --example generate_zimage -- "a cat sitting on a couch"

# Download FLUX.2-klein model
huggingface-cli download black-forest-labs/FLUX.2-klein-4B --local-dir ./models/flux-klein

# Generate image with FLUX.2-klein
cargo run --release -p flux-klein-mlx --example generate_klein -- "a beautiful sunset over mountains"
```

## Performance

Benchmarks on Apple M3 Max (128GB):

| Task | Model | Performance | Memory |
|------|-------|-------------|--------|
| LLM | Qwen3-4B | 45 tok/s | 8GB |
| LLM | GLM4-9B-4bit | 35 tok/s | 6GB |
| LLM | Mixtral-8x7B-4bit | 25 tok/s | 26GB |
| ASR | Paraformer | 18x real-time | 500MB |
| TTS | GPT-SoVITS | 4x real-time | 2GB |
| Image | Z-Image | ~3s/image | 8GB |
| Image | FLUX.2-klein | ~5s/image | 13GB |

## Documentation

| Crate | README | Description |
|-------|--------|-------------|
| mlx-rs-core | [README](mlx-rs-core/README.md) | Shared infrastructure |
| qwen3-mlx | [README](qwen3-mlx/README.md) | Qwen model family |
| glm4-mlx | [README](glm4-mlx/README.md) | GLM4 models |
| glm4-moe-mlx | [README](glm4-moe-mlx/README.md) | GLM4-MoE |
| mixtral-mlx | [README](mixtral-mlx/README.md) | Mixtral MoE |
| mistral-mlx | [README](mistral-mlx/README.md) | Mistral 7B |
| funasr-mlx | [README](funasr-mlx/README.md) | Paraformer ASR |
| funasr-nano-mlx | [README](funasr-nano-mlx/README.md) | FunASR-Nano |
| gpt-sovits-mlx | [README](gpt-sovits-mlx/README.md) | Voice cloning |

## Feature Flags

| Flag | Description | Default |
|------|-------------|---------|
| `metal` | Enable Metal GPU acceleration | On |
| `accelerate` | Use Accelerate framework | On |

## License

Dual-licensed under MIT and Apache 2.0.

## Acknowledgments

- [Apple MLX Team](https://github.com/ml-explore/mlx) for the MLX framework
- [oxideai](https://github.com/oxideai) for the original mlx-rs bindings
