<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/3cc8697b-08f0-4180-91f7-8aca0ffdbdde">
  <img width="500" alt="OminiX-MLX" src="https://github.com/user-attachments/assets/b168cf1c-8e2f-4969-bffa-b57ee33950c0" />
</picture>

  
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
│  LLM / VLM    │         │   Audio Crates  │         │  Image Crates   │
├───────────────┤         ├─────────────────┤         ├─────────────────┤
│ qwen3-mlx     │         │ funasr-mlx      │         │ flux-klein-mlx  │
│ glm4-mlx      │         │ funasr-nano-mlx │         │ zimage-mlx      │
│ glm4-moe-mlx  │         │ gpt-sovits-mlx  │         │ qwen-image-mlx  │
│ mixtral-mlx   │         │                 │         │                 │
│ mistral-mlx   │         │                 │         │                 │
│ moxin-vlm-mlx │         │                 │         │                 │
│ minicpm-sala   │         │                 │         │                 │
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
├── moxin-vlm-mlx/       # Moxin-7B VLM (vision-language)
├── MiniCPM-SALA-MLX/    # MiniCPM-SALA 9B (hybrid attention, 1M context)
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
| MiniCPM-SALA | `minicpm-sala-mlx` | 9B | Hybrid attention (sparse + lightning), 1M context, OpenAI-compatible API |

### Vision-Language Models (VLMs)

| Model | Crate | Sizes | Notes |
|-------|-------|-------|-------|
| Moxin-7B VLM | `moxin-vlm-mlx` | 7B | DINOv2 + SigLIP + Mistral-7B, 8-bit quantization, 30 tok/s |

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

### Vision-Language Model

```bash
# Download 8-bit quantized Moxin-7B VLM
huggingface-cli download moxin-org/Moxin-7B-VLM-8bit-mlx --local-dir ./models/Moxin-7B-VLM-8bit-mlx

# Run VLM inference with 8-bit quantization
cargo run --release -p moxin-vlm-mlx --example generate -- \
    --model ./models/Moxin-7B-VLM-hf \
    --image ./photo.jpg \
    --prompt "What is in this image?" \
    --quantize 8
```

```rust
use moxin_vlm_mlx::{load_model, load_tokenizer, normalize_dino, normalize_siglip, Generate, KVCache};

let mut vlm = load_model("./models/Moxin-7B-VLM-hf")?;
let vlm = vlm.quantize(64, 8)?; // 8-bit quantization
let tokenizer = load_tokenizer("./models/Moxin-7B-VLM-hf")?;

// Preprocess image to [1, 224, 224, 3] and normalize
let dino_img = normalize_dino(&tensor)?;
let siglip_img = normalize_siglip(&tensor)?;

// Generate
let mut cache: Vec<KVCache> = Vec::new();
let generator = Generate::new(&mut vlm, &mut cache, 0.0, dino_img, siglip_img, input_ids);

for token in generator.take(256) {
    let token = token?;
    print!("{}", tokenizer.decode(&[token.item::<u32>()], true)?);
}
```

### MiniCPM-SALA (Long-Context LLM)

```bash
# Download 8-bit quantized model
huggingface-cli download moxin-org/MiniCPM4-SALA-9B-8bit-mlx --local-dir ./models/MiniCPM-SALA-8bit

# Run text generation
cargo run --release -p minicpm-sala-mlx --example generate -- \
    ./models/MiniCPM-SALA-8bit "Explain the theory of relativity."

# Run interactive chat
cargo run --release -p minicpm-sala-mlx --example chat -- \
    ./models/MiniCPM-SALA-8bit --no-think

# Start OpenAI-compatible API server
cargo run --release -p minicpm-sala-mlx --example server -- \
    --model ./models/MiniCPM-SALA-8bit --port 8080 --no-think
```

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | OpenAI-compatible chat completion |
| GET | `/v1/models` | List available models |
| GET | `/health` | Health check |

**Example API call:**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minicpm-sala-9b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
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
| LLM | MiniCPM-SALA-9B-8bit | 28 tok/s | 9.6GB |
| VLM | Moxin-7B-8bit | 30 tok/s | 10GB |
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
| moxin-vlm-mlx | [README](moxin-vlm-mlx/README.md) | Moxin-7B VLM |
| minicpm-sala-mlx | [README](MiniCPM-SALA-MLX/README.md) | MiniCPM-SALA 9B (hybrid attention, 1M context) |
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
