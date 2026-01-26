# Step-Audio 2 MLX

Step-Audio 2 mini implementation in MLX (Rust) for Apple Silicon.

## Overview

Step-Audio 2 is an end-to-end multimodal large language model for bidirectional audio understanding and generation. This implementation targets the **mini** variant (8B parameters) and leverages existing OminiX-MLX components.

## Features

| Feature | Status | Phase |
|---------|--------|-------|
| ASR (Speech → Text) | Planned | 1 |
| Think Mode | Planned | 2 |
| TTS (Text → Speech) | Planned | 3 |
| S2TT (Speech Translation) | Planned | 3 |
| S2ST (Speech-to-Speech) | Planned | 3 |
| Voice Cloning | Planned | 3 |
| Tool Calling (Web Search) | Planned | 4 |

## Architecture

```
Audio (16kHz)
    → Mel Spectrogram (128 bins)
    → Whisper-style Encoder (32 layers, 1280 dim)
    → Adaptor (Conv1d + Linear → 3584 dim)
    → Qwen2.5-7B LLM (28 layers)
    → Text tokens + Audio tokens
    → [TTS: S3Tokenizer → Flow Decoder → HiFi-GAN]
    → Audio output (24kHz)
```

## Model Variants

| Variant | Size | HuggingFace |
|---------|------|-------------|
| mini-Base | 8B | [stepfun-ai/Step-Audio-2-mini-Base](https://huggingface.co/stepfun-ai/Step-Audio-2-mini-Base) |
| mini | 8B | [stepfun-ai/Step-Audio-2-mini](https://huggingface.co/stepfun-ai/Step-Audio-2-mini) |
| mini-Think | 8B | [stepfun-ai/Step-Audio-2-mini-Think](https://huggingface.co/stepfun-ai/Step-Audio-2-mini-Think) |

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 13.0+
- 32GB+ RAM recommended (16GB minimum with quantization)
- Rust 1.70+

## Installation

```bash
# Clone the repository
git clone https://github.com/anthropics/OminiX-MLX
cd OminiX-MLX/step-audio2-mlx

# Build
cargo build --release

# With TTS support
cargo build --release --features tts
```

## Usage

### ASR (Speech to Text)

```rust
use step_audio2_mlx::StepAudio2;

let mut model = StepAudio2::load("path/to/Step-Audio-2-mini")?;
let text = model.transcribe("audio.wav")?;
println!("Transcription: {}", text);
```

### TTS (Text to Speech)

```rust
use step_audio2_mlx::StepAudio2;

let mut model = StepAudio2::load("path/to/Step-Audio-2-mini")?;
let audio = model.synthesize("Hello, world!")?;
model.save_audio(&audio, "output.wav")?;
```

### Think Mode

```rust
use step_audio2_mlx::{StepAudio2, ThinkConfig};

let mut model = StepAudio2::load("path/to/Step-Audio-2-mini-Think")?;
let output = model.think_and_respond("audio.wav", ThinkConfig::default())?;
println!("Thinking: {:?}", output.thinking);
println!("Response: {}", output.response_text);
```

## Memory Requirements

| Precision | Model Size | Runtime Total |
|-----------|------------|---------------|
| BF16 | 16.5 GB | ~22 GB |
| INT8 | 9 GB | ~14 GB |
| INT4 | 5.25 GB | ~10 GB |

## Documentation

- [Architecture Details](docs/architecture.md)
- [Development Plan](docs/dev-plan.md)

## Leveraged Components

This implementation reuses code from:

| Component | Source | Reuse |
|-----------|--------|-------|
| KVCache, Sampler | mlx-rs-core | 100% |
| Qwen2 LLM | qwen3-mlx | 85% |
| Whisper Encoder | funasr-nano-mlx | 80% |
| HiFi-GAN | gpt-sovits-mlx | 70% |
| Flow Sampler | flux-klein-mlx | 80% |

## License

Apache 2.0 (same as Step-Audio 2)

## References

- [Step-Audio 2 Paper](https://arxiv.org/abs/2507.16632)
- [Step-Audio 2 GitHub](https://github.com/stepfun-ai/Step-Audio2)
- [Step-Audio 2 HuggingFace](https://huggingface.co/stepfun-ai)
