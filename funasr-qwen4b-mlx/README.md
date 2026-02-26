# funasr-qwen4b-mlx

LLM-based automatic speech recognition using a SenseVoice encoder + Qwen3-4B language model, running on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

## Architecture

```
Audio (16kHz) → 80-mel Spectrogram → SenseVoice Encoder (70 layers, 512-dim)
                                           │ 4-layer Transformer Adaptor (512 → 2560)
                                           │ 2560-dim visual tokens
                                           │
                              Qwen3-4B LLM (36 layers, 2560-dim)
                                           │
                                        Transcript
```

| Component | Details |
|-----------|---------|
| Audio frontend | 80-mel spectrogram, FunASR-compatible preprocessing |
| SenseVoice encoder | 70 transformer layers, 512-dim output |
| Adaptor | 4-layer transformer (512 → 2560) |
| LLM decoder | Qwen3-4B (36 layers, GQA, SwiGLU, RoPE) |
| Punctuation | Optional CT-Transformer via ONNX (`funasr-mlx` crate) |

## Quick Start

### 1. Download models

```bash
# SenseVoice encoder + adaptor weights
huggingface-cli download <model-repo> \
    --local-dir ~/.OminiX/models/funasr-qwen4b

# Qwen3-4B LLM
huggingface-cli download Qwen/Qwen3-4B \
    --local-dir ~/.OminiX/models/qwen3-4b
```

### 2. Transcribe

```bash
cargo run --release --example transcribe -- audio.wav

# Disable punctuation restoration
cargo run --release --example transcribe --no-default-features -- audio.wav

# Translate to English
cargo run --release --example transcribe_translate -- audio.wav
```

### 3. As a library

```rust
use funasr_qwen4b_mlx::FunASRQwen4B;

let model = FunASRQwen4B::load(
    "~/.OminiX/models/funasr-qwen4b",
    "~/.OminiX/models/qwen3-4b",
)?;

let text = model.transcribe("audio.wav")?;
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `punctuation` | yes | CT-Transformer punctuation restoration via `funasr-mlx` |

## Examples

| Example | Description |
|---------|-------------|
| `transcribe` | Transcribe a WAV file |
| `transcribe_translate` | Transcribe and translate to English |

## Project Structure

```
funasr-qwen4b-mlx/
├── Cargo.toml
└── src/
    ├── lib.rs                  # Public API
    ├── audio.rs                # Mel spectrogram, resampling
    ├── sensevoice_encoder.rs   # SenseVoice transformer encoder
    ├── adaptor.rs              # 4-layer adaptor (512 → 2560)
    ├── model.rs                # Combined model, weight loading
    ├── qwen4b.rs               # Qwen3-4B integration
    └── error.rs                # Error types
```

## License

MIT OR Apache-2.0
