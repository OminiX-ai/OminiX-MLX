# funasr-nano-mlx

Fun-ASR-Nano (800M) speech recognition on Apple Silicon using MLX.

## Architecture

Fun-ASR-Nano is an LLM-based ASR system combining:

```
Audio (16kHz)
    │
    ▼
┌─────────────────────┐
│   Mel Spectrogram   │  80 bins, 25ms window, 10ms hop
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Whisper Encoder   │  Frozen, extracts audio features
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Audio Adaptor     │  Linear projection to LLM dim
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│      Qwen LLM       │  Causal language model
└─────────┬───────────┘
          │
          ▼
      Text Output
```

## Features

- **800M parameters** - Balanced size/quality tradeoff
- **31 languages** (MLT variant) or Chinese/English/Japanese (base)
- **7 Chinese dialects** + 26 regional accents
- **Far-field recognition** - ~93% accuracy in noisy environments
- **Apple Silicon optimized** - Metal GPU acceleration via MLX

## Model Variants

| Model | Languages | Parameters |
|-------|-----------|------------|
| Fun-ASR-Nano-2512 | ZH, EN, JA | 800M |
| Fun-ASR-MLT-Nano-2512 | 31 languages | 800M |

## Model Download

Models download from HuggingFace (no authentication required).

### Download URLs

**MLX-converted models (required for this library):**

| Model | Precision | HuggingFace URL |
|-------|-----------|-----------------|
| Fun-ASR-Nano-2512 | fp16 | https://huggingface.co/mlx-community/Fun-ASR-Nano-2512-fp16 |
| Fun-ASR-MLT-Nano-2512 | fp16 | https://huggingface.co/mlx-community/Fun-ASR-MLT-Nano-2512-fp16 |


**Original PyTorch models (for reference):**

| Source | URL |
|--------|-----|
| **HuggingFace (Original)** | https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512 |
| **ModelScope (Original)** | https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512 |
| **Original FunASR** | https://github.com/modelscope/FunASR |

### Environment Variables

```bash
# Set custom model path
export FUNASR_NANO_MODEL_DIR=/path/to/Fun-ASR-Nano-2512

# Set language for SenseVoice (zh, en, ja, ko, auto)
export ASR_NANO_LANGUAGE=auto

# Or specify when running
FUNASR_NANO_MODEL_DIR=./models/Fun-ASR-Nano-2512 cargo run --example transcribe --release
```

### Manual Download

```bash
# Fun-ASR-Nano MLX (fp16)
huggingface-cli download mlx-community/Fun-ASR-Nano-2512-fp16 --local-dir ./models/Fun-ASR-Nano-2512

# Fun-ASR-MLT-Nano MLX (31 languages, fp16)
huggingface-cli download mlx-community/Fun-ASR-MLT-Nano-2512-fp16 --local-dir ./models/Fun-ASR-MLT-Nano-2512

# Using git lfs
git lfs install
git clone https://huggingface.co/mlx-community/Fun-ASR-Nano-2512-fp16 ./models/Fun-ASR-Nano-2512
```

### Model Directory Structure

```
models/Fun-ASR-Nano-2512/
├── model.safetensors            # MLX weights (required)
├── config.json                  # Model configuration
├── tokenizer.json               # Tokenizer
├── vocab.json                   # Vocabulary
├── merges.txt                   # BPE merges
└── tokenizer_config.json        # Tokenizer settings
```

## CLI Usage

```bash
# Basic transcription
cargo run --release --example transcribe -- ./models/Fun-ASR-Nano-2512 ./audio.wav

# Benchmark
cargo run --release --example benchmark -- ./models/Fun-ASR-Nano-2512 ./audio.wav
```

## Usage

```rust
use funasr_nano_mlx::{FunASRNano, load_model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model
    let mut model = load_model("path/to/Fun-ASR-Nano-2512")?;

    // Transcribe audio
    let text = model.transcribe("audio.wav")?;
    println!("{}", text);

    Ok(())
}
```

## Project Structure

```
funasr-nano-mlx/
├── src/
│   ├── lib.rs              # Public API
│   ├── audio.rs            # Audio loading & mel spectrogram
│   ├── whisper_encoder.rs  # Whisper-based audio encoder
│   ├── adaptor.rs          # Audio-to-LLM adaptor
│   ├── qwen.rs             # Qwen LLM (from qwen3-mlx)
│   ├── model.rs            # Combined FunASRNano model
│   └── error.rs            # Error types
├── examples/
│   ├── transcribe.rs       # Basic transcription
│   └── benchmark.rs        # Performance benchmarking
└── Cargo.toml
```

## Performance (Expected)

On Apple M3 Max:

| Metric | Value |
|--------|-------|
| Prompt processing | ~100-150 tok/s |
| Decode | ~30-50 tok/s |
| Memory (4-bit) | ~2-3 GB |
| Real-time factor | < 0.1 |

## Troubleshooting

If transcription produces garbage output, see [MLX Inference Fixes](docs/MLX_INFERENCE_FIXES.md) for common issues:

- Audio preprocessing mismatch (most common)
- Float16 precision drift in deep encoders
- Model path case sensitivity

## References

- [Fun-ASR GitHub](https://github.com/FunAudioLLM/Fun-ASR)
- [Technical Report](https://arxiv.org/abs/2509.12508)
- [Model on HuggingFace](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)

## License

MIT
