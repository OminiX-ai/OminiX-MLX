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

Download models from Hugging Face:

```bash
# Fun-ASR-Nano (Chinese, English, Japanese)
huggingface-cli download FunAudioLLM/Fun-ASR-Nano-2512 --local-dir ./models/Fun-ASR-Nano-2512

# Fun-ASR-MLT-Nano (31 languages)
huggingface-cli download FunAudioLLM/Fun-ASR-MLT-Nano-2512 --local-dir ./models/Fun-ASR-MLT-Nano-2512
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

## References

- [Fun-ASR GitHub](https://github.com/FunAudioLLM/Fun-ASR)
- [Technical Report](https://arxiv.org/abs/2509.12508)
- [Model on HuggingFace](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)

## License

MIT
