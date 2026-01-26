# funasr-mlx

GPU-accelerated Chinese speech recognition on Apple Silicon using the FunASR Paraformer model.

## Features

- **18x+ real-time** transcription on Apple Silicon
- **Pure Rust** - no Python dependencies at runtime
- **Non-autoregressive** - predicts all tokens in parallel
- **GPU accelerated** - Metal GPU via MLX

## Quick Start

```rust
use funasr_mlx::{load_model, parse_cmvn_file, transcribe, Vocabulary};
use funasr_mlx::audio::{load_wav, resample};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load and resample audio to 16kHz
    let (samples, sample_rate) = load_wav("audio.wav")?;
    let samples = resample(&samples, sample_rate, 16000);

    // Load model with CMVN
    let mut model = load_model("paraformer.safetensors")?;
    let (addshift, rescale) = parse_cmvn_file("am.mvn")?;
    model.set_cmvn(addshift, rescale);

    // Load vocabulary and transcribe
    let vocab = Vocabulary::load("tokens.txt")?;
    let text = transcribe(&mut model, &samples, &vocab)?;

    println!("Transcription: {}", text);
    Ok(())
}
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
funasr-mlx = { path = "../funasr-mlx" }
```

Or from git:

```toml
[dependencies]
funasr-mlx = { git = "https://github.com/oxideai/mlx-rs" }
```

## Model Download

Download pre-converted MLX model from Hugging Face:

```bash
# Paraformer-large (Chinese ASR)
huggingface-cli download funaudiollm/paraformer-large-mlx --local-dir ./models/paraformer
```

Model directory structure:
```
models/paraformer/
├── paraformer.safetensors   # Model weights
├── am.mvn                   # CMVN normalization
└── tokens.txt               # Vocabulary (8404 tokens)
```

### Converting from FunASR (Alternative)

If you need to convert from the original FunASR checkpoint:

```python
# scripts/convert_paraformer.py
import torch
from safetensors.torch import save_file

# Load FunASR checkpoint
ckpt = torch.load("model.pb", map_location="cpu")

# Rename and save
weights = {}
for name, tensor in ckpt.items():
    # Apply weight name mapping...
    weights[new_name] = tensor

save_file(weights, "paraformer.safetensors")
```

## Examples

### Basic Transcription

```bash
cargo run --release --example transcribe -- audio.wav /path/to/model
```

### Benchmark

```bash
cargo run --release --example benchmark -- audio.wav /path/to/model 10
```

## Performance

Benchmarks on Apple M3 Max (48GB):

| Audio Duration | Inference Time | RTF | Speed |
|----------------|----------------|-----|-------|
| 3s | 50ms | 0.017 | 59x |
| 10s | 150ms | 0.015 | 67x |
| 30s | 400ms | 0.013 | 75x |

## Architecture

The Paraformer model consists of:

```
Audio (16kHz)
    ↓
[Mel Frontend] - 80 bins, 25ms window, 10ms hop, LFR 7/6
    ↓
[SAN-M Encoder] - 50 layers, 512 hidden, 4 heads
    ↓
[CIF Predictor] - Continuous Integrate-and-Fire
    ↓
[Bidirectional Decoder] - 16 layers, 512 hidden, 4 heads
    ↓
Tokens [8404 vocabulary]
```

## Requirements

- macOS 13.5+ (Ventura or later)
- Apple Silicon (M1/M2/M3/M4)
- Rust 1.82.0+

## License

MIT OR Apache-2.0
