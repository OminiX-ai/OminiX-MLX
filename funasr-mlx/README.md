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

**Important:** The original FunASR model uses PyTorch format. You must convert it to MLX-compatible safetensors format before use.

### Step 1: Download Original Model

The Paraformer-large model is available from ModelScope:

```bash
git lfs install
git clone https://modelscope.cn/models/damo/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git ./paraformer-src
```

### Step 2: Convert to MLX Format

The converter is **pure Rust** - no Python or libtorch required:

```bash
cargo run --release --features convert --example convert_model -- ./paraformer-src ./models/paraformer
```

This will:
- Load the PyTorch model using candle-core
- Convert 956 tensors to MLX-compatible format
- Save as safetensors (smaller and faster to load)
- Copy auxiliary files (am.mvn, tokens.txt)

### Environment Variables

```bash
# Set custom model path
export FUNASR_MODEL_DIR=/path/to/paraformer

# Or specify when running
FUNASR_MODEL_DIR=./models/paraformer cargo run --example transcribe --release
```

### Model Directory Structure

```
models/paraformer/
├── paraformer.safetensors   # Model weights (converted)
├── am.mvn                   # CMVN normalization
└── tokens.txt               # Vocabulary (8404 tokens)
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
