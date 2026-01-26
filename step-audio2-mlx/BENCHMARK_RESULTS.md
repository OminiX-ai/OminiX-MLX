# Step-Audio 2 MLX - Benchmark Results

## Implementation Status

### Complete Implementation (All 4 Phases)

| Phase | Component | Status | Tests |
|-------|-----------|--------|-------|
| 1 | ASR Foundation | Complete | 14 passed |
| 2 | Think Mode | Complete | 8 passed |
| 3 | TTS Decoder | Complete | 26 passed |
| 4 | Integration & Tools | Complete | 18 passed |

**Total: 66/67 tests passing** (1 pre-existing adaptor convolution dimension issue)

### Code Statistics

| Component | Lines of Code |
|-----------|---------------|
| Core (audio, encoder, adaptor, llm) | ~2,500 |
| Think Mode | ~450 |
| TTS (S3Tokenizer, Flow, HiFi-GAN) | ~2,000 |
| Tools & Pipeline | ~1,500 |
| Examples | ~450 |
| **Total** | **~6,900 LOC** |

## Benchmark Comparison

### funasr-nano-mlx (SenseVoice + Qwen)
- Audio: 41.26s Chinese speech
- Mean latency: 1,631 ms
- **RTF: 0.0395x (25.3x real-time)**

### step-audio2-mlx Architecture
- Encoder: Whisper-style (32 layers, 1280 dim)
- Adaptor: Conv1d + Linear projection
- LLM: Qwen2.5-7B (28 layers, 3584 dim)
- TTS: S3Tokenizer → Flow Decoder → HiFi-GAN

### Model Comparison

| Model | Encoder | LLM | Parameters |
|-------|---------|-----|------------|
| step-audio2-mlx | Whisper (32L) | Qwen2.5-7B | ~8B |
| funasr-nano-mlx | SenseVoice | Qwen 0.5B | ~0.6B |
| funasr-mlx | Paraformer | - | ~220M |

### Expected Performance

Based on architecture complexity:

| Model | Expected RTF | Expected Speed |
|-------|-------------|----------------|
| step-audio2-mlx (8B) | ~0.15-0.3x | 3-7x real-time |
| funasr-nano-mlx (0.6B) | 0.04x | 25x real-time |
| funasr-mlx (220M) | 0.02x | 50x real-time |

> Note: step-audio2-mlx has ~13x more parameters than funasr-nano-mlx,
> so expect proportionally slower inference but higher quality output.

## Features

### ASR (Phase 1)
- 16kHz audio input → 128-mel spectrogram
- Whisper-style encoder with positional embeddings
- Adaptor with temporal downsampling (8x)
- Greedy decoding with temperature sampling

### Think Mode (Phase 2)
- `<think>...</think>` reasoning tags
- Separate token limits for thinking vs response
- Phase-aware temperature adjustment
- Streaming support for real-time thinking display

### TTS (Phase 3)
- Audio token extraction (range 151696-158256)
- S3Tokenizer: VQ codebook + PostNet refinement
- Flow decoder: Rectified flow with 10 steps
- HiFi-GAN: 256x upsampling vocoder
- Output: 24kHz audio

### Integration (Phase 4)
- Tool calling (Web search, Calculator)
- Multi-turn conversation context
- Unified pipeline API
- CLI examples (asr, think, conversation)

## Running Benchmarks

```bash
# Step-Audio 2 (requires model)
cd step-audio2-mlx
cargo run --release --example benchmark_asr -- ./audio.wav 10

# FunASR Nano (included model)
cd funasr-nano-mlx
cargo run --release --example fair_benchmark

# Full comparison
cd step-audio2-mlx
./scripts/benchmark_all_asr.sh ./audio.wav 10
```

## Model Requirements

To run step-audio2-mlx with actual inference:
1. Download Step-Audio-2-mini from HuggingFace
2. Place in `./Step-Audio-2-mini/`
3. Ensure `config.json` and `model.safetensors*` are present

```bash
# Example download (requires huggingface-cli)
huggingface-cli download stepfun-ai/Step-Audio-2-mini --local-dir ./Step-Audio-2-mini
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Step-Audio 2 MLX                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Audio Input (16kHz)                                                │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────┐                                                │
│  │ Mel Spectrogram │  128 bins, 400 FFT, 160 hop                    │
│  │   (audio.rs)    │                                                │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │    Encoder      │  Whisper-style, 32 layers, 1280 dim            │
│  │  (encoder.rs)   │  + sinusoidal positional embeddings            │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │    Adaptor      │  Conv1d (k=3, s=2) + 2x Linear                 │
│  │  (adaptor.rs)   │  1280 → 3584 dim, 8x temporal compression      │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │      LLM        │  Qwen2.5-7B, 28 layers, GQA                    │
│  │    (llm.rs)     │  vocab=158720, RoPE                            │
│  └────────┬────────┘                                                │
│           │                                                         │
│     ┌─────┴─────┐                                                   │
│     │           │                                                   │
│     ▼           ▼                                                   │
│  ┌──────┐   ┌──────────┐                                            │
│  │ Text │   │  Audio   │  (TTS feature)                             │
│  │Tokens│   │  Tokens  │                                            │
│  └──────┘   └────┬─────┘                                            │
│                  │                                                  │
│                  ▼                                                  │
│  ┌─────────────────┐                                                │
│  │  S3Tokenizer    │  Codebook (6561 entries) + PostNet             │
│  │(s3tokenizer.rs) │  codes → semantic features                     │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │  Flow Decoder   │  Rectified flow, 10 steps                      │
│  │   (flow.rs)     │  semantic → mel spectrogram                    │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │    HiFi-GAN     │  256x upsampling (8×8×2×2)                     │
│  │  (hifigan.rs)   │  mel → 24kHz waveform                          │
│  └─────────────────┘                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```
