# MiniCPM-SALA MLX Port

Port of [MiniCPM-SALA](https://huggingface.co/openbmb/MiniCPM-SALA) to Apple MLX framework.

## Overview

MiniCPM-SALA is a 9B parameter hybrid attention model that achieves **million-token context** on consumer GPUs by combining:

- **25% Sparse Attention (InfLLM-v2)** — High-fidelity local details
- **75% Linear Attention (Lightning Attention)** — Global efficiency

### Key Features

| Feature | Value |
|---------|-------|
| Parameters | 9B |
| Max Context | 1M+ tokens |
| Inference Speed | 3.5× faster than Qwen3-8B at 256K |
| Memory Efficiency | Runs on RTX 5090 / A6000D |
| License | Apache-2.0 |

## Status

| Component | Status |
|-----------|--------|
| Weight Loading (fp32 + quantized) | Complete |
| Lightning Attention (GLA) | Complete |
| Sparse Attention (InfLLMv2) | Complete |
| Custom Metal Kernels | Complete |
| Self-Speculative Decoding | Complete |
| Batched Inference | Complete |
| Multi-turn Chat | Complete |
| OpenAI-compatible API Server | Complete |
| Model Management (download/delete) | Complete |
| 8-bit Quantization | Complete |

## Quick Start

### Prerequisites

- macOS 14.0+ (Sonoma), Apple Silicon (M1/M2/M3/M4)
- Rust 1.82+, Xcode Command Line Tools

### Download Model

```bash
# 8-bit quantized (recommended — 9.6 GB)
huggingface-cli download moxin-org/MiniCPM4-SALA-9B-8bit-mlx --local-dir ./models/MiniCPM-SALA-8bit
```

### Examples

```bash
# Text generation
cargo run --release -p minicpm-sala-mlx --example generate -- \
    ./models/MiniCPM-SALA-8bit "Explain quantum entanglement in simple terms."

# Interactive multi-turn chat (strips <think> blocks)
cargo run --release -p minicpm-sala-mlx --example chat -- \
    ./models/MiniCPM-SALA-8bit --no-think

# Batched inference (multiple prompts in parallel)
cargo run --release -p minicpm-sala-mlx --example batch_generate -- \
    ./models/MiniCPM-SALA-8bit

# Self-speculative decoding (draft from first 8 layers)
cargo run --release -p minicpm-sala-mlx --example speculative_generate -- \
    ./models/MiniCPM-SALA-8bit

# Needle-in-a-haystack long context test
cargo run --release -p minicpm-sala-mlx --example needle_test -- \
    ./models/MiniCPM-SALA-8bit --context-len 32000 --depth 0.5

# Save quantized weights (fp16 → 4-bit or 8-bit)
cargo run --release -p minicpm-sala-mlx --example save_quantized -- \
    ./models/MiniCPM-SALA --bits 8 --output ./models/MiniCPM-SALA-8bit
```

### OpenAI-Compatible API Server

```bash
# Start server
cargo run --release -p minicpm-sala-mlx --example server -- \
    --model ./models/MiniCPM-SALA-8bit --port 8080 --no-think
```

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (OpenAI-compatible) |
| `GET`  | `/v1/models` | List models with metadata (path, size, quantization, loaded status) |
| `POST` | `/v1/models/download` | Download a model from HuggingFace |
| `DELETE` | `/v1/models/{id}` | Delete a downloaded model |
| `GET`  | `/health` | Health check |

**Chat completion request:**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minicpm-sala-9b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

**Response:**

```json
{
  "id": "chatcmpl-189409a7a2804800",
  "object": "chat.completion",
  "model": "minicpm-sala-9b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "The capital of France is Paris."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 32, "completion_tokens": 90, "total_tokens": 122}
}
```

**Model management:**

```bash
# List models (includes path, size, quantization, loaded status)
curl http://localhost:8080/v1/models

# Download a model from HuggingFace
curl -X POST http://localhost:8080/v1/models/download \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "moxin-org/MiniCPM4-SALA-9B-8bit-mlx"}'

# Delete a model
curl -X DELETE http://localhost:8080/v1/models/MiniCPM4-SALA-9B-8bit-mlx
```

Model metadata and download state are persisted in `~/.ominix/config.json`. The server scans `--models-dir` (default `~/.ominix/models`) for model subdirectories on each list request. The currently loaded model is always included and marked with `"loaded": true`. Set `HF_TOKEN` env var or have `~/.cache/huggingface/token` for private repos.

**Server options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Path to model directory |
| `--port` | 8080 | Server port |
| `--temperature` | 0.7 | Default sampling temperature |
| `--max-tokens` | 2048 | Default max generation tokens |
| `--no-think` | false | Strip `<think>...</think>` from responses |
| `--models-dir` | `~/.ominix/models` | Directory for managed models |

## Performance (Apple M3 Max, 128 GB)

### Throughput

| Variant | Size | Prefill (tok/s) | Decode (tok/s) |
|---------|------|-----------------|----------------|
| fp16    | 18 GB | 0.4 – 313.9    | 3.5 – 3.6     |
| 8-bit   | 9.6 GB | 4.7 – 442.6   | 27.3 – 28.1   |
| 4-bit   | 5.4 GB | 2.2 – 260.3   | 34.4 – 35.6   |

Prefill speed scales with prompt length (low end = 2 tokens, high end = ~900 tokens). Decode speed is steady-state autoregressive generation.

### Speed vs Qwen3-8B (both 8-bit, Apple M3 Max)

MiniCPM-SALA (Rust/mlx-rs) vs Qwen3-8B (Python/mlx-lm):

| Context | SALA Prefill | Qwen3 Prefill | SALA Decode | Qwen3 Decode |
|---------|-------------|---------------|-------------|--------------|
| 1K   | — | 486 tok/s | 28 tok/s | 36 tok/s |
| 4K   | 309 tok/s | 488 tok/s | 26 tok/s | 35 tok/s |
| 8K   | 325 tok/s | 493 tok/s | 25 tok/s | 33 tok/s |
| 16K  | 325 tok/s | 417 tok/s | 23 tok/s | 25 tok/s |
| 32K  | 350 tok/s | 333 tok/s | 23 tok/s | 18 tok/s |
| 64K  | 220 tok/s | OOM/untested | 19 tok/s | — |
| 128K | 192 tok/s | OOM/untested | 9 tok/s | — |

**Analysis:**
- At short contexts (< 16K), Qwen3-8B is faster — dense GQA attention is well-optimized in mlx-lm
- At 32K, **SALA overtakes Qwen3** in both prefill (350 vs 333 tok/s) and decode (23 vs 18 tok/s)
- Beyond 32K, Qwen3's dense KV cache grows linearly and becomes impractical; **SALA continues to 128K+** thanks to lightning attention's O(1) state per layer
- SALA's decode advantage grows with context length because 75% of layers use fixed-size recurrent state instead of scanning a growing KV cache
- Note: SALA is Rust (mlx-rs), Qwen3 is Python (mlx-lm) — runtimes differ, but both use the same Metal backend

### Quality (8-bit, temp=0)

| Category | Questions | Answered | Correct |
|----------|-----------|----------|---------|
| Math / Arithmetic | 4 | 4 | 4 (100%) |
| Multi-step Reasoning | 3 | 3 | 2 (67%) |
| Constraint Satisfaction | 1 | 1 | 1 (100%) |
| Think-loop Failures | 3 | 0 (stuck) | — |
| **Total** | **11** | **8** | **7 (87.5%)** |

### Needle-in-a-Haystack (8-bit, greedy)

Tests whether the model can retrieve a specific fact ("The secret verification code for Project Aurora is 739258") buried at various depths in long filler text.

| Context | Depth | Found? | Prefill (tok/s) | Decode (tok/s) | Prefill Time |
|---------|-------|--------|-----------------|----------------|-------------|
| 4K | 50% | YES | 309 | 26.0 | 13s |
| 8K | 25% | YES | 325 | 24.9 | 25s |
| 16K | 25% | YES | 325 | 23.3 | 49s |
| 32K | 3% | YES | 180 | — | 179s |
| 32K | 25% | NO | 389 | 20.1 | 83s |
| 32K | 50% | NO | 316 | 22.2 | 102s |
| 32K | 95% | YES | 350 | 22.8 | 92s |
| 64K | 50% | weak | 181 | 6.5 | 356s |
| 64K | 95% | YES | 220 | 18.8 | 293s |
| 128K | 95% | YES | 192 | 9.0 | 671s |
| 256K | 95% | NO | 276 | 6.7 | 934s |

**Key findings:**
- Reliable retrieval within the **sliding window** (last ~2048 tokens) and **init/dense region** (first ~8K tokens)
- Middle-region retrieval depends on InfLLMv2 top-K block selection (`topk=64, block_size=64`) — can miss individual facts in repetitive filler
- Decode speed degrades with context length: 26 tok/s at 4K → 9 tok/s at 128K (sparse layers scan growing KV caches)
- 128K tokens prefills in ~11 min on M3 Max; 256K in ~16 min

### Recommendation

- **8-bit** — Best balance of speed (28 tok/s), quality (87.5%), and memory (9.6 GB)
- **4-bit** — Use when memory-constrained; 25% faster decode but potentially lower quality
- **fp16** — Reserve for accuracy-critical batch processing (3.6 tok/s too slow for interactive use)

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) — SALA hybrid attention mechanism
- [MLX Port Guide](docs/MLX_PORT_GUIDE.md) — Gaps and implementation details
- [Implementation Roadmap](docs/IMPLEMENTATION_ROADMAP.md) — Phased development plan

## References

- [HuggingFace Model (upstream)](https://huggingface.co/openbmb/MiniCPM-SALA)
- [MLX 8-bit Weights](https://huggingface.co/moxin-org/MiniCPM4-SALA-9B-8bit-mlx)
- [GitHub Repository](https://github.com/OpenBMB/MiniCPM)
- [Technical Report](https://arxiv.org/abs/2026.xxxxx)
- [SGLang Integration](https://github.com/OpenBMB/sglang/tree/minicpm_sala)

## Related Projects

This project is part of the [OminiX-MLX](https://github.com/OminiX-ai/OminiX-MLX) ecosystem, which includes:
- `funasr-mlx` — Speech recognition
- `moxin-vlm-mlx` — Vision-language model
- `qwen3-mlx` — Qwen3 language model
- `gpt-sovits-mlx` — Voice cloning

## License

Apache-2.0 (same as upstream MiniCPM-SALA)
