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

> **Early Planning Phase** — Currently documenting architecture and port requirements.

| Component | Status |
|-----------|--------|
| Architecture Analysis | Complete |
| MLX Gap Analysis | Complete |
| Weight Loading | Not Started |
| Standard Attention Fallback | Not Started |
| Linear Attention | Not Started |
| Sparse Attention | Not Started |

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) — SALA hybrid attention mechanism
- [MLX Port Guide](docs/MLX_PORT_GUIDE.md) — Gaps and implementation details
- [Implementation Roadmap](docs/IMPLEMENTATION_ROADMAP.md) — Phased development plan

## References

- [HuggingFace Model](https://huggingface.co/openbmb/MiniCPM-SALA)
- [GitHub Repository](https://github.com/OpenBMB/MiniCPM)
- [Technical Report](https://arxiv.org/abs/2026.xxxxx)
- [SGLang Integration](https://github.com/OpenBMB/sglang/tree/minicpm_sala)

## Related Projects

This project is part of the [OminiX-MLX](https://github.com/username/OminiX-MLX) ecosystem, which includes:
- `funasr-mlx` — Speech recognition
- `moxin-vlm-mlx` — Vision-language model
- `qwen3-mlx` — Qwen3 language model

## License

Apache-2.0 (same as upstream MiniCPM-SALA)
