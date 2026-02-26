# moxin-vlm-mlx

Moxin-7B Vision-Language Model inference on Apple Silicon, written in Rust with [MLX](https://github.com/ml-explore/mlx).

## Architecture

```
Image (224×224)
  ├─ DINOv2 ViT-L/14      → [B, 256, 1024]
  └─ SigLIP ViT-SO400M/14 → [B, 256, 1152]
              │ concat
        [B, 256, 2176]
              │ FusedMLPProjector
        [B, 256, 4096]   (256 visual tokens)
              │
  BOS + [visual tokens] + text tokens
              │ Mistral-7B decoder (36 layers, GQA 32Q/8KV)
        logits → autoregressive generation
```

## Features

- **Dual vision encoders** — DINOv2 ViT-L/14 (ImageNet stats) + SigLIP ViT-SO400M (unit norm), fused via FusedMLPProjector
- **8-bit quantization** — Mistral-7B decoder linear layers quantized to INT8; vision encoders kept in BF16
- **KV-cache generation** — standard prefill + cached decode loop
- **Sharded weights** — loads `model.safetensors.index.json` or single `model.safetensors`
- **Zero Python** — pure Rust, no Python runtime needed

## Quick Start

### 1. Download the model

```bash
huggingface-cli download moxin-org/moxin-llm-7b \
    --local-dir ~/.OminiX/models/moxin-vlm-7b
```

### 2. Generate from image + prompt

```bash
cargo run --release --example generate -- \
    --model ~/.OminiX/models/moxin-vlm-7b \
    --image photo.jpg \
    --prompt "Describe the image."
```

### 3. Save a quantized model

```bash
cargo run --release --example save_quantized -- \
    --model ~/.OminiX/models/moxin-vlm-7b \
    --output ~/.OminiX/models/moxin-vlm-7b-8bit
```

### 4. As a library

```rust
use moxin_vlm_mlx::{load_model, normalize_dino, normalize_siglip};

let mut vlm = load_model("~/.OminiX/models/moxin-vlm-7b")?;

// Optionally quantize the LLM decoder (8-bit, group_size=64)
let mut vlm = vlm.quantize(64, 8)?;

// Encode image + run generation ...
```

## Examples

| Example | Description |
|---------|-------------|
| `generate` | Basic image + text generation |
| `save_quantized` | Quantize and save the model |
| `server` | HTTP server with OpenAI-compatible `/v1/chat/completions` |

## Project Structure

```
moxin-vlm-mlx/
├── Cargo.toml
└── src/
    ├── lib.rs         # MoxinVLM, generation loop, model loading
    ├── vision.rs      # ViT encoder (DINOv2 / SigLIP)
    ├── projector.rs   # FusedMLPProjector
    └── error.rs       # Error types
```

## License

MIT OR Apache-2.0
