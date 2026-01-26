# zimage-mlx

Z-Image-Turbo image generation model for Apple Silicon using MLX.

## Features

- **Z-Image-Turbo transformer**: 6B parameter Single-Stream DiT (S3-DiT)
- **9-step Turbo inference**: Distilled for fast generation (~3s/image)
- **4-bit quantization**: Memory-efficient inference (~3GB vs ~12GB)
- **3-axis RoPE**: Optimized position encoding [32, 48, 48]

## Model Download

The model downloads automatically from HuggingFace:

```bash
# Model ID (public, no auth required)
# HuggingFace: Freepik/Z-Image-Turbo-mlx
```

Required files (downloaded automatically):
```
transformer_q4.safetensors     # ~3GB (quantized transformer)
text_encoder_q4.safetensors    # ~2GB (quantized Qwen3)
vae.safetensors                # ~160MB
tokenizer.json
```

## Usage

### Command Line

```bash
# Basic generation (512x512, 9 steps, quantized)
cargo run --example generate_zimage_quantized --release -- "a beautiful sunset over the ocean"

# Full precision (requires ~12GB VRAM)
cargo run --example generate_zimage --release -- "a cat sitting on a windowsill"
```

### Library Usage

```rust
use zimage_mlx::{
    ZImageTransformer, ZImageConfig,
    load_quantized_zimage_transformer,
    load_quantized_qwen3_encoder,
    Decoder, AutoEncoderConfig,
    load_safetensors, sanitize_vae_weights,
};

// Load quantized text encoder (lower memory)
let text_encoder = load_quantized_qwen3_encoder(&weights, config)?;

// Load quantized transformer
let transformer = load_quantized_zimage_transformer(&weights, ZImageConfig::default())?;

// Load VAE decoder
let vae_config = AutoEncoderConfig::flux2();
let mut vae = Decoder::new(vae_config)?;

// Generate
let txt_embed = text_encoder.encode(&input_ids, Some(&attention_mask))?;
let latent = transformer.forward(&noise, &txt_embed, &timestep, &rope)?;
let image = vae.forward(&latent)?;
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Z-Image-Turbo Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────┐  │
│  │   Qwen3-4B  │    │  S3-DiT Blocks   │    │    VAE    │  │
│  │   Encoder   │───▶│  Noise Refiner   │───▶│  Decoder  │  │
│  │  Layer 34   │    │  Context Refiner │    │  32 ch    │  │
│  │  (4-bit)    │    │  Joint Blocks    │    │           │  │
│  └─────────────┘    └──────────────────┘    └───────────┘  │
│        │                    │                      │        │
│    [B,512,2560]        [B,1024,3072]          [B,H,W,3]    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Denoising: 9 Euler steps (Turbo distilled)
RoPE: 3-axis [32, 48, 48] with theta=256
Latent: 32 channels, 2x2 patch
```

## Comparison with FLUX.2-klein

| Feature | Z-Image-Turbo | FLUX.2-klein |
|---------|---------------|--------------|
| Parameters | 6B | 4B |
| Steps | 9 | 4 |
| Architecture | S3-DiT (single stream) | Double + Single blocks |
| RoPE | 3-axis [32,48,48] | 4-axis [32,32,32,32] |
| Text encoder | Qwen3 layer 34 only | Qwen3 concat layers |
| Quantized memory | ~3GB | ~8GB |

## Performance

On Apple M3 Max (128GB):

| Mode | VRAM | Time (512x512) |
|------|------|----------------|
| Quantized (4-bit) | ~3GB | ~3s |
| Full precision | ~12GB | ~2.5s |

## License

MIT OR Apache-2.0
