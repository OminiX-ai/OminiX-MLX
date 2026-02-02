# flux-klein-mlx

FLUX.2-klein image generation model for Apple Silicon using MLX.

## Features

- **FLUX.2-klein transformer**: 4B parameter diffusion model
- **Qwen3-4B text encoder**: 36 layers, 2560 hidden dim
- **VAE decoder**: AutoencoderKL with 32 latent channels
- **4-step generation**: Fast inference with rectified flow
- **INT8 quantization**: Optional memory-efficient mode (~8GB vs ~13GB)

## Model Download

The model downloads automatically from HuggingFace. You need access to the gated model:

```bash
# Login to HuggingFace (required for gated model)
huggingface-cli login

# Or set token via environment variable
export HF_TOKEN=your_token_here
```

### Download URLs

| Source | URL |
|--------|-----|
| **HuggingFace** | https://huggingface.co/black-forest-labs/FLUX.2-klein-4B |
| **ModelScope** | https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-4B |
| **Original FLUX.1** | https://huggingface.co/black-forest-labs/FLUX.1-schnell |

### Environment Variables

```bash
# Use custom model path (optional)
export FLUX_MODEL_DIR=/path/to/your/flux-model

# HuggingFace token (for gated model)
export HF_TOKEN=your_token_here

# Or specify when running
FLUX_MODEL_DIR=./models/flux cargo run --example generate_klein --release
```

### Manual Download

```bash
# Using huggingface-cli (requires login for gated model)
huggingface-cli login
huggingface-cli download black-forest-labs/FLUX.2-klein-4B --local-dir ./models/flux

# Using git lfs
git lfs install
git clone https://huggingface.co/black-forest-labs/FLUX.2-klein-4B ./models/flux
```

### Required Files

```
models/flux/
├── transformer/
│   └── diffusion_pytorch_model.safetensors  # ~8GB
├── text_encoder/
│   ├── model-00001-of-00002.safetensors     # ~5GB
│   └── model-00002-of-00002.safetensors     # ~5GB
├── vae/
│   └── diffusion_pytorch_model.safetensors  # ~160MB
└── tokenizer/
    └── tokenizer.json
```

## Usage

### Command Line

```bash
# Basic generation (512x512, 4 steps)
cargo run --example generate_klein --release -- "a beautiful sunset over the ocean"

# With quantization (reduces VRAM from ~13GB to ~8GB)
cargo run --example generate_klein --release -- --quantize "a cat sitting on a windowsill"

# Custom number of steps
cargo run --example generate_klein --release -- --steps 8 "detailed portrait of a knight"
```

### Library Usage

```rust
use flux_klein_mlx::{
    FluxKlein, FluxKleinParams,
    Qwen3TextEncoder, Qwen3Config,
    Decoder, AutoEncoderConfig,
    load_safetensors, sanitize_klein_model_weights, sanitize_vae_weights,
};

// Load text encoder
let qwen3_config = Qwen3Config {
    hidden_size: 2560,
    num_hidden_layers: 36,
    intermediate_size: 9728,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    rms_norm_eps: 1e-6,
    vocab_size: 151936,
    max_position_embeddings: 40960,
    rope_theta: 1000000.0,
    head_dim: 128,
};
let mut text_encoder = Qwen3TextEncoder::new(qwen3_config)?;

// Load transformer
let params = FluxKleinParams::default();
let mut transformer = FluxKlein::new(params)?;

// Load VAE decoder
let vae_config = AutoEncoderConfig::flux2();
let mut vae = Decoder::new(vae_config)?;

// Load weights (after downloading)
// ... load and apply weights ...

// Generate
let txt_embed = text_encoder.encode(&input_ids, Some(&attention_mask))?;
let latent = transformer.forward_with_rope(&noise, &txt_embed, &timestep, &rope_cos, &rope_sin)?;
let image = vae.forward(&latent)?;
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FLUX.2-klein Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────┐  │
│  │   Qwen3-4B  │    │  FLUX Transformer │    │    VAE    │  │
│  │   Encoder   │───▶│  5 double blocks  │───▶│  Decoder  │  │
│  │  36 layers  │    │ 20 single blocks  │    │  32 ch    │  │
│  └─────────────┘    └──────────────────┘    └───────────┘  │
│        │                    │                      │        │
│    [B,512,2560]        [B,1024,128]          [B,H,W,3]     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Denoising: 4 Euler steps with SNR-shifted schedule
Latent: 32 channels, 2x2 patch (128 dim per patch)
```

## Performance

On Apple M3 Max (128GB):

| Mode | VRAM | Time (512x512) |
|------|------|----------------|
| FP32 | ~13GB | ~5s |
| INT8 | ~8GB | ~6s |

## License

MIT OR Apache-2.0
