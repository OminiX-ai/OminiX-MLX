# Qwen-Image MLX

Rust implementation of Qwen-Image text-to-image model using MLX.

## Requirements

- macOS with Apple Silicon
- Rust 1.70+

## Installation

```bash
cargo build --release
```

## Model Setup

Models are automatically downloaded from HuggingFace on first run. For faster loading, organize models in `~/.dora/models/`:

```
~/.dora/models/
├── qwen-image-2512/           # Full precision (BF16)
│   ├── transformer/
│   ├── text_encoder/
│   ├── vae/
│   └── tokenizer/
└── qwen-image-2512-4bit/      # 4-bit quantized
    ├── transformer/
    ├── text_encoder/
    ├── vae/
    └── tokenizer/
```

Override with environment variable:
```bash
export DORA_MODELS_PATH=/path/to/models
```

## CLI Usage

### Full Precision (Best Quality)

```bash
cargo run --release --example generate_fp32 -- -p "a fluffy cat" -o output.png
```

Options:
```
-p, --prompt <PROMPT>      Text prompt for image generation
-o, --output <FILE>        Output image path [default: output.png]
-W, --width <WIDTH>        Image width [default: 1024]
-H, --height <HEIGHT>      Image height [default: 1024]
-s, --steps <STEPS>        Number of diffusion steps [default: 20]
-g, --guidance <SCALE>     Classifier-free guidance scale [default: 4.0]
--seed <SEED>              Random seed for reproducibility
```

Example:
```bash
cargo run --release --example generate_fp32 -- \
  -p "a majestic lion in the savanna at sunset" \
  -o lion.png \
  -W 1024 -H 1024 \
  -s 30 \
  -g 5.0 \
  --seed 42
```

### 4-bit Quantized (Faster, Lower Memory)

```bash
cargo run --release --example generate_qwen_image -- -p "a fluffy cat" -o output.png
```

Same options as full precision. Uses 4-bit quantized transformer weights for reduced memory and faster inference at slight quality cost.

## Seed

The `--seed` parameter controls the initial random noise:
- Same seed + same prompt = identical image
- Different seed = different variation
- Omit for random seed each run

## License

Apache 2.0 - Derived from HuggingFace Diffusers and QwenLM/Qwen-Image.
