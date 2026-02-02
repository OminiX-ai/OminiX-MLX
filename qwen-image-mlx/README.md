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

### Available Models

| Model | Size | HuggingFace |
|-------|------|-------------|
| BF16 (full precision) | 57.7 GB | [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) |
| 8-bit quantized | 36.1 GB | [mlx-community/Qwen-Image-2512-8bit](https://huggingface.co/mlx-community/Qwen-Image-2512-8bit) |
| 4-bit quantized | 25.9 GB | [mlx-community/Qwen-Image-2512-4bit](https://huggingface.co/mlx-community/Qwen-Image-2512-4bit) |

### Download Models

```bash
# Install huggingface-cli if needed
pip install huggingface_hub

# 4-bit quantized (smallest, recommended for limited memory)
huggingface-cli download mlx-community/Qwen-Image-2512-4bit --local-dir ~/.dora/models/qwen-image-2512-4bit

# 8-bit quantized (better quality, larger)
huggingface-cli download mlx-community/Qwen-Image-2512-8bit --local-dir ~/.dora/models/qwen-image-2512-8bit

# Full precision BF16 (best quality)
huggingface-cli download Qwen/Qwen-Image-2512 --local-dir ~/.dora/models/qwen-image-2512
```

### Model Directory Structure

```
~/.dora/models/
├── qwen-image-2512/           # Full precision (BF16, 57.7 GB)
│   ├── transformer/
│   ├── text_encoder/
│   ├── vae/
│   └── tokenizer/
├── qwen-image-2512-4bit/      # 4-bit quantized (25.9 GB)
│   └── ...
└── qwen-image-2512-8bit/      # 8-bit quantized (36.1 GB)
    └── ...
```

### Custom Model Path

By default, models are loaded from `~/.dora/models/`. To use a custom location:

```bash
# Set environment variable
export DORA_MODELS_PATH=/path/to/your/models

# Then run generation
cargo run --release --example generate_fp32 -- -p "a fluffy cat"
```

Or inline:
```bash
DORA_MODELS_PATH=/path/to/models cargo run --release --example generate_fp32 -- -p "a fluffy cat"
```

## CLI Usage

### Full Precision BF16 (Best Quality)

```bash
cargo run --release --example generate_fp32 -- -p "a fluffy cat" -o output.png
```

### 4-bit Quantized (Smallest Memory)

```bash
cargo run --release --example generate_qwen_image -- -p "a fluffy cat" -o output.png
```

### 8-bit Quantized

```bash
cargo run --release --example generate_qwen_image -- --use-8bit -p "a fluffy cat" -o output.png
```

### Options

```
-p, --prompt <PROMPT>      Text prompt for image generation
-o, --output <FILE>        Output image path [default: output.png]
-W, --width <WIDTH>        Image width [default: 1024]
-H, --height <HEIGHT>      Image height [default: 1024]
-s, --steps <STEPS>        Number of diffusion steps [default: 20]
-g, --guidance <SCALE>     Classifier-free guidance scale [default: 4.0]
--seed <SEED>              Random seed for reproducibility
--use-8bit                 Use 8-bit quantization (generate_qwen_image only)
```

### Example

```bash
cargo run --release --example generate_fp32 -- \
  -p "a majestic lion in the savanna at sunset" \
  -o lion.png \
  -W 1024 -H 1024 \
  -s 30 \
  -g 5.0 \
  --seed 42
```

## Seed

The `--seed` parameter controls the initial random noise:
- Same seed + same prompt = identical image
- Different seed = different variation
- Omit for random seed each run

## License

Apache 2.0 - Derived from HuggingFace Diffusers and QwenLM/Qwen-Image.
