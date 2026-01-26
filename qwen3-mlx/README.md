# qwen3-mlx

Qwen3 LLM inference on Apple Silicon using MLX.

## Features

- Fast inference with Metal GPU acceleration
- Support for both dense and quantized (4-bit) models
- Async token pipelining for maximum throughput
- Step-based KV cache for memory efficiency

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
qwen3-mlx = { path = "../qwen3-mlx" }
```

## Quick Start

```rust
use qwen3_mlx::{load_model, load_tokenizer, Generate, KVCache};
use mlx_rs::ops::indexing::NewAxis;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = "path/to/Qwen3-4B-bf16";

    // Load model and tokenizer
    let tokenizer = load_tokenizer(model_dir)?;
    let mut model = load_model(model_dir)?;

    // Tokenize prompt
    let encoding = tokenizer.encode("Hello, I am", true)?;
    let prompt = mlx_rs::Array::from(encoding.get_ids()).index(NewAxis);

    // Generate
    let mut cache = Vec::new();
    let generator = Generate::<KVCache>::new(&mut model, &mut cache, 0.7, &prompt);

    for token in generator.take(100) {
        let token = token?;
        let text = tokenizer.decode(&[token.item::<u32>()], true)?;
        print!("{}", text);
    }

    Ok(())
}
```

## Examples

```bash
# Text generation
cargo run --release --example generate_qwen3 -- ./Qwen3-4B-bf16 "Hello, how are you?"

# Interactive chat
cargo run --release --example chat_qwen3 -- ./Qwen3-4B-bf16
```

## Model Download

Download pre-converted MLX models from Hugging Face:

```bash
# Qwen3-4B (recommended for testing)
huggingface-cli download mlx-community/Qwen3-4B-bf16 --local-dir ./models/Qwen3-4B

# Qwen3-8B
huggingface-cli download mlx-community/Qwen3-8B-bf16 --local-dir ./models/Qwen3-8B

# Qwen3-4B 4-bit quantized (smaller, faster)
huggingface-cli download mlx-community/Qwen3-4B-4bit --local-dir ./models/Qwen3-4B-4bit
```

Or convert from Hugging Face yourself:

```bash
pip install mlx-lm
mlx_lm.convert --hf-path Qwen/Qwen3-4B -q
```

## Supported Models

| Model | HuggingFace Path | Size |
|-------|------------------|------|
| Qwen3-0.6B | `mlx-community/Qwen3-0.6B-bf16` | 1.2 GB |
| Qwen3-1.7B | `mlx-community/Qwen3-1.7B-bf16` | 3.4 GB |
| Qwen3-4B | `mlx-community/Qwen3-4B-bf16` | 8 GB |
| Qwen3-8B | `mlx-community/Qwen3-8B-bf16` | 16 GB |
| Qwen3-14B | `mlx-community/Qwen3-14B-bf16` | 28 GB |
| Qwen3-32B | `mlx-community/Qwen3-32B-bf16` | 64 GB |

## Performance

On M3 Max (40-core GPU):

| Model | Prompt | Decode | Memory |
|-------|--------|--------|--------|
| Qwen3-4B (bf16) | 150 tok/s | 45 tok/s | 8 GB |
| Qwen3-4B (4-bit) | 250 tok/s | 75 tok/s | 3 GB |

## License

MIT OR Apache-2.0
