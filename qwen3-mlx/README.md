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

## Supported Models

- Qwen3-0.6B
- Qwen3-1.7B
- Qwen3-4B
- Qwen3-8B
- Qwen3-14B
- Qwen3-32B

Models can be downloaded from Hugging Face and converted using `mlx-lm`:

```bash
pip install mlx-lm
mlx_lm.convert --hf-path Qwen/Qwen3-4B -q
```

## Performance

On M3 Max (40-core GPU):

| Model | Prompt | Decode | Memory |
|-------|--------|--------|--------|
| Qwen3-4B (bf16) | 150 tok/s | 45 tok/s | 8 GB |
| Qwen3-4B (4-bit) | 250 tok/s | 75 tok/s | 3 GB |

## License

MIT OR Apache-2.0
