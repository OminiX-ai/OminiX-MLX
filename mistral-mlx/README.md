# mistral-mlx

Mistral LLM inference on Apple Silicon with MLX.

## Features

- **Sliding Window Attention**: Efficient long-context processing
- **Grouped Query Attention (GQA)**: Reduced memory for KV cache
- **RoPE Position Embeddings**: Rotary position encoding
- **Quantization Support**: 4-bit and 8-bit quantized models

## Installation

```toml
[dependencies]
mistral-mlx = { path = "../mistral-mlx" }
```

## Quick Start

```rust
use mistral_mlx::{load_model, load_tokenizer, Generate, KVCache};
use mlx_rs::ops::indexing::NewAxis;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model and tokenizer
    let mut model = load_model("path/to/Mistral-7B")?;
    let tokenizer = load_tokenizer("path/to/Mistral-7B")?;

    // Prepare prompt
    let encoding = tokenizer.encode("Hello, ", true)?;
    let prompt = mlx_rs::Array::from(encoding.get_ids()).index(NewAxis);
    let mut cache = Vec::new();

    // Generate
    let generator = Generate::<KVCache>::new(&mut model, &mut cache, 0.7, &prompt);

    for token in generator.take(50) {
        let token = token?;
        print!("{}", tokenizer.decode(&[token.item::<u32>()], true)?);
    }

    Ok(())
}
```

## CLI Usage

### Download Model

```bash
# Mistral 7B (4-bit quantized)
huggingface-cli download mlx-community/Mistral-7B-Instruct-v0.3-4bit \
    --local-dir ./models/Mistral-7B-4bit

# Mistral 7B (bf16)
huggingface-cli download mlx-community/Mistral-7B-Instruct-v0.3-bf16 \
    --local-dir ./models/Mistral-7B
```

### Run Examples

```bash
# Text generation
cargo run --release --example generate_mistral -- \
    ./models/Mistral-7B-4bit "Explain machine learning in simple terms"

# Benchmark
cargo run --release --example benchmark_mistral -- \
    ./models/Mistral-7B-4bit --iterations 10
```

## Architecture

Mistral uses an optimized transformer architecture:

```
Mistral-7B
├── Embedding (vocab_size=32000, dim=4096)
├── 32x DecoderLayer
│   ├── input_layernorm
│   ├── Attention (GQA with sliding window)
│   │   ├── q_proj (32 heads)
│   │   ├── k_proj (8 heads, GQA)
│   │   ├── v_proj (8 heads, GQA)
│   │   └── o_proj
│   ├── post_attention_layernorm
│   └── MLP (SwiGLU)
│       ├── gate_proj
│       ├── up_proj
│       └── down_proj
└── norm
```

### Key Optimizations

- **Sliding Window**: 4096 token attention window for long sequences
- **GQA**: 4:1 query:key-value head ratio reduces memory
- **SwiGLU**: Improved MLP activation function

## Model Files

A model directory should contain:
- `model.safetensors` or `model-*.safetensors` - Model weights
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer

## Supported Models

| Model | Parameters | Notes |
|-------|------------|-------|
| Mistral-7B | 7B | Base and Instruct variants |
| Mistral-7B-v0.3 | 7B | Latest version |

## Performance

Benchmarks on Apple M3 Max:

| Model | Memory | Tokens/sec |
|-------|--------|------------|
| Mistral-7B (bf16) | 14GB | 40 tok/s |
| Mistral-7B (4-bit) | 4GB | 55 tok/s |

## Re-exports

This crate re-exports common components from `mlx-rs-core`:
- `KVCache`, `ConcatKeyValueCache`, `KeyValueCache`
- `create_attention_mask`, `scaled_dot_product_attention`
- `initialize_rope`, `AttentionMask`, `SdpaMask`

## License

MIT OR Apache-2.0
