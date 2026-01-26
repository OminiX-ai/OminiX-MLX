# mixtral-mlx

Mixtral MoE (Mixture of Experts) LLM inference on Apple Silicon using MLX.

## Features

- **8 Expert MoE**: 8 experts with top-2 routing per token
- **Fused SwiGLU Kernel**: Custom Metal kernel for 10-12x faster expert MLP
- **Optimized Routing**: gather_qmm for efficient expert dispatch
- **4-bit Quantization**: Required for running on consumer hardware

## Installation

```toml
[dependencies]
mixtral-mlx = { path = "../mixtral-mlx" }
```

## Quick Start

```rust
use mixtral_mlx::{load_model, load_tokenizer, Generate, KVCache};
use mlx_rs::ops::indexing::NewAxis;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = "path/to/Mixtral-8x7B-4bit";

    let tokenizer = load_tokenizer(model_dir)?;
    let mut model = load_model(model_dir)?;

    let encoding = tokenizer.encode("Hello, I am", true)?;
    let prompt = mlx_rs::Array::from(encoding.get_ids()).index(NewAxis);

    let mut cache = Vec::new();
    let generator = Generate::<KVCache>::new(&mut model, &mut cache, 0.7, &prompt);

    for token in generator.take(100) {
        let token = token?;
        print!("{}", tokenizer.decode(&[token.item::<u32>()], true)?);
    }

    Ok(())
}
```

## MoE Architecture

### Expert Routing
Each token is routed to top-k (default: 2) experts based on gate logits:

```
x → gate_linear → softmax → top-k selection → dispatch to experts
```

### SwitchGLU
Experts use SwiGLU activation with a custom fused Metal kernel:

```rust
// Fused SwiGLU: silu(gate) * up
// This is 10-12x faster than separate operations
let activated = fused_swiglu(&up, &gate)?;
```

### Quantized Expert Dispatch
Uses `gather_qmm` for efficient batched expert computation:
- Tokens sorted by expert for coalesced memory access
- Single kernel call for all experts

## Model Download

Download pre-converted MLX models from Hugging Face:

```bash
# Mixtral-8x7B 4-bit (recommended, fits in 32GB)
huggingface-cli download mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit --local-dir ./models/Mixtral-8x7B-4bit

# Mixtral-8x22B 4-bit (requires 64GB+ memory)
huggingface-cli download mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit --local-dir ./models/Mixtral-8x22B-4bit
```

Or convert from Hugging Face yourself (4-bit quantization required for consumer hardware):

```bash
pip install mlx-lm
mlx_lm.convert --hf-path mistralai/Mixtral-8x7B-Instruct-v0.1 -q
```

## Supported Models

| Model | HuggingFace Path | Size |
|-------|------------------|------|
| Mixtral-8x7B-4bit | `mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit` | 26 GB |
| Mixtral-8x22B-4bit | `mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit` | 70 GB |

## Performance

On M3 Max (40-core GPU):

| Model | Prompt | Decode | Memory |
|-------|--------|--------|--------|
| Mixtral-8x7B (4-bit) | 80 tok/s | 25 tok/s | 26 GB |

## Example

```bash
cargo run --release --example generate_mixtral -- ./Mixtral-8x7B-4bit "Tell me about"
```

## License

MIT OR Apache-2.0
