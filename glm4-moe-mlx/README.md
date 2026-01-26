# glm4-moe-mlx

GLM-4.5 MoE (Mixture of Experts) LLM inference on Apple Silicon with MLX.

## Features

- **45 Experts**: Shared experts + routed experts architecture
- **Partial RoPE**: Rotary position embedding on partial head dimensions
- **Custom Metal Kernel**: Fused SwiGLU kernel (10x+ faster)
- **3-bit Quantization**: Ultra-low memory footprint
- **Top-k Routing**: Dynamic expert selection per token

## Installation

```toml
[dependencies]
glm4-moe-mlx = { path = "../glm4-moe-mlx" }
```

## Quick Start

```rust
use glm4_moe_mlx::{load_model, load_tokenizer, Generate, KVCache};
use mlx_rs::ops::indexing::NewAxis;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model and tokenizer
    let mut model = load_model("path/to/GLM-4.5-MoE-3bit")?;
    let tokenizer = load_tokenizer("path/to/GLM-4.5-MoE-3bit")?;

    // Prepare prompt
    let encoding = tokenizer.encode("你好，", true)?;
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
# GLM-4.5 MoE (3-bit quantized)
huggingface-cli download mlx-community/glm-4.5-chat-moe-3bit \
    --local-dir ./models/GLM-4.5-MoE-3bit
```

### Run Examples

```bash
# Text generation
cargo run --release --example generate_glm4_moe -- \
    ./models/GLM-4.5-MoE-3bit "请解释量子计算"
```

## Architecture

GLM-4.5 MoE uses a hybrid expert architecture:

```
GLM-4.5-MoE
├── Embedding (vocab_size=151552)
├── 60x MoEDecoderLayer
│   ├── input_layernorm
│   ├── Attention (partial RoPE)
│   │   ├── qkv_proj (fused)
│   │   └── o_proj
│   ├── post_attention_layernorm
│   └── MoEBlock
│       ├── gate (router)
│       ├── 2x shared_experts (always active)
│       └── 43x routed_experts (top-k selected)
└── final_layernorm
```

### Expert Routing

Each token is processed by:
1. **Shared experts**: Always active, provide base capabilities
2. **Routed experts**: Top-k (typically 2-4) selected per token

### Optimizations

- **Fused SwiGLU**: Custom Metal kernel for expert MLPs
- **gather_qmm**: Efficient batched expert dispatch
- **Partial RoPE**: Reduced computation for position encoding

## Model Files

A model directory should contain:
- `model.safetensors` or `model-*.safetensors` - Model weights
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer

## Performance

Benchmarks on Apple M3 Max:

| Model | Memory | Tokens/sec |
|-------|--------|------------|
| GLM-4.5-MoE (3-bit) | ~20GB | 15-20 tok/s |

Note: MoE models have variable compute per token depending on expert routing.

## Re-exports

This crate re-exports common components from `mlx-rs-core`:
- `KVCache`, `ConcatKeyValueCache`, `KeyValueCache`
- `fused_swiglu` - Custom Metal kernel
- `SdpaMask`, `FloatOrString`

## License

MIT OR Apache-2.0
