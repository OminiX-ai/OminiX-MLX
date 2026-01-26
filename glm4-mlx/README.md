# glm4-mlx

GLM-4 LLM inference on Apple Silicon using MLX.

## Features

- **Partial RoPE**: Rotary position embedding on half of head dimensions
- **Fused MLP**: Combined gate_up_proj for better efficiency
- **Extra LayerNorms**: post_self_attn and post_mlp normalization
- Support for quantized (4-bit) models
- Step-based KV cache for memory efficiency

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
glm4-mlx = { path = "../glm4-mlx" }
```

## Quick Start

```rust
use glm4_mlx::{load_model, load_tokenizer, Generate, KVCache};
use mlx_rs::ops::indexing::NewAxis;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = "path/to/GLM-4-9B";

    let tokenizer = load_tokenizer(model_dir)?;
    let mut model = load_model(model_dir)?;

    let encoding = tokenizer.encode("你好，", true)?;
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

## GLM-4 Architecture Notes

### Partial RoPE
GLM-4 only applies RoPE to the first half of head dimensions (`partial_rotary_factor = 0.5`).

### Fused gate_up_proj
The MLP uses a single projection to 2×hidden_dim, then splits for gate and up paths:
```
x → gate_up_proj → [gate, up] → silu(gate) * up → down_proj → output
```

### Extra LayerNorms
Each decoder layer has 4 LayerNorms:
- `input_layernorm` (before attention)
- `post_self_attn_layernorm` (after attention, before residual)
- `post_attention_layernorm` (before MLP)
- `post_mlp_layernorm` (after MLP, before residual)

## Examples

```bash
cargo run --release --example generate_glm4 -- ./GLM-4-9B "你好"
```

## Model Download

Download pre-converted MLX models from Hugging Face:

```bash
# GLM-4-9B Chat (recommended)
huggingface-cli download mlx-community/glm-4-9b-chat-4bit --local-dir ./models/GLM-4-9B-4bit

# GLM-4-9B bf16 (requires 18GB+ memory)
huggingface-cli download mlx-community/glm-4-9b-chat-bf16 --local-dir ./models/GLM-4-9B
```

Or convert from Hugging Face yourself:

```bash
pip install mlx-lm
mlx_lm.convert --hf-path THUDM/glm-4-9b-chat -q
```

## Supported Models

| Model | HuggingFace Path | Size |
|-------|------------------|------|
| GLM-4-9B | `mlx-community/glm-4-9b-chat-bf16` | 18 GB |
| GLM-4-9B-4bit | `mlx-community/glm-4-9b-chat-4bit` | 6 GB |

## License

MIT OR Apache-2.0
