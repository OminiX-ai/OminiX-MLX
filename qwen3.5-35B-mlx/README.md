# qwen3-5-35b-mlx

Qwen3.5-27B hybrid DeltaNet+Attention inference on Apple Silicon using MLX.

Qwen3.5-27B is a 27-billion parameter dense language model with a novel **hybrid architecture**: 48 layers of **Gated DeltaNet** (linear attention with fixed-size recurrent state) interleaved with 16 layers of **Gated Full Attention** (standard attention with partial RoPE and sigmoid output gate). This gives the model O(1) memory complexity for 75% of its layers while retaining full quadratic attention where it matters most.

## Features

- Hybrid DeltaNet + Gated Attention — first Rust implementation of this architecture
- 4-bit and 8-bit quantized inference via MLX affine quantization
- Fixed-size recurrent state for DeltaNet layers (constant memory, no growing KV cache)
- Optimized prefill with matmul-based recurrence and periodic async evaluation
- Gated attention with partial RoPE (25% of head dimensions) and sigmoid output gate
- Auto-detection of VLM weight prefix (`language_model.model.*` vs `model.*`)
- Async token pipelining for maximum decode throughput

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
qwen3-5-35b-mlx = { path = "../qwen3.5-35B-mlx" }
```

## Quick Start

```rust
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use qwen3_5_35b_mlx::{load_model, load_tokenizer, Generate};

fn main() -> anyhow::Result<()> {
    let model_dir = "./models/Qwen3.5-27B-4bit";

    // Load model and tokenizer
    let tokenizer = load_tokenizer(model_dir)?;
    let mut model = load_model(model_dir)?;

    // Tokenize prompt
    let prompt = "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n";
    let encoding = tokenizer.encode(prompt, false).unwrap();
    let tokens = mlx_rs::Array::from(encoding.get_ids()).index(NewAxis);

    // Generate
    let gen = Generate::new(&mut model, 0.7, &tokens);
    for token in gen.take(200) {
        let token = token?;
        let id = token.item::<u32>();
        if id == 248044 { break; } // EOS
        print!("{}", tokenizer.decode(&[id], true).unwrap());
    }

    Ok(())
}
```

## Examples

```bash
# Text generation with default prompt
cargo run --release --example generate -- ./models/Qwen3.5-27B-4bit

# Custom prompt, max tokens, temperature
cargo run --release --example generate -- ./models/Qwen3.5-27B-4bit \
  "<|im_start|>user\nExplain quantum computing.<|im_end|>\n<|im_start|>assistant\n" \
  500 0.7

# Force CPU mode (very slow, for debugging only)
cargo run --release --example generate -- --cpu ./models/Qwen3.5-27B-4bit
```

## Model Download

Download pre-quantized MLX models from Hugging Face:

```bash
# 4-bit quantized (~15.5 GB, recommended)
huggingface-cli download mlx-community/Qwen3.5-27B-4bit \
  --local-dir ./models/Qwen3.5-27B-4bit

# 8-bit quantized (~29.5 GB, higher quality)
huggingface-cli download mlx-community/Qwen3.5-27B-8bit \
  --local-dir ./models/Qwen3.5-27B-8bit
```

## Supported Models

| Model | HuggingFace Path | Size | Decode Speed |
|-------|------------------|------|-------------|
| Qwen3.5-27B 4-bit | `mlx-community/Qwen3.5-27B-4bit` | ~15.5 GB | ~15 tok/s |
| Qwen3.5-27B 8-bit | `mlx-community/Qwen3.5-27B-8bit` | ~29.5 GB | ~7.4 tok/s |

## Performance

Benchmarks on Apple M3 Max (40-core GPU, 128 GB):

| Metric | 4-bit | 8-bit |
|--------|-------|-------|
| TTFT (60 tokens) | 0.69s | ~1.4s |
| TTFT (257 tokens) | 2.37s | ~4.7s |
| Decode speed | 15.3 tok/s | 7.4 tok/s |

TTFT = time to first token (includes prefill of all prompt tokens).

## Architecture

Qwen3.5-27B uses a repeating pattern of `[DeltaNet, DeltaNet, DeltaNet, FullAttention] x 16` for its 64 layers:

```
Input tokens
    |
    v
[Embedding] (vocab=248320, hidden=5120)
    |
    v
[Layer 0]  DeltaNet (linear attention)  ─┐
[Layer 1]  DeltaNet                       │  x 16 blocks
[Layer 2]  DeltaNet                       │
[Layer 3]  Gated Full Attention          ─┘
    ...
[Layer 63] Gated Full Attention
    |
    v
[RMSNorm] -> [LM Head] -> logits
```

### Gated DeltaNet (48 layers)

Linear attention using the **delta rule** to maintain a fixed-size state matrix `S ∈ R^{K x V}` per head:

```
S_t = exp(g_t) * S_{t-1} + beta_t * k_t ⊗ (v_t - S_{t-1}^T @ k_t)
o_t = S_t^T @ q_t
```

- **16 key/query heads** (dim 128), **48 value heads** (dim 128)
- Causal conv1d (kernel=4) before splitting Q/K/V
- L2-normalized Q, K with per-head exponential decay gate
- Sigmoid beta gate controls state update magnitude
- Output gated by `RMSNorm(o) * SiLU(z)` where z is a learned gate

The recurrent state has **constant size** regardless of sequence length, unlike KV cache which grows linearly.

### Gated Full Attention (16 layers)

Standard grouped-query attention with additional features:

- **24 query heads**, **4 KV heads** (GQA ratio 6:1), head_dim=256
- **Partial RoPE**: only 25% of head dimensions (64 out of 256) use rotary embeddings
- **Sigmoid output gate**: `q_proj` outputs `[query | gate]`, attention result is multiplied by `sigmoid(gate)`
- Q and K normalization via RMSNorm (pre-attention)
- Scaled dot-product attention with KV cache for decode

### MLP (all 64 layers)

Standard SiLU-gated feedforward: `down_proj(SiLU(gate_proj(x)) * up_proj(x))`, intermediate_size=17408.

## Crate Structure

```
src/
  lib.rs         Public API: load_model, load_tokenizer, Generate iterator, sampling
  config.rs      ModelArgs, TextConfig deserialization from config.json
  model.rs       Model, TransformerBlock, weight loading from safetensors
  attention.rs   GatedAttention (full attention with output gate + partial RoPE)
  deltanet.rs    GatedDeltaNet (linear attention with recurrent state)
  mlp.rs         SiLU-gated MLP
  cache.rs       HybridCache enum (KVCache | RecurrentState)
examples/
  generate.rs    Text generation CLI with timing metrics
```

## License

MIT OR Apache-2.0
