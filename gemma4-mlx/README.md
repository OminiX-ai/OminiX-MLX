# gemma4-mlx

Gemma 4 12B text inference on Apple Silicon using MLX.

This crate implements the text-only path for the 4-bit
`mlx-community/gemma-4-12B-it-4bit` checkpoint in pure Rust. Vision and audio
components from the multimodal Gemma 4 family are intentionally out of scope.

## Features

- Gemma 4 12B text decoder with 48 transformer layers
- 4-bit attention projections and 8-bit MLP projections
- Global and sliding-window attention layer support
- Gemma 4 proportional RoPE for global layers
- KV-cache greedy generation
- Gemma 4 chat template support
- Tied embedding output projection with final logit softcapping
- Validation examples for layer parity, model parity, and decode consistency

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
gemma4-mlx = { path = "../gemma4-mlx" }
```

## Model Download

Download the MLX 4-bit checkpoint from Hugging Face:

```bash
huggingface-cli download mlx-community/gemma-4-12B-it-4bit \
    --local-dir ./models/gemma-4-12B-it-4bit
```

The model directory should contain:

- `config.json`
- `model.safetensors.index.json`
- `model-*.safetensors`
- tokenizer files such as `tokenizer.json`

## Quick Start

```rust
use gemma4_mlx::{encode_chat, eos_ids, generate_greedy, load_model, load_tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = "./models/gemma-4-12B-it-4bit";

    let tokenizer = load_tokenizer(model_dir)?;
    let prompt = encode_chat(&tokenizer, "Explain what MLX is in one sentence.")?;
    let eos = eos_ids();

    let mut model = load_model(model_dir)?;
    let generated = generate_greedy(&mut model, &prompt, 128, &eos)?;

    let generated_u32: Vec<u32> = generated.iter().map(|&id| id as u32).collect();
    let reply = tokenizer.decode(&generated_u32, true)?;
    println!("{reply}");

    Ok(())
}
```

## Examples

Run chat inference:

```bash
M=./models/gemma-4-12B-it-4bit

cargo run -p gemma4-mlx --release --example chat_gemma4 -- \
    $M "Explain what MLX is in one sentence."
```

Run raw token generation:

```bash
M=./models/gemma-4-12B-it-4bit

cargo run -p gemma4-mlx --release --example generate_gemma4 -- \
    $M "2 1024 2048 4096 8192 16384" 8
```

Expected raw-token output:

```text
2818 107 100 100 14937 100 45518 107
```

Inspect weight and quantization layout:

```bash
cargo run -p gemma4-mlx --release --example inspect_gemma4 -- $M
```

## Architecture

The text path is:

```text
tokens
  -> quantized embedding * sqrt(hidden_size)
  -> 48 decoder layers
       -> RMSNorm
       -> self-attention
            - sliding layers: standard RoPE, 8 KV heads, v_proj
            - global layers: proportional RoPE, 1 KV head, k_eq_v
       -> RMSNorm + residual
       -> GeGLU MLP
       -> RMSNorm + residual
       -> layer_scalar
  -> final RMSNorm
  -> tied embedding projection
  -> 30 * tanh(logits / 30)
```

Important implementation details:

- Global attention layers do not have `v_proj`; values come from the raw
  `k_proj` output and then pass through no-scale `v_norm`.
- Proportional RoPE rotates 128 dimensions for global layers, with frequencies
  derived using the full 512-dimensional head size.
- Sliding decode masks are only materialized after the cache offset exceeds the
  sliding window; before that, all cached keys are visible.
- Prefill projects only the final hidden position to logits to avoid allocating
  `[batch, prompt_len, vocab]` for the 262144-token vocabulary.

## Validation

Unit tests do not require model weights:

```bash
cargo test -p gemma4-mlx --release -- --test-threads=1
```

The parity examples compare against Python `mlx-vlm` references. Generate
golden files first:

```bash
M=./models/gemma-4-12B-it-4bit

python3 gemma4-mlx/scripts/dump_gemma4_layer_io.py $M /tmp/gemma4_golden
python3 gemma4-mlx/scripts/dump_gemma4_logits.py $M /tmp/gemma4_logits
python3 gemma4-mlx/scripts/dump_gemma4_greedy.py $M /tmp/gemma4_greedy 8
```

Then run the Rust checks:

```bash
cargo run -p gemma4-mlx --release --example layer_parity -- \
    $M /tmp/gemma4_golden

cargo run -p gemma4-mlx --release --example model_parity -- \
    $M /tmp/gemma4_logits

cargo run -p gemma4-mlx --release --example decode_consistency -- \
    $M 8
```

Expected results:

- `layer_parity`: layer 0 and layer 5 max absolute differences around `1e-6`
- `model_parity`: next-token argmax `2818`
- `decode_consistency`: cache decode matches no-cache re-forward token-for-token

## Performance

The implementation is functionally complete, but decode performance is still an
active optimization area. On a 16 GB Apple Silicon machine, short chat runs have
been observed around `1-2 tok/s` after the initial correctness work and first
safe M3 optimizations.

Known performance work remains:

- More detailed per-token timing for prefill, decode, and synchronization
- Rotating KV cache support for sliding-window layers
- More stable decode shapes without increasing short-sequence compute
- Sampling and streaming generation

## Known Limitations

- Text inference only; Gemma 4 vision and audio paths are not implemented.
- Only greedy decoding is currently exposed.
- Full-model logits do not bit-match `mlx-vlm` exactly after 48 layers due to
  benign bf16 accumulation differences. Argmax/top candidates and internal
  cache/no-cache decode consistency are the intended correctness checks.
- MLX tests should be run with `--test-threads=1`.

## License

MIT OR Apache-2.0
