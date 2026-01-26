# Qwen-Image MLX Implementation Guide

<!-- Evolution: 2025-01-25 | source: qwen-image-mlx debugging session -->

Guide for implementing and debugging Qwen-Image (DiT-based diffusion model) in Rust with mlx-rs.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  QwenQuantizedTransformer               │
├─────────────────────────────────────────────────────────┤
│  img_in: QuantizedLinear (64 → 3072)                    │
│  txt_norm: RMSNorm (3584)                               │
│  txt_in: QuantizedLinear (3584 → 3072)                  │
│  time_text_embed: QwenTimeTextEmbed                     │
│  transformer_blocks: [QwenTransformerBlock; 60]         │
│  norm_out: QwenAdaLayerNormOut                          │
│  proj_out: QuantizedLinear (3072 → 64)                  │
└─────────────────────────────────────────────────────────┘
```

## Key Components

### Time Embedding Pipeline

```rust
// 1. Sinusoidal embedding (256 dims, scale=1000)
let sinusoidal = get_timestep_embedding(timestep, 256)?;  // [-1, 1]

// 2. MLP: linear_1 → silu → linear_2
let emb = linear_1.forward(&sinusoidal)?;  // [-10, 14]
let emb = mlx_rs::nn::silu(&emb)?;         // [-0.3, 14]
let temb = linear_2.forward(&emb)?;        // [-34, 63]
```

### Transformer Block

```rust
pub fn forward(&mut self, hidden_states, encoder_hidden_states, text_embeddings, ...) {
    // 1. Modulation parameters
    let img_silu = mlx_rs::nn::silu(text_embeddings)?;
    let img_mod_params = self.img_mod_linear.forward(&img_silu)?;  // [-132, 112]
    let (img_mod1, img_mod2) = split_half(&img_mod_params)?;

    // 2. LayerNorm + Modulation
    let img_normed = layer_norm(hidden_states, 1e-6)?;  // [-9, 9]
    let (img_modulated, img_gate1) = modulate(&img_normed, &img_mod1)?;  // [-66, 54]

    // 3. Joint Attention
    let (img_attn_out, txt_attn_out) = self.attn.forward(...)?;  // [-88, 118]

    // 4. Gated Residual
    let hidden_states = hidden_states + img_gate1 * img_attn_out;  // [-246, 443]

    // 5. FFN with mod2
    let img_normed2 = layer_norm(&hidden_states, 1e-6)?;
    let (img_modulated2, img_gate2) = modulate(&img_normed2, &img_mod2)?;
    let img_mlp_out = self.img_ff.forward(&img_modulated2)?;  // [-25500, 25600] ← NORMAL!
    let hidden_states = hidden_states + img_gate2 * img_mlp_out;  // [-2M, 2M] ← ALSO NORMAL!

    Ok((encoder_hidden_states, hidden_states))
}
```

### Modulation Formula

```rust
fn modulate(x: &Array, mod_params: &Array) -> Result<(Array, Array), Exception> {
    let dim = mod_params.dim(-1) / 3;
    let shift = mod_params.index((.., ..dim)).expand_dims(1)?;
    let scale = mod_params.index((.., dim..dim * 2)).expand_dims(1)?;
    let gate = mod_params.index((.., dim * 2..));

    // Formula: (1 + scale) * x + shift
    let one = Array::from_f32(1.0);
    let scale_factor = ops::add(&one, &scale)?;
    let modulated = ops::add(&ops::multiply(x, &scale_factor)?, &shift)?;

    Ok((modulated, gate))
}
```

## Value Ranges (Normal Behavior)

**Critical:** These large intermediate values are EXPECTED. Don't try to clamp them!

| Checkpoint | Expected Range | Notes |
|------------|----------------|-------|
| After img_in | [-15, 16] | Initial projection |
| img_normed | [-9, 9] | After LayerNorm |
| scale/shift | [-30, 20] | Modulation params |
| img_modulated | [-70, 55] | After modulation |
| img_attn_out | [-90, 120] | Attention output |
| img_mlp_out | [-25000, 26000] | FFN output (large!) |
| After block 0 | [-2M, 2M] | Accumulating |
| After block 59 | [-45M, 46M] | Maximum explosion |
| After norm_out | [-21, 13] | LayerNorm normalizes! |
| Final output | [-4.5, 4.5] | Back to reasonable |

## Weight Loading

### Key Mapping for QuantizedLinear

```rust
// MLX safetensors structure:
// - {prefix}.weight      → inner.weight (uint32 packed)
// - {prefix}.scales      → scales (bfloat16)
// - {prefix}.biases      → biases (bfloat16, quantization bias)
// - {prefix}.bias        → inner.bias (optional, layer bias)

fn sanitize_weight_name(name: &str) -> String {
    if name.ends_with(".weight") && !name.contains(".inner.") {
        name.replace(".weight", ".inner.weight")
    } else if name.ends_with(".bias") && !name.contains(".inner.")
           && !name.ends_with(".biases") {
        name.replace(".bias", ".inner.bias")
    } else {
        name.to_string()
    }
}
```

### Expected Weight Shapes (4-bit, group_size=64)

| Weight | Shape |
|--------|-------|
| time_text_embed.linear_1.weight | [3072, 32] (packed) |
| time_text_embed.linear_1.scales | [3072, 4] |
| time_text_embed.linear_1.biases | [3072, 4] |
| transformer_blocks.N.img_mod_linear.weight | [18432, 384] |
| norm_out.linear.weight | [6144, 384] |

## Common Issues

### Issue: Output is all black/noise

**Cause:** Scale/shift order wrong in modulation or AdaLayerNorm.

**Fix:** Verify order matches reference:
- Modulation: `[shift, scale, gate]` from mod1
- AdaLayerNormOut: `[scale, shift]` (scale first!)

### Issue: Values go to inf after block 0

**Cause:** Missing attention scaling.

**Fix:**
```rust
let scale = (self.head_dim as f32).sqrt();
let scores = ops::divide(&scores, &Array::from_f32(scale))?;
```

### Issue: Image appears tiled 3x3

**Cause:** Strided view from transpose not made contiguous.

**Fix:**
```rust
let img = tensor.transpose_axes(&[1, 2, 0])?;
let n = img.dim(0) * img.dim(1) * img.dim(2);
let img = img.reshape(&[n])?.reshape(&[img.dim(0), img.dim(1), img.dim(2)])?;
img.eval()?;
```

## Debugging Checklist

```rust
// Add these debug prints to trace issues:

// 1. After img_in
eprintln!("[IMG_IN] [{:.2}, {:.2}]", hidden.min()?, hidden.max()?);

// 2. After LayerNorm
eprintln!("[NORMED] [{:.2}, {:.2}]", normed.min()?, normed.max()?);

// 3. Modulation params
eprintln!("[MOD] scale=[{:.2}, {:.2}]", scale.min()?, scale.max()?);

// 4. After modulation
eprintln!("[MODULATED] [{:.2}, {:.2}]", modulated.min()?, modulated.max()?);

// 5. Attention output
eprintln!("[ATTN] [{:.2}, {:.2}]", attn_out.min()?, attn_out.max()?);

// 6. FFN output (will be large!)
eprintln!("[FFN] [{:.2}, {:.2}]", ffn_out.min()?, ffn_out.max()?);

// 7. Final (should be small again)
eprintln!("[FINAL] [{:.2}, {:.2}]", result.min()?, result.max()?);
```

## Comparison with mflux

Our Rust implementation produces nearly identical results to mflux (Python):

| Metric | Rust | mflux |
|--------|------|-------|
| RMSE | - | 6.62 |
| Mean | 38.66 | 41.15 |
| Std | 48.18 | 48.08 |

Both implementations show the same "value explosion" pattern - this is correct DiT behavior.

## Performance

| Resolution | Steps | Time (M3 Max) |
|------------|-------|---------------|
| 512x512 | 4 | ~17s |
| 512x512 | 20 | ~76s |
| 768x768 | 20 | ~170s |

## References

- [mflux Qwen implementation](https://github.com/filipstrand/mflux)
- [diffusers QwenImageTransformer](https://github.com/huggingface/diffusers)
- [mlx-community Qwen-Image-2512-4bit](https://huggingface.co/mlx-community/Qwen-Image-2512-4bit)
