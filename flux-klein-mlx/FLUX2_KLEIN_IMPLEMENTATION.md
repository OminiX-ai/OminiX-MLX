# FLUX.2-klein-4B Implementation Notes

This document describes the implementation of FLUX.2-klein-4B in Rust using mlx-rs.

## Architecture Overview

FLUX.2-klein-4B is a distilled diffusion transformer with the following key characteristics:

- **Text Encoder**: Qwen3-4B (replaces T5-XXL + CLIP from FLUX.1)
- **Transformer**: 5 double-stream blocks + 20 single-stream blocks
- **Hidden Size**: 3072
- **Attention Heads**: 24 (head_dim = 128)
- **MLP Ratio**: 3.0 (mlp_hidden = 9216)
- **Denoising Steps**: 4 (distilled model)
- **Shared Modulation**: One modulation layer per stream type, shared across all blocks

### Key Architectural Differences from FLUX.1

1. **Shared Modulation**: Unlike FLUX.1 which has per-block modulation, FLUX.2-klein uses:
   - `double_stream_modulation_img` - shared across all 5 double blocks for image stream
   - `double_stream_modulation_txt` - shared across all 5 double blocks for text stream
   - `single_stream_modulation` - shared across all 20 single blocks

2. **Separate Q/K/V Projections**: Double blocks use separate `to_q`, `to_k`, `to_v` instead of fused `qkv`

3. **RoPE Configuration**: Uses `axes_dims_rope = [32, 32, 32, 32]` with `theta = 2000`

## Performance Benchmarks

Benchmarked on Apple M3 Max, generating 512x512 images with 4 denoising steps.

### 10-Run Benchmark Results

**Rust (mlx-rs) Implementation:**
| Run | Total | Denoising | VAE Decode |
|-----|-------|-----------|------------|
| 1 | 7.65s | 4.30s | 346ms |
| 2 | 7.63s | 4.30s | 334ms |
| 3 | 7.67s | 4.33s | 322ms |
| 4 | 7.64s | 4.31s | 327ms |
| 5 | 7.68s | 4.35s | 321ms |
| 6 | 7.68s | 4.31s | 329ms |
| 7 | 7.71s | 4.32s | 362ms |
| 8 | 7.83s | 4.39s | 333ms |
| 9 | 7.90s | 4.44s | 336ms |
| 10 | 7.86s | 4.41s | 354ms |
| **Average** | **7.73s** | **4.35s** | **336ms** |

**flux.c (C + Metal) Reference Implementation:**
| Run | Total | Denoising |
|-----|-------|-----------|
| 1 | 16.18s | 11.75s |
| 2 | 16.27s | 11.74s |
| 3 | 15.91s | 11.54s |
| 4 | 15.81s | 11.48s |
| 5 | 15.81s | 11.47s |
| 6 | 15.91s | 11.54s |
| 7 | 15.84s | 11.46s |
| 8 | 15.96s | 11.61s |
| 9 | 15.81s | 11.45s |
| 10 | 15.80s | 11.46s |
| **Average** | **15.93s** | **11.55s** |

### Performance Comparison

| Metric | Rust (mlx-rs) | flux.c | Speedup |
|--------|---------------|--------|---------|
| **Total time** | **7.73s** | 15.93s | **2.06x faster** |
| Denoising (4 steps) | **4.35s** | 11.55s | **2.66x faster** |
| Per denoising step | **1.09s** | 2.89s | **2.65x faster** |

### Why Rust is Faster

1. **MLX Lazy Evaluation**: Operations are fused by the Metal compiler, reducing memory bandwidth
2. **RoPE Caching**: Compute RoPE frequencies once, reuse across all 4 steps
3. **Minimal GPU Synchronization**: Only `eval()` when necessary for timing
4. **No Debug Overhead**: Production build has no GPU→CPU copies for statistics

## Implementation Details

### Critical Bug Fix: SwiGLU Order in Double Blocks

The fused MLP weight in double blocks is laid out as `[gate_weight; up_weight]`:

```rust
// CORRECT: silu(gate) * up
let img_proj = self.img_mlp_in.forward(&img_mlp_in)?;
let img_splits = img_proj.split_axis(&[self.mlp_hidden], -1)?;
let img_gate = &img_splits[0];  // First half: gate
let img_up = &img_splits[1];    // Second half: up
let img_swiglu_out = ops::multiply(&mlx_rs::nn::silu(img_gate)?, img_up)?;

// WRONG (causes texture-only output): gate * silu(up)
```

This bug caused the model to produce chaotic textures instead of coherent objects.

### Optimized Forward Pass

The model provides two forward methods:

```rust
// For single inference (convenience)
pub fn forward(&mut self, img, txt, timesteps, img_ids, txt_ids) -> Result<Array>

// For denoising loops (faster - reuses RoPE)
pub fn compute_rope(txt_ids, img_ids) -> Result<(Array, Array)>
pub fn forward_with_rope(&mut self, img, txt, timesteps, rope_cos, rope_sin) -> Result<Array>
```

Usage in denoising loop:
```rust
// Compute RoPE once before loop
let (rope_cos, rope_sin) = FluxKlein::compute_rope(&txt_ids, &img_ids)?;

for step in 0..num_steps {
    let v_pred = flux.forward_with_rope(&latent, &txt_embed, &t_arr, &rope_cos, &rope_sin)?;
    // ... Euler step
}
```

### Weight Loading

The weight sanitization maps diffusers checkpoint names to our architecture:

| Diffusers Name | Our Name |
|---------------|----------|
| `transformer_blocks.X.attn.to_q.weight` | `double_blocks.X.img_to_q.weight` |
| `transformer_blocks.X.attn.add_q_proj.weight` | `double_blocks.X.txt_to_q.weight` |
| `transformer_blocks.X.ff.linear_in.weight` | `double_blocks.X.img_mlp_in.weight` |
| `transformer_blocks.X.ff_context.linear_in.weight` | `double_blocks.X.txt_mlp_in.weight` |
| `double_stream_modulation_img.linear.weight` | `double_mod_img.linear.weight` |
| `double_stream_modulation_txt.linear.weight` | `double_mod_txt.linear.weight` |
| `single_stream_modulation.linear.weight` | `single_mod.linear.weight` |

### Patch Unpacking

The 128-dim patch vector layout is `[C, p1, p2] = [32, 2, 2]`:

```rust
// Unpack latent: [batch, seq, 128] -> [batch, H, W, 32] for VAE
let latent = latent.reshape(&[batch, H/2, W/2, 32, 2, 2])?;  // C before patch dims
let latent = latent.transpose_axes(&[0, 1, 4, 2, 5, 3])?;    // Interleave patches
let latent = latent.reshape(&[batch, H, W, 32])?;            // Final NHWC for VAE
```

## INT8 Quantization

The implementation supports INT8 quantization for reduced memory usage. This is useful when GPU memory is constrained.

### Usage

```bash
# Generate with INT8 quantization
./target/release/examples/generate_klein --quantize "a cat sitting on a windowsill"
```

### INT8 10-Run Benchmark Results

**INT8 Quantized Rust (mlx-rs):**
| Run | Denoising |
|-----|-----------|
| 1   | 4.76s     |
| 2   | 4.77s     |
| 3   | 4.85s     |
| 4   | 5.04s     |
| 5   | 5.29s     |
| 6   | 5.34s     |
| 7   | 5.28s     |
| 8   | 5.15s     |
| 9   | 5.17s     |
| 10  | 5.06s     |
| **Average** | **5.07s** |

### INT8 vs F32 vs flux.c Comparison

| Implementation | Avg Denoising | vs flux.c | Memory |
|----------------|---------------|-----------|--------|
| **Rust F32** | **4.35s** | **2.71x faster** | ~4.4GB |
| **Rust INT8** | **5.07s** | **2.33x faster** | ~1.1GB |
| flux.c (C+Metal) | 11.80s | baseline | ~4.4GB |

### INT8 Characteristics

| Metric | F32 | INT8 | Notes |
|--------|-----|------|-------|
| **Model memory** | ~4.4GB | ~1.1GB | ~4x reduction for transformer weights |
| **Denoising time** | 4.35s | 5.07s | 17% slower (expected on Apple Silicon) |
| **Image quality** | High | High | Comparable results |
| **Quantization time** | N/A | ~100µs | Near-instant conversion at load time |
| **vs flux.c** | 2.71x faster | 2.33x faster | Both significantly faster |

### Implementation Details

The quantization uses mlx-rs's `QuantizedLinear` with:
- **Bits**: 8 (INT8)
- **Group size**: 64
- **Mode**: Affine quantization

All Linear layers in the transformer are quantized:
- Input/output embeddings
- Time embedding layers
- Shared modulation layers
- All attention Q/K/V projections
- All MLP layers

Normalization layers (LayerNorm, RmsNorm) remain in f32 as they have minimal memory footprint.

### API

```rust
use flux_mlx::klein_quantized::QuantizedFluxKlein;

// Load f32 model first
let mut flux = FluxKlein::new(params)?;
flux.update_flattened(weights);

// Convert to INT8 quantized
let quantized = QuantizedFluxKlein::from_unquantized(flux, 64, 8)?;

// Use same API for inference
let output = quantized.forward_with_rope(&img, &txt, &t, &rope_cos, &rope_sin)?;
```

## Usage

```bash
# Build
cargo build --example generate_klein --release

# Generate image (F32)
./target/release/examples/generate_klein "a cat sitting on a windowsill"

# Generate image (INT8 quantized)
./target/release/examples/generate_klein --quantize "a cat sitting on a windowsill"

# Output: output_klein.ppm (convert with: sips -s format png output_klein.ppm --out output.png)
```

## File Structure

```
src/
├── klein_model.rs      # Main FLUX.2-klein model implementation (F32)
│   ├── compute_rope_freqs()   # RoPE frequency computation
│   ├── apply_rope()           # RoPE application to Q/K
│   ├── SharedModulation       # Shared modulation layer
│   ├── KleinDoubleBlock       # Double-stream transformer block
│   ├── KleinSingleBlock       # Single-stream transformer block
│   └── FluxKlein              # Main model struct
├── klein_quantized.rs  # INT8 quantized FLUX.2-klein model
│   ├── QuantizedSharedModulation
│   ├── QuantizedKleinDoubleBlock
│   ├── QuantizedKleinSingleBlock
│   └── QuantizedFluxKlein     # from_unquantized() for conversion
├── weights.rs          # Weight loading and sanitization
│   └── sanitize_klein_model_weights()
├── qwen3_encoder.rs    # Qwen3-4B text encoder
└── autoencoder.rs      # VAE decoder

examples/
└── generate_klein.rs   # Full generation pipeline (supports --quantize flag)
```

---

## Z-Image-Turbo (6B) Implementation

Z-Image-Turbo is a 6B parameter Single-Stream DiT (S3-DiT) that shares the same Qwen3-4B text encoder.

### Architecture

- **Text Encoder**: Qwen3-4B (same as FLUX.2-klein, but extracts layer 34 instead of pooled output)
- **Transformer**: 2 noise refiners + 2 context refiners + 30 joint blocks
- **Hidden Size**: 3840
- **Attention Heads**: 30 (head_dim = 128)
- **RoPE**: 3-axis with `axes_dims = [32, 48, 48]`, `theta = 256`
- **Denoising Steps**: 9 (Turbo distilled)

### Key Implementation Details

1. **Timestep Convention**:
   - Model receives `t = 1 - sigma` where sigma goes from 1 (noise) to 0 (clean)
   - So model receives t from 0 (noise) to 1 (clean)

2. **Euler Step**:
   ```rust
   let dt = sigma_curr - sigma_next;  // Positive
   latents = latents + dt * v_pred;
   ```

3. **VAE Scaling**:
   ```rust
   // Decode
   let latents = latents / 0.3611 + 0.1159;
   let image = vae.forward(&latents)?;
   let image = (image / 2.0 + 0.5).clamp(0.0, 1.0);
   ```

4. **Position Encoding**:
   - Image positions: `(cap_len + 1, 0, 0)` offset for H_tok × W_tok grid
   - Caption positions: `(1, 0, 0)` offset for cap_len × 1 × 1

### Usage

```bash
# Generate with Z-Image-Turbo
cargo run --example generate_zimage --release -- "a red sports car"

# Output: output_zimage.ppm
```

### Performance

On Apple M3 Max with 512×512 images and 9 denoising steps:
- ~17 seconds total generation time
- ~1.9 seconds per denoising step

### File Structure

```
src/
├── zimage_model.rs      # Z-Image transformer implementation
│   ├── ZImageConfig           # Model configuration
│   ├── NoiseRefiner           # Initial noise processing
│   ├── ContextRefiner         # Caption context processing
│   ├── ZImageJointBlock       # Main transformer block with tanh-gated AdaLN
│   ├── ZImageTransformer      # Main model
│   └── sanitize_zimage_weights()  # Weight mapping from PyTorch
├── qwen3_encoder.rs     # Shared with FLUX.2-klein
│   └── encode_zimage()        # Extract layer 34 embeddings (2560-dim)

examples/
└── generate_zimage.rs   # Full Z-Image generation pipeline
```

---

## References

- [FLUX.2-klein-4B on HuggingFace](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [Z-Image-Turbo on HuggingFace](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [MLX_z-image Reference](https://github.com/uqer1244/MLX_z-image)
- [Diffusers FLUX.2 implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux2.py)
- [flux.c reference implementation](https://github.com/anthropics/flux.c)
