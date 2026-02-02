# Z-Image-Turbo Implementation Plan

This document describes the plan for porting Z-Image-Turbo to Rust using mlx-rs, leveraging the existing FLUX.2-klein implementation.

## Model Overview

**Z-Image-Turbo** is a 6B parameter Single-Stream DiT (S3-DiT) optimized for fast image generation on Apple Silicon.

| Property | Z-Image-Turbo |
|----------|---------------|
| Parameters | 6B |
| Architecture | Single-Stream DiT (S3-DiT) |
| Text Encoder | Qwen3-4B (2560 hidden, 36 layers) |
| Steps | 9 (Turbo distilled) |
| Resolution | 1024x1024 default |

## Architecture Comparison

### Text Encoder (100% Reusable)

Both models use identical Qwen3-4B text encoders:

| Parameter | FLUX.2-klein | Z-Image-Turbo |
|-----------|--------------|---------------|
| Architecture | Qwen3ForCausalLM | Qwen3ForCausalLM |
| Hidden Size | 2560 | 2560 |
| Layers | 36 | 36 |
| Attention Heads | 32 | 32 |
| KV Heads | 8 | 8 |
| Head Dim | 128 | 128 |
| Intermediate | 9728 | 9728 |
| Vocab | 151936 | 151936 |
| RoPE Theta | 1,000,000 | 1,000,000 |

**Key Difference**: Embedding extraction method
- FLUX.2-klein: Concat layers 8, 17, 26 → 7680-dim
- Z-Image: Take layer 34 (second-to-last) → 2560-dim

### Transformer Architecture

Actual Z-Image-Turbo config from `transformer/config.json`:

| Component | FLUX.2-klein | Z-Image-Turbo |
|-----------|--------------|---------------|
| Block Structure | Double + Single | Refiner + Refiner + Joint |
| Double Blocks | 5 | N/A |
| Single Blocks | 20 | N/A |
| Noise Refiner Blocks | N/A | 2 (`n_refiner_layers`) |
| Context Refiner Blocks | N/A | 2 (`n_refiner_layers`) |
| Joint Blocks | N/A | 30 (`n_layers`) |
| Hidden Size | 3072 | 3840 (`dim`) |
| Num Heads | 24 | 30 (`n_heads`) |
| KV Heads | 24 | 30 (`n_kv_heads`) |
| In Channels | 128 | 16 (`in_channels`) |
| Cap Feat Dim | 7680 | 2560 (`cap_feat_dim`) |
| FFN | SwiGLU | SwiGLU |
| Modulation | Shared AdaLN | Per-block AdaLN with tanh gates |
| QK Norm | RmsNorm | RmsNorm (`qk_norm: true`) |
| T Scale | 1000 | 1000 (`t_scale`) |

### Position Encoding (RoPE)

From Z-Image config (`axes_dims`, `axes_lens`, `rope_theta`):

| Property | FLUX.2-klein | Z-Image-Turbo |
|----------|--------------|---------------|
| Axes | 4-axis (32,32,32,32) | 3-axis (32,48,48) |
| Axes Lens | N/A | (1536, 512, 512) |
| Position Grid | (T, H, W, L) | (H, W, T) coordinate grid |
| Theta | 2000 | 256 (`rope_theta`) |
| Head Dim | 128 (32*4) | 128 (32+48+48) |

### Scheduler

| Property | FLUX.2-klein | Z-Image-Turbo |
|----------|--------------|---------------|
| Type | Flow Matching Euler | Flow Matching Euler |
| Steps | 4 | 9 |
| Dynamic Shift | Yes (SNR-based mu) | Yes (calculate_shift function) |

## Component Reusability Matrix

| Component | Reuse Level | Notes |
|-----------|-------------|-------|
| `Qwen3TextEncoder` | 100% | Add `encode_zimage()` for layer-34 extraction |
| `RmsNorm` / `LayerNorm` | 100% | Identical implementation |
| `SwiGLU MLP` | 100% | Same `silu(w1) * w3` pattern |
| `TimestepEmbedder` | 100% | Identical sinusoidal embedding |
| `Attention (SDPA)` | 90% | Same core, different RoPE |
| `VAE Decoder` | 100% | Same architecture |
| `Flow Matching Scheduler` | 95% | Same Euler method |
| `Safetensors Loading` | 100% | Same format |
| `RoPE` | 70% | Need 3-axis variant |
| `Position Encoding` | 50% | Different grid structure |
| `Modulation/AdaLN` | 60% | Z-Image uses tanh gates |

## Implementation Plan

### Phase 1: Text Encoder Modification

Add alternative extraction mode to existing `Qwen3TextEncoder`:

```rust
// In qwen3_encoder.rs
pub enum EmbeddingMode {
    FluxKlein,  // Concat layers 8, 17, 26 → 7680-dim
    ZImage,     // Layer 34 (second-to-last) → 2560-dim
}

impl Qwen3TextEncoder {
    pub fn encode_zimage(&mut self, input_ids: &Array, attention_mask: Option<&Array>)
        -> Result<Array, Exception>
    {
        let mut h = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            h = layer.forward(&h, attention_mask)?;
        }
        // Return second-to-last hidden state (layer 34, before final norm)
        Ok(h)
    }
}
```

**Files to modify**: `src/qwen3_encoder.rs`

### Phase 2: 3-Axis RoPE Implementation

Add 3-axis RoPE variant alongside existing 4-axis:

```rust
// In new file src/zimage_model.rs or add to klein_model.rs

/// Compute 3-axis RoPE frequencies for Z-Image
/// axes_dim = [32, 48, 48] for (H, W, T)
pub fn compute_rope_3axis(
    positions: &Array,  // [batch, seq, 3] for (h, w, t)
    axes_dim: &[i32],   // [32, 48, 48]
    theta: f32,         // 256.0
) -> Result<(Array, Array), Exception> {
    // Similar to klein_model::compute_rope_freqs but with 3 axes
}

/// Create coordinate grid for Z-Image position encoding
pub fn create_coordinate_grid(
    size: (i32, i32, i32),   // (d0, d1, d2)
    start: (i32, i32, i32),  // (s0, s1, s2)
) -> Array {
    // Matches MLX_z-image's create_coordinate_grid function
}
```

### Phase 3: Z-Image Transformer Block

Create new transformer block with Z-Image architecture:

```rust
// src/zimage_model.rs

/// Z-Image transformer block with tanh-gated modulation
pub struct ZImageTransformerBlock {
    attention: Attention,
    feed_forward: FeedForward,
    attention_norm1: RmsNorm,
    ffn_norm1: RmsNorm,
    attention_norm2: RmsNorm,
    ffn_norm2: RmsNorm,
    adaLN_modulation: Option<Linear>,  // None for context refiner
}

impl ZImageTransformerBlock {
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        positions: &Array,
        adaln_input: Option<&Array>,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array, Exception> {
        if let Some(adaln) = adaln_input {
            // Modulated path (noise refiner + joint blocks)
            let chunks = self.adaLN_modulation.forward(adaln)?;
            let (scale_msa, gate_msa, scale_mlp, gate_mlp) = split_4(chunks)?;

            // Key difference: tanh gates
            let norm_x = self.attention_norm1.forward(x)? * (1 + scale_msa);
            let attn_out = self.attention.forward(&norm_x, mask, cos, sin)?;
            let x = x + ops::tanh(&gate_msa)? * self.attention_norm2.forward(&attn_out)?;

            let norm_ffn = self.ffn_norm1.forward(&x)? * (1 + scale_mlp);
            let x = x + ops::tanh(&gate_mlp)? * self.ffn_norm2.forward(&self.feed_forward.forward(&norm_ffn)?)?;
            Ok(x)
        } else {
            // Unmodulated path (context refiner)
            let x = x + self.attention_norm2.forward(&self.attention.forward(&self.attention_norm1.forward(x)?, mask, cos, sin)?)?;
            let x = x + self.ffn_norm2.forward(&self.feed_forward.forward(&self.ffn_norm1.forward(&x)?)?)?;
            Ok(x)
        }
    }
}
```

### Phase 4: Main Z-Image Transformer

```rust
// src/zimage_model.rs

pub struct ZImageTransformer {
    t_scale: f32,
    t_embedder: TimestepEmbedder,
    x_embedder: Linear,
    cap_embedder: Sequential,  // RmsNorm + Linear
    final_layer: FinalLayer,

    x_pad_token: Array,
    cap_pad_token: Array,

    noise_refiner: Vec<ZImageTransformerBlock>,    // With modulation
    context_refiner: Vec<ZImageTransformerBlock>,  // Without modulation
    layers: Vec<ZImageTransformerBlock>,           // Joint blocks with modulation
}

impl ZImageTransformer {
    pub fn forward(
        &mut self,
        x: &Array,           // [batch, img_seq, in_channels*4]
        t: &Array,           // [batch]
        cap_feats: &Array,   // [batch, cap_seq, cap_dim]
        x_pos: &Array,       // [batch, img_seq, 3]
        cap_pos: &Array,     // [batch, cap_seq, 3]
        cos: &Array,
        sin: &Array,
    ) -> Result<Array, Exception> {
        let temb = self.t_embedder.forward(&(t * self.t_scale))?;
        let x = self.x_embedder.forward(x)?;
        let cap_feats = self.cap_embedder.forward(cap_feats)?;

        // Noise refiner: process image tokens
        for layer in &mut self.noise_refiner {
            x = layer.forward(&x, None, x_pos, Some(&temb), cos_img, sin_img)?;
        }

        // Context refiner: process text tokens (no modulation)
        for layer in &mut self.context_refiner {
            cap_feats = layer.forward(&cap_feats, None, cap_pos, None, cos_txt, sin_txt)?;
        }

        // Joint blocks: unified sequence
        let unified = ops::concatenate(&[x, cap_feats], 1)?;
        for layer in &mut self.layers {
            unified = layer.forward(&unified, None, unified_pos, Some(&temb), cos, sin)?;
        }

        // Extract image tokens and apply final layer
        let img_out = unified.slice(..x_seq)?;
        self.final_layer.forward(&img_out, &temb)
    }
}
```

### Phase 5: Pipeline Integration

```rust
// src/zimage_pipeline.rs

pub struct ZImagePipeline {
    text_encoder: Qwen3TextEncoder,
    transformer: ZImageTransformer,
    vae: Decoder,
    scheduler: FlowMatchEulerScheduler,
}

impl ZImagePipeline {
    pub fn generate(
        &mut self,
        prompt: &str,
        width: i32,
        height: i32,
        steps: i32,
        seed: Option<u64>,
    ) -> Result<Array, Exception> {
        // 1. Encode text (using layer 34)
        let cap_feats = self.text_encoder.encode_zimage(&input_ids, attention_mask)?;

        // 2. Pad caption features to multiple of 32
        let cap_feats = pad_to_multiple(&cap_feats, 32)?;

        // 3. Prepare position encodings
        let (img_pos, cap_pos) = create_positions(height, width, cap_len)?;
        let (cos, sin) = compute_rope_3axis(&unified_pos, &[32, 48, 48], 256.0)?;

        // 4. Initialize latents
        let latents = random::normal(&[batch, img_seq, in_channels])?;

        // 5. Denoising loop
        let mu = calculate_shift(img_seq_len)?;
        self.scheduler.set_timesteps(steps, mu)?;

        for i in 0..steps {
            let t = 1.0 - self.scheduler.timesteps[i];
            let noise_pred = self.transformer.forward(&latents, &t, &cap_feats, ...)?;
            latents = self.scheduler.step(&noise_pred, i, &latents)?;
        }

        // 6. Decode to image
        self.vae.forward(&unpack_latents(latents)?)
    }
}
```

## File Structure

```
src/
├── qwen3_encoder.rs      # Add encode_zimage() method
├── klein_model.rs        # Existing FLUX.2-klein (reference)
├── zimage_model.rs       # NEW: Z-Image transformer
│   ├── compute_rope_3axis()
│   ├── create_coordinate_grid()
│   ├── ZImageTransformerBlock
│   ├── FinalLayer
│   └── ZImageTransformer
├── zimage_pipeline.rs    # NEW: Full pipeline
│   ├── calculate_shift()
│   ├── FlowMatchEulerScheduler
│   └── ZImagePipeline
├── weights.rs            # Add sanitize_zimage_weights()
└── lib.rs                # Export new modules

examples/
└── generate_zimage.rs    # NEW: Example usage
```

## Weight Mapping

| MLX Python Key | Rust Key |
|----------------|----------|
| `t_embedder.linear1.weight` | `t_embedder.linear1.weight` |
| `t_embedder.linear2.weight` | `t_embedder.linear2.weight` |
| `x_embedder.weight` | `x_embedder.weight` |
| `cap_embedder.layers.0.weight` | `cap_embedder.norm.weight` |
| `cap_embedder.layers.1.weight` | `cap_embedder.linear.weight` |
| `noise_refiner.X.attention.to_q.weight` | `noise_refiner.X.attention.to_q.weight` |
| `context_refiner.X.attention.to_q.weight` | `context_refiner.X.attention.to_q.weight` |
| `layers.X.attention.to_q.weight` | `layers.X.attention.to_q.weight` |
| `final_layer.linear.weight` | `final_layer.linear.weight` |
| `final_layer.adaLN_modulation.layers.1.weight` | `final_layer.adaLN_modulation.linear.weight` |

## Development Timeline

| Phase | Task | Estimated LOC |
|-------|------|---------------|
| 1 | Text encoder modification | ~30 |
| 2 | 3-axis RoPE implementation | ~80 |
| 3 | ZImageTransformerBlock | ~150 |
| 4 | ZImageTransformer | ~200 |
| 5 | Pipeline + scheduler | ~150 |
| 6 | Weight loading | ~100 |
| 7 | Example + testing | ~200 |
| **Total** | | **~910** |

## Model Files

Download from HuggingFace:
```bash
huggingface-cli download uqer1244/MLX-z-image --local-dir ./models/zimage-turbo-mlx
```

Files needed:
- `transformer/model.safetensors` (sharded)
- `text_encoder/model.safetensors` (4-bit quantized)
- `vae/diffusion_pytorch_model.safetensors`
- `tokenizer/tokenizer.json`

## Quantization Support

The Rust implementation supports both dequantized (f32) and native 4-bit quantized inference.

### Two Loading Modes

#### 1. Dequantized Mode (Default) - Maximum Speed

Loads 4-bit weights from MLX format and converts to f32 at startup.

```rust
use flux_mlx::{ZImageTransformer, ZImageConfig, load_safetensors, sanitize_mlx_weights};

let config = ZImageConfig::default();
let mut transformer = ZImageTransformer::new(config)?;

// Load and dequantize weights
let weights = load_safetensors("transformer/model.safetensors")?;
let weights = sanitize_mlx_weights(weights);
// Dequantize 4-bit to f32
let weights: HashMap<String, Array> = weights
    .into_iter()
    .map(|(k, v)| (k, v.as_type::<f32>().unwrap_or(v)))
    .collect();
transformer.update_flattened(weights);
```

#### 2. Quantized Mode - Minimum Memory

Keeps 4-bit weights and uses `quantized_matmul` for inference.

```rust
use flux_mlx::{load_quantized_zimage_transformer, ZImageConfig, load_safetensors};

let config = ZImageConfig::default();
let weights = load_safetensors("transformer/model.safetensors")?;
let mut transformer = load_quantized_zimage_transformer(weights, config)?;
```

### Performance Comparison

Benchmarked on Apple M3 Max, 512x512 image, 9 denoising steps:

| Implementation | Avg Step Time | Memory Usage | Total (9 steps) |
|---------------|---------------|--------------|-----------------|
| **Rust Dequantized (f32)** | 1.87s | ~12GB | 16.8s |
| **Python MLX Quantized** | 1.98s | ~3GB | 17.8s |
| **Rust Quantized (4-bit)** | 2.08s | ~3GB | 18.7s |

### When to Use Each Mode

**Use Dequantized (f32)** when:
- You have sufficient memory (12GB+ available)
- Maximum inference speed is the priority
- Running on high-memory Apple Silicon (M1 Max/Ultra, M2 Max/Ultra, M3 Max/Ultra)

**Use Quantized (4-bit)** when:
- Memory is constrained (8GB or less available)
- Running alongside other memory-intensive applications
- Willing to trade ~11% speed for 4x memory reduction
- Running on base M1/M2/M3 with 8GB unified memory

### Why Quantized is Slower

1. **Dequantized approach**: Load 4-bit → convert to f32 once at startup → use highly optimized f32 matmul kernels on Apple Silicon

2. **Quantized approach**: Keep 4-bit → `quantized_matmul` unpacks values during every forward pass → overhead from runtime unpacking

The main benefit of quantization is **memory efficiency**, not speed. Apple Silicon's f32 matmul is highly optimized, while quantized matmul has unpacking overhead.

### Implementation Files

| File | Description |
|------|-------------|
| `src/zimage_model.rs` | Dequantized transformer (f32 weights) |
| `src/zimage_model_quantized.rs` | Quantized transformer (4-bit weights) |
| `examples/generate_zimage.rs` | Full generation with dequantized model |
| `examples/generate_zimage_quantized.rs` | Benchmark with quantized model |

### Quantized Weight Format

MLX 4-bit quantization stores weights as:

```
layer.weight  -> uint32, packed 4-bit values [out_dim, in_dim/8]
layer.scales  -> float16, per-group scales [out_dim, in_dim/group_size]
layer.biases  -> float16, per-group biases [out_dim, in_dim/group_size]
layer.bias    -> float16, optional linear bias [out_dim] (different from biases!)
```

With `group_size=32` and `bits=4`, each uint32 packs 8 weight values.

### Memory Breakdown

| Component | Dequantized (f32) | Quantized (4-bit) |
|-----------|-------------------|-------------------|
| Transformer weights | ~11GB | ~2.8GB |
| Activations (inference) | ~1GB | ~1GB |
| **Total** | **~12GB** | **~3.8GB** |

## References

- [MLX_z-image Repository](https://github.com/uqer1244/MLX_z-image)
- [Z-Image-Turbo on HuggingFace](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [MLX-z-image Quantized Weights](https://huggingface.co/uqer1244/MLX-z-image)
