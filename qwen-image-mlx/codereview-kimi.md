# Qwen-Image-MLX Critical Code Review & Performance Enhancement Plan

**Date**: 2026-01-29
**Reviewer**: Claude Code (kimi-k2.5)
**Scope**: Full codebase review with architecture analysis and performance roadmap

---

## Executive Summary

The `qwen-image-mlx` crate is a Rust implementation of the Qwen-Image-2512 text-to-image diffusion model using Apple's MLX framework. The codebase successfully implements the model with correct algorithms but requires significant cleanup and optimization before production deployment.

**Overall Assessment**: Functional but needs refinement
- **Architecture**: Sound
- **Code Quality**: Needs cleanup (debug artifacts, duplication)
- **Performance**: Good foundation, significant headroom for optimization
- **Safety**: No unsafe code, proper error handling

---

## 1. Architecture Overview

### 1.1 High-Level Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌─────────────┐
│  Text Prompt    │────▶│  QwenTextEncoder │────▶│  Quantized      │────▶│  QwenVAE    │────▶ RGB Image
│  (Tokenizer)    │     │  (28 layers GQA) │     │  Transformer    │     │  Decoder    │
└─────────────────┘     └──────────────────┘     │  (60 blocks)    │     └─────────────┘
                                                  │  Flow Matching  │
                                                  └─────────────────┘
```

### 1.2 Component Breakdown

| Component | Location | Purpose | Lines |
|-----------|----------|---------|-------|
| **Text Encoder** | `src/text_encoder.rs` | Qwen2.5-VL style 28-layer transformer with GQA | 578 |
| **Quantized Transformer** | `src/qwen_quantized.rs` | Main 60-block DiT with 4/8-bit quantization | 1096 |
| **Full-Precision Transformer** | `src/qwen_full_precision.rs` | FP32/BF16 variant for training | - |
| **VAE** | `src/vae/vae.rs` | 3D causal convolutional encoder/decoder | 316 |
| **Pipeline** | `src/pipeline.rs` | Flow-match Euler scheduler + generation loop | 268 |
| **Weight Loading** | `src/weights.rs` | SafeTensors loading with name mapping | 207 |

### 1.3 Model Specifications

**Text Encoder** (`src/text_encoder.rs:17-44`):
- Hidden Size: 3584
- Layers: 28
- Query Heads: 28, KV Heads: 4 (GQA)
- Head Dim: 128
- Vocab Size: 152064

**Diffusion Transformer** (`src/qwen_quantized.rs:18-45`):
- Layers: 60 transformer blocks
- Inner Dim: 3072 (24 heads × 128 head_dim)
- Patch Size: 2×2
- Quantization: 4-bit or 8-bit
- Joint image-text attention

**VAE** (`src/vae/vae.rs:30-33`):
- Base Channels: 96
- Stage Multipliers: [1, 1, 2, 4, 4]
- Latent Channels: 16
- Downsampling: 8×

---

## 2. Critical Issues

### 2.1 Code Duplication (HIGH PRIORITY)

**Problem**: Two parallel transformer implementations exist with significant duplication.

| Component | Quantized Location | Full-Precision Location |
|-----------|-------------------|------------------------|
| FeedForward | `src/qwen_quantized.rs:71-124` | `src/transformer/feedforward.rs` |
| Attention | `src/qwen_quantized.rs:126-336` | `src/transformer/attention.rs` |
| TransformerBlock | `src/qwen_quantized.rs:338-362` | `src/transformer/block.rs` |

**Impact**: Maintenance burden, risk of divergence between implementations.

**Recommendation**: Extract common traits or use generic types to share logic:
```rust
pub trait TransformerBlock {
    type Linear: LinearLayer;
    fn new(dim: i32, num_heads: i32) -> Self;
    fn forward(&mut self, x: &Array, text: &Array) -> Result<(Array, Array), Exception>;
}
```

### 2.2 Debug Code in Production (HIGH PRIORITY)

**Problem**: Multiple static atomic flags control debug printing throughout the codebase.

**Locations Found**:
- `src/qwen_quantized.rs:93-102` - DEBUG_FFN
- `src/qwen_quantized.rs:220-243` - DEBUG_BEFORE_NORM
- `src/qwen_quantized.rs:252-260` - DEBUG_NORM
- `src/qwen_quantized.rs:273-287` - DEBUG_QK
- `src/qwen_quantized.rs:386-394` - DEBUG_BLOCK_INPUT
- `src/qwen_quantized.rs:397-405` - DEBUG_TEMB
- `src/qwen_quantized.rs:412-419` - DEBUG_MOD
- `src/qwen_quantized.rs:432-439` - DEBUG_IMG_NORMED
- `src/qwen_quantized.rs:444-451` - DEBUG_IMG_MODULATED
- `src/qwen_quantized.rs:458-469` - DEBUG_GATE
- `src/qwen_quantized.rs:481-490` - DEBUG_ATTN
- `src/qwen_quantized.rs:555-610` - DEBUG_TS
- `src/qwen_quantized.rs:775-803` - DEBUG_IMG_IN, DEBUG_TXT_RAW, DEBUG_TXT_NORMED
- `src/qwen_quantized.rs:850-861` - DEBUG_BLOCK0
- `src/qwen_quantized.rs:865-886` - DEBUG_PRE_NORM, DEBUG_POST_NORM
- `src/qwen_quantized.rs:891-898` - DEBUG_FINAL
- `src/qwen_quantized.rs:941-951` - DEBUG_MODULATE

**Example**:
```rust
static DEBUG_FFN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
let debug_ffn = !DEBUG_FFN.swap(true, std::sync::atomic::Ordering::SeqCst);
if debug_ffn {
    mlx_rs::transforms::eval([x, &hidden]).ok();
    eprintln!("[DEBUG FFN] input: [{:.2}, {:.2}]...", ...);
}
```

**Impact**:
- Runtime overhead from atomic operations on hot paths
- Console noise in production builds
- Unprofessional output for end users

**Recommendation**:
1. Remove all debug printing infrastructure
2. Use proper logging crate (e.g., `tracing` or `log`) with compile-time filters
3. Gate debug code behind `#[cfg(debug_assertions)]` or feature flags

### 2.3 Dead Code and Unused Imports (MEDIUM PRIORITY)

**File**: `src/pipeline.rs:11-13`
```rust
use crate::transformer::QwenTransformer;  // UNUSED - only uses quantized version
use crate::vae::QwenVAE;
```

The `QwenImagePipeline` struct at `src/pipeline.rs:82-86` uses the full-precision `QwenTransformer`, but the example at `examples/generate_qwen_image.rs:200` only uses `QwenQuantizedTransformer`. The pipeline is essentially orphaned code.

**Recommendation**: Either integrate `QwenTransformer` properly or remove it and consolidate on the quantized version.

### 2.4 Hardcoded Magic Numbers (MEDIUM PRIORITY)

**File**: `examples/generate_qwen_image.rs:380`
```rust
let template = "<|im_start|>system\nDescribe the image...";
let drop_idx = 34;  // Assumes template is exactly 34 tokens
```

**File**: `examples/generate_qwen_image.rs:393`
```rust
let max_input_len = 77 + 34;  // Magic numbers without explanation
```

**Recommendation**: Define constants with documentation:
```rust
const TEMPLATE_TOKEN_COUNT: usize = 34;
const MAX_OUTPUT_TOKENS: usize = 77;
const MAX_INPUT_TOKENS: usize = MAX_OUTPUT_TOKENS + TEMPLATE_TOKEN_COUNT;
```

### 2.5 Disabled Code Blocks (MEDIUM PRIORITY)

**File**: `src/qwen_quantized.rs:221`
```rust
if false {  // Entire debug block disabled but present
    mlx_rs::transforms::eval([&img_q, &txt_q]).ok();
    // ... 20 lines of dead code
}
```

Multiple `if false` blocks exist throughout the codebase (lines 221, 253, 274, 398, etc.).

**Recommendation**: Remove or convert to proper feature flags:
```rust
#[cfg(feature = "debug-attention")]
{
    // Debug code here
}
```

### 2.6 Inefficient Weight Mapping (LOW PRIORITY)

**File**: `src/weights.rs:53-91`
```rust
impl TransformerWeightMapper {
    pub fn map_name(hf_name: &str) -> String {
        let mut name = hf_name.to_string();
        name = name.replace("transformer_blocks.", "transformer_blocks.");  // No-op!
        name = name.replace("to_q.weight", "to_q.weight");  // No-op!
        // ... many no-op replacements
    }
}
```

The weight mapper performs many no-op string replacements.

**Recommendation**: Replace with actual mappings or remove no-ops:
```rust
static WEIGHT_MAPPINGS: &[(&str, &str)] = &[
    (".attn1.", ".attn."),
    ("to_out.0.", "attn_to_out."),
    ("ff.net.0.proj.", "mlp_in."),
    // ... actual mappings only
];
```

---

## 3. Architecture Strengths

### 3.1 Proper Quantization Support

The quantized transformer at `src/qwen_quantized.rs:708-733` correctly uses `QuantizedLinear` with configurable bits (4/8) and group size:

```rust
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenQuantizedTransformer {
    #[param]
    pub img_in: QuantizedLinear,
    #[param]
    pub transformer_blocks: Vec<QwenTransformerBlock>,
    // ... 60 transformer blocks
}
```

### 3.2 Correct VAE Latent Normalization

The VAE properly handles pre-computed normalization constants at `src/vae/vae.rs:18-28`:

```rust
pub const LATENTS_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, ...
];
pub const LATENTS_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, ...
];
```

### 3.3 Flow Matching Implementation

The scheduler at `src/pipeline.rs:14-79` correctly implements flow matching with time shifting:

```rust
pub fn step(&self, model_output: &Array, timestep_idx: usize, sample: &Array)
    -> Result<Array, Exception> {
    let dt = self.sigmas[timestep_idx + 1] - self.sigmas[timestep_idx];
    ops::add(sample, &ops::multiply(model_output, &Array::from_f32(dt))?)
}
```

### 3.4 Batched CFG Support

The example at `examples/generate_qwen_image.rs:537-604` implements an optimization for classifier-free guidance that batches conditional and unconditional passes when possible, providing 2x speedup for the forward pass.

### 3.5 Correct 3D Convolution Implementation

The VAE causal convolution at `src/vae/conv3d.rs:62-87` properly handles the NCTHW ↔ NTHWC transpositions required by MLX's conv3d:

```rust
// Transpose from NCTHW to NTHWC
let input = padded.transpose_axes(&[0, 2, 3, 4, 1])?;
// Transpose weight from [out, in, kT, kH, kW] to [out, kT, kH, kW, in]
let weight = self.weight.transpose_axes(&[0, 2, 3, 4, 1])?;
```

---

## 4. Detailed Component Analysis

### 4.1 Text Encoder (`src/text_encoder.rs`)

**Architecture**: 28-layer transformer with Grouped Query Attention (GQA)

**Strengths**:
- Correct RoPE implementation at `src/text_encoder.rs:123-184`
- Proper causal masking with padding support at `src/text_encoder.rs:379-414`
- SwiGLU activation in MLP at `src/text_encoder.rs:289-313`

**Issues**:
- Custom `Linear` implementation at `src/text_encoder.rs:72-103` duplicates `mlx_rs::nn::Linear`
- `repeat_kv` at `src/text_encoder.rs:272-287` could use MLX's built-in broadcast

### 4.2 Quantized Transformer (`src/qwen_quantized.rs`)

**Architecture**: 60-block DiT (Diffusion Transformer) with joint image-text attention

**Block Structure** (per `src/qwen_quantized.rs:338-362`):
1. Image modulation linear (6x dim for shift/scale/gate)
2. Text modulation linear (6x dim for shift/scale/gate)
3. Joint attention (QKV for both streams, concatenated)
4. Image FFN (4x expansion)
5. Text FFN (4x expansion)

**Key Algorithm - Modulation** (`src/qwen_quantized.rs:933-959`):
```rust
fn modulate(x: &Array, mod_params: &Array) -> Result<(Array, Array), Exception> {
    let shift = mod_params.index((.., ..dim));
    let scale = mod_params.index((.., dim..dim*2));
    let gate = mod_params.index((.., dim*2..));
    // (1 + scale) * x + shift
}
```

**Issues**:
- Comment at `src/qwen_quantized.rs:917` claims "LayerNorm" but code implements correct LayerNorm, not RMSNorm
- `clip_values` function at `src/qwen_quantized.rs:926-931` is defined but never used

### 4.3 VAE (`src/vae/vae.rs`)

**Architecture**: 3D causal convolutional autoencoder

**Encoder**:
```
Input [B, 4, 1, H, W] ──▶ Conv3D(4→96) ──▶ DownBlock x4 ──▶ MidBlock ──▶ [B, 32, 1, H/8, W/8]
```

**Decoder**:
```
Input [B, 16, 1, H/8, W/8] ──▶ Conv3D(16→384) ──▶ MidBlock ──▶ UpBlock x4 ──▶ [B, 3, 1, H, W]
```

**Building Blocks** (`src/vae/blocks.rs`):
- `QwenImageResBlock3D`: Residual block with RMSNorm, SiLU, causal conv3d
- `QwenImageMidBlock3D`: Mid-block with interleaved attention
- `QwenImageUpBlock3D`: Upsampling block for decoder
- `QwenImageDownBlock3D`: Downsampling block for encoder

### 4.4 Attention Implementation

**Current** (`src/qwen_quantized.rs:294-318`):
```rust
let q = joint_q.transpose_axes(&[0, 2, 1, 3])?;
let k = joint_k.transpose_axes(&[0, 2, 1, 3])?;
let v = joint_v.transpose_axes(&[0, 2, 1, 3])?;
let scale = 1.0 / (self.head_dim as f32).sqrt();
let attn_scores = ops::matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?;
let mut attn_scores = ops::multiply(&attn_scores, &Array::from_f32(scale))?;
// ... masking ...
let attn = mlx_rs::ops::softmax_axis(&attn_scores, -1, None)?;
let out = ops::matmul(&attn, &v)?;
```

This materializes the full O(N²) attention matrix, which is memory-intensive.

---

## 5. Safety and Correctness

### 5.1 Unsafe Code
**Status**: None found in the reviewed files. All operations use safe MLX Rust bindings.

### 5.2 Error Handling
- Good use of `thiserror` for error types at `src/error.rs:6-24`
- Proper propagation with `?` operator throughout
- Custom error type `QwenImageError` covers all failure modes

### 5.3 Numerical Stability
- Epsilon values properly specified: 1e-6 for LayerNorm, 1e-12 for VAE RMSNorm
- Float32 upcasting in RMSNorm computation at `src/qwen_quantized.rs:681-705`
- Gradient clipping infrastructure present (though not used)

---

## 6. Performance Analysis

### 6.1 Current Bottlenecks

1. **Attention Computation**: O(N²) memory, not using Flash Attention
2. **VAE Decode**: Full image decode, no tiling for large images
3. **Kernel Dispatch**: 60 layers × 20 steps = 1200+ individual kernel dispatches
4. **Weight Loading**: Sequential loading, no async prefetching

### 6.2 Memory Usage

| Component | 4-bit Model | 8-bit Model | Activation (512×512) |
|-----------|-------------|-------------|---------------------|
| Transformer | ~4 GB | ~7 GB | ~1.5 GB |
| VAE | ~500 MB | ~500 MB | ~1 GB |
| Text Encoder | ~2 GB | ~2 GB | ~200 MB |
| **Total** | **~6.5 GB** | **~9.5 GB** | **~2.7 GB** |

---

## 7. Performance Enhancement Plan

### 7.1 Critical Optimizations (P0)

#### 7.1.1 Flash Attention / Memory-Efficient Attention

**Current**: `src/qwen_quantized.rs:294-318`

**Problem**: Materializes full O(N²) attention matrix. For 1024 patches × 77 text tokens with 24 heads: ~756M float32 values = 3GB intermediate memory.

**Solution**: Use MLX's `fast::scaled_dot_product_attention`:
```rust
let out = mlx_rs::fast::scaled_dot_product_attention(
    &q, &k, &v,
    encoder_hidden_states_mask,
    scale,
    false,  // causal mask not needed for diffusion
)?;
```

**Expected Impact**:
- 2-3x attention speedup
- 30-40% overall generation improvement
- 50%+ memory reduction during attention

**Implementation**: Replace lines 294-318 in `src/qwen_quantized.rs`

#### 7.1.2 VAE Tiled Decoding

**Current**: `examples/generate_qwen_image.rs:945`

**Problem**: Full 512×512 decode requires ~1.5GB activation memory

**Solution**: Implement tiled VAE decode with overlap blending:
```rust
impl QwenVAE {
    pub fn decode_tiled(&mut self, latent: &Array, tile_size: i32, overlap: i32)
        -> Result<Array, Exception> {
        // Process 32×32 or 64×64 latent tiles sequentially
        // Blend overlapping regions with Gaussian weights
    }
}
```

**Expected Impact**:
- Enable 2-4x larger images
- 15-20% speedup from cache locality
- 4x memory reduction

### 7.2 High-Impact Optimizations (P1)

#### 7.2.1 Kernel Fusion for Modulation

**Current**: 4+ kernel dispatches per modulation (`src/qwen_quantized.rs:407-451`)

**Solution**: Fuse into single Metal kernel via `mlx-rs-core`:
```rust
mlx_rs_core::fused_modulate_norm(
    hidden_states,
    &img_mod_params,
    eps,
)?;
```

**Expected Impact**: 15-20% reduction in dispatch overhead

#### 7.2.2 INT8 VAE Quantization

**Current**: Only transformer uses quantization

**Solution**: Add INT8 VAE support:
```rust
#[derive(Debug, Clone, ModuleParameters)]
pub struct QuantizedVAE {
    #[param]
    pub decoder: QuantizedDecoder3D,
}
```

**Expected Impact**: 50% VAE memory reduction, 20-30% decode speedup

#### 7.2.3 Async Text Encoding

**Current**: Sequential loading and encoding

**Solution**: Text encoding in parallel with transformer weight loading:
```rust
let text_handle = std::thread::spawn(|| load_and_encode_text(&model_dir, &prompt));
let transformer_weights = load_sharded_weights(&transformer_files)?;
let text_embeddings = text_handle.join().unwrap();
```

**Expected Impact**: 10-15% wall-time reduction on cold starts

### 7.3 Medium-Impact Optimizations (P2)

#### 7.3.1 Pre-computed RoPE Tables

**Current**: Computed every generation (`examples/generate_qwen_image.rs:618-676`)

**Solution**: Cache for common image sizes:
```rust
lazy_static! {
    static ref ROPE_CACHE: RwLock<HashMap<(i32, i32), RoPECache>> =
        RwLock::new(HashMap::new());
}
```

**Expected Impact**: 50-100ms per generation

#### 7.3.2 Batched VAE Decode

**Solution**: For batch workflows, decode together:
```rust
let batched_decoded = vae.decode_batched(&batched_latents)?;
```

**Expected Impact**: 2x better GPU utilization for batch generation

#### 7.3.3 Weight Streaming

**Solution**: Load transformer blocks on-demand for systems with limited RAM:
```rust
pub struct StreamingTransformer {
    block_cache: LruCache<usize, QwenTransformerBlock>,
    max_resident_blocks: usize,  // e.g., 12 of 60
}
```

**Expected Impact**: Enable inference on 4GB Macs, 50% memory reduction

### 7.4 Algorithmic Improvements (P3)

#### 7.4.1 Dynamic CFG Scheduling

**Current**: Fixed CFG scale throughout denoising

**Solution**: High CFG early, low CFG late:
```rust
let cfg_schedule = |step: i32, total: i32| -> f32 {
    let t = step as f32 / total as f32;
    3.0 + 4.0 * (1.0 - t * t)  // Quadratic decay from 7.0 to 3.0
};
```

**Expected Impact**: Better quality at same step count, or 25% fewer steps

#### 7.4.2 Progressive Distillation

**Solution**: Train 4-step or 8-step student model

**Expected Impact**: 5x inference speedup (requires training)

### 7.5 Implementation Priority Matrix

| Priority | Optimization | Effort | Impact | Files to Modify |
|----------|--------------|--------|--------|-----------------|
| **P0** | Flash Attention | Low | Very High | `qwen_quantized.rs` |
| **P0** | VAE Tiled Decode | Medium | Very High | `vae/vae.rs`, `vae/blocks.rs` |
| **P1** | Async Text Encoding | Low | Medium | `generate_qwen_image.rs` |
| **P1** | INT8 VAE | Medium | Medium | `vae/`, new `quantized_vae.rs` |
| **P2** | RoPE Caching | Low | Low | `generate_qwen_image.rs` |
| **P2** | Kernel Fusion | High | Medium | `mlx-rs-core/src/metal_kernels.rs` |
| **P3** | Weight Streaming | Medium | Medium | New `streaming.rs` |
| **P3** | Progressive Distillation | Very High | Very High | New training code |

---

## 8. Recommendations Summary

### 8.1 Immediate Actions (This Week)

1. **Remove Debug Code**: Strip all `eprintln!` debug blocks from `qwen_quantized.rs`
2. **Add Flash Attention**: Single-line change for 30-40% speedup
3. **Document Magic Numbers**: Replace hardcoded values with named constants

### 8.2 Short-term (Next Month)

1. **VAE Tiling**: Implement tiled decode for 1024×1024 support
2. **Code Consolidation**: Merge duplicate transformer implementations
3. **INT8 VAE**: Add quantization support to VAE decoder

### 8.3 Medium-term (Next Quarter)

1. **Kernel Fusion**: Add custom Metal kernels to `mlx-rs-core`
2. **Streaming**: Implement block streaming for low-memory devices
3. **Distillation**: Train 4-step student model

### 8.4 Code Cleanup Checklist

- [ ] Remove all `static DEBUG_*` atomic flags
- [ ] Remove all `if false` blocks
- [ ] Remove no-op string replacements in weight mapping
- [ ] Consolidate `QwenTransformer` and `QwenQuantizedTransformer`
- [ ] Replace custom `Linear` with `mlx_rs::nn::Linear`
- [ ] Add proper documentation to all public APIs
- [ ] Add unit tests for core components (attention, modulation, VAE blocks)

---

## 9. Conclusion

The qwen-image-mlx crate successfully implements the Qwen-Image-2512 model with correct algorithms and proper quantization support. The architecture is sound and the implementation produces correct outputs.

**Key Strengths**:
- Correct flow matching implementation
- Proper 4/8-bit quantization support
- Sound 3D causal convolution VAE
- Batched CFG optimization

**Key Weaknesses**:
- Debug code pollution throughout
- Significant code duplication
- Not using optimized attention kernels
- Memory-intensive VAE for large images

**Priority**: After implementing Flash Attention and VAE tiling, inference time could be reduced by **40-50%** with minimal code changes.

---

*Review generated by Claude Code (kimi-k2.5) on 2026-01-29*
