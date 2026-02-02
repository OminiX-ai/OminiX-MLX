# Qwen-Image-MLX Cleanup Plan

*Date: 2026-01-29*

## Overview

Based on code reviews from GLM-4.7 and Kimi-K2.5, plus our own analysis.

## Phase 1: Remove Debug Code (P0)

### Target: qwen_quantized.rs
- **22 static DEBUG_* atomic flags** to remove
- **88+ eprintln! statements** to remove
- **9 "if false" blocks** to remove

### Target: generate_qwen_image.rs
- Remove weight debug printing (lines 238-354)
- Remove text encoder debug output (lines 471-519)

## Phase 2: Define Constants (P1)

Create `/Users/yuechen/home/OminiX-MLX/qwen-image-mlx/src/constants.rs`:

```rust
//! Common constants for Qwen-Image model

/// Timestep embedding dimension
pub const TIMESTEP_EMBED_DIM: i32 = 256;

/// Timestep scale factor (multiply sigma by this)
pub const TIMESTEP_SCALE: f32 = 1000.0;

/// RoPE theta parameter
pub const ROPE_THETA: f32 = 10000.0;

/// LayerNorm/RMSNorm epsilon
pub const LAYER_NORM_EPS: f32 = 1e-6;

/// Qwen VL template prefix token count
pub const QWEN_TEMPLATE_PREFIX_TOKENS: usize = 34;

/// Max text tokens after dropping template prefix
pub const MAX_TEXT_OUTPUT_TOKENS: usize = 77;

/// Max text input tokens (output + prefix)
pub const MAX_TEXT_INPUT_TOKENS: usize = MAX_TEXT_OUTPUT_TOKENS + QWEN_TEMPLATE_PREFIX_TOKENS;

/// RoPE lookup table maximum index
pub const ROPE_MAX_INDEX: i32 = 4096;

/// Patch embedding dimension (latent channels)
pub const PATCH_EMBEDDING_DIM: i32 = 64;

/// RoPE axes dimensions [frames, height, width]
pub const ROPE_AXES_DIM: [i32; 3] = [16, 56, 56];

/// VAE spatial downsample factor
pub const VAE_SPATIAL_DOWNSAMPLE: i32 = 16;

/// Minimum image dimension
pub const MIN_IMAGE_SIZE: i32 = 256;

/// Maximum image dimension
pub const MAX_IMAGE_SIZE: i32 = 2048;
```

## Phase 3: Add Input Validation (P1)

### Add to pipeline.rs
```rust
fn validate_dimensions(height: i32, width: i32) -> Result<(), QwenImageError> {
    if height % VAE_SPATIAL_DOWNSAMPLE != 0 {
        return Err(QwenImageError::InvalidDimension(
            format!("height {} must be divisible by {}", height, VAE_SPATIAL_DOWNSAMPLE)
        ));
    }
    if width % VAE_SPATIAL_DOWNSAMPLE != 0 {
        return Err(QwenImageError::InvalidDimension(
            format!("width {} must be divisible by {}", width, VAE_SPATIAL_DOWNSAMPLE)
        ));
    }
    if height < MIN_IMAGE_SIZE || height > MAX_IMAGE_SIZE {
        return Err(QwenImageError::InvalidDimension(
            format!("height {} must be between {} and {}", height, MIN_IMAGE_SIZE, MAX_IMAGE_SIZE)
        ));
    }
    if width < MIN_IMAGE_SIZE || width > MAX_IMAGE_SIZE {
        return Err(QwenImageError::InvalidDimension(
            format!("width {} must be between {} and {}", width, MIN_IMAGE_SIZE, MAX_IMAGE_SIZE)
        ));
    }
    Ok(())
}
```

### Add to error.rs
```rust
#[derive(Debug, thiserror::Error)]
pub enum QwenImageError {
    // ... existing variants ...

    #[error("Invalid dimension: {0}")]
    InvalidDimension(String),
}
```

## Phase 4: Document MLX Lazy Evaluation Lessons (P2)

Create `/Users/yuechen/home/OminiX-MLX/qwen-image-mlx/docs/MLX_BEST_PRACTICES.md`

## Phase 5: Code Consolidation (P3 - Future)

This is a larger refactoring effort:
1. Extract common traits for Transformer blocks
2. Unify RoPE implementations
3. Create shared modulation/normalization utilities

**Deferred** - Requires more careful planning to avoid breaking changes.

## Execution Order

1. [x] Create constants.rs ✅
2. [x] Update error.rs with InvalidDimension variant ✅
3. [x] Add validation function to pipeline.rs ✅
4. [x] Remove debug code from qwen_quantized.rs ✅ (22 DEBUG flags removed, ~300 lines cleaned)
5. [ ] Remove debug code from generate_qwen_image.rs (deferred - example file)
6. [x] Create MLX_BEST_PRACTICES.md ✅
7. [ ] (Future) Code consolidation

## Files to Modify

| File | Action |
|------|--------|
| `src/constants.rs` | Create new |
| `src/lib.rs` | Add `pub mod constants;` |
| `src/error.rs` | Add InvalidDimension variant |
| `src/pipeline.rs` | Add validation |
| `src/qwen_quantized.rs` | Remove debug code |
| `examples/generate_qwen_image.rs` | Remove debug, add validation, use constants |
| `docs/MLX_BEST_PRACTICES.md` | Create new |
