# Metal FlashAttention Integration for Qwen-Image

## Overview

This document outlines the feasibility of integrating [universal-metal-flash-attention](https://github.com/bghira/universal-metal-flash-attention) to accelerate the DiT transformer in Qwen-Image.

## API Summary

The library provides a C FFI with these key functions:

```c
// Context management
mfa_error_t mfa_create_context(mfa_context_t* context);
void mfa_destroy_context(mfa_context_t context);

// Buffer management
mfa_error_t mfa_create_buffer(mfa_context_t context, size_t size_bytes, mfa_buffer_t* buffer);
mfa_error_t mfa_buffer_from_ptr(mfa_context_t context, void* data_ptr, size_t size_bytes, mfa_buffer_t* buffer);
void* mfa_buffer_contents(mfa_buffer_t buffer);
void mfa_destroy_buffer(mfa_buffer_t buffer);

// Forward attention
mfa_error_t mfa_attention_forward(
    mfa_context_t context,
    mfa_buffer_t q, mfa_buffer_t k, mfa_buffer_t v, mfa_buffer_t out,
    uint32_t batch_size, uint32_t seq_len_q, uint32_t seq_len_kv,
    uint32_t num_heads, uint16_t head_dim,
    float softmax_scale, bool causal,
    mfa_precision_t input_precision,
    mfa_precision_t intermediate_precision,
    mfa_precision_t output_precision,
    bool transpose_q, bool transpose_k, bool transpose_v, bool transpose_o,
    const void* mask_ptr, size_t mask_size_bytes,
    const int64_t* mask_shape, const int64_t* mask_strides, uint32_t mask_ndim,
    mfa_mask_type_t mask_type, mfa_mask_scalar_t mask_scalar_type
);
```

## Precision Types

```c
MFA_PRECISION_FP16  // 0
MFA_PRECISION_BF16  // 1
MFA_PRECISION_FP32  // 2
MFA_PRECISION_INT8  // 3
MFA_PRECISION_INT4  // 4
```

## Requirements

- macOS 15+ (Sequoia)
- Xcode 15+ with Swift 5.10+
- Metal-capable device (M1+)

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Qwen-Image Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│  Text Encoder (MLX)  →  Transformer (MFA)  →  VAE (MLX)     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Metal Flash Attention Bridge                   │
├─────────────────────────────────────────────────────────────┤
│  1. Convert MLX Array → MFA Buffer (zero-copy if possible)  │
│  2. Call mfa_attention_forward()                            │
│  3. Convert MFA Buffer → MLX Array                          │
└─────────────────────────────────────────────────────────────┘
```

## Qwen-Image Attention Parameters

| Parameter | Value |
|-----------|-------|
| batch_size | 1 (typically) |
| seq_len_q | 1024 (32x32 patches) + txt_seq |
| seq_len_kv | Same as seq_len_q |
| num_heads | 24 |
| head_dim | 128 |
| precision | BF16 |
| causal | false (joint attention) |

## Implementation Steps

### Step 1: Build universal-metal-flash-attention

```bash
git clone https://github.com/bghira/universal-metal-flash-attention
cd universal-metal-flash-attention
git submodule update --init --recursive
swift build -c release
```

### Step 2: Create Rust FFI Bindings

```rust
// src/mfa_sys.rs
#[repr(C)]
pub struct MfaContext(*mut std::ffi::c_void);

#[repr(C)]
pub struct MfaBuffer(*mut std::ffi::c_void);

pub const MFA_PRECISION_BF16: i32 = 1;
pub const MFA_SUCCESS: i32 = 0;

#[link(name = "MFAFFI")]
extern "C" {
    pub fn mfa_create_context(context: *mut MfaContext) -> i32;
    pub fn mfa_destroy_context(context: MfaContext);
    pub fn mfa_create_buffer(
        context: MfaContext,
        size_bytes: usize,
        buffer: *mut MfaBuffer,
    ) -> i32;
    pub fn mfa_buffer_from_ptr(
        context: MfaContext,
        data_ptr: *mut std::ffi::c_void,
        size_bytes: usize,
        buffer: *mut MfaBuffer,
    ) -> i32;
    pub fn mfa_buffer_contents(buffer: MfaBuffer) -> *mut std::ffi::c_void;
    pub fn mfa_destroy_buffer(buffer: MfaBuffer);
    pub fn mfa_attention_forward(
        context: MfaContext,
        q: MfaBuffer, k: MfaBuffer, v: MfaBuffer, out: MfaBuffer,
        batch_size: u32, seq_len_q: u32, seq_len_kv: u32,
        num_heads: u32, head_dim: u16,
        softmax_scale: f32, causal: bool,
        input_precision: i32,
        intermediate_precision: i32,
        output_precision: i32,
        transpose_q: bool, transpose_k: bool, transpose_v: bool, transpose_o: bool,
        mask_ptr: *const std::ffi::c_void, mask_size_bytes: usize,
        mask_shape: *const i64, mask_strides: *const i64, mask_ndim: u32,
        mask_type: i32, mask_scalar_type: i32,
    ) -> i32;
    pub fn mfa_is_device_supported() -> bool;
}
```

### Step 3: Safe Rust Wrapper

```rust
// src/mfa.rs
pub struct FlashAttention {
    context: MfaContext,
}

impl FlashAttention {
    pub fn new() -> Result<Self, MfaError> {
        if !unsafe { mfa_is_device_supported() } {
            return Err(MfaError::DeviceNotSupported);
        }
        let mut context = MfaContext(std::ptr::null_mut());
        let err = unsafe { mfa_create_context(&mut context) };
        if err != MFA_SUCCESS {
            return Err(MfaError::from_code(err));
        }
        Ok(Self { context })
    }

    pub fn forward(
        &self,
        q: &Array, k: &Array, v: &Array,
        num_heads: u32, head_dim: u16,
        scale: f32,
    ) -> Result<Array, MfaError> {
        // Implementation: convert MLX arrays to MFA buffers,
        // call mfa_attention_forward, convert back
        todo!()
    }
}

impl Drop for FlashAttention {
    fn drop(&mut self) {
        unsafe { mfa_destroy_context(self.context) };
    }
}
```

### Step 4: MLX Array ↔ MFA Buffer Conversion

The key challenge is efficiently converting between MLX Array and MFA Buffer:

**Option A: Copy data**
```rust
// Slow but safe
let data = array.as_slice::<bf16>()?;
let mut buffer = MfaBuffer::null();
mfa_buffer_from_ptr(context, data.as_ptr(), data.len() * 2, &mut buffer);
```

**Option B: Zero-copy via Metal buffer sharing**
```rust
// Fast but requires Metal buffer access
// MLX arrays are backed by Metal buffers - extract and share
let mtl_buffer = array.metal_buffer();  // Need MLX API for this
mfa_buffer_from_mtl_buffer(context, mtl_buffer, size, &mut buffer);
```

## Challenges

### 1. MLX Metal Buffer Access
MLX doesn't expose the underlying MTLBuffer directly. Options:
- Use `mlx_sys` to access internal buffer
- Copy data through CPU (slower but works)
- Contribute Metal buffer access API to mlx-rs

### 2. Memory Layout
MFA expects specific tensor layouts:
- Q/K/V: `[batch, seq, heads, head_dim]` or `[batch, heads, seq, head_dim]`
- Must match MLX's layout or transpose

### 3. Synchronization
MLX uses lazy evaluation. Must ensure:
- Arrays are evaluated before MFA access
- MFA completion before MLX continues

### 4. Build Complexity
Need to:
- Build Swift package
- Link against MFAFFI library
- Set DYLD_LIBRARY_PATH or embed library

## Expected Performance

Based on Draw Things benchmarks:
- 20-25% faster attention on M3/M4
- 43-120% faster for full image generation
- More benefit with larger sequence lengths

For Qwen-Image (1024 patches + ~100 text tokens):
- Current: ~3.7s per step
- Expected: ~3.0s per step (20% improvement)
- 10 steps: 37s → 30s (7 second savings)

## Recommendation

### Proceed with caution if:
- You need maximum performance on M3/M4
- You're willing to maintain complex build setup
- macOS 15+ is acceptable as minimum requirement

### Skip if:
- Development velocity is priority
- macOS 14 support needed
- Waiting for MLX improvements is acceptable

## Alternative: Wait for MLX

Apple is actively improving MLX attention:
- [Issue #1582](https://github.com/ml-explore/mlx/issues/1582): Masked SDPA
- [Issue #2955](https://github.com/ml-explore/mlx/issues/2955): FlashAttention proposal

M5 TensorOps will provide 3.8x automatic speedup.

## Next Steps (If Proceeding)

1. [ ] Clone and build universal-metal-flash-attention
2. [ ] Create minimal Rust FFI test
3. [ ] Benchmark MFA vs MLX SDPA standalone
4. [ ] Implement MLX Array → MFA Buffer conversion
5. [ ] Integrate into QwenAttention
6. [ ] Full pipeline benchmark

## Feasibility Test Results (2026-01-30)

### Test Configuration
- **Chip**: Apple M3 Max
- **macOS**: 26.2 (Tahoe beta)
- **Library**: universal-metal-flash-attention 1.0.0

### Test Results

**FFI Binding**: ✅ SUCCESS
- Library links correctly with Rust FFI
- Context creation works
- Buffer management works
- Device detection works

**Attention Forward**: ❌ FAILED
- Error code: 5 (Pipeline creation error)
- Metal shader compilation fails at runtime

### Error Details

```
error: illegal string literal in 'asm'
  __asm("air.simdgroup_async_copy_1d.p3i8.p1i8");

error: use of undeclared identifier '__metal_simdgroup_async_copy_1d'
```

### Root Cause

The library uses inline Metal assembly (`__asm`) to access low-level AIR (Apple Intermediate Representation) instructions for simdgroup async copy operations. This syntax is **not compatible with macOS 26 (Tahoe) Metal compiler**.

The library was tested on macOS 15 (Sequoia). The Metal shader language APIs appear to have changed in macOS 26.

### Workarounds

1. **Wait for library update**: Report issue to universal-metal-flash-attention maintainers
2. **Use macOS 15**: Install macOS 15 for compatibility
3. **Use MLX SDPA**: Continue using MLX's built-in scaled_dot_product_attention

### Conclusion

Metal Flash Attention integration is **not feasible** on macOS 26 until the library is updated for the new Metal compiler. The FFI bindings work correctly, but the Metal shaders need to be updated.

For now, the recommended path is:
1. Continue using MLX's SDPA (works well)
2. Wait for MLX improvements
3. M5 TensorOps will provide 3.8x automatic speedup

## References

- [universal-metal-flash-attention](https://github.com/bghira/universal-metal-flash-attention)
- [metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
- [ccv MFA](https://github.com/liuliu/ccv/tree/unstable/lib/nnc/mfa)
- [Draw Things Engineering Blog](https://engineering.drawthings.ai/)
