# Mixed Precision Optimization for Qwen-Image

## Goal

Improve numerical stability using Draw Things-style mixed precision techniques.

## Background

Research shows ANE (Apple Neural Engine) cannot run DiT models due to FP32 requirements.
MLX already handles most precision issues well, but we add safety for extreme values.

## Implemented Optimizations

### 1. FP32 Accumulation Before Final LayerNorm (IMPLEMENTED)

DiT activations grow to extreme values (±50M) during transformer blocks.
Upcast to FP32 before final LayerNorm for numerical stability.

```rust
// Before norm_out - prevents potential overflow
let input_dtype = hidden_states.dtype();
let hidden_states = hidden_states.as_dtype(Dtype::Float32)?;
let hidden_states = self.norm_out.forward(&hidden_states, &text_embeddings)?;
let hidden_states = hidden_states.as_dtype(input_dtype)?;
```

**Result:** Added to both `qwen_quantized.rs` and `qwen_full_precision.rs`

## Techniques Tested But NOT Implemented

### 2. Pre-Attention Scaling (REVERTED)

Tested applying 1/√D scaling to Q and K separately before matmul.

**Result:** No benefit for MLX - it already uses FP32 for softmax internally.
Added overhead without performance gain.

### 3. FFN Activation Scaling (REVERTED)

Tested scaling down inputs to FFN by 0.25x and scaling back 4x after.

**Result:** Made things WORSE! The FFN output was already large, and scaling
back up amplified the values further. The issue is accumulated residuals,
not FFN intermediate values.

## Benchmark Results

10-step generation on 512x512:
- Baseline: ~37s
- With FP32 norm_out only: ~37s (no significant change)
- With FFN scaling: ~36s but output values larger (bad)
- With pre-attention scaling: ~37s (no benefit)

## Lessons Learned

1. **MLX handles precision well** - Most mixed precision tricks from PyTorch/CUDA
   don't apply because MLX uses FP32 internally for softmax and other ops.

2. **FFN scaling doesn't help DiT** - The activation explosion comes from
   accumulated residuals across 60 transformer blocks, not from any single
   FFN layer. Scaling individual layers makes the final residual worse.

3. **FP32 norm_out is still valid** - Even though it doesn't improve speed,
   it provides safety margin for extreme values (±50M range).

4. **Draw Things optimizations are for their custom kernels** - Their Metal
   FlashAttention implementation has different precision characteristics
   than MLX's built-in ops.

## Files Modified

| File | Change |
|------|--------|
| `src/qwen_quantized.rs` | FP32 accumulation before norm_out |
| `src/qwen_full_precision.rs` | FP32 accumulation before norm_out |

## Conclusion

For MLX-based implementations, the main optimization path is:
1. **Wait for MLX improvements** - Apple is actively improving SDPA
2. **Upgrade to M5** - 3.8x automatic speedup with TensorOps
3. **Use Metal FlashAttention** - High effort, 12-15% gain possible

Simple mixed precision tricks don't provide significant benefits for MLX.

## References

- [Draw Things: BF16 and Image Generation](https://engineering.drawthings.ai/p/bf16-and-image-generation-models-803cf0515bee)
- [Metal FlashAttention 2.0](https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c)
