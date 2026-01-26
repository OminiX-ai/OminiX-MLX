# MLX Debugging Workflows

Debugging workflows for mlx-rs models. Use when investigating issues, comparing outputs, or validating implementations.

## Comparing Rust vs Python Outputs

### Step-by-Step Comparison

1. **Export intermediate values from Python**:
```python
import mlx.core as mx
import numpy as np

# Save intermediate tensor
mx.save("debug_tensor.npy", tensor)
```

2. **Load and compare in Rust**:
```rust
use mlx_rs::Array;

let expected = Array::load("debug_tensor.npy")?;
let actual = model.forward(&input)?;

let diff = (&actual - &expected).abs().sum(None, None)?;
println!("Diff sum: {:?}", diff.item::<f32>());
```

### Common Debugging Prints

```rust
// Shape debugging
println!("Shape: {:?}", tensor.shape());

// Sum for numerical comparison
println!("Sum: {:?}", tensor.sum(None, None)?.item::<f32>());

// First few values
println!("First 5: {:?}", tensor.flatten(None, None)?
    .index(..5)?
    .to_vec::<f32>()?);
```

---

## Weight Loading Issues

### Symptom: Noisy/garbage output

**Common causes**:
1. Weight name mismatch (sanitization issue)
2. Wrong dtype conversion
3. Missing weights (silently using random init)

**Debug approach**:
```rust
// Check which weights were loaded
for (name, _) in &weights {
    println!("Loaded: {}", name);
}

// Check for missing weights after sanitization
let expected_weights = ["layer.weight", "layer.bias"];
for name in expected_weights {
    if !weights.contains_key(name) {
        println!("MISSING: {}", name);
    }
}
```

### Symptom: NaN/Inf in output

**Common causes**:
1. Overflow in attention (missing scaling)
2. Division by zero in normalization
3. Exploding gradients in deep networks

**Debug approach**:
```rust
fn check_tensor(name: &str, t: &Array) {
    let has_nan = t.is_nan().any(None, None)?.item::<bool>();
    let has_inf = t.is_inf().any(None, None)?.item::<bool>();
    if has_nan || has_inf {
        println!("WARNING: {} has nan={} inf={}", name, has_nan, has_inf);
    }
}
```

---

## Performance Debugging

### Memory Usage

```rust
use std::process::Command;

fn print_memory() {
    // Force evaluation of lazy ops
    mlx_rs::eval(&[&tensor])?;

    // Check process memory (macOS)
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()?;
    let kb: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()?;
    println!("Memory: {} MB", kb / 1024);
}
```

### Timing

```rust
use std::time::Instant;

let start = Instant::now();
let output = model.forward(&input)?;
mlx_rs::eval(&[&output])?;  // Force evaluation!
println!("Forward: {:?}", start.elapsed());
```

**Important**: MLX operations are lazy - always call `eval()` before timing!

---

## Quantization Issues

### Symptom: Quantized model gives different results

**Expected**: Small numerical differences (~1% relative error)

**Debug approach**:
```rust
// Compare quantized vs dequantized
let q_out = quantized_model.forward(&input)?;
let f_out = float_model.forward(&input)?;

let rel_diff = (&q_out - &f_out).abs() / (&f_out.abs() + 1e-8);
let max_diff = rel_diff.max(None, None)?.item::<f32>();
println!("Max relative diff: {:.4}%", max_diff * 100.0);
```

### Symptom: Quantized model crashes

**Common causes**:
1. Missing `.scales` or `.biases` weights
2. Wrong group_size (default is 32)
3. Incompatible input dtype

---

## Attention Debugging

### Check attention weights

```rust
// In attention forward
let attn_weights = ops::softmax(&scores, &[-1])?;

// Debug: check for uniform attention (bad) vs focused (good)
let entropy = -(&attn_weights * attn_weights.log()).sum(&[-1], None)?;
println!("Attention entropy: {:?}", entropy.mean(None, None)?.item::<f32>());
// High entropy = uniform attention (potential issue)
// Low entropy = focused attention (usually good)
```

### Check QKV shapes

```rust
println!("Q: {:?}, K: {:?}, V: {:?}", q.shape(), k.shape(), v.shape());
// Expected: [batch, heads, seq, head_dim]
```

---

## Common Pitfalls

| Issue | Symptom | Fix |
|-------|---------|-----|
| Forgot `eval()` | Wrong timing | Add `mlx_rs::eval(&[&output])?` |
| Wrong transpose | Shape mismatch | Check axis order carefully |
| Missing `&` | Move error | Most ops take references |
| Lazy evaluation | Memory keeps growing | Call `eval()` periodically |
| Dtype mismatch | Silent precision loss | Explicit `.as_dtype()` |
| **Strided view + as_slice()** | **Tiled/corrupted images** | **Flatten + reshape before as_slice()** |
| **AdaLN scale/shift order** | **Output 3x wrong magnitude** | **scale=first half, shift=second half** |

---

## Critical: Strided Views

<!-- Evolution: 2025-01-25 | source: qwen-image-mlx -->

**This is one of the most common bugs in mlx-rs!**

`index()` and `transpose_axes()` create strided views. `as_slice()` ignores strides and returns raw memory.

```rust
// BROKEN: Image appears as 3x3 tiled pattern
let img = tensor.transpose_axes(&[1, 2, 0])?;  // [C,H,W] -> [H,W,C]
let pixels: Vec<u8> = img.as_slice().to_vec();  // WRONG!

// FIXED: Force contiguity first
let img = tensor.transpose_axes(&[1, 2, 0])?;
let n = img.dim(0) * img.dim(1) * img.dim(2);
let img = img.reshape(&[n])?.reshape(&[img.dim(0), img.dim(1), img.dim(2)])?;
img.eval()?;
let pixels: Vec<u8> = img.as_slice().to_vec();  // CORRECT!
```

See `mlx-rs-patterns.md` for full details.

---

## DiT (Diffusion Transformer) Debugging

<!-- Evolution: 2025-01-25 | source: qwen-image-mlx -->

### Value Explosion is NORMAL

**Key insight**: Hidden states growing to millions during transformer blocks is **expected behavior** in DiT models.

```
Block 0 output:  [-2M, 2M]
Block 30 output: [-10M, 10M]
Block 59 output: [-45M, 46M]
After norm_out:  [-6, 6]      ← LayerNorm normalizes back!
Final output:    [-4.5, 4.5]
```

**Don't panic** if you see huge intermediate values. The `norm_out` (AdaLayerNormContinuous) applies LayerNorm which normalizes everything back to a reasonable range.

### DiT Modulation Formula

```rust
// Standard DiT modulation (AdaLN)
// Formula: (1 + scale) * norm(x) + shift

fn modulate(x: &Array, mod_params: &Array) -> (Array, Array) {
    let dim = mod_params.dim(-1) / 3;
    let shift = mod_params.index((.., ..dim)).expand_dims(1)?;
    let scale = mod_params.index((.., dim..dim * 2)).expand_dims(1)?;
    let gate = mod_params.index((.., dim * 2..));

    let one = Array::from_f32(1.0);
    let scale_factor = ops::add(&one, &scale)?;
    let modulated = ops::add(&ops::multiply(x, &scale_factor)?, &shift)?;

    (modulated, gate)
}
```

**Scale/Shift ordering varies by model**:
- FLUX/Qwen-Image: `[shift, scale, gate]` from mod1 split
- Some models: `[scale, shift, gate]`
- AdaLayerNormContinuous: `[scale, shift]` (no gate)

Always verify against reference implementation!

### DiT Debug Checkpoints

Add these debug prints to trace issues:

```rust
// 1. After img_in projection
eprintln!("[IMG_IN] range=[{:.2}, {:.2}]", hidden.min(), hidden.max());
// Expected: [-15, 16] for 512x512

// 2. After LayerNorm (before modulation)
eprintln!("[NORMED] range=[{:.2}, {:.2}]", normed.min(), normed.max());
// Expected: [-9, 9] (centered around 0)

// 3. Modulation parameters
eprintln!("[MOD] scale=[{:.2}, {:.2}], shift=[{:.2}, {:.2}]",
    scale.min(), scale.max(), shift.min(), shift.max());
// Expected: scale [-30, 20], shift [-30, 20] (large is OK!)

// 4. After modulation
eprintln!("[MODULATED] range=[{:.2}, {:.2}]", modulated.min(), modulated.max());
// Expected: [-70, 55]

// 5. Attention output
eprintln!("[ATTN_OUT] range=[{:.2}, {:.2}]", attn_out.min(), attn_out.max());
// Expected: [-90, 120]

// 6. After FFN (will be large!)
eprintln!("[FFN_OUT] range=[{:.2}, {:.2}]", ffn_out.min(), ffn_out.max());
// Expected: [-25000, 25000] (yes, this is normal!)

// 7. Final output (after norm_out + proj_out)
eprintln!("[FINAL] range=[{:.2}, {:.2}]", result.min(), result.max());
// Expected: [-5, 6] (back to reasonable range)
```

### Common DiT Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Output all black | scale/shift order wrong | Check reference impl |
| Output all noise | Missing LayerNorm | Add `affine=false` LayerNorm |
| Output exploding after proj_out | norm_out not normalizing | Verify norm_out has LayerNorm |
| Values ±inf after block 0 | Missing attention scaling | Add `/ sqrt(head_dim)` |
| Gate values wrong sign | Wrong split indices | Verify dim*2..dim*3 for gate |

### Validating Against mflux

```python
# Monkey-patch mflux to capture intermediate values
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlock

original_call = QwenTransformerBlock.__call__
def patched_call(self, hidden_states, ...):
    print(f"hidden: [{hidden_states.min():.2f}, {hidden_states.max():.2f}]")
    return original_call(self, hidden_states, ...)
QwenTransformerBlock.__call__ = patched_call
```

Compare values at each checkpoint - they should match within ~0.01.
