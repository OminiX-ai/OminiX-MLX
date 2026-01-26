# MLX-rs Development Patterns

## Overview

mlx-rs is the Rust binding for Apple's MLX framework, optimized for Apple Silicon. This skill covers common patterns and best practices.

## Core Concepts

### Array Operations

```rust
use mlx_rs::{Array, ops, array};

// Create arrays
let x = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
let y = array!(2.0f32);  // Scalar

// Operations (lazy evaluation)
let z = ops::multiply(&x, &y)?;
let z = ops::add(&z, &array!(1.0f32))?;

// Force evaluation
z.eval()?;
```

### Lazy Evaluation

MLX uses lazy evaluation - operations are queued and fused by Metal compiler:

```rust
// These operations are NOT executed immediately
let a = ops::matmul(&x, &w)?;
let b = ops::add(&a, &bias)?;
let c = mlx_rs::nn::relu(&b)?;

// Only evaluated when needed
c.eval()?;  // Now Metal compiles and executes fused kernel
```

### Module System

```rust
use mlx_rs::module::{Module, ModuleParameters, Param};
use mlx_macros::ModuleParameters;

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct MyLayer {
    #[param]
    pub linear: Linear,
    #[param]
    pub norm: LayerNorm,
    pub hidden_size: i32,  // Not a param - no #[param] attribute
}

impl Module<&Array> for MyLayer {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array) -> Result<Array, Self::Error> {
        let x = self.linear.forward(x)?;
        self.norm.forward(&x)
    }
}
```

### Weight Loading

```rust
use mlx_rs::module::ModuleParameters;
use std::collections::HashMap;

// Load from safetensors
let weights: HashMap<String, Array> = load_safetensors(path)?;

// Convert to Rc<str> keys for update_flattened
let weights_rc: HashMap<std::rc::Rc<str>, Array> = weights
    .into_iter()
    .map(|(k, v)| (std::rc::Rc::from(k.as_str()), v))
    .collect();

// Apply to model
model.update_flattened(weights_rc);
```

## Neural Network Layers

### Linear

```rust
use mlx_rs::nn::{Linear, LinearBuilder};

// With bias (default)
let linear = LinearBuilder::new(in_features, out_features).build()?;

// Without bias
let linear = LinearBuilder::new(in_features, out_features)
    .bias(false)
    .build()?;

let output = linear.forward(&input)?;
```

### LayerNorm / RmsNorm

```rust
use mlx_rs::nn::{LayerNorm, LayerNormBuilder, RmsNorm};

// LayerNorm with learnable affine
let norm = LayerNormBuilder::new(hidden_size)
    .eps(1e-6)
    .build()?;

// LayerNorm without affine (for AdaLN)
let norm = LayerNormBuilder::new(hidden_size)
    .affine(false)
    .eps(1e-6)
    .build()?;

// RmsNorm
let norm = RmsNorm::new(hidden_size)?;
```

### Activations

```rust
use mlx_rs::nn;

let x = nn::relu(&x)?;
let x = nn::gelu(&x)?;
let x = nn::silu(&x)?;  // SiLU/Swish
let x = nn::softmax(&x, -1)?;
```

## Quantization

### INT8 Quantization

```rust
use mlx_rs::nn::QuantizedLinear;

// Convert existing Linear to QuantizedLinear
let quantized = QuantizedLinear::try_from_linear(
    linear,
    64,   // group_size
    8,    // bits (8 for INT8)
)?;

// Forward pass uses quantized_matmul internally
let output = quantized.forward(&input)?;
```

### Supported Bit Widths

- 2, 3, 4, 5, 6, 8 bits
- Group sizes: 32, 64, 128
- Mode: Affine quantization

## Tensor Operations

### Reshaping

```rust
let x = x.reshape(&[batch, seq, heads, head_dim])?;
let x = x.transpose_axes(&[0, 2, 1, 3])?;  // [batch, heads, seq, head_dim]
let x = x.flatten(None, None)?;
```

### Indexing

```rust
use mlx_rs::ops::indexing::IndexOp;

let slice = x.index((.., ..seq_len, ..));  // x[:, :seq_len, :]
let elem = x.index((.., .., 0_i32));       // x[:, :, 0]
```

### Splitting and Concatenating

```rust
// Split along axis
let chunks = x.split(num_chunks, axis)?;
let parts = x.split_axis(&[split_point], axis)?;

// Concatenate
let combined = ops::concatenate_axis(&[&a, &b], axis)?;
```

### Matrix Operations

```rust
let y = ops::matmul(&a, &b)?;
let y = ops::add(&a, &b)?;
let y = ops::multiply(&a, &b)?;  // Element-wise
let y = ops::divide(&a, &b)?;
```

## Performance Tips

### 1. Minimize eval() Calls

```rust
// BAD: Forces evaluation after each op
let a = op1(&x)?;
a.eval()?;
let b = op2(&a)?;
b.eval()?;

// GOOD: Let MLX fuse operations
let a = op1(&x)?;
let b = op2(&a)?;
b.eval()?;  // Single fused kernel
```

### 2. Avoid GPU→CPU Transfers

```rust
// BAD: Transfers data to CPU
let min = arr.min(None, None)?.item::<f32>();

// GOOD: Keep on GPU unless needed
// Only transfer for final output or debugging
```

### 3. Cache Reusable Computations

```rust
// BAD: Recompute every iteration
for i in 0..steps {
    let freqs = compute_rope_freqs(&ids)?;
    let out = forward_with_rope(&x, &freqs)?;
}

// GOOD: Compute once, reuse
let freqs = compute_rope_freqs(&ids)?;
for i in 0..steps {
    let out = forward_with_rope(&x, &freqs)?;
}
```

### 4. Use Release Builds

```bash
cargo run --release  # 10-100x faster than debug
```

## Common Patterns

### Attention

```rust
fn attention(q: &Array, k: &Array, v: &Array, scale: f32) -> Result<Array> {
    // q, k, v: [batch, heads, seq, head_dim]
    let scores = ops::matmul(q, &k.transpose_axes(&[0, 1, 3, 2])?)?;
    let scores = ops::divide(&scores, &array!(scale))?;
    let weights = ops::softmax_axis(&scores, -1, None)?;
    ops::matmul(&weights, v)
}
```

### SwiGLU MLP

```rust
fn swiglu_mlp(x: &Array, w_in: &Linear, w_out: &Linear, hidden: i32) -> Result<Array> {
    let proj = w_in.forward(x)?;
    let splits = proj.split_axis(&[hidden], -1)?;
    let gate = &splits[0];
    let up = &splits[1];
    let hidden = ops::multiply(&mlx_rs::nn::silu(gate)?, up)?;
    w_out.forward(&hidden)
}
```

### AdaLN Modulation

```rust
fn modulate(x: &Array, shift: &Array, scale: &Array) -> Result<Array> {
    // Expand for broadcasting: [batch, dim] -> [batch, 1, dim]
    let shift = shift.reshape(&[shift.dim(0), 1, shift.dim(-1)])?;
    let scale = scale.reshape(&[scale.dim(0), 1, scale.dim(-1)])?;

    // (1 + scale) * x + shift
    let scaled = ops::multiply(x, &ops::add(&array!(1.0f32), &scale)?)?;
    ops::add(&scaled, &shift)
}
```

## Error Handling

```rust
use mlx_rs::error::Exception;

// MLX operations return Result<T, Exception>
fn my_function(x: &Array) -> Result<Array, Exception> {
    let y = ops::add(x, &array!(1.0f32))?;
    Ok(y)
}

// For custom errors, use thiserror
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MyError {
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),

    #[error("Invalid shape")]
    InvalidShape,
}
```

## Critical Gotchas

<!-- Evolution: 2025-01-25 | source: qwen-image-mlx | author: debugging session -->

### Strided Views and as_slice()

**Problem:** `index()` and `transpose_axes()` create strided views, but `as_slice()` returns raw memory ignoring strides. This causes data corruption when reading pixel data or any indexed/transposed array.

**Symptom:** Image shows tiling pattern (e.g., 3x3 repeated tiles), or data appears shuffled/corrupted.

**Solution:** Force contiguity by flattening and reshaping before calling `as_slice()`:

```rust
// BAD: as_slice() ignores strides from index/transpose
let indexed = arr.index((.., .., 0, .., ..));  // Creates strided view
let data: Vec<f32> = indexed.as_slice().to_vec();  // WRONG DATA!

// BAD: transpose also creates strided view
let img = arr.transpose_axes(&[1, 2, 0])?;  // Creates strided view
let pixels: Vec<u8> = img.as_slice().to_vec();  // WRONG DATA!

// GOOD: Force contiguous by flatten + reshape
let indexed = arr.index((.., .., 0, .., ..));
let numel = indexed.dim(0) * indexed.dim(1) * indexed.dim(2) * indexed.dim(3);
let contiguous = indexed.reshape(&[numel])?.reshape(&[
    indexed.dim(0), indexed.dim(1), indexed.dim(2), indexed.dim(3)
])?;
contiguous.eval()?;
let data: Vec<f32> = contiguous.as_slice().to_vec();  // CORRECT!

// GOOD: Same pattern for transposed image data
let img = arr.transpose_axes(&[1, 2, 0])?;  // [C,H,W] -> [H,W,C]
let numel = img.dim(0) * img.dim(1) * img.dim(2);
let img = img.reshape(&[numel])?.reshape(&[img.dim(0), img.dim(1), img.dim(2)])?;
img.eval()?;
let pixels: Vec<u8> = img.as_slice().to_vec();  // CORRECT!
```

### AdaLayerNorm Scale/Shift Order

**Convention:** When splitting an embedding into scale and shift components, **scale is the first half, shift is the second half**.

```rust
// CORRECT: scale = first half, shift = second half
let half = emb.dim(-1) / 2;
let scale = emb.index((.., ..half)).expand_dims(1)?;   // First half
let shift = emb.index((.., half..)).expand_dims(1)?;   // Second half

// Apply: (1 + scale) * norm(x) + shift
let normed = layer_norm(x, eps)?;
let scaled = ops::multiply(&normed, &ops::add(&Array::from_f32(1.0), &scale)?)?;
ops::add(&scaled, &shift)
```

**Note:** This matches the convention in diffusers/mflux. Getting this wrong causes output values to be ~3x off in magnitude.

---

## Debugging

### Print Array Info

```rust
println!("Shape: {:?}", arr.shape());
println!("Dtype: {:?}", arr.dtype());

// Get values (forces eval + CPU transfer)
arr.eval()?;
let data: Vec<f32> = arr.as_slice().to_vec();
println!("Values: {:?}", &data[..10.min(data.len())]);
```

### Check for NaN

```rust
fn has_nan(arr: &Array) -> bool {
    let flat = arr.reshape(&[-1]).unwrap();
    flat.eval().unwrap();
    let data: Vec<f32> = flat.try_as_slice().unwrap().to_vec();
    data.iter().any(|x| x.is_nan())
}
```
