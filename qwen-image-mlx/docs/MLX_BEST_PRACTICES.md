# MLX Best Practices for Diffusion Models

*Lessons learned from optimizing Qwen-Image-MLX*

## The Golden Rule: Don't Fight Lazy Evaluation

MLX builds a computation graph and optimizes it holistically. Attempting to "optimize" by:
- Moving constants outside loops
- Caching Arrays in static variables
- Forcing early evaluation

...often **hurts** performance because it fragments the graph.

## What NOT to Do

### 1. Don't call `eval()` on every step

```rust
// BAD - eval every step
for step in 0..num_steps {
    latents = transformer.forward(&latents, ...)?;
    mlx_rs::transforms::eval([&latents])?;  // DON'T DO THIS
}

// GOOD - eval only when needed (progress reporting)
for step in 0..num_steps {
    latents = transformer.forward(&latents, ...)?;
    if (step + 1) % 5 == 0 {
        mlx_rs::transforms::eval([&latents])?;
        println!("Step {}", step + 1);
    }
}
```

### 2. Don't call `mlx_clear_cache()` frequently

Cache clearing adds significant overhead (tested: 86s vs 76s when clearing every 5 steps).

### 3. Don't pre-allocate scalar constants outside loops

```rust
// SLOWER - breaks graph fusion
let cfg_arr = Array::from_f32(cfg_scale);  // Outside loop
for step in 0..num_steps {
    let scaled = ops::multiply(&diff, &cfg_arr)?;
    ...
}

// FASTER - MLX handles this efficiently
for step in 0..num_steps {
    let cfg_arr = Array::from_f32(cfg_scale);  // Inside loop
    let scaled = ops::multiply(&diff, &cfg_arr)?;
    ...
}
```

### 4. Don't cache Arrays in static variables

```rust
// WON'T COMPILE - Array isn't Sync
static MODULATE_EPS: std::sync::OnceLock<Array> = std::sync::OnceLock::new();

// Even if it compiled, it would fragment the computation graph
```

## What TO Do

### 1. Use MLX's optimized fast operations

```rust
// Use fast::scaled_dot_product_attention instead of manual attention
let out = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, scale, None)?;

// Use fast::rms_norm instead of manual RMSNorm
let out = mlx_rs::fast::rms_norm(&x, &weight, eps)?;
```

### 2. Pre-compute expensive things once at initialization

```rust
// Good: Pre-compute RoPE frequencies at model creation
impl TimestepEmbedder {
    pub fn new(dim: i32) -> Self {
        let half_dim = dim / 2;
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| (-(i as f32) * (10000.0f32.ln()) / half_dim as f32).exp())
            .collect();
        let cached_freqs = Array::from_slice(&freqs, &[1, half_dim]);
        Self { cached_freqs, ... }
    }
}
```

### 3. Let MLX batch operations naturally

```rust
// Let the computation graph grow, then eval at the end
let final_latents = diffusion_loop(transformer, latents, num_steps)?;
mlx_rs::transforms::eval([&final_latents])?;  // Single eval at the end
```

### 4. Release unused resources to free memory

```rust
// Release text encoder after encoding (saves ~2-3GB)
drop(text_encoder);
// Note: Don't call mlx_clear_cache() - it adds overhead
```

## Custom Metal Kernels

Custom kernels are only worth it for operations MLX doesn't optimize well.

### When NOT to use custom kernels:
- For common operations (LayerNorm, elementwise ops, attention)
- When MLX already has a fast::* variant
- When the overhead of launching a custom kernel exceeds the fusion benefit

### Our experiment:
We implemented a fused modulate kernel:
```
Manual implementation: 4.31s per step
Fused Metal kernel: 4.52s per step (SLOWER)
```

MLX's built-in ops with lazy evaluation are already highly optimized.

## Memory Bandwidth is the Bottleneck

On Apple Silicon, memory bandwidth is often the limiting factor:

```
M2 Max specs:
- Memory bandwidth: 400 GB/s
- Model size (fp32): 13GB

Per forward pass (just weight reads):
- fp32: 13GB / 400GB/s = 32.5ms minimum
- int4: 1.6GB / 400GB/s = 4ms minimum
```

**Reality**: ~4s per step vs ~65ms theoretical minimum. The gap accounts for:
- Compute operations (attention, matmul, etc.)
- Activation memory reads/writes
- Kernel launch overhead
- Memory allocation
- Graph compilation

## Profiling Tips

1. **Compare with mflux** - Since mflux uses MLX too, it's a good reference
2. **Test changes in isolation** - One optimization at a time
3. **Measure total time** - Not just the operation you changed
4. **Watch for thermal throttling** - Can add variance to benchmarks

## Summary Table

| Optimization | Expected Impact | Actually Works? |
|--------------|-----------------|-----------------|
| Fast SDPA | 20-30% faster | ✅ Yes |
| Fast RMS Norm | 5-10% faster | ✅ Yes |
| RoPE pre-computation | Reduces overhead | ✅ Yes |
| Pre-allocate outside loop | Should help | ❌ No (slower) |
| Fused Metal kernel | Should help | ❌ No (slower) |
| Cache clearing | Should help | ❌ No (much slower) |
| Eval every step | More control | ❌ No (slower) |
| Step reduction | Linear speedup | ✅ Yes |
| Quantization | 1.5-2x faster | ⚠️ Quality tradeoff |

## References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [mflux Implementation](https://github.com/filipstrand/mflux)
- [OPTIMIZATION_FINDINGS.md](./OPTIMIZATION_FINDINGS.md) - Detailed optimization results
