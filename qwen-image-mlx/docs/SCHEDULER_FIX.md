# Qwen-Image Scheduler Fix

## Problem

The Rust Qwen-Image implementation was producing different results from the official HuggingFace diffusers pipeline due to incorrect scheduler formulas.

## Root Cause

Three issues were identified by comparing against the official diffusers implementation:

### 1. Incorrect Mu Calculation

**Before (Wrong):**
```rust
// Logarithmic interpolation + exp()
let m = (max_shift.ln() - base_shift.ln()) / (max_image_seq_len - base_image_seq_len);
let b = base_shift.ln() - m * base_image_seq_len;
let mu = (m * image_seq_len + b).exp();
```

**After (Correct):**
```rust
// Linear interpolation, NO ln/exp
let m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len);
let b = base_shift - m * base_image_seq_len;
let mu = m * image_seq_len + b;
```

The diffusers `calculate_shift` function uses simple linear interpolation:
```python
def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b  # LINEAR, no exp()!
    return mu
```

### 2. Incorrect Input Sigmas

**Before (Wrong):**
```rust
// Linear from 1.0 to shift_terminal
let t = 1.0 - (i as f32 / (num_steps - 1) as f32) * (1.0 - shift_terminal);
```

**After (Correct):**
```rust
// Linear from 1.0 to 1/num_steps (matching diffusers np.linspace)
let t = 1.0 - (i as f32 / (num_steps - 1) as f32) * (1.0 - 1.0 / num_steps as f32);
```

Diffusers pipeline (line 639):
```python
sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
```

### 3. Missing stretch_shift_to_terminal

**Before:** Not applied

**After (Correct):**
```rust
// Apply stretch_shift_to_terminal to scale endpoint to shift_terminal (0.02)
let last_sigma = shifted_sigmas[shifted_sigmas.len() - 1];
let scale_factor = (1.0 - last_sigma) / (1.0 - shift_terminal);
let sigmas: Vec<f32> = shifted_sigmas.iter().map(|&t| {
    1.0 - (1.0 - t) / scale_factor
}).collect();
```

Diffusers scheduler applies this after the exponential time shift:
```python
def stretch_shift_to_terminal(t, shift_terminal=0.02):
    one_minus_z = 1 - t
    scale_factor = one_minus_z[-1] / (1 - shift_terminal)
    stretched_t = 1 - (one_minus_z / scale_factor)
    return stretched_t
```

## Complete Scheduler Formula

The correct scheduler computation for Qwen-Image:

```rust
// Config values from Qwen-Image scheduler
let base_shift = 0.5f32;
let max_shift = 0.9f32;
let base_image_seq_len = 256.0f32;
let max_image_seq_len = 8192.0f32;
let shift_terminal = 0.02f32;
let image_seq_len = num_patches as f32;  // 1024 for 512x512

// Step 1: Compute mu (LINEAR interpolation)
let m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len);
let b = base_shift - m * base_image_seq_len;
let mu = m * image_seq_len + b;  // ~0.5387 for 1024 patches

// Step 2: Generate input sigmas
let input_sigmas: Vec<f32> = (0..num_steps).map(|i| {
    1.0 - (i as f32 / (num_steps - 1) as f32) * (1.0 - 1.0 / num_steps as f32)
}).collect();

// Step 3: Apply exponential time shift
let exp_mu = mu.exp();
let shifted_sigmas: Vec<f32> = input_sigmas.iter().map(|&t| {
    if t >= 1.0 { 1.0 }
    else if t <= 0.0 { 0.0 }
    else { exp_mu / (exp_mu + (1.0 / t - 1.0).powf(1.0)) }
}).collect();

// Step 4: Apply stretch_shift_to_terminal
let last_sigma = shifted_sigmas[shifted_sigmas.len() - 1];
let scale_factor = (1.0 - last_sigma) / (1.0 - shift_terminal);
let sigmas: Vec<f32> = shifted_sigmas.iter().map(|&t| {
    1.0 - (1.0 - t) / scale_factor
}).collect();
```

## Verification

After the fix, the Rust scheduler produces **identical timesteps** to diffusers:

| Step | Rust | Diffusers | Diff |
|------|------|-----------|------|
| 0 | 1000.00 | 1000.00 | 0.00 |
| 1 | 968.17 | 968.17 | 0.00 |
| 2 | 934.95 | 934.95 | 0.00 |
| 3 | 900.26 | 900.26 | 0.00 |
| ... | ... | ... | ... |
| 17 | 179.69 | 179.69 | 0.00 |
| 18 | 102.51 | 102.51 | 0.00 |
| 19 | 20.00 | 20.00 | 0.00 |

## Key Values

For 512x512 images (1024 patches, 20 steps):
- **mu**: 0.538710 (linear interpolation)
- **First sigma**: 1.0000
- **Last sigma**: 0.0200 (shift_terminal)
- **Timesteps**: [1000, 968.17, 934.95, ..., 179.69, 102.51, 20.00]

## References

- Diffusers pipeline: `diffusers/pipelines/qwenimage/pipeline_qwenimage.py`
- Diffusers scheduler: `diffusers/schedulers/scheduling_flow_match_euler_discrete.py`
- Fixed file: `examples/generate_qwen_image.rs` (lines 599-650)

## Date

Fixed: 2025-01-28
