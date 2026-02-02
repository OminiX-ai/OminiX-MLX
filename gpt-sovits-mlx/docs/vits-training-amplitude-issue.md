# VITS Training Amplitude Issue Analysis

## Problem Summary

Rust-trained VITS weights produce audio with 7-10x lower amplitude compared to pretrained weights, even though the individual weight values appear similar (correlation > 0.95).

## Observed Behavior

| Checkpoint | Max Amplitude | UPS.0 Weight Sum |
|------------|---------------|------------------|
| Pretrained | 24,201 | -101.88 |
| Step 50 | 1,926 | -151.24 |
| Step 100 | 3,485 | -180.41 |
| Final (4 epochs) | 3,423 | -180.48 |

The amplitude issue appears from the very first training steps (step 50), not just at the end.

## Root Cause Analysis

### 1. Missing Weight Normalization

**PyTorch VITS Training:**
```python
# Weights stored as separate g (magnitude) and v (direction)
weight_g: [out_channels, 1, 1]  # learnable magnitude
weight_v: [out_channels, in_channels, kernel]  # learnable direction

# Actual weight computed on-the-fly
weight = g * v / ||v||  # magnitude is constrained by g
```

**Rust MLX Training:**
```rust
// Weights stored as combined weight (no normalization)
weight: [out_channels, kernel, in_channels]  // directly learnable

// No magnitude constraint - weights can drift freely
```

**Impact:** Without weight normalization, the optimizer can change both the direction AND magnitude of weights freely. This leads to:
- Unbounded weight magnitude drift
- Loss of the implicit regularization that weight_norm provides
- Weights that produce valid loss values but wrong audio amplitude

### 2. High Learning Rate

The trainer uses `learning_rate_g: 1e-4` (or 2e-4), which is appropriate for training from scratch with weight normalization, but too aggressive for:
- Finetuning (typically uses 1e-5 to 1e-6)
- Training without weight normalization (weights change faster)

### 3. Weight Sum Drift Pattern

The ConvTranspose1d (upsample) layers show consistent negative drift:

| Layer | Pretrained Sum | Finetuned Sum | Drift |
|-------|----------------|---------------|-------|
| ups.0 | -101.88 | -180.48 | -77% |
| ups.1 | +87.88 | +127.80 | +45% |
| ups.2 | +34.81 | +38.15 | +10% |
| ups.3 | +20.67 | +20.05 | -3% |
| ups.4 | +1.87 | +2.04 | +9% |

The first two upsample layers (8x each) drift the most. These have the most parameters and highest capacity for drift.

## Technical Details

### Weight Format Flow

```
Pretrained (safetensors)     Training (MLX)           Saved (safetensors)
─────────────────────────────────────────────────────────────────────────
weight_g + weight_v    →    computed weight    →    direct weight
[PyTorch format]            [MLX format]            [PyTorch format]

load_vits_weights:          training step:          save_generator:
weight_norm_convt(g,v)      optimizer.update()      transpose_to_pytorch()
then transpose_convt()      (no normalization)
```

### Why Correlation is High but Output is Wrong

- Per-element changes are small (~0.001 mean absolute difference)
- But changes are biased (mean shift of -3.7e-5 per element)
- With 2M+ elements, this creates large sum differences
- Decoder is sensitive to cumulative weight biases across layers

## Proposed Solutions

### Solution 1: Lower Learning Rate (Quick Fix)

**Effort:** Low (config change)
**Effectiveness:** Medium

```rust
// In VITSTrainingConfig::default()
learning_rate_g: 1e-5,  // was 1e-4
learning_rate_d: 1e-5,  // was 1e-4
```

Pros:
- Simple to implement
- Reduces drift rate

Cons:
- Doesn't fix underlying issue
- May need many more steps to learn
- Still no magnitude constraint

### Solution 2: L2 Regularization Towards Pretrained (Better Fix)

**Effort:** Medium (code change)
**Effectiveness:** High

Add regularization loss that penalizes deviation from pretrained weights:

```rust
// In generator loss computation
let pretrained_reg_loss = compute_pretrained_regularization(
    &current_weights,
    &pretrained_weights,
    reg_strength: 0.01,
)?;

let total_loss = loss_gen
    .add(&loss_fm)?
    .add(&loss_mel)?
    .add(&loss_kl)?
    .add(&pretrained_reg_loss)?;  // NEW
```

Pros:
- Keeps weights close to pretrained
- Allows learning while preventing drift
- Simple to implement

Cons:
- Requires storing pretrained weights during training
- Adds hyperparameter (reg_strength)

### Solution 3: Implement Weight Normalization (Best Fix)

**Effort:** High (architecture change)
**Effectiveness:** Best

Implement proper weight normalization in MLX:

```rust
struct WeightNormConv1d {
    weight_g: Param<Array>,  // [out, 1, 1]
    weight_v: Param<Array>,  // [out, kernel, in]
    bias: Option<Param<Array>>,
}

impl WeightNormConv1d {
    fn weight(&self) -> Array {
        let v_norm = self.weight_v.l2_norm(axes: [-2, -1], keepdim: true);
        &self.weight_g * &self.weight_v / v_norm
    }

    fn forward(&self, x: &Array) -> Array {
        conv1d(x, &self.weight(), self.bias.as_ref())
    }
}
```

Pros:
- Matches PyTorch training exactly
- Proper magnitude/direction separation
- Best training stability

Cons:
- Significant refactoring of model architecture
- Need to update both model and trainer
- More complex weight loading/saving

## Recommended Approach

**Phase 1 (Immediate):** Apply Solution 1 + 2
1. Lower learning rate to 1e-5
2. Add simple L2 regularization towards pretrained

**Phase 2 (Later):** Implement Solution 3
1. Create WeightNormConv1d/WeightNormConvTranspose1d modules
2. Update HiFiGANGenerator to use weight-normalized layers
3. Update trainer to save/load weight_g and weight_v separately

## Verification Plan

After fix, verify:
1. Training step 50 amplitude should be close to pretrained (>15000)
2. Final checkpoint amplitude should be >10000
3. Weight sums should stay within 20% of pretrained
4. Audio quality should match Python-trained finetuning

## Files to Modify

| File | Changes |
|------|---------|
| `src/training/vits_trainer.rs` | Add regularization, lower LR |
| `src/training/config.rs` | Add reg_strength config |
| `src/models/vits.rs` | (Phase 2) Add weight_norm modules |
| `src/models/hifigan.rs` | (Phase 2) Use weight_norm layers |
