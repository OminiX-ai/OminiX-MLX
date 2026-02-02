# VITS Training Fix - Development Plan

## Goal
Fix the amplitude issue in Rust VITS training so finetuned models produce audio with proper amplitude (>15000 max, matching pretrained).

## Status Summary

| Phase | Task | Status |
|-------|------|--------|
| Phase 1 | 1.1 Lower LR | ✅ Complete |
| Phase 1 | 1.2 Add regularization | ✅ Complete |
| Phase 1 | 1.3 Update example | ✅ Complete |
| Phase 2 | 2.1 WeightNorm modules | ✅ Complete |
| Phase 2 | 2.2 Update Generator | ⏳ TODO |
| Phase 2 | 2.3 Update loading | ⏳ TODO |
| Phase 2 | 2.4 Update saving | ⏳ TODO |

## Phase 1: Quick Fixes ✅ COMPLETE

### Task 1.1: Lower Default Learning Rate ✅
**File:** `src/training/vits_trainer.rs`

Changed default learning rate from 1e-4 to 1e-5 for both generator and discriminator.

### Task 1.2: Add Pretrained Weight Regularization ✅
**File:** `src/training/vits_trainer.rs`

Implemented:
- `pretrained_reg_strength` config field (default: 0.001)
- `pretrained_weights` field in VITSTrainer
- `load_generator_weights_with_regularization()` method
- `compute_pretrained_reg_loss()` method
- `compute_pretrained_reg_gradients()` method
- Regularization gradients integrated into training step
- `loss_reg` field in VITSLosses

### Task 1.3: Update train_vits Example ✅
**File:** `examples/train_vits.rs`

Updated:
- Default learning rates changed to 1e-5
- Added `--pretrained-reg` CLI argument
- Uses `load_generator_weights_with_regularization()` for finetuning
- Logging includes regularization loss

## Phase 2: Weight Normalization (Partial)

### Task 2.1: Create WeightNorm Module Types ✅
**New File:** `src/nn/weight_norm.rs`

Created:
- `WeightNormConv1d` - Conv1d with separate weight_g and weight_v
- `WeightNormConvTranspose1d` - ConvTranspose1d with weight normalization
- Both implement `weight()` method: `weight = g * v / ||v||`
- Both implement `forward()` method with computed weights
- Tests for shape verification and forward pass

### Task 2.2: Update HiFiGANGenerator ⏳ TODO
**File:** `src/models/vits.rs` (HiFiGANGenerator)

Need to replace Conv1d/ConvTranspose1d with weight-normalized versions:

```rust
pub struct HiFiGANGenerator {
    conv_pre: WeightNormConv1d,      // was: nn::Conv1d
    ups: Vec<WeightNormConvTranspose1d>,  // was: Vec<nn::ConvTranspose1d>
    resblocks: Vec<ResBlock>,        // keep as-is (resblocks use regular conv)
    conv_post: WeightNormConv1d,     // was: nn::Conv1d
    cond: nn::Conv1d,                // keep as-is (conditioning layer)
}
```

### Task 2.3: Update Weight Loading ⏳ TODO
**File:** `src/models/vits.rs` (load_vits_weights)

Need to modify to load weight_g/weight_v directly instead of computing weight:

```rust
// For weight-normalized layers
for (i, up) in model.dec.ups.iter_mut().enumerate() {
    if let Some(g) = get_weight(&format!("dec.ups.{}.weight_g", i)) {
        up.weight_g = Param::new(g);
    }
    if let Some(v) = get_weight(&format!("dec.ups.{}.weight_v", i)) {
        // Transpose v from PyTorch to MLX format
        up.weight_v = Param::new(transpose_convt(v)?);
    }
}
```

### Task 2.4: Update Weight Saving ⏳ TODO
**File:** `src/training/vits_trainer.rs` (save_generator)

Need to save weight_g and weight_v separately:

```rust
// For weight-normalized layers
for (i, up) in self.generator.dec.ups.iter().enumerate() {
    let g = up.weight_g.as_ref();
    let v = transpose_convt_to_pytorch(up.weight_v.as_ref())?;

    g_converted.insert(format!("dec.ups.{}.weight_g", i), g.clone());
    g_converted.insert(format!("dec.ups.{}.weight_v", i), v);
}
```

## Testing Plan

### Phase 1 Tests
1. **Amplitude Test**: Train 100 steps, verify step50 amplitude > 15000
2. **Convergence Test**: Verify loss decreases normally with lower LR
3. **Quality Test**: Generate audio, compare with pretrained subjectively

### Phase 2 Tests
1. **Weight Loading Test**: Load pretrained, verify weight computation matches
2. **Training Test**: Full training run, compare with Phase 1 results
3. **Compatibility Test**: Ensure saved weights work with Python inference

## Files Modified/Created

| File | Status | Changes |
|------|--------|---------|
| `src/training/vits_trainer.rs` | ✅ Modified | Lower LR, add regularization |
| `examples/train_vits.rs` | ✅ Modified | Use new loading method, add CLI args |
| `src/nn/mod.rs` | ✅ Created | New module for weight normalization |
| `src/nn/weight_norm.rs` | ✅ Created | WeightNormConv1d, WeightNormConvTranspose1d |
| `src/lib.rs` | ✅ Modified | Export nn module |
| `src/models/vits.rs` | ⏳ TODO | Update HiFiGANGenerator to use weight-norm layers |

## Next Steps

1. **Test Phase 1 fixes**: Run training with new LR and regularization, verify amplitude improves
2. **Complete Phase 2 (optional)**: If Phase 1 fixes are insufficient, complete weight normalization:
   - Update HiFiGANGenerator struct
   - Update weight loading to handle weight_g/weight_v
   - Update weight saving to preserve weight_g/weight_v

## Success Criteria

1. Finetuned model amplitude within 30% of pretrained
2. Audio quality subjectively matches pretrained
3. Training is stable (no NaN, no divergence)
4. Can successfully finetune on 1-minute voice sample
