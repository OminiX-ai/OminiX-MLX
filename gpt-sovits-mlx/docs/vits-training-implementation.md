# VITS Training Implementation Guide

This document describes the VITS (SoVITS) training implementation in Rust + MLX, including key differences from the Python GPT-SoVITS implementation and how they were resolved.

## Overview

The VITS training pipeline performs GAN-based finetuning of the HiFi-GAN decoder for voice cloning. For fewshot voice cloning, we freeze most of the model and only train:

- `dec` - HiFi-GAN decoder (generates audio from latents)
- `ref_enc` - Reference encoder (extracts voice style from reference audio)
- `ssl_proj` - SSL projection layer (adapts HuBERT features)

## Training Architecture

```
                                    ┌─────────────┐
                                    │  ref_enc    │ ◄── spec[:,:704] (SAME as enc_q input!)
                                    └──────┬──────┘
                                           │ ge (style embedding)
                                           ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ HuBERT   │───►│ ssl_proj │───►│ quantizer│───►│  enc_p   │───►│   flow   │
│ features │    │          │    │   (VQ)   │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                     │                               │               │
                     │ commit_loss                   │ m_p, logs_p   │ z_p
                     ▼                               ▼               ▼

┌──────────┐    ┌──────────┐                                    ┌──────────┐
│   spec   │───►│  enc_q   │───────────────────────────────────►│   dec    │───► y_hat
│          │    │          │  z, m_q, logs_q                    │ (HiFiGAN)│
└──────────┘    └──────────┘                                    └──────────┘
```

## Loss Functions

The total generator loss matches Python exactly:

```
loss_total = loss_gen + loss_fm + loss_mel * c_mel + loss_kl * c_kl + commit_loss
```

Where:
- `loss_gen`: Adversarial loss (LSGAN) - generator wants discriminator to output 1
- `loss_fm`: Feature matching loss - L1 distance of discriminator feature maps × 2
- `loss_mel`: Mel reconstruction loss - L1 between real and generated mel spectrograms
- `loss_kl`: KL divergence between posterior and prior distributions
- `commit_loss`: VQ commitment loss (`kl_ssl` in Python code)

### Loss Weights (from Python s2.json)

| Weight | Value | Description |
|--------|-------|-------------|
| `c_mel` | 45.0 | Mel reconstruction weight |
| `c_kl` | 1.0 | KL divergence weight |
| `commit_loss` | 1.0 | VQ commitment weight |

## Critical Implementation Details

### 1. Feature Matching Loss - Stop Gradient

Python detaches real features so gradients only flow through fake features:

```python
# Python (losses.py line 11)
rl = rl.float().detach()  # CRITICAL: detach real features
```

Rust implementation:

```rust
// vits_loss.rs
let real_detached = stop_gradient(real)?;  // CRITICAL!
let diff = real_detached.subtract(fake)?;
```

### 2. Mel Loss Computation Order

Python computes mel from full spectrogram, THEN slices:

```python
# Python (s2_train.py lines 342-352)
mel = spec_to_mel_torch(spec, ...)           # Full spec → mel
y_mel = commons.slice_segments(mel, ids_slice, ...)  # Then slice
y_hat_mel = mel_spectrogram_torch(y_hat, ...)        # Generated audio → mel
```

Our implementation matches this:

```rust
// vits_trainer.rs
let mel_full = spec_to_mel(spec_f, &mel_config)?;
let mel_real = slice_mel_segments(&mel_full, &ids_slice, segment_frames)?;
let mel_fake = mel_spectrogram_mlx(&y_hat_sliced, &mel_config)?;
```

### 3. Reference Encoder Input (Training)

**Critical**: In training, `ref_enc` uses the SAME spectrogram as `enc_q`, NOT a separate reference:

```python
# Python (models.py line 921)
ge = self.ref_enc(y[:,:704] * y_mask, y_mask)  # y IS the spec!
```

```rust
// vits.rs forward_train()
let spec_sliced = spec.index((.., ..704, ..));
let spec_masked = spec_sliced.multiply(&y_mask)?;
let ge = self.ref_enc.forward(&spec_masked)?;
```

### 4. Random Segment Slicing

Training randomly slices segments from z for memory efficiency:

```python
# Python
z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, segment_size)
o = self.dec(z_slice, g=ge)
```

The `ids_slice` must be used consistently:
- Slice `z` for decoder input
- Slice full audio `y` for discriminator (using `ids_slice * hop_length`)
- Slice mel spectrogram for mel loss

## Optimizer Configuration

Python uses specific AdamW hyperparameters that differ from defaults:

| Parameter | Python | PyTorch Default | MLX Default |
|-----------|--------|-----------------|-------------|
| beta1 | 0.8 | 0.9 | 0.9 |
| beta2 | 0.99 | 0.999 | 0.999 |
| eps | 1e-9 | 1e-8 | 1e-8 |
| learning_rate | 1e-4 | - | - |

The lower beta1 (0.8) means less momentum - the optimizer adapts faster to new gradients.
The lower beta2 (0.99) means faster adaptation to gradient variance.

```rust
// vits_trainer.rs
let optim_g = AdamWBuilder::new(config.learning_rate_g)
    .betas((config.beta1, config.beta2))  // (0.8, 0.99)
    .eps(config.eps)                       // 1e-9
    .build()?;
```

## Gradient Handling

Python uses AMP (Automatic Mixed Precision) with a GradScaler that automatically handles gradient explosions. It does NOT explicitly clip gradients:

```python
# Python (s2_train.py)
grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)  # None = no clipping!
```

Without AMP, we need explicit gradient clipping to prevent explosions:

```rust
// vits_trainer.rs - Default config
grad_clip: 100.0,  // Moderate clipping for stability without AMP
```

## Training Configuration

Default configuration matching Python:

```rust
VITSTrainingConfig {
    learning_rate_g: 1e-4,
    learning_rate_d: 1e-4,
    batch_size: 4,
    segment_size: 20480,  // 640ms @ 32kHz
    c_mel: 45.0,
    c_kl: 1.0,
    c_fm: 2.0,
    grad_clip: 100.0,
    beta1: 0.8,
    beta2: 0.99,
    eps: 1e-9,
    ...
}
```

## Fewshot Training Workflow

1. **Load pretrained weights**:
   ```rust
   trainer.load_generator_weights(&pretrained_path)?;
   ```

2. **Freeze non-decoder layers**:
   ```rust
   trainer.freeze_non_decoder_layers();
   // Unfreezes: dec, ref_enc, ssl_proj
   // Keeps frozen: enc_p, enc_q, flow, quantizer
   ```

3. **Train on target voice data**:
   ```rust
   for batch in dataset.iter_batches(...) {
       let losses = trainer.train_step(&batch)?;
   }
   ```

4. **Save finetuned weights**:
   ```rust
   trainer.save_generator(&output_path)?;
   ```

## Weight Format

### Pretrained Weights (from Python)

Uses weight normalization with separate `weight_g` and `weight_v`:
```
dec.resblocks.0.convs1.0.weight_g: [256, 1, 1]
dec.resblocks.0.convs1.0.weight_v: [256, 256, 3]
```

### MLX Weights (after loading)

Weight normalization is merged: `weight = weight_g * weight_v / ||weight_v||`
```
dec.resblocks.0.convs1.0.weight: [256, 256, 3]  # Merged
```

### Saved Finetuned Weights

Saved in PyTorch format for compatibility:
- Conv1d: MLX `[out, kernel, in]` → PyTorch `[out, in, kernel]`
- ConvTranspose1d: MLX `[out, kernel, in]` → PyTorch `[in, out, kernel]`

## Debugging Tips

### Check if training is having effect

Compare weight changes between pretrained and finetuned:

```python
from safetensors.numpy import load_file
import numpy as np

pre = load_file("pretrained.safetensors")
fin = load_file("finetuned.safetensors")

for key in set(pre.keys()) & set(fin.keys()):
    diff = np.abs(fin[key] - pre[key]).mean()
    rel = diff / (np.abs(pre[key]).mean() + 1e-8) * 100
    if rel > 1.0:  # > 1% change
        print(f"{key}: {rel:.2f}% change")
```

### Expected behavior

- Decoder biases should change 5-15%
- Later layers (resblocks 10-14) change more than early layers
- ssl_proj should change ~0.5-1%
- Frozen layers should have 0% change

### Common issues

1. **NaN losses**: Reduce learning rate or increase grad_clip
2. **No weight changes**: Check that freeze/unfreeze is working correctly
3. **Loss not decreasing**: Verify data loading and mel computation

## References

- Python implementation: `moyoyo_tts/s2_train.py`
- Loss functions: `moyoyo_tts/module/losses.py`
- Model architecture: `moyoyo_tts/module/models.py`
- Training config: `moyoyo_tts/configs/s2.json`
