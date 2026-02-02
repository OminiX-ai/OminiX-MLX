# VITS Training Fix Plan

## Summary of What's Wrong

My Rust implementation is **fundamentally broken** because:

1. **Generator loss is missing adversarial and feature matching components** - This is why `G: 0.0000` in all training logs
2. **Generator never receives signal from discriminator** - The two networks aren't connected during G training
3. **Forward pass structure is wrong** - Python does forward twice (once for D, once for G with gradients)

## Python Training Flow (Correct)

```python
# Step 1: Forward pass (autocast)
y_hat, kl_ssl, ids_slice, ... = net_g(ssl, spec, spec_lengths, text, text_lengths)

# Step 2: Compute mels
y_mel = slice_segments(mel, ids_slice, segment_size)  # GT mel sliced
y_hat_mel = mel_spectrogram(y_hat.squeeze(1))         # Generated mel

# Step 3: Slice real audio to match
y = slice_segments(y, ids_slice * hop_length, segment_size)

# Step 4: Discriminator step
y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())  # DETACH fake
loss_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)
optim_d.zero_grad()
loss_disc.backward()
optim_d.step()

# Step 5: Generator step (KEY DIFFERENCE!)
y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)  # NO detach - gradients flow through!
loss_mel = F.l1_loss(y_mel, y_hat_mel) * c_mel          # 45
loss_kl = kl_loss(...) * c_kl                           # 1.0
loss_fm = feature_loss(fmap_r, fmap_g)                  # 2.0 (implicit)
loss_gen = generator_loss(y_d_hat_g)                    # Adversarial
loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl + loss_kl
optim_g.zero_grad()
loss_gen_all.backward()
optim_g.step()
```

## My Rust Implementation (Broken)

```rust
// Forward pass
let (y_hat, ...) = generator.forward_train(...);

// D step - OK
let (d_real, d_fake, _, _) = discriminator.forward_ex(y_real, &y_hat);
// ... discriminator loss and update

// G step - BROKEN!
// Only computes mel loss and KL loss
// NEVER passes y_hat through discriminator
// loss_gen = 0, loss_fm = 0
let loss = loss_mel * c_mel + loss_kl * c_kl;  // Missing adversarial!
```

## Key Issues to Fix

### Issue 1: Generator Loss Missing Adversarial Component
**Python:**
```python
loss_gen = mean((1 - D(y_hat))^2)  # Wants D to output 1 for fake
```
**My code:** Not computed at all

### Issue 2: Feature Matching Loss Missing
**Python:**
```python
loss_fm = sum(|fmap_real - fmap_fake|) * 2  # Intermediate layer outputs
```
**My code:** Not computed at all

### Issue 3: Discriminator Not Called During G Step
The discriminator must be called with **non-detached** y_hat during G step so gradients flow back to generator.

### Issue 4: Segment Slicing
Python uses `ids_slice` from the model to slice audio/mel to fixed `segment_size`. I'm not doing this properly.

## Loss Function Reference

### discriminator_loss (Python)
```python
def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = mean((1 - dr)^2)  # Real should be 1
        g_loss = mean(dg^2)         # Fake should be 0
        loss += r_loss + g_loss
    return loss
```

### generator_loss (Python)
```python
def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        loss += mean((1 - dg)^2)  # Generator wants D to output 1 for fake
    return loss
```

### feature_loss (Python)
```python
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.detach()  # Don't backprop through real
            loss += mean(|rl - gl|)
    return loss * 2
```

### kl_loss (Python)
```python
def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p)^2) * exp(-2 * logs_p)
    kl = sum(kl * z_mask)
    return kl / sum(z_mask)
```

## Hyperparameters (from s2.json)

| Parameter | Value |
|-----------|-------|
| learning_rate | 0.0001 |
| betas | [0.8, 0.99] |
| eps | 1e-9 |
| c_mel | 45 |
| c_kl | 1.0 |
| segment_size | 20480 |
| lr_decay | 0.999875 |

## Implementation Plan

### Step 1: Fix Generator Training Step

The G step must:
1. Forward through generator to get y_hat
2. Forward through discriminator with y_hat (NOT detached)
3. Compute all losses:
   - `loss_gen` = adversarial loss from discriminator outputs
   - `loss_fm` = feature matching loss from discriminator intermediate outputs
   - `loss_mel` = L1 mel reconstruction loss
   - `loss_kl` = KL divergence loss
4. Total: `loss_gen + loss_fm + loss_mel * 45 + loss_kl * 1.0`
5. Backprop and update generator

### Step 2: Fix Discriminator Output Format

Discriminator needs to return:
- `y_d_hat_r`: List of real audio discriminator outputs
- `y_d_hat_g`: List of fake audio discriminator outputs
- `fmap_r`: List of intermediate feature maps for real
- `fmap_g`: List of intermediate feature maps for fake

### Step 3: Implement Proper Loss Functions

Create correct implementations:
```rust
fn generator_loss(disc_outputs: &[Array]) -> Array {
    // sum of mean((1 - dg)^2) for each discriminator
}

fn feature_loss(fmap_r: &[Vec<Array>], fmap_g: &[Vec<Array>]) -> Array {
    // sum of mean(|rl - gl|) for each layer, * 2
}

fn discriminator_loss(d_real: &[Array], d_fake: &[Array]) -> Array {
    // sum of mean((1-dr)^2) + mean(dg^2)
}
```

### Step 4: Fix Training Loop Structure

```rust
pub fn train_step(&mut self, batch: &VITSBatch) -> Result<VITSLosses, Error> {
    // 1. Generator forward (outside grad computation)
    let (y_hat, ...) = self.generator.forward_train(...)?;

    // 2. Discriminator step (y_hat detached)
    let loss_d = self.train_discriminator_step(&y_real, &y_hat.stop_gradient())?;

    // 3. Generator step (NEW - includes discriminator forward)
    let (loss_gen, loss_fm, loss_mel, loss_kl) = self.train_generator_step_v2(...)?;

    Ok(losses)
}

fn train_generator_step_v2(&mut self, ...) -> Result<...> {
    let loss_fn = |gen: &mut SynthesizerTrn, ...| {
        // Forward through G
        let (y_hat, z_p, m_p, logs_p, ...) = gen.forward_train(...)?;

        // Forward through D (no detach - gradients flow!)
        let (_, d_fake, _, fmap_g) = self.discriminator.forward_ex(&y_real, &y_hat)?;
        let (_, _, fmap_r, _) = self.discriminator.forward_ex(&y_real, &y_real)?;

        // Compute losses
        let loss_gen = generator_loss(&d_fake);
        let loss_fm = feature_loss(&fmap_r, &fmap_g);
        let loss_mel = mel_l1_loss(&mel_real, &mel_fake) * 45.0;
        let loss_kl = kl_loss(...) * 1.0;

        Ok(loss_gen + loss_fm + loss_mel + loss_kl)
    };

    // value_and_grad on generator
    let (loss, grads) = value_and_grad(loss_fn)(&mut self.generator, ...)?;
    self.optim_g.update(&mut self.generator, &grads)?;
}
```

### Step 5: Verify Gradient Flow

Add logging to verify:
- Discriminator gradients are non-zero during D step
- Generator gradients are non-zero during G step
- Loss values match expected ranges:
  - loss_d: ~2-6 initially
  - loss_gen: ~2-5
  - loss_fm: ~5-15
  - loss_mel: ~1-3 (before *45)
  - loss_kl: ~0.5-2

## Files to Modify

1. `src/training/vits_trainer.rs` - Main training loop
2. `src/training/vits_loss.rs` - Loss functions (add generator_loss, feature_loss)
3. `src/models/discriminator.rs` - Verify forward returns feature maps

## Testing

After fixes:
1. Train for 1 epoch, verify all losses are non-zero
2. Loss values should be in expected ranges
3. Weight changes should be significant (not 0.0001)
4. Generated audio should improve, not stay same/degrade

## Expected Training Log (After Fix)

```
Step 0 | D: 5.2 | G: 3.8 | FM: 12.4 | Mel: 2.1 | KL: 1.2
Step 10 | D: 3.1 | G: 2.5 | FM: 8.2 | Mel: 1.8 | KL: 0.9
...
```

Compare to current broken output:
```
Step 0 | D: 5.9 | G: 0.0 | FM: 0.0 | Mel: 1.1 | KL: 5.0  # G/FM are 0!
```
