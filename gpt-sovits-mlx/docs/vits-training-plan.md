# VITS Training in Rust + MLX - Development Plan

## Overview

Implement VITS (SoVITS) training in Rust + MLX, enabling full voice cloning training on Apple Silicon using MLX acceleration.

## Architecture Overview

VITS is a GAN-based vocoder with these components:

```
Training Data Flow:
┌─────────────┐      ┌──────────────┐      ┌───────────────┐
│ SSL Features │─────>│ TextEncoder  │─────>│ Posterior     │
│ (HuBERT)     │      │ (enc_p)      │      │ Encoder (enc_q)│
└─────────────┘      └──────────────┘      └───────────────┘
                                                   │
                           ┌───────────────────────┘
                           │ z (latent)
                           ▼
                    ┌─────────────┐
                    │  HiFi-GAN   │
                    │  Generator  │──────> y_hat (audio)
                    └─────────────┘
                           │
                           ▼
              ┌───────────────────────┐
              │ Multi-Period/Scale    │
              │    Discriminator      │
              └───────────────────────┘
```

## Components to Implement

### 1. VITS Generator `forward_train()` Method

The generator needs a training-specific forward pass that exposes intermediate values for loss computation.

**Location**: `src/models/vits.rs`

**Interface**:
```rust
impl SynthesizerTrn {
    /// Training forward pass - returns audio and intermediate latents for loss computation
    pub fn forward_train(
        &mut self,
        ssl_features: &Array,      // [batch, ssl_dim, ssl_len] - HuBERT features
        spec: &Array,              // [batch, n_fft/2+1, spec_len] - Target spectrogram
        spec_lengths: &Array,      // [batch] - Spectrogram lengths
        text: &Array,              // [batch, text_len] - Phoneme indices
        text_lengths: &Array,      // [batch] - Text lengths
    ) -> Result<VITSTrainOutput, Exception> {
        // 1. Encode SSL -> quantized features
        // 2. Encode text + SSL via enc_p (TextEncoder)
        // 3. Encode spectrogram via enc_q (PosteriorEncoder) to get z
        // 4. Apply normalizing flow
        // 5. Random segment slice for efficient training
        // 6. Decode z slice to audio via HiFi-GAN
        // Return audio + all latents needed for loss
    }
}

pub struct VITSTrainOutput {
    pub y_hat: Array,          // Generated audio [batch, 1, segment_len]
    pub ids_slice: Array,      // Slice indices for segment extraction
    pub z_mask: Array,         // Mask for latent
    pub z: Array,              // Posterior latent
    pub z_p: Array,            // Flow-transformed latent
    pub m_p: Array,            // Prior mean
    pub logs_p: Array,         // Prior log-variance
    pub m_q: Array,            // Posterior mean
    pub logs_q: Array,         // Posterior log-variance
    pub kl_ssl: f32,           // SSL KL divergence term
}
```

**Key sub-components needed**:
- `PosteriorEncoder` (enc_q) - Encode spectrogram to latent distribution
- Random segment slicing for memory-efficient training
- Flow forward pass (not reverse)

### 2. Multi-Period/Scale Discriminator

**Location**: `src/models/discriminator.rs` (new file)

```rust
/// Period discriminator for different audio periodicities
pub struct DiscriminatorP {
    convs: Vec<nn::Conv2d>,
    conv_post: nn::Conv2d,
    period: i32,
}

impl DiscriminatorP {
    pub fn new(period: i32) -> Result<Self, Exception>;

    /// Forward pass returns (prediction, feature_maps)
    pub fn forward(&mut self, x: &Array) -> Result<(Array, Vec<Array>), Exception>;
}

/// Scale discriminator for multi-scale analysis
pub struct DiscriminatorS {
    convs: Vec<nn::Conv1d>,
    conv_post: nn::Conv1d,
}

impl DiscriminatorS {
    pub fn new() -> Result<Self, Exception>;

    pub fn forward(&mut self, x: &Array) -> Result<(Array, Vec<Array>), Exception>;
}

/// Combined multi-period discriminator
pub struct MultiPeriodDiscriminator {
    discriminators: Vec<Box<dyn Discriminator>>,
}

impl MultiPeriodDiscriminator {
    pub fn new() -> Result<Self, Exception> {
        // periods = [2, 3, 5, 7, 11] + 1 scale discriminator
    }

    /// Forward pass on real and generated audio
    pub fn forward(
        &mut self,
        y: &Array,      // Real audio
        y_hat: &Array,  // Generated audio
    ) -> Result<DiscriminatorOutput, Exception>;
}

pub struct DiscriminatorOutput {
    pub y_d_real: Vec<Array>,       // Real predictions
    pub y_d_fake: Vec<Array>,       // Fake predictions
    pub fmap_real: Vec<Vec<Array>>, // Real feature maps
    pub fmap_fake: Vec<Vec<Array>>, // Fake feature maps
}
```

### 3. Loss Functions

**Location**: `src/training/vits_loss.rs` (new file)

```rust
/// Generator adversarial loss (LSGAN)
/// loss = mean((1 - D(G(x)))^2)
pub fn generator_loss(disc_outputs: &[Array]) -> Result<Array, Exception>;

/// Discriminator adversarial loss (LSGAN)
/// loss = mean((1 - D(real))^2) + mean(D(fake)^2)
pub fn discriminator_loss(
    disc_real: &[Array],
    disc_fake: &[Array],
) -> Result<Array, Exception>;

/// Feature matching loss (L1 on discriminator features)
pub fn feature_loss(
    fmap_real: &[Vec<Array>],
    fmap_fake: &[Vec<Array>],
) -> Result<Array, Exception>;

/// KL divergence loss for VAE
/// KL(q(z|x) || p(z|c))
pub fn kl_loss(
    z_p: &Array,
    logs_q: &Array,
    m_p: &Array,
    logs_p: &Array,
    z_mask: &Array,
) -> Result<Array, Exception>;
```

### 4. Mel Spectrogram Computation in MLX

**Location**: `src/audio/mel.rs` (new file)

```rust
/// STFT parameters for mel spectrogram
pub struct STFTConfig {
    pub n_fft: i32,           // 2048
    pub hop_size: i32,        // 640
    pub win_size: i32,        // 2048
    pub sampling_rate: i32,   // 32000
    pub n_mels: i32,          // 128
    pub fmin: f32,            // 0.0
    pub fmax: f32,            // None (Nyquist)
}

impl Default for STFTConfig {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            hop_size: 640,
            win_size: 2048,
            sampling_rate: 32000,
            n_mels: 128,
            fmin: 0.0,
            fmax: 16000.0,
        }
    }
}

/// Compute STFT using MLX
pub fn stft(
    y: &Array,              // [batch, time]
    n_fft: i32,
    hop_length: i32,
    win_length: i32,
) -> Result<Array, Exception> {
    // 1. Pad audio: reflect padding
    // 2. Frame into overlapping windows
    // 3. Apply Hann window
    // 4. FFT each frame
    // 5. Return magnitude: [batch, n_fft/2+1, frames]
}

/// Mel filterbank (precomputed)
pub fn mel_filterbank(
    n_fft: i32,
    n_mels: i32,
    sr: i32,
    fmin: f32,
    fmax: f32,
) -> Array;

/// Compute mel spectrogram
pub fn mel_spectrogram(
    y: &Array,
    config: &STFTConfig,
) -> Result<Array, Exception> {
    let spec = stft(y, config.n_fft, config.hop_size, config.win_size)?;
    let mel_basis = mel_filterbank(config.n_fft, config.n_mels, config.sampling_rate, config.fmin, config.fmax);
    let mel = matmul(&mel_basis, &spec)?;
    // Dynamic range compression: log(clamp(x, 1e-5))
    Ok(mel.maximum(array!(1e-5))?.log()?)
}
```

**Multi-Resolution STFT Loss**:
```rust
/// Multi-resolution STFT loss for high-quality audio
pub struct MultiResolutionSTFTLoss {
    configs: Vec<STFTConfig>,
}

impl MultiResolutionSTFTLoss {
    pub fn new() -> Self {
        // Multiple resolutions: (n_fft, hop, win)
        // - (512, 50, 240)   - Fine detail
        // - (1024, 120, 600) - Medium
        // - (2048, 240, 1200) - Coarse structure
    }

    /// Compute spectral convergence + log STFT magnitude loss
    pub fn forward(&self, y: &Array, y_hat: &Array) -> Result<Array, Exception> {
        let mut loss = array!(0.0f32);
        for config in &self.configs {
            let spec_y = stft(y, config.n_fft, config.hop_size, config.win_size)?;
            let spec_y_hat = stft(y_hat, config.n_fft, config.hop_size, config.win_size)?;

            // Spectral convergence: ||S_y - S_y_hat||_F / ||S_y||_F
            let sc = frobenius_norm(&spec_y.subtract(&spec_y_hat)?)?
                .divide(&frobenius_norm(&spec_y)?)?;

            // Log magnitude loss
            let log_y = spec_y.log()?;
            let log_y_hat = spec_y_hat.log()?;
            let mag_loss = log_y.subtract(&log_y_hat)?.abs()?.mean(None, None)?;

            loss = loss.add(&sc)?.add(&mag_loss)?;
        }
        Ok(loss)
    }
}
```

### 5. GAN Training Loop

**Location**: `src/training/vits_trainer.rs` (new file)

```rust
pub struct VITSTrainer {
    generator: SynthesizerTrn,
    discriminator: MultiPeriodDiscriminator,
    optim_g: AdamW,
    optim_d: AdamW,
    config: VITSTrainingConfig,
    step: usize,
}

pub struct VITSTrainingConfig {
    pub learning_rate_g: f32,      // 2e-4
    pub learning_rate_d: f32,      // 2e-4
    pub batch_size: usize,         // 4
    pub segment_size: i32,         // 8192 samples
    pub c_mel: f32,                // 45.0 (mel loss weight)
    pub c_kl: f32,                 // 1.0 (KL loss weight)
    pub c_fm: f32,                 // 2.0 (feature matching weight)
    pub grad_clip: f32,            // 5.0
}

impl VITSTrainer {
    /// Single training step with alternating G/D updates
    pub fn train_step(&mut self, batch: &VITSBatch) -> Result<VITSLosses, Error> {
        // ===============================
        // Step 1: Discriminator Update
        // ===============================

        // Forward pass through generator (no grad for D update)
        let g_output = self.generator.forward_train(
            &batch.ssl_features,
            &batch.spec,
            &batch.spec_lengths,
            &batch.text,
            &batch.text_lengths,
        )?;

        // Slice real audio to match generated segment
        let y_real = slice_segments(&batch.audio, &g_output.ids_slice, self.config.segment_size);

        // Discriminator forward
        let d_output = self.discriminator.forward(&y_real, &g_output.y_hat.stop_gradient()?)?;

        // Discriminator loss
        let loss_d = discriminator_loss(&d_output.y_d_real, &d_output.y_d_fake)?;

        // Update discriminator
        let (_, grads_d) = nn::value_and_grad(|d| {
            discriminator_loss_fn(d, &y_real, &g_output.y_hat)
        })(&mut self.discriminator)?;
        self.optim_d.update(&mut self.discriminator, &grads_d)?;

        // ===============================
        // Step 2: Generator Update
        // ===============================

        // Need fresh discriminator output (with gradients flowing to G)
        let d_output = self.discriminator.forward(&y_real, &g_output.y_hat)?;

        // Compute all generator losses
        let loss_gen = generator_loss(&d_output.y_d_fake)?;
        let loss_fm = feature_loss(&d_output.fmap_real, &d_output.fmap_fake)?;
        let loss_mel = mel_loss(&y_real, &g_output.y_hat)? * self.config.c_mel;
        let loss_kl = kl_loss(
            &g_output.z_p,
            &g_output.logs_q,
            &g_output.m_p,
            &g_output.logs_p,
            &g_output.z_mask,
        )? * self.config.c_kl;

        let loss_g_total = loss_gen
            .add(&loss_fm.multiply(array!(self.config.c_fm))?)?
            .add(&loss_mel)?
            .add(&loss_kl)?
            .add(array!(g_output.kl_ssl))?;

        // Update generator
        let (_, grads_g) = nn::value_and_grad(|g| {
            generator_loss_fn(g, &self.discriminator, batch)
        })(&mut self.generator)?;
        self.optim_g.update(&mut self.generator, &grads_g)?;

        self.step += 1;

        Ok(VITSLosses {
            loss_d: loss_d.item::<f32>(),
            loss_gen: loss_gen.item::<f32>(),
            loss_fm: loss_fm.item::<f32>(),
            loss_mel: loss_mel.item::<f32>(),
            loss_kl: loss_kl.item::<f32>(),
        })
    }
}
```

## Implementation Order

### Phase 1: Core Components (Priority: High)

1. **Mel Spectrogram** (`src/audio/mel.rs`)
   - STFT implementation in MLX
   - Mel filterbank computation
   - Used for both training and inference

2. **PosteriorEncoder** (add to `src/models/vits.rs`)
   - Encodes spectrogram to latent
   - Required for `forward_train()`

3. **Generator `forward_train()`**
   - Expose intermediate latents
   - Segment slicing for memory efficiency

### Phase 2: Discriminator & Losses (Priority: High)

4. **Multi-Period Discriminator** (`src/models/discriminator.rs`)
   - DiscriminatorP (period-based 2D conv)
   - DiscriminatorS (scale-based 1D conv)
   - Feature map extraction

5. **Loss Functions** (`src/training/vits_loss.rs`)
   - Generator/discriminator adversarial loss
   - Feature matching loss
   - KL divergence loss
   - Mel L1 loss

### Phase 3: Training Loop (Priority: Medium)

6. **VITS Trainer** (`src/training/vits_trainer.rs`)
   - Alternating G/D updates
   - Gradient computation with `value_and_grad`
   - Checkpointing

7. **Training CLI** (`examples/train_vits.rs`)
   - Command-line interface
   - Data loading integration

### Phase 4: Enhancements (Priority: Low)

8. **Multi-Resolution STFT Loss**
   - Optional higher-quality loss

9. **Mixed Precision Training**
   - FP16/BF16 support for faster training

## File Structure

```
src/
├── audio/
│   └── mel.rs                    # NEW: STFT & mel spectrogram
├── models/
│   ├── vits.rs                   # MODIFY: Add forward_train(), PosteriorEncoder
│   └── discriminator.rs          # NEW: Multi-period discriminator
├── training/
│   ├── mod.rs                    # MODIFY: Export VITS training
│   ├── vits_loss.rs              # NEW: VITS loss functions
│   └── vits_trainer.rs           # NEW: VITS training loop
examples/
└── train_vits.rs                 # NEW: Training CLI
```

## Data Requirements

Training data format (same as T2S but with audio):
```
/path/to/training_data/
├── metadata.json
├── phoneme_ids/*.npy      # [seq_len] int32
├── bert_features/*.npy    # [1024, seq_len] float32
├── semantic_ids/*.npy     # [seq_len] int32
├── audio/*.npy            # [samples] float32 (32kHz)
└── spec/*.npy             # [n_fft/2+1, frames] float32
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Training throughput | 2+ steps/sec (batch=4) |
| Memory usage | <16GB for batch=4 |
| Quality | Match PyTorch baseline |

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| STFT in MLX may be slow | Precompute spectrograms offline |
| GAN training instability | Start with pretrained generator |
| Memory for full audio | Segment slicing (8192 samples) |
| Discriminator gradients | Careful stop_gradient() placement |

## Verification Steps

1. **Unit tests**: Each component independently
2. **Loss sanity check**: Compare with PyTorch on same inputs
3. **Training curve**: Should see D/G loss converge
4. **Audio quality**: A/B test generated samples
5. **Voice similarity**: Compare cloned voice to reference

## References

- GPT-SoVITS: `moyoyo_tts/module/models.py`
- VITS paper: https://arxiv.org/abs/2106.06103
- HiFi-GAN: https://arxiv.org/abs/2010.05646
