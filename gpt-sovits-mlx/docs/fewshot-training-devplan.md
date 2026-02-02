# Fewshot VITS Training Development Plan

## Goal
Enable voice cloning with 1-minute of audio by training VITS (SoVITS) in Rust + MLX.

## Why VITS Training (Not T2S)

| Model | What it learns | Data needed | Fewshot? |
|-------|----------------|-------------|----------|
| **VITS** | Voice timbre (how voice sounds) | 10-60 seconds | ✅ Yes |
| **T2S** | Prosody/rhythm (how words are spoken) | 10+ minutes | ❌ No |

The Python GPT-SoVITS fewshot training uses `s2_train.py` which trains **VITS only**.

## Architecture

```
Training Input:
  - HuBERT features (SSL): [batch, 768, ssl_len] - extracted from audio
  - Spectrogram: [batch, 1025, spec_len] - computed from audio
  - Phonemes: [batch, text_len] - from G2P
  - Audio: [batch, 1, samples] - target waveform

Generator (SynthesizerTrn):
  SSL + Spec + Text → y_hat (generated audio)
                    → z_p, m_p, logs_p, logs_q (for KL loss)

Discriminator (MultiPeriodDiscriminator):
  Audio → real/fake scores + feature maps

Losses:
  - Discriminator: L2 loss on real=1, fake=0
  - Generator: L2 loss on fake=1 (adversarial)
  - Feature Matching: L1 on discriminator feature maps
  - Mel Reconstruction: L1 on mel spectrograms
  - KL Divergence: VAE regularization
```

## Current Status (Updated 2026-01-30)

### ✅ Fully Implemented
- `SynthesizerTrn` (Generator) - `src/models/vits.rs`
- `MultiPeriodDiscriminator` - `src/models/discriminator.rs`
- `VITSTrainer` - `src/training/vits_trainer.rs`
- All loss functions - `src/training/vits_loss.rs`
- AdamW optimizer, value_and_grad, gradient clipping
- **Data converter script** - `scripts/convert_vits_training_data.py`
- **VITSDataset** - `src/training/vits_dataset.rs`
- **Spectrogram computation** - `src/audio/mel.rs`
- **Training CLI** - `examples/train_vits.rs`

### ⚠️ Known Limitations
1. **Grouped convolutions** - Disabled in DiscriminatorS due to weight initialization issues
2. **No discriminator checkpoint loading** - Uses fresh discriminator, not pretrained

### 🎉 Training Working
Training verified with 1-minute fewshot data:
```
Epoch 4 Summary: D=2.77, G=1.95, Mel=0.78
```

## Implementation Plan

### Phase 1: Data Pipeline (Day 1)

#### Task 1.1: Convert Python preprocessed data
Create script to convert GPT-SoVITS format to training format:

```
Input (Python GPT-SoVITS):
  /tmp/fewshot_1min/
  ├── 4-cnhubert/*.wav.pt    # HuBERT features [1, 768, T]
  ├── 5-wav32k/*.wav         # Audio files
  └── 2-name2text.txt        # phonemes

Output (Rust training):
  /tmp/fewshot_1min_vits/
  ├── ssl/*.npy              # HuBERT features [768, T]
  ├── audio/*.npy            # Audio [samples]
  ├── phonemes/*.npy         # Phoneme IDs [T]
  └── metadata.json
```

#### Task 1.2: Implement VITSDataset
```rust
pub struct VITSDataset {
    samples: Vec<VITSSample>,
}

pub struct VITSSample {
    ssl_path: PathBuf,      // HuBERT features
    audio_path: PathBuf,    // Audio waveform
    phonemes_path: PathBuf, // Phoneme IDs
}

impl VITSDataset {
    pub fn load(dir: &Path) -> Result<Self>;
    pub fn get_batch(&self, indices: &[usize]) -> Result<VITSBatch>;
}
```

#### Task 1.3: Implement slice_segments
```rust
/// Extract random segments from audio/features for training
pub fn slice_segments(
    x: &Array,           // [batch, channels, time]
    segment_size: i32,   // Target segment length
) -> Result<(Array, Array), Exception> {
    // Returns (sliced_x, start_indices)
}
```

### Phase 2: Training Loop (Day 1-2)

#### Task 2.1: Implement forward_train for SynthesizerTrn
The generator needs a training-specific forward that returns all intermediate values:

```rust
impl SynthesizerTrn {
    pub fn forward_train(
        &mut self,
        ssl: &Array,           // [B, 768, T_ssl]
        spec: &Array,          // [B, 1025, T_spec]
        spec_lengths: &Array,  // [B]
        text: &Array,          // [B, T_text]
        refer: &Array,         // [B, 128, T_ref]
    ) -> Result<(
        Array,  // y_hat: generated audio
        Array,  // kl_ssl: SSL KL loss
        Array,  // ids_slice: segment indices
        Array,  // x_mask
        Array,  // z_mask
        Array,  // z_p, m_p, logs_p, m_q, logs_q (VAE outputs)
    ), Exception>;
}
```

#### Task 2.2: Update VITSTrainer
```rust
impl VITSTrainer {
    pub fn train_epoch(&mut self, dataset: &VITSDataset) -> Result<f32>;
    pub fn save_generator(&self, path: &Path) -> Result<()>;
}
```

### Phase 3: Training CLI (Day 2)

#### Task 3.1: Create examples/train_vits.rs
```rust
/// VITS Training CLI
///
/// Usage:
///   cargo run --release --example train_vits -- \
///     --data-dir /tmp/fewshot_1min_vits \
///     --pretrained ~/.dora/models/.../vits_pretrained.safetensors \
///     --output /tmp/vits_finetuned.safetensors \
///     --epochs 4 \
///     --batch-size 2
```

### Phase 4: Testing & Validation (Day 2-3)

#### Task 4.1: Test with 1-minute data
1. Preprocess audio with Python (existing scripts)
2. Convert to Rust training format
3. Train for 4 epochs
4. Test inference with trained model

#### Task 4.2: Compare with Python baseline
- Train same data with Python s2_train.py
- Compare audio quality

## File Changes

### New Files
```
scripts/convert_vits_training_data.py  # Convert Python → Rust format
src/training/vits_dataset.rs           # VITS training dataset
examples/train_vits.rs                 # Training CLI
```

### Modified Files
```
src/models/vits.rs                     # Add forward_train method
src/training/vits_trainer.rs           # Update training loop
src/training/mod.rs                    # Export new types
```

## Training Configuration

Based on Python GPT-SoVITS defaults for fewshot:

```rust
VITSTrainingConfig {
    learning_rate_g: 1e-4,      // Generator LR
    learning_rate_d: 1e-4,      // Discriminator LR
    batch_size: 2,              // Small for fewshot
    segment_size: 20480,        // ~640ms at 32kHz
    epochs: 4,                  // 4 epochs for fewshot
    c_mel: 45.0,                // Mel loss weight
    c_kl: 1.0,                  // KL loss weight
    c_fm: 2.0,                  // Feature matching weight
    lr_decay: 0.999875,         // Per-step decay
    save_every_epoch: 2,
}
```

## Success Criteria

1. Training completes without errors
2. Loss decreases over epochs:
   - D loss: ~2-4 → ~1-2
   - G loss: ~10+ → ~2-5
   - Mel loss: ~50+ → ~10-20
3. Generated audio sounds like target speaker
4. Quality comparable to Python training

## Timeline

| Day | Tasks | Deliverable |
|-----|-------|-------------|
| 1 | Data pipeline, slice_segments | Can load training data |
| 2 | Training loop, CLI | Can run training |
| 3 | Testing, debugging | Working fewshot training |

## Commands

```bash
# Step 1: Preprocess with Python (already done)
# Data at /tmp/fewshot_1min/

# Step 2: Convert to Rust format
python scripts/convert_vits_training_data.py \
  --input /tmp/fewshot_1min \
  --output /tmp/fewshot_1min_vits

# Step 3: Train VITS
cargo run --release --example train_vits -- \
  --data-dir /tmp/fewshot_1min_vits \
  --pretrained ~/.dora/models/primespeech/gpt-sovits-mlx/vits_pretrained_v2.safetensors \
  --output /tmp/vits_finetuned.safetensors \
  --epochs 4

# Step 4: Test inference
cargo run --release --example voice_clone -- \
  --vits /tmp/vits_finetuned.safetensors \
  --ref /tmp/fewshot_1min/5-wav32k/seg_0000.wav \
  --ref-text "参考文本" \
  "测试语音克隆"
```

## Notes

- Weight norm/spectral norm: Skip for fewshot (small data, short training)
- GRU in ReferenceEncoder: Use existing MelStyleEncoder instead
- Distributed training: Not needed for fewshot
- FP16: Skip for simplicity (MLX handles mixed precision automatically)
