# GPT-SoVITS Training Support Development Plan

## Goal
Enable voice cloning by fine-tuning GPT-SoVITS models on user-provided voice samples.

**Important: What to train depends on your data length:**
- **Few-shot (10-30 seconds)**: Train **SoVITS only** — learns voice timbre
- **Medium (1-5 minutes)**: Train SoVITS (8-15 epochs) + optional T2S
- **Full fine-tune (10+ minutes)**: Train both SoVITS and T2S

**What each model learns:**
| Model | Learns | Effect |
|-------|--------|--------|
| SoVITS (VITS) | Voice **timbre** | How the voice sounds (tone quality, texture) |
| T2S (GPT) | **Prosody/rhythm** | How words are spoken (pacing, emphasis, flow) |

**Reference: Doubao model training config:**
- SoVITS: 8 epochs, ~384 samples (192 iterations), float16
- T2S/GPT: 15 epochs

## Architecture Overview

```
User's Voice Samples (audio + transcripts)
    ↓
[Phase 1: Data Preprocessing] (Python scripts)
    ├── Audio → HuBERT → Semantic IDs (training targets)
    ├── Text → G2P → Phoneme IDs
    └── Text → BERT → BERT Features (1024-dim)
    ↓
[Phase 2: Model Training]
    ├── FEW-SHOT (10-30s): SoVITS only (Python/PyTorch)
    │   └── 4-8 epochs, learn voice timbre
    ├── MEDIUM (1-5 min): SoVITS + optional T2S
    │   └── SoVITS 8-15 epochs, T2S 5-10 epochs
    └── FULL (10+ min): Both SoVITS and T2S
        └── SoVITS 8-20 epochs, T2S 15+ epochs
    ↓
Fine-tuned Voice Model (.safetensors)
```

## Phase 1: Data Preprocessing (Python)

### Files to Create
```
scripts/
├── preprocess_voice.py      # Main preprocessing script
└── extract_semantic_codes.py # HuBERT semantic extraction
```

### preprocess_voice.py
Inputs:
- Directory with .wav files
- Corresponding transcript files (.txt)
- Path to pretrained models (HuBERT, BERT, VITS)

Outputs:
- `phoneme_ids.npy` - Phoneme sequences for each utterance
- `bert_features.npy` - BERT features for each utterance
- `semantic_ids.npy` - Target semantic codes (from HuBERT)
- `metadata.json` - File paths and lengths

### Data Format
```
dataset/
├── metadata.json
├── phoneme_ids/
│   ├── 0000.npy  # [seq_len] int32
│   ├── 0001.npy
│   └── ...
├── bert_features/
│   ├── 0000.npy  # [1024, seq_len] float32
│   └── ...
└── semantic_ids/
    ├── 0000.npy  # [seq_len] int32 (target)
    └── ...
```

## Phase 2: T2S Training Module (Rust/MLX)

### Files to Create
```
src/training/
├── mod.rs              # Module exports
├── config.rs           # TrainingConfig struct
├── dataset.rs          # TrainingDataset, DataLoader
├── trainer.rs          # T2STrainer implementation
├── lr_scheduler.rs     # Learning rate schedulers
└── checkpoint.rs       # Save/load training state

examples/
└── train_voice.rs      # CLI for voice training
```

### Core Components

#### 1. TrainingConfig
```rust
pub struct TrainingConfig {
    // Optimization
    pub learning_rate: f32,           // 1e-4
    pub weight_decay: f32,            // 0.01
    pub warmup_steps: usize,          // 1000
    pub max_steps: usize,             // 100000
    pub batch_size: usize,            // 4-16

    // Checkpointing
    pub checkpoint_dir: PathBuf,
    pub save_every_n_steps: usize,    // 1000
    pub keep_last_n_checkpoints: usize, // 5

    // Logging
    pub log_every_n_steps: usize,     // 100

    // Model
    pub base_model_path: PathBuf,     // Pretrained T2S
    pub freeze_bert_proj: bool,       // true (don't fine-tune BERT projection)
}
```

#### 2. TrainingDataset
```rust
pub struct TrainingDataset {
    metadata: Vec<SampleMetadata>,
    phoneme_dir: PathBuf,
    bert_dir: PathBuf,
    semantic_dir: PathBuf,
}

impl TrainingDataset {
    pub fn load(dataset_path: &Path) -> Result<Self>;
    pub fn get_batch(&self, indices: &[usize]) -> Result<TrainingBatch>;
    pub fn len(&self) -> usize;
}

pub struct TrainingBatch {
    pub phoneme_ids: Array,      // [batch, max_phone_len]
    pub phoneme_lens: Array,     // [batch]
    pub bert_features: Array,    // [batch, 1024, max_phone_len]
    pub semantic_ids: Array,     // [batch, max_semantic_len]
    pub semantic_lens: Array,    // [batch]
}
```

#### 3. T2STrainer
```rust
pub struct T2STrainer {
    config: TrainingConfig,
    model: T2SModel,
    optimizer: AdamW,
    scheduler: CosineScheduler,
    step: usize,
    best_loss: f32,
}

impl T2STrainer {
    pub fn new(config: TrainingConfig) -> Result<Self>;
    pub fn load_pretrained(&mut self, path: &Path) -> Result<()>;
    pub fn load_dataset(&mut self, path: &Path) -> Result<TrainingDataset>;

    pub fn train(&mut self, dataset: &TrainingDataset) -> Result<()>;
    pub fn train_step(&mut self, batch: &TrainingBatch) -> Result<f32>;

    pub fn save_checkpoint(&self, path: &Path) -> Result<()>;
    pub fn load_checkpoint(&mut self, path: &Path) -> Result<()>;
}
```

#### 4. Training Loop
```rust
fn train_step(&mut self, batch: &TrainingBatch) -> Result<f32> {
    // 1. Forward pass
    let logits = self.model.forward_train(
        &batch.phoneme_ids,
        &batch.phoneme_lens,
        &batch.bert_features,
        &batch.semantic_ids,
        &batch.semantic_lens,
    )?;

    // 2. Compute loss (CrossEntropy)
    let loss = cross_entropy_loss(&logits, &batch.semantic_ids)?;

    // 3. Backward pass (compute gradients)
    let gradients = compute_gradients(&self.model, &loss)?;

    // 4. Clip gradients
    let (clipped_grads, _) = clip_grad_norm(&gradients, 1.0)?;

    // 5. Update parameters
    self.optimizer.update(&mut self.model, &clipped_grads)?;

    // 6. Update learning rate
    self.scheduler.step();

    Ok(loss.item())
}
```

#### 5. LR Scheduler
```rust
pub struct CosineScheduler {
    base_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl CosineScheduler {
    pub fn get_lr(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (self.current_step as f32 / self.warmup_steps as f32)
        } else {
            // Cosine decay
            let progress = (self.current_step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps) as f32;
            self.base_lr * 0.5 * (1.0 + (PI * progress).cos())
        }
    }
}
```

## Phase 3: CLI Tool

### examples/train_voice.rs
```
USAGE:
    train_voice [OPTIONS] --dataset <PATH>

OPTIONS:
    --dataset <PATH>        Path to preprocessed dataset
    --base-model <PATH>     Path to pretrained T2S model
    --output <PATH>         Output directory for checkpoints
    --batch-size <N>        Batch size (default: 4)
    --learning-rate <F>     Learning rate (default: 1e-4)
    --max-steps <N>         Maximum training steps (default: 10000)
    --resume <PATH>         Resume from checkpoint
```

## Implementation Order

### Step 1: Data Preprocessing Scripts (Python)
- [ ] `scripts/preprocess_voice.py`
- [ ] `scripts/extract_semantic_codes.py`
- [ ] Test with sample voice data

### Step 2: Training Infrastructure (Rust)
- [ ] `src/training/mod.rs` - Module setup
- [ ] `src/training/config.rs` - Configuration
- [ ] `src/training/dataset.rs` - Data loading
- [ ] `src/training/lr_scheduler.rs` - LR scheduling

### Step 3: T2S Trainer (Rust)
- [ ] Add `forward_train` to T2SModel (returns logits, not sampled tokens)
- [ ] `src/training/trainer.rs` - Main trainer
- [ ] `src/training/checkpoint.rs` - Checkpoint management

### Step 4: CLI and Integration
- [ ] `examples/train_voice.rs` - Training CLI
- [ ] Integration tests
- [ ] Documentation

## Testing Plan

1. **Unit Tests**
   - Dataset loading
   - Batch collation
   - LR scheduler correctness
   - Gradient computation

2. **Integration Tests**
   - Single training step
   - Checkpoint save/load
   - Full training loop (small dataset)

3. **End-to-End Test**
   - Preprocess sample voice (5-10 utterances)
   - Train for 100 steps
   - Verify loss decreases
   - Generate with fine-tuned model

## Estimated Timeline

| Task | Duration |
|------|----------|
| Data preprocessing scripts | 1 day |
| Training config & dataset | 1 day |
| LR scheduler | 0.5 day |
| T2S forward_train | 0.5 day |
| Trainer implementation | 2 days |
| Checkpoint management | 0.5 day |
| CLI tool | 0.5 day |
| Testing & debugging | 2 days |
| **Total** | **~8 days** |

## Dependencies

No new external dependencies needed. Uses existing:
- `mlx-rs` (optimizers, transforms, losses)
- `serde` (config serialization)
- `clap` (CLI parsing)

## Training Priority by Use Case

### Few-shot Voice Cloning (10-30s audio)
**Priority: Train SoVITS only** (in Python/PyTorch)
- This is the most common use case
- T2S/GPT doesn't benefit from <30s of data (prosody needs more samples)
- SoVITS learns timbre quickly from few samples
- Expected: ~4-8 epochs, ~8-16 iterations total

### Medium Data (1-5 min audio)
**Train SoVITS first, then optionally T2S**
- SoVITS: 8-15 epochs
- T2S: 5-10 epochs (optional, if prosody matters)

### Full Fine-tune (10+ min audio)
**Train both models**
- SoVITS: 8-20 epochs
- T2S: 15+ epochs

## Future Enhancements (Phase 4+)

1. **SoVITS Training in Rust** - GAN training for vocoder (complex, requires discriminator)
2. **Distributed Training** - Multi-GPU support
3. **Mixed Precision** - FP16 training
4. **Data Augmentation** - Speed perturbation, noise injection
5. **Validation** - Held-out set evaluation
