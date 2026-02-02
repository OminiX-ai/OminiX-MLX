# Voice Cloning Improvement Plan

## Executive Summary

This document outlines approaches to improve voice cloning quality in gpt-sovits-mlx.

**Critical Understanding: What to Train Based on Data Length**

| Data Length | Train | Why |
|-------------|-------|-----|
| **Zero-shot (0s)** | Nothing | Use reference audio at inference only |
| **Few-shot (10-30s)** | **SoVITS only** | Learns voice timbre; not enough data for prosody |
| **Medium (1-5 min)** | SoVITS + optional T2S | Can start learning prosody patterns |
| **Full (10+ min)** | Both SoVITS and T2S | Full voice adaptation |

**What each model learns:**
- **SoVITS (VITS)**: Voice **timbre** — how the voice sounds (tone, texture, quality)
- **T2S (GPT)**: **Prosody** — how words are spoken (rhythm, pacing, emphasis)

**Reference: Doubao model training:**
- SoVITS: 8 epochs, ~384 samples (192 iterations)
- T2S/GPT: 15 epochs

---

## Approaches

1. **Train SoVITS** using MoYoYo.tts Python pipeline (immediate improvement for few-shot)
2. **Add V2Pro Support** by porting Eres2Net to Rust MLX (better zero-shot)
3. **Complete Python Training Pipeline** integration (full solution)

---

## Current State

### What We Have
- ✅ T2S (Text-to-Semantic) model trained on user's voice (1000 steps)
- ✅ T2S inference in Rust MLX
- ✅ VITS inference in Rust MLX (using pretrained weights)
- ✅ HuBERT feature extraction in Rust MLX
- ✅ BERT feature extraction in Rust MLX

### What's Missing
- ❌ VITS trained on user's voice → voice timbre doesn't match
- ❌ Eres2Net (Speaker Verification) → weak speaker conditioning
- ❌ Complete training pipeline in Rust → still depends on Python

### Current Voice Cloning Quality
| Component | Trained | Effect |
|-----------|---------|--------|
| T2S | ✅ Yes | Prosody matches (rhythm, pacing) |
| VITS | ❌ No | Timbre doesn't match (sounds generic) |

---

## Approach 1: Train VITS Using MoYoYo.tts Pipeline

### Overview
Use the existing Python training pipeline to fine-tune VITS on user's voice data.

### Prerequisites
- Audio segments: `/tmp/voice_training/audio_segments/` (already prepared)
- Transcription: `/tmp/voice_training/train.list` (already prepared)
- MoYoYo.tts repo: `/Users/yuechen/home/OminiX-MLX/gpt-sovits-clone-mlx/MoYoYo.tts/`

### Steps

#### Step 1: Prepare Feature Directory Structure
```bash
# Create experiment directory
EXP_DIR="/tmp/voice_training/gpt_sovits_exp"
mkdir -p $EXP_DIR/{3-bert,4-cnhubert,5-wav32k,logs_s1,logs_s2}
```

#### Step 2: Extract BERT Features (Text)
```bash
cd /Users/yuechen/home/OminiX-MLX/gpt-sovits-clone-mlx/MoYoYo.tts

export inp_text="/tmp/voice_training/train.list"
export inp_wav_dir="/tmp/voice_training/audio_segments"
export exp_name="yuechen_voice"
export opt_dir="$EXP_DIR"
export bert_pretrained_dir="/Users/yuechen/.dora/models/primespeech/moyoyo/chinese-roberta-wwm-ext-large"
export i_part="0"
export all_parts="1"
export is_half="False"

python GPT_SoVITS/prepare_datasets/1-get-text.py
```

#### Step 3: Extract HuBERT Features (Audio SSL)
```bash
export cnhubert_base_dir="/Users/yuechen/.dora/models/primespeech/moyoyo/chinese-hubert-base"

python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py
```

#### Step 4: Extract Semantic Tokens
```bash
export pretrained_s2G="/Users/yuechen/.dora/models/primespeech/moyoyo/SoVITS_weights/doubao-mixed.pth"
export s2config_path="GPT_SoVITS/configs/s2.json"

python GPT_SoVITS/prepare_datasets/3-get-semantic.py
```

#### Step 5: Train SoVITS (VITS)

**IMPORTANT: Training parameters depend on your data length**

| Data Length | Epochs | Iterations | Notes |
|-------------|--------|------------|-------|
| 10-30s (~3-10 segments) | 4-8 | 8-40 total | Few-shot: SoVITS only |
| 1-5 min (~20-100 segments) | 8-15 | 100-500 total | Medium: can add T2S |
| 10+ min (~200+ segments) | 8-20 | 500+ total | Full fine-tune |

**Reference: Doubao was trained with ~384 samples, 8 epochs (192 iterations)**

```python
# Create training config
import json

config = {
    "train": {
        "batch_size": 4,  # Adjust based on memory (Doubao used batch 2)
        "epochs": 8,      # 4-8 for few-shot, 8-15 for medium
        "fp16_run": False,  # Set True if GPU has enough VRAM
        "text_low_lr_rate": 0.4,
        "pretrained_s2G": "/Users/yuechen/.dora/models/primespeech/moyoyo/SoVITS_weights/doubao-mixed.pth",
        # IMPORTANT: Must specify discriminator for proper GAN training
        "pretrained_s2D": "/path/to/pretrained/s2D2333k.pth",
        "if_save_latest": True,
        "if_save_every_weights": True,
        "save_every_epoch": 4,
    },
    "data": {
        "exp_dir": "/tmp/voice_training/gpt_sovits_exp",
    },
    "model": {
        "version": "v2",
    },
    "name": "yuechen_voice",
}

with open("/tmp/voice_training/s2_config.json", "w") as f:
    json.dump(config, f)
```

```bash
python GPT_SoVITS/s2_train.py --config /tmp/voice_training/s2_config.json
```

#### Step 6: Convert to Safetensors for Rust MLX
```python
import torch
from safetensors.torch import save_file

# Load trained VITS
ckpt = torch.load("/tmp/voice_training/gpt_sovits_exp/SoVITS_weights/yuechen_voice.pth")
weights = ckpt.get("weight", ckpt)

# Save as safetensors
save_file(weights, "~/.dora/models/primespeech/gpt-sovits-mlx/yuechen_vits.safetensors")
```

### Expected Outcome
- VITS model trained on user's voice
- Voice timbre should match reference audio
- Combined with trained T2S → full voice cloning

### Time Estimate
- Feature extraction: ~10-15 minutes
- SoVITS training (8 epochs): ~30-60 minutes
- Total: ~45-75 minutes

---

## Approach 2: Add V2Pro Support (Port Eres2Net)

### Overview
Port the Eres2Net speaker verification model to Rust MLX for better zero-shot voice cloning.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ERES2NET ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Audio (16kHz) → Fbank (80-dim) → ERes2NetV2 → sv_emb (192) │
│                                                              │
│  ERes2NetV2:                                                │
│  ├── Conv1d stem                                            │
│  ├── ResNet-like blocks with SE attention                   │
│  ├── Attentive Statistics Pooling                          │
│  └── FC layer → 192-dim embedding                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Plan

#### Phase 1: Fbank Feature Extraction (1-2 days)
```rust
// src/audio/fbank.rs

/// Compute 80-bin filterbank features (Kaldi-compatible)
pub fn compute_fbank(
    audio: &[f32],
    sample_rate: i32,
    num_mel_bins: i32,
) -> Result<Array, Error> {
    // 1. Pre-emphasis
    // 2. Frame extraction (25ms window, 10ms hop)
    // 3. Windowing (Povey window)
    // 4. FFT
    // 5. Mel filterbank
    // 6. Log compression
}
```

#### Phase 2: ERes2NetV2 Model (3-5 days)
```rust
// src/models/eres2net.rs

/// SE (Squeeze-and-Excitation) block
#[derive(ModuleParameters)]
pub struct SEBlock {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

/// Res2Net block with SE attention
#[derive(ModuleParameters)]
pub struct Res2NetBlock {
    conv1: nn::Conv1d,
    bn1: nn::BatchNorm,
    conv2: nn::Conv1d,  // Split into scales
    bn2: nn::BatchNorm,
    conv3: nn::Conv1d,
    bn3: nn::BatchNorm,
    se: SEBlock,
    downsample: Option<nn::Conv1d>,
}

/// Attentive Statistics Pooling
#[derive(ModuleParameters)]
pub struct AttentiveStatsPool {
    attention: nn::Conv1d,
}

/// Complete ERes2NetV2 model
#[derive(ModuleParameters)]
pub struct ERes2NetV2 {
    conv1: nn::Conv1d,
    bn1: nn::BatchNorm,
    layer1: Vec<Res2NetBlock>,
    layer2: Vec<Res2NetBlock>,
    layer3: Vec<Res2NetBlock>,
    layer4: Vec<Res2NetBlock>,
    pool: AttentiveStatsPool,
    fc: nn::Linear,
}

impl ERes2NetV2 {
    /// Extract speaker embedding
    pub fn forward(&self, fbank: &Array) -> Result<Array, Error> {
        // fbank: [batch, time, 80]
        // output: [batch, 192]
    }
}
```

#### Phase 3: Integration with VITS (1-2 days)
```rust
// Modify src/models/vits.rs

impl VITSModel {
    /// Forward with speaker verification embedding
    pub fn forward_with_sv(
        &self,
        ssl_content: &Array,
        text: &Array,
        refer: &Array,
        sv_emb: Option<&Array>,  // NEW: Speaker embedding from Eres2Net
    ) -> Result<Array, Error> {
        // Extract style from reference mel
        let ge = self.ref_enc.forward(refer)?;

        // Combine with speaker embedding if available
        let conditioning = if let Some(sv) = sv_emb {
            // Blend ge and sv_emb
            concatenate_axis(&[&ge, sv], -1)?
        } else {
            ge
        };

        // Continue with VITS forward
        // ...
    }
}
```

#### Phase 4: Weight Conversion (0.5 days)
```python
# scripts/convert_eres2net.py

import torch
from safetensors.torch import save_file

# Load PyTorch weights
ckpt = torch.load("GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt")

# Convert to safetensors
tensors = {}
for name, param in ckpt.items():
    tensors[name] = param

save_file(tensors, "~/.dora/models/primespeech/gpt-sovits-mlx/eres2net.safetensors")
```

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/audio/fbank.rs` | CREATE | Kaldi-compatible fbank extraction |
| `src/models/eres2net.rs` | CREATE | ERes2NetV2 model |
| `src/models/mod.rs` | MODIFY | Add eres2net module |
| `src/models/vits.rs` | MODIFY | Add sv_emb parameter |
| `src/voice_clone.rs` | MODIFY | Integrate Eres2Net |
| `scripts/convert_eres2net.py` | CREATE | Weight conversion |

### Expected Outcome
- Better zero-shot voice cloning without training
- Speaker identity preserved from 10-second reference
- Works with existing pretrained models

### Time Estimate
- Fbank extraction: 1-2 days
- ERes2NetV2 model: 3-5 days
- Integration: 1-2 days
- Testing/debugging: 1-2 days
- **Total: 6-11 days**

---

## Approach 3: Complete Python Training Pipeline Integration

### Overview
Create a unified training interface that uses MoYoYo.tts for training and Rust MLX for inference.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     UNIFIED TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────────┐ │
│  │   Rust CLI       │ →  │  Python Training │ →  │  Rust MLX     │ │
│  │  (Orchestrator)  │    │    Pipeline      │    │  Inference    │ │
│  └──────────────────┘    └──────────────────┘    └───────────────┘ │
│                                                                      │
│  Commands:                                                          │
│  - gpt-sovits train --audio audio.mp3 --name my_voice              │
│  - gpt-sovits infer --text "Hello" --voice my_voice                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation Plan

#### Phase 1: Python Training Wrapper (2-3 days)

```python
# python/training/pipeline.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import subprocess
import json

@dataclass
class TrainingConfig:
    exp_name: str
    audio_path: Path
    transcript_path: Optional[Path] = None
    version: str = "v2"
    sovits_epochs: int = 8
    gpt_epochs: int = 15
    batch_size: int = 4
    output_dir: Path = Path("~/.dora/models/primespeech/gpt-sovits-mlx/models")

class VoiceTrainingPipeline:
    """Unified voice training pipeline"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.moyoyo_path = Path.home() / "home/OminiX-MLX/gpt-sovits-clone-mlx/MoYoYo.tts"

    def run(self):
        """Execute full training pipeline"""
        yield from self._preprocess()
        yield from self._extract_features()
        yield from self._train_sovits()
        yield from self._train_gpt()
        yield from self._convert_to_safetensors()

    def _preprocess(self):
        """Audio slicing and ASR"""
        # 1. Slice audio into segments
        # 2. Run ASR for transcription
        yield {"stage": "preprocess", "progress": 1.0}

    def _extract_features(self):
        """Extract BERT, HuBERT, and semantic features"""
        # 1. Text → BERT features
        # 2. Audio → HuBERT features
        # 3. HuBERT → Semantic tokens
        yield {"stage": "features", "progress": 1.0}

    def _train_sovits(self):
        """Train SoVITS model"""
        yield {"stage": "sovits", "progress": 1.0}

    def _train_gpt(self):
        """Train GPT/T2S model"""
        yield {"stage": "gpt", "progress": 1.0}

    def _convert_to_safetensors(self):
        """Convert models to safetensors format"""
        yield {"stage": "convert", "progress": 1.0}
```

#### Phase 2: Rust CLI Interface (2-3 days)

```rust
// src/bin/train.rs

use clap::Parser;
use std::process::Command;

#[derive(Parser)]
#[command(name = "gpt-sovits-train")]
struct Args {
    /// Input audio file (mp3, wav, etc.)
    #[arg(short, long)]
    audio: PathBuf,

    /// Voice name for the trained model
    #[arg(short, long)]
    name: String,

    /// Model version (v1, v2, v2Pro)
    #[arg(long, default_value = "v2")]
    version: String,

    /// SoVITS training epochs
    #[arg(long, default_value = "8")]
    sovits_epochs: u32,

    /// GPT training epochs
    #[arg(long, default_value = "15")]
    gpt_epochs: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Call Python training pipeline
    let status = Command::new("python")
        .args(&[
            "-m", "gpt_sovits_mlx.training",
            "--audio", args.audio.to_str().unwrap(),
            "--name", &args.name,
            "--version", &args.version,
        ])
        .status()?;

    if !status.success() {
        return Err("Training failed".into());
    }

    println!("Training complete! Model saved as: {}", args.name);
    Ok(())
}
```

#### Phase 3: Model Registry (1-2 days)

```rust
// src/models/registry.rs

use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct VoiceModel {
    pub name: String,
    pub version: String,
    pub t2s_path: PathBuf,
    pub vits_path: PathBuf,
    pub ref_audio_path: Option<PathBuf>,
    pub ref_text: Option<String>,
}

pub struct ModelRegistry {
    models_dir: PathBuf,
    models: HashMap<String, VoiceModel>,
}

impl ModelRegistry {
    pub fn new(models_dir: PathBuf) -> Self {
        let mut registry = Self {
            models_dir,
            models: HashMap::new(),
        };
        registry.scan_models();
        registry
    }

    pub fn scan_models(&mut self) {
        // Scan models directory for trained voices
    }

    pub fn get(&self, name: &str) -> Option<&VoiceModel> {
        self.models.get(name)
    }

    pub fn list(&self) -> Vec<&VoiceModel> {
        self.models.values().collect()
    }
}
```

#### Phase 4: Unified CLI (1-2 days)

```rust
// src/bin/gpt-sovits.rs

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "gpt-sovits")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new voice model
    Train {
        #[arg(short, long)]
        audio: PathBuf,
        #[arg(short, long)]
        name: String,
    },

    /// List available voice models
    List,

    /// Synthesize speech
    Speak {
        #[arg(short, long)]
        text: String,
        #[arg(short, long)]
        voice: String,
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Interactive TTS mode
    Interactive {
        #[arg(short, long)]
        voice: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train { audio, name } => train_voice(audio, name),
        Commands::List => list_voices(),
        Commands::Speak { text, voice, output } => speak(text, voice, output),
        Commands::Interactive { voice } => interactive_mode(voice),
    }
}
```

### Directory Structure

```
gpt-sovits-mlx/
├── src/
│   ├── bin/
│   │   ├── gpt-sovits.rs      # Main CLI
│   │   └── train.rs           # Training CLI
│   ├── models/
│   │   ├── registry.rs        # Model registry
│   │   └── ...
│   └── ...
├── python/
│   ├── gpt_sovits_mlx/
│   │   ├── __init__.py
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py    # Training pipeline
│   │   │   ├── preprocess.py  # Audio preprocessing
│   │   │   └── convert.py     # Model conversion
│   │   └── ...
│   └── setup.py
├── models/                     # Trained models directory
│   ├── yuechen/
│   │   ├── config.json
│   │   ├── t2s.safetensors
│   │   ├── vits.safetensors
│   │   └── ref_audio.wav
│   └── ...
└── docs/
```

### Expected Outcome
- Single command to train a new voice: `gpt-sovits train --audio voice.mp3 --name my_voice`
- Single command to synthesize: `gpt-sovits speak --text "Hello" --voice my_voice`
- Model management: `gpt-sovits list`
- Interactive mode: `gpt-sovits interactive --voice my_voice`

### Time Estimate
- Python wrapper: 2-3 days
- Rust CLI: 2-3 days
- Model registry: 1-2 days
- Unified CLI: 1-2 days
- Testing/documentation: 1-2 days
- **Total: 7-12 days**

---

## Approach 4: Rust MLX T2S Fine-tuning

### Overview
Add T2S (GPT) fine-tuning capability directly in Rust MLX. Unlike VITS, T2S uses standard cross-entropy loss without GAN, making it feasible to implement in Rust.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  T2S TRAINING IN RUST MLX                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: semantic_tokens, phone_ids, bert_features           │
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Text Encoder│ →  │ AR Decoder  │ →  │ Next Token  │     │
│  │ (BERT emb)  │    │ (GPT-style) │    │ Prediction  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                              │
│  Loss: CrossEntropy(predicted_token, target_token)          │
│  Optimizer: AdamW (already in mlx-rs)                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### MLX-rs Training Primitives Available

| Component | Status | Location |
|-----------|--------|----------|
| `value_and_grad` | ✅ Available | `mlx-rs/src/transforms/value_and_grad.rs` |
| `AdamW` optimizer | ✅ Available | `mlx-rs/src/optimizers/adamw.rs` |
| `CrossEntropy` loss | ✅ Available | `mlx-rs/src/losses.rs` |
| `clip_grad_norm` | ✅ Available | `mlx-rs/src/optimizers/mod.rs` |
| `ModuleParameters` trait | ✅ Available | `mlx-rs/src/module.rs` |

### Implementation Plan

#### Phase 1: Data Loading (2-3 days)
```rust
// src/training/data.rs

/// Training sample for T2S
pub struct T2SSample {
    pub semantic_tokens: Vec<i32>,    // Target: [T_semantic]
    pub phone_ids: Vec<i32>,          // Input: [T_phone]
    pub bert_features: Array,         // Input: [T_phone, 1024]
}

/// Dataset loader for T2S training
pub struct T2SDataset {
    samples: Vec<T2SSample>,
}

impl T2SDataset {
    /// Load from preprocessed features directory
    pub fn load(exp_dir: &Path) -> Result<Self> {
        // Load 6-name2semantic.tsv
        // Load 2-name2text.txt
        // Load 3-bert/*.npy files
    }

    /// Get a batch of samples
    pub fn batch(&self, indices: &[usize]) -> T2SBatch {
        // Pad sequences and create batch tensors
    }
}
```

#### Phase 2: Training Loop (3-4 days)
```rust
// src/training/t2s_trainer.rs

use mlx_rs::{
    optimizers::{AdamW, Optimizer},
    transforms::value_and_grad,
    losses::cross_entropy,
};

pub struct T2STrainer {
    model: T2SModel,
    optimizer: AdamW,
    config: TrainingConfig,
}

impl T2STrainer {
    pub fn new(model: T2SModel, config: TrainingConfig) -> Self {
        let optimizer = AdamW::new(config.learning_rate)
            .with_betas([0.9, 0.95])
            .with_weight_decay(0.01);

        Self { model, optimizer, config }
    }

    /// Single training step
    pub fn step(&mut self, batch: &T2SBatch) -> Result<f32> {
        // Define loss function
        let loss_fn = |params: &[Array]| -> Vec<Array> {
            let logits = self.model.forward_with_params(
                params,
                &batch.phone_ids,
                &batch.bert_features,
            );
            let loss = cross_entropy(&logits, &batch.semantic_tokens, None)?;
            vec![loss]
        };

        // Compute gradients
        let mut grad_fn = value_and_grad(loss_fn);
        let (loss_val, grads) = grad_fn(&self.model.parameters())?;

        // Update parameters
        self.optimizer.update(&mut self.model, &grads)?;

        Ok(loss_val[0].item())
    }

    /// Full training loop
    pub fn train(&mut self, dataset: &T2SDataset, epochs: u32) -> Result<()> {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut steps = 0;

            for batch in dataset.iter_batches(self.config.batch_size) {
                let loss = self.step(&batch)?;
                total_loss += loss;
                steps += 1;

                if steps % 100 == 0 {
                    println!("Epoch {}, Step {}: loss = {:.4}",
                        epoch, steps, total_loss / steps as f32);
                }
            }

            // Save checkpoint
            if (epoch + 1) % self.config.save_every == 0 {
                self.save_checkpoint(epoch)?;
            }
        }
        Ok(())
    }
}
```

#### Phase 3: CLI Integration (1-2 days)
```rust
// examples/train_t2s.rs

use clap::Parser;

#[derive(Parser)]
struct Args {
    /// Preprocessed features directory
    #[arg(long)]
    exp_dir: PathBuf,

    /// Pretrained T2S model to fine-tune
    #[arg(long)]
    pretrained: PathBuf,

    /// Output model path
    #[arg(long)]
    output: PathBuf,

    /// Training epochs
    #[arg(long, default_value = "15")]
    epochs: u32,

    /// Batch size
    #[arg(long, default_value = "4")]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value = "0.0001")]
    lr: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load pretrained model
    let model = T2SModel::load(&args.pretrained)?;

    // Load dataset
    let dataset = T2SDataset::load(&args.exp_dir)?;
    println!("Loaded {} training samples", dataset.len());

    // Create trainer
    let config = TrainingConfig {
        learning_rate: args.lr,
        batch_size: args.batch_size,
        save_every: 5,
    };
    let mut trainer = T2STrainer::new(model, config);

    // Train
    trainer.train(&dataset, args.epochs)?;

    // Save final model
    trainer.save(&args.output)?;

    Ok(())
}
```

### Files to Create

| File | Description |
|------|-------------|
| `src/training/mod.rs` | Training module |
| `src/training/data.rs` | Data loading |
| `src/training/t2s_trainer.rs` | T2S training loop |
| `examples/train_t2s.rs` | CLI for T2S training |

### Expected Outcome
- Fine-tune T2S directly in Rust on Apple Silicon
- Leverage Metal acceleration for training
- No Python dependency for T2S fine-tuning

### Time Estimate
- Data loading: 2-3 days
- Training loop: 3-4 days
- CLI integration: 1-2 days
- Testing/debugging: 2-3 days
- **Total: 8-12 days**

---

## Approach 5: Port MultiPeriodDiscriminator to Rust MLX

### Overview
Port the GAN discriminator to Rust MLX to enable full VITS training in Rust (future milestone).

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              MULTIPERIOD DISCRIMINATOR (MPD)                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Audio Waveform                                              │
│       ↓                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Period = 2  │  │ Period = 3  │  │ Period = 5  │ ...     │
│  │ SubDiscrim  │  │ SubDiscrim  │  │ SubDiscrim  │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         ↓                ↓                ↓                 │
│     [scores]         [scores]         [scores]              │
│         └────────────────┼────────────────┘                 │
│                          ↓                                  │
│              discriminator_loss / generator_loss            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Components to Port

| Component | Python | Rust Effort |
|-----------|--------|-------------|
| `DiscriminatorP` (period sub-discriminator) | Conv2d stack | Medium |
| `DiscriminatorS` (scale sub-discriminator) | Conv1d stack | Medium |
| `MultiPeriodDiscriminator` | Wrapper | Low |
| `MultiScaleDiscriminator` | Wrapper | Low |
| Weight normalization | `weight_norm` | Medium |
| Spectral normalization | `spectral_norm` | High |

### Implementation Plan

#### Phase 1: Core Discriminator Blocks (3-4 days)
```rust
// src/models/discriminator.rs

use mlx_nn::{Conv1d, Conv2d, LeakyReLU};

/// Single period discriminator
#[derive(ModuleParameters)]
pub struct DiscriminatorP {
    period: usize,
    convs: Vec<Conv2d>,
    conv_post: Conv2d,
}

impl DiscriminatorP {
    pub fn new(period: usize, use_spectral_norm: bool) -> Self {
        // Build conv stack with appropriate normalization
    }

    pub fn forward(&self, x: &Array) -> (Array, Vec<Array>) {
        // Reshape to 2D based on period
        // Apply conv stack
        // Return (score, feature_maps)
    }
}

/// Single scale discriminator
#[derive(ModuleParameters)]
pub struct DiscriminatorS {
    convs: Vec<Conv1d>,
    conv_post: Conv1d,
}
```

#### Phase 2: Multi-discriminator Wrappers (1-2 days)
```rust
/// Multi-period discriminator (MPD)
#[derive(ModuleParameters)]
pub struct MultiPeriodDiscriminator {
    discriminators: Vec<DiscriminatorP>,  // periods: [2, 3, 5, 7, 11]
}

impl MultiPeriodDiscriminator {
    pub fn forward(&self, y: &Array, y_hat: &Array)
        -> (Vec<Array>, Vec<Array>, Vec<Vec<Array>>, Vec<Vec<Array>>)
    {
        // y_d_rs: real scores
        // y_d_gs: generated scores
        // fmap_rs: real feature maps
        // fmap_gs: generated feature maps
    }
}

/// Multi-scale discriminator (MSD)
#[derive(ModuleParameters)]
pub struct MultiScaleDiscriminator {
    discriminators: Vec<DiscriminatorS>,
}
```

#### Phase 3: GAN Loss Functions (1 day)
```rust
// src/training/gan_losses.rs

/// Discriminator loss (LSGAN style)
pub fn discriminator_loss(
    disc_real_outputs: &[Array],
    disc_gen_outputs: &[Array],
) -> (Array, Vec<f32>, Vec<f32>) {
    // L = E[(1 - D(y))²] + E[D(G(z))²]
}

/// Generator adversarial loss
pub fn generator_loss(disc_outputs: &[Array]) -> (Array, Vec<f32>) {
    // L = E[(1 - D(G(z)))²]
}

/// Feature matching loss
pub fn feature_loss(
    fmap_real: &[Vec<Array>],
    fmap_gen: &[Vec<Array>],
) -> Array {
    // L = Σ |fmap_real - fmap_gen|
}

/// KL divergence loss for VAE
pub fn kl_loss(
    z_p: &Array, logs_q: &Array,
    m_p: &Array, logs_p: &Array,
    z_mask: &Array,
) -> Array {
    // KL(q||p) for latent distribution
}
```

#### Phase 4: Full VITS Training Loop (3-4 days)
```rust
// src/training/vits_trainer.rs

pub struct VITSTrainer {
    generator: VITSModel,
    discriminator: MultiPeriodDiscriminator,
    optim_g: AdamW,
    optim_d: AdamW,
}

impl VITSTrainer {
    pub fn step(&mut self, batch: &VITSBatch) -> Result<TrainingMetrics> {
        // 1. Generator forward
        let (y_hat, kl_ssl, z, z_p, m_p, logs_p, m_q, logs_q) =
            self.generator.forward_train(&batch)?;

        // 2. Discriminator forward (detached)
        let (y_d_hat_r, y_d_hat_g, _, _) =
            self.discriminator.forward(&batch.audio, &y_hat.stop_gradient())?;

        // 3. Update discriminator
        let loss_disc = discriminator_loss(&y_d_hat_r, &y_d_hat_g)?;
        let grads_d = value_and_grad(|_| loss_disc.clone())?;
        self.optim_d.update(&mut self.discriminator, &grads_d)?;

        // 4. Discriminator forward (attached)
        let (y_d_hat_r, y_d_hat_g, fmap_r, fmap_g) =
            self.discriminator.forward(&batch.audio, &y_hat)?;

        // 5. Update generator
        let loss_mel = l1_loss(&y_mel, &y_hat_mel)?;
        let loss_kl = kl_loss(&z_p, &logs_q, &m_p, &logs_p, &z_mask)?;
        let loss_fm = feature_loss(&fmap_r, &fmap_g)?;
        let (loss_gen, _) = generator_loss(&y_d_hat_g)?;
        let loss_g = loss_gen + loss_fm + loss_mel + kl_ssl + loss_kl;

        let grads_g = value_and_grad(|_| loss_g.clone())?;
        self.optim_g.update(&mut self.generator, &grads_g)?;

        Ok(TrainingMetrics { loss_disc, loss_gen, loss_mel, loss_kl })
    }
}
```

### Files to Create

| File | Description |
|------|-------------|
| `src/models/discriminator.rs` | MPD and MSD models |
| `src/training/gan_losses.rs` | GAN loss functions |
| `src/training/vits_trainer.rs` | Full VITS training loop |
| `examples/train_vits.rs` | CLI for VITS training |

### Challenges

| Challenge | Difficulty | Notes |
|-----------|------------|-------|
| Spectral normalization | High | Need to implement in mlx-rs |
| Weight normalization | Medium | May need mlx-rs PR |
| Mel spectrogram in training | Medium | Need differentiable STFT |
| Memory management | High | GAN training is memory-intensive |

### Expected Outcome
- Full VITS training in Rust MLX
- No Python dependency for any training
- Native Apple Silicon acceleration

### Time Estimate
- Discriminator blocks: 3-4 days
- Multi-discriminator wrappers: 1-2 days
- GAN losses: 1 day
- VITS training loop: 3-4 days
- Testing/debugging: 3-4 days
- **Total: 11-15 days**

---

## Implementation Priority

**Note: Priority depends on your use case:**

### For Few-shot Voice Cloning (10-30s audio) — Most Common

| Approach | Priority | Why |
|----------|----------|-----|
| 1. Train SoVITS (Python) | **HIGHEST** | Direct timbre improvement, few-shot friendly |
| 2. V2Pro Support (Eres2Net) | HIGH | Better zero-shot speaker conditioning |
| 3. Full Pipeline | MEDIUM | Better UX but not core improvement |
| 4. T2S Fine-tuning | LOW | Not useful for <30s data (prosody needs more) |
| 5. MPD Discriminator | LOW | Future, for Rust-native training |

### For Full Fine-tune (10+ min audio)

| Approach | Priority | Why |
|----------|----------|-----|
| 1. Train SoVITS (Python) | **HIGH** | Core voice quality |
| 4. T2S Fine-tuning | **HIGH** | Prosody adaptation with enough data |
| 3. Full Pipeline | HIGH | End-to-end workflow |
| 2. V2Pro Support | MEDIUM | Less critical with fine-tuned models |
| 5. MPD Discriminator | LOW | Future enhancement |

### Recommended Implementation Order

1. **Immediate: Run Python SoVITS Training**
   - This is the PRIMARY improvement for few-shot voice cloning
   - Train SoVITS only (4-8 epochs for few-shot, 8-15 for medium data)
   - Convert checkpoint to safetensors
   - Test with pretrained T2S + trained SoVITS
   - **Result: Voice timbre matches reference**

2. **For Few-shot: V2Pro Support (Approach 2)**
   - Port Eres2Net to Rust MLX
   - Better speaker embedding for zero-shot conditioning
   - **Result: Improved zero-shot quality without training**

3. **For Full Fine-tune: T2S Fine-tuning (Approach 4)**
   - Only useful if you have 1+ minutes of audio
   - Implement data loading for preprocessed features
   - Build training loop with value_and_grad
   - **Result: Prosody/rhythm matches speaker**

4. **Full Pipeline Integration (Approach 3)**
   - Create unified CLI
   - Implement model registry
   - Add training orchestration
   - **Result: Better user experience**

5. **Future: MultiPeriodDiscriminator (Approach 5)**
   - Port discriminator architecture
   - Full VITS training in Rust
   - **Result: Complete Rust solution**

---

## Quick Start: Train SoVITS Now

**IMPORTANT: For few-shot (10-30s audio), train SoVITS ONLY. Do NOT train T2S/GPT.**

### Few-shot Training (10-30 seconds of audio)

```bash
# 1. Prepare 3-10 audio segments with transcripts

# 2. Run SoVITS training ONLY (4-8 epochs)
cd /Users/yuechen/home/OminiX-MLX/gpt-sovits-clone-mlx/MoYoYo.tts

python scripts/train_vits.py \
  --exp_name my_voice \
  --audio_dir /path/to/audio_segments \
  --transcript /path/to/train.list \
  --output_dir /tmp/voice_training/gpt_sovits_exp \
  --sovits_epochs 8 \  # 4-8 for few-shot
  --batch_size 4
  # NOTE: Do NOT add --gpt_epochs for few-shot!

# 3. Test inference with trained SoVITS + PRETRAINED T2S
cargo run --release --example voice_clone -- \
  --text "测试语音克隆" \
  --ref /path/to/reference.wav \
  --ref-text "参考音频文本" \
  --t2s-model /path/to/pretrained_t2s.safetensors \  # Use PRETRAINED T2S
  --vits-model ~/.dora/models/primespeech/gpt-sovits-mlx/my_voice_vits.safetensors \  # Use TRAINED SoVITS
  --output /tmp/output.wav
```

### Full Fine-tune Training (10+ minutes of audio)

```bash
# For 10+ minutes of audio, train BOTH models
python scripts/train_vits.py \
  --exp_name my_voice \
  --audio_dir /path/to/audio_segments \
  --transcript /path/to/train.list \
  --sovits_epochs 15 \  # 8-20 for full fine-tune
  --gpt_epochs 15       # Only add this with 10+ min of data
```

## Next Steps After SoVITS Training

### For Few-shot (10-30s): You're Done!

For few-shot voice cloning, SoVITS training is sufficient. The pretrained T2S handles prosody well enough. No need to train T2S.

### For Medium Data (1-5 min): Optional T2S Training

If you have 1-5 minutes of audio and want better prosody matching:

```bash
# Train T2S with Python (existing pipeline)
python scripts/train_gpt.py \
  --exp-dir /tmp/voice_training/gpt_sovits_exp \
  --epochs 10 \  # 5-10 for medium data
  --batch-size 4
```

### For Full Fine-tune (10+ min): T2S Fine-tuning

```bash
# Option A: Python training (works now)
python scripts/train_gpt.py \
  --exp-dir /tmp/voice_training/gpt_sovits_exp \
  --epochs 15 \  # 15+ for full fine-tune
  --batch-size 4

# Option B: Rust MLX (after implementing src/training/t2s_trainer.rs)
cargo run --release --example train_t2s -- \
  --exp-dir /tmp/voice_training/gpt_sovits_exp \
  --pretrained ~/.dora/models/primespeech/gpt-sovits-mlx/doubao_mixed_codes.bin \
  --output ~/.dora/models/primespeech/gpt-sovits-mlx/yuechen_t2s_rust.safetensors \
  --epochs 15 \
  --batch-size 4 \
  --lr 0.0001
```

### Future: Full Rust Training

Once MultiPeriodDiscriminator is ported:
```bash
cargo run --release --example train_vits -- \
  --exp-dir /tmp/voice_training/gpt_sovits_exp \
  --pretrained-g /path/to/pretrained_g.safetensors \
  --pretrained-d /path/to/pretrained_d.safetensors \
  --output ~/.dora/models/primespeech/gpt-sovits-mlx/yuechen_vits_rust.safetensors \
  --epochs 8
```

---

## Success Metrics

### Few-shot Scenario (10-30s audio)

| Metric | Current | After SoVITS Training | After V2Pro | Notes |
|--------|---------|----------------------|-------------|-------|
| Voice timbre | 20% | **80%+** | 85%+ | SoVITS is the key improvement |
| Prosody match | 70% | 70% | 75% | Can't improve much without data |
| Overall quality | 40% | **80%+** | 85%+ | Timbre is dominant factor |

### Full Fine-tune Scenario (10+ min audio)

| Metric | Current | After SoVITS | After SoVITS + T2S | Notes |
|--------|---------|--------------|-------------------|-------|
| Voice timbre | 20% | 85%+ | 85%+ | SoVITS handles timbre |
| Prosody match | 70% | 75% | **90%+** | T2S learns prosody |
| Overall quality | 40% | 85% | **95%+** | Full voice adaptation |

### System Metrics

| Metric | Current | After SoVITS | After T2S Rust | After Full Rust |
|--------|---------|--------------|----------------|-----------------|
| Training speed | N/A | 1x (Python) | 1.5x (MLX) | 2x (MLX) |
| Inference speed | 1x | 1x | 1x | 1x |
| Python dependency | Full | Partial | Minimal | None |

---

## Development Roadmap

### For Few-shot Voice Cloning (Most Common Use Case)

```
┌─────────────────────────────────────────────────────────────────────────┐
│              DEVELOPMENT ROADMAP (FEW-SHOT: 10-30s audio)                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  NOW ─────────────────────────────────────────────────────────► FUTURE  │
│                                                                          │
│  [1] Python SoVITS  [2] V2Pro Port   [3] Full      [4] Rust     [5] MPD │
│      Training           (Eres2Net)    Pipeline      SoVITS       Port   │
│      (1 day)           (2 weeks)     (2 weeks)    Training   (2 weeks) │
│         │                  │             │           │           │      │
│         ▼                  ▼             ▼           ▼           ▼      │
│   Voice TIMBRE       Better zero-    Unified    Native MLX   Complete   │
│   matches ref         shot clone       CLI      training    Rust TTS   │
│                                                                          │
│  Key insight: For few-shot, SoVITS (timbre) is PRIMARY.                 │
│  T2S training not useful with <30s of audio.                            │
│                                                                          │
│  Current state:                                                          │
│  ✅ T2S inference (pretrained works well for prosody)                   │
│  ✅ VITS inference                                                       │
│  ❌ SoVITS training → Train to match voice TIMBRE                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### For Full Fine-tune (10+ min audio)

```
┌─────────────────────────────────────────────────────────────────────────┐
│              DEVELOPMENT ROADMAP (FULL: 10+ min audio)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  NOW ─────────────────────────────────────────────────────────► FUTURE  │
│                                                                          │
│  [1] Python SoVITS  [2] Python T2S   [3] Rust T2S  [4] Full     [5] MPD │
│      Training           Training        Training   Pipeline      Port   │
│      (1 day)           (1 day)        (2 weeks)   (2 weeks)  (2 weeks) │
│         │                  │               │           │          │     │
│         ▼                  ▼               ▼           ▼          ▼     │
│   Voice TIMBRE       Voice PROSODY    Native MLX   Unified   Complete   │
│   matches ref        matches ref       training     CLI     Rust TTS   │
│                                                                          │
│  With 10+ min of audio, T2S training adds prosody adaptation.           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## References

- MoYoYo.tts Training Pipeline: `/Users/yuechen/home/OminiX-MLX/gpt-sovits-clone-mlx/MoYoYo.tts/training_pipeline/`
- GPT-SoVITS Original: https://github.com/RVC-Boss/GPT-SoVITS
- Eres2Net Paper: https://arxiv.org/abs/2305.02995
- MLX Documentation: https://ml-explore.github.io/mlx/
