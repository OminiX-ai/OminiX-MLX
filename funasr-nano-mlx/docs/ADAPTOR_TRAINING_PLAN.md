# Adaptor Training Plan: FunASR-Nano → Qwen3-4B

## Objective

Train a new audio adaptor to project SenseVoice encoder output (512-dim) into Qwen3-4B embedding space (2560-dim), enabling ASR + translation in a single model.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SenseVoice Encoder                            │
│                    (FROZEN - 500M params)                        │
│                    Output: [batch, time, 512]                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Audio Adaptor (NEW)                           │
│                    (TRAINABLE - ~20M params)                     │
│                                                                  │
│    linear1: 512 → 2048  (keep from original)                    │
│    linear2: 2048 → 2560 (new output dim)                        │
│    transformer_blocks: 2-4 layers @ 2560-dim                    │
│                                                                  │
│                    Output: [batch, time, 2560]                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Qwen3-4B LLM                                  │
│                    (FROZEN or LoRA - 4B params)                  │
│                    hidden_size: 2560                             │
│                    layers: 36, heads: 32, kv_heads: 8            │
└─────────────────────────────────────────────────────────────────┘
```

## Training Phases

### Phase 1: Audio-Text Alignment

**Goal:** Adaptor learns to produce embeddings that match text embeddings in LLM space.

**Method:** Contrastive learning between audio and text representations.

**Data Required:**
- Audio-text pairs (transcriptions)
- ~1,000-10,000 hours

**Loss Function:**
```python
# InfoNCE contrastive loss
def alignment_loss(audio_embeds, text_embeds, temperature=0.07):
    # audio_embeds: [batch, seq, 2560] -> pool to [batch, 2560]
    # text_embeds: [batch, seq, 2560] -> pool to [batch, 2560]

    audio_pooled = audio_embeds.mean(dim=1)
    text_pooled = text_embeds.mean(dim=1)

    # Normalize
    audio_pooled = F.normalize(audio_pooled, dim=-1)
    text_pooled = F.normalize(text_pooled, dim=-1)

    # Cosine similarity matrix
    logits = audio_pooled @ text_pooled.T / temperature

    # Labels: diagonal is positive
    labels = torch.arange(len(logits), device=logits.device)

    loss = F.cross_entropy(logits, labels)
    return loss
```

**Expected Outcome:**
- Audio embeddings cluster near corresponding text embeddings
- LLM can "understand" audio as pseudo-text

---

### Phase 2: End-to-End ASR Fine-tuning

**Goal:** Full pipeline generates correct transcriptions from audio.

**Method:** Standard autoregressive language modeling loss.

**Data Required:**
- Same audio-text pairs as Phase 1
- Can use same dataset

**Loss Function:**
```python
def generation_loss(model, audio, text_tokens):
    # Encode audio
    audio_features = encoder(audio)           # [B, T, 512]
    adapted = adaptor(audio_features)         # [B, T, 2560]

    # Build input: [audio_embeds, text_embeds[:-1]]
    text_embeds = llm.embed(text_tokens[:, :-1])
    input_embeds = torch.cat([adapted, text_embeds], dim=1)

    # Forward through LLM
    logits = llm(inputs_embeds=input_embeds)

    # Only compute loss on text portion
    text_logits = logits[:, adapted.size(1):, :]

    loss = F.cross_entropy(
        text_logits.reshape(-1, vocab_size),
        text_tokens[:, 1:].reshape(-1)
    )
    return loss
```

**Expected Outcome:**
- Model accurately transcribes audio to text
- CER < 5% on Chinese ASR benchmarks

---

### Phase 3: Translation Fine-tuning (Optional)

**Goal:** Direct audio-to-English translation.

**Method:** Same as Phase 2, but with translation targets.

**Data Required:**
- Audio + English translation pairs
- CoVoST-2: ~500 hours Chinese→English

**Prompt Format:**
```
<|im_start|>system
You are a speech translation assistant.<|im_end|>
<|im_start|>user
Translate the following speech to English:<|startofspeech|>{AUDIO}<|endofspeech|><|im_end|>
<|im_start|>assistant
{ENGLISH_TRANSLATION}<|im_end|>
```

**Expected Outcome:**
- Direct speech-to-English translation
- BLEU > 20 on CoVoST-2

---

## Dataset Preparation

### Dataset Options

| Dataset | Size | Purpose | Download | License |
|---------|------|---------|----------|---------|
| **Emilia** | **50,000h Chinese** | ASR (recommended) | [HuggingFace](https://huggingface.co/datasets/amphion/Emilia-Dataset) | CC BY-NC 4.0 |
| AISHELL-1 | 170h | Chinese ASR | [OpenSLR](http://www.openslr.org/33/) | Apache 2.0 |
| AISHELL-2 | 1000h | Chinese ASR | [Registration](https://www.aishelltech.com/aishell_2) | Research |
| CoVoST-2 | 500h | Translation | `datasets` library | CC0 |

### Recommended: Emilia Dataset

Emilia provides 50,000 hours of Chinese speech - 50x more than AISHELL. Use streaming to avoid downloading all 4.5TB:

```python
from datasets import load_dataset

# Stream Chinese subset only
dataset = load_dataset(
    "amphion/Emilia-Dataset",
    data_files={"train": "Emilia/zh/**/*.tar"},
    split="train",
    streaming=True
)

# Sample 100K utterances (~1000 hours)
train_data = list(dataset.take(100000))
```

### Alternative: AISHELL (Smaller, Faster Start)

| Dataset | Size | Purpose | Download |
|---------|------|---------|----------|
| AISHELL-1 | 170h | Chinese ASR | [OpenSLR](http://www.openslr.org/33/) |
| AISHELL-2 | 1000h | Chinese ASR | [Registration](https://www.aishelltech.com/aishell_2) |
| CoVoST-2 | 500h | Translation | `datasets` library |

### Data Format

```json
{
  "audio_path": "/data/aishell/S0001/001.wav",
  "transcript": "开放时间是早上九点至下午五点",
  "translation": "Opening hours are from 9am to 5pm",
  "duration": 3.2,
  "language": "zh"
}
```

### Preprocessing Script

```python
# training/prepare_data.py

import json
from pathlib import Path
from datasets import load_dataset

def prepare_aishell(data_dir, output_file):
    """Convert AISHELL to training format."""
    samples = []

    transcript_file = data_dir / "transcript" / "aishell_transcript_v0.8.txt"
    with open(transcript_file) as f:
        for line in f:
            parts = line.strip().split()
            utt_id = parts[0]
            text = "".join(parts[1:])

            # Find audio file
            audio_path = find_audio(data_dir, utt_id)

            samples.append({
                "audio_path": str(audio_path),
                "transcript": text,
                "language": "zh"
            })

    with open(output_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

def prepare_covost2(output_file):
    """Download and prepare CoVoST-2."""
    ds = load_dataset("facebook/covost2", "zh-CN_en", split="train")

    samples = []
    for item in ds:
        samples.append({
            "audio_path": item["path"],
            "transcript": item["sentence"],
            "translation": item["translation"],
            "language": "zh"
        })

    with open(output_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
```

---

## Training Scripts

### Main Training Script

```python
# training/train_adaptor.py

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class AudioAdaptorQwen4B(nn.Module):
    """Adaptor for Qwen3-4B (2560 hidden_size)."""

    def __init__(self, encoder_dim=512, ffn_dim=2048, llm_dim=2560, n_layers=2):
        super().__init__()

        self.linear1 = nn.Linear(encoder_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, llm_dim)
        self.activation = nn.ReLU()

        # Transformer blocks at LLM dimension
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=llm_dim,
            nhead=8,
            dim_feedforward=llm_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        # x: [batch, time, 512]
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.transformer(x)
        return x  # [batch, time, 2560]


def train_phase1(adaptor, encoder, llm, dataloader, optimizer, epochs=10):
    """Phase 1: Contrastive alignment training."""

    adaptor.train()
    encoder.eval()
    llm.eval()

    for epoch in range(epochs):
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            audio = batch["audio"].cuda()
            text_tokens = batch["text_tokens"].cuda()

            # Get audio embeddings (frozen encoder)
            with torch.no_grad():
                audio_features = encoder(audio)

            # Adaptor (trainable)
            audio_embeds = adaptor(audio_features)

            # Get text embeddings (frozen LLM)
            with torch.no_grad():
                text_embeds = llm.get_input_embeddings()(text_tokens)

            # Contrastive loss
            loss = alignment_loss(audio_embeds, text_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")


def train_phase2(adaptor, encoder, llm, dataloader, optimizer, epochs=5):
    """Phase 2: End-to-end generation training."""

    adaptor.train()
    encoder.eval()
    # LLM: frozen or LoRA

    for epoch in range(epochs):
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            audio = batch["audio"].cuda()
            text_tokens = batch["text_tokens"].cuda()

            with torch.no_grad():
                audio_features = encoder(audio)

            audio_embeds = adaptor(audio_features)

            # Prepare LLM input
            text_embeds = llm.get_input_embeddings()(text_tokens[:, :-1])
            input_embeds = torch.cat([audio_embeds, text_embeds], dim=1)

            # Forward
            outputs = llm(inputs_embeds=input_embeds)
            logits = outputs.logits

            # Loss on text tokens only
            text_logits = logits[:, audio_embeds.size(1):, :]
            loss = nn.functional.cross_entropy(
                text_logits.reshape(-1, logits.size(-1)),
                text_tokens[:, 1:].reshape(-1),
                ignore_index=-100
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--encoder-path", type=str, required=True)
    parser.add_argument("--llm", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--output", type=str, default="adaptor_qwen4b.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # Load components
    print("Loading encoder...")
    encoder = load_sensevoice_encoder(args.encoder_path)
    encoder.cuda().eval()

    print(f"Loading LLM: {args.llm}")
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.llm)

    print("Creating adaptor...")
    adaptor = AudioAdaptorQwen4B(
        encoder_dim=512,
        llm_dim=2560  # Qwen3-4B hidden_size
    ).cuda()

    # Load data
    dataset = AudioTextDataset(args.data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(adaptor.parameters(), lr=args.lr)

    if args.phase == 1:
        train_phase1(adaptor, encoder, llm, dataloader, optimizer, args.epochs)
    elif args.phase == 2:
        train_phase2(adaptor, encoder, llm, dataloader, optimizer, args.epochs)

    # Save adaptor
    torch.save(adaptor.state_dict(), args.output)
    print(f"Saved adaptor to {args.output}")


if __name__ == "__main__":
    main()
```

---

## Remote Execution Plan

### 1. Transfer Files

```bash
# Create project directory on remote
ssh user@gpu-server "mkdir -p ~/funasr-qwen4b/training"

# Transfer training scripts
scp -r training/ user@gpu-server:~/funasr-qwen4b/

# Transfer encoder weights (from existing model)
scp ~/.dora/models/funasr-nano/model.safetensors user@gpu-server:~/funasr-qwen4b/
```

### 2. Setup Environment

```bash
ssh user@gpu-server << 'EOF'
cd ~/funasr-qwen4b

# Create conda environment
conda create -n funasr-qwen4b python=3.11 -y
conda activate funasr-qwen4b

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install safetensors tqdm
EOF
```

### 3. Download Data

**Option A: Emilia (Recommended - 50K hours Chinese)**

```bash
ssh user@gpu-server << 'EOF'
cd ~/funasr-qwen4b

# Login to HuggingFace (required for Emilia access)
huggingface-cli login

# Stream and prepare Emilia Chinese subset
python -c "
from datasets import load_dataset
import json

dataset = load_dataset(
    'amphion/Emilia-Dataset',
    data_files={'train': 'Emilia/zh/**/*.tar'},
    split='train',
    streaming=True
)

# Take 100K samples (~1000 hours)
with open('data/emilia_train.jsonl', 'w') as f:
    for i, sample in enumerate(dataset.take(100000)):
        f.write(json.dumps({
            'audio': sample['mp3'],
            'transcript': sample['json']['text'],
            'language': 'zh'
        }, ensure_ascii=False) + '\n')
        if i % 10000 == 0:
            print(f'Processed {i} samples')
"
EOF
```

**Option B: AISHELL (Smaller, faster download)**

```bash
ssh user@gpu-server << 'EOF'
cd ~/funasr-qwen4b

# Download AISHELL-1
wget https://www.openslr.org/resources/33/data_aishell.tgz
tar -xzf data_aishell.tgz
EOF
```

**For Translation (Phase 3)**

```bash
ssh user@gpu-server << 'EOF'
# Download CoVoST-2 Chinese→English
python -c "from datasets import load_dataset; load_dataset('facebook/covost2', 'zh-CN_en')"
EOF
```

### 4. Run Training

```bash
# Phase 1: Alignment (background)
ssh user@gpu-server "cd ~/funasr-qwen4b && \
  nohup python training/train_adaptor.py \
    --phase 1 \
    --data data/aishell_train.jsonl \
    --encoder-path model.safetensors \
    --epochs 10 \
    > phase1.log 2>&1 &"

# Monitor progress
ssh user@gpu-server "tail -f ~/funasr-qwen4b/phase1.log"
```

### 5. Retrieve Results

```bash
# Download trained adaptor
scp user@gpu-server:~/funasr-qwen4b/adaptor_qwen4b.pt ./

# Download logs
scp user@gpu-server:~/funasr-qwen4b/*.log ./
```

---

## Training Schedule

| Phase | Duration | GPU Memory | Dataset |
|-------|----------|------------|---------|
| Phase 1 | 1-2 days | ~20GB | AISHELL-1/2 |
| Phase 2 | 1-2 days | ~24GB | AISHELL-1/2 |
| Phase 3 | 1 day | ~24GB | CoVoST-2 |

**Total: 3-5 days on single A100/H100**

---

## Validation Checkpoints

### After Phase 1
```python
# Check embedding alignment
audio_embed = adaptor(encoder(test_audio))
text_embed = llm.embed(tokenizer("测试文本"))
similarity = cosine_similarity(audio_embed.mean(1), text_embed.mean(1))
# Should be > 0.7 for matching pairs
```

### After Phase 2
```python
# Check ASR accuracy
transcription = generate(model, test_audio)
cer = compute_cer(transcription, ground_truth)
# Should be < 10% CER
```

### After Phase 3
```python
# Check translation quality
translation = generate(model, test_audio, task="translate")
bleu = compute_bleu(translation, reference)
# Should be > 15 BLEU
```

---

## Output Files

```
funasr-qwen4b/
├── adaptor_qwen4b_phase1.pt    # After alignment
├── adaptor_qwen4b_phase2.pt    # After ASR fine-tuning
├── adaptor_qwen4b_final.pt     # After translation (optional)
├── phase1.log
├── phase2.log
└── config.json                  # Adaptor config
```

## Integration Back to MLX

After training, convert the adaptor to MLX format:

```python
# Convert PyTorch adaptor to safetensors
import torch
from safetensors.torch import save_file

state_dict = torch.load("adaptor_qwen4b_final.pt")
save_file(state_dict, "adaptor_qwen4b.safetensors")
```

Then update the Rust code to load the new adaptor with Qwen3-4B.
