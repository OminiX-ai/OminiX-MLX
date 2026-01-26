# Step-Audio 2 mini Architecture

## Overview

Step-Audio 2 mini is an 8B parameter end-to-end multimodal large language model for bidirectional audio understanding and generation. It supports:

- **ASR**: Automatic Speech Recognition (speech → text)
- **TTS**: Text-to-Speech synthesis (text → speech)
- **S2TT**: Speech-to-Text Translation (speech in lang A → text in lang B)
- **S2ST**: Speech-to-Speech Translation (speech in lang A → speech in lang B)
- **Conversation**: Multi-turn speech dialogue
- **Voice Cloning**: Clone voice from reference audio
- **Think Mode**: Extended reasoning before response (mini-Think variant)
- **Tool Calling**: Web search integration

## Model Variants

| Variant | Parameters | Training | Use Case |
|---------|------------|----------|----------|
| **mini-Base** | 8B | Pre-training only | Fine-tuning foundation |
| **mini** | 8B | Pre-training + SFT + RL | Production inference |
| **mini-Think** | 8B | + Think RL | Extended reasoning |

All variants share identical architecture; differences are in training only.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Step-Audio 2 mini                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT                                                                       │
│  ─────                                                                       │
│  Audio (16kHz) ──→ [Mel Spectrogram] ──→ [128, T]                           │
│                         128 mels                                             │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                     AUDIO ENCODER (Whisper-style)                   │     │
│  │                                                                     │     │
│  │  Mel [128, T]                                                       │     │
│  │       ↓                                                             │     │
│  │  Conv1d(128→1280, k=3, p=1) + GELU                                 │     │
│  │       ↓                                                             │     │
│  │  Conv1d(1280→1280, k=3, s=2, p=1) + GELU     ← 2x downsample       │     │
│  │       ↓                                                             │     │
│  │  + Positional Embedding [1500, 1280]                                │     │
│  │       ↓                                                             │     │
│  │  Transformer Blocks × 32                                            │     │
│  │  ┌─────────────────────────────────────┐                           │     │
│  │  │  LayerNorm                          │                           │     │
│  │  │  Multi-Head Attention (20 heads)    │                           │     │
│  │  │  + Residual                         │                           │     │
│  │  │  LayerNorm                          │                           │     │
│  │  │  MLP (1280 → 5120 → 1280) + GELU    │                           │     │
│  │  │  + Residual                         │                           │     │
│  │  └─────────────────────────────────────┘                           │     │
│  │       ↓                                                             │     │
│  │  AvgPool1d(k=2, s=2)                         ← 2x downsample       │     │
│  │       ↓                                                             │     │
│  │  Output: [B, T/4, 1280]                      (4x total downsample) │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                          ADAPTOR                                    │     │
│  │                                                                     │     │
│  │  [B, T/4, 1280]                                                     │     │
│  │       ↓                                                             │     │
│  │  Conv1d(1280→1280, k=3, s=2, p=1) + GELU     ← 2x downsample       │     │
│  │       ↓                                                             │     │
│  │  Linear(1280 → 2048) + GELU                                         │     │
│  │       ↓                                                             │     │
│  │  Linear(2048 → 3584)                          ← Project to LLM dim │     │
│  │       ↓                                                             │     │
│  │  Output: [B, T/8, 3584]                       (12.5 Hz frame rate) │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                     LLM (Qwen2.5-7B based)                          │     │
│  │                                                                     │     │
│  │  Config:                                                            │     │
│  │  ├─ hidden_size: 3584                                               │     │
│  │  ├─ num_hidden_layers: 28                                           │     │
│  │  ├─ num_attention_heads: 28                                         │     │
│  │  ├─ num_key_value_heads: 4 (GQA 7:1)                               │     │
│  │  ├─ intermediate_size: 18944                                        │     │
│  │  ├─ vocab_size: 158720                                              │     │
│  │  ├─ max_position_embeddings: 16384                                  │     │
│  │  └─ rope_theta: 1000000.0                                           │     │
│  │                                                                     │     │
│  │  Input Embedding:                                                   │     │
│  │  ├─ Text tokens → Embedding(158720, 3584)                          │     │
│  │  └─ Audio features → Direct injection (already 3584 dim)           │     │
│  │                                                                     │     │
│  │  Transformer Blocks × 28:                                           │     │
│  │  ┌─────────────────────────────────────┐                           │     │
│  │  │  RMSNorm(3584)                      │                           │     │
│  │  │  Attention:                         │                           │     │
│  │  │  ├─ Q: Linear(3584→3584, bias=True) │                           │     │
│  │  │  ├─ K: Linear(3584→512, bias=True)  │  (4 KV heads × 128 dim)  │     │
│  │  │  ├─ V: Linear(3584→512, bias=True)  │                           │     │
│  │  │  ├─ RoPE positional encoding        │                           │     │
│  │  │  ├─ Scaled Dot-Product Attention    │                           │     │
│  │  │  └─ O: Linear(3584→3584, bias=False)│                           │     │
│  │  │  + Residual                         │                           │     │
│  │  │  RMSNorm(3584)                      │                           │     │
│  │  │  MLP (SwiGLU):                      │                           │     │
│  │  │  ├─ gate: Linear(3584→18944)        │                           │     │
│  │  │  ├─ up: Linear(3584→18944)          │                           │     │
│  │  │  ├─ SiLU(gate) * up                 │                           │     │
│  │  │  └─ down: Linear(18944→3584)        │                           │     │
│  │  │  + Residual                         │                           │     │
│  │  └─────────────────────────────────────┘                           │     │
│  │                                                                     │     │
│  │  Output:                                                            │     │
│  │  ├─ RMSNorm(3584)                                                   │     │
│  │  └─ LM Head: Linear(3584→158720, bias=False)                       │     │
│  │                                                                     │     │
│  │  Generated Tokens:                                                  │     │
│  │  ├─ Text tokens: 0 - 151687                                         │     │
│  │  └─ Audio tokens: 151696 - 158256 (6561 codes)                     │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                     TTS DECODER (for audio output)                  │     │
│  │                                                                     │     │
│  │  Audio Tokens [151696-158256]                                       │     │
│  │       ↓                                                             │     │
│  │  S3Tokenizer (decode)                        ← ONNX model          │     │
│  │       ↓                                                             │     │
│  │  Semantic Codes [B, T]                                              │     │
│  │       ↓                                                             │     │
│  │  Flow-Matching Decoder (CosyVoice2)                                 │     │
│  │  ┌─────────────────────────────────────┐                           │     │
│  │  │  Input: semantic codes + prompt     │                           │     │
│  │  │  CFM Estimator (UNet-like):         │                           │     │
│  │  │  ├─ Encoder blocks                  │                           │     │
│  │  │  ├─ Cross-attention (prompt)        │                           │     │
│  │  │  └─ Decoder blocks                  │                           │     │
│  │  │  Denoising: 10 steps (rectified)    │                           │     │
│  │  │  Output: Mel spectrogram [80, T']   │                           │     │
│  │  └─────────────────────────────────────┘                           │     │
│  │       ↓                                                             │     │
│  │  HiFi-GAN Vocoder                                                   │     │
│  │  ┌─────────────────────────────────────┐                           │     │
│  │  │  Mel [80, T'] → Conv1d              │                           │     │
│  │  │  Upsample blocks × 4:               │                           │     │
│  │  │  ├─ ConvTranspose1d (8x, 8x, 2x, 2x)│  = 256x total            │     │
│  │  │  └─ ResBlocks (k=[3,7,11])          │                           │     │
│  │  │  Conv1d → tanh                      │                           │     │
│  │  │  Output: Waveform [1, T'×256]       │  24kHz                    │     │
│  │  └─────────────────────────────────────┘                           │     │
│  │       ↓                                                             │     │
│  │  OUTPUT: 24kHz Audio Waveform                                       │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Audio Frontend

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample rate | 16000 Hz | Input audio |
| N_FFT | 400 | 25ms window |
| Hop length | 160 | 10ms stride |
| N_mels | 128 | Mel filterbank channels |
| Output rate | 100 Hz | Before encoder downsampling |

### 2. Audio Encoder

| Parameter | Value |
|-----------|-------|
| Architecture | Whisper-style Transformer |
| Input channels | 128 (n_mels) |
| Hidden dim | 1280 |
| Attention heads | 20 |
| Head dim | 64 |
| Layers | 32 |
| MLP ratio | 4x (5120) |
| Activation | GELU |
| Positional encoding | Learned (max 1500) |
| Downsampling | 4x (conv stride 2 + avgpool 2) |
| Output rate | 25 Hz |
| Parameters | ~500M |

```python
# Encoder forward pass
def forward(self, mel):  # mel: [B, 128, T]
    x = gelu(self.conv1(mel))      # [B, 1280, T]
    x = gelu(self.conv2(x))        # [B, 1280, T/2]
    x = x.transpose(1, 2)          # [B, T/2, 1280]
    x = x + self.pos_embed[:, :x.size(1)]
    for block in self.blocks:      # 32 layers
        x = block(x)
    x = x.transpose(1, 2)          # [B, 1280, T/2]
    x = self.avg_pooler(x)         # [B, 1280, T/4]
    return x.transpose(1, 2)       # [B, T/4, 1280]
```

### 3. Audio Adaptor

| Parameter | Value |
|-----------|-------|
| Input dim | 1280 |
| Hidden dim | 2048 |
| Output dim | 3584 (LLM hidden) |
| Downsampling | 2x (conv stride 2) |
| Output rate | 12.5 Hz |
| Parameters | ~15M |

```python
# Adaptor forward pass
def forward(self, x):  # x: [B, T, 1280]
    x = x.transpose(1, 2)          # [B, 1280, T]
    x = gelu(self.conv(x))         # [B, 1280, T/2]
    x = x.transpose(1, 2)          # [B, T/2, 1280]
    x = gelu(self.linear1(x))      # [B, T/2, 2048]
    x = self.linear2(x)            # [B, T/2, 3584]
    return x
```

### 4. LLM (Qwen2.5-7B based)

| Parameter | Value |
|-----------|-------|
| Model type | Qwen2 |
| Hidden size | 3584 |
| Intermediate size | 18944 |
| Num layers | 28 |
| Num attention heads | 28 |
| Num KV heads | 4 |
| Head dim | 128 |
| Vocab size | 158720 |
| Max position | 16384 |
| RoPE theta | 1000000.0 |
| RMS norm eps | 1e-6 |
| Attention | GQA (7:1 ratio) |
| Attention bias | Q/K/V: True, O: False |
| QK norm | No (unlike Qwen3) |
| Activation | SiLU (SwiGLU) |
| Tie embeddings | No |
| Parameters | ~7.5B |

### 5. Token Vocabulary

| Range | Count | Purpose |
|-------|-------|---------|
| 0 - 151642 | 151643 | Standard text tokens |
| 151643 | 1 | `<\|endoftext\|>` / pad |
| 151644 | 1 | `<\|im_start\|>` |
| 151645 | 1 | `<\|im_end\|>` |
| 151646 | 1 | `<\|BOT\|>` |
| 151687 | 1 | `<\|EOT\|>` |
| 151688 | 1 | `<audio_patch>` |
| 151689-151695 | 7 | Reserved |
| 151696-158256 | 6561 | Audio tokens (CosyVoice2 codebook) |

### 6. S3Tokenizer

| Parameter | Value |
|-----------|-------|
| Model | ONNX (`speech_tokenizer_v2_25hz.onnx`) |
| Frame rate | 25 Hz |
| Codebook size | 6561 |
| Input | Audio waveform (24kHz) |
| Output | Discrete codes |

### 7. Flow-Matching Decoder (CosyVoice2)

| Parameter | Value |
|-----------|-------|
| Architecture | Conditional Flow Matching |
| Estimator | UNet-like with cross-attention |
| Denoising steps | 10 (inference) |
| Schedule | Rectified flow (linear) |
| Output | 80-dim mel spectrogram |
| Parameters | ~200M |

### 8. HiFi-GAN Vocoder

| Parameter | Value |
|-----------|-------|
| Input | 80-dim mel spectrogram |
| Upsample rates | [8, 8, 2, 2] = 256x |
| Upsample kernels | [16, 16, 4, 4] |
| ResBlock kernels | [3, 7, 11] |
| ResBlock dilations | [[1,3,5], [1,3,5], [1,3,5]] |
| Output | 24kHz waveform |
| Parameters | ~50M |

---

## Data Flow

### ASR Mode (Speech → Text)

```
Audio (16kHz)
    → Mel (128 bins, 100Hz)
    → Encoder (25Hz, 1280 dim)
    → Adaptor (12.5Hz, 3584 dim)
    → LLM (generate text tokens)
    → Tokenizer decode
    → Text output
```

### TTS Mode (Text → Speech)

```
Text input
    → Tokenizer encode
    → LLM (generate text + audio tokens)
    → Extract audio tokens (151696+)
    → S3Tokenizer decode
    → Flow decoder (→ mel)
    → HiFi-GAN (→ waveform)
    → Audio output (24kHz)
```

### Speech-to-Speech Mode

```
Audio input (16kHz)
    → Mel → Encoder → Adaptor
    → LLM (audio features + prompt)
    → LLM generates: text tokens + audio tokens (interleaved)
    → Audio tokens → S3Tokenizer → Flow → HiFi-GAN
    → Audio output (24kHz)
```

### Think Mode

```
Audio input
    → [Encoder → Adaptor → LLM]
    → Generate until </think>:
      "<think>\n{reasoning text}\n</think>"
    → Continue generating:
      Text response + Audio tokens
    → [TTS Decoder]
    → Audio output
```

---

## Memory Requirements

| Component | BF16 Size | Notes |
|-----------|-----------|-------|
| Audio Encoder | ~1.0 GB | 500M params |
| Adaptor | ~0.03 GB | 15M params |
| LLM | ~15.0 GB | 7.5B params |
| S3Tokenizer | ~0.05 GB | ONNX |
| Flow Decoder | ~0.4 GB | 200M params |
| HiFi-GAN | ~0.1 GB | 50M params |
| **Total weights** | **~16.5 GB** | |
| KV Cache (16K ctx) | ~4 GB | Runtime |
| Activations | ~2 GB | Runtime |
| **Total runtime** | **~22 GB** | Minimum |

### Quantization Options

| Precision | LLM Size | Total Model | Quality |
|-----------|----------|-------------|---------|
| BF16 | 15 GB | 16.5 GB | Best |
| INT8 | 7.5 GB | 9 GB | Good |
| INT4 | 3.75 GB | 5.25 GB | Acceptable |

---

## Comparison with Fun-ASR-Nano

| Aspect | Fun-ASR-Nano | Step-Audio 2 mini |
|--------|--------------|-------------------|
| Parameters | 985M | 8B |
| LLM | Qwen3-0.6B | Qwen2.5-7B |
| LLM hidden | 1024 | 3584 |
| Encoder | SenseVoice (SAN-M) | Whisper-style |
| Encoder dim | 512 | 1280 |
| N_mels | 80 (LFR: 560) | 128 |
| Adaptor | 2-layer Transformer | Conv + Linear |
| TTS | No | Yes (full) |
| Think mode | No | Yes |
| Tool calling | No | Yes (web search) |
| Memory | ~2.5 GB | ~22 GB |

---

## References

- Paper: [arXiv:2507.16632](https://arxiv.org/abs/2507.16632)
- Code: [github.com/stepfun-ai/Step-Audio2](https://github.com/stepfun-ai/Step-Audio2)
- Models: [huggingface.co/stepfun-ai](https://huggingface.co/stepfun-ai/Step-Audio-2-mini)
- License: Apache 2.0
