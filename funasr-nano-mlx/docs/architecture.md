# Fun-ASR-Nano-2512 Architecture Analysis

## Overview

**Model Size:** ~985M parameters (1.97GB in BFloat16)
**Architecture:** SenseVoice Encoder → Transformer Adaptor → Qwen3-0.6B LLM

This is **NOT** a Whisper-based model. It uses the SenseVoice encoder with SAN-M attention.

---

## Component Breakdown

| Component | Params | Layers | Description |
|-----------|--------|--------|-------------|
| Audio Encoder | 221M | 70 | SenseVoice with SAN-M attention |
| Audio Adaptor | 12.6M | 2 | Transformer adaptor |
| LLM | 751M | 28 | Qwen3-0.6B |
| **Total** | **~985M** | | |

---

## 1. Frontend (WavFrontend)

```yaml
fs: 16000            # Sample rate
window: hamming      # Window function
n_mels: 80           # Mel filterbank bins
frame_length: 25ms   # 400 samples
frame_shift: 10ms    # 160 samples
lfr_m: 7             # LFR stacking (7 frames)
lfr_n: 6             # LFR subsampling (every 6th frame)
```

**LFR (Low Frame Rate):**
- Stacks 7 consecutive frames → 80 * 7 = 560 features
- Subsamples by factor of 6 → 6x faster processing
- Input to encoder: `[batch, time/6, 560]`

---

## 2. Audio Encoder (SenseVoice, 221M params)

### Structure
```
encoders0 (1 layer)     # Initial block, input_dim=560 → output_dim=512
    ↓
encoders (49 layers)    # Main encoder blocks, dim=512
    ↓
tp_encoders (20 layers) # Temporal-parallel encoders, dim=512
    ↓
after_norm              # Final layer normalization
```

### Encoder Layer (SAN-M Attention)

Each layer contains:
```python
# Self-Attention with FSMN memory block
self_attn.linear_q_k_v: [1536, 512]  # Fused QKV projection
self_attn.fsmn_block:   [512, 1, 11] # FSMN memory, kernel=11
self_attn.linear_out:   [512, 512]   # Output projection

# Feed-Forward Network
feed_forward.w_1: [2048, 512]  # Up projection
feed_forward.w_2: [512, 2048]  # Down projection

# Layer Norms
norm1: [512]  # Pre-attention norm
norm2: [512]  # Pre-FFN norm
```

### SAN-M (Self-Attention with Memory)

SAN-M combines:
1. **Multi-Head Self-Attention** (4 heads, head_dim=128)
2. **FSMN Memory Block** - 1D convolution with kernel size 11

```python
# Pseudo-code
q, k, v = split(linear_q_k_v(x), 3)
attn_out = self_attention(q, k, v)
memory_out = fsmn_block(attn_out)  # 1D conv along time
out = linear_out(memory_out)
```

---

## 3. Audio Adaptor (12.6M params)

### Structure
```
linear1: [2048, 512]   # Input projection (encoder_dim → ffn_dim)
linear2: [1024, 2048]  # Output projection (ffn_dim → llm_dim)
    ↓
blocks (2 layers)      # Transformer blocks, dim=1024
```

### Adaptor Block

```python
# Self-Attention (separate Q/K/V projections)
self_attn.linear_q: [1024, 1024]
self_attn.linear_k: [1024, 1024]
self_attn.linear_v: [1024, 1024]
self_attn.linear_out: [1024, 1024]

# Feed-Forward (bottleneck)
feed_forward.w_1: [256, 1024]   # Down projection
feed_forward.w_2: [1024, 256]   # Up projection

# Layer Norms
norm1: [1024]  # Pre-attention
norm2: [1024]  # Pre-FFN
```

---

## 4. LLM (Qwen3-0.6B, 751M params)

### Configuration

```json
{
  "hidden_size": 1024,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "intermediate_size": 3072,
  "vocab_size": 151936,
  "rope_theta": 1000000,
  "rms_norm_eps": 1e-6,
  "tie_word_embeddings": true
}
```

### Layer Structure

```python
# Attention (GQA with QK-Norm)
self_attn.q_proj: [2048, 1024]  # 16 heads * 128
self_attn.k_proj: [1024, 1024]  # 8 kv_heads * 128
self_attn.v_proj: [1024, 1024]  # 8 kv_heads * 128
self_attn.o_proj: [1024, 2048]  # Output projection
self_attn.q_norm: [128]         # QK normalization
self_attn.k_norm: [128]

# MLP (SwiGLU)
mlp.gate_proj: [3072, 1024]
mlp.up_proj:   [3072, 1024]
mlp.down_proj: [1024, 3072]

# Layer Norms (RMSNorm)
input_layernorm: [1024]
post_attention_layernorm: [1024]
```

---

## 5. Weight Key Mapping

### From PyTorch to MLX

```
PyTorch Key                              → MLX Key
─────────────────────────────────────────────────────────────
audio_encoder.encoders.{i}.self_attn.*   → encoder.layers.{i}.attn.*
audio_encoder.encoders.{i}.feed_forward.* → encoder.layers.{i}.ffn.*
audio_encoder.encoders.{i}.norm{1,2}.*   → encoder.layers.{i}.norm{1,2}.*

audio_adaptor.linear{1,2}.*              → adaptor.proj{1,2}.*
audio_adaptor.blocks.{i}.*               → adaptor.layers.{i}.*

llm.model.embed_tokens.weight            → llm.embed_tokens.weight
llm.model.layers.{i}.*                   → llm.layers.{i}.*
llm.model.norm.weight                    → llm.norm.weight
llm.lm_head.weight                       → llm.lm_head.weight
```

---

## 6. Code Changes Required

### Replace whisper_encoder.rs → sensevoice_encoder.rs

1. Implement SAN-M attention with FSMN memory block
2. Three-stage encoder: encoders0 → encoders → tp_encoders
3. LFR frontend integration

### Update adaptor.rs

1. Change from simple Linear to 2-layer Transformer
2. Add input/output projections (linear1, linear2)
3. Implement bottleneck FFN (1024 → 256 → 1024)

### Update qwen.rs

1. hidden_size: 2048 → 1024
2. num_hidden_layers: 24 → 28
3. Add QK normalization (q_norm, k_norm)
4. Fix o_proj shape for GQA

### Add LFR to audio.rs

1. Implement frame stacking (m=7)
2. Implement subsampling (n=6)
3. Output shape: [batch, time/6, 560]

---

## 7. Inference Pipeline

```
Audio (16kHz)
    ↓
WavFrontend (Mel + LFR)
    ↓ [batch, T/6, 560]
SenseVoice Encoder (70 layers)
    ↓ [batch, T/6, 512]
Audio Adaptor (projection + 2 layers)
    ↓ [batch, T/6, 1024]
Qwen3 LLM (28 layers, autoregressive)
    ↓
Text tokens
```

---

## 8. Special Tokens

From Qwen3 tokenizer:
- `<|im_start|>`: 151644
- `<|im_end|>`: 151645
- `<|endoftext|>`: 151643

Speech markers (to be verified):
- `<|startofspeech|>`: TBD
- `<|endofspeech|>`: TBD
