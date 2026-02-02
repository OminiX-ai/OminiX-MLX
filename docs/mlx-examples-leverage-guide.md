# MLX-Examples Leverage Guide for GPT-SoVITS

Reference guide for leveraging existing mlx-examples components in the GPT-SoVITS migration.

**Repository:** https://github.com/ml-explore/mlx-examples

---

## Component Mapping

| GPT-SoVITS Need | MLX-Examples Source | Files to Study |
|-----------------|---------------------|----------------|
| Audio encoder (CNHubert) | Whisper | `whisper/whisper/audio.py`, `whisper/whisper/model.py` |
| Mel-spectrogram | Whisper | `whisper/whisper/audio.py` |
| Transformer decoder | MusicGen | `musicgen/musicgen/transformer.py` |
| KV cache | MusicGen | `musicgen/musicgen/transformer.py` |
| Cross-attention | Whisper, T5 | `whisper/whisper/model.py`, `t5/t5.py` |
| RVQ (vector quantization) | EnCodec | `encodec/encodec/model.py` |
| Audio decoder/vocoder | EnCodec | `encodec/encodec/model.py` |
| Conv1d + upsampling | EnCodec | `encodec/encodec/model.py` |
| LSTM layers | EnCodec | `encodec/encodec/model.py` |
| Top-k sampling | MusicGen, LLMs | `musicgen/musicgen/generate.py` |
| Streaming generation | MusicGen | `musicgen/musicgen/generate.py` |

---

## 1. Whisper (Speech Recognition)

**Location:** `mlx-examples/whisper/`

### Key Components

#### Audio Preprocessing (`whisper/audio.py`)
```python
# Mel-spectrogram extraction - DIRECTLY USABLE
def log_mel_spectrogram(audio, n_mels=80):
    # STFT with 400-sample window, 160-sample hop
    # Mel filterbank (80 bins)
    # Log compression

# Usage:
mel = log_mel_spectrogram(audio_array)  # [n_mels, time]
```

#### Audio Encoder (`whisper/model.py`)
```python
class AudioEncoder(nn.Module):
    # Conv1d preprocessing (2 layers)
    # Sinusoidal positional embeddings
    # N transformer blocks with self-attention

    def __call__(self, x):
        # x: [batch, n_mels, time]
        x = self.conv1(x)  # → [batch, hidden, time/2]
        x = self.conv2(x)  # → [batch, hidden, time/4]
        x = x + self.positional_embedding
        for block in self.blocks:
            x = block(x)
        return x  # → [batch, time/4, hidden]
```

**Adaptation for CNHubert:**
- Change conv kernel sizes to match HuBERT
- Adjust hidden dimensions (768 for HuBERT)
- Add feature projection layer

### Files to Copy/Adapt
```
whisper/
├── whisper/
│   ├── audio.py        # ◄── Mel-spectrogram (copy)
│   ├── model.py        # ◄── AudioEncoder (adapt)
│   └── transcribe.py   # Generation patterns
```

---

## 2. EnCodec (Audio Codec)

**Location:** `mlx-examples/encodec/`

### Key Components

#### Residual Vector Quantizer (`encodec/model.py`)
```python
class ResidualVectorQuantizer(nn.Module):
    # Multi-codebook RVQ - MATCHES SOVITS EXACTLY
    # 8 codebooks with 1024 entries each

    def encode(self, x):
        # x: [batch, channels, time]
        codes = []
        residual = x
        for quantizer in self.layers:
            indices = quantizer.encode(residual)
            quantized = quantizer.decode(indices)
            residual = residual - quantized
            codes.append(indices)
        return codes  # List of [batch, time] tensors

    def decode(self, codes):
        # Reconstruct from codes
        quantized = sum(q.decode(c) for q, c in zip(self.layers, codes))
        return quantized
```

**Adaptation for SoVITS:**
- Use same 8-codebook structure
- Adjust codebook dimensions if needed
- Add commitment loss for training

#### SEANet Decoder (Vocoder)
```python
class SEANetDecoder(nn.Module):
    # Transposed convolutions for upsampling
    # Residual blocks with dilations
    # LSTM for temporal modeling

    def __call__(self, x):
        # x: [batch, channels, time]
        for block in self.model:
            x = block(x)
        return x  # Upsampled audio
```

**Adaptation for SoVITS:**
- Match upsampling ratios (16kHz → 32kHz)
- Add MRTE blocks if needed
- Integrate with duration predictor output

### Files to Copy/Adapt
```
encodec/
├── encodec/
│   ├── model.py        # ◄── RVQ + SEANet (adapt heavily)
│   └── utils.py        # Audio I/O utilities
```

---

## 3. MusicGen (Audio Generation)

**Location:** `mlx-examples/musicgen/`

### Key Components

#### Transformer Decoder (`musicgen/transformer.py`)
```python
class TransformerBlock(nn.Module):
    def __call__(self, x, cross_attn_src=None, mask=None, cache=None):
        # Self-attention with KV cache
        h, cache = self.self_attn(self.norm1(x), mask=mask, cache=cache)
        x = x + h

        # Cross-attention (if conditioning)
        if cross_attn_src is not None:
            h = self.cross_attn(self.norm2(x), cross_attn_src)
            x = x + h

        # FFN
        x = x + self.ffn(self.norm3(x))
        return x, cache
```

**DIRECTLY USABLE for GPT stage!**

#### KV Cache (`musicgen/transformer.py`)
```python
class KVCache:
    def __init__(self):
        self.keys = None
        self.values = None

    def update(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        return self.keys, self.values
```

**Note:** This is concat-based. For better performance, use step-allocated cache from mlx-rs.

#### Generation Loop (`musicgen/generate.py`)
```python
def generate(model, prompt, max_tokens, temperature=1.0, top_k=250):
    tokens = prompt
    cache = [None] * model.num_layers

    for _ in range(max_tokens):
        logits, cache = model(tokens[:, -1:], cache=cache)

        if temperature == 0:
            next_token = mx.argmax(logits, axis=-1)
        else:
            logits = logits / temperature
            if top_k > 0:
                # Top-k filtering
                indices = mx.argpartition(-logits, top_k, axis=-1)
                mask = mx.arange(logits.shape[-1]) >= top_k
                logits = mx.where(mask, -float('inf'), logits[indices])
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(probs)

        tokens = mx.concatenate([tokens, next_token], axis=1)

    return tokens
```

**Adaptation for GPT-SoVITS:**
- Add phoneme conditioning
- Integrate reference audio features
- Match sampling parameters (top_k=3, temp=0.8)

### Files to Copy/Adapt
```
musicgen/
├── musicgen/
│   ├── transformer.py  # ◄── TransformerBlock, Attention (copy)
│   ├── generate.py     # ◄── Generation loop (adapt)
│   └── model.py        # Full model structure
```

---

## 4. T5 (Encoder-Decoder)

**Location:** `mlx-examples/t5/`

### Key Components

#### Cross-Attention Pattern
```python
class TransformerDecoderLayer(nn.Module):
    def __call__(self, x, encoder_output, mask=None, cache=None):
        # Self-attention
        h, cache = self.self_attn(x, mask=mask, cache=cache)
        x = x + h

        # Cross-attention to encoder
        h = self.cross_attn(
            queries=x,
            keys=encoder_output,
            values=encoder_output
        )
        x = x + h

        # FFN
        x = x + self.ffn(x)
        return x, cache
```

**Use for SoVITS conditioning** (semantic tokens conditioned on speaker features)

#### Relative Position Bias
```python
class RelativePositionBias(nn.Module):
    # Learned relative position embeddings
    # More flexible than sinusoidal for variable lengths
```

---

## 5. Speech Commands / KWT

**Location:** `mlx-examples/speechcommands/`

### Audio Feature Embedding
```python
class PatchEmbed(nn.Module):
    def __call__(self, x):
        # x: [batch, freq, time]
        # Patch: [freq_patch, time_patch]
        B, F, T = x.shape
        x = x.reshape(B, F // self.patch_h, self.patch_h,
                      T // self.patch_w, self.patch_w)
        x = x.transpose(0, 1, 3, 2, 4).reshape(B, -1, self.patch_h * self.patch_w)
        return self.proj(x)
```

**Alternative to conv-based embedding** for audio transformers.

---

## Implementation Checklist

### Phase 1: GPT Stage (Python MLX)

- [ ] Copy `TransformerBlock` from MusicGen
- [ ] Copy `KVCache` (or implement step-allocated)
- [ ] Adapt generation loop for phoneme input
- [ ] Add semantic token output head
- [ ] Implement weight loading from PyTorch

### Phase 2: Audio Encoder

- [ ] Copy `log_mel_spectrogram` from Whisper
- [ ] Adapt `AudioEncoder` for HuBERT architecture
- [ ] Match hidden dimensions (768)
- [ ] Implement feature caching

### Phase 3: Vocoder

- [ ] Copy `ResidualVectorQuantizer` from EnCodec
- [ ] Adapt `SEANetDecoder` for SoVITS
- [ ] Implement duration predictor (new)
- [ ] Add MRTE blocks (new)

### Phase 4: Integration

- [ ] End-to-end pipeline
- [ ] Streaming synthesis
- [ ] Dora node integration
- [ ] Performance optimization

---

## Quick Start: Minimal GPT Stage

```python
# gpt_sovits_mlx.py - Minimal GPT stage implementation

import mlx.core as mx
import mlx.nn as nn

# Copy from musicgen
from musicgen.transformer import TransformerBlock, Attention

class GPTSoVITS(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.phoneme_embed = nn.Embedding(config.phoneme_vocab, config.hidden_size)
        self.semantic_embed = nn.Embedding(config.semantic_vocab, config.hidden_size)

        self.layers = [
            TransformerBlock(
                dims=config.hidden_size,
                num_heads=config.num_heads,
                mlp_dims=config.hidden_size * 4,
            )
            for _ in range(config.num_layers)
        ]

        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.semantic_vocab)

    def __call__(self, phoneme_ids, semantic_ids, cache=None):
        # Embed inputs
        phoneme_emb = self.phoneme_embed(phoneme_ids)
        semantic_emb = self.semantic_embed(semantic_ids)
        x = phoneme_emb + semantic_emb  # or concatenate

        # Transformer layers
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            x, new_layer_cache = layer(x, cache=layer_cache)
            new_cache.append(new_layer_cache)

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, new_cache


def generate_semantic_tokens(model, phoneme_ids, max_tokens=500, temperature=0.8, top_k=3):
    """Generate semantic tokens from phoneme sequence."""
    cache = [None] * len(model.layers)
    semantic_ids = mx.array([[0]])  # Start token

    for _ in range(max_tokens):
        logits, cache = model(phoneme_ids, semantic_ids[:, -1:], cache=cache)

        # Top-k sampling
        if temperature > 0:
            logits = logits[:, -1, :] / temperature
            top_k_indices = mx.argpartition(-logits, top_k, axis=-1)[:, :top_k]
            top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)
            probs = mx.softmax(top_k_logits, axis=-1)
            sampled_idx = mx.random.categorical(mx.log(probs))
            next_token = mx.take_along_axis(top_k_indices, sampled_idx[:, None], axis=-1)
        else:
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)

        semantic_ids = mx.concatenate([semantic_ids, next_token], axis=1)

        # Check for EOS (token 1024)
        if next_token.item() == 1024:
            break

    return semantic_ids[:, 1:]  # Remove start token
```

---

## Performance Tips

1. **Use `mx.compile`** for repeated functions:
   ```python
   @mx.compile
   def forward_step(model, x, cache):
       return model(x, cache=cache)
   ```

2. **Batch mel-spectrogram computation**:
   ```python
   # Process multiple audio chunks at once
   mels = mx.stack([log_mel_spectrogram(chunk) for chunk in chunks])
   ```

3. **Pre-allocate KV cache**:
   ```python
   # Step-allocated for better performance
   cache = [KVCache(max_len=512, step=64) for _ in range(num_layers)]
   ```

4. **Use `mx.async_eval`** for pipelining:
   ```python
   next_token = sample(logits)
   mx.async_eval(next_token)
   # Prepare next iteration while GPU computes
   ```

---

## Resources

- MLX Documentation: https://ml-explore.github.io/mlx/
- MLX Examples: https://github.com/ml-explore/mlx-examples
- GPT-SoVITS Paper: https://arxiv.org/abs/2401.13193
- This repository's mlx-rs: `/Users/yuechen/home/OminiX-MLX/mlx-rs-lm/`
