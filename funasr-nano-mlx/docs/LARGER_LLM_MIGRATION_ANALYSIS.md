# FunASR-Nano: Migration to Larger Qwen3 LLM Analysis

## Executive Summary

Switching funasr-nano to a larger Qwen3 model (e.g., 1.7B, 4B, 8B) that has built-in translation capability is **technically feasible** but requires **retraining the audio adaptor**. The SenseVoice encoder (~500M params) is **fully reusable**.

## Qwen3 Model Specifications

| Model | Params | hidden_size | Layers | Attention Heads | KV Heads | Context |
|-------|--------|-------------|--------|-----------------|----------|---------|
| Qwen3-0.6B | 0.6B | **1024** | 28 | 16 | 8 | 32K |
| Qwen3-1.7B | 1.7B | **2048** | 28 | 16 | 8 | 40K |
| Qwen3-4B | 4B | **2560** | 36 | 32 | 8 | 40K |
| Qwen3-8B | 8B | **4096** | 36 | 32 | 8 | 128K |

## Current FunASR-Nano Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      SenseVoice Encoder                          │
│  - 70 transformer blocks with SAN-M attention                   │
│  - Input: mel-spectrogram [batch, time, 560] (after LFR)        │
│  - Output: [batch, time', 512]                                  │
│  - ~500M parameters                                              │
│  - STATUS: FULLY REUSABLE                                        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Audio Adaptor                               │
│  - linear1: 512 → 2048 (input projection)                       │
│  - linear2: 2048 → 1024 (output to LLM hidden_size)             │
│  - 2 transformer blocks @ 1024-dim                              │
│  - ~10M parameters                                               │
│  - STATUS: NEEDS RETRAINING (output dimension changes)           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Qwen3-0.6B LLM                              │
│  - hidden_size: 1024                                            │
│  - 28 layers, 16 attention heads                                │
│  - ~600M parameters                                              │
│  - STATUS: REPLACED ENTIRELY                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Component Reusability Analysis

### 1. SenseVoice Encoder (FULLY REUSABLE)

```
Status: 100% Reusable
Params: ~500M
Changes: None required
```

The encoder processes audio into 512-dim feature vectors. This is independent of the LLM choice:
- Input: Mel-spectrogram (80 mels × 7 LFR stacking = 560-dim)
- Output: 512-dim audio features
- **No dependency on LLM hidden_size**

### 2. Audio Processing Pipeline (FULLY REUSABLE)

```
Status: 100% Reusable
Components:
  - WAV loading
  - Resampling to 16kHz
  - Mel-spectrogram computation
  - LFR (Low Frame Rate) stacking
```

These are pure signal processing operations with no model dependencies.

### 3. Audio Adaptor (NEEDS MODIFICATION)

```
Status: Partial reuse possible
Params: ~10M (current) → ~20-50M (larger LLM)
Changes: Output dimension, transformer blocks
```

**Current Adaptor Architecture:**
```
encoder_dim=512 → ffn_dim=2048 → llm_dim=1024
```

**Required Changes by Target LLM:**

| Target LLM | New llm_dim | linear2 change | Block changes |
|------------|-------------|----------------|---------------|
| Qwen3-1.7B | 2048 | 2048→2048 | Rebuild @ 2048 |
| Qwen3-4B | 2560 | 2048→2560 | Rebuild @ 2560 |
| Qwen3-8B | 4096 | 2048→4096 | Rebuild @ 4096 |

**Adaptor Training Requirement:**
- The adaptor must learn to project audio features into the target LLM's embedding space
- This requires training with paired (audio, text) data
- Estimated training: 10K-100K samples for convergence

### 4. Tokenizer (MOSTLY REUSABLE)

```
Status: 90% Reusable (same Qwen3 family)
Changes: May need special token ID adjustments
```

All Qwen3 models use the same tokenizer family. However:
- `<|startofspeech|>` and `<|endofspeech|>` token IDs may differ
- Need to verify special token mappings

### 5. LLM Weights (REPLACED ENTIRELY)

```
Status: 0% Reusable
Reason: Complete architecture change
```

The entire LLM is replaced with the new model (Qwen3-1.7B/4B/8B).

## Migration Path Analysis

### Option A: Qwen3-1.7B (Recommended First Step)

**Pros:**
- Smallest jump (2x hidden_size change: 1024→2048)
- Built-in instruction following and translation
- Reasonable memory footprint (~3GB VRAM)
- Performance: ~2x better than 0.6B on translation benchmarks

**Cons:**
- May still lack nuanced translation quality
- Moderate retraining effort

**Estimated Work:**
1. Adapt linear2: 2048→2048 (simplified - ffn_dim already matches)
2. Rebuild transformer blocks at 2048-dim
3. Train adaptor: ~1-2 days on M-series Mac

### Option B: Qwen3-4B (Balanced Choice)

**Pros:**
- Significant capability improvement
- Good translation quality
- Still fits on consumer hardware (~8GB VRAM)

**Cons:**
- Larger adaptor retraining effort
- 2.5x hidden_size change: 1024→2560

**Estimated Work:**
1. Rebuild adaptor for 2560-dim output
2. Train adaptor: ~2-3 days on M-series Mac

### Option C: Qwen3-8B (Maximum Quality)

**Pros:**
- Best translation quality (comparable to GPT-3.5)
- 128K context window
- Closest to Step-Audio 2's capability level

**Cons:**
- ~16GB VRAM requirement
- Largest retraining effort
- 4x hidden_size change: 1024→4096

**Estimated Work:**
1. Rebuild adaptor for 4096-dim output
2. Train adaptor: ~3-5 days on M-series Mac

## Step-Audio 2 Architecture Comparison

| Aspect | FunASR-Nano | Step-Audio 2 | Migration Target |
|--------|-------------|--------------|------------------|
| Encoder | SenseVoice (500M) | Whisper-large (550M) | Keep SenseVoice |
| Adaptor | 2-layer (10M) | Q-Former (50M) | Expand as needed |
| LLM | Qwen3-0.6B (600M) | Qwen2.5-7B (7B) | Qwen3-4B or 8B |
| Translation | None | Built-in | Built-in with Instruct |
| Total Params | ~1.1B | ~8B | ~5-9B |

**Key Insight from Step-Audio 2:**
- The LLM already has translation knowledge
- The adaptor only needs to learn audio→embedding projection
- Larger LLM = better semantic understanding = better translation

## Training Data Requirements

For adaptor retraining, you need:

1. **Audio-Text Pairs (Required)**
   - Chinese ASR data: LibriSpeech-like Chinese datasets
   - Estimate: 10K-50K hours for good quality

2. **Translation Pairs (Optional but Helpful)**
   - Audio → Chinese + English pairs
   - Helps if doing end-to-end translation training
   - Estimate: 1K-10K hours

**Recommended Datasets:**
- AISHELL-1 (170 hours Chinese ASR)
- AISHELL-2 (1000 hours Chinese ASR)
- WenetSpeech (10K hours Chinese ASR)
- CoVoST-2 (Speech translation)

## Implementation Checklist

### Phase 1: Architecture Changes

```rust
// src/adaptor.rs - Update for larger LLM
pub struct AdaptorConfig {
    pub encoder_dim: i32,  // Keep: 512
    pub ffn_dim: i32,      // Keep: 2048
    pub llm_dim: i32,      // Change: 1024 → 2048/2560/4096
    pub n_layer: i32,      // May increase: 2 → 4
}
```

```rust
// src/qwen.rs - Add support for larger configs
pub struct QwenConfig {
    pub hidden_size: i32,  // 1024 → 2048/2560/4096
    pub num_layers: i32,   // 28 → 28/36/36
    pub num_heads: i32,    // 16 → 16/32/32
    pub kv_heads: i32,     // 8 → 8/8/8
}
```

### Phase 2: Weight Loading

1. Download target Qwen3 model from HuggingFace
2. Convert to safetensors format compatible with MLX
3. Update weight mapping in `map_safetensors_key()`

### Phase 3: Adaptor Training

```python
# Training script pseudocode
for batch in dataloader:
    audio, text = batch

    # Freeze encoder
    with torch.no_grad():
        audio_features = encoder(audio)

    # Train adaptor
    adapted = adaptor(audio_features)

    # Freeze LLM or use LoRA
    logits = llm(adapted, text_tokens)

    loss = cross_entropy(logits, text_targets)
    loss.backward()
```

### Phase 4: Validation

1. Test Chinese ASR accuracy (should remain similar)
2. Test translation quality (should improve)
3. Benchmark latency (will increase with LLM size)

## Expected Outcomes

| Metric | Current (0.6B) | With 1.7B | With 4B | With 8B |
|--------|----------------|-----------|---------|---------|
| ASR CER | ~5% | ~5% | ~5% | ~5% |
| Translation BLEU | N/A | ~20 | ~25 | ~30 |
| Latency (1s audio) | 0.3s | 0.5s | 1.0s | 2.0s |
| Memory (inference) | 2GB | 4GB | 8GB | 16GB |

## Conclusion

Switching to a larger Qwen3 LLM is **the right approach** for adding translation capability:

1. **What's Reusable:**
   - SenseVoice encoder (100%)
   - Audio processing pipeline (100%)
   - Tokenizer (90%)
   - Training infrastructure (partially)

2. **What Needs Work:**
   - Audio adaptor: Rebuild output layers + retrain
   - LLM: Replace entirely
   - Integration: Update dimension constants

3. **Recommended Path:**
   - Start with Qwen3-1.7B (smallest change)
   - Validate adaptor training pipeline
   - Scale up to 4B/8B if quality insufficient

4. **Estimated Total Effort:**
   - Code changes: 1-2 days
   - Adaptor training: 2-5 days (depending on target LLM)
   - Validation: 1 day

## References

- [Qwen3 GitHub](https://github.com/QwenLM/Qwen3)
- [Qwen3 Technical Report](https://arxiv.org/pdf/2506.05176)
- [Step-Audio 2 Architecture](../step-audio2-mlx/)
- [FunASR-Nano Implementation](../src/model.rs)
