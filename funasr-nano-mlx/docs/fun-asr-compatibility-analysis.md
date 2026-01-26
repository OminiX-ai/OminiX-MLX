# Fun-ASR Compatibility Analysis with funasr-mlx

## Executive Summary

**Can funasr-mlx be used for Fun-ASR?**

**NO - Major refactoring required.** Fun-ASR is a fundamentally different architecture (LLM-based) compared to Paraformer (CIF-based). Reusing funasr-mlx would require essentially a complete rewrite.

---

## Architecture Comparison

### Current funasr-mlx (Paraformer)

```
Audio (16kHz)
    ↓
[Mel Frontend] ─────────────────┐
    ↓                           │
[SAN-M Encoder] (50 layers)     │ Pure Audio Processing
    ↓                           │
[CIF Predictor] ────────────────┘
    ↓
[Bidirectional Decoder] (16 layers)
    ↓
Token IDs → Vocabulary → Text
```

**Characteristics:**
- Non-autoregressive (parallel decoding)
- CIF-based alignment
- 220M parameters
- Chinese only (8404 vocab)
- No LLM component
- Direct audio-to-text

### Fun-ASR (LLM-based)

```
Audio (16kHz)
    ↓
[Audio Encoder] (from ModelScope, architecture unknown)
    ↓
[Audio Adaptor] (Linear projection)
    ↓
[LLM Injection] → <|startofspeech|>...<|endofspeech|>
    ↓
[Causal LLM] (800M params, likely Qwen-based)
    ↓
Autoregressive Token Generation → Text
    │
    └──→ [CTC Branch] (Optional auxiliary task)
```

**Characteristics:**
- Autoregressive (sequential decoding)
- LLM-based generation
- 800M parameters (3.6x larger)
- 31 languages (MLT-Nano)
- Multimodal (audio injected into LLM)
- ChatML conversation format

---

## Component-by-Component Analysis

| Component | funasr-mlx (Paraformer) | Fun-ASR | Reusable? |
|-----------|-------------------------|---------|-----------|
| **Audio Frontend** | Mel spectrogram + LFR | Unknown (abstracted) | ❌ Likely different |
| **Encoder** | SAN-M (50 layers, 512 dim) | ModelScope (unknown arch) | ❌ Different |
| **Alignment** | CIF Predictor | Audio Adaptor (linear) | ❌ Completely different |
| **Decoder** | Bidirectional Transformer (16L) | Causal LLM (GPT-style) | ❌ Completely different |
| **Decoding** | Non-autoregressive | Autoregressive | ❌ Opposite approaches |
| **Output** | Token IDs → vocab lookup | LLM text generation | ❌ Different |
| **Vocabulary** | 8404 Chinese tokens | LLM tokenizer (50k+) | ❌ Different |

**Reusability Score: ~5%** (only basic utilities)

---

## What Can Be Reused

### 1. Audio Loading (audio.rs)
```rust
// These utilities can be reused:
- load_wav() - WAV file parsing
- resample() - Audio resampling to 16kHz
- stereo_to_mono() - Channel mixing
```

### 2. Basic Infrastructure
```rust
// Reusable patterns:
- Error handling (error.rs)
- MLX tensor operations
- Model parameter loading framework
```

### 3. Potentially Reusable
```rust
// If Fun-ASR encoder is Paraformer-based:
- Mel spectrogram computation (with modifications)
- Some attention mechanisms
```

---

## What Needs Complete Rewrite

### 1. New LLM Component (Major)

Fun-ASR requires a full causal language model:

```rust
// New implementation needed:
struct CausalLLM {
    embed_tokens: nn::Embedding,      // Token embeddings
    layers: Vec<TransformerBlock>,    // Causal attention layers
    lm_head: nn::Linear,              // Output projection
    // ... KV cache for efficient generation
}

struct TransformerBlock {
    self_attn: CausalSelfAttention,   // Masked attention
    mlp: MLP,                         // Feed-forward
    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
}
```

**Estimated effort:** 800-1500 lines of new code

### 2. Audio Adaptor

```rust
// New implementation:
struct AudioAdaptor {
    projection: nn::Linear,  // encoder_dim → llm_dim
    // Possibly upsampling/downsampling layers
}
```

**Estimated effort:** 50-100 lines

### 3. Multimodal Injection

```rust
// New implementation:
fn inject_audio_into_llm(
    text_tokens: &Array,      // LLM tokens with <|startofspeech|> marker
    audio_features: &Array,   // Encoded audio
) -> Array {
    // Replace speech markers with audio features
    // Handle variable-length audio sequences
}
```

**Estimated effort:** 100-200 lines

### 4. Autoregressive Decoding

```rust
// Complete rewrite of decoding:
fn generate(
    &mut self,
    prompt: &Array,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> Vec<i32> {
    // KV cache management
    // Token-by-token generation
    // Stopping criteria (EOS, max length)
}
```

**Estimated effort:** 200-400 lines

### 5. New Audio Encoder (Unknown Architecture)

The Fun-ASR audio encoder is loaded from ModelScope and its architecture is not publicly documented. Options:

**Option A:** Reverse-engineer from weights
```rust
// Load weights, inspect shapes, deduce architecture
// Risk: May not match exactly
```

**Option B:** Use existing Paraformer encoder (if compatible)
```rust
// Reuse SAN-M encoder if Fun-ASR uses similar arch
// Risk: May not be compatible
```

**Option C:** Implement generic transformer encoder
```rust
// Implement flexible encoder that can load various architectures
// Risk: Over-engineering
```

**Estimated effort:** 500-1000 lines (depending on approach)

---

## Effort Estimation

### Minimal Port (Inference Only)

| Component | Lines of Code | Effort |
|-----------|---------------|--------|
| LLM Core | 800-1500 | High |
| Audio Adaptor | 50-100 | Low |
| Multimodal Injection | 100-200 | Medium |
| Autoregressive Decoding | 200-400 | Medium |
| Audio Encoder (if new) | 500-1000 | High |
| Integration & Testing | 300-500 | Medium |
| **Total** | **2000-3700** | **High** |

**Timeline estimate:** 2-4 weeks for experienced Rust/ML developer

### Full Port (With Streaming)

Add:
| Component | Lines of Code | Effort |
|-----------|---------------|--------|
| KV Cache Management | 200-300 | Medium |
| Streaming Inference | 300-500 | Medium |
| Context Continuation | 100-200 | Low |
| **Additional** | **600-1000** | **Medium** |

**Timeline estimate:** 3-6 weeks

---

## Recommended Approach

### Option 1: Port Fun-ASR (New Project)

Create a new `fun-asr-mlx` project:

```
fun-asr-mlx/
├── src/
│   ├── lib.rs
│   ├── audio.rs          # Reuse from funasr-mlx
│   ├── encoder.rs        # New (reverse engineer or generic)
│   ├── adaptor.rs        # New
│   ├── llm.rs            # New (main effort)
│   ├── generation.rs     # New (autoregressive)
│   └── error.rs          # Reuse from funasr-mlx
└── Cargo.toml
```

**Pros:**
- Clean architecture
- No legacy constraints
- 800M model = better accuracy

**Cons:**
- Significant development effort
- LLM adds memory requirements (~3GB for 800M model)

### Option 2: Enhance funasr-mlx (Keep Paraformer)

Keep Paraformer and add features:
- Streaming support
- Batch processing
- English model (Paraformer-en)

**Pros:**
- Builds on working code
- Smaller model (220M) = faster
- Known architecture

**Cons:**
- Won't have Fun-ASR's 31-language support
- Won't have dialect/accent robustness

### Option 3: Hybrid Approach

1. Keep funasr-mlx for fast Chinese ASR (Paraformer)
2. Create separate fun-asr-mlx for multilingual/robust ASR
3. Share common utilities (audio loading, error handling)

**Pros:**
- Best of both worlds
- Different use cases
- Code reuse where possible

**Cons:**
- Two codebases to maintain

---

## Technical Challenges

### 1. Unknown Encoder Architecture

Fun-ASR's encoder is loaded from ModelScope without architecture details:

```python
# From model.py - encoder is a black box
self.audio_encoder = AutoModel.from_pretrained(encoder_config, hub="ms")
```

**Solutions:**
- Inspect weight shapes to reverse-engineer
- Contact FunAudioLLM team for architecture details
- Use generic transformer encoder

### 2. LLM Token Injection

Audio features must be injected at specific positions:

```python
# Speech markers in input
<|startofspeech|>![audio_url]<|endofspeech|>
```

This requires:
- Token position tracking
- Variable-length feature handling
- Attention mask management

### 3. Memory Requirements

| Model | Parameters | FP16 Memory | FP32 Memory |
|-------|------------|-------------|-------------|
| Paraformer | 220M | ~440MB | ~880MB |
| Fun-ASR-Nano | 800M | ~1.6GB | ~3.2GB |

Fun-ASR requires 3-4x more memory.

### 4. Autoregressive vs Non-autoregressive

Current funasr-mlx:
```rust
// Single forward pass, all tokens at once
let tokens = model.forward(&audio)?;  // O(1) passes
```

Fun-ASR would need:
```rust
// Token-by-token generation
for _ in 0..max_tokens {
    let next_token = model.forward_one(&input, &kv_cache)?;  // O(n) passes
    // Update cache, check stopping
}
```

This is a fundamental architectural difference.

---

## Conclusion

### Verdict: Major Refactoring Required

Fun-ASR cannot use the existing funasr-mlx inference code. The architectures are fundamentally different:

| Aspect | Paraformer | Fun-ASR |
|--------|------------|---------|
| Decoding | Non-autoregressive | Autoregressive |
| Core | CIF + Decoder | LLM |
| Alignment | CIF Predictor | Audio Adaptor |
| Generation | Parallel | Sequential |
| Model size | 220M | 800M |

### Recommendation

**If you need multilingual/robust ASR:** Create new `fun-asr-mlx` project (2-4 weeks effort)

**If Chinese ASR is sufficient:** Enhance funasr-mlx with streaming and batch support (1 week effort)

**Best long-term:** Hybrid approach with shared utilities

---

## Code Reuse Summary

| From funasr-mlx | Reusable in Fun-ASR Port |
|-----------------|--------------------------|
| `audio.rs` (load_wav, resample) | ✅ Yes |
| `error.rs` | ✅ Yes |
| `paraformer.rs` (MelFrontend) | ⚠️ Maybe (if encoder uses mel) |
| `paraformer.rs` (SAN-M Encoder) | ❌ No (different arch) |
| `paraformer.rs` (CIF Predictor) | ❌ No (not used) |
| `paraformer.rs` (Decoder) | ❌ No (need LLM) |
| `lib.rs` (Vocabulary) | ❌ No (need LLM tokenizer) |

**Bottom line:** ~200 lines reusable out of ~1800 lines (~11%)
