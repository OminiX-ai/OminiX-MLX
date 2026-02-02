# FunASR-Qwen4B-MLX Development Plan

Port the trained PyTorch audio adaptor + Qwen3-4B to Rust MLX for native Apple Silicon inference.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    funasr-qwen4b-mlx                            │
├─────────────────────────────────────────────────────────────────┤
│  Audio (WAV 16kHz)                                              │
│       ↓                                                         │
│  [Mel + LFR] ────────────────── Reuse from funasr-nano-mlx     │
│       ↓ (560-dim)                                               │
│  [SenseVoice Encoder] ───────── Reuse from funasr-nano-mlx     │
│       ↓ (512-dim)                                               │
│  [Audio Adaptor 4-layer] ────── NEW: Port from PyTorch         │
│       ↓ (2560-dim)                                              │
│  [Qwen3-4B LLM] ─────────────── NEW: Extend qwen.rs            │
│       ↓                                                         │
│  Text Output                                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    OminiX-API Integration                       │
├─────────────────────────────────────────────────────────────────┤
│  ~/home/OminiX-API/                                             │
│  ├── src/asr.rs         ← Paraformer (CTC-based, existing)     │
│  ├── src/asr_qwen4b.rs  ← NEW: Qwen4B (LLM-based)              │
│  └── src/main.rs        ← Route: /v1/audio/transcriptions      │
│                                                                 │
│  POST /v1/audio/transcriptions                                  │
│  { "model": "funasr-qwen4b", "file": "base64...", ... }        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Priority Levels

| Priority | Meaning | Timeline |
|----------|---------|----------|
| **P0.x** | Critical path - blocks everything | Day 1-2 |
| **P1.x** | Core functionality | Day 2-4 |
| **P2.x** | Full feature parity | Day 4-6 |
| **P3.x** | Optimization & polish | Day 6+ |

---

## P0: Foundation (Critical Path)

### P0.0 - Project Setup
**Create Rust project structure**

```bash
funasr-qwen4b-mlx/
├── Cargo.toml              # Dependencies
├── src/
│   ├── lib.rs              # Public API
│   ├── model.rs            # Main model (copy from nano, modify)
│   ├── adaptor.rs          # NEW: 4-layer adaptor for 2560-dim
│   ├── qwen4b.rs           # NEW: Qwen3-4B config
│   ├── sensevoice_encoder.rs  # Copy from nano (unchanged)
│   ├── audio.rs            # Copy from nano (unchanged)
│   └── error.rs            # Copy from nano (unchanged)
├── examples/
│   └── transcribe.rs       # Basic inference example
└── scripts/
    └── convert_weights.py  # PyTorch → safetensors
```

**Tasks:**
- [ ] Initialize Cargo.toml with mlx-rs dependencies
- [ ] Copy reusable files from funasr-nano-mlx
- [ ] Create module structure

**Deliverable:** Project compiles (no functionality yet)

---

### P0.1 - Weight Conversion Script
**Convert PyTorch weights to MLX safetensors format**

```python
# scripts/convert_weights.py
# Input: adaptor_phase2_final.pt (PyTorch)
# Output: adaptor.safetensors (MLX compatible)

# Key mappings:
# PyTorch                    → Safetensors
# input_proj.weight          → adaptor.input_proj.weight
# input_proj.bias            → adaptor.input_proj.bias
# transformer.layers.0.*     → adaptor.blocks.0.*
# output_proj.weight         → adaptor.output_proj.weight
# norm.weight                → adaptor.norm.weight
```

**Tasks:**
- [ ] Write conversion script for adaptor weights
- [ ] Verify tensor shapes match expected dimensions
- [ ] Test loading in Python first

**Deliverable:** `adaptor.safetensors` file (~800MB)

---

### P0.2 - Adaptor Architecture (Rust)
**Implement 4-layer transformer adaptor matching PyTorch**

```rust
// src/adaptor.rs

pub struct AudioAdaptorQwen4B {
    pub input_proj: nn::Linear,      // 512 → 2048
    pub blocks: Vec<AdaptorBlock>,   // 4 transformer layers @ 2048
    pub output_proj: nn::Linear,     // 2048 → 2560
    pub norm: nn::LayerNorm,         // LayerNorm(2560)
}

pub struct AdaptorBlock {
    pub self_attn: TransformerAttention,  // 8 heads, 256 head_dim
    pub feed_forward: AdaptorFFN,          // 2048 → 8192 → 2048
    pub norm1: nn::LayerNorm,
    pub norm2: nn::LayerNorm,
}

// Dimensions:
// - hidden_dim: 2048
// - n_heads: 8
// - head_dim: 256 (2048 / 8)
// - ffn_dim: 8192 (2048 * 4)
// - output_dim: 2560
```

**Tasks:**
- [ ] Define AdaptorBlock struct
- [ ] Implement forward() with pre-norm residual
- [ ] Match PyTorch nn.TransformerEncoderLayer behavior
- [ ] Add weight loading from safetensors

**Deliverable:** Adaptor compiles and loads weights

---

## P1: Core Functionality

### P1.0 - Qwen3-4B Configuration ✅ COMPLETE
**Reuse qwen3-mlx crate instead of custom implementation**

```rust
// Cargo.toml - Uses existing qwen3-mlx crate
qwen3-mlx = { path = "../qwen3-mlx" }

// Actual Qwen3-4B config (from config.json):
hidden_size: 2560
num_hidden_layers: 36
num_attention_heads: 32      // Not 20!
num_key_value_heads: 8       // GQA
head_dim: 128
intermediate_size: 9728      // Not 6912!
vocab_size: 151936
rope_theta: 1000000.0
tie_word_embeddings: true    // No separate lm_head
```

**Tasks:**
- [x] Extract config from Qwen3-4B HuggingFace repo
- [x] Verify GQA head configuration (32 heads, 8 KV heads)
- [x] Reuse qwen3-mlx crate (no custom qwen4b.rs needed)

**Deliverable:** Config via qwen3-mlx ✅

---

### P1.1 - Qwen3-4B Weight Loading ✅ COMPLETE
**Handled by qwen3-mlx's load_model()**

```rust
// src/model.rs
use qwen3_mlx::{load_model, load_tokenizer};

let llm = load_model(&qwen_path)?;  // Handles sharded safetensors
let tokenizer = load_tokenizer(&qwen_path)?;
```

**Tasks:**
- [x] Download Qwen3-4B weights (~7.5GB)
- [x] qwen3-mlx handles key mapping automatically
- [x] qwen3-mlx handles sharded safetensors
- [x] Verified loaded weights match expected shapes

**Deliverable:** Qwen3-4B loads via qwen3-mlx ✅

---

### P1.2 - Qwen3-4B Forward Pass ✅ COMPLETE
**Uses qwen3-mlx's Generate iterator**

```rust
// Standard text generation via qwen3-mlx
let generator = Generate::<KVCache>::new(&mut model, &mut cache, temp, &prompt_tokens);
for token_result in generator.take(max_tokens) {
    // ...
}

// Multimodal: Access internals directly
model.model.embed_tokens  // Get embeddings
model.model.layers        // Run through layers
model.model.norm          // Final normalization
```

**Tasks:**
- [x] qwen3-mlx handles attention with 32 heads / 8 KV heads
- [x] qwen3-mlx handles MLP with correct intermediate size
- [x] Text generation works (~9 tok/s)
- [x] Verified output is coherent

**Deliverable:** Qwen3-4B generates text correctly ✅

---

### P1.3 - Adaptor-LLM Integration ✅ COMPLETE
**Connect adaptor output to Qwen3-4B embeddings**

**Implementation (aligned with funasr-nano-mlx):**

```rust
// src/model.rs - Key methods

/// Get token embeddings (for multimodal injection)
fn get_token_embeddings(&mut self, tokens: &Array) -> Result<Array>

/// Forward pass with embedding inputs (for multimodal)
fn forward_embeddings(&mut self, embeddings: &Array, cache: &mut Vec<Option<KVCache>>) -> Result<Array>

/// Generate from audio features
fn generate_from_audio_features(&mut self, audio_features: &Array, prompt: &str) -> Result<String>
```

**Multimodal Embedding Injection Pattern:**
1. Build full prompt with zeros as audio placeholders
2. Get all embeddings at once via `get_token_embeddings()`
3. Replace placeholder embeddings with actual audio features
4. Run `forward_embeddings()` through transformer layers
5. Autoregressive decoding with KV cache

**ChatML Format (matches funasr-nano-mlx):**
```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
<|im_start|>user\n{prompt}<|startofspeech|>{AUDIO}<|endofspeech|><|im_end|>\n
<|im_start|>assistant\n
```

**Tasks:**
- [x] Update embedding concatenation for 2560-dim
- [x] Test adaptor → LLM connection
- [x] Verify gradient-free inference works
- [x] Add repetition detection (like nano)

**Deliverable:** Audio features flow through full pipeline ✅

---

## P2: Full Feature Parity

### P2.0 - End-to-End Inference ✅ COMPLETE
**Complete audio → text pipeline**

```rust
// examples/transcribe.rs
fn main() {
    let model = FunASRQwen4B::load("models/")?;
    let text = model.transcribe("audio.wav")?;
    println!("{}", text);
}

// Also available:
model.transcribe_samples(&samples, sample_rate)?;  // From raw samples
model.encode_audio(&samples, sample_rate)?;         // Get audio features only
```

**API Methods Implemented:**
- `transcribe(path)` - Load WAV and transcribe
- `transcribe_samples(&[f32], sample_rate)` - From raw samples (streaming)
- `encode_audio(&[f32], sample_rate)` - Get adapted features without generation
- `transcribe_and_translate(path)` - Chinese + English output
- `translate(chinese)` - Text-only translation
- `generate_text(prompt, max_tokens)` - Text-only LLM
- `complete(prompt, max_tokens, temperature)` - With temperature

**Tasks:**
- [x] Implement `transcribe()` API
- [x] Handle audio loading + preprocessing
- [x] Add sampling parameters (temperature via `complete()`)
- [x] Test with real audio - WORKING but accuracy limited (see Known Issues)

**Deliverable:** Working transcription from audio files ✅

**Test Results (zh.wav - "开放时间：早上九点至下午五点"):**
- Output: "上早八九点至时末五点日。"
- Key content recognized: "九点", "五点", "至"
- Speed: 7.4 tok/s, 38ms encoder, 30ms adaptor

---

### P2.1 - Translation Support
**Add Chinese → English translation**

```rust
pub fn transcribe_and_translate(&self, audio_path: &str) -> Result<(String, String)> {
    let chinese = self.transcribe(audio_path)?;
    let english = self.translate(&chinese)?;
    Ok((chinese, english))
}

fn translate(&self, chinese: &str) -> Result<String> {
    let prompt = format!(
        "Translate to English: {}\nEnglish:",
        chinese
    );
    // Use enable_thinking=false equivalent
    self.generate_text(&prompt)
}
```

**Tasks:**
- [ ] Add translation prompt template
- [ ] Implement thinking mode disable (prompt-based)
- [ ] Test translation quality

**Deliverable:** Chinese + English output

---

### P2.2 - Streaming Output
**Token-by-token generation for real-time display**

```rust
pub fn transcribe_streaming<F>(&self, audio_path: &str, callback: F) -> Result<String>
where
    F: FnMut(&str),
{
    // Call callback for each generated token
}
```

**Tasks:**
- [ ] Implement token callback mechanism
- [ ] Add to CLI example
- [ ] Test streaming performance

**Deliverable:** Real-time text output

---

### P2.3 - Batch Processing
**Process multiple audio files efficiently**

```rust
pub fn transcribe_batch(&self, paths: &[&str]) -> Result<Vec<String>> {
    // Batch audio preprocessing
    // Single forward pass through encoder
    // Sequential LLM generation (or batched if memory allows)
}
```

**Tasks:**
- [ ] Implement batch audio loading
- [ ] Optimize memory for batch processing
- [ ] Add progress reporting

**Deliverable:** Efficient batch transcription

---

## P2.5: API Integration (OminiX-API)

### P2.5.0 - Add Crate Dependency
**Add funasr-qwen4b-mlx to OminiX-API**

```toml
# ~/home/OminiX-API/Cargo.toml
[dependencies]
funasr-qwen4b-mlx = { path = "../OminiX-MLX/funasr-qwen4b-mlx" }
```

**Tasks:**
- [ ] Ensure funasr-qwen4b-mlx exports clean public API
- [ ] Add to Cargo.toml dependencies
- [ ] Verify compilation

---

### P2.5.1 - New ASR Engine (LLM-based)
**Create AsrQwen4BEngine in OminiX-API**

```rust
// ~/home/OminiX-API/src/asr_qwen4b.rs

pub struct AsrQwen4BEngine {
    model: funasr_qwen4b_mlx::FunASRQwen4B,
}

impl AsrQwen4BEngine {
    pub fn new(model_dir: &str) -> Result<Self> {
        let model = funasr_qwen4b_mlx::FunASRQwen4B::load(model_dir)?;
        Ok(Self { model })
    }

    pub fn transcribe(&self, request: &TranscriptionRequest) -> Result<TranscriptionResponse> {
        // Similar to existing asr.rs but using LLM-based model
    }

    pub fn transcribe_and_translate(&self, request: &TranscriptionRequest)
        -> Result<TranslationResponse> {
        // Chinese + English output
    }
}
```

**Tasks:**
- [ ] Create asr_qwen4b.rs
- [ ] Implement transcribe() API
- [ ] Implement transcribe_and_translate() API
- [ ] Add streaming support

---

### P2.5.2 - API Endpoint
**Add /v1/audio/transcriptions/qwen4b endpoint**

```rust
// Option 1: New endpoint
POST /v1/audio/transcriptions/qwen4b

// Option 2: Model parameter
POST /v1/audio/transcriptions
{
    "model": "funasr-qwen4b",  // vs "paraformer"
    "file": "base64...",
    "language": "zh",
    "translate": true  // Optional: also return English
}
```

**Tasks:**
- [ ] Add route in main.rs
- [ ] Add model selection parameter
- [ ] Update types.rs for new response fields
- [ ] Test with curl/Postman

---

### P2.5.3 - Model Loading Strategy
**Lazy loading to save memory**

```rust
pub struct AsrEngines {
    paraformer: Option<AsrEngine>,         // CTC-based (smaller)
    qwen4b: Option<AsrQwen4BEngine>,       // LLM-based (larger)
}

impl AsrEngines {
    pub fn load_on_demand(&mut self, model: &str) -> Result<()> {
        match model {
            "paraformer" => self.load_paraformer(),
            "funasr-qwen4b" => self.load_qwen4b(),
            _ => Err(...)
        }
    }
}
```

**Tasks:**
- [ ] Implement lazy loading
- [ ] Add model unloading to free memory
- [ ] Config for default model

**Deliverable:** API serves both Paraformer and Qwen4B ASR

---

## P3: Optimization & Polish

### P3.0 - Memory Optimization
**Reduce memory footprint for smaller Macs**

- [ ] Implement weight quantization (INT8/INT4)
- [ ] Add KV cache memory management
- [ ] Profile memory usage per component
- [ ] Target: Run on 16GB Mac

### P3.1 - Performance Optimization
**Maximize throughput on Apple Silicon**

- [ ] Profile Metal GPU utilization
- [ ] Optimize attention kernels
- [ ] Benchmark vs PyTorch MPS
- [ ] Target: 10x+ real-time factor

### P3.2 - API Polish
**Clean public API**

- [ ] Add builder pattern for config
- [ ] Improve error messages
- [ ] Add logging/tracing
- [ ] Write API documentation

### P3.3 - Testing & CI
**Comprehensive test suite**

- [ ] Unit tests for each component
- [ ] Integration tests with sample audio
- [ ] Accuracy comparison with PyTorch
- [ ] GitHub Actions CI

---

## File Reuse Summary

| File | Source | Action |
|------|--------|--------|
| `sensevoice_encoder.rs` | funasr-nano-mlx | Copy unchanged |
| `audio.rs` | funasr-nano-mlx | Copy unchanged |
| `error.rs` | funasr-nano-mlx | Copy unchanged |
| `qwen.rs` | funasr-nano-mlx | Modify for 4B config |
| `adaptor.rs` | funasr-nano-mlx | Rewrite for 4-layer |
| `model.rs` | funasr-nano-mlx | Modify for 2560-dim |
| `lib.rs` | funasr-nano-mlx | Modify exports |

---

## Dependencies

```toml
[dependencies]
mlx-rs = "0.21"
mlx-rs-core = { path = "../mlx-rs-core" }
tokenizers = "0.21"
hound = "3.5"
rubato = "0.16"
rustfft = "6.2"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
safetensors = "0.4"
anyhow = "1.0"
```

---

## Timeline Estimate

| Phase | Tasks | Days |
|-------|-------|------|
| P0 | Foundation | 1-2 |
| P1 | Core functionality | 2-3 |
| P2.0-2.3 | Full features | 2-3 |
| P2.5 | API Integration (OminiX-API) | 1-2 |
| P3 | Optimization | 2+ |
| **Total** | | **8-12 days** |

---

## Success Criteria

1. **P0 Complete:** Adaptor loads and produces 2560-dim output
2. **P1 Complete:** Qwen3-4B generates text from audio embeddings
3. **P2 Complete:** Full transcription matches PyTorch quality
4. **P3 Complete:** Runs efficiently on M1 Pro 16GB

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Qwen4B too large for 16GB Mac | Implement INT8 quantization (P3.0) |
| Weight format incompatibility | Thorough conversion testing (P0.1) |
| Accuracy regression vs PyTorch | Side-by-side comparison tests |
| MLX API changes | Pin mlx-rs version |

---

## Known Issues

### Encoder Mismatch
**Issue:** Adaptor was trained with `iic/SenseVoiceSmall` (FunASR), but inference uses funasr-nano-mlx's SenseVoice encoder which has different weights.

**Impact:** Reduced transcription accuracy. Model recognizes key content but with substitutions:
- Expected: "开放时间：早上九点至下午五点"
- Actual: "上早八九点至时末五点日。"

**Solutions (choose one):**
1. **Convert SenseVoiceSmall weights** - Download from ModelScope, convert to safetensors, load in Rust
2. **Retrain adaptor** - Use funasr-nano-mlx's encoder embeddings for training
3. **Accept current accuracy** - Works for proof-of-concept, key content is recognized

### Training Format vs Inference
**Issue:** Training used simple `[audio_features, text_tokens]` concatenation without ChatML.

**Solution:** Inference now matches training format - no system/user/assistant markers.

### GELU vs ReLU
**Issue:** Initially implemented ReLU, but training used GELU activation.

**Status:** Fixed - now uses GELU to match training.
