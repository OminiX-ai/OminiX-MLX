# Fun-ASR-Nano MLX Development Plan

## Overview

**Goal:** Port Fun-ASR-Nano-2512 (800M) to Rust/MLX for Apple Silicon inference.

**Architecture:** Whisper Encoder → Audio Adaptor → Qwen LLM

**Current Status:** Project scaffolding complete (~1,400 lines), needs weight loading and validation.

---

## Phase 1: Architecture Discovery (1-2 days)

### 1.1 Inspect Fun-ASR Model Weights

**Objective:** Understand exact architecture from weight shapes.

**Tasks:**
- [ ] Download Fun-ASR-Nano-2512 from HuggingFace/ModelScope
- [ ] Inspect `config.json` for all configuration parameters
- [ ] List all weight keys and shapes from safetensors
- [ ] Document encoder architecture (layers, dims, heads)
- [ ] Document LLM architecture (Qwen variant, layers, dims)
- [ ] Identify adaptor weight structure

**Commands:**
```bash
# Download model
huggingface-cli download FunAudioLLM/Fun-ASR-Nano-2512

# Or from ModelScope
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('iic/Fun-ASR-Nano-2512')"
```

**Analysis script:**
```python
import safetensors
from safetensors import safe_open

with safe_open("model.safetensors", framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(f"{key}: {tensor.shape} {tensor.dtype}")
```

### 1.2 Verify Architecture Assumptions

**Questions to answer:**
- Is the encoder actually Whisper? What variant (tiny/base/small/medium/large)?
- What Qwen variant is used? (Qwen1.5? Qwen2? dimensions?)
- Is the adaptor a single linear layer or more complex?
- What are the special tokens for speech markers?

---

## Phase 2: Weight Loading (2-3 days)

### 2.1 Implement Safetensors Loader

**File:** `src/model.rs` - `load_weights()`

**Tasks:**
- [ ] Parse `model.safetensors.index.json`
- [ ] Load weights by layer prefix
- [ ] Handle sharded weight files
- [ ] Map weight names to model parameters

**Weight mapping pattern:**
```rust
// Expected patterns (to be verified):
// encoder.conv1.weight
// encoder.conv2.weight
// encoder.layers.{i}.self_attn.{q,k,v,out}_proj.weight
// encoder.layers.{i}.mlp.fc{1,2}.weight
// encoder.ln_post.weight

// adaptor.projection.weight
// adaptor.projection.bias

// llm.embed_tokens.weight
// llm.layers.{i}.self_attn.{q,k,v,o}_proj.weight
// llm.layers.{i}.mlp.{gate,up,down}_proj.weight
// llm.norm.weight
// llm.lm_head.weight
```

### 2.2 Update Configuration Structs

**Tasks:**
- [ ] Update `WhisperEncoderConfig` with actual values
- [ ] Update `QwenConfig` with actual values
- [ ] Add adaptor config if needed
- [ ] Load config from `config.json`

### 2.3 Test Weight Loading

**Validation:**
- [ ] All weights loaded without error
- [ ] Weight shapes match model parameters
- [ ] No missing or extra weights

---

## Phase 3: Forward Pass Validation (3-4 days)

### 3.1 Encoder Validation

**Tasks:**
- [ ] Compare encoder output with Python reference
- [ ] Test with same input mel spectrogram
- [ ] Validate layer-by-layer if mismatch

**Test approach:**
```python
# Python reference
import torch
from funasr import AutoModel

model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512")
# Extract encoder output for comparison
```

```rust
// Rust comparison
let mel = load_test_mel()?;
let encoder_out = model.encoder.forward(&mel)?;
// Compare with Python output
```

### 3.2 Adaptor Validation

**Tasks:**
- [ ] Verify adaptor output dimensions
- [ ] Compare with Python reference

### 3.3 LLM Validation

**Tasks:**
- [ ] Test token embedding
- [ ] Test single layer forward pass
- [ ] Test full model logits output
- [ ] Validate KV cache behavior

### 3.4 End-to-End Validation

**Tasks:**
- [ ] Compare final logits with Python
- [ ] Verify sampling produces same tokens (with temp=0)
- [ ] Test full transcription pipeline

---

## Phase 4: Generation Pipeline (2-3 days)

### 4.1 Multimodal Input Handling

**Current gap:** Need to properly inject audio features into LLM.

**Tasks:**
- [ ] Identify speech marker tokens from tokenizer
- [ ] Implement `inject_audio_features()` function
- [ ] Handle variable-length audio sequences

**Implementation:**
```rust
fn inject_audio_features(
    text_embeds: &Array,      // [B, L_text, dim]
    audio_features: &Array,   // [B, L_audio, dim]
    start_pos: usize,         // Position of <|startofspeech|>
    end_pos: usize,           // Position of <|endofspeech|>
) -> Result<Array> {
    // Replace markers with audio features
    // text_embeds[start_pos:end_pos] = audio_features
}
```

### 4.2 Prompt Template

**Tasks:**
- [ ] Implement ChatML formatting
- [ ] Add system prompt for ASR task
- [ ] Handle language parameter (zh/en/ja)

**Template:**
```
<|im_start|>system
You are a speech recognition assistant.<|im_end|>
<|im_start|>user
<|startofspeech|>[audio]<|endofspeech|><|im_end|>
<|im_start|>assistant
```

### 4.3 Streaming Generation

**Tasks:**
- [ ] Implement token-by-token generation
- [ ] Add async prefetching (from qwen3-mlx)
- [ ] Implement streaming state management

---

## Phase 5: Optimization (2-3 days)

### 5.1 Quantization Support

**Tasks:**
- [ ] Add 4-bit quantization loading
- [ ] Implement `QuantizedLinear` for all layers
- [ ] Test accuracy vs full precision

### 5.2 Performance Optimization

**Tasks:**
- [ ] Profile inference bottlenecks
- [ ] Optimize KV cache allocation
- [ ] Add fused SwiGLU kernel (from mlx-rs-core)
- [ ] Benchmark against Python reference

**Target performance:**
| Metric | Target |
|--------|--------|
| Decode speed | 30-50 tok/s |
| Memory (4-bit) | < 3 GB |
| RTF | < 0.1 |

### 5.3 Batch Processing

**Tasks:**
- [ ] Support batch inference (multiple audio files)
- [ ] Implement dynamic batching

---

## Phase 6: Polish & Documentation (1-2 days)

### 6.1 Error Handling

**Tasks:**
- [ ] Add comprehensive error messages
- [ ] Handle edge cases (empty audio, very long audio)
- [ ] Add input validation

### 6.2 Examples & Documentation

**Tasks:**
- [ ] Update README with usage examples
- [ ] Add API documentation
- [ ] Create benchmark comparison with Python

### 6.3 Testing

**Tasks:**
- [ ] Add unit tests for each module
- [ ] Add integration tests
- [ ] Add regression tests with reference outputs

---

## Timeline Summary

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Architecture Discovery | 1-2 days | Not started |
| Phase 2: Weight Loading | 2-3 days | Stub implemented |
| Phase 3: Forward Pass Validation | 3-4 days | Not started |
| Phase 4: Generation Pipeline | 2-3 days | Basic structure |
| Phase 5: Optimization | 2-3 days | Not started |
| Phase 6: Polish | 1-2 days | Not started |
| **Total** | **11-17 days** | |

---

## Dependencies

### External
- Fun-ASR-Nano-2512 weights (HuggingFace/ModelScope)
- Python Fun-ASR for reference comparisons

### Internal (OminiX-MLX)
- `mlx-rs` - MLX Rust bindings
- `mlx-rs-core` - Shared LLM infrastructure
- `funasr-mlx` - Audio utilities (optional reuse)

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Unknown encoder architecture | Medium | High | Inspect weights, compare with Whisper variants |
| Weight format incompatibility | Low | Medium | Write custom loader if needed |
| Performance gap vs Python | Medium | Medium | Profile and optimize hot paths |
| Qwen version mismatch | Low | Low | Verify config against qwen3-mlx |

---

## Success Criteria

1. **Functional:** Transcribes audio with same output as Python reference
2. **Performance:** Achieves < 0.1 RTF on M1/M2/M3 Mac
3. **Memory:** Runs in < 4 GB with quantization
4. **Quality:** WER within 1% of Python reference

---

## Next Immediate Steps

1. **Download Fun-ASR-Nano-2512 model**
2. **Inspect config.json and weight keys**
3. **Update architecture configs based on findings**
4. **Implement weight loading**
5. **Validate encoder forward pass**
