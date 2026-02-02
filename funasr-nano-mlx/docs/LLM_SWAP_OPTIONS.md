# FunASR-Nano: LLM Swap Options Analysis

## Can You Swap in Other LLMs?

**Yes, absolutely.** The architecture is designed as a modular pipeline:

```
Audio → Encoder → Adaptor → [ANY LLM] → Text
```

The only requirement is that the **adaptor output dimension must match the LLM's hidden_size**.

## Available LLM Options

### Tier 1: Small Models (1-3B) - Best for Edge/Mobile

| Model | Params | hidden_size | Context | Translation | Chinese |
|-------|--------|-------------|---------|-------------|---------|
| **Qwen3-0.6B** (current) | 0.6B | 1024 | 32K | Poor | Excellent |
| **Qwen3-1.7B** | 1.7B | 2048 | 40K | Good | Excellent |
| **Llama-3.2-1B** | 1.2B | 2048 | 128K | Good | Fair |
| **Llama-3.2-3B** | 3.2B | 3072 | 128K | Good | Fair |
| **Gemma-2-2B** | 2.6B | 2304 | 8K | Good | Fair |
| **DeepSeek-Coder-1.3B** | 1.3B | 2048 | 16K | N/A | Good |
| **DeepSeek-V2-Lite** | 15.7B | 2048 | 32K | Good | Excellent |

### Tier 2: Medium Models (4-9B) - Best Balance

| Model | Params | hidden_size | Context | Translation | Chinese |
|-------|--------|-------------|---------|-------------|---------|
| **Qwen3-4B** | 4B | 2560 | 40K | Very Good | Excellent |
| **Qwen3-8B** | 8B | 4096 | 128K | Excellent | Excellent |
| **Gemma-2-9B** | 9B | 3584 | 8K | Very Good | Good |
| **Llama-3.1-8B** | 8B | 4096 | 128K | Excellent | Good |
| **DeepSeek-7B** | 7B | 4096 | 4K | Good | Excellent |

### Tier 3: Large Models (10B+) - Server Only

| Model | Params | hidden_size | Notes |
|-------|--------|-------------|-------|
| DeepSeek-V3 | 671B (37B active) | 7168 | MoE, requires 8×H200 |
| Qwen3-14B | 14B | 5120 | Good for translation |
| Qwen3-32B | 32B | 5120 | Best quality |

## Key Considerations for Each Model Family

### Qwen3 (Recommended for Chinese + Translation)

**Pros:**
- Same tokenizer family as current model
- Excellent Chinese understanding
- Good translation capability (Instruct versions)
- Well-documented architecture

**Cons:**
- Proprietary license for some versions

**Best Choice:** Qwen3-1.7B or Qwen3-4B

### Llama 3.2 (Good for Multilingual)

**Pros:**
- Open weights (Meta license)
- 128K context window even on 1B model
- Strong English capability
- Well-optimized for edge devices

**Cons:**
- Chinese capability weaker than Qwen
- Different tokenizer (needs remapping)

**Best Choice:** Llama-3.2-3B (3072 hidden_size, good balance)

### Gemma 2 (Google Quality)

**Pros:**
- Strong reasoning capability
- Efficient GQA architecture
- Good multilingual support

**Cons:**
- Shorter context (8K)
- Unusual hidden sizes (2304, 3584)
- Different tokenizer

**Best Choice:** Gemma-2-2B for edge, Gemma-2-9B for quality

### DeepSeek (Best for Code + Chinese)

**Pros:**
- Excellent Chinese
- Strong reasoning (R1 series)
- DeepSeek-V2-Lite is efficient MoE

**Cons:**
- V3 too large for consumer hardware
- Older models (7B) have short context

**Best Choice:** DeepSeek-V2-Lite (MoE, 2048 hidden)

## Adaptor Modification Requirements

For each LLM, you need to modify:

```rust
// src/adaptor.rs
pub struct AdaptorConfig {
    pub encoder_dim: i32,  // Keep: 512 (from SenseVoice)
    pub ffn_dim: i32,      // Keep: 2048
    pub llm_dim: i32,      // CHANGE: match target LLM hidden_size
    pub n_layer: i32,      // May adjust: 2-4
}
```

### Adaptor Changes by Target LLM

| Target LLM | Current llm_dim | New llm_dim | Change |
|------------|-----------------|-------------|--------|
| Qwen3-0.6B | 1024 | 1024 | None |
| Qwen3-1.7B | 1024 | 2048 | +100% |
| Llama-3.2-1B | 1024 | 2048 | +100% |
| Gemma-2-2B | 1024 | 2304 | +125% |
| Qwen3-4B | 1024 | 2560 | +150% |
| Llama-3.2-3B | 1024 | 3072 | +200% |
| Gemma-2-9B | 1024 | 3584 | +250% |
| Qwen3-8B | 1024 | 4096 | +300% |

## Tokenizer Compatibility

### Same Tokenizer Family (Easy Swap)
- Qwen3-0.6B → Qwen3-1.7B → Qwen3-4B → Qwen3-8B
- Only need to verify special token IDs

### Different Tokenizer (More Work)
- Qwen → Llama: Need token ID remapping
- Qwen → Gemma: Need token ID remapping
- Qwen → DeepSeek: Partially compatible

**Special Tokens to Handle:**
```
<|startofspeech|>  → May need custom token
<|endofspeech|>    → May need custom token
<|im_start|>       → Llama uses [INST], Gemma uses <start_of_turn>
<|im_end|>         → Varies by model
```

## Recommended Migration Paths

### Path A: Stay in Qwen Family (Easiest)
```
Qwen3-0.6B → Qwen3-1.7B → Qwen3-4B
```
- Same tokenizer
- Predictable architecture
- Best Chinese support

### Path B: Maximize Multilingual
```
Qwen3-0.6B → Llama-3.2-3B
```
- 128K context
- Better English
- Requires tokenizer work

### Path C: Maximize Efficiency (MoE)
```
Qwen3-0.6B → DeepSeek-V2-Lite
```
- Only 2.4B active params
- 2048 hidden_size (same as Qwen3-1.7B)
- Excellent Chinese

## Implementation Checklist

### For Any LLM Swap:

1. **Architecture Support**
   ```rust
   // Add new model config in src/qwen.rs (or new file)
   pub struct LlamaConfig { ... }
   pub struct GemmaConfig { ... }
   ```

2. **Weight Loading**
   ```rust
   // Update weight mapping for new model
   fn map_safetensors_key_llama(key: &str) -> String { ... }
   ```

3. **Adaptor Output Dimension**
   ```rust
   // Modify adaptor.rs
   llm_dim: NEW_HIDDEN_SIZE,
   ```

4. **Tokenizer Integration**
   ```rust
   // Handle different chat templates
   fn build_prompt_llama(...) { ... }
   fn build_prompt_gemma(...) { ... }
   ```

5. **Special Token Mapping**
   ```rust
   // Map speech tokens to new vocabulary
   pub struct SpeechMarkers {
       start_token: i32,  // Find equivalent or add custom
       end_token: i32,
   }
   ```

6. **Adaptor Retraining**
   - Train new adaptor to project audio→LLM embedding space
   - ~10K-100K audio-text pairs
   - 2-5 days on M-series Mac

## Quick Comparison: Which LLM to Choose?

| Priority | Best Choice | Why |
|----------|-------------|-----|
| **Chinese ASR + Translation** | Qwen3-1.7B/4B | Best Chinese, same tokenizer |
| **Multilingual** | Llama-3.2-3B | 128K context, good all-around |
| **Edge/Mobile** | Llama-3.2-1B | Smallest with good quality |
| **Quality/Server** | Qwen3-8B | Best translation quality |
| **Efficiency** | DeepSeek-V2-Lite | MoE, only 2.4B active |

## Conclusion

Swapping LLMs is **definitely possible** and the architecture supports it well. The key work is:

1. **Adaptor retraining** (~80% of effort)
2. **Tokenizer adaptation** (~15% of effort)
3. **Code changes** (~5% of effort)

**Recommendation:** Start with **Qwen3-1.7B** (same family, 2x hidden_size) to validate the pipeline, then consider Llama/DeepSeek if you need specific capabilities.

## Sources

- [DeepSeek-V3 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- [DeepSeek Architecture](https://mccormickml.com/2025/02/12/the-inner-workings-of-deep-seek-v3/)
- [Llama 3.2 Model Card](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/)
- [Llama 3.2 1B Specs](https://apxml.com/models/llama-3-2-1b)
- [Gemma 2 Architecture](https://developers.googleblog.com/en/gemma-explained-overview-gemma-model-family-architectures/)
- [Gemma 2 HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/gemma2)
- [DeepSeek LLM GitHub](https://github.com/deepseek-ai/DeepSeek-LLM)
