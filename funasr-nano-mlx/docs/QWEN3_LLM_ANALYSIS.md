# Qwen3-0.6B LLM Analysis for funasr-nano

**Date:** 2026-01-30
**Status:** Research Complete

---

## Executive Summary

The Qwen3-0.6B LLM in funasr-nano is the **base pre-trained model**, not the instruction-tuned version. This explains why it cannot follow prompts for translation or other tasks. The official `Qwen/Qwen3-0.6B` from Alibaba IS instruction-tuned and could potentially enable translation capability.

---

## Model Variants Inventory

### Official Qwen3-0.6B Family

| Model | HuggingFace ID | Type | Params | Purpose |
|-------|----------------|------|--------|---------|
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | Instruct | 0.6B | General instruction-following, chat |
| Qwen3-0.6B-Base | `Qwen/Qwen3-0.6B-Base` | Base | 0.6B | Pre-trained LM, no instruction tuning |
| Qwen3-0.6B-MLX-bf16 | `Qwen/Qwen3-0.6B-MLX-bf16` | Instruct | 0.6B | MLX-optimized for Apple Silicon |
| Qwen3-ASR-0.6B | `Qwen/Qwen3-ASR-0.6B` | ASR | 0.9B | Speech recognition (Qwen3-Omni based) |
| Qwen3-Embedding-0.6B | `Qwen/Qwen3-Embedding-0.6B` | Embedding | 0.6B | Text embeddings |

### Architecture Comparison

```
funasr-nano LLM config:
{
  "architectures": ["Qwen3ForCausalLM"],
  "hidden_size": 1024,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "intermediate_size": 3072,
  "vocab_size": 151936,
  "max_position_embeddings": 40960
}
```

**Key Finding:** The architecture is IDENTICAL between Base and Instruct versions. Only the weights differ.

---

## funasr-nano LLM Configuration

From `~/.dora/models/funasr-nano/config.yaml`:

```yaml
llm: Qwen3-0.6b
llm_conf:
  hub: hf
  freeze: true           # <-- LLM weights FROZEN during training
  llm_dtype: bf16
  init_param_path: Qwen3-0.6B
  use_lora: false
```

### What Was Trained

| Component | Frozen? | Notes |
|-----------|---------|-------|
| SenseVoice Encoder | Yes | Pre-trained audio encoder |
| Audio Adaptor | Yes* | Projects audio → LLM dimension |
| Qwen3-0.6B LLM | Yes | Original weights unchanged |
| CTC Decoder | No | Only this was trained |

*The adaptor was trained to project audio embeddings that the frozen LLM interprets as Chinese text.

### Why It Can't Follow Instructions

1. **Base Model:** Qwen3-0.6B-Base has no instruction-following capability
2. **Single Task Training:** Only trained on `Audio → Chinese Transcription`
3. **Fixed Prompt:** Always uses "语音转写成中文：" regardless of user prompt
4. **Audio Dominance:** Audio embeddings override any text instructions

---

## Empirical Test Results

### Test 1: Translation Capability

```
Input: "今天天气很好，我们去公园散步吧。"
Prompt: "请将以下中文翻译为英文：{input}"
Output: "!!!!!!!!!!"
Result: FAILED - Model outputs garbage
```

### Test 2: Custom Audio Prompt

```
Audio: Chinese speech
Prompt: "Transcribe and translate to English:"
Output: "开放时间：早上九点至下午五点。" (Chinese)
Result: FAILED - Ignores prompt, outputs Chinese
```

### Test 3: Standard ASR

```
Audio: Chinese speech
Prompt: "语音转写成中文："
Output: "开放时间：早上九点至下午五点。"
Result: SUCCESS - ASR works correctly
```

**Conclusion:** The model is hardwired for Chinese ASR only.

---

## Special Token Mapping

funasr-nano repurposes existing Qwen3 special tokens:

| Token ID | Original Qwen3 Purpose | funasr-nano Usage |
|----------|------------------------|-------------------|
| 151643 | `<\|endoftext\|>` | EOS token |
| 151644 | `<\|im_start\|>` | ChatML start |
| 151645 | `<\|im_end\|>` | ChatML end / EOS |
| 151646 | `<\|object_ref_start\|>` | Start of speech |
| 151647 | `<\|object_ref_end\|>` | End of speech |

---

## Qwen3-ASR-0.6B (NOT Compatible)

The official `Qwen/Qwen3-ASR-0.6B` is a **different model**:

- Built on Qwen3-Omni foundation (not Qwen3-0.6B)
- 0.9B parameters (not 0.6B)
- Different architecture and pipeline
- Supports 52 languages natively
- Has its own `qwen-asr` Python package
- **Cannot be swapped into funasr-nano**

---

## Fine-tuning Requirements

### Hardware Requirements (LoRA + 4-bit Quantization)

| Model Size | VRAM Required | Apple Silicon |
|------------|---------------|---------------|
| Qwen3-0.6B | 4-6 GB | M1/M2 8GB works |
| Qwen3-1.7B | 8-12 GB | M1/M2 16GB+ |

### Training Frameworks

1. **Unsloth** - 2x faster, 70% less VRAM
2. **ms-swift** - Official Alibaba framework
3. **transformers + PEFT** - Standard approach

### Recommended LoRA Configuration

```python
lora_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "learning_rate": 5e-5,
    "epochs": 1-2,
    "batch_size": 4-8,
}
```

### Dataset Requirements

| Task | Dataset Size | Source |
|------|--------------|--------|
| Instruction Tuning | 10k-50k | Alpaca, Dolly, etc. |
| Chinese→English Translation | 10k-100k | WMT, OPUS, etc. |
| Mixed (Reasoning + Direct) | 75% / 25% | Recommended for Qwen3 |

---

## References

- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) - Official Instruct model
- [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) - Base model
- [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) - Dedicated ASR model
- [Unsloth Qwen3 Guide](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune) - Fine-tuning guide
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) - arXiv paper

---

*Document created: 2026-01-30*
