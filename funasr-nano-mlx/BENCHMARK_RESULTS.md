# Benchmark Results: funasr-nano-mlx vs funasr-mlx

**Test Audio**: 41.26 seconds Chinese speech (/tmp/voice_memo.wav)
**Hardware**: Apple Silicon (M-series)
**Date**: 2025-01-25

## Results Summary

| Model | Architecture | Latency (Mean) | RTF | Real-Time Factor |
|-------|-------------|----------------|-----|------------------|
| **funasr-mlx** (Paraformer) | Non-autoregressive | 734 ms | 0.018x | **56x real-time** |
| **funasr-nano-mlx** (Fun-ASR-Nano) | Autoregressive LLM | 13,604 ms | 0.33x | **3x real-time** |

## Architectural Comparison

### Paraformer (funasr-mlx)
- **Encoder**: SAN-M encoder (~200M params)
- **Decoder**: CIF (Continuous Integrate-and-Fire) + parallel predictor
- **Decoding**: Non-autoregressive - all 151 tokens in single forward pass
- **Strengths**: Very fast, low latency
- **Limitations**: Fixed vocabulary, less flexible for complex tasks

### Fun-ASR-Nano (funasr-nano-mlx)
- **Encoder**: SenseVoice (70-layer SAN-M, ~400M params)
- **Adaptor**: 2-layer transformer projecting to LLM dimension
- **Decoder**: Qwen3-0.6B LLM (28 layers, ~600M params)
- **Decoding**: Autoregressive - each token requires full LLM forward pass
- **Total**: ~985M parameters
- **Strengths**: LLM-based, can be extended for translation, summarization, etc.
- **Limitations**: Slower due to sequential token generation

## Performance Breakdown (Fun-ASR-Nano)

| Stage | Time |
|-------|------|
| Audio loading + preprocessing | 77 ms |
| SenseVoice encoding | 547 ms |
| LLM generation (119 chars) | 10,945 ms |
| **Per-token latency** | **92 ms** |

## Analysis

The 18x speed difference is expected due to fundamental architectural differences:

1. **Paraformer** uses CIF for acoustic boundary detection and parallel token prediction. The decoder runs once regardless of output length.

2. **Fun-ASR-Nano** uses an autoregressive LLM which must run 28 transformer layers for each output token sequentially.

For a 41-second audio:
- Paraformer: 1 forward pass → 151 tokens in 734ms
- Fun-ASR-Nano: 119 forward passes → 119 tokens in 10.9s

The 92ms/token is competitive for a 600M LLM on Apple Silicon (typical: 10-20 tokens/sec = 50-100ms/token).

## Optimization Opportunities

| Optimization | Potential Speedup | Effort |
|--------------|------------------|--------|
| Speculative decoding | 2-3x | High |
| INT8 quantization | 1.5-2x | Medium |
| INT4 quantization | 2-3x | Medium |
| Batched inference | N/A for single stream | - |

## Conclusion

Both models are correctly implemented. The speed difference reflects their design tradeoffs:

- **Use Paraformer** when speed is critical (real-time transcription)
- **Use Fun-ASR-Nano** when LLM capabilities are needed (translation, summarization, instruction following) and 3x real-time is acceptable
