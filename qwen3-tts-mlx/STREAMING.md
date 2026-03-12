# Streaming TTS: Architecture Notes & Future Work

## Current State (2026-03-12)

Streaming generation works — the autoregressive token loop yields codec frames in
chunks via `GenerationState::next_chunk()`. However, the **speech tokenizer decoder
is stateless**, which causes audio artifacts at chunk boundaries.

### The Problem

The decoder (`SpeechTokenizerDecoder::decode()`) contains:
- Conv1d layers (pre_conv, upsample_convs, initial_conv, final_conv)
- A transformer with sliding-window attention (pre_transformer)
- ConvNeXt blocks (upsample_convnext)
- SnakeBeta activations + decoder blocks

None of these maintain state between calls. When streaming calls
`decoder.decode(&chunk_frames)` per chunk, the convolutions at chunk boundaries
have no access to neighboring frames. This creates **audible discontinuities
(blips/pops)** at every chunk boundary.

### Current Workaround

The OminiX-API defaults to `response_format: "wav"`, which synthesizes all frames
first, decodes them in a single `decode()` call, and returns a complete WAV file.
The streaming HTTP path (`response_format: "pcm"`) also synthesizes fully before
sending PCM in 32KB chunks over chunked transfer encoding ("pseudo-streaming").

This eliminates artifacts but loses time-to-first-audio advantage.

## Eval Frequency Constraint (8-bit Quantized Models)

`mlx_rs::transforms::eval()` must be called **every step** in the autoregressive
generation loop, not batched (e.g., every 4 steps). With 8-bit quantized weights,
deferring eval accumulates precision errors in the lazy computation graph. By the
time eval finally runs, the compounded quantization drift produces wrong codec
tokens → noise and distortion in the decoded audio.

This applies to:
- **Main generation loop** (`generate.rs`): `eval([&logits])` after each `forward_step`
- **Code predictor** (`talker.rs`): `eval()` after each layer pass AND after each
  `lm_heads[g]` logit computation, for all 15 codebook predictions per timestep

The code predictor is especially sensitive because its 15 sequential predictions
each depend on the previous — errors compound across all codebooks.

Float32 models may tolerate batched eval, but 8-bit models cannot. Do not remove
these eval calls for performance without testing audio quality end-to-end.

## True Streaming: Implementation Plan

To support real streaming (hear audio while generation continues), the decoder
needs to become **stateful** with overlap-add crossfading.

### Approach: Overlap-Add Decoding

1. **Buffer with overlap**: Instead of decoding `[frame_0..frame_N]` then
   `[frame_N..frame_2N]`, decode `[frame_0..frame_N+K]` then
   `[frame_N-K..frame_2N+K]` where `K` is the overlap size.

2. **Crossfade at boundaries**: For the overlapping region (2K frames), apply a
   fade-out to the tail of chunk 1 and fade-in to the head of chunk 2, then sum.
   A raised-cosine (Hann) window works well:
   ```
   window[i] = 0.5 * (1 - cos(pi * i / K))
   chunk1_tail[i] *= (1 - window[i])
   chunk2_head[i] *= window[i]
   output[i] = chunk1_tail[i] + chunk2_head[i]
   ```

3. **Overlap size**: The decoder has 4 upsample stages (each 2x or 4x) and conv
   kernels. Minimum overlap should cover the receptive field of the deepest
   convolution. Start with K=4 frames (~333ms at 12Hz) and test.

### Required Changes

- `SpeechTokenizerDecoder`: Add `decode_streaming()` that takes a chunk + overlap
  context and returns audio with crossfade applied
- `StreamingSession`: Buffer `K` extra frames from the previous chunk
- `Synthesizer::start_streaming()`: Accept overlap parameter
- OminiX-API `tts_pool.rs`: Use streaming decode path, send PCM chunks as they're
  produced

### Alternative: Stateful Convolutions

Instead of overlap-add, make the conv layers stateful by carrying their internal
padding buffers between calls. This is more complex but avoids redundant computation:

- Each Conv1d stores its left-padding state (last `kernel_size - 1` inputs)
- The transformer uses a persistent KV cache (already exists for attention)
- ConvNeXt blocks carry their depthwise conv state

This is what audio codecs (Opus, etc.) do internally but requires modifying every
conv layer in the decoder.

### Recommendation

Start with overlap-add — it's simpler, doesn't require modifying the decoder
internals, and the redundant computation (re-decoding K frames) is negligible
compared to the autoregressive generation time.

## Text Chunking for Long Input

For text exceeding ~500 words / ~800 Chinese characters, the API layer should
chunk input at sentence boundaries and make multiple synthesis calls:

1. Split text on sentence-ending punctuation (。！？.!?)
2. Synthesize each sentence independently (fresh KV cache per call)
3. Concatenate PCM output
4. Optionally add 50-100ms silence between sentences for natural pacing

This avoids KV cache memory pressure and keeps generation time per-call reasonable.
Prosody discontinuity between sentences is minimal since sentence boundaries are
natural pause points.

### Limits

| Constraint | Value | Notes |
|---|---|---|
| max_new_tokens | 8192 frames | ~683s audio at 12Hz, hard cap in config |
| Practical per-call | ~500 words EN / ~800 chars ZH | Memory-limited on 16GB Macs |
| Input text | No validation | Silently truncates at max_new_tokens |
