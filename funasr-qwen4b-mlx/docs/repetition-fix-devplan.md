# Dev Plan: Fix ASR Repetition

## Problem

~10% of audio chunks trigger repetition loops where the model generates the same phrase until max tokens. Current mitigation (n-gram detection + truncation) is reactive — it catches loops after 2-3 repetitions and strips them. This wastes inference time and produces truncated output.

Root causes identified through deep research:

1. **Greedy decoding (temperature=0.0)** — Qwen3 team explicitly warns this causes "endless repetitions"
2. **Training/inference format mismatch** — Phase 2 training used raw `[audio|text]` concat, inference wraps in full ChatML with `<think>` tags
3. **Wrong speech token IDs** — 151646/151647 are `<|object_ref_start|>`/`<|object_ref_end|>`, not speech tokens
4. **No penalty mechanism** — nothing discourages repeating previously-generated tokens
5. **Single-token detection in model.rs** — core generate function only catches 10 identical tokens in a row

## Files to Modify

| File | Changes |
|------|---------|
| `src/model.rs` | Add sampling (top-k, top-p), presence penalty, entropy monitoring, upgrade repetition detection |
| `examples/test_real_audio.rs` | Wire new sampling params, add embedding collapse detection, update constants |

## Implementation

### Step 1: Add top-k and top-p sampling to `model.rs`

Current `sample()` function (referenced at lines 333, 398 in test_real_audio.rs; lines 381, 415 in model.rs) only does argmax when temperature=0.0.

Add to `src/model.rs`:

```rust
fn sample_with_params(logits: &Array, temperature: f32, top_k: usize, top_p: f32) -> Array {
    if temperature == 0.0 {
        // greedy fallback
        return argmax(logits, -1);
    }
    let scaled = logits / temperature;
    // top-k: zero out everything below the k-th largest
    // top-p: sort by probability, cumsum, mask below threshold
    // then sample from the filtered distribution
}
```

Parameters to use (from Qwen3-4B generation_config.json):
- `temperature = 0.6`
- `top_k = 20`
- `top_p = 0.95`

### Step 2: Add presence penalty to `model.rs`

Before sampling each token, subtract a flat penalty from logits of any token already generated in the current sequence. This discourages repetition without over-penalizing naturally repeated words (unlike frequency_penalty which increases with count).

```rust
fn apply_presence_penalty(logits: &Array, generated_tokens: &[i32], penalty: f32) -> Array {
    // For each unique token in generated_tokens, subtract penalty from that logit index
    // penalty = 1.0-1.2 for ASR (conservative to preserve real repeated words)
}
```

Apply at line ~330 in test_real_audio.rs (before `sample()`) and line ~380 in model.rs.

### Step 3: Fix speech token IDs in `test_real_audio.rs`

Current (lines 204-205):
```rust
let speech_start: i32 = 151646;  // Actually <|object_ref_start|>
let speech_end: i32 = 151647;    // Actually <|object_ref_end|>
```

Options:
- **Option A**: Remove speech markers entirely — just inject audio features between user prefix and suffix with no special boundary tokens. This is closest to the training format where audio and text were directly concatenated.
- **Option B**: Use `<|vision_start|>` (151652) / `<|vision_end|>` (151653) — these are the closest multimodal tokens available. However the model wasn't trained with these either.

**Choose Option A** — removing the markers is the least disruptive change and reduces the train/inference gap.

### Step 4: Upgrade repetition detection in `model.rs`

Current detection in `generate_from_embeddings` (lines 395-402) only checks if the last 10 tokens are all identical. Replace with the proven n-gram detection already working in `test_real_audio.rs` (lines 348-381):

```rust
// In generate_from_embeddings loop:
if tokens.len() >= 4 {
    let max_n = 64.min(tokens.len() / 2);
    for n in 1..=max_n {
        let reps_needed = if n <= 2 { 3 } else { 2 };
        // check last reps_needed * n tokens for repeated pattern
        // if found, truncate to 1 copy and break
    }
}
```

This makes the core model robust regardless of which example/caller invokes it.

### Step 5: Add entropy monitoring to `model.rs`

After each forward pass, compute Shannon entropy of the softmax distribution. If entropy stays below a threshold for consecutive steps, the model is stuck in a degenerate state — stop early.

```rust
fn compute_entropy(logits: &Array) -> f32 {
    let probs = softmax(logits, -1);
    let log_probs = log(probs);
    -sum(probs * log_probs)  // Shannon entropy
}

// In generation loop:
if entropy < 0.5 for 5+ consecutive steps {
    break;  // Model is stuck, stop generating
}
```

This catches repetition patterns that are too long or complex for n-gram detection.

### Step 6: Add embedding collapse detection in `test_real_audio.rs`

After the adaptor projects audio features, check if the output embeddings have collapsed (all audio tokens map to nearly the same vector). If so, skip the chunk — feeding degenerate embeddings to the LLM produces garbage.

```rust
fn check_embedding_collapse(embeddings: &Array, threshold: f32) -> bool {
    // Compute mean pairwise cosine similarity of audio token embeddings
    // If > threshold (e.g., 0.95), embeddings have collapsed
    // Return true = collapsed, should skip this chunk
}
```

Insert after adaptor forward pass in `transcribe_chunk()` (~line 310).

### Step 7: Update constants in `test_real_audio.rs`

```rust
const TEMPERATURE: f32 = 0.6;       // Was 0.0 (greedy)
const TOP_K: usize = 20;            // New
const TOP_P: f32 = 0.95;            // New
const PRESENCE_PENALTY: f32 = 1.0;  // New (conservative for ASR)
const MAX_ASR_TOKENS: usize = 100;  // Keep as-is
```

## Verification

```bash
# Full pipeline test on real audio
cargo run --example test_real_audio --release -- /tmp/rust_talk.wav 10 2>/dev/null
```

Check:
- Zero or near-zero chunks hitting max tokens (was ~3 with current n-gram stripping, ~17 without)
- No meta-commentary in output
- Legitimate repeated words preserved (e.g., "好好好" should stay if spoken)
- RTF comparable or better (sampling adds minimal overhead vs greedy)
- Total Chinese chars in reasonable range (7000-9000 for this audio)

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Sampling (temp=0.6) | Could introduce randomness/noise in clean chunks | top-k=20 limits vocabulary; top-p=0.95 filters tail; can tune down to 0.3 if needed |
| Presence penalty | Could suppress real repeated words | Use conservative 1.0 (not Qwen's default 1.5); penalty is flat, not cumulative |
| Remove speech markers | Could change output quality since training Python script used these markers | Check: training script train_phase2_asr.py does NOT use speech markers — raw concat only |
| Entropy early stop | Could truncate valid slow-but-correct generation | Threshold 0.5 with 5-step window is conservative; most normal generation has entropy > 2.0 |

## Implementation Order

1. Steps 1+2+7 together (sampling + penalty + constants) — biggest impact
2. Step 4 (upgrade model.rs detection) — defense in depth
3. Step 3 (fix token IDs) — correctness
4. Steps 5+6 (entropy + collapse detection) — polish

## References

- [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B) — "DO NOT use greedy decoding"
- [LZ Penalty (arxiv 2504.20131)](https://arxiv.org/abs/2504.20131) — information-theoretic repetition prevention
- [ERGO (arxiv 2510.14077)](https://arxiv.org/html/2510.14077v1) — entropy-guided generation optimization
- [Calm-Whisper (arxiv 2505.12969)](https://arxiv.org/html/2505.12969v1) — hallucination reduction in audio models
- [Qwen3-VL repetition issue #1611](https://github.com/QwenLM/Qwen3-VL/issues/1611) — presence_penalty recommendation
