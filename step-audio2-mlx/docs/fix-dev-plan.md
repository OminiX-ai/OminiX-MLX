# step-audio2-mlx Fix Dev Plan

Based on critical code review. Ordered by impact and dependency.

---

## Phase 1: Generation Quality (User-Visible)

### 1.1 Add repetition penalty to generation loop
**Files:** `src/model.rs` (generate_text, generate_audio_tokens, generate_with_audio)
**Problem:** Model gets stuck repeating same sentence indefinitely.
**Fix:**
- Add frequency penalty: track token counts, penalize logits of recently generated tokens
- Add n-gram blocking: prevent repeating any 3-gram
- Add to all three generation methods (generate_text, generate_audio_tokens, generate_with_audio)
- Default: frequency_penalty=1.2, no_repeat_ngram_size=3

### 1.2 Fix max_tokens inconsistency
**Files:** `src/model.rs:510, 527`
**Problem:** `transcribe()` and `transcribe_samples()` use max_tokens=512, too low for dense speech.
**Fix:** Change both to 2048 to match `transcribe_long()`.

### 1.3 Make `transcribe()` use chunked processing by default
**Files:** `src/model.rs`
**Problem:** `transcribe()` silently truncates audio >15s.
**Fix:** Have `transcribe()` call `transcribe_long()` internally, so all public transcription methods handle long audio.

---

## Phase 2: Crash Prevention

### 2.1 Replace panics with Results in TTS weight access
**Files:** `src/tts/flow.rs:165`, `src/tts/hifigan.rs:121`
**Problem:** `w()` panics on missing weight key.
**Fix:**
```rust
fn w(&self, key: &str) -> Result<&Array> {
    self.weights.get(key).ok_or_else(||
        Error::ModelLoad(format!("Missing weight: {}", key)))
}
```
- Update all call sites to propagate `?`
- ~50 call sites in flow.rs, ~20 in hifigan.rs

### 2.2 Add weight load validation
**Files:** `src/model.rs:245-269`, `src/tts/flow.rs`, `src/tts/hifigan.rs`
**Problem:** Silent weight load failures — model runs with random/zero weights.
**Fix:**
- After loading, count matched vs total weights
- Log warning if <90% matched
- Return error if <50% matched
- Add `validate_weights()` method that checks all expected keys exist

### 2.3 Fix STFT off-by-one for short audio
**Files:** `src/audio.rs:381`
**Problem:** Frame calculation edge case for very short audio.
**Fix:**
- Return error for audio shorter than one STFT frame instead of producing garbage
- Add minimum audio duration check (e.g., 0.1s)

---

## Phase 3: Performance

### 3.1 Remove unnecessary Array clones in encoder
**Files:** `src/encoder.rs:274, 280`
**Problem:** 64 GPU memory allocations per inference (32 layers × 2 clones).
**Fix:**
- MLX Arrays are reference-counted. Use `&x` for residual instead of `x.clone()`
- If MLX requires owned Array for add(), check if there's an in-place or borrow-based add
- If clone is truly needed, document why

### 3.2 Add KV cache size limit
**Files:** `src/llm.rs`, `src/model.rs`
**Problem:** Unbounded cache growth, OOM on long generations.
**Fix:**
- Add `max_cache_len` config (default 4096)
- When cache exceeds limit, truncate oldest entries (sliding window)
- Alternative: implement StreamingLLM-style attention sink

### 3.3 Optimize generation loop eval pattern
**Files:** `src/model.rs:447+`
**Problem:** `item::<i32>()` forces synchronous eval every token.
**Fix:** This is inherent to autoregressive generation (need token ID to decide next input). Document as known limitation. Future: implement speculative decoding for batched eval.

---

## Phase 4: Robustness

### 4.1 Integer type safety in audio embedding insertion
**Files:** `src/model.rs:405, 424`
**Fix:**
- Use `usize` consistently for indices
- Add bounds checks before casting
- Assert audio_len > 0 before use

### 4.2 Improve tokenizer error handling
**Files:** `src/model.rs:121-127`
**Fix:**
- Make tokenizer required (return error if missing/broken)
- Or: clearly log at startup that output will be raw token IDs
- Add `--no-tokenizer` flag for intentional skip

### 4.3 STFT memory optimization for long audio
**Files:** `src/audio.rs:441`
**Fix:**
- Process STFT in chunks (e.g., 1000 frames at a time)
- Stream results into output buffer
- Cap CPU fallback to MAX_AUDIO_DURATION_SECS worth of frames

### 4.4 Clean up dead code
**Files:** `src/tts/hifigan.rs:124` (w_opt), `src/model.rs:536` (build_tts_prompt)
**Fix:** Remove unused methods. Run `cargo fix` for unused imports/variables.

---

## Phase 5: Testing

### 5.1 Add integration tests with real audio
- Short audio (<1s): edge case handling
- Medium audio (5-15s): normal path
- Long audio (>30s): chunking path
- Silence: graceful handling
- Non-speech noise: no crash

### 5.2 Add weight validation tests
- Load model, verify all expected weight keys present
- Verify weight shapes match expected dimensions
- Test with deliberately missing weights — should error not panic

### 5.3 Add generation quality tests
- Verify repetition penalty prevents loops
- Verify max_tokens respected
- Verify stop tokens work

---

## Execution Order

```
Phase 1 (Generation Quality)  → Most user-visible impact
  1.1 Repetition penalty       ← START HERE
  1.2 Fix max_tokens
  1.3 Unify transcribe methods

Phase 2 (Crash Prevention)    → Reliability
  2.1 Replace panics
  2.2 Weight validation
  2.3 STFT edge case

Phase 3 (Performance)         → Speed
  3.1 Remove encoder clones
  3.2 KV cache limit
  3.3 Document eval limitation

Phase 4 (Robustness)          → Polish
  4.1-4.4 in any order

Phase 5 (Testing)             → Confidence
  After each phase, add relevant tests
```
