# Unified Translation Pipeline Analysis

**Date:** 2026-01-30
**Scope:** Leveraging funasr-nano's integrated LLM for correction + translation
**Status:** TESTED - Key Limitations Identified

---

## CRITICAL FINDING

**The Qwen3-0.6B in funasr-nano was fine-tuned EXCLUSIVELY for ASR.** It cannot perform:
- Text translation
- Modified prompt tasks
- Any non-ASR generation

**Test Results:**
```
Input: "今天天气很好，我们去公园散步吧。"
Prompt: "请将以下中文翻译为英文：{input}"
Output: "!!!!!!!!!!"  <-- Model outputs garbage for non-ASR tasks

Audio-to-English prompt: Still outputs Chinese transcription
Custom prompts: Ignored, model always does ASR
```

**Conclusion:** Cannot leverage funasr-nano's LLM for translation. Must use separate translation LLM.

---

## Current Pain Points

1. **ASR Segmentation Errors**: Wrong word boundaries from variable speaker pacing, especially in Chinese/English mixed speech
2. **LLM Correction Latency**: Separate LLM call adds ~200-500ms for error correction
3. **Pipeline Overhead**: ASR -> LLM (correction) -> LLM (translation) -> TTS = multiple serialized calls

---

## funasr-nano Architecture

```
Audio -> SenseVoice Encoder (70 layers) -> Audio Adaptor (2 layers) -> Qwen3-0.6B (28 layers)
                                                                              |
                                                                              v
                                                                       Transcription
```

**Current Prompt** (hardcoded in `src/model.rs:547-575`):
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
语音转写成中文：<|startofspeech|>{AUDIO}<|endofspeech|><|im_end|>
<|im_start|>assistant
```

---

## Approach 1: Modified Prompt for Unified Task

**Concept:** Change the prompt to request correction + translation in one pass.

**Proposed Prompt:**
```
<|im_start|>system
You are a speech translation assistant. Transcribe audio accurately, correct any recognition errors, and translate to English.<|im_end|>
<|im_start|>user
请转写并翻译以下语音为英文：<|startofspeech|>{AUDIO}<|endofspeech|><|im_end|>
<|im_start|>assistant
```

**Implementation Changes:**
```rust
// src/model.rs - new method
pub fn translate_to_english(&mut self, audio_path: impl AsRef<Path>) -> Result<String> {
    // ... audio processing same as transcribe()
    self.generate_text_with_prompt(
        &audio_features,
        "请转写并翻译以下语音为英文：",  // Custom prompt
        &SamplingConfig::default(),
    )
}
```

**Pros:**
- Single inference pass
- No additional model loading
- Leverages audio context for better accuracy

**Cons:**
- Qwen3-0.6B (620M params) is small for translation
- Model was fine-tuned for ASR, not translation
- Unknown if training data included translation pairs
- May hallucinate translations

**Verdict:** Requires testing. May work for simple sentences but likely poor for complex translation.

---

## Approach 2: Two-Pass with Shared Model

**Concept:** Use same Qwen3-0.6B twice - once for ASR, once for text correction/translation.

**Pass 1 (ASR):**
```
语音转写成中文：<|startofspeech|>{AUDIO}<|endofspeech|>
-> "今天天气很好" (with potential errors)
```

**Pass 2 (Text-only correction + translation):**
```
<|im_start|>user
请纠正并翻译以下中文为英文：今天天气很好<|im_end|>
<|im_start|>assistant
The weather is nice today.
```

**Implementation:**
```rust
impl FunASRNano {
    pub fn transcribe_and_translate(&mut self, audio_path: impl AsRef<Path>) -> Result<(String, String)> {
        // Pass 1: ASR
        let transcription = self.transcribe(audio_path)?;

        // Pass 2: Correction + Translation (text only)
        let translation = self.translate_text(&transcription)?;

        Ok((transcription, translation))
    }

    pub fn translate_text(&mut self, text: &str) -> Result<String> {
        // Use LLM directly without audio
        let prompt = format!("请纠正并翻译以下中文为英文：{}", text);
        self.llm.generate(&prompt, &mut vec![], &SamplingConfig::default())
    }
}
```

**Pros:**
- Clear separation of concerns
- Can debug each step independently
- Same model weights, no extra memory

**Cons:**
- Still two inference passes (but faster than loading separate model)
- Small LLM translation quality questionable

**Estimated Latency:**
- ASR pass: ~300-500ms (current)
- Translation pass: ~200-300ms (text-only, shorter sequence)
- Total: ~500-800ms (vs current ~800-1200ms with separate LLM)

---

## Approach 3: Speculative-like Audio Verification

**Concept:** Use ASR output as "draft", have translation LLM verify against audio context.

**Architecture:**
```
Audio -> Encoder -> Adaptor -> Audio Embeddings
                                    |
                                    +---> ASR Decoder (fast draft) -> "今天天气很好"
                                    |
                                    +---> Translation LLM (verify + translate)
                                          Input: Audio Embeddings + ASR Draft
                                          Output: Verified Translation
```

**Key Insight:** The audio embeddings contain rich contextual information that can help the translation model:
1. Detect when ASR made segmentation errors (audio context doesn't match text)
2. Understand prosody for better translation (question marks, emphasis)
3. Handle code-switching (Chinese/English mix) by attending to audio

**Implementation Concept:**
```rust
pub fn translate_with_audio_context(
    &mut self,
    audio_features: &Array,
    asr_draft: &str,
    target_lang: &str,
) -> Result<String> {
    // Build prompt with both audio and ASR draft
    let prompt = format!(
        "语音内容已转写为：{}\n请验证并翻译为{}：",
        asr_draft, target_lang
    );

    // Generate with audio context
    self.generate_text_with_audio_and_prompt(audio_features, &prompt)
}
```

**Pros:**
- Audio context helps verify/correct ASR errors
- Single encoder pass, shared audio embeddings
- Elegant speculative-like verification

**Cons:**
- Requires architectural changes
- Need to modify prompt injection
- More complex implementation

**Verdict:** Most promising for accuracy, but requires significant development.

---

## Approach 4: Concurrent Shared Encoder

**Concept:** Share encoder output, run ASR and translation decoders in parallel.

**Architecture:**
```
                           +---> Qwen3-0.6B (ASR) --> Chinese
Audio -> Encoder -> Adaptor
                           +---> Separate LLM (Translation) --> English
```

**Benefits:**
- Encoder runs once (~50% of total time)
- Decoders run in parallel
- Best latency if translation LLM can use same audio embeddings

**Challenge:** Translation LLM needs to accept audio embeddings from funasr-nano's adaptor.

**Options:**
1. Fine-tune translation LLM with audio adaptor (expensive)
2. Use audio embeddings as prefix for translation LLM (may work with frozen LLM)
3. Train small projection layer to map embeddings (low-cost fine-tuning)

---

## Approach 5: Streaming Pipeline Optimization

**Concept:** Overlap ASR and translation for lower perceived latency.

**Timeline:**
```
Time: 0ms    200ms   400ms   600ms   800ms   1000ms
      |-------|-------|-------|-------|-------|
ASR:  [===ENCODE===][=DECODE=]
                       |
Translation:           [==LLM==]
                            |
TTS:                        [==SYNTH==]
```

**With Overlap:**
```
ASR:  [===ENCODE===][=DECODE=]
Translation:    [==LLM==] (starts when first tokens available)
TTS:               [==SYNTH==] (starts with first translation tokens)
```

**Implementation:**
```rust
pub fn translate_streaming(
    &mut self,
    audio_path: impl AsRef<Path>,
    callback: impl FnMut(&str),
) -> Result<String> {
    // Start ASR
    let asr_stream = self.transcribe_streaming(audio_path)?;

    // Start translation as ASR tokens arrive
    for partial_text in asr_stream {
        if partial_text.ends_with(['。', '，', '？', '！']) {
            // Sentence boundary - translate this chunk
            let translation = self.translate_text(&partial_text)?;
            callback(&translation);
        }
    }
}
```

---

## Recommendations

### Short-term (Low Effort)

1. **Test Approach 1** - Try modified prompts to see if Qwen3-0.6B can do basic translation
2. **Implement Approach 2** - Two-pass with shared model is straightforward

### Medium-term (Medium Effort)

3. **Implement Approach 5** - Streaming pipeline for lower perceived latency
4. **Benchmark Qwen3-0.6B translation quality** - Determine if it's viable

### Long-term (High Effort)

5. **Implement Approach 3** - Audio-conditioned verification for best accuracy
6. **Consider fine-tuning** - If translation quality insufficient, fine-tune on translation pairs

---

## Next Steps

1. Create test script to evaluate Qwen3-0.6B translation capability
2. Implement `translate_text()` method for text-only LLM inference
3. Add configurable prompt support to `FunASRNano`
4. Benchmark latency comparison between current pipeline and unified approach

---

## Appendix: Speculative Decoding for LLM Latency

### Current Limitation

Speculative decoding requires a smaller draft model with compatible vocabulary. For Qwen3-0.6B:
- No smaller Qwen3 model available (0.6B is smallest)
- Qwen2.5-0.5B has different vocabulary (incompatible)
- Training custom draft model would be expensive

### Alternative: Self-Speculative Decoding

Recent research shows self-speculative decoding without draft model:
- **Medusa**: Add prediction heads to main model
- **Lookahead**: Use n-gram cache from previous generations
- **Draft and Verify with Exit Layers**: Use early layers as draft

These could potentially reduce Qwen3-0.6B latency by 1.3-1.5x without external draft model.

### Recommendation

Focus on pipeline optimization (streaming, overlap) rather than speculative decoding, given:
- No suitable draft model for Qwen3
- Self-speculative requires model modification
- Streaming overlap is simpler and effective

---

## REVISED RECOMMENDATIONS (Post-Testing)

Based on empirical testing, the original approaches are NOT viable because the Qwen3-0.6B
was fine-tuned to ignore prompts and always output Chinese ASR.

### Viable Options

#### Option 1: Replace LLM with Translation-Capable Model

Fine-tune or swap Qwen3-0.6B with a version trained for translation:

```
Audio -> Encoder -> Adaptor -> Qwen3-0.6B (translation-tuned)
                                     |
                                     v
                               English Translation
```

**Effort:** High (requires fine-tuning data and training)
**Impact:** Would enable unified pipeline

#### Option 2: Add Secondary LLM Head

Keep ASR head, add translation head:

```
Audio -> Encoder -> Adaptor --> ASR Head (existing) --> Chinese
                           \-> Translation Head (new) --> English
```

**Effort:** High (new model training)
**Impact:** Parallel inference possible

#### Option 3: Efficient External Translation (Recommended)

Accept that translation needs a separate model, but optimize latency:

**Pipeline:**
```
Time: 0ms    100ms   200ms   300ms   400ms   500ms
      |-------|-------|-------|-------|-------|
ASR:  [==ENCODE==][==DECODE==]
                        |
Translation:            [===SMALL-LLM===]
```

**Optimizations:**
1. **Use smaller translation LLM** (e.g., Qwen2.5-0.5B-Instruct, ~300ms for short text)
2. **Stream ASR output to translation** (start translating as soon as sentence boundary detected)
3. **Cache translation LLM** (keep loaded in memory)
4. **Batch short segments** (reduce LLM call overhead)

**Implementation:**
```rust
// In your pipeline code (not funasr-nano itself)
fn translate_pipeline(audio: &[f32], asr: &mut FunASRNano, translator: &mut TranslationLLM) -> String {
    // Run ASR
    let chinese = asr.transcribe_samples(audio, 16000)?;

    // Run translation (use a real translation model)
    let english = translator.translate(&chinese)?;

    english
}
```

#### Option 4: Streaming Pipeline with Sentence Detection

For real-time translation, don't wait for full transcription:

```rust
fn streaming_translate(
    audio_stream: impl Iterator<Item = Vec<f32>>,
    asr: &mut FunASRNano,
    translator: &mut TranslationLLM,
) {
    let mut buffer = String::new();

    for chunk in audio_stream {
        if let Some(partial) = asr.transcribe_chunk(&mut ctx, &chunk, 16000)? {
            buffer.push_str(&partial);

            // Check for sentence boundary
            if partial.ends_with(['。', '！', '？', '.', '!', '?']) {
                // Translate complete sentence
                let translated = translator.translate(&buffer)?;
                emit_translation(&translated);
                buffer.clear();
            }
        }
    }
}
```

---

## Next Steps

1. **Accept that unified approach won't work** with current funasr-nano model
2. **Benchmark small translation models** (Qwen2.5-0.5B-Instruct, NLLB-200)
3. **Implement streaming pipeline** with sentence-boundary detection
4. **Consider model quantization** for translation LLM if memory is a concern
5. **Long-term:** Investigate fine-tuning funasr-nano for translation tasks

---

*Document created: 2026-01-30*
*Updated: 2026-01-30 - Added empirical test results and revised recommendations*
