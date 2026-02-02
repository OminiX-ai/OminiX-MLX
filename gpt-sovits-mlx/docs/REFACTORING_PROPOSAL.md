# GPT-SoVITS Rust Codebase Refactoring Proposal

## Current State Analysis

### Code Size Summary

| File | Lines | Size | Concern |
|------|-------|------|---------|
| `voice_clone.rs` | 2,257 | 97KB | **Monolithic god object** |
| `text/preprocessor.rs` | 2,469 | 98KB | Overlaps with voice_clone |
| `models/vits.rs` | 2,239 | 82KB | Large but acceptable |
| `models/bert.rs` | 1,054 | 35KB | OK |
| `models/t2s.rs` | 1,051 | 35KB | OK |
| `text/tone_sandhi.rs` | 1,039 | 39KB | OK (data-heavy) |

**Total codebase:** ~15,000 lines across src/

### Problems Identified

#### 1. `voice_clone.rs` is a God Object (2,257 lines)

Contains **6 distinct responsibilities** that should be separate modules:

```
voice_clone.rs
├── Pipeline orchestration (VoiceCloner struct)
├── Sampling logic (sample_top_k_with_penalty, sample_top_k, detect_repetition)
├── Text chunking (cut5_split, split_text_by_language, chunk_segments_by_length)
├── Language detection (get_lang_detector, is_cjk_char)
├── Audio output handling (AudioOutput, save_wav, play)
└── Configuration (VoiceClonerConfig)
```

#### 2. Duplicated Functionality

| Function in voice_clone.rs | Already exists in |
|---------------------------|-------------------|
| `cut5_split()` | `text/preprocessor.rs` has text splitting |
| `split_text_by_language()` | `text/lang_segment.rs` |
| `is_cjk_char()` | Common pattern, should be in `text/utils.rs` |
| `compute_word2ph()` | Should be in `text/` module |
| `estimate_phoneme_count()` | Should be in `text/` module |

#### 3. Sampling Logic Hardcoded

The sampling functions (`sample_top_k_with_penalty`, `sample_top_k`) are:
- 120+ lines of dense logic
- Hardcoded parameters scattered throughout
- Not reusable for other models

#### 4. No Clear Trait Boundaries

The `VoiceCloner` struct does everything directly. No traits for:
- Text preprocessing
- Token generation
- Audio synthesis
- Reference encoding

This makes testing difficult and prevents swapping implementations.

---

## Proposed Refactoring

### Direction 1: Extract Sampling Module

**New file:** `src/sampling.rs`

```rust
pub struct SamplingConfig {
    pub top_k: i32,
    pub top_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
}

pub struct Sampler {
    config: SamplingConfig,
    previous_tokens: Vec<i32>,
}

impl Sampler {
    pub fn sample(&mut self, logits: &Array) -> Result<(i32, i32), Error>;
    pub fn sample_with_eos_mask(&mut self, logits: &Array) -> Result<(i32, i32), Error>;
    pub fn reset(&mut self);
}
```

**Benefits:**
- Reusable across different models
- Testable in isolation
- Clear configuration surface

---

### Direction 2: Consolidate Text Processing

**Merge duplicates into `text/` module:**

```
src/text/
├── mod.rs
├── chunking.rs        ← NEW: cut5_split, merge_short, split_at_punctuation
├── language.rs        ← NEW: language detection, is_cjk_char
├── g2p/
│   ├── mod.rs
│   ├── chinese.rs
│   ├── english.rs
│   └── mixed.rs       ← g2pw integration
├── normalization.rs   ← existing text_normalizer.rs
├── symbols.rs         ← existing
└── bert_features.rs   ← existing
```

**Remove from voice_clone.rs:**
- `cut5_split()` → `text::chunking::cut5`
- `split_text_by_language()` → `text::chunking::split_by_language`
- `is_cjk_char()` → `text::language::is_cjk`
- `get_lang_detector()` → `text::language::detector()`
- `compute_word2ph()` → `text::g2p::word2ph`
- `count_english_word2ph_entries()` → `text::g2p::english::word2ph_count`

**Estimated reduction:** ~400 lines from voice_clone.rs

---

### Direction 3: Extract Audio I/O Module

**New file:** `src/audio/output.rs`

```rust
pub struct AudioOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl AudioOutput {
    pub fn duration_secs(&self) -> f32;
    pub fn to_i16(&self) -> Vec<i16>;
    pub fn fade_in(&mut self, ms: f32);
    pub fn trim_start(&mut self, ms: f32);
    pub fn save_wav(&self, path: &Path) -> Result<()>;
    pub fn play(&self) -> Result<()>;
    pub fn play_blocking(&self) -> Result<()>;
}
```

**Move from voice_clone.rs:**
- `AudioOutput` struct and all its methods
- `save_wav()` implementation
- `play()` / `play_blocking()` implementations
- `array_to_f32_samples()` helper

**Estimated reduction:** ~150 lines from voice_clone.rs

---

### Direction 4: Pipeline Trait Abstraction

**New file:** `src/pipeline.rs`

```rust
/// Text-to-semantic token generation
pub trait TextToSemantic {
    fn generate(&mut self, text: &str, config: &T2SConfig) -> Result<Vec<i32>>;
}

/// Semantic tokens to audio
pub trait SemanticToAudio {
    fn synthesize(&mut self, tokens: &[i32], phones: &[i32]) -> Result<AudioOutput>;
}

/// Reference audio encoder
pub trait ReferenceEncoder {
    fn encode(&mut self, audio_path: &Path) -> Result<ReferenceContext>;
    fn encode_with_text(&mut self, audio_path: &Path, text: &str) -> Result<ReferenceContext>;
}

/// Full TTS pipeline
pub trait TTSPipeline: TextToSemantic + SemanticToAudio {
    fn synthesize_text(&mut self, text: &str) -> Result<AudioOutput>;
}
```

**Benefits:**
- Swap MLX VITS for ONNX VITS transparently
- Mock components for testing
- Clear API boundaries

---

### Direction 5: Configuration Consolidation

**Current:** Config scattered across multiple places
- `VoiceClonerConfig` in voice_clone.rs
- Hardcoded paths in voice_clone.rs
- Model configs in each model file

**Proposed:** Single hierarchical config

```rust
// src/config.rs
pub struct GPTSoVITSConfig {
    pub models: ModelPaths,
    pub t2s: T2SConfig,
    pub vits: VITSConfig,
    pub sampling: SamplingConfig,
    pub audio: AudioConfig,
}

pub struct ModelPaths {
    pub t2s_weights: PathBuf,
    pub vits_weights: PathBuf,
    pub bert_path: PathBuf,
    pub hubert_path: PathBuf,
    pub g2pw_path: Option<PathBuf>,
    pub vits_onnx_path: Option<PathBuf>,
}

impl GPTSoVITSConfig {
    pub fn from_yaml(path: &Path) -> Result<Self>;
    pub fn with_defaults() -> Self;
}
```

---

## Refactored Structure

```
src/
├── lib.rs
├── config.rs              ← NEW: unified configuration
├── error.rs
├── pipeline.rs            ← NEW: trait definitions
├── sampling.rs            ← NEW: extracted from voice_clone
├── voice_clone.rs         ← SLIMMED: ~800 lines (orchestration only)
├── audio/
│   ├── mod.rs
│   ├── io.rs              ← load/save WAV
│   ├── output.rs          ← NEW: AudioOutput struct
│   ├── mel.rs             ← mel spectrogram extraction
│   └── playback.rs        ← NEW: play functions
├── text/
│   ├── mod.rs
│   ├── chunking.rs        ← NEW: cut5, merge, split
│   ├── language.rs        ← NEW: detection, is_cjk
│   ├── g2p/
│   │   ├── mod.rs
│   │   ├── chinese.rs
│   │   ├── english.rs
│   │   └── mixed.rs
│   ├── normalization.rs
│   ├── symbols.rs
│   └── bert_features.rs
└── models/
    ├── mod.rs
    ├── bert.rs
    ├── hubert.rs
    ├── t2s.rs
    ├── vits.rs
    └── vits_onnx.rs
```

---

## Implementation Priority

| Priority | Task | Impact | Effort |
|----------|------|--------|--------|
| **P0** | Extract `sampling.rs` | High (cleanest win) | Low |
| **P0** | Extract `audio/output.rs` | Medium | Low |
| **P1** | Consolidate text chunking | High (removes duplication) | Medium |
| **P1** | Create `config.rs` | Medium (maintainability) | Medium |
| **P2** | Define pipeline traits | High (composability) | Medium |
| **P2** | Reorganize `text/g2p/` | Medium | Medium |

---

## Expected Outcomes

### Before
```
voice_clone.rs: 2,257 lines (god object)
```

### After
```
voice_clone.rs:    ~800 lines (orchestration only)
sampling.rs:       ~200 lines
audio/output.rs:   ~150 lines
audio/playback.rs: ~50 lines
text/chunking.rs:  ~300 lines
text/language.rs:  ~100 lines
config.rs:         ~150 lines
pipeline.rs:       ~100 lines
```

**Total:** Same lines, but properly organized with clear responsibilities.

---

## Quick Win: Extract Sampling (P0) ✅ COMPLETED

This was completed successfully:

1. ✅ Created `src/sampling.rs` (372 lines)
2. ✅ Moved `sample_top_k_with_penalty()` and `sample_top_k()`
3. ✅ Created `Sampler` struct to hold state
4. ✅ Updated `voice_clone.rs` to use `Sampler`
5. ✅ Added unit tests for sampling logic

**Results:**
- `voice_clone.rs`: 2,257 → 2,061 lines (-196 lines, -8.7%)
- `sampling.rs`: 372 lines (new)
- All tests pass
- Functional verification successful

**New `sampling.rs` API:**
```rust
pub struct SamplingConfig {
    pub top_k: i32,
    pub top_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub eos_token: i32,
}

pub struct Sampler {
    // Holds config + previous_tokens for repetition penalty
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Self;
    pub fn sample(&mut self, logits: &Array) -> Result<(i32, i32), Error>;  // (sampled, argmax)
    pub fn sample_with_eos_mask(&mut self, logits: &Array) -> Result<(i32, i32), Error>;
    pub fn add_token(&mut self, token: i32);
    pub fn reset(&mut self);
}

pub fn detect_repetition(tokens: &[i32], n: usize, min_count: usize) -> bool;
```
