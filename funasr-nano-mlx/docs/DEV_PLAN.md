# Development Plan: funasr-nano-mlx

**Created**: 2026-01-25
**Target**: Production-ready ASR library for Apple Silicon

---

## Goals

1. **Performance**: Achieve 30x+ real-time transcription (currently 16x)
2. **Features**: Streaming API, batch processing, advanced sampling
3. **Quality**: Comprehensive error handling, tests, documentation
4. **Reliability**: Handle edge cases, validate inputs

---

## Lessons from funasr-mlx (Paraformer Implementation)

The funasr-mlx crate provides a well-structured reference implementation. Key patterns to adopt:

### 1. FFT with Cached Planner (45x speedup)

**funasr-mlx/src/paraformer.rs:155-222**
```rust
use rustfft::{num_complex::Complex, FftPlanner};

pub struct MelFrontend {
    // Cached FFT instance for efficient repeated STFT computation
    fft: Arc<dyn rustfft::Fft<f32>>,
    mel_filters: Vec<f32>,  // Pre-computed filterbank
    window: Vec<f32>,       // Pre-computed Hamming window
}

impl MelFrontend {
    pub fn new(config: &Config) -> Self {
        // Pre-create FFT planner for efficient repeated use
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);
        // ...
    }
}
```

### 2. Error Handling with thiserror

**funasr-mlx/src/error.rs**
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),  // Auto-convert from MLX errors

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model error: {0}")]
    Model(String),
}
```

### 3. Clean Configuration Struct

**funasr-mlx/src/paraformer.rs:55-108**
```rust
#[derive(Debug, Clone)]
pub struct ParaformerConfig {
    // Audio frontend
    pub sample_rate: i32,
    pub n_mels: i32,
    pub n_fft: i32,
    pub hop_length: i32,
    pub lfr_m: i32,
    pub lfr_n: i32,

    // Encoder
    pub encoder_dim: i32,
    pub encoder_layers: i32,
    // ...
}

impl Default for ParaformerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_mels: 80,
            n_fft: 400,      // 25ms window - documented!
            hop_length: 160, // 10ms hop
            // ...
        }
    }
}
```

### 4. Weight Loading Helpers

**funasr-mlx/src/paraformer.rs:1286-1298**
```rust
fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::Model(format!("Missing weight: {}", key)))
}

fn get_conv_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    let weight = get_weight(weights, key)?;
    weight.transpose_axes(&[0, 2, 1])
        .map_err(|e| Error::Model(format!("Failed to transpose: {}", e)))
}
```

### 5. High-Level API

**funasr-mlx/src/lib.rs:121-138**
```rust
/// High-level transcription function
pub fn transcribe(
    model: &mut Paraformer,
    audio: &[f32],
    vocab: &Vocabulary,
) -> Result<String> {
    let audio_array = Array::from_slice(audio, &[audio.len() as i32]);
    let token_ids = model.transcribe(&audio_array)?;
    eval([&token_ids])?;

    let token_ids_vec: Vec<i32> = token_ids.try_as_slice::<i32>()?.to_vec();
    Ok(vocab.decode(&token_ids_vec))
}
```

### 6. Vocabulary with Special Token Filtering

**funasr-mlx/src/lib.rs:62-110**
```rust
pub struct Vocabulary {
    tokens: Vec<String>,
}

impl Vocabulary {
    pub fn decode(&self, token_ids: &[i32]) -> String {
        token_ids.iter()
            .filter_map(|&id| {
                let token = &self.tokens[id as usize];
                // Filter special tokens
                if token == "<blank>" || token == "<s>" || token == "</s>" {
                    None
                } else {
                    Some(token.clone())
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }
}
```

### 7. Architecture Documentation in Code

**funasr-mlx/src/paraformer.rs:1-28**
```rust
//! # Architecture
//!
//! ```text
//! Audio (16kHz)
//!     ↓
//! [Mel Frontend] - 80 bins, 25ms window, 10ms hop, LFR 7/6
//!     ↓
//! [SAN-M Encoder] - 50 layers, 512 hidden, 4 heads
//!     ↓
//! [CIF Predictor] - Continuous Integrate-and-Fire
//!     ↓
//! [Bidirectional Decoder] - 16 layers, 512 hidden, 4 heads
//!     ↓
//! Tokens [batch, num_tokens]
//! ```
```

---

## Milestones

### Milestone 1: Performance Optimization (P0)
**Target**: 2x speedup (16x → 32x real-time)

### Milestone 2: Core Features (P1)
**Target**: Streaming & batch APIs

### Milestone 3: Robustness (P2)
**Target**: Production-ready error handling & tests

### Milestone 4: Polish (P3)
**Target**: Documentation & minor features

---

## Phase 1: Performance (P0)

### Task 1.1: Replace DFT with FFT
**Priority**: P0 | **Effort**: Medium | **Impact**: HIGH (10-100x audio processing speedup)

```rust
// Add to Cargo.toml:
rustfft = "6.2"

// Replace audio.rs:154-163 with:
use rustfft::{FftPlanner, num_complex::Complex};

fn compute_fft(frame: &[f32], n_fft: usize) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    // ... FFT implementation
}
```

**Files**: `src/audio.rs`
**Tests**: Verify mel spectrogram output matches current implementation

---

### Task 1.2: Use Optimized SDPA in Encoder
**Priority**: P0 | **Effort**: Low | **Impact**: HIGH (2-3x encoder speedup)

```rust
// Replace sensevoice_encoder.rs:232-240 with:
let attn_out = mlx_rs::fast::scaled_dot_product_attention(
    q, k, v, self.scale, None::<ScaledDotProductAttentionMask>
)?;
```

**Files**: `src/sensevoice_encoder.rs`
**Tests**: Compare encoder output before/after

---

### Task 1.3: Reduce Array Cloning
**Priority**: P0 | **Effort**: Medium | **Impact**: MEDIUM (memory reduction)

```rust
// Before (qwen.rs:263):
let residual = x.clone();
let h = self.input_layernorm.forward(&x)?;

// After:
let h = self.input_layernorm.forward(x)?;
let h = self.self_attn.forward_with_cache(&h, cache, mask)?;
x.add(&h)?  // x is still valid
```

**Files**: `src/qwen.rs`, `src/sensevoice_encoder.rs`, `src/adaptor.rs`
**Tests**: Memory profiling before/after

---

### Task 1.4: GPU-Accelerated LFR
**Priority**: P0 | **Effort**: Medium | **Impact**: MEDIUM

```rust
// Replace CPU-based LFR (audio.rs:241-307) with MLX operations:
pub fn apply_lfr_gpu(mel: &Array, lfr_m: i32, lfr_n: i32) -> Result<Array> {
    // Use MLX reshape, pad, concatenate instead of CPU Vec operations
}
```

**Files**: `src/audio.rs`

---

## Phase 2: Core Features (P1)

### Task 2.1: Streaming Transcription API
**Priority**: P1 | **Effort**: High | **Impact**: HIGH

```rust
// New API in model.rs:

/// Streaming context holding encoder state and partial results
pub struct StreamingContext {
    encoder_state: Option<Array>,
    cache: Vec<Option<KVCache>>,
    pending_audio: Vec<f32>,
    partial_text: String,
}

impl FunASRNano {
    /// Create a new streaming context
    pub fn create_streaming_context(&self) -> StreamingContext;

    /// Process an audio chunk (16kHz f32 samples)
    /// Returns partial transcription if available
    pub fn transcribe_chunk(
        &mut self,
        ctx: &mut StreamingContext,
        chunk: &[f32]
    ) -> Result<Option<String>>;

    /// Finalize streaming and return complete transcription
    pub fn finalize_stream(&mut self, ctx: StreamingContext) -> Result<String>;
}
```

**Files**: `src/model.rs`, `src/lib.rs`
**New Files**: `src/streaming.rs`
**Tests**: `tests/streaming_test.rs`

---

### Task 2.2: Batch Processing API
**Priority**: P1 | **Effort**: Medium | **Impact**: HIGH

```rust
// New API in model.rs:

impl FunASRNano {
    /// Transcribe multiple audio files in parallel
    pub fn transcribe_batch<P: AsRef<Path>>(
        &mut self,
        paths: &[P]
    ) -> Result<Vec<String>>;

    /// Transcribe multiple audio arrays
    pub fn transcribe_arrays(
        &mut self,
        audios: &[&[f32]],
        sample_rates: &[u32],
    ) -> Result<Vec<String>>;
}
```

**Files**: `src/model.rs`
**Tests**: `tests/batch_test.rs`

---

### Task 2.3: Sampling Strategies
**Priority**: P1 | **Effort**: Medium | **Impact**: MEDIUM

```rust
// New file: src/sampling.rs

#[derive(Clone, Debug)]
pub enum SamplingStrategy {
    /// Greedy decoding (temperature = 0)
    Greedy,
    /// Top-k sampling
    TopK { k: usize, temperature: f32 },
    /// Nucleus (top-p) sampling
    TopP { p: f32, temperature: f32 },
    /// Temperature sampling
    Temperature { temperature: f32 },
}

impl SamplingStrategy {
    pub fn sample(&self, logits: &Array) -> Result<i32>;
}

// Update TranscriptionOptions:
pub struct TranscriptionOptions {
    pub sampling: SamplingStrategy,
    pub max_tokens: usize,
    pub repetition_penalty: f32,
}
```

**Files**: `src/sampling.rs`, `src/model.rs`

---

### Task 2.4: Configurable Prompt Templates
**Priority**: P1 | **Effort**: Low | **Impact**: MEDIUM

```rust
// New in model.rs:

pub struct PromptTemplate {
    pub system_message: String,
    pub user_prefix: String,
    pub assistant_prefix: String,
}

impl Default for PromptTemplate {
    fn default() -> Self {
        Self {
            system_message: "You are a helpful assistant.".into(),
            user_prefix: "语音转写成中文：".into(),
            assistant_prefix: "".into(),
        }
    }
}
```

**Files**: `src/model.rs`

---

## Phase 3: Robustness (P2)

### Task 3.1: Comprehensive Error Handling
**Priority**: P2 | **Effort**: Medium | **Impact**: HIGH

```rust
// Improve error.rs:

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Audio processing failed: {message}")]
    Audio {
        message: String,
        #[source] source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Invalid configuration: {field} - {reason}")]
    InvalidConfig { field: String, reason: String },

    #[error("Model loading failed: {path}")]
    ModelLoad {
        path: PathBuf,
        #[source] source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Tokenizer not available - cannot decode output")]
    TokenizerMissing,

    #[error("Audio too short: {duration_ms}ms (minimum: {min_ms}ms)")]
    AudioTooShort { duration_ms: u64, min_ms: u64 },

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<i32>, actual: Vec<i32> },
}
```

**Files**: `src/error.rs`, all files using errors

---

### Task 3.2: Input Validation
**Priority**: P2 | **Effort**: Medium | **Impact**: HIGH

```rust
// Add to model.rs:

impl FunASRNanoConfig {
    pub fn validate(&self) -> Result<()> {
        // Dimension compatibility
        if self.adaptor.encoder_dim != self.encoder.output_size {
            return Err(Error::InvalidConfig {
                field: "adaptor.encoder_dim".into(),
                reason: format!(
                    "must equal encoder.output_size ({})",
                    self.encoder.output_size
                ),
            });
        }
        // ... more validations
        Ok(())
    }
}

// Add to audio.rs:
fn validate_audio(samples: &[f32], sample_rate: u32) -> Result<()> {
    if samples.is_empty() {
        return Err(Error::AudioTooShort { duration_ms: 0, min_ms: 100 });
    }
    let duration_ms = (samples.len() as u64 * 1000) / sample_rate as u64;
    if duration_ms < 100 {
        return Err(Error::AudioTooShort { duration_ms, min_ms: 100 });
    }
    Ok(())
}
```

**Files**: `src/model.rs`, `src/audio.rs`

---

### Task 3.3: Comprehensive Test Suite
**Priority**: P2 | **Effort**: High | **Impact**: HIGH

```rust
// tests/encoder_test.rs
#[test]
fn test_sensevoice_encoder_output_shape() { }

#[test]
fn test_encoder_deterministic() { }

// tests/audio_test.rs
#[test]
fn test_mel_spectrogram_shape() { }

#[test]
fn test_lfr_padding_edge_cases() { }

#[test]
fn test_empty_audio_error() { }

// tests/model_test.rs
#[test]
fn test_transcribe_short_audio() { }

#[test]
fn test_transcribe_long_audio() { }

#[test]
fn test_invalid_config_error() { }

// tests/integration_test.rs
#[test]
fn test_end_to_end_chinese() { }

#[test]
fn test_end_to_end_english() { }
```

**New Files**: `tests/*.rs`

---

### Task 3.4: Remove Unused Code
**Priority**: P2 | **Effort**: Low | **Impact**: LOW

- Remove `sample_with_penalty` (model.rs:479-517)
- Remove or complete `whisper_encoder.rs` if unused
- Clean up dead code warnings

**Files**: `src/model.rs`, `src/whisper_encoder.rs`

---

## Phase 4: Polish (P3)

### Task 4.1: Architecture Documentation
**Priority**: P3 | **Effort**: Medium | **Impact**: HIGH

```markdown
# docs/ARCHITECTURE.md

## Data Flow

1. **Audio Input** (WAV, 16kHz mono)
   └── Shape: [samples]

2. **Mel Spectrogram** (audio.rs)
   └── Shape: [1, 80, time_frames]
   └── 80 mel bins, 25ms window, 10ms hop

3. **LFR (Low Frame Rate)** (audio.rs)
   └── Shape: [1, time_frames/6, 560]
   └── Stack 7 frames, subsample by 6

4. **SenseVoice Encoder** (sensevoice_encoder.rs)
   └── Shape: [1, time_frames/6, 512]
   └── 50 SAN-M layers + 20 TP layers

5. **Audio Adaptor** (adaptor.rs)
   └── Shape: [1, time_frames/6, 1024]
   └── Projects to LLM dimension

6. **Qwen3 LLM** (qwen.rs)
   └── Autoregressive generation
   └── 28 transformer layers
```

**New Files**: `docs/ARCHITECTURE.md`

---

### Task 4.2: API Examples
**Priority**: P3 | **Effort**: Medium | **Impact**: MEDIUM

```rust
// examples/basic_transcription.rs
// examples/streaming_transcription.rs
// examples/batch_processing.rs
// examples/custom_sampling.rs
```

**New Files**: `examples/*.rs`

---

### Task 4.3: Multiple Audio Format Support
**Priority**: P3 | **Effort**: Medium | **Impact**: LOW

```toml
# Cargo.toml
[features]
default = ["wav"]
wav = ["hound"]
mp3 = ["symphonia"]
all-formats = ["wav", "mp3"]
```

**Files**: `Cargo.toml`, `src/audio.rs`

---

### Task 4.4: Constants for Magic Numbers
**Priority**: P3 | **Effort**: Low | **Impact**: LOW

```rust
// New file: src/constants.rs

// Qwen3 special tokens
pub const QWEN_IM_START: i32 = 151644;
pub const QWEN_IM_END: i32 = 151645;
pub const QWEN_START_OF_SPEECH: i32 = 151646;
pub const QWEN_END_OF_SPEECH: i32 = 151647;
pub const QWEN_EOS: i32 = 151643;

// Audio processing
pub const DEFAULT_SAMPLE_RATE: u32 = 16000;
pub const DEFAULT_N_MELS: i32 = 80;
pub const DEFAULT_N_FFT: usize = 400;
pub const DEFAULT_HOP_LENGTH: usize = 160;
```

**New Files**: `src/constants.rs`

---

## Timeline

```
Week 1-2: Phase 1 (Performance)
├── Task 1.1: FFT implementation
├── Task 1.2: SDPA in encoder
├── Task 1.3: Reduce cloning
└── Task 1.4: GPU LFR

Week 3-4: Phase 2 (Features)
├── Task 2.1: Streaming API
├── Task 2.2: Batch API
├── Task 2.3: Sampling strategies
└── Task 2.4: Prompt templates

Week 5: Phase 3 (Robustness)
├── Task 3.1: Error handling
├── Task 3.2: Input validation
├── Task 3.3: Test suite
└── Task 3.4: Remove unused code

Week 6: Phase 4 (Polish)
├── Task 4.1: Architecture docs
├── Task 4.2: API examples
├── Task 4.3: Audio formats
└── Task 4.4: Constants
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Real-time factor | 16x | 32x |
| Test coverage | ~5% | 80% |
| API completeness | Basic | Streaming + Batch |
| Documentation | Minimal | Comprehensive |

---

## Dependencies to Add

```toml
[dependencies]
rustfft = "6.2"           # FFT for mel spectrogram
thiserror = "1.0"         # Better error handling

[dev-dependencies]
criterion = "0.5"         # Benchmarking
approx = "0.5"            # Float comparisons in tests
tempfile = "3.0"          # Test fixtures
```

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| FFT implementation breaks mel accuracy | Compare output against Python reference |
| SDPA change affects encoder quality | Validate transcription accuracy before/after |
| Streaming adds complexity | Start with simple chunk-based approach |
| Performance regression | Add benchmarks to CI |

---

## Appendix: funasr-mlx vs funasr-nano-mlx Comparison

| Aspect | funasr-mlx (Paraformer) | funasr-nano-mlx (Current) |
|--------|-------------------------|---------------------------|
| **FFT** | `rustfft` with cached planner | Manual O(n²) DFT |
| **Error Handling** | `thiserror` with `#[from]` | Manual string formatting |
| **Config** | Single `ParaformerConfig` struct | Split across modules |
| **Weight Loading** | Helper functions `get_weight()` | String replacement chain |
| **Vocabulary** | Separate `Vocabulary` struct | Built into tokenizer |
| **Architecture Docs** | ASCII art in module docs | Minimal |
| **Tests** | 3 unit tests | 2 unit tests |
| **CMVN** | Built into MelFrontend | Separate application |

### Files to Reference

| funasr-mlx File | What to Learn |
|-----------------|---------------|
| `src/paraformer.rs:155-222` | MelFrontend with cached FFT |
| `src/paraformer.rs:369-411` | Efficient STFT computation |
| `src/paraformer.rs:55-108` | Clean config struct |
| `src/paraformer.rs:1286-1298` | Weight loading helpers |
| `src/error.rs` | thiserror usage |
| `src/lib.rs:62-110` | Vocabulary struct |
| `src/lib.rs:121-138` | High-level API |

### Code to Port

1. **MelFrontend** → Adapt for FunASR-Nano's audio processing
2. **Error enum** → Adopt thiserror pattern
3. **Config validation** → Add dimension checks
4. **Weight helpers** → Simplify loading code
5. **Vocabulary** → Better token handling
