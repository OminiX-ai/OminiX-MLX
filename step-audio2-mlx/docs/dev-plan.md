# Step-Audio 2 MLX Development Plan

## Executive Summary

This plan outlines the implementation of Step-Audio 2 mini in MLX (Rust), leveraging existing components from the OminiX-MLX codebase. The implementation is structured in 4 phases, progressing from ASR-only to full speech-to-speech capabilities.

**Target**: Step-Audio 2 mini (8B parameters) with Think mode support

**Estimated Total Effort**: ~4,700 lines of new/adapted code
**Reuse Rate**: ~55% from existing codebase

---

## Existing Components to Leverage

### From `mlx-rs-core`

| Component | File | Reuse | Adaptation |
|-----------|------|-------|------------|
| KVCache | `src/cache.rs` | 100% | None |
| Sampler | `src/sampler.rs` | 100% | None |
| RoPE init | `src/utils.rs` | 100% | None |
| Attention masks | `src/utils.rs` | 100% | None |
| SDPA | `src/utils.rs` | 100% | None |
| Token generation | `src/generate/` | 100% | None |
| Audio I/O | `src/audio.rs` | 90% | Add 128-mel config |
| Mel spectrogram | `src/audio.rs` | 80% | Update for 128 mels |

### From `qwen3-mlx`

| Component | File | Reuse | Adaptation |
|-----------|------|-------|------------|
| Qwen2 Model | `src/qwen2.rs` | 85% | Scale dimensions, extend vocab |
| Qwen2 Attention | `src/qwen2.rs` | 90% | Verify bias settings |
| Qwen2 MLP | `src/qwen2.rs` | 100% | None |
| Weight loading | `src/qwen2.rs` | 80% | Update key mapping |
| Quantization | `src/qwen2.rs` | 100% | None |

### From `funasr-nano-mlx`

| Component | File | Reuse | Adaptation |
|-----------|------|-------|------------|
| WhisperEncoder | `src/whisper_encoder.rs` | 80% | Add AvgPool, update config |
| WhisperAttention | `src/whisper_encoder.rs` | 100% | None |
| WhisperMLP | `src/whisper_encoder.rs` | 100% | None |
| Model integration | `src/model.rs` | 60% | New adaptor, TTS path |
| Audio preprocessing | `src/audio.rs` | 70% | 128 mels |

### From `gpt-sovits-mlx`

| Component | File | Reuse | Adaptation |
|-----------|------|-------|------------|
| HiFi-GAN Generator | `src/models/vits.rs` | 70% | Extract, adapt upsampling |
| ResBlock | `src/models/vits.rs` | 100% | None |
| VQ Codebook | `src/models/vits.rs` | 60% | Different codebook size |

### From `flux-klein-mlx`

| Component | File | Reuse | Adaptation |
|-----------|------|-------|------------|
| FluxSampler | `src/sampler.rs` | 80% | Adapt for CosyVoice2 |
| Rectified flow | `src/sampler.rs` | 90% | Same algorithm |
| Timestep schedule | `src/sampler.rs` | 80% | Linear schedule |

---

## Project Structure

```
step-audio2-mlx/
├── Cargo.toml
├── README.md
├── docs/
│   ├── architecture.md
│   └── dev-plan.md
├── src/
│   ├── lib.rs                    # Public API
│   ├── config.rs                 # Model configurations
│   ├── error.rs                  # Error types
│   │
│   │── # Phase 1: ASR
│   ├── audio/
│   │   ├── mod.rs
│   │   ├── frontend.rs           # Mel spectrogram (128 mels)
│   │   └── io.rs                 # WAV load/save (reuse mlx-rs-core)
│   ├── encoder/
│   │   ├── mod.rs
│   │   ├── whisper.rs            # Whisper-style encoder
│   │   └── layers.rs             # Attention, MLP, AvgPool
│   ├── adaptor.rs                # Conv1d + Linear adaptor
│   ├── llm/
│   │   ├── mod.rs
│   │   ├── qwen2.rs              # Qwen2.5-7B (adapted from qwen3-mlx)
│   │   ├── attention.rs          # GQA attention
│   │   └── generate.rs           # Token generation
│   │
│   │── # Phase 2: Think Mode
│   ├── think.rs                  # Think tag handling
│   │
│   │── # Phase 3: TTS
│   ├── tts/
│   │   ├── mod.rs
│   │   ├── audio_tokens.rs       # Audio token handling (151696+)
│   │   ├── s3_tokenizer.rs       # S3Tokenizer wrapper
│   │   ├── flow_decoder.rs       # CosyVoice2 flow matching
│   │   └── vocoder.rs            # HiFi-GAN
│   │
│   │── # Phase 4: Integration
│   ├── model.rs                  # Full model integration
│   ├── pipeline.rs               # High-level inference API
│   └── tools.rs                  # Tool calling (web search)
│
├── examples/
│   ├── asr.rs                    # Speech recognition
│   ├── tts.rs                    # Text to speech
│   ├── s2st.rs                   # Speech to speech translation
│   ├── conversation.rs           # Multi-turn dialogue
│   └── think.rs                  # Think mode demo
│
└── tests/
    ├── encoder_test.rs
    ├── llm_test.rs
    ├── tts_test.rs
    └── integration_test.rs
```

---

## Phase 1: ASR Foundation

**Goal**: Speech-to-text recognition (matches Fun-ASR-Nano capability at larger scale)

**Duration**: 2-3 weeks

### 1.1 Audio Frontend

**File**: `src/audio/frontend.rs`

```rust
// Adapt from mlx-rs-core/src/audio.rs
pub struct StepAudio2AudioConfig {
    pub sample_rate: i32,      // 16000
    pub n_fft: i32,            // 400
    pub hop_length: i32,       // 160
    pub n_mels: i32,           // 128 (changed from 80)
    pub fmin: f32,             // 0.0
    pub fmax: Option<f32>,     // 8000.0
}

pub struct MelFrontend {
    config: StepAudio2AudioConfig,
    // Pre-computed mel filterbank
    mel_filters: Array,
    window: Array,
}

impl MelFrontend {
    pub fn compute_mel(&self, samples: &[f32]) -> Result<Array>;
}
```

**Tasks**:
- [ ] Copy `AudioConfig` from mlx-rs-core, modify for 128 mels
- [ ] Update mel filterbank generation for 128 bins
- [ ] Test against Python reference

**LOC**: ~100 (mostly reuse)

### 1.2 Whisper-style Audio Encoder

**File**: `src/encoder/whisper.rs`

```rust
// Adapt from funasr-nano-mlx/src/whisper_encoder.rs
pub struct StepAudio2EncoderConfig {
    pub n_mels: i32,           // 128
    pub n_ctx: i32,            // 1500
    pub n_state: i32,          // 1280
    pub n_head: i32,           // 20
    pub n_layer: i32,          // 32
}

pub struct StepAudio2Encoder {
    pub conv1: nn::Conv1d,     // (128, 1280, k=3, p=1)
    pub conv2: nn::Conv1d,     // (1280, 1280, k=3, s=2, p=1)
    pub positional_embedding: Param<Array>,
    pub blocks: Vec<EncoderLayer>,  // 32 layers
    pub avg_pooler: AvgPool1d,      // NEW: k=2, s=2
    pub ln_post: nn::LayerNorm,
}
```

**Tasks**:
- [ ] Copy `WhisperEncoder` from funasr-nano-mlx
- [ ] Add `AvgPool1d` layer (new implementation)
- [ ] Update config for 128 mels, 32 layers
- [ ] Update positional embedding shape

**LOC**: ~350 (80% reuse + AvgPool1d)

### 1.3 AvgPool1d Layer

**File**: `src/encoder/layers.rs`

```rust
// NEW: Not in mlx-rs currently
pub struct AvgPool1d {
    pub kernel_size: i32,
    pub stride: i32,
    pub padding: i32,
}

impl Module<&Array> for AvgPool1d {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        // x: [B, C, T]
        // Use unfold + mean or conv with uniform kernel
        let kernel = Array::ones(&[1, 1, self.kernel_size])?
            / self.kernel_size as f32;
        // Apply as depthwise conv
        mlx_rs::ops::conv1d(x, &kernel, self.stride, self.padding, 1, x.shape()[1])
    }
}
```

**Tasks**:
- [ ] Implement AvgPool1d using conv1d or unfold+mean
- [ ] Test output shapes match PyTorch

**LOC**: ~50

### 1.4 Audio Adaptor

**File**: `src/adaptor.rs`

```rust
// NEW: Different from funasr-nano (simpler)
pub struct StepAudio2Adaptor {
    pub conv: nn::Conv1d,      // (1280, 1280, k=3, s=2, p=1)
    pub linear1: nn::Linear,   // (1280, 2048)
    pub linear2: nn::Linear,   // (2048, 3584)
}

impl StepAudio2Adaptor {
    pub fn new(encoder_dim: i32, llm_dim: i32) -> Result<Self> {
        Ok(Self {
            conv: nn::Conv1dBuilder::new(encoder_dim, encoder_dim, 3)
                .stride(2).padding(1).build()?,
            linear1: nn::LinearBuilder::new(encoder_dim, 2048).build()?,
            linear2: nn::LinearBuilder::new(2048, llm_dim).build()?,
        })
    }
}

impl Module<&Array> for StepAudio2Adaptor {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        // x: [B, T, 1280]
        let x = x.transpose_axes(&[0, 2, 1])?;  // [B, 1280, T]
        let x = nn::gelu(&self.conv.forward(&x)?)?;
        let x = x.transpose_axes(&[0, 2, 1])?;  // [B, T/2, 1280]
        let x = nn::gelu(&self.linear1.forward(&x)?)?;
        self.linear2.forward(&x)  // [B, T/2, 3584]
    }
}
```

**Tasks**:
- [ ] Implement Conv1d + Linear adaptor
- [ ] Verify output dimensions

**LOC**: ~80

### 1.5 Qwen2.5-7B LLM

**File**: `src/llm/qwen2.rs`

```rust
// Adapt from qwen3-mlx/src/qwen2.rs
pub struct StepAudio2LLMConfig {
    pub hidden_size: i32,           // 3584
    pub intermediate_size: i32,     // 18944
    pub num_hidden_layers: i32,     // 28
    pub num_attention_heads: i32,   // 28
    pub num_key_value_heads: i32,   // 4
    pub vocab_size: i32,            // 158720 (extended)
    pub max_position_embeddings: i32, // 16384
    pub rope_theta: f32,            // 1000000.0
    pub rms_norm_eps: f32,          // 1e-6
}

pub struct StepAudio2LLM {
    pub embed_tokens: nn::Embedding,  // (158720, 3584)
    pub layers: Vec<TransformerBlock>,
    pub norm: nn::RmsNorm,
    pub lm_head: nn::Linear,  // (3584, 158720) - NOT tied
}
```

**Tasks**:
- [ ] Copy Qwen2 model from qwen3-mlx
- [ ] Update config for 3584 hidden, 158720 vocab
- [ ] Ensure `tie_word_embeddings: false` handled
- [ ] Update weight loading key mapping
- [ ] Add embedding injection for audio features

**LOC**: ~400 (85% reuse)

### 1.6 Model Integration

**File**: `src/model.rs`

```rust
pub struct StepAudio2 {
    pub encoder: StepAudio2Encoder,
    pub adaptor: StepAudio2Adaptor,
    pub llm: StepAudio2LLM,
    pub config: StepAudio2Config,
    pub tokenizer: Tokenizer,
}

impl StepAudio2 {
    pub fn load(model_dir: &Path) -> Result<Self>;

    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        let mel = self.compute_mel(audio)?;
        let encoder_out = self.encoder.forward(&mel)?;
        let adapted = self.adaptor.forward(&encoder_out)?;
        self.generate_text(&adapted)
    }

    fn generate_text(&mut self, audio_features: &Array) -> Result<String> {
        // Build prompt with audio injection
        // Generate tokens until EOS
        // Decode and return text
    }
}
```

**Tasks**:
- [ ] Implement model loading from safetensors
- [ ] Implement prompt building with audio injection
- [ ] Implement text generation loop
- [ ] Add streaming support

**LOC**: ~400

### Phase 1 Total

| Component | Lines | New | Reuse |
|-----------|-------|-----|-------|
| Audio frontend | 100 | 20% | 80% |
| Encoder | 350 | 20% | 80% |
| AvgPool1d | 50 | 100% | 0% |
| Adaptor | 80 | 70% | 30% |
| LLM | 400 | 15% | 85% |
| Integration | 400 | 40% | 60% |
| **Total** | **1380** | **~30%** | **~70%** |

---

## Phase 2: Think Mode

**Goal**: Add reasoning before response capability

**Duration**: 3-5 days

### 2.1 Think Tag Handler

**File**: `src/think.rs`

```rust
pub struct ThinkConfig {
    pub enabled: bool,
    pub think_start: String,   // "<think>\n"
    pub think_end: String,     // "\n</think>"
    pub max_think_tokens: usize,
}

pub struct ThinkOutput {
    pub thinking: Option<String>,  // Content inside <think>...</think>
    pub response_text: String,
    pub response_audio: Option<Vec<i32>>,  // Audio tokens
}

impl StepAudio2 {
    pub fn generate_with_think(
        &mut self,
        audio_features: &Array,
        config: &ThinkConfig,
    ) -> Result<ThinkOutput> {
        // 1. Start generation with "<think>\n" prefix
        // 2. Generate until "</think>" or max tokens
        // 3. Extract thinking content
        // 4. Continue generating response (text + audio)
        // 5. Return structured output
    }
}
```

**Tasks**:
- [ ] Implement think tag parsing
- [ ] Add two-phase generation (think → respond)
- [ ] Handle stop sequences
- [ ] Support streaming think output

**LOC**: ~150

### Phase 2 Total

| Component | Lines | New | Reuse |
|-----------|-------|-----|-------|
| Think handler | 150 | 100% | 0% |

---

## Phase 3: TTS Decoder

**Goal**: Full speech synthesis capability

**Duration**: 4-6 weeks

### 3.1 Audio Token Handler

**File**: `src/tts/audio_tokens.rs`

```rust
pub const AUDIO_TOKEN_START: i32 = 151696;
pub const AUDIO_TOKEN_END: i32 = 158256;
pub const AUDIO_CODEBOOK_SIZE: i32 = 6561;

pub fn is_audio_token(token_id: i32) -> bool {
    token_id >= AUDIO_TOKEN_START && token_id <= AUDIO_TOKEN_END
}

pub fn token_to_code(token_id: i32) -> i32 {
    token_id - AUDIO_TOKEN_START
}

pub fn code_to_token(code: i32) -> i32 {
    code + AUDIO_TOKEN_START
}

pub fn extract_audio_tokens(tokens: &[i32]) -> Vec<i32> {
    tokens.iter()
        .filter(|&&t| is_audio_token(t))
        .map(|&t| token_to_code(t))
        .collect()
}
```

**Tasks**:
- [ ] Define audio token constants
- [ ] Implement token/code conversion
- [ ] Extract interleaved audio tokens from generation

**LOC**: ~80

### 3.2 S3Tokenizer Wrapper

**File**: `src/tts/s3_tokenizer.rs`

```rust
// Option A: Use onnxruntime-rs
pub struct S3Tokenizer {
    session: ort::Session,
}

impl S3Tokenizer {
    pub fn load(onnx_path: &Path) -> Result<Self> {
        let session = ort::Session::builder()?
            .with_model_from_file(onnx_path)?;
        Ok(Self { session })
    }

    pub fn decode(&self, codes: &[i32]) -> Result<Array> {
        // Run ONNX inference
        // Return semantic features
    }

    pub fn encode(&self, audio: &Array) -> Result<Vec<i32>> {
        // Run ONNX inference
        // Return discrete codes
    }
}

// Option B: Port to native MLX (more work but better performance)
pub struct S3TokenizerMLX {
    encoder: S3Encoder,
    quantizer: VectorQuantizer,
}
```

**Tasks**:
- [ ] Evaluate ONNX runtime integration vs native port
- [ ] Implement encode (audio → codes)
- [ ] Implement decode (codes → features)
- [ ] Test against Python reference

**LOC**: ~300 (ONNX wrapper) or ~800 (native port)

### 3.3 Flow-Matching Decoder

**File**: `src/tts/flow_decoder.rs`

```rust
// Adapt FluxSampler from flux-klein-mlx
pub struct FlowDecoderConfig {
    pub hidden_dim: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    pub mel_dim: i32,          // 80
    pub num_steps: i32,        // 10 for inference
}

pub struct FlowEstimator {
    // UNet-like architecture
    pub encoder_blocks: Vec<EncoderBlock>,
    pub decoder_blocks: Vec<DecoderBlock>,
    pub cross_attention: Vec<CrossAttention>,
}

pub struct FlowDecoder {
    pub estimator: FlowEstimator,
    pub sampler: FlowSampler,  // Reuse from flux-klein-mlx
}

impl FlowDecoder {
    pub fn generate(
        &mut self,
        semantic_codes: &Array,
        prompt_features: Option<&Array>,  // For voice cloning
        num_steps: i32,
    ) -> Result<Array> {
        // 1. Sample from prior (Gaussian)
        let latents = self.sampler.sample_prior(&shape)?;

        // 2. Iterative denoising
        let mel = self.sampler.denoise_loop(
            |x, t| self.estimator.forward(x, semantic_codes, prompt_features, t),
            latents,
            Some(num_steps),
        )?;

        Ok(mel)  // [B, 80, T]
    }
}
```

**Tasks**:
- [ ] Analyze CosyVoice2 flow model architecture
- [ ] Implement FlowEstimator (UNet-like)
- [ ] Adapt FluxSampler for audio
- [ ] Implement CFM denoising loop
- [ ] Test mel output quality

**LOC**: ~1200

### 3.4 HiFi-GAN Vocoder

**File**: `src/tts/vocoder.rs`

```rust
// Adapt from gpt-sovits-mlx/src/models/vits.rs
pub struct HiFiGANConfig {
    pub upsample_rates: Vec<i32>,         // [8, 8, 2, 2]
    pub upsample_kernel_sizes: Vec<i32>,  // [16, 16, 4, 4]
    pub resblock_kernel_sizes: Vec<i32>,  // [3, 7, 11]
    pub resblock_dilation_sizes: Vec<Vec<i32>>,
    pub num_mels: i32,                    // 80
}

pub struct HiFiGAN {
    pub conv_pre: nn::Conv1d,
    pub ups: Vec<nn::ConvTranspose1d>,
    pub resblocks: Vec<Vec<ResBlock>>,
    pub conv_post: nn::Conv1d,
}

impl HiFiGAN {
    pub fn forward(&mut self, mel: &Array) -> Result<Array> {
        // mel: [B, 80, T]
        let mut x = self.conv_pre.forward(mel)?;

        for (up, resblock_group) in self.ups.iter_mut().zip(&mut self.resblocks) {
            x = nn::leaky_relu(&x, 0.1)?;
            x = up.forward(&x)?;

            let mut xs = None;
            for resblock in resblock_group {
                let xr = resblock.forward(&x)?;
                xs = Some(match xs {
                    None => xr,
                    Some(prev) => prev.add(&xr)?,
                });
            }
            x = xs.unwrap().divide(&array!(resblock_group.len() as f32))?;
        }

        x = nn::leaky_relu(&x, 0.1)?;
        x = self.conv_post.forward(&x)?;
        mlx_rs::ops::tanh(&x)
    }
}
```

**Tasks**:
- [ ] Copy ResBlock from gpt-sovits-mlx
- [ ] Implement ConvTranspose1d if not available
- [ ] Build HiFi-GAN with correct upsample rates
- [ ] Load weights from CosyVoice2 checkpoint
- [ ] Test audio quality

**LOC**: ~600

### Phase 3 Total

| Component | Lines | New | Reuse |
|-----------|-------|-----|-------|
| Audio tokens | 80 | 100% | 0% |
| S3Tokenizer | 300 | 100% | 0% |
| Flow decoder | 1200 | 60% | 40% |
| HiFi-GAN | 600 | 30% | 70% |
| **Total** | **2180** | **~60%** | **~40%** |

---

## Phase 4: Integration & Polish

**Goal**: Production-ready inference with all features

**Duration**: 2-3 weeks

### 4.1 Full Pipeline

**File**: `src/pipeline.rs`

```rust
pub struct StepAudio2Pipeline {
    pub model: StepAudio2,
    pub tts: Option<TTSDecoder>,
    pub config: PipelineConfig,
}

pub struct PipelineConfig {
    pub enable_tts: bool,
    pub enable_think: bool,
    pub enable_tools: bool,
    pub sampling: SamplingConfig,
}

impl StepAudio2Pipeline {
    /// ASR: Speech to text
    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String>;

    /// TTS: Text to speech
    pub fn synthesize(&mut self, text: &str) -> Result<Vec<f32>>;

    /// S2ST: Speech to speech (translation)
    pub fn translate_speech(
        &mut self,
        audio: &[f32],
        target_lang: &str,
    ) -> Result<Vec<f32>>;

    /// Conversation: Multi-turn dialogue
    pub fn chat(
        &mut self,
        audio: Option<&[f32]>,
        text: Option<&str>,
    ) -> Result<ChatResponse>;

    /// Think mode
    pub fn think_and_respond(
        &mut self,
        audio: &[f32],
    ) -> Result<ThinkResponse>;
}

pub struct ChatResponse {
    pub text: String,
    pub audio: Option<Vec<f32>>,
    pub thinking: Option<String>,
}
```

**Tasks**:
- [ ] Implement unified pipeline API
- [ ] Add conversation context management
- [ ] Implement streaming output
- [ ] Add voice cloning support

**LOC**: ~400

### 4.2 Tool Calling

**File**: `src/tools.rs`

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn execute(&self, params: &serde_json::Value) -> Result<String>;
}

pub struct WebSearchTool {
    api_key: String,
}

impl Tool for WebSearchTool {
    fn name(&self) -> &str { "web_search" }
    fn execute(&self, params: &Value) -> Result<String> {
        let query = params["query"].as_str().ok_or("Missing query")?;
        // Call search API
        // Return results
    }
}

pub struct ToolManager {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolManager {
    pub fn register(&mut self, tool: Box<dyn Tool>);
    pub fn parse_tool_call(&self, output: &str) -> Option<ToolCall>;
    pub fn execute(&self, call: &ToolCall) -> Result<String>;
}
```

**Tasks**:
- [ ] Implement tool calling protocol
- [ ] Add web search tool
- [ ] Integrate with generation loop

**LOC**: ~300

### 4.3 Optimization

**Tasks**:
- [ ] Add INT4/INT8 quantization support
- [ ] Implement KV cache optimization
- [ ] Add batch inference
- [ ] Profile and optimize hot paths
- [ ] Memory optimization for 8B model

**LOC**: ~200

### Phase 4 Total

| Component | Lines | New | Reuse |
|-----------|-------|-----|-------|
| Pipeline | 400 | 50% | 50% |
| Tools | 300 | 80% | 20% |
| Optimization | 200 | 50% | 50% |
| **Total** | **900** | **~60%** | **~40%** |

---

## Summary

### Lines of Code by Phase

| Phase | Description | Lines | Weeks |
|-------|-------------|-------|-------|
| 1 | ASR Foundation | 1380 | 2-3 |
| 2 | Think Mode | 150 | 0.5 |
| 3 | TTS Decoder | 2180 | 4-6 |
| 4 | Integration | 900 | 2-3 |
| **Total** | | **4610** | **8-12** |

### Code Reuse Summary

| Source | Components | Reuse Rate |
|--------|------------|------------|
| mlx-rs-core | Cache, sampler, utils, audio | 95% |
| qwen3-mlx | Qwen2 model | 85% |
| funasr-nano-mlx | Whisper encoder, model structure | 70% |
| gpt-sovits-mlx | HiFi-GAN, ResBlocks | 70% |
| flux-klein-mlx | Flow sampler | 80% |
| **Overall** | | **~55%** |

### Dependencies

```toml
[dependencies]
mlx-rs = { path = "../mlx-rs" }
mlx-rs-core = { path = "../mlx-rs-core" }
tokenizers = "0.15"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_yaml = "0.9"

# For S3Tokenizer (Option A)
ort = "2.0"  # ONNX Runtime

# Audio processing
rubato = "0.14"  # High-quality resampling
```

### Milestones

| Milestone | Target | Deliverable |
|-----------|--------|-------------|
| M1 | Week 3 | ASR working (transcribe audio) |
| M2 | Week 4 | Think mode working |
| M3 | Week 8 | TTS working (synthesize speech) |
| M4 | Week 10 | S2ST working |
| M5 | Week 12 | Full pipeline, optimized |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Memory (8B model) | Start with INT4 quantization |
| S3Tokenizer complexity | Use ONNX runtime first, port later |
| Flow decoder accuracy | Validate against Python reference |
| Performance | Profile early, optimize critical paths |
| Weight compatibility | Test weight loading incrementally |

---

## Testing Strategy

### Unit Tests
- Encoder output shapes
- Adaptor dimensions
- LLM forward pass
- Audio token conversion

### Integration Tests
- Full ASR pipeline
- Full TTS pipeline
- S2ST end-to-end
- Think mode flow

### Quality Tests
- WER on LibriSpeech
- MOS on synthesized speech
- Compare with Python reference

---

## Getting Started

1. **Set up project structure**
   ```bash
   cd /Users/yuechen/home/OminiX-MLX/step-audio2-mlx
   cargo init --lib
   ```

2. **Start with Phase 1.5 (LLM)**
   - Copy qwen2.rs from qwen3-mlx
   - Modify config for 3584 hidden
   - Test with text-only generation

3. **Add encoder incrementally**
   - Copy whisper_encoder.rs
   - Add AvgPool1d
   - Test encoder output shapes

4. **Integrate and test ASR**
   - Connect encoder → adaptor → LLM
   - Test full transcription pipeline
