# Qwen3-TTS-MLX

High-performance Qwen3-TTS inference on Apple Silicon in pure Rust, powered by [MLX](https://github.com/ml-explore/mlx).

Part of the [OminiX-MLX](https://github.com/OminiX-ai/OminiX-MLX) ecosystem.

## Highlights

- **~2.3x realtime** on Apple Silicon (M1/M2/M3/M4)
- **9 preset speakers** across 12 languages (CustomVoice model)
- **Voice cloning** from a short reference audio clip (Base model)
- **Streaming** audio output — start playback before generation finishes
- **Deterministic** generation with seed control
- **8-bit quantized** — 1.7B parameter model fits comfortably in unified memory
- **Zero Python dependencies** at inference time

## Table of Contents

- [Quick Start](#quick-start)
- [Models](#models)
- [Synthesis Modes](#synthesis-modes)
  - [CustomVoice (Preset Speakers)](#customvoice-preset-speakers)
  - [Speaker + Instruct (Emotion/Style Control)](#speaker--instruct-emotionstyle-control)
  - [Voice Cloning](#voice-cloning)
  - [VoiceDesign (Text-Described Voices)](#voicedesign-text-described-voices)
  - [Streaming](#streaming)
  - [Speed Control](#speed-control)
- [API Reference](#api-reference)
  - [Synthesizer](#synthesizer)
  - [SynthesizeOptions](#synthesizeoptions)
  - [SynthesisTiming](#synthesistiming)
  - [StreamingSession](#streamingsession)
  - [Utility Functions](#utility-functions)
- [CLI Reference](#cli-reference)
- [Architecture](#architecture)
  - [Talker (28-layer Qwen3 Transformer)](#talker)
  - [Code Predictor (5-layer Sub-Transformer)](#code-predictor)
  - [Speech Tokenizer Decoder (ConvNet)](#speech-tokenizer-decoder)
  - [Speaker Encoder (ECAPA-TDNN)](#speaker-encoder-ecapa-tdnn)
  - [Speech Encoder (Mimi)](#speech-encoder-mimi)
- [Generation Parameters](#generation-parameters)
- [Supported Speakers](#supported-speakers)
- [Supported Languages](#supported-languages)
- [Performance](#performance)
- [Known Limitations](#known-limitations)
- [Building from Source](#building-from-source)

## Quick Start

```rust
use qwen3_tts_mlx::{Synthesizer, SynthesizeOptions, save_wav, normalize_audio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model
    let mut synth = Synthesizer::load("./models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")?;

    // Synthesize with a preset speaker
    let opts = SynthesizeOptions {
        speaker: "vivian",
        language: "english",
        ..Default::default()
    };
    let samples = synth.synthesize("Hello! Welcome to Qwen3 TTS.", &opts)?;

    // Save to WAV (24kHz, 16-bit PCM, mono)
    let samples = normalize_audio(&samples, 0.95);
    save_wav(&samples, synth.sample_rate, "output.wav")?;
    Ok(())
}
```

Or from the command line:

```bash
cargo run --release --example synthesize -- \
    --model-dir ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
    "Hello! Welcome to Qwen3 TTS." \
    --speaker vivian --language english \
    --output output.wav
```

## Models

| Model | Type | Size | Speakers | Voice Clone | Download |
|-------|------|------|----------|-------------|----------|
| `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | CustomVoice | ~1.8 GB | 9 presets | No | [mlx-community](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit) |
| `Qwen3-TTS-12Hz-1.7B-Base` | Base | ~3.6 GB | None | Yes (x-vector) | [QwenLM](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) |

**Model directory structure:**

```
model_dir/
├── config.json                 # Model + talker config
├── generation_config.json      # Sampling defaults
├── model.safetensors           # Talker weights (8-bit quantized or bf16)
├── model.safetensors.index.json # (optional) weight shard index
├── vocab.json                  # BPE tokenizer vocabulary
├── merges.txt                  # BPE merge rules
└── speech_tokenizer/
    ├── config.json             # Decoder architecture config
    └── model.safetensors       # Decoder weights (float32)
                                # + Mimi encoder weights (Base model only)
```

## Synthesis Modes

### CustomVoice (Preset Speakers)

Use one of 9 built-in speakers across 12 languages. Requires the CustomVoice model.

```rust
let opts = SynthesizeOptions {
    speaker: "serena",
    language: "chinese",
    temperature: Some(0.9),
    seed: Some(42), // deterministic output
    ..Default::default()
};
let samples = synth.synthesize("你好，欢迎使用语音合成。", &opts)?;
```

### Speaker + Instruct (Emotion/Style Control)

Combine a preset speaker with a natural-language style instruction. This allows emotion and delivery control while keeping a specific voice identity. Requires the CustomVoice model.

```rust
let opts = SynthesizeOptions {
    language: "chinese",
    ..Default::default()
};
let samples = synth.synthesize_with_speaker_instruct(
    "我太开心了！终于完成了！",
    "serena",                                       // preset speaker
    "用兴奋激动的语气说话，充满热情和活力",              // style instruction
    "chinese",
    &opts,
)?;
```

Or from the command line:

```bash
cargo run --release --example synthesize -- \
    -m ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
    "我太开心了！终于完成了！" \
    -s serena -l chinese \
    --instruct "用兴奋激动的语气说话，充满热情和活力"
```

**Verified emotion prompts** (Chinese — tested to produce accurate style control):

| Style | Prompt |
|-------|--------|
| Excited | `用兴奋激动的语气说话，充满热情和活力` |
| Sad | `用悲伤失望的语气说话，声音低沉，语速缓慢` |
| Cheerful | `用开朗愉快的语气说话，声音明亮上扬，节奏轻快` |
| Shout | `用大声喊叫的方式说话，声音高亢有力，语速快` |
| Sarcastic | `用讽刺嘲讽的语气说话，语调阴阳怪气，拖长尾音` |
| Soft | `用温柔轻柔的语气说话` |
| Panic | `用惊慌恐惧的语气说话，声音颤抖，语速急促` (use with `repetition_penalty: 1.3`) |

Custom free-form prompts are also supported. Include emotion + timbre + pace descriptors for strongest control. The prompt can be in any language, but Chinese prompts are best tested.

**Dispatch logic** — the method called determines the mode:

| Condition | Method | Mode |
|-----------|--------|------|
| `reference_audio` provided | `synthesize_voice_clone()` | Voice cloning (x-vector) |
| `reference_audio` + `instruct` | `synthesize_voice_clone_instruct()` | Voice cloning + emotion/style |
| `instruct` + `speaker` | `synthesize_with_speaker_instruct()` | Combined speaker + style |
| `voice_description` only | `synthesize_voice_design()` | Text-described voice |
| Default | `synthesize()` | Preset speaker |

### Voice Cloning

Clone a voice from a short reference audio clip (3-10 seconds recommended). Requires the Base model.

**x-vector mode** (recommended) — uses a speaker embedding extracted from reference audio:

```rust
// Load reference audio (any sample rate, will be resampled to 24kHz)
let (ref_samples, ref_sr) = mlx_rs_core::audio::load_wav("reference.wav")?;
let ref_samples = if ref_sr != 24000 {
    mlx_rs_core::audio::resample(&ref_samples, ref_sr, 24000)
} else {
    ref_samples
};

let opts = SynthesizeOptions {
    language: "chinese",
    ..Default::default()
};
let samples = synth.synthesize_voice_clone(
    "你好，很高兴认识你。",
    &ref_samples,
    "chinese",
    &opts,
)?;
```

**ICL mode** (experimental) — additionally uses Mimi-encoded reference audio codes + transcript for richer conditioning:

```rust
let samples = synth.synthesize_voice_clone_icl(
    "你好，很高兴认识你。",
    &ref_samples,
    "这是参考音频的文本。", // transcript of reference audio
    "chinese",
    &opts,
)?;
```

> **Note**: ICL mode is experimental and unreliable on Apple Silicon. Use x-vector mode for production.

**Voice cloning + emotion/style control** — combines x-vector cloning with an instruct prompt:

```rust
let samples = synth.synthesize_voice_clone_instruct(
    "你好，很高兴认识你。",
    &ref_samples,
    "用悲伤的语气说",  // emotion instruction
    "chinese",
    &opts,
)?;
```

> **Limitations**: Voice cloning + instruct is an experimental feature. The Base model was not
> specifically trained for this combination — emotion/style effects are weaker and less consistent
> than with preset speakers (Speaker + Instruct mode). Some emotions (sad, angry, soft) work
> reasonably well; others (fearful, surprised) may sound flat. Short, direct instructions in
> the `"用...语气说"` format work best. The upcoming **Qwen3-TTS-25Hz-VoiceEditing** model from
> Alibaba is expected to provide native support for emotion control on cloned voices.

### VoiceDesign (Text-Described Voices)

Describe the desired voice in natural language. Requires the VoiceDesign model.

```rust
let samples = synth.synthesize_voice_design(
    "Hello, this is a test.",
    "A young woman with a warm, gentle voice and a slight British accent",
    "english",
    &opts,
)?;
```

### Streaming

Generate and decode audio in chunks for low-latency playback. Each chunk contains `chunk_frames` codec frames (~83ms per frame at 12Hz).

```rust
let mut session = synth.start_streaming("Hello, this is streaming!", &opts, 10)?;

while let Some(chunk) = session.next_chunk()? {
    // Play chunk immediately — Vec<f32> at 24kHz
    play_audio(&chunk);
    eprintln!("{:.2}s generated so far", session.duration_secs());
}
```

### Speed Control

Control speech speed without changing pitch or voice character via the `speed_factor` option:

```rust
let opts = SynthesizeOptions {
    speed_factor: Some(1.2), // 20% faster
    ..Default::default()
};
let samples = synth.synthesize_voice_clone(text, &ref_audio, "chinese", &opts)?;
```

| speed_factor | Effect | Method |
|-------------|--------|--------|
| 0.5 - 0.99 | Slower speech | EOS logit steering (model generates longer output) |
| 1.0 | Normal speed | No modification |
| 1.01 - 2.0 | Faster speech | WSOLA time-stretching (post-processing) |

**How it works:**

- **Slower (< 1.0):** Biases the codec EOS logit during generation, suppressing early stopping so the model naturally extends its output with slower pacing. Based on the Segment-Aware Conditioning approach.
- **Faster (> 1.0):** Generates audio at normal speed, then applies WSOLA (Waveform Similarity Overlap-Add) time-compression. This preserves pitch and voice character while shortening the audio. No words are skipped or cut.

The `time_stretch_wsola()` function is also available as a standalone utility for post-processing any audio.

## API Reference

### Synthesizer

The main entry point for all TTS operations.

```rust
impl Synthesizer {
    /// Load model from a directory
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self>;

    /// Detected model type
    pub fn model_type(&self) -> ModelType;

    /// Available preset speaker names (CustomVoice only)
    pub fn speakers(&self) -> Vec<&str>;

    /// Available language codes
    pub fn languages(&self) -> Vec<&str>;

    /// Output sample rate (always 24000)
    pub sample_rate: u32;
}
```

**Synthesis methods** — each has a `_with_timing` variant returning `(Vec<f32>, SynthesisTiming)`:

| Method | Model Type | Description |
|--------|-----------|-------------|
| `synthesize(text, opts)` | CustomVoice | Preset speaker synthesis |
| `synthesize_with_speaker_instruct(text, speaker, instruct, lang, opts)` | CustomVoice | Speaker + emotion/style instruction |
| `synthesize_voice_design(text, instruct, language, opts)` | VoiceDesign | Text-described voice |
| `synthesize_voice_clone(text, ref_audio, language, opts)` | Base | x-vector voice cloning |
| `synthesize_voice_clone_instruct(text, ref_audio, instruct, lang, opts)` | Base | Voice cloning + emotion/style (experimental) |
| `synthesize_voice_clone_icl(text, ref_audio, ref_text, language, opts)` | Base | ICL voice cloning (disabled) |
| `start_streaming(text, opts, chunk_frames)` | CustomVoice | Streaming chunks |

**Model capability queries:**

| Method | Returns `true` for |
|--------|-------------------|
| `supports_preset_speakers()` | CustomVoice |
| `supports_voice_cloning()` | Base |
| `supports_voice_design()` | VoiceDesign |

### SynthesizeOptions

```rust
pub struct SynthesizeOptions<'a> {
    pub speaker: &'a str,              // Preset speaker name (default: "vivian")
    pub language: &'a str,             // Language code (default: "english")
    pub temperature: Option<f32>,      // Sampling temperature (config default: 0.9)
    pub top_k: Option<i32>,            // Top-k sampling (config default: 50)
    pub top_p: Option<f32>,            // Nucleus sampling threshold (config default: 1.0)
    pub max_new_tokens: Option<i32>,   // Max codec frames (config default: 8192)
    pub seed: Option<u64>,             // Random seed for deterministic generation
    pub speed_factor: Option<f32>,     // Speech speed: >1.0 faster, <1.0 slower (default: 1.0)
    pub repetition_penalty: Option<f32>, // Penalizes repeated tokens (config default: 1.05)
}
```

All `Option` fields override the values from `generation_config.json` when `Some`.

### SynthesisTiming

Returned by all `_with_timing` methods.

```rust
pub struct SynthesisTiming {
    pub prefill_ms: f64,         // Initial forward pass (embedding + first token)
    pub generation_ms: f64,      // Autoregressive codec generation
    pub generation_frames: usize, // Total codec frames generated
    pub decode_ms: f64,          // Codec-to-waveform decoding
    pub total_ms: f64,           // End-to-end wall time
}
```

### StreamingSession

```rust
impl StreamingSession<'_> {
    /// Generate next audio chunk. Returns None when done.
    pub fn next_chunk(&mut self) -> Result<Option<Vec<f32>>>;

    /// Whether generation has finished
    pub fn is_finished(&self) -> bool;

    /// Total codec frames generated so far
    pub fn total_frames(&self) -> usize;

    /// Total audio samples generated so far
    pub fn total_samples(&self) -> usize;

    /// Total audio duration in seconds
    pub fn duration_secs(&self) -> f32;
}
```

### Utility Functions

```rust
/// Save f32 samples as 16-bit PCM WAV
pub fn save_wav(samples: &[f32], sample_rate: u32, path: impl AsRef<Path>) -> Result<()>;

/// Normalize audio to target peak amplitude
pub fn normalize_audio(samples: &[f32], target_peak: f32) -> Vec<f32>;

/// WSOLA time-stretching: change tempo without changing pitch
/// speed_factor > 1.0 = faster, < 1.0 = slower
pub fn time_stretch_wsola(samples: &[f32], speed_factor: f32, sample_rate: u32) -> Vec<f32>;
```

## CLI Reference

```
qwen3-tts [OPTIONS] --model-dir <PATH> <TEXT>
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model-dir <PATH>` | `-m` | (required) | Path to model directory |
| `<TEXT>` | | (required) | Text to synthesize (positional) |
| `--output <FILE>` | `-o` | `output.wav` | Output WAV file path |
| `--speaker <NAME>` | `-s` | `vivian` | Preset speaker name |
| `--language <LANG>` | `-l` | `english` | Language code |
| `--temperature <FLOAT>` | `-t` | 0.9 | Sampling temperature |
| `--top-k <INT>` | `-k` | 50 | Top-k sampling |
| `--top-p <FLOAT>` | `-p` | 1.0 | Nucleus sampling threshold |
| `--max-tokens <INT>` | `-n` | 8192 | Max codec frames to generate |
| `--seed <UINT>` | | None | Random seed for determinism |
| `--streaming` | | off | Enable streaming mode |
| `--chunk-frames <INT>` | | 10 | Frames per streaming chunk (~83ms each) |
| `--speed <FLOAT>` | | 1.0 | Speech speed factor (0.5-2.0) |
| `--repetition-penalty <FLOAT>` | | 1.05 | Repetition penalty (1.0-2.0) |
| `--instruct <TEXT>` | | None | Style/emotion instruction (speaker+instruct mode) |
| `--reference-audio <PATH>` | | None | Reference WAV for voice cloning (Base model) |
| `--reference-text <TEXT>` | | None | Reference transcript (enables ICL mode) |

**Examples:**

```bash
# CustomVoice — Chinese with preset speaker
cargo run --release --example synthesize -- \
    -m ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
    "你好，今天天气真不错。" \
    -s uncle_fu -l chinese -o chinese_output.wav

# CustomVoice — English, deterministic
cargo run --release --example synthesize -- \
    -m ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
    "The quick brown fox jumps over the lazy dog." \
    -s ryan -l english --seed 42

# Streaming mode
cargo run --release --example synthesize -- \
    -m ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
    "This is a longer sentence to demonstrate streaming output." \
    --streaming --chunk-frames 10

# Speaker + instruct (emotion control)
cargo run --release --example synthesize -- \
    -m ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
    "我太开心了！终于完成了！" \
    -s serena -l chinese \
    --instruct "用兴奋激动的语气说话，充满热情和活力"

# Speed control (slower)
cargo run --release --example synthesize -- \
    -m ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
    "This is slower, more deliberate speech." \
    -s ryan -l english --speed 0.7

# Speed control (faster)
cargo run --release --example synthesize -- \
    -m ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
    "This is faster speech." \
    -s vivian -l english --speed 1.5

# Voice cloning (x-vector, recommended)
cargo run --release --example synthesize -- \
    -m ./models/Qwen3-TTS-12Hz-1.7B-Base \
    "你好，很高兴认识你。" \
    -l chinese --reference-audio ./reference.wav

# Voice cloning (ICL, experimental)
cargo run --release --example synthesize -- \
    -m ./models/Qwen3-TTS-12Hz-1.7B-Base \
    "你好，很高兴认识你。" \
    -l chinese --reference-audio ./reference.wav \
    --reference-text "这是参考音频的文本。"

# VoiceDesign — describe the voice
cargo run --release --example synthesize -- \
    -m ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    "Hello, this is a test." \
    -l english --instruct "A deep male voice with a calm, authoritative tone"

# Verbose logging
RUST_LOG=debug cargo run --release --example synthesize -- \
    -m ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
    "Test" -o test.wav
```

## Architecture

Qwen3-TTS uses a multi-stage pipeline: text tokenization, transformer-based codec generation, and convolutional waveform synthesis.

```
Text ──→ Tokenizer ──→ Talker (28L Qwen3) ──→ Code Predictor (5L) ──→ Decoder (ConvNet) ──→ 24kHz Audio
              │              │                       │
              │         codec book 0            codebooks 1-15
              │         (sampled)               (greedy, autoregressive)
              │
         text_projection (2-layer MLP)
         maps text embeddings into talker's hidden space
```

### Talker

The main transformer that generates codebook-0 tokens.

- **Architecture**: 28-layer Qwen3 decoder-only transformer
- **Hidden size**: 2048
- **Attention**: GQA with QK-normalization, standard RoPE (theta=1e6, stride-based rotation)
- **Activation**: SwiGLU (gate_proj * up_proj → down_proj)
- **Quantization**: 8-bit affine (group_size=64) for linear layers
- **Dual embedding streams**: text embeddings (projected via 2-layer MLP) + codec embeddings, summed per position
- **Special tokens**: `tts_pad=151671`, `tts_bos=151672`, `tts_eos=151673`, `codec_pad=2148`, `codec_bos=2149`, `codec_eos=2150`

### Code Predictor

Generates codebooks 1-15 from the talker's hidden state + codebook-0 embedding.

- **Architecture**: 5-layer transformer
- **Hidden size**: 1024 (projected from talker's 2048 via linear layer)
- **Decoding**: Always greedy (argmax), no sampling
- **Process**: 2-token prefill [projected_hidden, code0_embed], then autoregressive for codebooks 1-15

### Speech Tokenizer Decoder

Converts 16-codebook discrete codes to a 24kHz waveform.

- **Architecture**: Mimi-based ConvNet with SnakeBeta activations
- **Components**:
  - RVQ codebook embeddings (semantic + 15 acoustic)
  - Pre-transformer (8-layer, RoPE, sliding window=250)
  - Upsampling ConvTranspose1d blocks (rates: 8, 5)
  - ConvNeXt blocks with layer scale
  - SnakeBeta activation: `x + (1/beta) * sin²(alpha * x)`
- **Codec rate**: 12Hz (83.3ms per frame)
- **Output**: 24kHz mono float32 audio

### Speaker Encoder (ECAPA-TDNN)

Extracts a speaker identity embedding from reference audio. Used for voice cloning (Base model only).

- **Architecture**: ECAPA-TDNN with SE-Res2Net blocks + attentive statistics pooling
- **Input**: Log mel spectrogram (128 mels, n_fft=1024, hop=256, 24kHz)
- **Output**: Speaker embedding vector (2048-dim for 1.7B model)
- **Weights**: float32, stored in main `model.safetensors` under `speaker_encoder.*` prefix

### Speech Encoder (Mimi)

Encodes reference audio into codec frames for ICL voice cloning. Base model only.

- **Architecture**: SEANet convolutional encoder + 8-layer transformer + RVQ quantization
- **Input**: 24kHz audio samples
- **Output**: 16-codebook codec frames at 12Hz
- **Components**: Conv1d downsampling, transformer with causal mask (sliding window=250), 1 semantic + 15 acoustic codebooks
- **Weights**: float32, stored in `speech_tokenizer/model.safetensors` under `encoder.*` prefix

## Generation Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `temperature` | 0.9 | 0.0–2.0 | Controls randomness. 0.0 = greedy, higher = more varied |
| `top_k` | 50 | 1–vocab | Only sample from top-k most likely tokens |
| `top_p` | 1.0 | 0.0–1.0 | Nucleus sampling — only sample from tokens with cumulative probability ≤ p |
| `repetition_penalty` | 1.05 | 1.0–2.0 | Penalizes repeated codec tokens. Higher = less repetition |
| `max_new_tokens` | 8192 | 1+ | Maximum codec frames to generate (at 12Hz: 8192 frames ≈ 682 seconds) |
| `seed` | None | u64 | Random seed. Same seed + same input = same output |

**Token suppression**: Tokens in range [2048, 3072) except `codec_eos` (2150) are always suppressed during sampling. This prevents the model from generating invalid control tokens.

**Minimum generation**: The first 2 steps suppress EOS to prevent premature termination.

## Supported Speakers

CustomVoice model provides 9 preset speakers:

| Speaker | Language Affinity | ID |
|---------|-------------------|-----|
| `vivian` | English (default) | 3065 |
| `serena` | English | 3066 |
| `ryan` | English | 3061 |
| `aiden` | English | 2861 |
| `eric` | English | 2875 |
| `dylan` | English | 2878 |
| `uncle_fu` | Chinese | 3010 |
| `ono_anna` | Japanese | 2873 |
| `sohee` | Korean | 2864 |

All speakers can synthesize in any supported language, but quality is best when the speaker's natural language matches the target language.

## Supported Languages

| Language | Code | ID |
|----------|------|-----|
| English | `english` | 2050 |
| Chinese (Mandarin) | `chinese` | 2055 |
| Japanese | `japanese` | 2058 |
| Korean | `korean` | 2064 |
| French | `french` | 2061 |
| German | `german` | 2053 |
| Spanish | `spanish` | 2054 |
| Russian | `russian` | 2069 |
| Italian | `italian` | 2070 |
| Portuguese | `portuguese` | 2071 |
| Beijing Dialect | `beijing_dialect` | 2074 |
| Sichuan Dialect | `sichuan_dialect` | 2062 |

## Performance

Benchmarked on Apple Silicon with the 8-bit CustomVoice model:

| Stage | Typical Time | Notes |
|-------|-------------|-------|
| Model load | 1-3s | Includes tokenizer + decoder |
| Prefill | 50-150ms | 10-position batched forward pass |
| Generation | ~35ms/frame | 12Hz codec, ~28 frames/sec |
| Decode | 100-300ms | ConvNet, scales with output length |
| **Total** | | **~2.3x realtime** |

**Realtime factor** = audio_duration / total_time. A value >1.0 means faster-than-realtime synthesis.

Streaming mode adds minimal overhead per chunk — first audio chunk is available after prefill + `chunk_frames` generation steps.

## Known Limitations

1. **ICL voice cloning is unreliable on Apple Silicon**: The ICL (In-Context Learning) voice cloning mode produces inconsistent results on MPS/Metal. This appears to be a precision issue (bf16/quantized inference). **Use x-vector mode instead** — it reliably captures speaker identity. ICL methods are disabled in the current release.

2. **Voice cloning + emotion is experimental**: The `synthesize_voice_clone_instruct()` method injects instruct tokens into the voice cloning pipeline, but the Base model was not specifically trained for this combination. Emotion effects are weaker and less consistent than with preset speakers (Speaker + Instruct mode). Some emotions (sad, angry, soft) work reasonably well; others (fearful, surprised) may sound flat. Short direct prompts in the `"用...语气说"` format work best. Community parameter-tuning approaches (adjusting temperature/top_p per emotion) have negligible effect on actual prosody. The upcoming **Qwen3-TTS-25Hz-VoiceEditing** model from Alibaba is expected to provide native emotion control for cloned voices.

3. **No real-time audio playback**: The library outputs WAV files or `Vec<f32>` buffers. Integrate with an audio playback library (e.g., `cpal`, `rodio`) for real-time output.

4. **VoiceDesign model not publicly available**: The VoiceDesign synthesis path is implemented but requires the VoiceDesign model variant, which has not been officially released by Qwen as of this writing.

4. **Long text handling**: Very long texts (>100 characters) may require manual chunking for best quality. The model generates a single continuous codec sequence, so extremely long inputs may degrade.

5. **Reference audio quality matters**: For voice cloning, use clean reference audio (minimal background noise, 3-10 seconds of natural speech). Poor reference audio degrades cloning quality.

## Building from Source

**Requirements:**
- Rust 1.82.0+
- macOS with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools (for Metal compiler)

```bash
# Clone the repository
git clone https://github.com/OminiX-ai/OminiX-MLX.git
cd OminiX-MLX

# Build the TTS crate
cargo build --release --package qwen3-tts-mlx

# Run the example
cargo run --release --example synthesize -- \
    --model-dir ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
    "Hello world" --speaker vivian --language english

# Run with debug logging
RUST_LOG=debug cargo run --release --example synthesize -- \
    --model-dir ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
    "Hello world"
```

**Dependencies** (from Cargo.toml):

| Crate | Purpose |
|-------|---------|
| `mlx-rs` | Safe Rust bindings to Apple MLX |
| `mlx-sys` | Low-level FFI to MLX C API |
| `mlx-rs-core` | Shared inference utilities (KV cache, RoPE, attention, audio I/O) |
| `serde` / `serde_json` | Config deserialization |
| `tokenizers` | HuggingFace BPE tokenizer |
| `tracing` | Structured logging |
| `thiserror` | Error type derivation |

## License

MIT / Apache-2.0
