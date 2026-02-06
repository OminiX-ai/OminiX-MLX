# MLX Inference Fixes for FunASR

This document captures critical fixes discovered while debugging the funasr-qwen4b-mlx (SenseVoice + Qwen3-4B) inference pipeline. These lessons apply to all FunASR-based MLX implementations.

## Problem Summary

The trained adaptor from A100 GPU training loaded correctly but produced garbage output instead of proper Chinese transcription. The root cause was **preprocessing mismatch** between training (FunASR/PyTorch) and inference (MLX/Rust).

## Critical Fixes

### 1. Audio Preprocessing (Most Important)

**Problem**: Kaldi-style preprocessing was applied but FunASR SenseVoice training doesn't use it.

**Wrong approach** (caused mel mean = +11.3):
```rust
// DON'T DO THIS for FunASR SenseVoice
let mut audio: Vec<f32> = audio_slice.iter().map(|&s| s * 32768.0).collect();

// Dithering
for s in audio.iter_mut() {
    *s += rng.gen::<f32>() - 0.5;
}

// Pre-emphasis
for i in (1..audio.len()).rev() {
    audio[i] -= 0.97 * audio[i - 1];
}
```

**Correct approach** (mel mean = -5.87, matches training):
```rust
// FunASR SenseVoice uses normalized audio directly
// NO 32768 scaling
// NO dithering
// NO pre-emphasis
let audio: Vec<f32> = audio_slice.to_vec();
```

**Why this matters**:
- 32768 scaling adds ~17 to log mel values
- Dithering corrupts silent frames (adds random energy)
- Pre-emphasis changes spectral balance
- Combined effect: ~21 difference in mel mean, causing encoder to produce wrong features

### 2. Float16 to Float32 Conversion

**Problem**: MLX weights loaded as Float16 caused numerical drift across deep networks.

**Solution**: Convert all encoder weights to Float32 during loading:

```rust
/// Convert weight to float32 for numerical precision
fn to_f32(w: &Array) -> Array {
    if w.dtype() == mlx_rs::Dtype::Float32 {
        w.clone()
    } else {
        w.as_dtype(mlx_rs::Dtype::Float32).unwrap_or_else(|_| w.clone())
    }
}

// Apply to all weight assignments
self.after_norm_weight = Self::to_f32(&weights["after_norm.weight"]);
self.after_norm_bias = Self::to_f32(&weights["after_norm.bias"]);
// ... etc for all weights
```

**Why this matters**:
- SenseVoice has 70 transformer layers (1 encoders0 + 49 encoders + 20 tp_encoders)
- Small Float16 precision errors compound across layers
- After fix: encoder output correlates 0.999995 with PyTorch reference

### 3. Early Stopping Entropy Threshold (Generation Truncation)

**Problem**: Model produces correct characters but stops generating after only a few tokens.

**Root cause**: The entropy-based early stopping threshold was too aggressive. When the model is confident about ASR transcription (which is expected behavior), entropy drops to near 0. The old threshold (0.5) with window (5) would stop generation after just 5 tokens.

**Wrong settings** (caused truncation to ~5 tokens):
```rust
const ENTROPY_THRESHOLD: f32 = 0.5;  // Too high for ASR
const ENTROPY_WINDOW: usize = 5;     // Too short
```

**Correct settings** (allows full transcription):
```rust
const ENTROPY_THRESHOLD: f32 = 0.05;  // Only catch truly degenerate states
const ENTROPY_WINDOW: usize = 15;     // Require sustained low entropy
```

**Why this matters**:
- In ASR, low entropy = confident prediction, which is CORRECT
- The entropy check was designed to catch infinite loops/garbage
- When preprocessing is correct, the model becomes more confident
- After preprocessing fix, the overly aggressive entropy check caused truncation

**Impact**:
| Metric | Before fix | After fix |
|--------|-----------|-----------|
| AISHELL CER | 11.10% | 5.10% |
| Rust talk output | 5 chars (truncated) | 44+ chars (full sentence) |

### 4. Model Path Case Sensitivity

**Problem**: Code looked for `models/qwen3-4b` but actual directory was `models/Qwen3-4B`.

**Solution**: macOS has case-insensitive filesystem by default, so paths work. For Linux, create symlink:
```bash
ln -s Qwen3-4B qwen3-4b
```

## Mel Spectrogram Configuration

Different FunASR encoders require different mel configurations:

### SenseVoice Encoder (funasr-qwen4b-mlx)

| Parameter | Value |
|-----------|-------|
| Sample rate | 16000 Hz |
| N FFT | 512 |
| Hop length | 160 (10ms) |
| Window length | 400 (25ms) |
| Window type | **Hamming** |
| N mels | 80 |
| Frequency range | 0 - 8000 Hz |
| Power | 1.0 (magnitude) |
| Log offset | 1e-6 |

### Whisper Encoder (funasr-nano-mlx)

| Parameter | Value |
|-----------|-------|
| Sample rate | 16000 Hz |
| N FFT | 400 |
| Hop length | 160 (10ms) |
| Window length | 400 (25ms) |
| Window type | **Hann** |
| N mels | 80 |
| Frequency range | 0 - 8000 Hz |
| Power | 1.0 (magnitude) |
| Log offset | 1e-10 |

**Key differences:**
- SenseVoice uses Hamming window, Whisper uses Hann window
- SenseVoice uses n_fft=512, Whisper uses n_fft=400
- Different log offsets (1e-6 vs 1e-10)

## LFR (Low Frame Rate) Transform

FunASR uses LFR to reduce sequence length:

| Parameter | Value |
|-----------|-------|
| LFR M (stack) | 7 |
| LFR N (stride) | 6 |

This means 7 consecutive frames are stacked, then strided by 6, reducing sequence length by ~6x.

## Debugging Checklist

When FunASR MLX inference produces garbage:

1. **Check mel spectrogram statistics**
   - Expected mean: approximately -5 to -10 (varies by audio)
   - If mean is positive (+10 or higher), preprocessing is wrong

2. **Compare encoder output with PyTorch reference**
   - Save encoder output: `mlx_rs::save("encoder_output.npy", &output)`
   - Load in Python and compare with FunASR PyTorch
   - Correlation should be > 0.999

3. **Verify weight dtypes**
   - Print dtype of loaded weights
   - Convert Float16 to Float32 for encoder

4. **Test text-only generation**
   - If LLM produces garbage even without audio, the LLM loading is broken
   - Verify model path exists and weights load correctly

5. **Check CMVN (if applicable)**
   - Some FunASR variants use CMVN normalization
   - Verify cmvn_stats match between training and inference

## Architecture Reference

```
Audio (16kHz, normalized [-1,1])
    │
    ▼
┌─────────────────────┐
│   Mel Spectrogram   │  80 mels, no scaling/dithering/pre-emphasis
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   LFR Transform     │  7x stack, 6 stride
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  SenseVoice Encoder │  70 layers, 512-dim output (Float32!)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Audio Adaptor     │  Projects 512 → LLM dim
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│      Qwen LLM       │  Generates text tokens
└─────────┬───────────┘
          │
          ▼
      Text Output
```

## Domain-Specific Prompts

For better transcription accuracy with domain-specific content, customize the system prompt with:

1. **Terminology glossary** - List common terms that may be misheard
2. **Context description** - Describe the domain (e.g., "Rust programming talk about trading systems")
3. **Language mixing rules** - Specify how to handle mixed Chinese/English content

Example for Rust + trading system talks:

```rust
let system_prompt = r#"你是专业的技术演讲语音转写系统。这是一场关于Rust编程语言在量化交易系统开发中应用的技术演讲。

常见术语对照（正确写法）：
- Rust相关：Rust、Cargo、crate、trait、impl、struct、enum、match、Option、Result、unwrap、clone、borrow、lifetime、async、await、tokio、FFI、unsafe、macro、workspace、Cargo.toml、lib.rs、main.rs、pub、mod、use、extern、#[derive]、Vec、HashMap、Arc、Mutex、RefCell、Box、Rc、dyn、where、Send、Sync、rlib、dylib、cdylib、staticlib
- 交易相关：量化交易、高频交易、策略、行情、下单、撮合、延迟、吞吐、回测、实盘、API、TCP/IP、UDP、FIX协议、交易所、订单簿、K线、tick数据
- 项目相关：workspace、binary、library、dependency、编译器、链接器、静态链接、动态链接

重要：英文人名必须保留英文原文，不要音译成中文：
- Alice、Bob、Josh、Richard、Mike、David、John、Steve、Alex、Chris、Tom、Jack、Ryan、Kevin、Brian、Andrew、Daniel、James、William、Michael、Matthew、Peter、Paul、George、Henry、Edward、Frank、Gary、Eric、Mark、Nick、Tony、Ben、Sam、Max、Luke、Adam、Carl、Dean、Jeff、Ken、Larry、Leo、Neil、Oscar、Patrick、Phil、Ray、Rick、Roger、Scott、Sean、Ted、Tim、Victor、Wayne、Zach
- 韩东 是中文名，保持中文

直接输出语音内容的中文文字，保留英文技术术语和英文人名的原始拼写，不要添加解释或评论。"#;
```

This helps the model:
- Correctly transcribe "FFI" instead of mishearing as similar-sounding Chinese
- Keep "Rust", "Cargo", "trait" in English
- Keep English names like "Alice", "Bob", "Josh" in English (not 艾丽斯, 鲍勃)
- Preserve Chinese names like 韩东 in Chinese

## Benchmark Results

After applying all fixes, the model achieves competitive results on AISHELL-1 test set:

| Model | CER (Character Error Rate) |
|-------|---------------------------|
| Fun-ASR (7.7B) | 3.38% |
| Fun-ASR-Nano (0.8B) | 4.22% |
| **funasr-qwen4b-mlx** | **5.10%** |
| Paraformer v2 (0.2B) | 6.23% |
| funasr-qwen4b-mlx (before fixes) | 11.10% |

**54% reduction in CER** after fixing preprocessing and entropy threshold.

## Lessons Learned

1. **Training/inference preprocessing must match exactly** - Even small differences (scaling factor, dithering) cause large output differences.

2. **Use Float32 for deep encoders** - 70+ layers amplify Float16 precision errors.

3. **Always compare intermediate outputs** - Don't just check final output. Compare mel, encoder output, adaptor output step by step.

4. **Silent audio is a good test** - If silent audio produces non-zero mel energy, something is wrong (likely dithering).

5. **Check the training code** - The reference is always the training pipeline, not documentation or other implementations.

6. **Don't be too aggressive with early stopping** - Low entropy in ASR means confidence, not degeneration. The entropy check should only catch truly stuck states (entropy < 0.05 for 15+ consecutive tokens).
