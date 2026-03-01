//! ECAPA-TDNN Speaker Encoder for voice cloning.
//!
//! Extracts a speaker embedding from reference audio for the Base model's voice clone mode.
//! Architecture: TDNN → 3× SE-Res2Net → MFA → ASP → FC
//!
//! Weight keys: `speaker_encoder.blocks.{0-3}.*`, `speaker_encoder.mfa.*`,
//! `speaker_encoder.asp.*`, `speaker_encoder.fc.*`

use std::collections::HashMap;

use mlx_rs::{array, Array};
use mlx_rs::module::{Module, Param};
use mlx_rs::nn;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;

use crate::error::{Error, Result};

// ============================================================================
// Configuration
// ============================================================================

/// Speaker encoder mel spectrogram config.
pub struct SpeakerMelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_mels: usize,
    pub fmin: f32,
    pub fmax: f32,
}

impl Default for SpeakerMelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            n_fft: 1024,
            hop_length: 256,
            win_length: 1024,
            n_mels: 128,
            fmin: 0.0,
            fmax: 12000.0,
        }
    }
}

/// Speaker encoder architecture config (from config.json `speaker_encoder_config`).
#[derive(Debug, Clone)]
pub struct SpeakerEncoderConfig {
    pub mel_dim: i32,
    pub enc_dim: i32,
    pub enc_channels: Vec<i32>,
    pub enc_kernel_sizes: Vec<i32>,
    pub enc_dilations: Vec<i32>,
    pub enc_attention_channels: i32,
    pub enc_res2net_scale: i32,
    pub enc_se_channels: i32,
}

impl SpeakerEncoderConfig {
    /// Default config for 1.7B Base model (enc_dim=2048).
    pub fn default_1_7b() -> Self {
        Self {
            mel_dim: 128,
            enc_dim: 2048,
            enc_channels: vec![512, 512, 512, 512, 1536],
            enc_kernel_sizes: vec![5, 3, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 4, 1],
            enc_attention_channels: 128,
            enc_res2net_scale: 8,
            enc_se_channels: 128,
        }
    }

    /// Default config for 0.6B Base model (enc_dim=1024).
    pub fn default_0_6b() -> Self {
        Self {
            mel_dim: 128,
            enc_dim: 1024,
            enc_channels: vec![512, 512, 512, 512, 1536],
            enc_kernel_sizes: vec![5, 3, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 4, 1],
            enc_attention_channels: 128,
            enc_res2net_scale: 8,
            enc_se_channels: 128,
        }
    }

    /// Infer config from enc_dim value in config.json.
    pub fn from_enc_dim(enc_dim: i32) -> Self {
        if enc_dim <= 1024 {
            Self::default_0_6b()
        } else {
            Self::default_1_7b()
        }
    }
}

// ============================================================================
// TDNN Block (Conv1d + ReLU)
// ============================================================================

struct TdnnBlock {
    conv: nn::Conv1d,
}

impl TdnnBlock {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        // x: [B, T, C] (MLX Conv1d format: NLC)
        let y = self.conv.forward(x)?;
        Ok(ops::maximum(&y, &array!(0.0f32))?) // ReLU
    }
}

// ============================================================================
// Res2Net Block
// ============================================================================

/// Res2Net: splits channels into `scale` chunks, each processed sequentially
/// through its own TDNN. Chunk 0 passes through unchanged, chunk i (i>=2) gets
/// input = chunk_i + output_{i-1} before its TDNN.
struct Res2NetBlock {
    blocks: Vec<TdnnBlock>, // scale - 1 blocks (for chunks 1..scale)
    scale: i32,
    chunk_size: i32,
}

impl Res2NetBlock {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        // x: [B, T, C]  (NLC format)
        // Split along channel dimension into `scale` chunks
        let mut chunks: Vec<Array> = Vec::with_capacity(self.scale as usize);
        for s in 0..self.scale {
            let start = s * self.chunk_size;
            let end = start + self.chunk_size;
            // Index [B, T, start..end]
            let chunk = x.index((.., .., start..end));
            chunks.push(chunk);
        }

        // Process chunks
        let mut outputs: Vec<Array> = Vec::with_capacity(self.scale as usize);

        // Chunk 0: pass through unchanged
        outputs.push(chunks[0].clone());

        // Chunks 1..scale: each has its own TDNN
        for i in 1..self.scale as usize {
            let input = if i >= 2 {
                // chunk_i + output_{i-1}
                chunks[i].add(&outputs[i - 1])?
            } else {
                chunks[i].clone()
            };
            let out = self.blocks[i - 1].forward(&input)?;
            outputs.push(out);
        }

        // Concatenate all chunks back along channel dim
        let refs: Vec<&Array> = outputs.iter().collect();
        Ok(ops::concatenate_axis(&refs, 2)?) // concat along C (axis 2 in NLC)
    }
}

// ============================================================================
// Squeeze-and-Excitation Block
// ============================================================================

/// SE block: global avg pool → conv1 → ReLU → conv2 → Sigmoid → multiply
struct SeBlock {
    conv1: nn::Conv1d, // C → se_channels, k=1
    conv2: nn::Conv1d, // se_channels → C, k=1
}

impl SeBlock {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        // x: [B, T, C]
        // Global average pooling over time: [B, 1, C]
        let pooled = ops::mean_axis(x, 1, true)?;

        // SE path
        let y = self.conv1.forward(&pooled)?;
        let y = ops::maximum(&y, &array!(0.0f32))?; // ReLU
        let y = self.conv2.forward(&y)?;

        // Sigmoid
        let y = ops::sigmoid(&y)?;

        // Scale input
        Ok(x.multiply(&y)?)
    }
}

// ============================================================================
// SE-Res2Net Block
// ============================================================================

/// Full SE-Res2Net block: TDNN1 → Res2Net → TDNN2 → SE → residual add
struct SeRes2NetBlock {
    tdnn1: TdnnBlock,
    res2net_block: Res2NetBlock,
    tdnn2: TdnnBlock,
    se_block: SeBlock,
}

impl SeRes2NetBlock {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        let residual = x.clone();
        let y = self.tdnn1.forward(x)?;
        let y = self.res2net_block.forward(&y)?;
        let y = self.tdnn2.forward(&y)?;
        let y = self.se_block.forward(&y)?;
        Ok(y.add(&residual)?)
    }
}

// ============================================================================
// Attentive Statistics Pooling (ASP)
// ============================================================================

/// ASP: computes attention-weighted mean and std over time dimension.
/// Output: [B, 1, 2*C]
struct AttentiveStatisticsPooling {
    tdnn: TdnnBlock, // 3*C → attn_channels, k=1
    conv: nn::Conv1d, // attn_channels → C, k=1
}

impl AttentiveStatisticsPooling {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        // x: [B, T, C]
        let _b = x.dim(0) as i32;
        let t = x.dim(1) as i32;
        let _c = x.dim(2) as i32;

        // Compute mean and std over time
        let mean = ops::mean_axis(x, 1, true)?; // [B, 1, C]
        // Broadcast mean to [B, T, C]
        let mean_broadcast = ops::broadcast_to(
            &mean,
            &[x.dim(0) as i32, t, x.dim(2) as i32],
        )?;

        // Std: sqrt(mean((x - mean)^2))
        let diff = x.subtract(&mean_broadcast)?;
        let var = ops::mean_axis(&diff.multiply(&diff)?, 1, true)?;
        let std = ops::sqrt(&var.add(&array!(1e-5f32))?)?;
        let std_broadcast = ops::broadcast_to(
            &std,
            &[x.dim(0) as i32, t, x.dim(2) as i32],
        )?;

        // Concat [x, mean, std] along channel: [B, T, 3*C]
        let cat = ops::concatenate_axis(&[x, &mean_broadcast, &std_broadcast], 2)?;

        // Attention: TDNN(3*C → attn_ch) + Tanh + Conv(attn_ch → C) + Softmax
        let attn = self.tdnn.forward(&cat)?;
        let attn = ops::tanh(&attn)?;
        let attn = self.conv.forward(&attn)?; // [B, T, C]
        let attn = ops::softmax_axis(&attn, 1, None::<bool>)?; // softmax over T

        // Weighted mean: sum(x * attn, dim=T)
        let weighted = x.multiply(&attn)?;
        let w_mean = ops::sum_axis(&weighted, 1, true)?; // [B, 1, C]

        // Weighted std: sqrt(sum((x - w_mean)^2 * attn, dim=T))
        let w_mean_broadcast = ops::broadcast_to(
            &w_mean,
            &[x.dim(0) as i32, t, x.dim(2) as i32],
        )?;
        let diff2 = x.subtract(&w_mean_broadcast)?;
        let w_var = ops::sum_axis(
            &diff2.multiply(&diff2)?.multiply(&attn)?,
            1,
            true,
        )?;
        let w_std = ops::sqrt(&w_var.add(&array!(1e-5f32))?)?;

        // Output: cat([w_mean, w_std], channel) → [B, 1, 2*C]
        Ok(ops::concatenate_axis(&[&w_mean, &w_std], 2)?)
    }
}

// ============================================================================
// Full ECAPA-TDNN Speaker Encoder
// ============================================================================

/// ECAPA-TDNN speaker encoder.
/// Input: mel spectrogram [B, T, n_mels]
/// Output: speaker embedding [B, enc_dim]
pub struct SpeakerEncoder {
    initial_tdnn: TdnnBlock,           // blocks.0: mel_dim → enc_channels[0]
    se_res2net_blocks: Vec<SeRes2NetBlock>, // blocks.1-3
    mfa: TdnnBlock,                    // Multi-feature aggregation
    asp: AttentiveStatisticsPooling,
    fc: nn::Conv1d,                    // 2*enc_channels[4] → enc_dim, k=1
    fc_bias: Option<Array>,
    enc_dim: i32,
}

impl SpeakerEncoder {
    /// Compute speaker embedding from mel spectrogram.
    /// Input: mel [B, T, n_mels] (log mel spectrogram)
    /// Output: speaker embedding [B, enc_dim]
    pub fn forward(&mut self, mel: &Array) -> Result<Array> {
        // Initial TDNN: [B, T, n_mels] → [B, T', channels]
        let mut x = self.initial_tdnn.forward(mel)?;

        // SE-Res2Net blocks, collecting outputs for MFA
        let mut block_outputs: Vec<Array> = Vec::with_capacity(3);
        for block in self.se_res2net_blocks.iter_mut() {
            x = block.forward(&x)?;
            block_outputs.push(x.clone());
        }

        // MFA: concatenate block outputs along channel dim, then 1x1 conv
        let refs: Vec<&Array> = block_outputs.iter().collect();
        let cat = ops::concatenate_axis(&refs, 2)?; // [B, T, 3*channels]
        let mfa_out = self.mfa.forward(&cat)?;

        // Attentive Statistics Pooling: [B, T, C] → [B, 1, 2*C]
        let pooled = self.asp.forward(&mfa_out)?;

        // FC: [B, 1, 2*C] → [B, 1, enc_dim]
        let mut y = self.fc.forward(&pooled)?;
        if let Some(ref bias) = self.fc_bias {
            y = y.add(bias)?;
        }

        // Squeeze time dim: [B, 1, enc_dim] → [B, enc_dim]
        let shape = y.shape();
        let out = y.reshape(&[shape[0] as i32, self.enc_dim])?;
        mlx_rs::transforms::eval(std::iter::once(&out))?;

        Ok(out)
    }
}

// ============================================================================
// Mel Spectrogram Computation
// ============================================================================

/// Compute log mel spectrogram for speaker encoder input.
/// Input: audio samples at 24kHz, f32 in [-1, 1]
/// Output: [1, T, 128] log mel spectrogram
///
/// Matches Python: F.pad(y, (pad, pad), mode="reflect") + torch.stft(center=False)
/// where pad = (n_fft - hop_length) / 2.
pub fn compute_speaker_mel(samples: &[f32], config: &SpeakerMelConfig) -> Result<Array> {
    let n_fft = config.n_fft;
    let hop_length = config.hop_length;
    let win_length = config.win_length;
    let n_mels = config.n_mels;

    // Reflect padding: (n_fft - hop_length) / 2 on each side, matching Python's
    // F.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
    let pad = (n_fft - hop_length) / 2; // = 384 for n_fft=1024, hop=256
    let n = samples.len();
    if n < pad + 1 {
        return Err(Error::Model("Audio too short for speaker encoder".into()));
    }
    let padded_len = n + 2 * pad;
    let mut padded = vec![0.0f32; padded_len];

    // Left reflect: padded[0..pad] = samples[pad], samples[pad-1], ..., samples[1]
    for i in 0..pad {
        padded[i] = samples[pad - i];
    }
    // Copy original
    padded[pad..pad + n].copy_from_slice(samples);
    // Right reflect: padded[pad+n..] = samples[n-2], samples[n-3], ..., samples[n-1-pad]
    for i in 0..pad {
        padded[pad + n + i] = samples[n - 2 - i];
    }

    // Hann window
    let window: Vec<f32> = (0..win_length)
        .map(|i| {
            let t = i as f32 / (win_length - 1) as f32;
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * t).cos()
        })
        .collect();

    // STFT on padded signal (center=False): compute magnitude spectrum
    let n_freqs = n_fft / 2 + 1;
    let n_frames = (padded_len.saturating_sub(win_length)) / hop_length + 1;
    if n_frames == 0 {
        return Err(Error::Model("Audio too short for speaker encoder".into()));
    }

    let mut magnitudes = vec![0.0f32; n_frames * n_freqs];

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;

        // Apply window to padded signal
        let mut windowed = vec![0.0f32; n_fft];
        for i in 0..win_length {
            windowed[i] = padded[start + i] * window[i];
        }

        // Real FFT (magnitude)
        for k in 0..n_freqs {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for fft_n in 0..n_fft {
                let angle = -2.0 * std::f64::consts::PI * k as f64 * fft_n as f64 / n_fft as f64;
                re += windowed[fft_n] as f64 * angle.cos();
                im += windowed[fft_n] as f64 * angle.sin();
            }
            // Magnitude: sqrt(re^2 + im^2 + eps)
            magnitudes[frame_idx * n_freqs + k] = ((re * re + im * im + 1e-9).sqrt()) as f32;
        }
    }

    // Mel filterbank (Slaney normalization)
    let filterbank = slaney_mel_filterbank(n_fft as i32, n_mels as i32, config.sample_rate as i32, config.fmin, config.fmax);

    // Apply mel filterbank: [n_frames, n_freqs] × [n_mels, n_freqs]^T → [n_frames, n_mels]
    let mut mel_spec = vec![0.0f32; n_frames * n_mels];
    for t in 0..n_frames {
        for m in 0..n_mels {
            let mut sum = 0.0f32;
            for k in 0..n_freqs {
                sum += magnitudes[t * n_freqs + k] * filterbank[m * n_freqs + k];
            }
            mel_spec[t * n_mels + m] = sum;
        }
    }

    // Log mel: log(clamp(mel, 1e-5))
    for v in &mut mel_spec {
        *v = (*v).max(1e-5).ln();
    }

    // Convert to Array [1, n_frames, n_mels]
    let arr = Array::from_slice(&mel_spec, &[n_frames as i32, n_mels as i32]);
    let arr = arr.reshape(&[1, n_frames as i32, n_mels as i32])?;
    Ok(arr)
}

/// Slaney-normalized mel filterbank.
fn slaney_mel_filterbank(n_fft: i32, n_mels: i32, sample_rate: i32, fmin: f32, fmax: f32) -> Vec<f32> {
    let n_freqs = (n_fft / 2 + 1) as usize;

    // Hz to mel (Slaney: linear below 1000Hz, log above)
    let hz_to_mel = |hz: f32| -> f32 {
        if hz < 1000.0 {
            hz * 3.0 / 200.0
        } else {
            15.0 + (hz / 1000.0).ln() / (6400.0f32 / 1000.0).ln() * 27.0
        }
    };
    let mel_to_hz = |mel: f32| -> f32 {
        if mel < 15.0 {
            mel * 200.0 / 3.0
        } else {
            1000.0 * ((mel - 15.0) / 27.0 * (6400.0f32 / 1000.0).ln()).exp()
        }
    };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // Mel points: n_mels + 2
    let mut mel_points = Vec::with_capacity(n_mels as usize + 2);
    for i in 0..=(n_mels + 1) as usize {
        let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
        mel_points.push(mel_to_hz(mel));
    }

    // FFT frequencies
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
        .collect();

    // Create filterbank with Slaney normalization
    let mut filterbank = vec![0.0f32; n_mels as usize * n_freqs];

    for m in 0..n_mels as usize {
        let f_left = mel_points[m];
        let f_center = mel_points[m + 1];
        let f_right = mel_points[m + 2];

        // Slaney normalization factor: 2 / (f_right - f_left)
        let enorm = 2.0 / (f_right - f_left);

        for k in 0..n_freqs {
            let freq = fft_freqs[k];
            let val = if freq >= f_left && freq <= f_center {
                (freq - f_left) / (f_center - f_left)
            } else if freq > f_center && freq <= f_right {
                (f_right - freq) / (f_right - f_center)
            } else {
                0.0
            };
            filterbank[m * n_freqs + k] = val * enorm;
        }
    }

    filterbank
}

// ============================================================================
// Weight Loading
// ============================================================================

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

/// Load a Conv1d from weight keys. MLX Conv1d weight shape: [out, kernel, in] (NLC format).
/// PyTorch saves [out, in, kernel], so we transpose only when needed.
/// `needs_transpose`: true if weights are in PyTorch format and need transposing.
fn load_conv1d(
    weights: &HashMap<String, Array>,
    prefix: &str,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
    needs_transpose: bool,
) -> Result<(nn::Conv1d, Option<Array>)> {
    let weight_key = format!("{prefix}.weight");
    let bias_key = format!("{prefix}.bias");

    let weight = get_weight(weights, &weight_key)?;
    let bias = weights.get(&bias_key).cloned();

    // Only transpose if weights are in PyTorch format [out, in, kernel]
    // MLX format is already [out, kernel, in]
    let weight = if needs_transpose {
        weight.transpose_axes(&[0, 2, 1])?
    } else {
        weight
    };

    let conv = nn::Conv1d {
        weight: Param::new(weight),
        bias: Param::new(bias.clone()),
        stride,
        padding,
        dilation,
        groups,
    };

    Ok((conv, None)) // bias already in conv
}

/// Load a TDNN block (Conv1d + ReLU).
fn load_tdnn(
    weights: &HashMap<String, Array>,
    prefix: &str,
    kernel_size: i32,
    dilation: i32,
    needs_transpose: bool,
) -> Result<TdnnBlock> {
    let padding = ((kernel_size - 1) * dilation) / 2; // same padding
    let (conv, _) = load_conv1d(weights, &format!("{prefix}.conv"), 1, padding, dilation, 1, needs_transpose)?;
    Ok(TdnnBlock { conv })
}

/// Load a Res2Net block.
fn load_res2net(
    weights: &HashMap<String, Array>,
    prefix: &str,
    channels: i32,
    kernel_size: i32,
    dilation: i32,
    scale: i32,
    needs_transpose: bool,
) -> Result<Res2NetBlock> {
    let chunk_size = channels / scale;
    let padding = ((kernel_size - 1) * dilation) / 2;

    let mut blocks = Vec::with_capacity((scale - 1) as usize);
    for i in 0..(scale - 1) {
        let (conv, _) = load_conv1d(
            weights,
            &format!("{prefix}.blocks.{i}.conv"),
            1,
            padding,
            dilation,
            1,
            needs_transpose,
        )?;
        blocks.push(TdnnBlock { conv });
    }

    Ok(Res2NetBlock {
        blocks,
        scale,
        chunk_size,
    })
}

/// Load an SE block.
fn load_se_block(
    weights: &HashMap<String, Array>,
    prefix: &str,
    needs_transpose: bool,
) -> Result<SeBlock> {
    let (conv1, _) = load_conv1d(weights, &format!("{prefix}.conv1"), 1, 0, 1, 1, needs_transpose)?;
    let (conv2, _) = load_conv1d(weights, &format!("{prefix}.conv2"), 1, 0, 1, 1, needs_transpose)?;
    Ok(SeBlock { conv1, conv2 })
}

/// Load a SE-Res2Net block.
fn load_se_res2net_block(
    weights: &HashMap<String, Array>,
    prefix: &str,
    channels: i32,
    kernel_size: i32,
    dilation: i32,
    scale: i32,
    needs_transpose: bool,
) -> Result<SeRes2NetBlock> {
    let tdnn1 = load_tdnn(weights, &format!("{prefix}.tdnn1"), 1, 1, needs_transpose)?;
    let res2net_block = load_res2net(
        weights,
        &format!("{prefix}.res2net_block"),
        channels,
        kernel_size,
        dilation,
        scale,
        needs_transpose,
    )?;
    let tdnn2 = load_tdnn(weights, &format!("{prefix}.tdnn2"), 1, 1, needs_transpose)?;
    let se_block = load_se_block(weights, &format!("{prefix}.se_block"), needs_transpose)?;

    Ok(SeRes2NetBlock {
        tdnn1,
        res2net_block,
        tdnn2,
        se_block,
    })
}

/// Load the full ECAPA-TDNN speaker encoder from weight map.
/// Weight keys should have the `speaker_encoder.` prefix already stripped.
pub fn load_speaker_encoder(
    weights: &HashMap<String, Array>,
    config: &SpeakerEncoderConfig,
) -> Result<SpeakerEncoder> {
    let prefix = "speaker_encoder";

    // Detect weight format: PyTorch [out, in, kernel] vs MLX [out, kernel, in]
    // Check initial conv (kernel_size=5, mel_dim=128): unambiguous since 5 ≠ 128
    let test_key = format!("{prefix}.blocks.0.conv.weight");
    let test_weight = get_weight(weights, &test_key)?;
    let shape = test_weight.shape();
    // PyTorch: [512, 128, 5] → shape[2]=5 < shape[1]=128 → needs transpose
    // MLX:     [512, 5, 128] → shape[2]=128 > shape[1]=5 → already correct
    let needs_transpose = shape[2] < shape[1];

    // blocks.0: initial TDNN (mel_dim → enc_channels[0], k=5, d=1)
    let padding0 = ((config.enc_kernel_sizes[0] - 1) * config.enc_dilations[0]) / 2;
    let (initial_conv, _) = load_conv1d(
        weights,
        &format!("{prefix}.blocks.0.conv"),
        1,
        padding0,
        config.enc_dilations[0],
        1,
        needs_transpose,
    )?;
    let initial_tdnn = TdnnBlock { conv: initial_conv };

    // blocks.1-3: SE-Res2Net blocks
    let mut se_res2net_blocks = Vec::with_capacity(3);
    for i in 1..=3 {
        let block = load_se_res2net_block(
            weights,
            &format!("{prefix}.blocks.{i}"),
            config.enc_channels[i],
            config.enc_kernel_sizes[i],
            config.enc_dilations[i],
            config.enc_res2net_scale,
            needs_transpose,
        )?;
        se_res2net_blocks.push(block);
    }

    // MFA: 1x1 conv on concatenated block outputs (3*512=1536 → 1536)
    let (mfa_conv, _) = load_conv1d(
        weights,
        &format!("{prefix}.mfa.conv"),
        1,
        0,
        1,
        1,
        needs_transpose,
    )?;
    let mfa = TdnnBlock { conv: mfa_conv };

    // ASP: Attentive Statistics Pooling
    // TDNN: 3*1536=4608 → attn_channels
    let asp_tdnn = load_tdnn(weights, &format!("{prefix}.asp.tdnn"), 1, 1, needs_transpose)?;
    let (asp_conv, _) = load_conv1d(
        weights,
        &format!("{prefix}.asp.conv"),
        1,
        0,
        1,
        1,
        needs_transpose,
    )?;
    let asp = AttentiveStatisticsPooling {
        tdnn: asp_tdnn,
        conv: asp_conv,
    };

    // FC: 2*1536=3072 → enc_dim, k=1
    // Note: load_conv1d already includes bias in the Conv1d, no separate fc_bias needed
    let (fc, _) = load_conv1d(
        weights,
        &format!("{prefix}.fc"),
        1,
        0,
        1,
        1,
        needs_transpose,
    )?;

    Ok(SpeakerEncoder {
        initial_tdnn,
        se_res2net_blocks,
        mfa,
        asp,
        fc,
        fc_bias: None, // bias is already in Conv1d
        enc_dim: config.enc_dim,
    })
}

/// Check if speaker encoder weights are present in the weight map.
pub fn has_speaker_encoder_weights(weights: &HashMap<String, Array>) -> bool {
    weights.keys().any(|k| k.starts_with("speaker_encoder."))
}
