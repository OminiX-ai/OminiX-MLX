//! FunASR Paraformer Model for Chinese ASR
//!
//! This module implements the Paraformer-large (220M) model for non-autoregressive
//! Chinese speech recognition using MLX for GPU acceleration.
//!
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
//!
//! # Key Features
//!
//! - **Non-autoregressive**: Predicts all tokens in parallel (3-5x faster than Whisper)
//! - **SAN-M Attention**: Self-attention with memory enhancement (FSMN block)
//! - **CIF Mechanism**: Continuous integrate-and-fire for length prediction
//! - **GPU Accelerated**: Metal GPU via MLX for all operations

use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use rustfft::{num_complex::Complex, FftPlanner};

use mlx_rs::{
    argmax_axis, array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp, softmax_axis},
    Array,
};

use crate::error::{Error, Result};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Paraformer model
#[derive(Debug, Clone)]
pub struct ParaformerConfig {
    // Audio frontend
    /// Sample rate (must be 16000)
    pub sample_rate: i32,
    /// Number of mel bins
    pub n_mels: i32,
    /// FFT window size in samples (400 = 25ms at 16kHz)
    pub n_fft: i32,
    /// Hop length in samples (160 = 10ms at 16kHz)
    pub hop_length: i32,
    /// LFR multiply factor (stack this many frames)
    pub lfr_m: i32,
    /// LFR divide factor (subsample by this factor)
    pub lfr_n: i32,

    // Encoder
    /// Encoder hidden dimension
    pub encoder_dim: i32,
    /// Number of encoder layers
    pub encoder_layers: i32,
    /// Number of attention heads
    pub encoder_heads: i32,
    /// FFN intermediate dimension
    pub encoder_ffn_dim: i32,
    /// SAN-M kernel size
    pub sanm_kernel_size: i32,
    /// Dropout rate
    pub dropout: f32,

    // CIF Predictor
    /// CIF threshold for firing
    pub cif_threshold: f32,
    /// CIF tail threshold
    pub cif_tail_threshold: f32,
    /// CIF conv left order
    pub cif_l_order: i32,
    /// CIF conv right order
    pub cif_r_order: i32,

    // Decoder
    /// Decoder hidden dimension (same as encoder)
    pub decoder_dim: i32,
    /// Number of decoder layers
    pub decoder_layers: i32,
    /// Number of decoder attention heads
    pub decoder_heads: i32,
    /// Decoder FFN intermediate dimension
    pub decoder_ffn_dim: i32,

    // Output
    /// Vocabulary size
    pub vocab_size: i32,
}

impl Default for ParaformerConfig {
    fn default() -> Self {
        Self {
            // Audio frontend (16kHz, 80 mel, LFR 7/6)
            sample_rate: 16000,
            n_mels: 80,
            n_fft: 400,      // 25ms window
            hop_length: 160, // 10ms hop
            lfr_m: 7,        // Stack 7 frames
            lfr_n: 6,        // Subsample by 6

            // Encoder (Paraformer-large): 1 first_layer + 49 regular = 50 total
            encoder_dim: 512,
            encoder_layers: 50,
            encoder_heads: 4,
            encoder_ffn_dim: 2048,
            sanm_kernel_size: 11,
            dropout: 0.1,

            // CIF Predictor
            cif_threshold: 1.0,
            cif_tail_threshold: 0.45,
            cif_l_order: 1,
            cif_r_order: 1,

            // Decoder (16 layers)
            decoder_dim: 512,
            decoder_layers: 16,
            decoder_heads: 4,
            decoder_ffn_dim: 2048,

            // Output
            vocab_size: 8404,
        }
    }
}

// ============================================================================
// Audio Frontend
// ============================================================================

/// Mel spectrogram frontend for Paraformer
///
/// Computes 80-bin mel spectrogram with LFR (Low Frame Rate) stacking.
/// Uses FFT for efficient STFT computation (O(N log N) instead of O(N²)).
pub struct MelFrontend {
    config: ParaformerConfig,
    mel_filters: Vec<f32>,
    window: Vec<f32>,
    cmvn_addshift: Option<Vec<f32>>,
    cmvn_rescale: Option<Vec<f32>>,
    /// Cached FFT instance for efficient repeated STFT computation
    fft: Arc<dyn rustfft::Fft<f32>>,
}

impl std::fmt::Debug for MelFrontend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MelFrontend")
            .field("config", &self.config)
            .field("mel_filters_len", &self.mel_filters.len())
            .field("window_len", &self.window.len())
            .field("cmvn_addshift", &self.cmvn_addshift.is_some())
            .field("cmvn_rescale", &self.cmvn_rescale.is_some())
            .field("fft_len", &self.fft.len())
            .finish()
    }
}

impl Clone for MelFrontend {
    fn clone(&self) -> Self {
        // Re-create FFT planner for clone since Arc<dyn Fft> is not Clone
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.config.n_fft as usize);

        Self {
            config: self.config.clone(),
            mel_filters: self.mel_filters.clone(),
            window: self.window.clone(),
            cmvn_addshift: self.cmvn_addshift.clone(),
            cmvn_rescale: self.cmvn_rescale.clone(),
            fft,
        }
    }
}

impl MelFrontend {
    pub fn new(config: &ParaformerConfig) -> Self {
        let n_fft = config.n_fft as usize;
        let n_mels = config.n_mels as usize;
        let sample_rate = config.sample_rate as f32;

        // Create Hamming window
        let window: Vec<f32> = (0..n_fft)
            .map(|i| {
                let t = i as f32 / (n_fft - 1) as f32;
                0.54 - 0.46 * (2.0 * PI * t).cos()
            })
            .collect();

        let mel_filters = Self::create_mel_filterbank(n_fft, n_mels, sample_rate);

        // Pre-create FFT planner for efficient repeated use
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        Self {
            config: config.clone(),
            mel_filters,
            window,
            cmvn_addshift: None,
            cmvn_rescale: None,
            fft,
        }
    }

    /// Set CMVN normalization parameters
    pub fn set_cmvn(&mut self, addshift: Vec<f32>, rescale: Vec<f32>) {
        self.cmvn_addshift = Some(addshift);
        self.cmvn_rescale = Some(rescale);
    }

    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    fn create_mel_filterbank(n_fft: usize, n_mels: usize, sample_rate: f32) -> Vec<f32> {
        let n_freqs = n_fft / 2 + 1;
        let fmin = 0.0f32;
        let fmax = sample_rate / 2.0;

        let mel_min = Self::hz_to_mel(fmin);
        let mel_max = Self::hz_to_mel(fmax);

        let mut mel_points = Vec::with_capacity(n_mels + 2);
        for i in 0..=(n_mels + 1) {
            let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
            mel_points.push(Self::mel_to_hz(mel));
        }

        let fft_freqs: Vec<f32> = (0..n_freqs)
            .map(|i| i as f32 * sample_rate / n_fft as f32)
            .collect();

        let mut filterbank = vec![0.0f32; n_mels * n_freqs];

        for m in 0..n_mels {
            let f_left = mel_points[m];
            let f_center = mel_points[m + 1];
            let f_right = mel_points[m + 2];

            for k in 0..n_freqs {
                let freq = fft_freqs[k];
                if freq >= f_left && freq <= f_center {
                    filterbank[m * n_freqs + k] = (freq - f_left) / (f_center - f_left);
                } else if freq > f_center && freq <= f_right {
                    filterbank[m * n_freqs + k] = (f_right - freq) / (f_right - f_center);
                }
            }
        }

        filterbank
    }

    /// Compute mel spectrogram from audio samples
    pub fn forward(&self, audio: &Array) -> Result<Array> {
        let audio_data: Vec<f32> = audio
            .try_as_slice::<f32>()
            .map_err(|_| Error::Audio("Failed to get audio slice".into()))?
            .to_vec();

        if audio_data.iter().any(|x| x.is_nan() || x.is_infinite()) {
            return Err(Error::Audio("Audio contains NaN or Inf values".into()));
        }

        // Scale audio by 2^15 (FunASR/Kaldi convention)
        let audio_scaled: Vec<f32> = audio_data.iter().map(|&x| x * 32768.0).collect();

        // Apply pre-emphasis (coeff=0.97)
        let preemph_coeff = 0.97f32;
        let mut audio_preemph = Vec::with_capacity(audio_scaled.len());
        for i in 0..audio_scaled.len() {
            if i == 0 {
                audio_preemph.push(audio_scaled[i]);
            } else {
                audio_preemph.push(audio_scaled[i] - preemph_coeff * audio_scaled[i - 1]);
            }
        }

        // Compute STFT power spectrum
        let stft_mag = self.compute_stft(&audio_preemph);
        let n_freqs = (self.config.n_fft / 2 + 1) as usize;
        let n_frames = stft_mag.len() / n_freqs;

        if n_frames == 0 {
            return Err(Error::Audio("Audio too short for mel spectrogram".into()));
        }

        // Apply mel filterbank
        let n_mels = self.config.n_mels as usize;
        let mut mel_spec = vec![0.0f32; n_frames * n_mels];

        for t in 0..n_frames {
            for m in 0..n_mels {
                let mut sum = 0.0f32;
                for k in 0..n_freqs {
                    sum += stft_mag[t * n_freqs + k] * self.mel_filters[m * n_freqs + k];
                }
                mel_spec[t * n_mels + m] = (sum.max(1e-10)).ln();
            }
        }

        // Apply LFR stacking
        let lfr_m = self.config.lfr_m as usize;
        let lfr_n = self.config.lfr_n as usize;
        let left_padding = (lfr_m - 1) / 2;
        let padded_frames = n_frames + left_padding;
        let lfr_frames = (padded_frames + lfr_n - 1) / lfr_n;
        let lfr_dim = n_mels * lfr_m;

        let mut lfr_spec = vec![0.0f32; lfr_frames * lfr_dim];

        for t in 0..lfr_frames {
            let start = t * lfr_n;
            for m in 0..lfr_m {
                let padded_idx = start + m;
                let src_frame = if padded_idx < left_padding {
                    0
                } else if padded_idx - left_padding < n_frames {
                    padded_idx - left_padding
                } else {
                    n_frames - 1
                };

                for f in 0..n_mels {
                    lfr_spec[t * lfr_dim + m * n_mels + f] = mel_spec[src_frame * n_mels + f];
                }
            }
        }

        // Apply CMVN
        if let (Some(addshift), Some(rescale)) = (&self.cmvn_addshift, &self.cmvn_rescale) {
            for t in 0..lfr_frames {
                for d in 0..lfr_dim {
                    let idx = t * lfr_dim + d;
                    lfr_spec[idx] = (lfr_spec[idx] + addshift[d]) * rescale[d];
                }
            }
        }

        Ok(Array::from_slice(
            &lfr_spec,
            &[1, lfr_frames as i32, lfr_dim as i32],
        ))
    }

    /// Compute STFT using cached FFT (O(N log N) instead of O(N²) manual DFT)
    ///
    /// Performance improvement: ~45x faster than manual DFT for n_fft=400
    /// - Manual DFT: ~160,000 operations per frame
    /// - FFT: ~3,500 operations per frame
    fn compute_stft(&self, samples: &[f32]) -> Vec<f32> {
        let n_fft = self.config.n_fft as usize;
        let hop_length = self.config.hop_length as usize;
        let n_freqs = n_fft / 2 + 1;

        let n_frames = if samples.len() >= n_fft {
            (samples.len() - n_fft) / hop_length + 1
        } else {
            0
        };

        if n_frames == 0 {
            return vec![0.0f32; n_freqs];
        }

        let mut power_spec = vec![0.0f32; n_frames * n_freqs];
        let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n_fft];

        for frame in 0..n_frames {
            let start = frame * hop_length;

            // Apply window and convert to complex
            for i in 0..n_fft {
                buffer[i] = Complex::new(samples[start + i] * self.window[i], 0.0);
            }

            // Compute FFT in-place using cached FFT instance
            self.fft.process(&mut buffer);

            // Extract power spectrum (only positive frequencies)
            for k in 0..n_freqs {
                let c = buffer[k];
                power_spec[frame * n_freqs + k] = c.re * c.re + c.im * c.im;
            }
        }

        power_spec
    }
}

// ============================================================================
// Positional Encoding
// ============================================================================

fn sinusoidal_position_encoding(max_len: i32, dim: i32) -> std::result::Result<Array, Exception> {
    let half_dim = dim / 2;
    let mut pe = vec![0.0f32; (max_len * dim) as usize];

    let log_timescale_increment = 10000.0_f32.ln() / (half_dim as f32 - 1.0);

    let inv_timescales: Vec<f32> = (0..half_dim)
        .map(|i| (-(i as f32) * log_timescale_increment).exp())
        .collect();

    for pos in 0..max_len {
        let position = (pos + 1) as f32;

        for i in 0..half_dim {
            let scaled_time = position * inv_timescales[i as usize];
            pe[(pos * dim + i) as usize] = scaled_time.sin();
            pe[(pos * dim + half_dim + i) as usize] = scaled_time.cos();
        }
    }

    Ok(Array::from_slice(&pe, &[max_len, dim]))
}

// ============================================================================
// SAN-M Attention
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct SanmAttention {
    #[param]
    pub linear_q_k_v: nn::Linear,
    #[param]
    pub out_proj: nn::Linear,
    #[param]
    pub fsmn_block: nn::Conv1d,
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub input_dim: i32,
}

impl SanmAttention {
    pub fn new(
        input_dim: i32,
        dim: i32,
        num_heads: i32,
        kernel_size: i32,
    ) -> std::result::Result<Self, Exception> {
        let head_dim = dim / num_heads;
        let scale = (head_dim as f32).powf(-0.5);

        let linear_q_k_v = nn::LinearBuilder::new(input_dim, 3 * dim).bias(true).build()?;
        let out_proj = nn::LinearBuilder::new(dim, dim).bias(true).build()?;

        let padding = kernel_size / 2;
        let fsmn_block = nn::Conv1dBuilder::new(dim, dim, kernel_size)
            .stride(1)
            .padding(padding)
            .groups(dim)
            .bias(false)
            .build()?;

        Ok(Self {
            linear_q_k_v,
            out_proj,
            fsmn_block,
            num_heads,
            head_dim,
            scale,
            input_dim,
        })
    }
}

impl Module<&Array> for SanmAttention {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> std::result::Result<Self::Output, Self::Error> {
        let shape = x.shape();
        let (batch, seq_len, _dim) = (shape[0], shape[1], shape[2]);

        let qkv = self.linear_q_k_v.forward(x)?;

        let dim = self.num_heads * self.head_dim;
        let q = qkv.index((.., .., ..dim));
        let k = qkv.index((.., .., dim..2 * dim));
        let v = qkv.index((.., .., 2 * dim..));

        let q = q
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[batch, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let k_t = k.transpose_axes(&[0, 1, 3, 2])?;
        let scores = q.matmul(&k_t)?.multiply(array!(self.scale))?;
        let attn_weights = softmax_axis(&scores, -1, None)?;
        let attn_out = attn_weights.matmul(&v)?;

        let attn_out = attn_out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;

        let v_proj = qkv.index((.., .., 2 * dim..));
        let fsmn_conv = self.fsmn_block.forward(&v_proj)?;
        let fsmn_out = ops::add(&fsmn_conv, &v_proj)?;

        let attn_proj = self.out_proj.forward(&attn_out)?;
        ops::add(&attn_proj, &fsmn_out)
    }

    fn training_mode(&mut self, mode: bool) {
        self.linear_q_k_v.training_mode(mode);
        self.out_proj.training_mode(mode);
        self.fsmn_block.training_mode(mode);
    }
}

// ============================================================================
// Feed-Forward Network
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct FeedForward {
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
}

impl FeedForward {
    pub fn new(dim: i32, ffn_dim: i32) -> std::result::Result<Self, Exception> {
        let up_proj = nn::LinearBuilder::new(dim, ffn_dim).bias(true).build()?;
        let down_proj = nn::LinearBuilder::new(ffn_dim, dim).bias(true).build()?;
        Ok(Self { up_proj, down_proj })
    }
}

impl Module<&Array> for FeedForward {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> std::result::Result<Self::Output, Self::Error> {
        let h = self.up_proj.forward(x)?;
        let h = nn::relu(&h)?;
        self.down_proj.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

// ============================================================================
// SAN-M Encoder Layer
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct SanmEncoderLayer {
    #[param]
    pub self_attn: SanmAttention,
    #[param]
    pub ffn: FeedForward,
    #[param]
    pub norm1: nn::LayerNorm,
    #[param]
    pub norm2: nn::LayerNorm,
}

impl SanmEncoderLayer {
    pub fn new(
        input_dim: i32,
        dim: i32,
        config: &ParaformerConfig,
    ) -> std::result::Result<Self, Exception> {
        let self_attn =
            SanmAttention::new(input_dim, dim, config.encoder_heads, config.sanm_kernel_size)?;
        let ffn = FeedForward::new(dim, config.encoder_ffn_dim)?;
        let norm1 = nn::LayerNormBuilder::new(input_dim).eps(1e-5).build()?;
        let norm2 = nn::LayerNormBuilder::new(dim).eps(1e-5).build()?;

        Ok(Self {
            self_attn,
            ffn,
            norm1,
            norm2,
        })
    }
}

impl Module<&Array> for SanmEncoderLayer {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> std::result::Result<Self::Output, Self::Error> {
        let h = self.norm1.forward(x)?;
        let h = self.self_attn.forward(&h)?;

        let attn_input_dim = self.self_attn.input_dim;
        let attn_output_dim = self.self_attn.num_heads * self.self_attn.head_dim;

        let x = if attn_input_dim == attn_output_dim {
            ops::add(x, &h)?
        } else {
            h
        };

        let h = self.norm2.forward(&x)?;
        let h = self.ffn.forward(&h)?;
        ops::add(&x, &h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.self_attn.training_mode(mode);
        self.ffn.training_mode(mode);
        self.norm1.training_mode(mode);
        self.norm2.training_mode(mode);
    }
}

// ============================================================================
// SAN-M Encoder
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct SanmEncoder {
    #[param]
    pub first_layer: SanmEncoderLayer,
    #[param]
    pub layers: Vec<SanmEncoderLayer>,
    #[param]
    pub after_norm: nn::LayerNorm,
    pub max_len: i32,
}

impl SanmEncoder {
    pub fn new(config: &ParaformerConfig) -> std::result::Result<Self, Exception> {
        let input_dim = config.n_mels * config.lfr_m;
        let first_layer = SanmEncoderLayer::new(input_dim, config.encoder_dim, config)?;

        let num_regular_layers = config.encoder_layers - 1;
        let mut layers = Vec::with_capacity(num_regular_layers as usize);
        for _ in 0..num_regular_layers {
            layers.push(SanmEncoderLayer::new(
                config.encoder_dim,
                config.encoder_dim,
                config,
            )?);
        }

        let after_norm = nn::LayerNormBuilder::new(config.encoder_dim)
            .eps(1e-5)
            .build()?;

        Ok(Self {
            first_layer,
            layers,
            after_norm,
            max_len: 5000,
        })
    }
}

impl Module<&Array> for SanmEncoder {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> std::result::Result<Self::Output, Self::Error> {
        let shape = x.shape();
        let seq_len = shape[1];
        let input_dim = shape[2];

        let scale_factor = (512.0_f32).sqrt();
        let mut h = x.multiply(array!(scale_factor))?;

        let pe = sinusoidal_position_encoding(seq_len, input_dim)?;
        let pe = pe.reshape(&[1, seq_len, input_dim])?;
        h = ops::add(&h, &pe)?;

        h = self.first_layer.forward(&h)?;

        for layer in &mut self.layers {
            h = layer.forward(&h)?;
        }

        self.after_norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.first_layer.training_mode(mode);
        for layer in &mut self.layers {
            layer.training_mode(mode);
        }
        self.after_norm.training_mode(mode);
    }
}

// ============================================================================
// CIF Predictor
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct CIFPredictor {
    #[param]
    pub conv: nn::Conv1d,
    #[param]
    pub output_proj: nn::Linear,
    pub threshold: f32,
    pub tail_threshold: f32,
    pub l_order: i32,
    pub r_order: i32,
}

impl CIFPredictor {
    pub fn new(config: &ParaformerConfig) -> std::result::Result<Self, Exception> {
        let kernel_size = config.cif_l_order + config.cif_r_order + 1;

        if config.cif_l_order != config.cif_r_order {
            return Err(Exception::from(
                "CIF asymmetric padding (l_order != r_order) not yet supported",
            ));
        }

        let conv = nn::Conv1dBuilder::new(config.encoder_dim, config.encoder_dim, kernel_size)
            .stride(1)
            .padding(config.cif_l_order)
            .build()?;

        let output_proj = nn::LinearBuilder::new(config.encoder_dim, 1)
            .bias(true)
            .build()?;

        Ok(Self {
            conv,
            output_proj,
            threshold: config.cif_threshold,
            tail_threshold: config.cif_tail_threshold,
            l_order: config.cif_l_order,
            r_order: config.cif_r_order,
        })
    }

    fn compute_alphas(&mut self, encoder_out: &Array) -> std::result::Result<Array, Exception> {
        let h = self.conv.forward(encoder_out)?;
        let h = nn::relu(&h)?;
        let alphas = self.output_proj.forward(&h)?;
        // Squeeze the last dimension (1) but keep batch dimension
        let alphas = alphas.squeeze_axes(&[-1])?;
        ops::sigmoid(&alphas)
    }

    /// CIF fire mechanism with batch support
    ///
    /// Now supports batch_size > 1 for improved throughput.
    /// For batched input, pads output to max token count across batch.
    fn cif_fire(
        &self,
        hidden: &Array,
        alphas: &Array,
    ) -> std::result::Result<(Array, Array), Exception> {
        let shape = hidden.shape();
        let (batch, len_time, hidden_size) = (shape[0], shape[1], shape[2]);

        // Get data as contiguous slices
        let hidden_data: Vec<f32> = hidden
            .try_as_slice::<f32>()
            .map_err(|_| Exception::from("Failed to get hidden slice"))?
            .to_vec();

        // Handle potentially 2D or 1D alphas based on batch size
        let alphas_flat: Vec<f32> = alphas
            .try_as_slice::<f32>()
            .map_err(|_| Exception::from("Failed to get alphas slice"))?
            .to_vec();

        // Process each batch item
        let mut all_batch_frames: Vec<Vec<Vec<f32>>> = Vec::with_capacity(batch as usize);
        let mut token_counts: Vec<i32> = Vec::with_capacity(batch as usize);

        for b in 0..batch as usize {
            let mut integrate = 0.0f32;
            let mut frame = vec![0.0f32; hidden_size as usize];
            let mut list_frames: Vec<Vec<f32>> = Vec::new();

            for t in 0..len_time as usize {
                // Index into flattened arrays
                let alpha_idx = b * len_time as usize + t;
                let hidden_offset = b * (len_time as usize * hidden_size as usize)
                    + t * hidden_size as usize;

                let alpha = alphas_flat[alpha_idx];
                let distribution_completion = 1.0 - integrate;

                integrate += alpha;

                let fire_place = integrate >= self.threshold;
                if fire_place {
                    integrate -= 1.0;
                }

                let cur = if fire_place {
                    distribution_completion
                } else {
                    alpha
                };
                let remainds = alpha - cur;

                for d in 0..hidden_size as usize {
                    frame[d] += cur * hidden_data[hidden_offset + d];
                }

                if fire_place {
                    list_frames.push(frame.clone());
                    for d in 0..hidden_size as usize {
                        frame[d] = remainds * hidden_data[hidden_offset + d];
                    }
                }
            }

            // Handle tail
            if integrate > self.tail_threshold {
                list_frames.push(frame);
            }

            token_counts.push(list_frames.len() as i32);
            all_batch_frames.push(list_frames);
        }

        // Find max token count for padding
        let max_tokens = token_counts.iter().copied().max().unwrap_or(0) as usize;

        if max_tokens == 0 {
            return Ok((
                Array::zeros::<f32>(&[batch, 0, hidden_size])?,
                Array::from_slice(&token_counts, &[batch]),
            ));
        }

        // Create padded output array
        let mut flat_embeds = vec![0.0f32; batch as usize * max_tokens * hidden_size as usize];

        for (b, batch_frames) in all_batch_frames.into_iter().enumerate() {
            for (t, frame) in batch_frames.into_iter().enumerate() {
                let offset = b * max_tokens * hidden_size as usize + t * hidden_size as usize;
                for (d, &val) in frame.iter().enumerate() {
                    flat_embeds[offset + d] = val;
                }
            }
        }

        let embeds_array =
            Array::from_slice(&flat_embeds, &[batch, max_tokens as i32, hidden_size]);
        let token_num = Array::from_slice(&token_counts, &[batch]);

        Ok((embeds_array, token_num))
    }
}

impl Module<&Array> for CIFPredictor {
    type Output = (Array, Array);
    type Error = Exception;

    fn forward(&mut self, encoder_out: &Array) -> std::result::Result<Self::Output, Self::Error> {
        let alphas = self.compute_alphas(encoder_out)?;
        self.cif_fire(encoder_out, &alphas)
    }

    fn training_mode(&mut self, mode: bool) {
        self.conv.training_mode(mode);
        self.output_proj.training_mode(mode);
    }
}

// ============================================================================
// Decoder Layer
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct ParaformerDecoderLayer {
    #[param]
    pub self_attn_fsmn: nn::Conv1d,
    #[param]
    pub src_attn_q: nn::Linear,
    #[param]
    pub src_attn_kv: nn::Linear,
    #[param]
    pub src_attn_out: nn::Linear,
    #[param]
    pub ffn: FeedForward,
    #[param]
    pub ffn_norm: nn::LayerNorm,
    #[param]
    pub norm1: nn::LayerNorm,
    #[param]
    pub norm2: nn::LayerNorm,
    #[param]
    pub norm3: nn::LayerNorm,
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
}

impl ParaformerDecoderLayer {
    pub fn new(config: &ParaformerConfig) -> std::result::Result<Self, Exception> {
        let head_dim = config.decoder_dim / config.decoder_heads;
        let scale = (head_dim as f32).powf(-0.5);

        let padding = config.sanm_kernel_size / 2;
        let self_attn_fsmn =
            nn::Conv1dBuilder::new(config.decoder_dim, config.decoder_dim, config.sanm_kernel_size)
                .stride(1)
                .padding(padding)
                .groups(config.decoder_dim)
                .bias(false)
                .build()?;

        let src_attn_q = nn::LinearBuilder::new(config.decoder_dim, config.decoder_dim)
            .bias(true)
            .build()?;
        let src_attn_kv = nn::LinearBuilder::new(config.encoder_dim, 2 * config.decoder_dim)
            .bias(true)
            .build()?;
        let src_attn_out = nn::LinearBuilder::new(config.decoder_dim, config.decoder_dim)
            .bias(true)
            .build()?;

        let ffn = FeedForward::new(config.decoder_dim, config.decoder_ffn_dim)?;
        let ffn_norm = nn::LayerNormBuilder::new(config.decoder_ffn_dim)
            .eps(1e-5)
            .build()?;

        let norm1 = nn::LayerNormBuilder::new(config.decoder_dim)
            .eps(1e-5)
            .build()?;
        let norm2 = nn::LayerNormBuilder::new(config.decoder_dim)
            .eps(1e-5)
            .build()?;
        let norm3 = nn::LayerNormBuilder::new(config.decoder_dim)
            .eps(1e-5)
            .build()?;

        Ok(Self {
            self_attn_fsmn,
            src_attn_q,
            src_attn_kv,
            src_attn_out,
            ffn,
            ffn_norm,
            norm1,
            norm2,
            norm3,
            num_heads: config.decoder_heads,
            head_dim,
            scale,
        })
    }

    fn cross_attention(
        &mut self,
        x: &Array,
        encoder_out: &Array,
    ) -> std::result::Result<Array, Exception> {
        let shape = x.shape();
        let (batch, tgt_len, _) = (shape[0], shape[1], shape[2]);
        let src_len = encoder_out.shape()[1];

        let q = self.src_attn_q.forward(x)?;
        let kv = self.src_attn_kv.forward(encoder_out)?;

        let dim = self.num_heads * self.head_dim;
        let k = kv.index((.., .., ..dim));
        let v = kv.index((.., .., dim..));

        let q = q
            .reshape(&[batch, tgt_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[batch, src_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[batch, src_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let k_t = k.transpose_axes(&[0, 1, 3, 2])?;
        let scores = q.matmul(&k_t)?.multiply(array!(self.scale))?;
        let attn_weights = softmax_axis(&scores, -1, None)?;
        let attn_out = attn_weights.matmul(&v)?;

        let attn_out = attn_out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, tgt_len, self.num_heads * self.head_dim])?;

        self.src_attn_out.forward(&attn_out)
    }
}

/// Input for decoder layer
pub struct DecoderLayerInput<'a> {
    pub x: &'a Array,
    pub encoder_out: &'a Array,
}

impl<'a> Module<DecoderLayerInput<'a>> for ParaformerDecoderLayer {
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        input: DecoderLayerInput<'a>,
    ) -> std::result::Result<Self::Output, Self::Error> {
        let x = input.x;
        let encoder_out = input.encoder_out;
        let residual = x;

        let h = self.norm1.forward(x)?;
        let h = self.ffn.up_proj.forward(&h)?;
        let h = nn::relu(&h)?;
        let h = self.ffn_norm.forward(&h)?;
        let tgt = self.ffn.down_proj.forward(&h)?;

        let h = self.norm2.forward(&tgt)?;
        let h_fsmn = self.self_attn_fsmn.forward(&h)?;
        let h = ops::add(&h_fsmn, &h)?;
        let x = ops::add(residual, &h)?;

        let residual = &x;
        let h = self.norm3.forward(&x)?;
        let h = self.cross_attention(&h, encoder_out)?;
        ops::add(residual, &h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.self_attn_fsmn.training_mode(mode);
        self.src_attn_q.training_mode(mode);
        self.src_attn_kv.training_mode(mode);
        self.src_attn_out.training_mode(mode);
        self.ffn.training_mode(mode);
        self.ffn_norm.training_mode(mode);
        self.norm1.training_mode(mode);
        self.norm2.training_mode(mode);
        self.norm3.training_mode(mode);
    }
}

// ============================================================================
// Decoder
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct ParaformerDecoder {
    #[param]
    pub embed: nn::Embedding,
    #[param]
    pub layers: Vec<ParaformerDecoderLayer>,
    #[param]
    pub final_ffn_norm1: nn::LayerNorm,
    #[param]
    pub final_ffn_up: nn::Linear,
    #[param]
    pub final_ffn_norm: nn::LayerNorm,
    #[param]
    pub final_ffn_down: nn::Linear,
    #[param]
    pub after_norm: nn::LayerNorm,
    #[param]
    pub output_proj: nn::Linear,
}

impl ParaformerDecoder {
    pub fn new(config: &ParaformerConfig) -> std::result::Result<Self, Exception> {
        let embed = nn::Embedding::new(config.vocab_size, config.decoder_dim)?;

        let mut layers = Vec::with_capacity(config.decoder_layers as usize);
        for _ in 0..config.decoder_layers {
            layers.push(ParaformerDecoderLayer::new(config)?);
        }

        let final_ffn_norm1 = nn::LayerNormBuilder::new(config.decoder_dim)
            .eps(1e-5)
            .build()?;
        let final_ffn_up = nn::LinearBuilder::new(config.decoder_dim, config.decoder_ffn_dim)
            .bias(true)
            .build()?;
        let final_ffn_norm = nn::LayerNormBuilder::new(config.decoder_ffn_dim)
            .eps(1e-5)
            .build()?;
        let final_ffn_down = nn::LinearBuilder::new(config.decoder_ffn_dim, config.decoder_dim)
            .bias(false)
            .build()?;

        let after_norm = nn::LayerNormBuilder::new(config.decoder_dim)
            .eps(1e-5)
            .build()?;
        let output_proj = nn::LinearBuilder::new(config.decoder_dim, config.vocab_size)
            .bias(true)
            .build()?;

        Ok(Self {
            embed,
            layers,
            final_ffn_norm1,
            final_ffn_up,
            final_ffn_norm,
            final_ffn_down,
            after_norm,
            output_proj,
        })
    }
}

/// Input for decoder
pub struct DecoderInput<'a> {
    pub acoustic_embeds: &'a Array,
    pub encoder_out: &'a Array,
}

impl<'a> Module<DecoderInput<'a>> for ParaformerDecoder {
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        input: DecoderInput<'a>,
    ) -> std::result::Result<Self::Output, Self::Error> {
        let mut h = input.acoustic_embeds.clone();

        for layer in &mut self.layers {
            h = layer.forward(DecoderLayerInput {
                x: &h,
                encoder_out: input.encoder_out,
            })?;
        }

        let h = self.final_ffn_norm1.forward(&h)?;
        let h = self.final_ffn_up.forward(&h)?;
        let h = nn::relu(&h)?;
        let h = self.final_ffn_norm.forward(&h)?;
        let h = self.final_ffn_down.forward(&h)?;

        let h = self.after_norm.forward(&h)?;
        self.output_proj.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed.training_mode(mode);
        for layer in &mut self.layers {
            layer.training_mode(mode);
        }
        self.final_ffn_norm1.training_mode(mode);
        self.final_ffn_up.training_mode(mode);
        self.final_ffn_norm.training_mode(mode);
        self.final_ffn_down.training_mode(mode);
        self.after_norm.training_mode(mode);
        self.output_proj.training_mode(mode);
    }
}

// ============================================================================
// Full Model
// ============================================================================

/// Paraformer ASR model
#[derive(Debug, Clone, ModuleParameters)]
pub struct Paraformer {
    pub frontend: MelFrontend,
    #[param]
    pub encoder: SanmEncoder,
    #[param]
    pub predictor: CIFPredictor,
    #[param]
    pub decoder: ParaformerDecoder,
    pub config: ParaformerConfig,
}

impl Paraformer {
    pub fn new(config: ParaformerConfig) -> std::result::Result<Self, Exception> {
        let frontend = MelFrontend::new(&config);
        let encoder = SanmEncoder::new(&config)?;
        let predictor = CIFPredictor::new(&config)?;
        let decoder = ParaformerDecoder::new(&config)?;

        Ok(Self {
            frontend,
            encoder,
            predictor,
            decoder,
            config,
        })
    }

    /// Transcribe audio to token IDs
    pub fn transcribe(&mut self, audio: &Array) -> Result<Array> {
        let mel = self.frontend.forward(audio)?;
        let encoder_out = self.encoder.forward(&mel)?;
        let (acoustic_embeds, _token_num) = self.predictor.forward(&encoder_out)?;

        if acoustic_embeds.shape()[1] == 0 {
            return Ok(Array::from_slice::<i32>(&[], &[1, 0]));
        }

        let logits = self.decoder.forward(DecoderInput {
            acoustic_embeds: &acoustic_embeds,
            encoder_out: &encoder_out,
        })?;

        let token_ids = argmax_axis!(logits, -1)?;
        Ok(token_ids.as_dtype(mlx_rs::Dtype::Int32)?)
    }

    /// Transcribe from pre-computed mel features (for batched processing)
    ///
    /// Use this when you have already computed mel spectrograms and want
    /// to process them in a batch for better throughput.
    pub fn transcribe_from_mel(&mut self, mel: &Array) -> Result<(Array, Array)> {
        let encoder_out = self.encoder.forward(mel)?;
        let (acoustic_embeds, token_num) = self.predictor.forward(&encoder_out)?;

        if acoustic_embeds.shape()[1] == 0 {
            let batch = mel.shape()[0];
            return Ok((
                Array::from_slice::<i32>(&[], &[batch, 0]),
                Array::zeros::<i32>(&[batch])?,
            ));
        }

        let logits = self.decoder.forward(DecoderInput {
            acoustic_embeds: &acoustic_embeds,
            encoder_out: &encoder_out,
        })?;

        let token_ids = argmax_axis!(logits, -1)?;
        Ok((token_ids.as_dtype(mlx_rs::Dtype::Int32)?, token_num))
    }

    /// Set CMVN normalization parameters
    pub fn set_cmvn(&mut self, addshift: Vec<f32>, rescale: Vec<f32>) {
        self.frontend.set_cmvn(addshift, rescale);
    }
}

impl Module<&Array> for Paraformer {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, audio: &Array) -> std::result::Result<Self::Output, Self::Error> {
        self.transcribe(audio).map_err(|e| match e {
            Error::Mlx(ex) => ex,
            _ => Exception::from(e.to_string().as_str()),
        })
    }

    fn training_mode(&mut self, mode: bool) {
        self.encoder.training_mode(mode);
        self.predictor.training_mode(mode);
        self.decoder.training_mode(mode);
    }
}

// ============================================================================
// Weight Loading
// ============================================================================

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::Model(format!("Missing weight: {}", key)))
}

fn get_conv_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    let weight = get_weight(weights, key)?;
    weight
        .transpose_axes(&[0, 2, 1])
        .map_err(|e| Error::Model(format!("Failed to transpose conv weight: {}", e)))
}

fn load_paraformer_weights(model: &mut Paraformer, weights: &HashMap<String, Array>) -> Result<()> {
    eprintln!("Loading {} weight tensors...", weights.len());

    // Encoder First Layer
    {
        let layer = &mut model.encoder.first_layer;
        let prefix = "encoder.encoders0.0";

        layer.self_attn.linear_q_k_v.weight =
            Param::new(get_weight(weights, &format!("{}.self_attn.linear_q_k_v.weight", prefix))?);
        layer.self_attn.linear_q_k_v.bias = Param::new(Some(get_weight(
            weights,
            &format!("{}.self_attn.linear_q_k_v.bias", prefix),
        )?));
        layer.self_attn.out_proj.weight =
            Param::new(get_weight(weights, &format!("{}.self_attn.out_proj.weight", prefix))?);
        layer.self_attn.out_proj.bias = Param::new(Some(get_weight(
            weights,
            &format!("{}.self_attn.out_proj.bias", prefix),
        )?));
        layer.self_attn.fsmn_block.weight =
            Param::new(get_conv_weight(weights, &format!("{}.self_attn.fsmn_block.weight", prefix))?);

        layer.ffn.up_proj.weight =
            Param::new(get_weight(weights, &format!("{}.ffn.up_proj.weight", prefix))?);
        layer.ffn.up_proj.bias =
            Param::new(Some(get_weight(weights, &format!("{}.ffn.up_proj.bias", prefix))?));
        layer.ffn.down_proj.weight =
            Param::new(get_weight(weights, &format!("{}.ffn.down_proj.weight", prefix))?);
        layer.ffn.down_proj.bias =
            Param::new(Some(get_weight(weights, &format!("{}.ffn.down_proj.bias", prefix))?));

        layer.norm1.weight =
            Param::new(Some(get_weight(weights, &format!("{}.norm1.weight", prefix))?));
        layer.norm1.bias =
            Param::new(Some(get_weight(weights, &format!("{}.norm1.bias", prefix))?));
        layer.norm2.weight =
            Param::new(Some(get_weight(weights, &format!("{}.norm2.weight", prefix))?));
        layer.norm2.bias =
            Param::new(Some(get_weight(weights, &format!("{}.norm2.bias", prefix))?));
    }

    // Regular Encoder Layers
    for (i, layer) in model.encoder.layers.iter_mut().enumerate() {
        let prefix = format!("encoder.layers.{}", i);

        layer.self_attn.linear_q_k_v.weight =
            Param::new(get_weight(weights, &format!("{}.self_attn.linear_q_k_v.weight", prefix))?);
        layer.self_attn.linear_q_k_v.bias = Param::new(Some(get_weight(
            weights,
            &format!("{}.self_attn.linear_q_k_v.bias", prefix),
        )?));
        layer.self_attn.out_proj.weight =
            Param::new(get_weight(weights, &format!("{}.self_attn.out_proj.weight", prefix))?);
        layer.self_attn.out_proj.bias = Param::new(Some(get_weight(
            weights,
            &format!("{}.self_attn.out_proj.bias", prefix),
        )?));
        layer.self_attn.fsmn_block.weight =
            Param::new(get_conv_weight(weights, &format!("{}.self_attn.fsmn_block.weight", prefix))?);

        layer.ffn.up_proj.weight =
            Param::new(get_weight(weights, &format!("{}.ffn.up_proj.weight", prefix))?);
        layer.ffn.up_proj.bias =
            Param::new(Some(get_weight(weights, &format!("{}.ffn.up_proj.bias", prefix))?));
        layer.ffn.down_proj.weight =
            Param::new(get_weight(weights, &format!("{}.ffn.down_proj.weight", prefix))?);
        layer.ffn.down_proj.bias =
            Param::new(Some(get_weight(weights, &format!("{}.ffn.down_proj.bias", prefix))?));

        layer.norm1.weight =
            Param::new(Some(get_weight(weights, &format!("{}.norm1.weight", prefix))?));
        layer.norm1.bias =
            Param::new(Some(get_weight(weights, &format!("{}.norm1.bias", prefix))?));
        layer.norm2.weight =
            Param::new(Some(get_weight(weights, &format!("{}.norm2.weight", prefix))?));
        layer.norm2.bias =
            Param::new(Some(get_weight(weights, &format!("{}.norm2.bias", prefix))?));
    }

    model.encoder.after_norm.weight =
        Param::new(Some(get_weight(weights, "encoder.after_norm.weight")?));
    model.encoder.after_norm.bias =
        Param::new(Some(get_weight(weights, "encoder.after_norm.bias")?));

    // CIF Predictor
    model.predictor.conv.weight = Param::new(get_conv_weight(weights, "predictor.conv.weight")?);
    model.predictor.conv.bias = Param::new(Some(get_weight(weights, "predictor.conv.bias")?));
    model.predictor.output_proj.weight =
        Param::new(get_weight(weights, "predictor.output_proj.weight")?);
    model.predictor.output_proj.bias =
        Param::new(Some(get_weight(weights, "predictor.output_proj.bias")?));

    // Decoder
    model.decoder.embed.weight = Param::new(get_weight(weights, "decoder.embed.0.weight")?);

    for (i, layer) in model.decoder.layers.iter_mut().enumerate() {
        let prefix = format!("decoder.layers.{}", i);

        layer.self_attn_fsmn.weight =
            Param::new(get_conv_weight(weights, &format!("{}.self_attn.fsmn_block.weight", prefix))?);

        layer.src_attn_q.weight =
            Param::new(get_weight(weights, &format!("{}.src_attn.q_proj.weight", prefix))?);
        layer.src_attn_q.bias =
            Param::new(Some(get_weight(weights, &format!("{}.src_attn.q_proj.bias", prefix))?));
        layer.src_attn_kv.weight =
            Param::new(get_weight(weights, &format!("{}.src_attn.linear_k_v.weight", prefix))?);
        layer.src_attn_kv.bias =
            Param::new(Some(get_weight(weights, &format!("{}.src_attn.linear_k_v.bias", prefix))?));
        layer.src_attn_out.weight =
            Param::new(get_weight(weights, &format!("{}.src_attn.out_proj.weight", prefix))?);
        layer.src_attn_out.bias =
            Param::new(Some(get_weight(weights, &format!("{}.src_attn.out_proj.bias", prefix))?));

        layer.ffn.up_proj.weight =
            Param::new(get_weight(weights, &format!("{}.ffn.up_proj.weight", prefix))?);
        layer.ffn.up_proj.bias =
            Param::new(Some(get_weight(weights, &format!("{}.ffn.up_proj.bias", prefix))?));
        layer.ffn.down_proj.weight =
            Param::new(get_weight(weights, &format!("{}.ffn.down_proj.weight", prefix))?);
        layer.ffn.down_proj.bias = Param::new(None);

        layer.ffn_norm.weight = Param::new(Some(get_weight(
            weights,
            &format!("{}.feed_forward.norm.weight", prefix),
        )?));
        layer.ffn_norm.bias = Param::new(Some(get_weight(
            weights,
            &format!("{}.feed_forward.norm.bias", prefix),
        )?));

        layer.norm1.weight =
            Param::new(Some(get_weight(weights, &format!("{}.norm1.weight", prefix))?));
        layer.norm1.bias =
            Param::new(Some(get_weight(weights, &format!("{}.norm1.bias", prefix))?));
        layer.norm2.weight =
            Param::new(Some(get_weight(weights, &format!("{}.norm2.weight", prefix))?));
        layer.norm2.bias =
            Param::new(Some(get_weight(weights, &format!("{}.norm2.bias", prefix))?));
        layer.norm3.weight =
            Param::new(Some(get_weight(weights, &format!("{}.norm3.weight", prefix))?));
        layer.norm3.bias =
            Param::new(Some(get_weight(weights, &format!("{}.norm3.bias", prefix))?));
    }

    // Final FFN layer
    model.decoder.final_ffn_norm1.weight =
        Param::new(Some(get_weight(weights, "decoder.decoders3.0.norm1.weight")?));
    model.decoder.final_ffn_norm1.bias =
        Param::new(Some(get_weight(weights, "decoder.decoders3.0.norm1.bias")?));
    model.decoder.final_ffn_up.weight =
        Param::new(get_weight(weights, "decoder.decoders3.0.ffn.up_proj.weight")?);
    model.decoder.final_ffn_up.bias = Param::new(Some(get_weight(
        weights,
        "decoder.decoders3.0.ffn.up_proj.bias",
    )?));
    model.decoder.final_ffn_norm.weight = Param::new(Some(get_weight(
        weights,
        "decoder.decoders3.0.feed_forward.norm.weight",
    )?));
    model.decoder.final_ffn_norm.bias = Param::new(Some(get_weight(
        weights,
        "decoder.decoders3.0.feed_forward.norm.bias",
    )?));
    model.decoder.final_ffn_down.weight =
        Param::new(get_weight(weights, "decoder.decoders3.0.ffn.down_proj.weight")?);

    model.decoder.after_norm.weight =
        Param::new(Some(get_weight(weights, "decoder.after_norm.weight")?));
    model.decoder.after_norm.bias =
        Param::new(Some(get_weight(weights, "decoder.after_norm.bias")?));
    model.decoder.output_proj.weight =
        Param::new(get_weight(weights, "decoder.output_proj.weight")?);
    model.decoder.output_proj.bias =
        Param::new(Some(get_weight(weights, "decoder.output_proj.bias")?));

    eprintln!("Weights loaded successfully");
    Ok(())
}

/// Parse FunASR am.mvn file for CMVN parameters
pub fn parse_cmvn_file(path: impl AsRef<Path>) -> Result<(Vec<f32>, Vec<f32>)> {
    let content = fs::read_to_string(path.as_ref())?;

    let mut addshift = Vec::new();
    let mut rescale = Vec::new();
    let mut in_addshift = false;
    let mut in_rescale = false;
    let mut in_values = false;

    for line in content.lines() {
        let line = line.trim();

        if line.contains("<AddShift>") {
            in_addshift = true;
            in_rescale = false;
            in_values = false;
            continue;
        }
        if line.contains("<Rescale>") {
            in_addshift = false;
            in_rescale = true;
            in_values = false;
            continue;
        }
        if line.contains("</Nnet>") {
            break;
        }
        if line.contains("<Splice>") || line.contains("<Nnet>") {
            continue;
        }

        if (in_addshift || in_rescale) && (line.contains('[') || in_values) {
            let mut parse_str = line;

            if let Some(start) = line.find('[') {
                in_values = true;
                parse_str = &line[start + 1..];
            }

            let at_end = parse_str.contains(']');
            if at_end {
                if let Some(end) = parse_str.find(']') {
                    parse_str = &parse_str[..end];
                }
                in_values = false;
            }

            let values: Vec<f32> = parse_str
                .split_whitespace()
                .filter_map(|s| s.parse::<f32>().ok())
                .collect();

            if in_addshift {
                addshift.extend(values);
            } else if in_rescale {
                rescale.extend(values);
            }
        }
    }

    if addshift.is_empty() || rescale.is_empty() {
        return Err(Error::Config(format!(
            "Failed to parse CMVN (addshift={}, rescale={})",
            addshift.len(),
            rescale.len()
        )));
    }

    if addshift.len() != 560 || rescale.len() != 560 {
        return Err(Error::Config(format!(
            "CMVN dimension mismatch: expected 560, got addshift={}, rescale={}",
            addshift.len(),
            rescale.len()
        )));
    }

    Ok((addshift, rescale))
}

/// Load Paraformer model from safetensors
pub fn load_model(weights_path: impl AsRef<Path>) -> Result<Paraformer> {
    let config = ParaformerConfig::default();
    let mut model = Paraformer::new(config).map_err(Error::Mlx)?;

    let weights = Array::load_safetensors(weights_path.as_ref())
        .map_err(|e| Error::Model(format!("Failed to load weights: {:?}", e)))?;
    load_paraformer_weights(&mut model, &weights)?;

    Ok(model)
}

/// Load Paraformer model with custom config
pub fn load_model_with_config(
    weights_path: impl AsRef<Path>,
    config: ParaformerConfig,
) -> Result<Paraformer> {
    let mut model = Paraformer::new(config).map_err(Error::Mlx)?;

    let weights = Array::load_safetensors(weights_path.as_ref())
        .map_err(|e| Error::Model(format!("Failed to load weights: {:?}", e)))?;
    load_paraformer_weights(&mut model, &weights)?;

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ParaformerConfig::default();
        assert_eq!(config.encoder_layers, 50);
        assert_eq!(config.decoder_layers, 16);
        assert_eq!(config.vocab_size, 8404);
    }

    #[test]
    fn test_sinusoidal_encoding() {
        let pe = sinusoidal_position_encoding(100, 512).unwrap();
        assert_eq!(pe.shape(), &[100, 512]);
    }

    #[test]
    fn test_model_creation() {
        let config = ParaformerConfig::default();
        let model = Paraformer::new(config);
        assert!(model.is_ok());
    }
}
