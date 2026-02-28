//! Mimi Speech Encoder: encodes reference audio to 16-codebook discrete codes.
//!
//! Used for ICL voice cloning in Base model mode. Encodes 24kHz audio
//! into 12Hz codec frames with 16 codebooks (1 semantic + 15 acoustic).
//!
//! Architecture: SEANet Conv Encoder → Transformer → Downsample → RVQ
//!
//! Weight keys from `speech_tokenizer/model.safetensors`:
//!   `encoder.encoder.layers.*`          — SEANet convolutional encoder
//!   `encoder.encoder_transformer.layers.*` — 8-layer transformer
//!   `encoder.downsample.conv.*`         — 25Hz → 12.5Hz
//!   `encoder.quantizer.*`               — Semantic + Acoustic RVQ

use std::collections::HashMap;

use mlx_rs::{array, Array};
use mlx_rs::module::{Module, Param};
use mlx_rs::nn;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::eval;

use crate::error::{Error, Result};

// ============================================================================
// Conv helpers (reusing patterns from speech_tokenizer decoder)
// ============================================================================

/// Transpose Conv1d weight from PyTorch [out, in, kernel] to MLX [out, kernel, in].
fn transpose_conv_weight(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 1])?)
}

/// A causal Conv1d with configurable padding mode (constant/replicate).
/// Matches HF MimiConv1d: causal left padding + dynamic extra right padding
/// to ensure output length = ceil(input_length / stride).
struct CausalConv1d {
    conv: nn::Conv1d,
    left_pad: i32,
    kernel_size: i32,
    stride: i32,
    replicate: bool, // true = replicate padding (edge values), false = constant/zero padding
}

impl CausalConv1d {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        // x: [B, T, C] in MLX (channel-last)
        let length = x.dim(1) as i32;

        // Compute extra right padding (matches Python _get_extra_padding_for_conv1d)
        let n_frames_num = length - self.kernel_size + self.left_pad;
        let n_frames_ceil = (n_frames_num + self.stride - 1) / self.stride + 1;
        let ideal_length = (n_frames_ceil - 1) * self.stride + self.kernel_size - self.left_pad;
        let extra_pad = (ideal_length - length).max(0);

        let x = if self.left_pad > 0 || extra_pad > 0 {
            let b = x.dim(0) as i32;
            let c = x.dim(2) as i32;
            let mut parts: Vec<Array> = Vec::new();
            if self.left_pad > 0 {
                if self.replicate {
                    let first = x.index((.., ..1i32, ..));
                    let left = ops::broadcast_to(&first, &[b, self.left_pad, c])?;
                    parts.push(left);
                } else {
                    parts.push(ops::zeros::<f32>(&[b, self.left_pad, c])?);
                }
            }
            parts.push(x.clone());
            if extra_pad > 0 {
                if self.replicate {
                    let last = x.index((.., -1i32.., ..));
                    let right = ops::broadcast_to(&last, &[b, extra_pad, c])?;
                    parts.push(right);
                } else {
                    parts.push(ops::zeros::<f32>(&[b, extra_pad, c])?);
                }
            }
            let refs: Vec<&Array> = parts.iter().collect();
            ops::concatenate_axis(&refs[..], 1)?
        } else {
            x.clone()
        };
        Ok(self.conv.forward(&x)?)
    }
}

// ============================================================================
// Residual Block (for SEANet encoder)
// ============================================================================

/// Residual block in the SEANet encoder.
/// Two convolutions with optional dimension change + skip connection.
struct EncoderResBlock {
    conv1: CausalConv1d, // bottleneck: C → C/2, k=3
    conv2: CausalConv1d, // expand: C/2 → C, k=1
    shortcut: Option<CausalConv1d>, // if input/output dims differ
}

impl EncoderResBlock {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        // ELU activation before each conv (pre-activation residual block)
        let h = elu(x)?;
        let h = self.conv1.forward(&h)?;
        let h = elu(&h)?;
        let h = self.conv2.forward(&h)?;

        let skip = if let Some(ref mut sc) = self.shortcut {
            sc.forward(x)?
        } else {
            x.clone()
        };

        Ok(h.add(&skip)?)
    }
}

/// ELU activation: x if x > 0, else alpha * (exp(x) - 1) where alpha=1.0
fn elu(x: &Array) -> Result<Array> {
    let zero = array!(0.0f32);
    let one = array!(1.0f32);
    let positive = x.gt(&zero)?;
    let exp_x = ops::exp(x)?;
    let neg_part = exp_x.subtract(&one)?;
    Ok(ops::r#where(&positive, x, &neg_part)?)
}

// ============================================================================
// Affine LayerNorm (encoder transformer uses LN with bias)
// ============================================================================

struct AffineLayerNorm {
    weight: Array,
    bias: Array,
    eps: f32,
}

impl AffineLayerNorm {
    fn forward(&self, x: &Array) -> Result<Array> {
        // x: [B, T, D]
        let mean = ops::mean_axis(x, -1, true)?;
        let diff = x.subtract(&mean)?;
        let var = ops::mean_axis(&diff.multiply(&diff)?, -1, true)?;
        let inv_std = ops::rsqrt(&var.add(&array!(self.eps))?)?;
        let normed = diff.multiply(&inv_std)?;
        let scaled = normed.multiply(&self.weight)?;
        Ok(scaled.add(&self.bias)?)
    }
}

// ============================================================================
// Encoder Transformer Layer
// ============================================================================

struct EncoderTransformerLayer {
    input_layernorm: AffineLayerNorm,
    q_proj: Array, // [D, D]
    k_proj: Array,
    v_proj: Array,
    o_proj: Array,
    self_attn_layer_scale: Array, // [D]
    post_attention_layernorm: AffineLayerNorm,
    fc1: Array, // [4D, D]
    fc2: Array, // [D, 4D]
    mlp_layer_scale: Array, // [D]
    num_heads: i32,
    head_dim: i32,
}

impl EncoderTransformerLayer {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        // Self-attention with pre-norm
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attention(&normed)?;
        let attn_scaled = attn_out.multiply(&self.self_attn_layer_scale)?;
        let x = x.add(&attn_scaled)?;

        // MLP with pre-norm
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp(&normed)?;
        let mlp_scaled = mlp_out.multiply(&self.mlp_layer_scale)?;
        Ok(x.add(&mlp_scaled)?)
    }

    fn self_attention(&self, x: &Array) -> Result<Array> {
        // x: [B, T, D]
        let b = x.dim(0) as i32;
        let t = x.dim(1) as i32;
        let t_usize = t as usize;

        // Q, K, V projections
        let mut q = ops::matmul(x, &self.q_proj.t())?.reshape(&[b, t, self.num_heads, self.head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
        let mut k = ops::matmul(x, &self.k_proj.t())?.reshape(&[b, t, self.num_heads, self.head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
        let v = ops::matmul(x, &self.v_proj.t())?.reshape(&[b, t, self.num_heads, self.head_dim])?.transpose_axes(&[0, 2, 1, 3])?;

        // RoPE (theta=10000, stride-based like Qwen)
        let hd = self.head_dim as usize;
        let half = hd / 2;
        let mut freqs_data = vec![0.0f32; half];
        for i in 0..half {
            freqs_data[i] = 1.0 / (10000.0f32).powf(2.0 * i as f32 / hd as f32);
        }
        let mut cos_data = vec![0.0f32; t_usize * half];
        let mut sin_data = vec![0.0f32; t_usize * half];
        for pos in 0..t_usize {
            for i in 0..half {
                let angle = pos as f32 * freqs_data[i];
                cos_data[pos * half + i] = angle.cos();
                sin_data[pos * half + i] = angle.sin();
            }
        }
        let cos_arr = Array::from_slice(&cos_data, &[1, 1, t, half as i32]);
        let sin_arr = Array::from_slice(&sin_data, &[1, 1, t, half as i32]);

        // Apply RoPE: split first/second half
        let q1 = q.index((.., .., .., ..half as i32));
        let q2 = q.index((.., .., .., half as i32..));
        q = mlx_rs::ops::concatenate_axis(
            &[&q1.multiply(&cos_arr)?.subtract(&q2.multiply(&sin_arr)?)?,
              &q2.multiply(&cos_arr)?.add(&q1.multiply(&sin_arr)?)?],
            -1,
        )?;
        let k1 = k.index((.., .., .., ..half as i32));
        let k2 = k.index((.., .., .., half as i32..));
        k = mlx_rs::ops::concatenate_axis(
            &[&k1.multiply(&cos_arr)?.subtract(&k2.multiply(&sin_arr)?)?,
              &k2.multiply(&cos_arr)?.add(&k1.multiply(&sin_arr)?)?],
            -1,
        )?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        let scores = ops::matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?.multiply(array!(1.0 / scale))?;

        // Causal mask with sliding window (window=250)
        let window = 250usize;
        let mut mask_data = vec![0.0f32; t_usize * t_usize];
        for row in 0..t_usize {
            for col in 0..t_usize {
                if col > row || row - col >= window {
                    mask_data[row * t_usize + col] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Array::from_slice(&mask_data, &[1, 1, t, t]);
        let scores = scores.add(&mask)?;

        let attn_weights = ops::softmax_axis(&scores, -1, None::<bool>)?;
        let attn_out = ops::matmul(&attn_weights, &v)?;

        // Reshape and output projection
        let attn_out = attn_out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, t, -1])?;
        Ok(ops::matmul(&attn_out, &self.o_proj.t())?)
    }

    fn mlp(&self, x: &Array) -> Result<Array> {
        // Standard MLP: fc1 → GELU → fc2
        let h = ops::matmul(x, &self.fc1.t())?;
        let h = nn::gelu(&h)?;
        Ok(ops::matmul(&h, &self.fc2.t())?)
    }
}

// ============================================================================
// RVQ Codebook
// ============================================================================

struct RvqCodebook {
    embedding: Array, // [codebook_size, codebook_dim] (normalized)
}

impl RvqCodebook {
    /// Find nearest codebook entry for each vector.
    /// Input: [B, T, D] (already projected to codebook_dim)
    /// Output: [B, T] codes (u32)
    fn quantize(&self, x: &Array) -> Result<(Array, Array)> {
        // L2 distance: ||x - e||^2 = ||x||^2 - 2*x*e^T + ||e||^2
        let x_sq = ops::sum_axis(&x.multiply(x)?, -1, true)?; // [B, T, 1]
        let e_sq = ops::sum_axis(&self.embedding.multiply(&self.embedding)?, -1, true)?; // [1, codebook_size]
        let x_e = ops::matmul(x, &self.embedding.t())?; // [B, T, codebook_size]

        let dists = x_sq.subtract(&x_e.multiply(array!(2.0f32))?)?.add(&e_sq.t())?;

        // Argmin
        let codes = ops::indexing::argmin_axis(&dists, -1, None)?; // [B, T]

        // Lookup embeddings
        let flat_codes = codes.reshape(&[-1])?;
        let quantized = self.embedding.index(flat_codes);
        let quantized = quantized.reshape(&[x.dim(0) as i32, x.dim(1) as i32, -1])?;

        Ok((codes, quantized))
    }
}

/// Normalize codebook: embedding = embed_sum / cluster_usage.clamp(min=epsilon)
/// Matches Python MimiEuclideanCodebook: epsilon=1e-5
fn normalize_codebook(embed_sum: &Array, cluster_usage: &Array) -> Result<Array> {
    let usage = cluster_usage.reshape(&[-1, 1])?;
    let usage = ops::maximum(&usage, &array!(1e-5f32))?;
    Ok(embed_sum.divide(&usage)?)
}

// ============================================================================
// Split Residual Vector Quantizer
// ============================================================================

#[allow(dead_code)]
struct SplitRvq {
    semantic_input_proj: nn::Conv1d,
    semantic_codebook: RvqCodebook,
    semantic_output_proj: nn::Conv1d,
    acoustic_input_proj: nn::Conv1d,
    acoustic_codebooks: Vec<RvqCodebook>,
    acoustic_output_proj: nn::Conv1d, // kept for full reconstruction if needed
}

impl SplitRvq {
    /// Encode input to 16 codebook codes.
    /// Input: [B, T, 512]
    /// Output: [T, 16] codes as u32 (batch squeezed)
    ///
    /// NOTE: Semantic and acoustic RVQs encode the SAME input independently.
    /// Each has its own input_proj. The residual subtraction happens within
    /// each RVQ's layers, not between semantic and acoustic.
    fn encode(&mut self, x: &Array) -> Result<Vec<[u32; 16]>> {
        let t = x.dim(1) as usize;

        // Semantic quantization (codebook 0)
        let sem_proj = self.semantic_input_proj.forward(x)?; // [B, T, 256]
        let (sem_codes, _sem_quantized) = self.semantic_codebook.quantize(&sem_proj)?;

        // Acoustic quantization (codebooks 1-15)
        // IMPORTANT: acoustic RVQ receives the ORIGINAL input x, not the semantic residual.
        // Both RVQs independently project the same input into their own spaces.
        let acou_proj = self.acoustic_input_proj.forward(x)?; // [B, T, 256]
        let mut acou_residual = acou_proj;
        let mut all_acoustic_codes: Vec<Array> = Vec::with_capacity(15);

        for codebook in &self.acoustic_codebooks {
            let (codes, quantized) = codebook.quantize(&acou_residual)?;
            acou_residual = acou_residual.subtract(&quantized)?;
            all_acoustic_codes.push(codes);
        }

        // Convert to Vec<[u32; 16]>
        eval(std::iter::once(&sem_codes))?;
        for c in &all_acoustic_codes {
            eval(std::iter::once(c))?;
        }

        let sem_codes_vec: Vec<u32> = sem_codes.reshape(&[-1])?.as_slice::<u32>().to_vec();
        let acou_codes_vecs: Vec<Vec<u32>> = all_acoustic_codes
            .iter()
            .map(|c| c.reshape(&[-1]).unwrap().as_slice::<u32>().to_vec())
            .collect();

        let n_acoustic = acou_codes_vecs.len().min(15);
        let mut frames = Vec::with_capacity(t);
        for frame_idx in 0..t {
            let mut frame = [0u32; 16];
            frame[0] = sem_codes_vec[frame_idx];
            for g in 0..n_acoustic {
                frame[g + 1] = acou_codes_vecs[g][frame_idx];
            }
            frames.push(frame);
        }

        Ok(frames)
    }
}

// ============================================================================
// Full Mimi Encoder
// ============================================================================

/// Mimi speech encoder: 24kHz audio → 12Hz codec frames with 16 codebooks.
pub struct SpeechEncoder {
    /// SEANet convolutional encoder layers
    encoder_layers: Vec<EncoderLayer>,
    /// Transformer layers
    transformer_layers: Vec<EncoderTransformerLayer>,
    /// 25Hz → 12.5Hz downsampler
    downsample: CausalConv1d,
    /// RVQ quantizer
    quantizer: SplitRvq,
}

/// Different types of encoder layers in SEANet
enum EncoderLayer {
    Conv(CausalConv1d),     // Regular or stride conv
    ResBlock(EncoderResBlock), // Residual block
    Elu,                    // ELU activation (between layers)
}

impl SpeechEncoder {
    /// Encode audio samples to codec frames.
    /// Input: audio samples [1, N_samples, 1] (or we handle reshaping)
    /// Output: Vec<[u32; 16]> codec frames at ~12Hz
    pub fn encode(&mut self, samples: &[f32]) -> Result<Vec<[u32; 16]>> {
        // Reshape audio: [1, N, 1]
        let n = samples.len() as i32;
        let x = Array::from_slice(samples, &[1, n, 1]);

        // SEANet conv encoder
        let mut h = x;
        for layer in self.encoder_layers.iter_mut() {
            h = match layer {
                EncoderLayer::Conv(conv) => conv.forward(&h)?,
                EncoderLayer::ResBlock(block) => block.forward(&h)?,
                EncoderLayer::Elu => elu(&h)?,
            };
        }

        // Transformer
        for layer in self.transformer_layers.iter_mut() {
            h = layer.forward(&h)?;
        }
        eval(std::iter::once(&h))?;

        // Downsample: 25Hz → 12.5Hz (uses replicate padding)
        h = self.downsample.forward(&h)?;
        eval(std::iter::once(&h))?;

        // RVQ encode
        self.quantizer.encode(&h)
    }
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

fn load_conv1d_with_stride(
    weights: &HashMap<String, Array>,
    prefix: &str,
    stride: i32,
    dilation: i32,
) -> Result<CausalConv1d> {
    load_conv1d_with_stride_and_pad(weights, prefix, stride, dilation, false)
}

fn load_conv1d_with_stride_and_pad(
    weights: &HashMap<String, Array>,
    prefix: &str,
    stride: i32,
    dilation: i32,
    replicate: bool,
) -> Result<CausalConv1d> {
    let w = transpose_conv_weight(&get_weight(weights, &format!("{prefix}.weight"))?)?;
    let bias = weights.get(&format!("{prefix}.bias")).cloned();

    let kernel_size = w.dim(1) as i32;

    // Match Python MimiConv1d:
    // effective_kernel_size = (kernel_size - 1) * dilation + 1
    // left_pad = effective_kernel_size - stride
    let effective_kernel = (kernel_size - 1) * dilation + 1;
    let left_pad = effective_kernel - stride;

    let conv = nn::Conv1d {
        weight: Param::new(w),
        bias: Param::new(bias),
        stride,
        padding: 0, // we do manual padding
        dilation,
        groups: 1,
    };

    Ok(CausalConv1d { conv, left_pad, kernel_size: effective_kernel, stride, replicate })
}

fn load_encoder_res_block(
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<EncoderResBlock> {
    // ResBlock has conv1 (bottleneck) and conv2 (expand)
    // Try to load shortcut conv if it exists
    let conv1 = load_conv1d_with_stride(weights, &format!("{prefix}.block.1.conv"), 1, 1)?;
    let conv2 = load_conv1d_with_stride(weights, &format!("{prefix}.block.3.conv"), 1, 1)?;

    let shortcut = if weights.contains_key(&format!("{prefix}.shortcut.conv.weight")) {
        Some(load_conv1d_with_stride(weights, &format!("{prefix}.shortcut.conv"), 1, 1)?)
    } else {
        None
    };

    Ok(EncoderResBlock { conv1, conv2, shortcut })
}

fn load_affine_layer_norm(
    weights: &HashMap<String, Array>,
    prefix: &str,
    eps: f32,
) -> Result<AffineLayerNorm> {
    let weight = get_weight(weights, &format!("{prefix}.weight"))?;
    let bias = get_weight(weights, &format!("{prefix}.bias"))?;
    Ok(AffineLayerNorm { weight, bias, eps })
}

fn load_encoder_transformer_layer(
    weights: &HashMap<String, Array>,
    prefix: &str,
    hidden_dim: i32,
    num_heads: i32,
) -> Result<EncoderTransformerLayer> {
    let head_dim = hidden_dim / num_heads;

    let input_layernorm = load_affine_layer_norm(weights, &format!("{prefix}.input_layernorm"), 1e-5)?;
    let post_attention_layernorm = load_affine_layer_norm(weights, &format!("{prefix}.post_attention_layernorm"), 1e-5)?;

    let q_proj = get_weight(weights, &format!("{prefix}.self_attn.q_proj.weight"))?;
    let k_proj = get_weight(weights, &format!("{prefix}.self_attn.k_proj.weight"))?;
    let v_proj = get_weight(weights, &format!("{prefix}.self_attn.v_proj.weight"))?;
    let o_proj = get_weight(weights, &format!("{prefix}.self_attn.o_proj.weight"))?;
    let self_attn_layer_scale = get_weight(weights, &format!("{prefix}.self_attn_layer_scale.scale"))?;

    let fc1 = get_weight(weights, &format!("{prefix}.mlp.fc1.weight"))?;
    let fc2 = get_weight(weights, &format!("{prefix}.mlp.fc2.weight"))?;
    let mlp_layer_scale = get_weight(weights, &format!("{prefix}.mlp_layer_scale.scale"))?;

    Ok(EncoderTransformerLayer {
        input_layernorm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        self_attn_layer_scale,
        post_attention_layernorm,
        fc1,
        fc2,
        mlp_layer_scale,
        num_heads,
        head_dim,
    })
}

fn load_rvq_codebook(
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<RvqCodebook> {
    let embed_sum = get_weight(weights, &format!("{prefix}.codebook.embed_sum"))?;
    let cluster_usage = get_weight(weights, &format!("{prefix}.codebook.cluster_usage"))?;
    let embedding = normalize_codebook(&embed_sum, &cluster_usage)?;
    Ok(RvqCodebook { embedding })
}

fn load_conv1d_proj(
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<nn::Conv1d> {
    let w = transpose_conv_weight(&get_weight(weights, &format!("{prefix}.weight"))?)?;

    Ok(nn::Conv1d {
        weight: Param::new(w),
        bias: Param::new(None),
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
    })
}

/// Load the Mimi speech encoder from the speech_tokenizer weights.
/// Expects all encoder.* keys in the weight map.
pub fn load_speech_encoder(weights: &HashMap<String, Array>) -> Result<SpeechEncoder> {
    let prefix = "encoder";

    // ========================================================================
    // SEANet convolutional encoder
    // ========================================================================
    // The encoder has a specific layer structure:
    //   layer 0: initial conv (1 → 64, k=7)
    //   layer 1: resblock (64 → 64)
    //   layer 3: stride conv (64 → 128, k=8, stride=8)
    //   layer 4: resblock (128 → 128)
    //   layer 6: stride conv (128 → 256, k=10, stride=5 or 6)
    //   layer 7: resblock (256 → 256)
    //   layer 9: stride conv (256 → 512, k=12, stride=4 or 5)
    //   layer 10: resblock (512 → 512)
    //   layer 12: stride conv (512 → 1024, k=16, stride=4)
    //   layer 14: final conv (1024 → 512, k=3)
    //
    // We detect layers by probing weight keys.

    let mut encoder_layers: Vec<EncoderLayer> = Vec::new();

    // The SEANet encoder structure from moshi/HuggingFace:
    //   [initial_conv, ResBlock, ELU, stride_conv, ResBlock, ELU, stride_conv, ...,
    //    ResBlock, ELU, stride_conv, ELU, final_conv]
    //
    // ELU layers (2, 5, 8, 11, 13) have no weights. We must insert them explicitly.
    // Probe which layers exist and determine their type.
    for layer_idx in 0..20 {
        let conv_key = format!("{prefix}.encoder.layers.{layer_idx}.conv.weight");
        let block_key = format!("{prefix}.encoder.layers.{layer_idx}.block.1.conv.weight");

        if weights.contains_key(&conv_key) {
            // It's a plain conv layer — detect stride from weight shape
            let w = get_weight(weights, &conv_key)?;
            // Shape after transpose: [out, kernel, in]
            let w_t = transpose_conv_weight(&w)?;
            let out_ch = w_t.dim(0) as i32;
            let kernel = w_t.dim(1) as i32;
            let in_ch = w_t.dim(2) as i32;

            // Detect stride: for SEANet Mimi encoder, downsample convs have
            // kernel > 3 AND both in_ch and out_ch > 1 (not the initial or final conv)
            // The initial conv (1→64, k=7) and final conv (1024→512, k=3) are stride=1.
            // From moshi source: stride = kernel_size / 2 for each downsample layer.
            let stride = if kernel > 3 && out_ch > in_ch && in_ch > 1 {
                // This is a downsample layer. stride = kernel / 2 (from moshi convention)
                // k=8 → s=4, k=10 → s=5, k=12 → s=6, k=16 → s=8
                kernel / 2
            } else {
                1
            };

            let conv = load_conv1d_with_stride(
                weights,
                &format!("{prefix}.encoder.layers.{layer_idx}.conv"),
                stride,
                1,
            )?;
            encoder_layers.push(EncoderLayer::Conv(conv));
        } else if weights.contains_key(&block_key) {
            // It's a residual block
            let block = load_encoder_res_block(
                weights,
                &format!("{prefix}.encoder.layers.{layer_idx}"),
            )?;
            encoder_layers.push(EncoderLayer::ResBlock(block));
        } else {
            // This layer index has no weights — it's an ELU activation.
            // Only insert ELU if we've already loaded at least one layer
            // (the layer indices with no weights are 2, 5, 8, 11, 13).
            if !encoder_layers.is_empty() && layer_idx <= 14 {
                // Check: is there a next layer after this? (up to layer 14)
                let has_next = (layer_idx + 1..=14).any(|j| {
                    weights.contains_key(&format!("{prefix}.encoder.layers.{j}.conv.weight"))
                        || weights.contains_key(&format!("{prefix}.encoder.layers.{j}.block.1.conv.weight"))
                });
                if has_next {
                    encoder_layers.push(EncoderLayer::Elu);
                }
            }
        }
    }

    tracing::info!("Loaded {} SEANet encoder layers", encoder_layers.len());

    // ========================================================================
    // Encoder Transformer (8 layers, hidden=512, 8 heads)
    // ========================================================================
    let hidden_dim = 512;
    let num_heads = 8;

    let mut transformer_layers = Vec::new();
    for i in 0..8 {
        let layer_prefix = format!("{prefix}.encoder_transformer.layers.{i}");
        if weights.contains_key(&format!("{layer_prefix}.self_attn.q_proj.weight")) {
            transformer_layers.push(load_encoder_transformer_layer(
                weights,
                &layer_prefix,
                hidden_dim,
                num_heads,
            )?);
        }
    }
    tracing::info!("Loaded {} encoder transformer layers", transformer_layers.len());

    // ========================================================================
    // Downsample (25Hz → 12.5Hz)
    // ========================================================================
    let downsample = load_conv1d_with_stride_and_pad(
        weights,
        &format!("{prefix}.downsample.conv"),
        2,  // stride 2 for 25Hz → 12.5Hz
        1,
        true, // replicate padding (downsample uses pad_mode="replicate")
    )?;

    // ========================================================================
    // RVQ Quantizer
    // ========================================================================
    let sem_prefix = format!("{prefix}.quantizer.semantic_residual_vector_quantizer");
    let acou_prefix = format!("{prefix}.quantizer.acoustic_residual_vector_quantizer");

    let semantic_input_proj = load_conv1d_proj(weights, &format!("{sem_prefix}.input_proj"))?;
    let semantic_codebook = load_rvq_codebook(weights, &format!("{sem_prefix}.layers.0"))?;
    let semantic_output_proj = load_conv1d_proj(weights, &format!("{sem_prefix}.output_proj"))?;

    let acoustic_input_proj = load_conv1d_proj(weights, &format!("{acou_prefix}.input_proj"))?;
    let mut acoustic_codebooks = Vec::with_capacity(15);
    for i in 0..15 {
        // Try to load; some models may have fewer layers
        match load_rvq_codebook(weights, &format!("{acou_prefix}.layers.{i}")) {
            Ok(cb) => acoustic_codebooks.push(cb),
            Err(_) => break,
        }
    }
    let acoustic_output_proj = load_conv1d_proj(weights, &format!("{acou_prefix}.output_proj"))?;

    tracing::info!(
        "Loaded RVQ: 1 semantic + {} acoustic codebooks",
        acoustic_codebooks.len()
    );

    let quantizer = SplitRvq {
        semantic_input_proj,
        semantic_codebook,
        semantic_output_proj,
        acoustic_input_proj,
        acoustic_codebooks,
        acoustic_output_proj,
    };

    Ok(SpeechEncoder {
        encoder_layers,
        transformer_layers,
        downsample,
        quantizer,
    })
}

/// Check if speech encoder weights are present.
pub fn has_encoder_weights(weights: &HashMap<String, Array>) -> bool {
    weights.keys().any(|k| k.starts_with("encoder.encoder."))
}
