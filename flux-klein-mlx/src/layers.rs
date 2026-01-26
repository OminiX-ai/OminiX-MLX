//! FLUX transformer layers
//!
//! Ported from MLX Python implementation:
//! https://github.com/ml-explore/mlx-examples/blob/main/flux/flux/layers.py

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    module::{Module, ModuleParameters},
    nn::{Linear, LinearBuilder, RmsNorm},
    ops,
    Array,
};
use mlx_macros::ModuleParameters;

// ============================================================================
// LayerNorm (for AdaLN - without learnable parameters)
// ============================================================================

/// Layer normalization without learnable parameters (elementwise_affine=False)
/// Used in FLUX.2-klein's AdaLN modulation
pub fn layer_norm(x: &Array, eps: f32) -> Result<Array, Exception> {
    // Compute mean along last dimension
    let mean = ops::mean_axis(x, -1, true)?;

    // Compute variance: E[(x - mean)^2]
    let x_centered = ops::subtract(x, &mean)?;
    let x_sq = ops::multiply(&x_centered, &x_centered)?;
    let variance = ops::mean_axis(&x_sq, -1, true)?;

    // Normalize: (x - mean) / sqrt(variance + eps)
    let std_inv = ops::rsqrt(&ops::add(&variance, &array!(eps))?)?;
    ops::multiply(&x_centered, &std_inv)
}

// ============================================================================
// Modulation - Adaptive Layer Normalization components
// ============================================================================

/// Modulation layer for AdaLN-Zero
///
/// Produces shift, scale, and gate parameters from conditioning input.
/// For double modulation (used in DoubleStreamBlock), produces 6 outputs.
/// For single modulation (used in SingleStreamBlock), produces 3 outputs.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Modulation {
    #[param]
    pub linear: Linear,
    pub is_double: bool,
}

impl Modulation {
    /// Create a new Modulation layer
    ///
    /// # Arguments
    /// * `dim` - Hidden dimension
    /// * `is_double` - If true, output 6 modulation params (for double blocks)
    pub fn new(dim: i32, is_double: bool) -> Result<Self, Exception> {
        let multiplier = if is_double { 6 } else { 3 };
        let linear = LinearBuilder::new(dim, multiplier * dim).bias(false).build()?;
        Ok(Self { linear, is_double })
    }

    /// Forward pass
    ///
    /// Returns (shift, scale, gate) tuples. For double modulation, returns two tuples.
    pub fn forward(&mut self, x: &Array) -> Result<ModulationOutput, Exception> {
        // x: [batch, dim] -> [batch, multiplier * dim]
        let x = mlx_rs::nn::silu(x)?;
        let out = self.linear.forward(&x)?;

        if self.is_double {
            // Split into 6 parts
            let chunks = out.split(6, -1)?;
            Ok(ModulationOutput::Double {
                shift1: chunks[0].clone(),
                scale1: chunks[1].clone(),
                gate1: chunks[2].clone(),
                shift2: chunks[3].clone(),
                scale2: chunks[4].clone(),
                gate2: chunks[5].clone(),
            })
        } else {
            // Split into 3 parts
            let chunks = out.split(3, -1)?;
            Ok(ModulationOutput::Single {
                shift: chunks[0].clone(),
                scale: chunks[1].clone(),
                gate: chunks[2].clone(),
            })
        }
    }
}

/// Output of the Modulation layer
#[derive(Debug, Clone)]
pub enum ModulationOutput {
    Single {
        shift: Array,
        scale: Array,
        gate: Array,
    },
    Double {
        shift1: Array,
        scale1: Array,
        gate1: Array,
        shift2: Array,
        scale2: Array,
        gate2: Array,
    },
}

// ============================================================================
// QKNorm - Query-Key Normalization
// ============================================================================

/// Query-Key normalization layer
///
/// Applies separate RMSNorm to queries and keys before attention.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct QKNorm {
    #[param]
    pub query_norm: RmsNorm,
    #[param]
    pub key_norm: RmsNorm,
}

impl QKNorm {
    /// Create a new QKNorm layer
    ///
    /// # Arguments
    /// * `head_dim` - Dimension per attention head
    pub fn new(head_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            query_norm: RmsNorm::new(head_dim)?,
            key_norm: RmsNorm::new(head_dim)?,
        })
    }

    /// Normalize queries and keys
    pub fn forward(&mut self, q: &Array, k: &Array) -> Result<(Array, Array), Exception> {
        let q_norm = self.query_norm.forward(q)?;
        let k_norm = self.key_norm.forward(k)?;
        Ok((q_norm, k_norm))
    }
}

// ============================================================================
// Rotary Position Embedding (RoPE)
// ============================================================================

/// Apply rotary position embedding to input tensor
///
/// # Arguments
/// * `x` - Input tensor of shape [batch, seq, heads, head_dim]
/// * `freqs` - Precomputed frequency tensor with cos/sin stacked
///             Shape: [batch, seq, 1, head_dim, 2] where last dim is [cos, sin]
pub fn apply_rope(x: &Array, freqs: &Array) -> Result<Array, Exception> {
    let shape = x.shape();
    let batch = shape[0];
    let seq = shape[1];
    let heads = shape[2];
    let head_dim = shape[3];

    // freqs shape: [batch, seq, 1, head_dim, 2] where 2 = [cos, sin]
    // Extract cos and sin
    let cos_idx = Array::from_slice(&[0i32], &[1]);
    let sin_idx = Array::from_slice(&[1i32], &[1]);
    let cos_freqs = freqs.take_axis(&cos_idx, -1)?.squeeze_axes(&[-1])?; // [batch, seq, 1, head_dim]
    let sin_freqs = freqs.take_axis(&sin_idx, -1)?.squeeze_axes(&[-1])?; // [batch, seq, 1, head_dim]

    // Broadcast cos/sin to [batch, seq, heads, head_dim]
    let cos_freqs = ops::broadcast_to(&cos_freqs, &[batch, seq, heads, head_dim])?;
    let sin_freqs = ops::broadcast_to(&sin_freqs, &[batch, seq, heads, head_dim])?;

    // Apply 2D RoPE rotation in pairs: (x0, x1) -> (x0*cos - x1*sin, x1*cos + x0*sin)
    // Reshape x to [batch, seq, heads, head_dim/2, 2]
    let x_pairs = x.reshape(&[batch, seq, heads, head_dim / 2, 2])?;
    let x_split = x_pairs.split(2, -1)?;
    let x0 = x_split[0].squeeze_axes(&[-1])?; // [batch, seq, heads, head_dim/2]
    let x1 = x_split[1].squeeze_axes(&[-1])?;

    // Reshape cos/sin to [batch, seq, heads, head_dim/2, 2] and take first of pair
    let cos_pairs = cos_freqs.reshape(&[batch, seq, heads, head_dim / 2, 2])?;
    let sin_pairs = sin_freqs.reshape(&[batch, seq, heads, head_dim / 2, 2])?;
    let cos_split = cos_pairs.split(2, -1)?;
    let sin_split = sin_pairs.split(2, -1)?;
    let cos_val = cos_split[0].squeeze_axes(&[-1])?; // [batch, seq, heads, head_dim/2]
    let sin_val = sin_split[0].squeeze_axes(&[-1])?;

    // Complex rotation: (x0 + i*x1) * (cos + i*sin) = (x0*cos - x1*sin) + i*(x1*cos + x0*sin)
    let out0 = ops::subtract(&ops::multiply(&x0, &cos_val)?, &ops::multiply(&x1, &sin_val)?)?;
    let out1 = ops::add(&ops::multiply(&x1, &cos_val)?, &ops::multiply(&x0, &sin_val)?)?;

    // Stack back: [batch, seq, heads, head_dim/2, 2] -> [batch, seq, heads, head_dim]
    let out = ops::stack_axis(&[out0, out1], -1)?;
    out.reshape(shape)
}

/// Generate 2D rotary position embeddings for image patches
///
/// # Arguments
/// * `height` - Image height in patches
/// * `width` - Image width in patches
/// * `dim` - Embedding dimension (head_dim)
/// * `theta` - Base frequency (default 10000.0)
pub fn get_2d_rope_freqs(
    height: i32,
    width: i32,
    dim: i32,
    theta: f32,
) -> Result<Array, Exception> {
    let half_dim = dim / 2;

    // Create frequency bands
    let freq_seq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / dim as f32))
        .collect();
    let freqs = Array::from_slice(&freq_seq, &[half_dim]);

    // Create position indices
    let h_pos: Vec<f32> = (0..height).map(|i| i as f32).collect();
    let w_pos: Vec<f32> = (0..width).map(|i| i as f32).collect();
    let h_arr = Array::from_slice(&h_pos, &[height]);
    let w_arr = Array::from_slice(&w_pos, &[width]);

    // Compute position * frequency for height and width
    let h_freqs = ops::outer(&h_arr, &freqs)?; // [height, half_dim/2]
    let w_freqs = ops::outer(&w_arr, &freqs)?; // [width, half_dim/2]

    // Combine into 2D grid
    // For each (h, w) position, concatenate h_freqs and w_freqs
    let h_freqs_exp = ops::broadcast_to(&h_freqs.reshape(&[height, 1, half_dim])?, &[height, width, half_dim])?;
    let w_freqs_exp = ops::broadcast_to(&w_freqs.reshape(&[1, width, half_dim])?, &[height, width, half_dim])?;

    // Interleave or concatenate based on FLUX convention
    let combined = ops::concatenate_axis(&[h_freqs_exp, w_freqs_exp], -1)?;

    // Reshape to [height * width, dim]
    combined.reshape(&[height * width, dim])
}

// ============================================================================
// Timestep Embedding
// ============================================================================

/// Sinusoidal timestep embedding
///
/// # Arguments
/// * `t` - Timestep values [batch]
/// * `dim` - Embedding dimension
/// * `max_period` - Maximum period for frequencies (default 10000.0)
pub fn timestep_embedding(t: &Array, dim: i32, max_period: f32) -> Result<Array, Exception> {
    let half = dim / 2;

    // Create frequency bands: exp(-log(max_period) * i / half)
    let freq_seq: Vec<f32> = (0..half)
        .map(|i| (-max_period.ln() * i as f32 / half as f32).exp())
        .collect();
    let freqs = Array::from_slice(&freq_seq, &[1, half]);

    // t: [batch] -> [batch, 1]
    let t_exp = t.reshape(&[-1, 1])?;

    // args: [batch, half]
    let args = ops::multiply(&t_exp, &freqs)?;

    // Concatenate cos and sin
    let cos_emb = ops::cos(&args)?;
    let sin_emb = ops::sin(&args)?;
    let emb = ops::concatenate_axis(&[cos_emb, sin_emb], -1)?;

    // Handle odd dimension
    if dim % 2 == 1 {
        let zeros = ops::zeros::<f32>(&[emb.dim(0), 1])?;
        ops::concatenate_axis(&[emb, zeros], -1)
    } else {
        Ok(emb)
    }
}

// ============================================================================
// MLP Embedder
// ============================================================================

/// Two-layer MLP for embedding conditioning signals
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct MlpEmbedder {
    #[param]
    pub linear1: Linear,
    #[param]
    pub linear2: Linear,
}

impl MlpEmbedder {
    /// Create a new MLP embedder (no bias for FLUX.2-klein)
    ///
    /// # Arguments
    /// * `in_dim` - Input dimension
    /// * `hidden_dim` - Hidden/output dimension
    pub fn new(in_dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            linear1: LinearBuilder::new(in_dim, hidden_dim).bias(false).build()?,
            linear2: LinearBuilder::new(hidden_dim, hidden_dim).bias(false).build()?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let x = self.linear1.forward(x)?;
        let x = mlx_rs::nn::silu(&x)?;
        self.linear2.forward(&x)
    }
}

// ============================================================================
// Self-Attention with QK-Norm
// ============================================================================

/// Self-attention layer with QK normalization and RoPE support
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct SelfAttention {
    pub num_heads: i32,
    pub head_dim: i32,
    #[param]
    pub qkv: Linear,
    #[param]
    pub norm: QKNorm,
    #[param]
    pub proj: Linear,
}

impl SelfAttention {
    /// Create a new self-attention layer
    ///
    /// # Arguments
    /// * `dim` - Model dimension
    /// * `num_heads` - Number of attention heads
    pub fn new(dim: i32, num_heads: i32) -> Result<Self, Exception> {
        let head_dim = dim / num_heads;

        Ok(Self {
            num_heads,
            head_dim,
            qkv: LinearBuilder::new(dim, dim * 3).bias(false).build()?,
            norm: QKNorm::new(head_dim)?,
            proj: LinearBuilder::new(dim, dim).bias(false).build()?,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq, dim]
    /// * `rope_freqs` - Optional RoPE frequencies
    pub fn forward(
        &mut self,
        x: &Array,
        rope_freqs: Option<&Array>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // QKV projection
        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.reshape(&[batch, seq_len, 3, self.num_heads, self.head_dim])?;

        // Split into Q, K, V using take_axis on dimension 2
        let q = qkv.take_axis(&Array::from_slice(&[0i32], &[1]), 2)?.squeeze_axes(&[2])?;
        let k = qkv.take_axis(&Array::from_slice(&[1i32], &[1]), 2)?.squeeze_axes(&[2])?;
        let v = qkv.take_axis(&Array::from_slice(&[2i32], &[1]), 2)?.squeeze_axes(&[2])?;

        // Apply QK normalization
        let (q, k) = self.norm.forward(&q, &k)?;

        // Apply RoPE if provided
        let (q, k) = if let Some(freqs) = rope_freqs {
            (apply_rope(&q, freqs)?, apply_rope(&k, freqs)?)
        } else {
            (q, k)
        };

        // Scaled dot-product attention
        // q, k, v: [batch, seq, heads, head_dim]
        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let scale = (self.head_dim as f32).sqrt();
        let attn_weights = ops::matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?;
        let attn_weights = ops::divide(&attn_weights, &array!(scale))?;
        let attn_weights = ops::softmax_axis(&attn_weights, -1, None)?;

        let out = ops::matmul(&attn_weights, &v)?;

        // Transpose back and reshape: [batch, heads, seq, head_dim] -> [batch, seq, dim]
        let out = out.transpose_axes(&[0, 2, 1, 3])?;
        let out = out.reshape(&[batch, seq_len, -1])?;

        // Output projection
        self.proj.forward(&out)
    }
}

// ============================================================================
// Double Stream Block
// ============================================================================

/// Double stream transformer block for FLUX.2-klein
///
/// Processes image and text streams in parallel with joint attention.
/// Uses RMSNorm before modulation for numerical stability.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct DoubleStreamBlock {
    pub hidden_size: i32,
    pub num_heads: i32,
    pub mlp_ratio: f32,

    // Image stream
    #[param]
    pub img_mod: Modulation,
    #[param]
    pub img_norm1: RmsNorm,
    #[param]
    pub img_attn: SelfAttention,
    #[param]
    pub img_norm2: RmsNorm,
    #[param]
    pub img_mlp: Mlp,

    // Text stream
    #[param]
    pub txt_mod: Modulation,
    #[param]
    pub txt_norm1: RmsNorm,
    #[param]
    pub txt_attn: SelfAttention,
    #[param]
    pub txt_norm2: RmsNorm,
    #[param]
    pub txt_mlp: Mlp,
}

/// SwiGLU MLP for FLUX.2-klein
///
/// Uses gated linear unit: output = (gate * silu(gate)) * up
/// fc1 produces gate and up concatenated, fc2 projects back down
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Mlp {
    #[param]
    pub fc1: Linear,
    #[param]
    pub fc2: Linear,
    pub hidden_features: i32,
}

impl Mlp {
    /// Create SwiGLU MLP
    ///
    /// For FLUX.2-klein (SwiGLU), fc1 outputs gate+up (2 * hidden_features)
    /// and fc2 takes hidden_features and outputs in_features.
    pub fn new(in_features: i32, hidden_features: i32) -> Result<Self, Exception> {
        // SwiGLU: fc1 outputs gate+up = 2 * hidden_features
        Ok(Self {
            fc1: LinearBuilder::new(in_features, hidden_features * 2).bias(false).build()?,
            fc2: LinearBuilder::new(hidden_features, in_features).bias(false).build()?,
            hidden_features,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let proj = self.fc1.forward(x)?;
        // Split into gate and up
        let splits = proj.split_axis(&[self.hidden_features], -1)?;
        let gate = &splits[0];
        let up = &splits[1];
        // SwiGLU: silu(gate) * up
        let x = ops::multiply(&mlx_rs::nn::silu(gate)?, up)?;
        self.fc2.forward(&x)
    }
}

impl DoubleStreamBlock {
    /// Create a new double stream block
    pub fn new(hidden_size: i32, num_heads: i32, mlp_ratio: f32) -> Result<Self, Exception> {
        let mlp_hidden = (hidden_size as f32 * mlp_ratio) as i32;

        Ok(Self {
            hidden_size,
            num_heads,
            mlp_ratio,

            // Image stream (RMSNorm for stability, initialized to identity)
            img_mod: Modulation::new(hidden_size, true)?,
            img_norm1: RmsNorm::new(hidden_size)?,
            img_attn: SelfAttention::new(hidden_size, num_heads)?,
            img_norm2: RmsNorm::new(hidden_size)?,
            img_mlp: Mlp::new(hidden_size, mlp_hidden)?,

            // Text stream
            txt_mod: Modulation::new(hidden_size, true)?,
            txt_norm1: RmsNorm::new(hidden_size)?,
            txt_attn: SelfAttention::new(hidden_size, num_heads)?,
            txt_norm2: RmsNorm::new(hidden_size)?,
            txt_mlp: Mlp::new(hidden_size, mlp_hidden)?,
        })
    }

    /// Forward pass with joint attention
    ///
    /// Joint attention: image and text tokens attend to concatenated K/V from both streams.
    /// This allows cross-modal communication essential for text-to-image generation.
    ///
    /// # Arguments
    /// * `img` - Image tokens [batch, img_seq, hidden]
    /// * `txt` - Text tokens [batch, txt_seq, hidden]
    /// * `vec` - Conditioning vector [batch, hidden]
    /// * `img_rope` - RoPE frequencies for image [batch, img_seq, 1, head_dim, 2]
    /// * `txt_rope` - RoPE frequencies for text [batch, txt_seq, 1, head_dim, 2]
    pub fn forward(
        &mut self,
        img: &Array,
        txt: &Array,
        vec: &Array,
        img_rope: Option<&Array>,
        txt_rope: Option<&Array>,
    ) -> Result<(Array, Array), Exception> {
        let batch = img.dim(0);
        let img_seq = img.dim(1);
        let txt_seq = txt.dim(1);
        let num_heads = self.num_heads;
        let head_dim = self.hidden_size / num_heads;

        // Get modulation parameters
        let img_mod = self.img_mod.forward(vec)?;
        let txt_mod = self.txt_mod.forward(vec)?;

        let (img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2) =
            match img_mod {
                ModulationOutput::Double { shift1, scale1, gate1, shift2, scale2, gate2 } =>
                    (shift1, scale1, gate1, shift2, scale2, gate2),
                _ => unreachable!(),
            };

        let (txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2) =
            match txt_mod {
                ModulationOutput::Double { shift1, scale1, gate1, shift2, scale2, gate2 } =>
                    (shift1, scale1, gate1, shift2, scale2, gate2),
                _ => unreachable!(),
            };

        // Apply RMSNorm then modulation
        let img_modulated = modulate(&self.img_norm1.forward(img)?, &img_shift1, &img_scale1)?;
        let txt_modulated = modulate(&self.txt_norm1.forward(txt)?, &txt_shift1, &txt_scale1)?;

        // ========== JOINT ATTENTION ==========
        // Project to Q, K, V for both streams
        let img_qkv = self.img_attn.qkv.forward(&img_modulated)?;
        let txt_qkv = self.txt_attn.qkv.forward(&txt_modulated)?;

        // Reshape to [batch, seq, 3, heads, head_dim]
        let img_qkv = img_qkv.reshape(&[batch, img_seq, 3, num_heads, head_dim])?;
        let txt_qkv = txt_qkv.reshape(&[batch, txt_seq, 3, num_heads, head_dim])?;

        // Split into Q, K, V
        let idx0 = Array::from_slice(&[0i32], &[1]);
        let idx1 = Array::from_slice(&[1i32], &[1]);
        let idx2 = Array::from_slice(&[2i32], &[1]);

        let img_q = img_qkv.take_axis(&idx0, 2)?.squeeze_axes(&[2])?;
        let img_k = img_qkv.take_axis(&idx1, 2)?.squeeze_axes(&[2])?;
        let img_v = img_qkv.take_axis(&idx2, 2)?.squeeze_axes(&[2])?;

        let txt_q = txt_qkv.take_axis(&idx0, 2)?.squeeze_axes(&[2])?;
        let txt_k = txt_qkv.take_axis(&idx1, 2)?.squeeze_axes(&[2])?;
        let txt_v = txt_qkv.take_axis(&idx2, 2)?.squeeze_axes(&[2])?;

        // Apply QK normalization
        let (img_q, img_k) = self.img_attn.norm.forward(&img_q, &img_k)?;
        let (txt_q, txt_k) = self.txt_attn.norm.forward(&txt_q, &txt_k)?;

        // Apply RoPE if provided
        let (img_q, img_k) = if let Some(rope) = img_rope {
            (apply_rope(&img_q, rope)?, apply_rope(&img_k, rope)?)
        } else {
            (img_q, img_k)
        };

        let (txt_q, txt_k) = if let Some(rope) = txt_rope {
            (apply_rope(&txt_q, rope)?, apply_rope(&txt_k, rope)?)
        } else {
            (txt_q, txt_k)
        };

        // Concatenate K and V from both streams: [txt, img] order (matching flux.c)
        // Shape: [batch, txt_seq + img_seq, heads, head_dim]
        let combined_k = ops::concatenate_axis(&[txt_k, img_k], 1)?;
        let combined_v = ops::concatenate_axis(&[txt_v, img_v], 1)?;

        // Transpose for attention: [batch, heads, seq, head_dim]
        let img_q = img_q.transpose_axes(&[0, 2, 1, 3])?;
        let txt_q = txt_q.transpose_axes(&[0, 2, 1, 3])?;
        let combined_k = combined_k.transpose_axes(&[0, 2, 1, 3])?;
        let combined_v = combined_v.transpose_axes(&[0, 2, 1, 3])?;

        // Scaled dot-product attention
        let scale = (head_dim as f32).sqrt();

        // Image attends to combined K/V
        let img_attn_weights = ops::matmul(&img_q, &combined_k.transpose_axes(&[0, 1, 3, 2])?)?;
        let img_attn_weights = ops::divide(&img_attn_weights, &array!(scale))?;
        let img_attn_weights = ops::softmax_axis(&img_attn_weights, -1, None)?;
        let img_attn_out = ops::matmul(&img_attn_weights, &combined_v)?;

        // Text attends to combined K/V
        let txt_attn_weights = ops::matmul(&txt_q, &combined_k.transpose_axes(&[0, 1, 3, 2])?)?;
        let txt_attn_weights = ops::divide(&txt_attn_weights, &array!(scale))?;
        let txt_attn_weights = ops::softmax_axis(&txt_attn_weights, -1, None)?;
        let txt_attn_out = ops::matmul(&txt_attn_weights, &combined_v)?;

        // Transpose back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let img_attn_out = img_attn_out.transpose_axes(&[0, 2, 1, 3])?;
        let img_attn_out = img_attn_out.reshape(&[batch, img_seq, -1])?;
        let txt_attn_out = txt_attn_out.transpose_axes(&[0, 2, 1, 3])?;
        let txt_attn_out = txt_attn_out.reshape(&[batch, txt_seq, -1])?;

        // Output projections
        let img_attn_out = self.img_attn.proj.forward(&img_attn_out)?;
        let txt_attn_out = self.txt_attn.proj.forward(&txt_attn_out)?;

        // Apply gate and residual for attention
        let img = ops::add(img, &gate(&img_attn_out, &img_gate1)?)?;
        let txt = ops::add(txt, &gate(&txt_attn_out, &txt_gate1)?)?;

        // ========== MLP ==========
        // MLP with RMSNorm then modulation
        let img_mlp_in = modulate(&self.img_norm2.forward(&img)?, &img_shift2, &img_scale2)?;
        let txt_mlp_in = modulate(&self.txt_norm2.forward(&txt)?, &txt_shift2, &txt_scale2)?;

        let img_mlp_out = self.img_mlp.forward(&img_mlp_in)?;
        let txt_mlp_out = self.txt_mlp.forward(&txt_mlp_in)?;

        // Apply gate and residual for MLP
        let img = ops::add(&img, &gate(&img_mlp_out, &img_gate2)?)?;
        let txt = ops::add(&txt, &gate(&txt_mlp_out, &txt_gate2)?)?;

        Ok((img, txt))
    }
}

// ============================================================================
// Single Stream Block
// ============================================================================

/// Single stream transformer block for FLUX.2-klein
///
/// Uses fused QKV + SwiGLU MLP projections.
/// linear1 outputs: QKV (3 * hidden) + gate (mlp_hidden) + up (mlp_hidden)
/// linear2 takes: attention (hidden) + mlp_out (mlp_hidden) -> hidden
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct SingleStreamBlock {
    pub hidden_size: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    pub mlp_hidden: i32,

    #[param]
    pub mod_layer: Modulation,
    #[param]
    pub pre_norm: RmsNorm,

    // Fused QKV + SwiGLU (gate + up) projection
    #[param]
    pub linear1: Linear,
    #[param]
    pub linear2: Linear,

    #[param]
    pub norm_q: RmsNorm,
    #[param]
    pub norm_k: RmsNorm,
}

impl SingleStreamBlock {
    /// Create a new single stream block
    pub fn new(hidden_size: i32, num_heads: i32, mlp_ratio: f32) -> Result<Self, Exception> {
        let head_dim = hidden_size / num_heads;
        let mlp_hidden = (hidden_size as f32 * mlp_ratio) as i32;

        // Fused projection for FLUX.2-klein:
        // QKV (3 * hidden) + gate (mlp_hidden) + up (mlp_hidden) = 9216 + 9216 + 9216 = 27648
        let linear1_out = hidden_size * 3 + mlp_hidden * 2;

        // linear2 input: attention (hidden) + mlp_out (mlp_hidden) = 3072 + 9216 = 12288
        let linear2_in = hidden_size + mlp_hidden;

        Ok(Self {
            hidden_size,
            num_heads,
            head_dim,
            mlp_hidden,

            mod_layer: Modulation::new(hidden_size, false)?,
            pre_norm: RmsNorm::new(hidden_size)?,

            linear1: LinearBuilder::new(hidden_size, linear1_out).bias(false).build()?,
            linear2: LinearBuilder::new(linear2_in, hidden_size).bias(false).build()?,

            norm_q: RmsNorm::new(head_dim)?,
            norm_k: RmsNorm::new(head_dim)?,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Combined tokens [batch, seq, hidden]
    /// * `vec` - Conditioning vector [batch, hidden]
    /// * `rope_freqs` - RoPE frequencies
    pub fn forward(
        &mut self,
        x: &Array,
        vec: &Array,
        rope_freqs: Option<&Array>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Get modulation parameters
        let mod_out = self.mod_layer.forward(vec)?;
        let (shift, scale, gate_val) = match mod_out {
            ModulationOutput::Single { shift, scale, gate } => (shift, scale, gate),
            _ => unreachable!(),
        };

        // Apply RMSNorm then modulation
        let x_mod = modulate(&self.pre_norm.forward(x)?, &shift, &scale)?;

        // Fused projection
        let proj = self.linear1.forward(&x_mod)?;

        // Split: QKV (9216) + gate (9216) + up (9216)
        let qkv_size = self.hidden_size * 3;
        let splits = proj.split_axis(&[qkv_size, qkv_size + self.mlp_hidden], -1)?;
        let qkv = &splits[0];
        let gate_proj = &splits[1];
        let up_proj = &splits[2];

        // Reshape QKV
        let qkv = qkv.reshape(&[batch, seq_len, 3, self.num_heads, self.head_dim])?;
        let q = qkv.take_axis(&Array::from_slice(&[0i32], &[1]), 2)?.squeeze_axes(&[2])?;
        let k = qkv.take_axis(&Array::from_slice(&[1i32], &[1]), 2)?.squeeze_axes(&[2])?;
        let v = qkv.take_axis(&Array::from_slice(&[2i32], &[1]), 2)?.squeeze_axes(&[2])?;

        // QK norm
        let q = self.norm_q.forward(&q)?;
        let k = self.norm_k.forward(&k)?;

        // Apply RoPE
        let (q, k) = if let Some(freqs) = rope_freqs {
            (apply_rope(&q, freqs)?, apply_rope(&k, freqs)?)
        } else {
            (q, k)
        };

        // Attention
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let scale = (self.head_dim as f32).sqrt();
        let attn = ops::matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?;
        let attn = ops::divide(&attn, &array!(scale))?;
        let attn = ops::softmax_axis(&attn, -1, None)?;
        let attn_out = ops::matmul(&attn, &v)?;

        let attn_out = attn_out.transpose_axes(&[0, 2, 1, 3])?;
        let attn_out = attn_out.reshape(&[batch, seq_len, -1])?;

        // SwiGLU MLP: silu(gate) * up
        let mlp_out = ops::multiply(&mlx_rs::nn::silu(gate_proj)?, up_proj)?;

        // Concatenate attention output and MLP output, then project
        let combined = ops::concatenate_axis(&[attn_out, mlp_out], -1)?;
        let out = self.linear2.forward(&combined)?;

        // Gate and residual
        let out = gate(&out, &gate_val)?;
        ops::add(x, &out)
    }
}

// ============================================================================
// Final Layer
// ============================================================================

/// Final output layer with AdaLN modulation
///
/// FLUX.2-klein applies modulation directly without norm.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct FinalLayer {
    #[param]
    pub linear: Linear,
    #[param]
    pub ada_linear: Linear,
}

impl FinalLayer {
    /// Create final layer
    ///
    /// # Arguments
    /// * `hidden_size` - Model hidden dimension
    /// * `patch_size` - Patch size
    /// * `out_channels` - Output channels (latent channels)
    pub fn new(hidden_size: i32, patch_size: i32, out_channels: i32) -> Result<Self, Exception> {
        let output_dim = patch_size * patch_size * out_channels;

        Ok(Self {
            linear: LinearBuilder::new(hidden_size, output_dim).bias(false).build()?,
            ada_linear: LinearBuilder::new(hidden_size, hidden_size * 2).bias(false).build()?,
        })
    }

    pub fn forward(&mut self, x: &Array, vec: &Array) -> Result<Array, Exception> {
        // Get scale and shift from conditioning
        // Note: flux.c has scale FIRST, shift SECOND
        let ada = mlx_rs::nn::silu(vec)?;
        let ada = self.ada_linear.forward(&ada)?;
        let chunks = ada.split(2, -1)?;
        let scale = &chunks[0];  // scale is first half
        let shift = &chunks[1];  // shift is second half

        // Apply modulation directly (no pre-norm in FLUX.2-klein)
        let x = modulate(x, shift, scale)?;
        self.linear.forward(&x)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Apply AdaLN modulation: (1 + scale) * x + shift
fn modulate(x: &Array, shift: &Array, scale: &Array) -> Result<Array, Exception> {
    // Expand shift and scale for broadcasting: [batch, hidden] -> [batch, 1, hidden]
    let shift = shift.reshape(&[shift.dim(0), 1, -1])?;
    let scale = scale.reshape(&[scale.dim(0), 1, -1])?;

    let scaled = ops::multiply(x, &ops::add(&array!(1.0), &scale)?)?;
    ops::add(&scaled, &shift)
}

/// Apply gating: x * gate (with broadcasting)
fn gate(x: &Array, gate: &Array) -> Result<Array, Exception> {
    // Expand gate for broadcasting: [batch, hidden] -> [batch, 1, hidden]
    let gate = gate.reshape(&[gate.dim(0), 1, -1])?;
    ops::multiply(x, &gate)
}

/// Check if array contains NaN values
fn has_nan_arr(arr: &Array) -> bool {
    let flat = match arr.reshape(&[-1]) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let data: Vec<f32> = match flat.try_as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => return false,
    };
    data.iter().any(|x| x.is_nan())
}

/// Get min and max values of array
fn get_range(arr: &Array) -> (f32, f32) {
    let flat = match arr.reshape(&[-1]) {
        Ok(f) => f,
        Err(_) => return (f32::NAN, f32::NAN),
    };
    let data: Vec<f32> = match flat.try_as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => return (f32::NAN, f32::NAN),
    };
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modulation() {
        let mut mod_layer = Modulation::new(64, false).unwrap();
        let x = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[2, 64], None).unwrap();
        let out = mod_layer.forward(&x).unwrap();

        match out {
            ModulationOutput::Single { shift, scale, gate } => {
                assert_eq!(shift.shape(), &[2, 64]);
                assert_eq!(scale.shape(), &[2, 64]);
                assert_eq!(gate.shape(), &[2, 64]);
            }
            _ => panic!("Expected single modulation output"),
        }
    }

    #[test]
    fn test_qk_norm() {
        let mut norm = QKNorm::new(64).unwrap();
        let q = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[2, 16, 8, 64], None).unwrap();
        let k = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[2, 16, 8, 64], None).unwrap();
        let (q_norm, k_norm) = norm.forward(&q, &k).unwrap();

        assert_eq!(q_norm.shape(), &[2, 16, 8, 64]);
        assert_eq!(k_norm.shape(), &[2, 16, 8, 64]);
    }

    #[test]
    fn test_timestep_embedding() {
        let t = Array::from_slice(&[0.0f32, 0.5, 1.0], &[3]);
        let emb = timestep_embedding(&t, 256, 10000.0).unwrap();
        assert_eq!(emb.shape(), &[3, 256]);
    }
}
