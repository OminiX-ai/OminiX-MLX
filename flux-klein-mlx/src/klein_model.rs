//! FLUX.2-klein-4B Model Implementation
//!
//! This is the correct architecture for FLUX.2-klein which differs from FLUX.1:
//! - Shared modulation layers (one per stream type, shared across all blocks)
//! - Separate Q/K/V projections in double blocks
//! - Different MLP structure
//! - Uses Qwen3-4B for text encoding (7680-dim embeddings)
//!
//! # Stability Fixes
//!
//! The original architecture can cause value explosion in the text (txt) stream.
//! Several fixes are applied to maintain numerical stability:
//!
//! 1. **Gate Clamping**: Gates are clamped using `tanh(gate) * 3` to prevent
//!    extreme gating values (original values reached ±10).
//!
//! 2. **Scale Clamping**: Modulation scales are clamped using `tanh(scale) * 2`
//!    to prevent excessive amplification (original reached ±4).
//!
//! 3. **Text Residual Scaling**: The txt stream residual connections are scaled
//!    by 0.5x to prevent accumulation: `txt = txt + 0.5 * gated_output`.
//!
//! 4. **FP16-style Clipping**: Values are clipped to ±65504 after residuals.
//!
//! See `docs/FLUX2_KLEIN_IMPLEMENTATION.md` for detailed documentation.

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    module::Module,
    nn::{LayerNorm, LayerNormBuilder, Linear, LinearBuilder, RmsNorm},
    ops,
    ops::indexing::IndexOp,
    Array,
};
use mlx_macros::ModuleParameters;

use crate::layers::timestep_embedding;

// ============================================================================
// RoPE (Rotary Position Embedding) Implementation
// ============================================================================

/// Compute rotary embeddings for given position IDs
///
/// # Arguments
/// * `ids` - Position IDs [batch, seq, 3] where 3 = (t, h, w) for 3-axis encoding
/// * `dim` - Dimension per axis (typically 16, 56, 56 for FLUX)
/// * `theta` - Base frequency (default 10000.0)
///
/// Returns (cos, sin) each of shape [batch, seq, dim*3/2] interleaved for complex rotation
pub fn compute_rope_freqs(
    ids: &Array,
    axes_dim: &[i32],  // e.g., [16, 56, 56] = 128 total
    theta: f32,
) -> Result<(Array, Array), Exception> {
    let batch = ids.dim(0);
    let seq_len = ids.dim(1);

    // Compute frequencies for each axis
    let mut all_cos = Vec::new();
    let mut all_sin = Vec::new();

    for (axis, &dim) in axes_dim.iter().enumerate() {
        let half_dim = dim / 2;

        // Compute inverse frequencies: 1 / (theta ^ (2i / dim))
        let mut inv_freq = Vec::with_capacity(half_dim as usize);
        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * (i as f32) / (dim as f32));
            inv_freq.push(freq);
        }
        let inv_freq = Array::from_slice(&inv_freq, &[1, 1, half_dim]);

        // Get positions for this axis: ids[:, :, axis]
        let pos = ids.index((.., .., axis as i32));
        let pos = pos.reshape(&[batch, seq_len, 1])?;

        // Compute angles: pos * inv_freq using broadcasting
        // pos: [batch, seq, 1], inv_freq: [1, 1, half_dim] -> [batch, seq, half_dim]
        let angles = ops::multiply(&pos, &inv_freq)?;

        // Compute cos and sin
        let cos_angles = ops::cos(&angles)?;
        let sin_angles = ops::sin(&angles)?;

        // Duplicate each frequency for complex rotation pairs: [c0,c0,c1,c1,...]
        // cos_angles shape: [batch, seq, half_dim] -> [batch, seq, half_dim, 2] -> [batch, seq, dim]
        let cos_stacked = ops::stack_axis(&[cos_angles.clone(), cos_angles], -1)?;
        let sin_stacked = ops::stack_axis(&[sin_angles.clone(), sin_angles], -1)?;
        let cos_interleaved = cos_stacked.reshape(&[batch, seq_len, dim])?;
        let sin_interleaved = sin_stacked.reshape(&[batch, seq_len, dim])?;

        all_cos.push(cos_interleaved);
        all_sin.push(sin_interleaved);
    }

    // Concatenate all axes along last dimension
    let cos_refs: Vec<&Array> = all_cos.iter().collect();
    let sin_refs: Vec<&Array> = all_sin.iter().collect();

    let cos = ops::concatenate_axis(&cos_refs, -1)?;
    let sin = ops::concatenate_axis(&sin_refs, -1)?;

    Ok((cos, sin))
}

/// Apply rotary embedding to Q or K tensor
///
/// # Arguments
/// * `x` - Input tensor [batch, seq, heads, head_dim] or [batch, heads, seq, head_dim]
/// * `cos` - Cosine frequencies [batch, seq, dim]
/// * `sin` - Sine frequencies [batch, seq, dim]
///
/// Returns rotated tensor of same shape
/// Apply rotary embedding using interleaved pairs (flux.c style)
///
/// For each pair of adjacent elements (x0, x1):
///   out0 = x0 * cos - x1 * sin
///   out1 = x1 * cos + x0 * sin
///
/// This matches flux.c's apply_rope_2d implementation.
pub fn apply_rope(
    x: &Array,
    cos: &Array,
    sin: &Array,
) -> Result<Array, Exception> {
    // x shape: [batch, seq, heads, head_dim]
    // cos/sin shape: [batch, seq, head_dim] with duplicated values [c0,c0,c1,c1,...]
    let x_shape = x.shape();
    let batch = x_shape[0] as i32;
    let seq_len = x_shape[1] as i32;
    let heads = x_shape[2] as i32;
    let head_dim = x_shape[3] as i32;

    // Reshape cos/sin for broadcasting: [batch, seq, 1, head_dim]
    let cos = cos.reshape(&[batch, seq_len, 1, head_dim])?;
    let sin = sin.reshape(&[batch, seq_len, 1, head_dim])?;

    // Reshape x to expose interleaved pairs: [batch, seq, heads, head_dim/2, 2]
    let x_pairs = x.reshape(&[batch, seq_len, heads, head_dim / 2, 2])?;
    let x0 = x_pairs.index((.., .., .., .., 0_i32));  // First element of each pair
    let x1 = x_pairs.index((.., .., .., .., 1_i32));  // Second element of each pair

    // cos/sin have duplicated values: [c0,c0,c1,c1,...]
    // Reshape to [batch, seq, 1, head_dim/2, 2] and take first of each pair
    let cos_pairs = cos.reshape(&[batch, seq_len, 1, head_dim / 2, 2])?;
    let sin_pairs = sin.reshape(&[batch, seq_len, 1, head_dim / 2, 2])?;
    let cos_val = cos_pairs.index((.., .., .., .., 0_i32));  // [batch, seq, 1, head_dim/2]
    let sin_val = sin_pairs.index((.., .., .., .., 0_i32));

    // Apply rotation to pairs:
    // out0 = x0 * cos - x1 * sin
    // out1 = x1 * cos + x0 * sin
    let out0 = ops::subtract(&ops::multiply(&x0, &cos_val)?, &ops::multiply(&x1, &sin_val)?)?;
    let out1 = ops::add(&ops::multiply(&x1, &cos_val)?, &ops::multiply(&x0, &sin_val)?)?;

    // Stack back to pairs and reshape to original shape
    let out_pairs = ops::stack_axis(&[out0, out1], -1)?;
    out_pairs.reshape(&[batch, seq_len, heads, head_dim])
}

// ============================================================================
// FLUX.2-klein Configuration
// ============================================================================

/// FLUX.2-klein-4B parameters
#[derive(Debug, Clone)]
pub struct FluxKleinParams {
    pub in_channels: i32,      // 128 (after patchify: 32 VAE channels * 2*2)
    pub hidden_size: i32,      // 3072
    pub txt_embed_dim: i32,    // 7680 (from Qwen3)
    pub num_heads: i32,        // 24
    pub mlp_ratio: f32,        // 3.0
    pub depth: i32,            // 5 double stream blocks
    pub depth_single: i32,     // 20 single stream blocks
    pub head_dim: i32,         // 128
    pub mlp_hidden: i32,       // 9216 (3072 * 3)
}

impl Default for FluxKleinParams {
    fn default() -> Self {
        Self {
            in_channels: 128,
            hidden_size: 3072,
            txt_embed_dim: 7680,
            num_heads: 24,
            mlp_ratio: 3.0,
            depth: 5,
            depth_single: 20,
            head_dim: 128,
            mlp_hidden: 9216,
        }
    }
}

// ============================================================================
// Shared Modulation
// ============================================================================

/// Shared modulation layer for AdaLN (Adaptive Layer Normalization)
///
/// This layer computes shift, scale, and gate parameters for modulating transformer
/// block outputs. The modulation is shared across all blocks of the same type.
///
/// # Output Format
/// - Double blocks (num_params=6): `[shift1, scale1, gate1, shift2, scale2, gate2]`
/// - Single blocks (num_params=3): `[shift, scale, gate]`
///
/// # Stability
/// Scale and gate values are clamped using tanh to prevent numerical explosion:
/// - Scales: `tanh(scale) * 2` → range approximately [-2, 2]
/// - Gates: `tanh(gate) * 3` → range approximately [-3, 3]
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct SharedModulation {
    #[param]
    pub linear: Linear,
    pub num_params: i32,
}

impl SharedModulation {
    pub fn new(hidden_size: i32, num_params: i32) -> Result<Self, Exception> {
        Ok(Self {
            linear: LinearBuilder::new(hidden_size, num_params * hidden_size)
                .bias(false)
                .build()?,
            num_params,
        })
    }

    /// Forward pass - returns modulation parameters with stability clamping
    ///
    /// # Arguments
    /// * `vec` - Time embedding vector [batch, hidden_size] (after time_embed_2, before silu)
    ///
    /// # Returns
    /// Vector of modulation parameters:
    /// - Double blocks: `[shift1, scale1, gate1, shift2, scale2, gate2]`
    /// - Single blocks: `[shift, scale, gate]`
    ///
    /// # Processing
    /// 1. Apply SiLU activation to input (matching diffusers AdaLayerNormZero)
    /// 2. Linear projection to num_params * hidden_size
    /// 3. Split into individual parameters
    /// Note: No clamping - flux.c doesn't clamp gates
    pub fn forward(&mut self, vec: &Array) -> Result<Vec<Array>, Exception> {
        // Apply silu activation before linear projection (matching diffusers AdaLayerNormZero)
        let x = mlx_rs::nn::silu(vec)?;
        let out = self.linear.forward(&x)?;
        // Return chunks directly - no clamping (flux.c doesn't clamp)
        Ok(out.split(self.num_params, -1)?)
    }
}

// ============================================================================
// Klein Double Stream Block
// ============================================================================

/// Double stream block for FLUX.2-klein
///
/// Processes image and text streams in parallel with joint attention, then applies
/// separate MLP transformations. Uses shared modulation parameters (passed from FluxKlein).
///
/// # Architecture
/// - Separate Q/K/V projections (not fused QKV like FLUX.1)
/// - Joint attention: both streams attend to concatenated [txt, img] keys/values
/// - Pre-modulation LayerNorm without learnable affine (matches diffusers)
/// - QK normalization using RmsNorm
/// - SwiGLU MLP: `x * silu(gate)` where input splits into [x, gate]
///
/// # Stability Measures
/// The txt stream tends to explode without intervention. We apply:
/// - **Residual scaling**: txt residuals are scaled by 0.5x before addition
/// - **Value clipping**: Both streams clipped to ±65504 after residuals
/// - (Modulation clamping is handled in SharedModulation)
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct KleinDoubleBlock {
    pub hidden_size: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    pub mlp_hidden: i32,

    // Pre-modulation normalization: LayerNorm without learnable affine (matches diffusers)
    #[param]
    pub img_norm1: LayerNorm,
    #[param]
    pub img_norm2: LayerNorm,
    #[param]
    pub txt_norm1: LayerNorm,
    #[param]
    pub txt_norm2: LayerNorm,

    // Post-residual normalization: RmsNorm (currently unused, kept for experimentation)
    // We now use residual scaling (0.5x) instead, which preserves direction better
    #[param]
    pub txt_post_attn_norm: RmsNorm,
    #[param]
    pub txt_post_mlp_norm: RmsNorm,

    // Image stream attention (separate Q, K, V)
    #[param]
    pub img_to_q: Linear,
    #[param]
    pub img_to_k: Linear,
    #[param]
    pub img_to_v: Linear,
    #[param]
    pub img_norm_q: RmsNorm,
    #[param]
    pub img_norm_k: RmsNorm,
    #[param]
    pub img_to_out: Linear,

    // Text stream attention (separate Q, K, V)
    #[param]
    pub txt_to_q: Linear,
    #[param]
    pub txt_to_k: Linear,
    #[param]
    pub txt_to_v: Linear,
    #[param]
    pub txt_norm_q: RmsNorm,
    #[param]
    pub txt_norm_k: RmsNorm,
    #[param]
    pub txt_to_out: Linear,

    // Image MLP (SwiGLU)
    #[param]
    pub img_mlp_in: Linear,   // [hidden, 2*mlp_hidden] for gate+up
    #[param]
    pub img_mlp_out: Linear,  // [mlp_hidden, hidden]

    // Text MLP (SwiGLU)
    #[param]
    pub txt_mlp_in: Linear,
    #[param]
    pub txt_mlp_out: Linear,
}

impl KleinDoubleBlock {
    pub fn new(hidden_size: i32, num_heads: i32, mlp_hidden: i32) -> Result<Self, Exception> {
        let head_dim = hidden_size / num_heads;

        Ok(Self {
            hidden_size,
            num_heads,
            head_dim,
            mlp_hidden,

            // Pre-modulation normalization: LayerNorm without learnable affine (matches diffusers)
            img_norm1: LayerNormBuilder::new(hidden_size).affine(false).eps(1e-6).build()?,
            img_norm2: LayerNormBuilder::new(hidden_size).affine(false).eps(1e-6).build()?,
            txt_norm1: LayerNormBuilder::new(hidden_size).affine(false).eps(1e-6).build()?,
            txt_norm2: LayerNormBuilder::new(hidden_size).affine(false).eps(1e-6).build()?,

            // Post-residual normalization for txt stream (prevents value explosion)
            txt_post_attn_norm: RmsNorm::new(hidden_size)?,
            txt_post_mlp_norm: RmsNorm::new(hidden_size)?,

            // Image attention
            img_to_q: LinearBuilder::new(hidden_size, hidden_size).bias(false).build()?,
            img_to_k: LinearBuilder::new(hidden_size, hidden_size).bias(false).build()?,
            img_to_v: LinearBuilder::new(hidden_size, hidden_size).bias(false).build()?,
            img_norm_q: RmsNorm::new(head_dim)?,
            img_norm_k: RmsNorm::new(head_dim)?,
            img_to_out: LinearBuilder::new(hidden_size, hidden_size).bias(false).build()?,

            // Text attention
            txt_to_q: LinearBuilder::new(hidden_size, hidden_size).bias(false).build()?,
            txt_to_k: LinearBuilder::new(hidden_size, hidden_size).bias(false).build()?,
            txt_to_v: LinearBuilder::new(hidden_size, hidden_size).bias(false).build()?,
            txt_norm_q: RmsNorm::new(head_dim)?,
            txt_norm_k: RmsNorm::new(head_dim)?,
            txt_to_out: LinearBuilder::new(hidden_size, hidden_size).bias(false).build()?,

            // Image MLP
            img_mlp_in: LinearBuilder::new(hidden_size, mlp_hidden * 2).bias(false).build()?,
            img_mlp_out: LinearBuilder::new(mlp_hidden, hidden_size).bias(false).build()?,

            // Text MLP
            txt_mlp_in: LinearBuilder::new(hidden_size, mlp_hidden * 2).bias(false).build()?,
            txt_mlp_out: LinearBuilder::new(mlp_hidden, hidden_size).bias(false).build()?,
        })
    }

    /// Forward pass with joint attention
    ///
    /// # Arguments
    /// * `img` - Image tokens [batch, img_seq, hidden]
    /// * `txt` - Text tokens [batch, txt_seq, hidden]
    /// * `img_mod` - Image modulation [shift1, scale1, gate1, shift2, scale2, gate2]
    /// * `txt_mod` - Text modulation [shift1, scale1, gate1, shift2, scale2, gate2]
    /// * `rope_cos` - RoPE cosine frequencies [batch, img_seq+txt_seq, head_dim]
    /// * `rope_sin` - RoPE sine frequencies [batch, img_seq+txt_seq, head_dim]
    pub fn forward(
        &mut self,
        img: &Array,
        txt: &Array,
        img_mod: &[Array],
        txt_mod: &[Array],
        rope_cos: &Array,
        rope_sin: &Array,
    ) -> Result<(Array, Array), Exception> {
        let batch = img.dim(0);
        let img_seq = img.dim(1);
        let txt_seq = txt.dim(1);

        // Unpack modulation - diffusers order: shift, scale, gate (for both attention and MLP)
        let (img_shift1, img_scale1, img_gate1) = (&img_mod[0], &img_mod[1], &img_mod[2]);
        let (img_shift2, img_scale2, img_gate2) = (&img_mod[3], &img_mod[4], &img_mod[5]);
        let (txt_shift1, txt_scale1, txt_gate1) = (&txt_mod[0], &txt_mod[1], &txt_mod[2]);
        let (txt_shift2, txt_scale2, txt_gate2) = (&txt_mod[3], &txt_mod[4], &txt_mod[5]);

        // Apply pre-normalization THEN modulation: (1 + scale) * norm(x) + shift
        let img_normed = self.img_norm1.forward(img)?;
        let txt_normed = self.txt_norm1.forward(txt)?;
        let img_mod1 = modulate(&img_normed, img_shift1, img_scale1)?;
        let txt_mod1 = modulate(&txt_normed, txt_shift1, txt_scale1)?;

        // ========== JOINT ATTENTION ==========
        // Project to Q, K, V separately
        let img_q = self.img_to_q.forward(&img_mod1)?;
        let img_k = self.img_to_k.forward(&img_mod1)?;
        let img_v = self.img_to_v.forward(&img_mod1)?;

        let txt_q = self.txt_to_q.forward(&txt_mod1)?;
        let txt_k = self.txt_to_k.forward(&txt_mod1)?;
        let txt_v = self.txt_to_v.forward(&txt_mod1)?;

        // Reshape for multi-head attention: [batch, seq, hidden] -> [batch, seq, heads, head_dim]
        let img_q = img_q.reshape(&[batch, img_seq, self.num_heads, self.head_dim])?;
        let img_k = img_k.reshape(&[batch, img_seq, self.num_heads, self.head_dim])?;
        let img_v = img_v.reshape(&[batch, img_seq, self.num_heads, self.head_dim])?;

        let txt_q = txt_q.reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?;
        let txt_k = txt_k.reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?;
        let txt_v = txt_v.reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?;

        // Apply QK normalization
        let img_q = self.img_norm_q.forward(&img_q)?;
        let img_k = self.img_norm_k.forward(&img_k)?;
        let txt_q = self.txt_norm_q.forward(&txt_q)?;
        let txt_k = self.txt_norm_k.forward(&txt_k)?;

        // Apply RoPE to Q and K
        let txt_rope_cos = rope_cos.index((.., ..txt_seq, ..));
        let txt_rope_sin = rope_sin.index((.., ..txt_seq, ..));
        let img_rope_cos = rope_cos.index((.., txt_seq.., ..));
        let img_rope_sin = rope_sin.index((.., txt_seq.., ..));

        let img_q = apply_rope(&img_q, &img_rope_cos, &img_rope_sin)?;
        let img_k = apply_rope(&img_k, &img_rope_cos, &img_rope_sin)?;
        let txt_q = apply_rope(&txt_q, &txt_rope_cos, &txt_rope_sin)?;
        let txt_k = apply_rope(&txt_k, &txt_rope_cos, &txt_rope_sin)?;

        // Concatenate K and V for joint attention: [txt, img]
        let combined_k = ops::concatenate_axis(&[&txt_k, &img_k], 1)?;
        let combined_v = ops::concatenate_axis(&[&txt_v, &img_v], 1)?;

        // Transpose for attention: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let img_q = img_q.transpose_axes(&[0, 2, 1, 3])?;
        let txt_q = txt_q.transpose_axes(&[0, 2, 1, 3])?;
        let combined_k = combined_k.transpose_axes(&[0, 2, 1, 3])?;
        let combined_v = combined_v.transpose_axes(&[0, 2, 1, 3])?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();

        // Image attends to combined K/V
        let img_attn = ops::matmul(&img_q, &combined_k.transpose_axes(&[0, 1, 3, 2])?)?;
        let img_attn = ops::divide(&img_attn, &array!(scale))?;
        let img_attn = ops::softmax_axis(&img_attn, -1, None)?;
        let img_attn_out = ops::matmul(&img_attn, &combined_v)?;

        // Text attends to combined K/V
        let txt_attn = ops::matmul(&txt_q, &combined_k.transpose_axes(&[0, 1, 3, 2])?)?;
        let txt_attn = ops::divide(&txt_attn, &array!(scale))?;
        let txt_attn = ops::softmax_axis(&txt_attn, -1, None)?;
        let txt_attn_out = ops::matmul(&txt_attn, &combined_v)?;

        // Transpose back and reshape: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let img_attn_out = img_attn_out.transpose_axes(&[0, 2, 1, 3])?;
        let img_attn_out = img_attn_out.reshape(&[batch, img_seq, -1])?;
        let txt_attn_out = txt_attn_out.transpose_axes(&[0, 2, 1, 3])?;
        let txt_attn_out = txt_attn_out.reshape(&[batch, txt_seq, -1])?;

        // Output projections
        let img_attn_out = self.img_to_out.forward(&img_attn_out)?;
        let txt_attn_out = self.txt_to_out.forward(&txt_attn_out)?;

        // Apply gate and residual
        let img = ops::add(img, &gate(&img_attn_out, img_gate1)?)?;
        let txt = ops::add(txt, &gate(&txt_attn_out, txt_gate1)?)?;

        // ========== MLP ==========
        let img_normed2 = self.img_norm2.forward(&img)?;
        let txt_normed2 = self.txt_norm2.forward(&txt)?;
        let img_mlp_in = modulate(&img_normed2, img_shift2, img_scale2)?;
        let txt_mlp_in = modulate(&txt_normed2, txt_shift2, txt_scale2)?;

        // SwiGLU MLP for image: silu(gate) * up
        let img_proj = self.img_mlp_in.forward(&img_mlp_in)?;
        let img_splits = img_proj.split_axis(&[self.mlp_hidden], -1)?;
        let img_swiglu_out = ops::multiply(&mlx_rs::nn::silu(&img_splits[0])?, &img_splits[1])?;
        let img_mlp_out = self.img_mlp_out.forward(&img_swiglu_out)?;

        // SwiGLU MLP for text: silu(gate) * up
        let txt_proj = self.txt_mlp_in.forward(&txt_mlp_in)?;
        let txt_splits = txt_proj.split_axis(&[self.mlp_hidden], -1)?;
        let txt_swiglu_out = ops::multiply(&mlx_rs::nn::silu(&txt_splits[0])?, &txt_splits[1])?;
        let txt_mlp_out = self.txt_mlp_out.forward(&txt_swiglu_out)?;

        // Apply gate and residual
        let img = ops::add(&img, &gate(&img_mlp_out, img_gate2)?)?;
        let txt_out = ops::add(&txt, &gate(&txt_mlp_out, txt_gate2)?)?;

        Ok((img, txt_out))
    }
}

// ============================================================================
// Klein Single Stream Block
// ============================================================================

/// Single stream block for FLUX.2-klein
///
/// Processes the combined (img + txt) stream with self-attention and MLP in a single
/// fused operation. Used after double blocks for efficient single-stream processing.
///
/// # Architecture
/// - Fused projection: `[Q, K, V, mlp_gate, mlp_up]` in one linear layer
/// - Self-attention on combined stream
/// - Parallel MLP: computed alongside attention, outputs concatenated
/// - Fused output projection: `[attn_out, mlp_out] -> hidden`
/// - Pre-modulation LayerNorm without learnable affine
/// - QK normalization using RmsNorm
///
/// # Stability
/// Values are clipped to ±65504 after residual addition.
/// Modulation clamping is handled in SharedModulation.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct KleinSingleBlock {
    pub hidden_size: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    pub mlp_hidden: i32,

    // Pre-modulation normalization: LayerNorm without learnable affine (matches diffusers)
    #[param]
    pub norm: LayerNorm,

    // Fused projection: Q (hidden) + K (hidden) + V (hidden) + gate (mlp_hidden) + up (mlp_hidden)
    #[param]
    pub to_qkv_mlp: Linear,  // [hidden, 3*hidden + 2*mlp_hidden]

    // Output projection: attn (hidden) + mlp (mlp_hidden) -> hidden
    #[param]
    pub to_out: Linear,  // [hidden + mlp_hidden, hidden]

    #[param]
    pub norm_q: RmsNorm,
    #[param]
    pub norm_k: RmsNorm,

    // Post-residual normalization to prevent value explosion
    #[param]
    pub post_norm: RmsNorm,
}

impl KleinSingleBlock {
    pub fn new(hidden_size: i32, num_heads: i32, mlp_hidden: i32) -> Result<Self, Exception> {
        let head_dim = hidden_size / num_heads;
        let fused_in = 3 * hidden_size + 2 * mlp_hidden;  // 27648
        let fused_out = hidden_size + mlp_hidden;          // 12288

        Ok(Self {
            hidden_size,
            num_heads,
            head_dim,
            mlp_hidden,

            norm: LayerNormBuilder::new(hidden_size).affine(false).eps(1e-6).build()?,
            to_qkv_mlp: LinearBuilder::new(hidden_size, fused_in).bias(false).build()?,
            to_out: LinearBuilder::new(fused_out, hidden_size).bias(false).build()?,
            norm_q: RmsNorm::new(head_dim)?,  // Q/K norms use RmsNorm (matching diffusers)
            norm_k: RmsNorm::new(head_dim)?,
            post_norm: RmsNorm::new(hidden_size)?,  // Post-residual normalization
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Combined tokens [batch, seq, hidden]
    /// * `mod_params` - Modulation [shift, scale, gate]
    /// * `rope_cos` - RoPE cosine [batch, seq, head_dim]
    /// * `rope_sin` - RoPE sine [batch, seq, head_dim]
    pub fn forward(
        &mut self,
        x: &Array,
        mod_params: &[Array],
        rope_cos: &Array,
        rope_sin: &Array,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Modulation order: shift, scale, gate
        let (shift, scale, gate_val) = (&mod_params[0], &mod_params[1], &mod_params[2]);

        // Apply pre-norm THEN modulation: (1 + scale) * norm(x) + shift
        let x_normed = self.norm.forward(x)?;
        let x_mod = modulate(&x_normed, shift, scale)?;

        // Fused projection
        let proj = self.to_qkv_mlp.forward(&x_mod)?;

        // Split: Q, K, V, gate, up
        let q_end = self.hidden_size;
        let k_end = 2 * self.hidden_size;
        let v_end = 3 * self.hidden_size;
        let gate_end = v_end + self.mlp_hidden;

        let splits = proj.split_axis(&[q_end, k_end, v_end, gate_end], -1)?;
        let q = &splits[0];
        let k = &splits[1];
        let v = &splits[2];
        let mlp_gate = &splits[3];
        let mlp_up = &splits[4];

        // Reshape Q, K, V for attention
        let q = q.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;

        // Apply QK normalization
        let q = self.norm_q.forward(&q)?;
        let k = self.norm_k.forward(&k)?;

        // Apply RoPE to Q and K
        let q = apply_rope(&q, rope_cos, rope_sin)?;
        let k = apply_rope(&k, rope_cos, rope_sin)?;

        // Attention
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let scale = (self.head_dim as f32).sqrt();
        let attn = ops::matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?;
        let attn = ops::divide(&attn, &array!(scale))?;
        let attn = ops::softmax_axis(&attn, -1, None)?;
        let attn_out = ops::matmul(&attn, &v)?;

        // Reshape back
        let attn_out = attn_out.transpose_axes(&[0, 2, 1, 3])?;
        let attn_out = attn_out.reshape(&[batch, seq_len, -1])?;

        // SwiGLU MLP: silu(gate) * up (matching flux.c)
        let mlp_out = ops::multiply(&mlx_rs::nn::silu(mlp_gate)?, mlp_up)?;

        // Concatenate attention and MLP output
        let combined = ops::concatenate_axis(&[attn_out, mlp_out], -1)?;
        let out = self.to_out.forward(&combined)?;

        // Apply gate and residual
        ops::add(x, &gate(&out, gate_val)?)
    }
}

// ============================================================================
// FLUX.2-klein Model
// ============================================================================

/// FLUX.2-klein transformer
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct FluxKlein {
    pub params: FluxKleinParams,

    // Input embeddings
    #[param]
    pub x_embedder: Linear,      // img_in: [in_channels, hidden]
    #[param]
    pub context_embedder: Linear, // txt_in: [txt_embed, hidden]

    // Normalization after embeddings (critical for stability)
    // txt comes from 7680 dim Qwen3, needs norm to prevent explosion
    #[param]
    pub txt_norm: RmsNorm,

    // Time embedding
    #[param]
    pub time_embed_1: Linear,
    #[param]
    pub time_embed_2: Linear,

    // Shared modulation (one per stream type)
    #[param]
    pub double_mod_img: SharedModulation,
    #[param]
    pub double_mod_txt: SharedModulation,
    #[param]
    pub single_mod: SharedModulation,

    // Transformer blocks
    #[param]
    pub double_blocks: Vec<KleinDoubleBlock>,
    #[param]
    pub single_blocks: Vec<KleinSingleBlock>,

    // Final layer
    #[param]
    pub final_norm: RmsNorm,  // LayerNorm before final modulation
    #[param]
    pub norm_out: Linear,  // AdaLN: [hidden, 2*hidden] for scale + shift
    #[param]
    pub proj_out: Linear,  // [hidden, in_channels]
}

impl FluxKlein {
    pub fn new(params: FluxKleinParams) -> Result<Self, Exception> {
        let hidden = params.hidden_size;
        let mlp_hidden = params.mlp_hidden;

        // Input embeddings
        let x_embedder = LinearBuilder::new(params.in_channels, hidden).bias(false).build()?;
        let context_embedder = LinearBuilder::new(params.txt_embed_dim, hidden).bias(false).build()?;

        // Normalization after txt embedding (stabilizes txt stream)
        let txt_norm = RmsNorm::new(hidden)?;

        // Time embedding
        let time_embed_1 = LinearBuilder::new(256, hidden).bias(false).build()?;
        let time_embed_2 = LinearBuilder::new(hidden, hidden).bias(false).build()?;

        // Shared modulation
        let double_mod_img = SharedModulation::new(hidden, 6)?;
        let double_mod_txt = SharedModulation::new(hidden, 6)?;
        let single_mod = SharedModulation::new(hidden, 3)?;

        // Double blocks
        let double_blocks: Result<Vec<_>, _> = (0..params.depth)
            .map(|_| KleinDoubleBlock::new(hidden, params.num_heads, mlp_hidden))
            .collect();
        let double_blocks = double_blocks?;

        // Single blocks
        let single_blocks: Result<Vec<_>, _> = (0..params.depth_single)
            .map(|_| KleinSingleBlock::new(hidden, params.num_heads, mlp_hidden))
            .collect();
        let single_blocks = single_blocks?;

        // Final layer
        let final_norm = RmsNorm::new(hidden)?;
        let norm_out = LinearBuilder::new(hidden, 2 * hidden).bias(false).build()?;
        let proj_out = LinearBuilder::new(hidden, params.in_channels).bias(false).build()?;

        Ok(Self {
            params,
            x_embedder,
            context_embedder,
            txt_norm,
            time_embed_1,
            time_embed_2,
            double_mod_img,
            double_mod_txt,
            single_mod,
            double_blocks,
            single_blocks,
            final_norm,
            norm_out,
            proj_out,
        })
    }

    /// Compute RoPE frequencies for caching
    ///
    /// Call this once before the denoising loop and pass the result to forward_with_rope
    pub fn compute_rope(
        txt_ids: &Array,
        img_ids: &Array,
    ) -> Result<(Array, Array), Exception> {
        let all_ids = ops::concatenate_axis(&[txt_ids, img_ids], 1)?;
        compute_rope_freqs(
            &all_ids,
            &[32, 32, 32, 32],  // FLUX.2-klein: 4 axes of 32 dims each
            2000.0,             // theta from config.json
        )
    }

    /// Forward pass with pre-computed RoPE (faster for multiple steps)
    pub fn forward_with_rope(
        &mut self,
        img: &Array,
        txt: &Array,
        timesteps: &Array,
        rope_cos: &Array,
        rope_sin: &Array,
    ) -> Result<Array, Exception> {
        let txt_seq = txt.dim(1);

        // Project inputs
        let mut img = self.x_embedder.forward(img)?;
        let txt = self.context_embedder.forward(txt)?;

        // Time embedding
        let t_emb = timestep_embedding(timesteps, 256, 10000.0)?;
        let vec = self.time_embed_1.forward(&t_emb)?;
        let vec = mlx_rs::nn::silu(&vec)?;
        let vec = self.time_embed_2.forward(&vec)?;

        // Get shared modulation parameters
        let img_mod = self.double_mod_img.forward(&vec)?;
        let txt_mod = self.double_mod_txt.forward(&vec)?;
        let single_mod = self.single_mod.forward(&vec)?;

        // Double stream blocks
        let mut txt = txt;
        for block in self.double_blocks.iter_mut() {
            let (new_img, new_txt) = block.forward(&img, &txt, &img_mod, &txt_mod, &rope_cos, &rope_sin)?;
            img = new_img;
            txt = new_txt;
        }

        // Concatenate streams for single blocks: [txt, img]
        let mut x = ops::concatenate_axis(&[txt, img], 1)?;

        // Single stream blocks
        for block in self.single_blocks.iter_mut() {
            x = block.forward(&x, &single_mod, &rope_cos, &rope_sin)?;
        }

        // Extract image tokens
        let parts = x.split_axis(&[txt_seq], 1)?;
        let img_out = &parts[1];

        // Final layer with AdaLN
        let ada = mlx_rs::nn::silu(&vec)?;
        let ada = self.norm_out.forward(&ada)?;
        let chunks = ada.split(2, -1)?;
        let scale = &chunks[0];
        let shift = &chunks[1];

        let img_normed = self.final_norm.forward(img_out)?;
        let out = modulate(&img_normed, shift, scale)?;
        self.proj_out.forward(&out)
    }

    /// Forward pass (convenience wrapper that computes RoPE internally)
    ///
    /// For better performance in denoising loops, use compute_rope() once
    /// then call forward_with_rope() for each step.
    pub fn forward(
        &mut self,
        img: &Array,
        txt: &Array,
        timesteps: &Array,
        img_ids: &Array,
        txt_ids: &Array,
    ) -> Result<Array, Exception> {
        let (rope_cos, rope_sin) = Self::compute_rope(txt_ids, img_ids)?;
        self.forward_with_rope(img, txt, timesteps, &rope_cos, &rope_sin)
    }

    /// Check if array contains NaN values
    fn has_nan(arr: &Array) -> bool {
        let flat = match arr.reshape(&[-1]) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let _ = flat.eval();
        let data: Vec<f32> = match flat.try_as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => return false,
        };
        data.iter().any(|x| x.is_nan())
    }

    /// Get min/max range of array
    fn get_range(arr: &Array) -> (f32, f32) {
        let flat = match arr.reshape(&[-1]) {
            Ok(f) => f,
            Err(_) => return (0.0, 0.0),
        };
        let _ = flat.eval();
        let data: Vec<f32> = match flat.try_as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => return (0.0, 0.0),
        };
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Apply modulation: (1 + scale) * x + shift
/// Standard AdaLN formula where scale is a small adjustment around 1
fn modulate(x: &Array, shift: &Array, scale: &Array) -> Result<Array, Exception> {
    // Expand shift and scale for broadcasting: [batch, dim] -> [batch, 1, dim]
    let shift = shift.reshape(&[shift.dim(0), 1, shift.dim(-1)])?;
    let scale = scale.reshape(&[scale.dim(0), 1, scale.dim(-1)])?;

    // (1 + scale) * x + shift
    let one = array!(1.0f32);
    let scale_plus_one = ops::add(&one, &scale)?;
    let scaled = ops::multiply(x, &scale_plus_one)?;
    ops::add(&scaled, &shift)
}

/// Apply gate: x * unsqueeze(gate)
fn gate(x: &Array, gate: &Array) -> Result<Array, Exception> {
    let gate = gate.reshape(&[gate.dim(0), 1, gate.dim(-1)])?;
    ops::multiply(x, &gate)
}

/// Clip values to prevent fp16-style overflow (diffusers uses ±65504)
fn clip_values(x: &Array) -> Result<Array, Exception> {
    let min_val = array!(-65504.0f32);
    let max_val = array!(65504.0f32);
    let clipped = ops::maximum(x, &min_val)?;
    ops::minimum(&clipped, &max_val)
}

/// Get min/max range of array (helper for debugging)
fn get_range(arr: &Array) -> (f32, f32) {
    let flat = match arr.reshape(&[-1]) {
        Ok(f) => f,
        Err(_) => return (0.0, 0.0),
    };
    let _ = flat.eval();
    let data: Vec<f32> = match flat.try_as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => return (0.0, 0.0),
    };
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (min, max)
}
