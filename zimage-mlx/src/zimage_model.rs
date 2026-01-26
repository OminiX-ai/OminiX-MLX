//! Z-Image-Turbo Transformer Implementation
//!
//! Z-Image-Turbo is a 6B parameter Single-Stream DiT (S3-DiT) optimized for
//! fast image generation. It differs from FLUX in several key ways:
//!
//! # Architecture
//! - **Text Encoder**: Qwen3-4B, extracts layer 34 (2560-dim) instead of concat(8,17,26)
//! - **Transformer**: Noise Refiner → Context Refiner → Joint blocks
//! - **RoPE**: 3-axis (32,48,48) with theta=256, not 4-axis
//! - **Modulation**: Per-block AdaLN with tanh-gated outputs
//!
//! # Configuration (from transformer/config.json)
//! ```text
//! dim: 3840
//! n_heads: 30
//! n_kv_heads: 30
//! n_layers: 30 (joint blocks)
//! n_refiner_layers: 2
//! in_channels: 16
//! cap_feat_dim: 2560
//! axes_dims: [32, 48, 48]
//! rope_theta: 256.0
//! t_scale: 1000.0
//! ```

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    module::Module,
    nn::{self, Linear, LinearBuilder, RmsNorm},
    ops,
    ops::indexing::IndexOp,
    Array,
};
use mlx_macros::ModuleParameters;

// ============================================================================
// Configuration
// ============================================================================

/// Z-Image-Turbo configuration
#[derive(Debug, Clone)]
pub struct ZImageConfig {
    pub dim: i32,              // 3840
    pub n_heads: i32,          // 30
    pub n_kv_heads: i32,       // 30
    pub n_layers: i32,         // 30 joint blocks
    pub n_refiner_layers: i32, // 2 refiner blocks each
    pub in_channels: i32,      // 16
    pub cap_feat_dim: i32,     // 2560 (Qwen3 layer 34)
    pub axes_dims: [i32; 3],   // [32, 48, 48]
    pub rope_theta: f32,       // 256.0
    pub t_scale: f32,          // 1000.0
    pub norm_eps: f32,         // 1e-5
    pub patch_size: i32,       // 2
}

impl Default for ZImageConfig {
    fn default() -> Self {
        Self {
            dim: 3840,
            n_heads: 30,
            n_kv_heads: 30,
            n_layers: 30,
            n_refiner_layers: 2,
            in_channels: 16,
            cap_feat_dim: 2560,
            axes_dims: [32, 48, 48],
            rope_theta: 256.0,
            t_scale: 1000.0,
            norm_eps: 1e-5,
            patch_size: 2,
        }
    }
}

impl ZImageConfig {
    pub fn head_dim(&self) -> i32 {
        self.dim / self.n_heads
    }

    /// MLP hidden size (8/3 ratio like SwiGLU)
    pub fn mlp_hidden(&self) -> i32 {
        (self.dim as f32 / 3.0 * 8.0) as i32
    }
}

// ============================================================================
// 3-Axis RoPE Implementation
// ============================================================================

/// Create 3D coordinate grid for position encoding
///
/// Matches MLX_z-image's create_coordinate_grid function.
///
/// # Arguments
/// * `size` - (d0, d1, d2) grid dimensions
/// * `start` - (s0, s1, s2) starting coordinates
///
/// # Returns
/// Array of shape [d0*d1*d2, 3]
pub fn create_coordinate_grid(
    size: (i32, i32, i32),
    start: (i32, i32, i32),
) -> Result<Array, Exception> {
    let (d0, d1, d2) = size;
    let (s0, s1, s2) = start;
    let total = (d0 * d1 * d2) as usize;

    let mut coords = Vec::with_capacity(total * 3);
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                coords.push((s0 + i) as f32);
                coords.push((s1 + j) as f32);
                coords.push((s2 + k) as f32);
            }
        }
    }

    Ok(Array::from_slice(&coords, &[total as i32, 3]))
}

/// Pre-compute inverse frequencies for 3-axis RoPE
///
/// Returns arrays for each axis dimension that can be reused across calls.
pub fn precompute_rope_inv_freqs(axes_dims: &[i32; 3], theta: f32) -> [Array; 3] {
    let mut result = [
        Array::from_slice(&[0.0f32], &[1]),
        Array::from_slice(&[0.0f32], &[1]),
        Array::from_slice(&[0.0f32], &[1]),
    ];

    for (axis, &dim) in axes_dims.iter().enumerate() {
        let half_dim = dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| (-theta.ln() * (i as f32) / (half_dim as f32)).exp())
            .collect();
        result[axis] = Array::from_slice(&inv_freq, &[1, 1, 1, half_dim]);
    }

    result
}

/// Compute 3-axis RoPE frequencies for Z-Image using pre-computed inverse frequencies
///
/// # Arguments
/// * `positions` - Position coordinates [batch, seq, 3] for (h, w, t)
/// * `inv_freqs` - Pre-computed inverse frequencies from `precompute_rope_inv_freqs`
///
/// # Returns
/// (cos, sin) each of shape [batch, seq, 1, half_total_dim] where half_total_dim = sum(axes_dims)/2
pub fn compute_rope_3axis_cached(
    positions: &Array,
    inv_freqs: &[Array; 3],
) -> Result<(Array, Array), Exception> {
    let batch = positions.dim(0);
    let seq_len = positions.dim(1);

    let mut all_args = Vec::with_capacity(3);

    for (axis, inv_freq) in inv_freqs.iter().enumerate() {
        // Get positions for this axis: positions[:, :, axis]
        let pos = positions.index((.., .., axis as i32));
        let pos = pos.reshape(&[batch, seq_len, 1, 1])?;

        // Compute angles: pos * inv_freq -> [batch, seq, 1, half_dim]
        let angles = ops::multiply(&pos, inv_freq)?;
        all_args.push(angles);
    }

    // Concatenate all axes -> [batch, seq, 1, half_total]
    let args_refs: Vec<&Array> = all_args.iter().collect();
    let args = ops::concatenate_axis(&args_refs, -1)?;

    // Compute cos and sin
    let cos = ops::cos(&args)?;
    let sin = ops::sin(&args)?;

    Ok((cos, sin))
}

/// Compute 3-axis RoPE frequencies for Z-Image (convenience function)
///
/// # Arguments
/// * `positions` - Position coordinates [batch, seq, 3] for (h, w, t)
/// * `axes_dims` - Dimensions per axis [32, 48, 48]
/// * `theta` - Base frequency (256.0 for Z-Image)
///
/// # Returns
/// (cos, sin) each of shape [batch, seq, 1, half_total_dim] where half_total_dim = sum(axes_dims)/2
/// This matches the reference implementation which doesn't duplicate values.
pub fn compute_rope_3axis(
    positions: &Array,
    axes_dims: &[i32; 3],
    theta: f32,
) -> Result<(Array, Array), Exception> {
    let inv_freqs = precompute_rope_inv_freqs(axes_dims, theta);
    compute_rope_3axis_cached(positions, &inv_freqs)
}

/// Apply rotary embedding using even/odd split (matches reference)
///
/// Reference implementation:
///   q1, q2 = q[..., 0::2], q[..., 1::2]
///   q = mx.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1).reshape(...)
pub fn apply_rope_3axis(
    x: &Array,
    cos: &Array,
    sin: &Array,
) -> Result<Array, Exception> {
    // x shape: [batch, seq, heads, head_dim]
    // cos/sin shape: [batch, seq, 1, head_dim/2] (no duplication)
    let x_shape = x.shape();
    let batch = x_shape[0] as i32;
    let seq_len = x_shape[1] as i32;
    let heads = x_shape[2] as i32;
    let head_dim = x_shape[3] as i32;

    // Split into even and odd indices: x[..., 0::2] and x[..., 1::2]
    let x_pairs = x.reshape(&[batch, seq_len, heads, head_dim / 2, 2])?;
    let x1 = x_pairs.index((.., .., .., .., 0_i32));  // even indices
    let x2 = x_pairs.index((.., .., .., .., 1_i32));  // odd indices

    // Apply rotation:
    //   out_even = x_even * cos - x_odd * sin
    //   out_odd  = x_even * sin + x_odd * cos
    let out0 = ops::subtract(&ops::multiply(&x1, cos)?, &ops::multiply(&x2, sin)?)?;
    let out1 = ops::add(&ops::multiply(&x1, sin)?, &ops::multiply(&x2, cos)?)?;

    // Stack and reshape: [out0, out1] -> interleaved pairs
    let out_pairs = ops::stack_axis(&[out0, out1], -1)?;
    out_pairs.reshape(&[batch, seq_len, heads, head_dim])
}

// ============================================================================
// Feed Forward (SwiGLU)
// ============================================================================

/// SwiGLU feed-forward network
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct FeedForward {
    #[param]
    pub w1: Linear,  // gate projection
    #[param]
    pub w2: Linear,  // down projection
    #[param]
    pub w3: Linear,  // up projection
}

impl FeedForward {
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            w1: LinearBuilder::new(dim, hidden_dim).bias(false).build()?,
            w2: LinearBuilder::new(hidden_dim, dim).bias(false).build()?,
            w3: LinearBuilder::new(dim, hidden_dim).bias(false).build()?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // SwiGLU: silu(w1(x)) * w3(x)
        let gate = nn::silu(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        let hidden = ops::multiply(&gate, &up)?;
        self.w2.forward(&hidden)
    }
}

// ============================================================================
// Attention
// ============================================================================

/// Multi-head attention with QK normalization and optional fused QKV
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[param]
    pub to_q: Linear,
    #[param]
    pub to_k: Linear,
    #[param]
    pub to_v: Linear,
    #[param]
    pub to_out: Linear,

    #[param]
    pub norm_q: RmsNorm,
    #[param]
    pub norm_k: RmsNorm,
}

impl Attention {
    pub fn new(dim: i32, n_heads: i32, n_kv_heads: i32, eps: f32) -> Result<Self, Exception> {
        let head_dim = dim / n_heads;
        let scale = (head_dim as f32).powf(-0.5);

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            to_q: LinearBuilder::new(dim, n_heads * head_dim).bias(false).build()?,
            to_k: LinearBuilder::new(dim, n_kv_heads * head_dim).bias(false).build()?,
            to_v: LinearBuilder::new(dim, n_kv_heads * head_dim).bias(false).build()?,
            to_out: LinearBuilder::new(n_heads * head_dim, dim).bias(false).build()?,
            norm_q: nn::RmsNormBuilder::new(head_dim).eps(eps).build()?,
            norm_k: nn::RmsNormBuilder::new(head_dim).eps(eps).build()?,
        })
    }

    /// Forward pass with RoPE
    ///
    /// # Arguments
    /// * `x` - Input [batch, seq, dim]
    /// * `cos` - Cosine frequencies [batch, seq, 1, head_dim]
    /// * `sin` - Sine frequencies [batch, seq, 1, head_dim]
    /// * `mask` - Optional attention mask
    pub fn forward(
        &mut self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0] as i32;
        let seq_len = shape[1] as i32;

        // Project to Q, K, V
        let q = self.to_q.forward(x)?;
        let k = self.to_k.forward(x)?;
        let v = self.to_v.forward(x)?;

        // Reshape for multi-head: [batch, seq, heads, head_dim]
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.n_kv_heads, self.head_dim])?;

        // Apply QK normalization
        let q = self.norm_q.forward(&q)?;
        let k = self.norm_k.forward(&k)?;

        // Apply RoPE
        let q = apply_rope_3axis(&q, cos, sin)?;
        let k = apply_rope_3axis(&k, cos, sin)?;

        // Transpose for attention: [batch, heads, seq, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Handle GQA if n_kv_heads < n_heads
        let (k, v) = if self.n_kv_heads < self.n_heads {
            let n_rep = self.n_heads / self.n_kv_heads;
            let k = Array::repeat_axis::<f32>(k, n_rep, 1)?;
            let v = Array::repeat_axis::<f32>(v, n_rep, 1)?;
            (k, v)
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        // Q @ K^T / sqrt(d)
        let k_t = k.transpose_axes(&[0, 1, 3, 2])?;
        let attn = ops::matmul(&q, &k_t)?;
        let attn = ops::multiply(&attn, &array!(self.scale))?;

        // Apply mask if provided (additive mask: 0 = attend, -inf = mask)
        let attn = if let Some(m) = mask {
            ops::add(&attn, m)?
        } else {
            attn
        };

        // Softmax and apply to values
        let attn = ops::softmax_axis(&attn, -1, None)?;
        let out = ops::matmul(&attn, &v)?;

        // Transpose back and reshape: [batch, heads, seq, dim] -> [batch, seq, heads*dim]
        let out = out.transpose_axes(&[0, 2, 1, 3])?;
        let out = out.reshape(&[batch, seq_len, -1])?;

        self.to_out.forward(&out)
    }
}

// ============================================================================
// Timestep Embedder
// ============================================================================

// Constants for timestep embedding
const TIMESTEP_EMBED_DIM: i32 = 256;
const TIMESTEP_MLP_HIDDEN: i32 = 1024;

/// Timestep embedding with sinusoidal positional encoding
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct TimestepEmbedder {
    pub frequency_embedding_size: i32,
    /// Pre-computed sinusoidal frequencies (cached for performance)
    pub cached_freqs: Array,

    #[param]
    pub linear1: Linear,
    #[param]
    pub linear2: Linear,
}

impl TimestepEmbedder {
    pub fn new(out_size: i32, mid_size: Option<i32>, freq_size: i32) -> Result<Self, Exception> {
        let mid = mid_size.unwrap_or(out_size);
        let half = freq_size / 2;

        // Pre-compute sinusoidal frequencies once at initialization
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-10000.0f32.ln() * (i as f32) / (half as f32)).exp())
            .collect();
        let cached_freqs = Array::from_slice(&freqs, &[1, half]);

        Ok(Self {
            frequency_embedding_size: freq_size,
            cached_freqs,
            linear1: LinearBuilder::new(freq_size, mid).bias(true).build()?,
            linear2: LinearBuilder::new(mid, out_size).bias(true).build()?,
        })
    }

    pub fn forward(&mut self, t: &Array) -> Result<Array, Exception> {
        // Use pre-computed frequencies
        let t = t.as_dtype(mlx_rs::Dtype::Float32)?;
        let t = t.reshape(&[-1, 1])?;
        let args = ops::multiply(&t, &self.cached_freqs)?;

        let cos_args = ops::cos(&args)?;
        let sin_args = ops::sin(&args)?;
        let embedding = ops::concatenate_axis(&[&cos_args, &sin_args], -1)?;

        // MLP
        let x = self.linear1.forward(&embedding)?;
        let x = nn::silu(&x)?;
        self.linear2.forward(&x)
    }
}

// ============================================================================
// Z-Image Transformer Block
// ============================================================================

/// Z-Image transformer block with optional modulation
///
/// When modulation is enabled (noise_refiner, joint blocks), uses AdaLN with tanh gates.
/// When disabled (context_refiner), uses standard pre-norm.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct ZImageTransformerBlock {
    pub dim: i32,
    pub has_modulation: bool,

    #[param]
    pub attention: Attention,
    #[param]
    pub feed_forward: FeedForward,

    // Pre-attention normalization
    #[param]
    pub attention_norm1: RmsNorm,
    // Post-attention normalization (before residual)
    #[param]
    pub attention_norm2: RmsNorm,
    // Pre-FFN normalization
    #[param]
    pub ffn_norm1: RmsNorm,
    // Post-FFN normalization (before residual)
    #[param]
    pub ffn_norm2: RmsNorm,

    // AdaLN modulation (only for modulated blocks)
    // Outputs: [scale_msa, gate_msa, scale_mlp, gate_mlp]
    #[param]
    pub ada_ln_modulation: Option<Linear>,
}

impl ZImageTransformerBlock {
    pub fn new(config: &ZImageConfig, has_modulation: bool) -> Result<Self, Exception> {
        let dim = config.dim;
        let mlp_hidden = config.mlp_hidden();

        let ada_ln_modulation = if has_modulation {
            // 4 outputs: scale_msa, gate_msa, scale_mlp, gate_mlp
            Some(LinearBuilder::new(256, 4 * dim).bias(true).build()?)
        } else {
            None
        };

        Ok(Self {
            dim,
            has_modulation,
            attention: Attention::new(dim, config.n_heads, config.n_kv_heads, config.norm_eps)?,
            feed_forward: FeedForward::new(dim, mlp_hidden)?,
            attention_norm1: nn::RmsNormBuilder::new(dim).eps(config.norm_eps).build()?,
            attention_norm2: nn::RmsNormBuilder::new(dim).eps(config.norm_eps).build()?,
            ffn_norm1: nn::RmsNormBuilder::new(dim).eps(config.norm_eps).build()?,
            ffn_norm2: nn::RmsNormBuilder::new(dim).eps(config.norm_eps).build()?,
            ada_ln_modulation,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input [batch, seq, dim]
    /// * `cos`, `sin` - RoPE frequencies
    /// * `adaln_input` - Time embedding for modulation (only used if has_modulation)
    /// * `mask` - Optional attention mask
    pub fn forward(
        &mut self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        adaln_input: Option<&Array>,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        if self.has_modulation {
            // Modulated path (noise refiner + joint blocks)
            let adaln = adaln_input.expect("adaln_input required for modulated blocks");

            // Get modulation parameters
            let chunks = self.ada_ln_modulation.as_mut().unwrap().forward(adaln)?;
            let splits = chunks.split(4, -1)?;
            let scale_msa = &splits[0];
            let gate_msa = &splits[1];
            let scale_mlp = &splits[2];
            let gate_mlp = &splits[3];

            // Expand for broadcasting: [batch, dim] -> [batch, 1, dim]
            let scale_msa = scale_msa.reshape(&[scale_msa.dim(0), 1, self.dim])?;
            let gate_msa = gate_msa.reshape(&[gate_msa.dim(0), 1, self.dim])?;
            let scale_mlp = scale_mlp.reshape(&[scale_mlp.dim(0), 1, self.dim])?;
            let gate_mlp = gate_mlp.reshape(&[gate_mlp.dim(0), 1, self.dim])?;

            // Attention with modulation
            // norm(x) * (1 + scale)
            let norm_x = self.attention_norm1.forward(x)?;
            let one = array!(1.0f32);
            let modulated = ops::multiply(&norm_x, &ops::add(&one, &scale_msa)?)?;
            let attn_out = self.attention.forward(&modulated, cos, sin, mask)?;

            // tanh(gate) * norm2(attn_out)
            let gated_attn = ops::multiply(&ops::tanh(&gate_msa)?, &self.attention_norm2.forward(&attn_out)?)?;
            let x = ops::add(x, &gated_attn)?;

            // FFN with modulation
            let norm_x = self.ffn_norm1.forward(&x)?;
            let modulated = ops::multiply(&norm_x, &ops::add(&one, &scale_mlp)?)?;
            let ffn_out = self.feed_forward.forward(&modulated)?;

            // tanh(gate) * norm2(ffn_out)
            let gated_ffn = ops::multiply(&ops::tanh(&gate_mlp)?, &self.ffn_norm2.forward(&ffn_out)?)?;
            ops::add(&x, &gated_ffn)
        } else {
            // Unmodulated path (context refiner)
            let norm_x = self.attention_norm1.forward(x)?;
            let attn_out = self.attention.forward(&norm_x, cos, sin, mask)?;
            let x = ops::add(x, &self.attention_norm2.forward(&attn_out)?)?;

            let norm_x = self.ffn_norm1.forward(&x)?;
            let ffn_out = self.feed_forward.forward(&norm_x)?;
            ops::add(&x, &self.ffn_norm2.forward(&ffn_out)?)
        }
    }
}

// ============================================================================
// Final Layer
// ============================================================================

/// Final output layer with AdaLN modulation
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct FinalLayer {
    pub dim: i32,
    pub out_channels: i32,

    #[param]
    pub norm_final: nn::LayerNorm,
    #[param]
    pub linear: Linear,
    #[param]
    pub ada_ln_modulation: Linear,
}

impl FinalLayer {
    pub fn new(dim: i32, out_channels: i32) -> Result<Self, Exception> {
        Ok(Self {
            dim,
            out_channels,
            // LayerNorm without affine (weight/bias) - just normalization
            norm_final: nn::LayerNormBuilder::new(dim).affine(false).eps(1e-6).build()?,
            linear: LinearBuilder::new(dim, out_channels).bias(true).build()?,
            // SiLU + Linear for modulation
            ada_ln_modulation: LinearBuilder::new(256, dim).bias(true).build()?,
        })
    }

    pub fn forward(&mut self, x: &Array, c: &Array) -> Result<Array, Exception> {
        // Get scale from timestep embedding
        let scale = nn::silu(c)?;
        let scale = self.ada_ln_modulation.forward(&scale)?;
        let scale = scale.reshape(&[scale.dim(0), 1, self.dim])?;

        // Apply: linear(norm(x) * (1 + scale))
        let one = array!(1.0f32);
        let normed = self.norm_final.forward(x)?;
        let modulated = ops::multiply(&normed, &ops::add(&one, &scale)?)?;
        self.linear.forward(&modulated)
    }
}

// ============================================================================
// Z-Image Transformer
// ============================================================================

/// Z-Image-Turbo Transformer
///
/// Architecture:
/// 1. Embed inputs (x_embedder for images, cap_embedder for text)
/// 2. Noise Refiner: Process image tokens with modulation
/// 3. Context Refiner: Process text tokens without modulation
/// 4. Joint Blocks: Process concatenated [img, txt] with modulation
/// 5. Final Layer: Output projection with AdaLN
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct ZImageTransformer {
    pub config: ZImageConfig,

    // Pre-computed RoPE inverse frequencies (cached for performance)
    pub rope_inv_freqs: [Array; 3],

    // Timestep embedding
    #[param]
    pub t_embedder: TimestepEmbedder,

    // Input embeddings
    #[param]
    pub x_embedder: Linear,  // Image patch embedding

    // Caption embedding: RmsNorm + Linear
    #[param]
    pub cap_norm: RmsNorm,
    #[param]
    pub cap_linear: Linear,

    // Pad tokens for masking (not trainable parameters)
    pub x_pad_token: Array,
    pub cap_pad_token: Array,

    // Transformer blocks
    #[param]
    pub noise_refiner: Vec<ZImageTransformerBlock>,
    #[param]
    pub context_refiner: Vec<ZImageTransformerBlock>,
    #[param]
    pub layers: Vec<ZImageTransformerBlock>,

    // Final output
    #[param]
    pub final_layer: FinalLayer,
}

impl ZImageTransformer {
    pub fn new(config: ZImageConfig) -> Result<Self, Exception> {
        let dim = config.dim;

        // Pre-compute RoPE inverse frequencies (cached for performance)
        let rope_inv_freqs = precompute_rope_inv_freqs(&config.axes_dims, config.rope_theta);

        // Timestep embedder: outputs 256-dim for modulation (using constants)
        let t_embedder = TimestepEmbedder::new(TIMESTEP_EMBED_DIM, Some(TIMESTEP_MLP_HIDDEN), TIMESTEP_EMBED_DIM)?;

        // Image embedder: in_channels * patch_size^2 -> dim
        let patch_channels = config.in_channels * config.patch_size * config.patch_size;
        let x_embedder = LinearBuilder::new(patch_channels, dim).bias(true).build()?;

        // Caption embedder: RmsNorm + Linear
        let cap_norm = nn::RmsNormBuilder::new(config.cap_feat_dim).eps(config.norm_eps).build()?;
        let cap_linear = LinearBuilder::new(config.cap_feat_dim, dim).bias(true).build()?;

        // Pad tokens
        let x_pad_token = Array::zeros::<f32>(&[1, dim])?;
        let cap_pad_token = Array::zeros::<f32>(&[1, dim])?;

        // Noise refiner blocks (with modulation)
        let noise_refiner: Result<Vec<_>, _> = (0..config.n_refiner_layers)
            .map(|_| ZImageTransformerBlock::new(&config, true))
            .collect();
        let noise_refiner = noise_refiner?;

        // Context refiner blocks (without modulation)
        let context_refiner: Result<Vec<_>, _> = (0..config.n_refiner_layers)
            .map(|_| ZImageTransformerBlock::new(&config, false))
            .collect();
        let context_refiner = context_refiner?;

        // Joint blocks (with modulation)
        let layers: Result<Vec<_>, _> = (0..config.n_layers)
            .map(|_| ZImageTransformerBlock::new(&config, true))
            .collect();
        let layers = layers?;

        // Final layer
        let final_layer = FinalLayer::new(dim, patch_channels)?;

        Ok(Self {
            config,
            rope_inv_freqs,
            t_embedder,
            x_embedder,
            cap_norm,
            cap_linear,
            x_pad_token,
            cap_pad_token,
            noise_refiner,
            context_refiner,
            layers,
            final_layer,
        })
    }

    /// Compute RoPE frequencies for caching
    ///
    /// Call once before denoising loop and pass to forward_with_rope.
    /// Uses pre-computed inverse frequencies for better performance.
    pub fn compute_rope(
        &self,
        x_pos: &Array,
        cap_pos: &Array,
    ) -> Result<(Array, Array), Exception> {
        let unified_pos = ops::concatenate_axis(&[x_pos, cap_pos], 1)?;
        compute_rope_3axis_cached(&unified_pos, &self.rope_inv_freqs)
    }

    /// Forward pass with pre-computed RoPE
    ///
    /// # Arguments
    /// * `x` - Image latents [batch, img_seq, in_channels * patch^2]
    /// * `t` - Timesteps [batch]
    /// * `cap_feats` - Caption features from Qwen3 [batch, cap_seq, 2560]
    /// * `x_pos` - Image positions [batch, img_seq, 3]
    /// * `cap_pos` - Caption positions [batch, cap_seq, 3]
    /// * `cos`, `sin` - Pre-computed RoPE frequencies
    /// * `x_mask` - Optional image mask
    /// * `cap_mask` - Optional caption mask
    pub fn forward_with_rope(
        &mut self,
        x: &Array,
        t: &Array,
        cap_feats: &Array,
        x_pos: &Array,
        cap_pos: &Array,
        cos: &Array,
        sin: &Array,
        x_mask: Option<&Array>,
        cap_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let x_seq = x.dim(1);
        let cap_seq = cap_feats.dim(1);

        // Timestep embedding (scaled by t_scale)
        let t_scaled = ops::multiply(t, &array!(self.config.t_scale))?;
        let temb = self.t_embedder.forward(&t_scaled)?;

        // Embed image patches
        let mut x = self.x_embedder.forward(x)?;
        if let Some(mask) = x_mask {
            // Apply padding where mask is true
            let mask = mask.reshape(&[mask.dim(0), mask.dim(1), 1])?;
            x = ops::r#where(&mask, &self.x_pad_token, &x)?;
        }

        // Embed caption
        let cap_normed = self.cap_norm.forward(cap_feats)?;
        let mut cap = self.cap_linear.forward(&cap_normed)?;
        if let Some(mask) = cap_mask {
            let mask = mask.reshape(&[mask.dim(0), mask.dim(1), 1])?;
            cap = ops::r#where(&mask, &self.cap_pad_token, &cap)?;
        }

        // Split RoPE for image and caption
        let x_cos = cos.index((.., ..x_seq, .., ..));
        let x_sin = sin.index((.., ..x_seq, .., ..));
        let cap_cos = cos.index((.., x_seq.., .., ..));
        let cap_sin = sin.index((.., x_seq.., .., ..));

        // Noise Refiner: process image tokens
        for block in self.noise_refiner.iter_mut() {
            x = block.forward(&x, &x_cos, &x_sin, Some(&temb), None)?;
        }

        // Context Refiner: process caption tokens (no modulation)
        for block in self.context_refiner.iter_mut() {
            cap = block.forward(&cap, &cap_cos, &cap_sin, None, None)?;
        }

        // Concatenate for joint processing: [x, cap]
        let mut unified = ops::concatenate_axis(&[&x, &cap], 1)?;

        // Joint blocks
        for block in self.layers.iter_mut() {
            unified = block.forward(&unified, cos, sin, Some(&temb), None)?;
        }

        // Extract image tokens
        let img_out = unified.index((.., ..x_seq, ..));

        // Final layer
        self.final_layer.forward(&img_out, &temb)
    }

    /// Convenience forward that computes RoPE internally
    pub fn forward(
        &mut self,
        x: &Array,
        t: &Array,
        cap_feats: &Array,
        x_pos: &Array,
        cap_pos: &Array,
        x_mask: Option<&Array>,
        cap_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let (cos, sin) = self.compute_rope(x_pos, cap_pos)?;
        self.forward_with_rope(x, t, cap_feats, x_pos, cap_pos, &cos, &sin, x_mask, cap_mask)
    }
}

// ============================================================================
// Weight Sanitization
// ============================================================================

/// Sanitize Z-Image weight keys from MLX format to our Rust format
///
/// MLX naming conventions:
/// - cap_embedder.layers.0.weight -> cap_norm.weight
/// - cap_embedder.layers.1.{weight,bias} -> cap_linear.{weight,bias}
/// - final_layer.adaLN_modulation.layers.1.* -> final_layer.ada_ln_modulation.*
/// - *.adaLN_modulation.* -> *.ada_ln_modulation.* (for blocks)
/// - cap_pad_token, x_pad_token are special (non-trainable)
pub fn sanitize_mlx_weights(
    weights: std::collections::HashMap<String, Array>,
) -> std::collections::HashMap<String, Array> {
    let mut sanitized = std::collections::HashMap::new();

    for (key, value) in weights {
        // Skip pad tokens (they're non-trainable in our model)
        if key == "cap_pad_token" || key == "x_pad_token" {
            continue;
        }

        let new_key = key.clone();

        let new_key = new_key
            // Caption embedder (MLX uses .layers.0/.layers.1)
            .replace("cap_embedder.layers.0.", "cap_norm.")
            .replace("cap_embedder.layers.1.", "cap_linear.")
            // Final layer adaLN (MLX uses .layers.1)
            .replace("final_layer.adaLN_modulation.layers.1.", "final_layer.ada_ln_modulation.")
            // Block adaLN modulation (no .layers. in block names)
            .replace(".adaLN_modulation.", ".ada_ln_modulation.");

        sanitized.insert(new_key, value);
    }

    sanitized
}

/// Sanitize Z-Image weight keys from PyTorch diffusers format to our Rust format
///
/// Diffusers naming conventions:
/// - t_embedder.mlp.{0,2} -> t_embedder.linear{1,2} (mlp.1 is SiLU activation)
/// - cap_embedder.{0,1} -> cap_norm, cap_linear
/// - all_x_embedder.{ps}-{fps} -> x_embedder (patch size encoding)
/// - all_final_layer.{ps}-{fps}.* -> final_layer.*
/// - layers/noise_refiner.{idx}.adaLN_modulation.0 -> *.ada_ln_modulation (these blocks only have Linear at .0)
/// - final_layer.adaLN_modulation.1 -> final_layer.ada_ln_modulation (final_layer has SiLU+Linear, Linear is at .1)
/// - attention.to_out.0 -> attention.to_out (skip .1 dropout)
pub fn sanitize_zimage_weights(
    weights: std::collections::HashMap<String, Array>,
) -> std::collections::HashMap<String, Array> {
    use regex::Regex;

    let mut sanitized = std::collections::HashMap::new();

    // Pre-compile regex patterns for patch size keys
    let all_x_embedder_re = Regex::new(r"^all_x_embedder\.\d+-\d+\.").unwrap();
    let all_final_layer_re = Regex::new(r"^all_final_layer\.\d+-\d+\.").unwrap();

    for (key, value) in weights {
        // Skip dropout layers (to_out.1)
        if key.contains("attention.to_out.1.") {
            continue;
        }

        let new_key = key.clone();

        // Handle all_x_embedder.{ps}-{fps} -> x_embedder
        let new_key = all_x_embedder_re.replace(&new_key, "x_embedder.").to_string();

        // Handle all_final_layer.{ps}-{fps} -> final_layer
        let new_key = all_final_layer_re.replace(&new_key, "final_layer.").to_string();

        let new_key = new_key
            // TimestepEmbedder: mlp.0 -> linear1, mlp.2 -> linear2
            .replace("t_embedder.mlp.0.", "t_embedder.linear1.")
            .replace("t_embedder.mlp.2.", "t_embedder.linear2.")
            // Caption embedder
            .replace("cap_embedder.0.", "cap_norm.")
            .replace("cap_embedder.1.", "cap_linear.")
            // Attention output projection (skip dropout layer .1)
            .replace(".attention.to_out.0.", ".attention.to_out.")
            // Block adaLN modulation: blocks have just a Linear at index 0
            // (layers, noise_refiner have only Linear, no SiLU prefix)
            .replace(".adaLN_modulation.0.", ".ada_ln_modulation.")
            // Final layer adaLN: has SiLU(0) + Linear(1), map .1 -> our linear
            .replace("final_layer.adaLN_modulation.1.", "final_layer.ada_ln_modulation.");

        sanitized.insert(new_key, value);
    }

    sanitized
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = ZImageConfig::default();
        assert_eq!(config.dim, 3840);
        assert_eq!(config.n_heads, 30);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.mlp_hidden(), 10240);  // 3840 * 8 / 3
    }

    #[test]
    fn test_coordinate_grid() {
        let grid = create_coordinate_grid((2, 3, 1), (0, 0, 0)).unwrap();
        assert_eq!(grid.shape(), &[6, 3]);
    }

    #[test]
    fn test_rope_3axis() {
        let positions = Array::zeros::<f32>(&[1, 10, 3]).unwrap();
        let (cos, sin) = compute_rope_3axis(&positions, &[32, 48, 48], 256.0).unwrap();
        // Total dim = 32 + 48 + 48 = 128
        assert_eq!(cos.shape(), &[1, 10, 1, 128]);
        assert_eq!(sin.shape(), &[1, 10, 1, 128]);
    }

    #[test]
    fn test_transformer_block_creation() {
        let config = ZImageConfig::default();
        let block = ZImageTransformerBlock::new(&config, true);
        assert!(block.is_ok());
    }
}
