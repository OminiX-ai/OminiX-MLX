//! Quantized Z-Image-Turbo Transformer Implementation
//!
//! This module provides a 4-bit quantized version of the Z-Image transformer
//! that uses `QuantizedLinear` for memory-efficient inference.
//!
//! # Memory Usage
//!
//! | Mode | Memory |
//! |------|--------|
//! | Dequantized (f32) | ~12GB |
//! | Quantized (4-bit) | ~3GB |
//!
//! # Performance Trade-off
//!
//! Quantized inference is ~11% slower than dequantized due to `quantized_matmul`
//! unpacking overhead. Use quantized mode when memory is constrained.
//!
//! | Mode | Step Time |
//! |------|-----------|
//! | Dequantized (f32) | ~1.87s |
//! | Quantized (4-bit) | ~2.08s |
//!
//! # Usage
//!
//! ```rust,ignore
//! use zimage_mlx::{load_quantized_zimage_transformer, ZImageConfig, load_safetensors};
//!
//! let config = ZImageConfig::default();
//! let weights = load_safetensors("transformer/model.safetensors")?;
//! let mut transformer = load_quantized_zimage_transformer(weights, config)?;
//!
//! // Use transformer.forward_with_rope() for inference
//! ```
//!
//! # MLX Quantization Format
//!
//! The quantized format matches MLX Python's `nn.quantize()` output:
//! - `weight`: uint32 packed 4-bit values (8 values per uint32)
//! - `scales`: float16 per-group scaling factors
//! - `biases`: float16 per-group biases (quantization biases, not layer bias)
//! - `bias`: float16 optional layer bias
//! - `group_size`: 32

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    module::{Module, Param},
    nn::{self, Linear, RmsNorm, QuantizedLinear},
    ops,
    ops::indexing::IndexOp,
    Array,
};
use mlx_macros::ModuleParameters;
use std::collections::HashMap;

use crate::zimage_model::{
    ZImageConfig,
    apply_rope_3axis,
    compute_rope_3axis_cached,
    precompute_rope_inv_freqs,
};

// Constants for timestep embedding
const TIMESTEP_EMBED_DIM: i32 = 256;
const TIMESTEP_MLP_HIDDEN: i32 = 1024;

// ============================================================================
// Quantized Feed Forward (SwiGLU)
// ============================================================================

/// Quantized SwiGLU feed-forward network
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct QuantizedFeedForward {
    #[param]
    pub w1: QuantizedLinear,  // gate projection
    #[param]
    pub w2: QuantizedLinear,  // down projection
    #[param]
    pub w3: QuantizedLinear,  // up projection
}

impl QuantizedFeedForward {
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // SwiGLU: silu(w1(x)) * w3(x)
        let gate = nn::silu(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        let hidden = ops::multiply(&gate, &up)?;
        self.w2.forward(&hidden)
    }
}

// ============================================================================
// Quantized Attention
// ============================================================================

/// Quantized multi-head attention with QK normalization
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct QuantizedAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[param]
    pub to_q: QuantizedLinear,
    #[param]
    pub to_k: QuantizedLinear,
    #[param]
    pub to_v: QuantizedLinear,
    #[param]
    pub to_out: QuantizedLinear,

    #[param]
    pub norm_q: RmsNorm,
    #[param]
    pub norm_k: RmsNorm,
}

impl QuantizedAttention {
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
        let k_t = k.transpose_axes(&[0, 1, 3, 2])?;
        let attn = ops::matmul(&q, &k_t)?;
        let attn = ops::multiply(&attn, &array!(self.scale))?;

        // Apply mask if provided
        let attn = if let Some(m) = mask {
            ops::add(&attn, m)?
        } else {
            attn
        };

        // Softmax and apply to values
        let attn = ops::softmax_axis(&attn, -1, None)?;
        let out = ops::matmul(&attn, &v)?;

        // Transpose back and reshape
        let out = out.transpose_axes(&[0, 2, 1, 3])?;
        let out = out.reshape(&[batch, seq_len, -1])?;

        self.to_out.forward(&out)
    }
}

// ============================================================================
// Quantized Transformer Block
// ============================================================================

/// Quantized Z-Image transformer block
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct QuantizedTransformerBlock {
    pub dim: i32,
    pub has_modulation: bool,

    #[param]
    pub attention: QuantizedAttention,
    #[param]
    pub feed_forward: QuantizedFeedForward,

    #[param]
    pub attention_norm1: RmsNorm,
    #[param]
    pub attention_norm2: RmsNorm,
    #[param]
    pub ffn_norm1: RmsNorm,
    #[param]
    pub ffn_norm2: RmsNorm,

    // AdaLN modulation (quantized)
    #[param]
    pub ada_ln_modulation: Option<QuantizedLinear>,
}

impl QuantizedTransformerBlock {
    pub fn forward(
        &mut self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        adaln_input: Option<&Array>,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        if self.has_modulation {
            let adaln = adaln_input.expect("adaln_input required for modulated blocks");

            // Get modulation parameters
            let chunks = self.ada_ln_modulation.as_mut().unwrap().forward(adaln)?;
            let splits = chunks.split(4, -1)?;
            let scale_msa = &splits[0];
            let gate_msa = &splits[1];
            let scale_mlp = &splits[2];
            let gate_mlp = &splits[3];

            // Expand for broadcasting
            let scale_msa = scale_msa.reshape(&[scale_msa.dim(0), 1, self.dim])?;
            let gate_msa = gate_msa.reshape(&[gate_msa.dim(0), 1, self.dim])?;
            let scale_mlp = scale_mlp.reshape(&[scale_mlp.dim(0), 1, self.dim])?;
            let gate_mlp = gate_mlp.reshape(&[gate_mlp.dim(0), 1, self.dim])?;

            // Attention with modulation
            let norm_x = self.attention_norm1.forward(x)?;
            let one = array!(1.0f32);
            let modulated = ops::multiply(&norm_x, &ops::add(&one, &scale_msa)?)?;
            let attn_out = self.attention.forward(&modulated, cos, sin, mask)?;

            let gated_attn = ops::multiply(&ops::tanh(&gate_msa)?, &self.attention_norm2.forward(&attn_out)?)?;
            let x = ops::add(x, &gated_attn)?;

            // FFN with modulation
            let norm_x = self.ffn_norm1.forward(&x)?;
            let modulated = ops::multiply(&norm_x, &ops::add(&one, &scale_mlp)?)?;
            let ffn_out = self.feed_forward.forward(&modulated)?;

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
// Quantized Timestep Embedder
// ============================================================================

/// Quantized timestep embedding
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct QuantizedTimestepEmbedder {
    pub frequency_embedding_size: i32,
    pub cached_freqs: Array,

    #[param]
    pub linear1: QuantizedLinear,
    #[param]
    pub linear2: QuantizedLinear,
}

impl QuantizedTimestepEmbedder {
    pub fn forward(&mut self, t: &Array) -> Result<Array, Exception> {
        let t = t.as_dtype(mlx_rs::Dtype::Float32)?;
        let t = t.reshape(&[-1, 1])?;
        let args = ops::multiply(&t, &self.cached_freqs)?;

        let cos_args = ops::cos(&args)?;
        let sin_args = ops::sin(&args)?;
        let embedding = ops::concatenate_axis(&[&cos_args, &sin_args], -1)?;

        let x = self.linear1.forward(&embedding)?;
        let x = nn::silu(&x)?;
        self.linear2.forward(&x)
    }
}

// ============================================================================
// Quantized Final Layer
// ============================================================================

/// Quantized final output layer
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct QuantizedFinalLayer {
    pub dim: i32,
    pub out_channels: i32,

    #[param]
    pub norm_final: nn::LayerNorm,
    #[param]
    pub linear: QuantizedLinear,
    #[param]
    pub ada_ln_modulation: QuantizedLinear,
}

impl QuantizedFinalLayer {
    pub fn forward(&mut self, x: &Array, c: &Array) -> Result<Array, Exception> {
        let scale = nn::silu(c)?;
        let scale = self.ada_ln_modulation.forward(&scale)?;
        let scale = scale.reshape(&[scale.dim(0), 1, self.dim])?;

        let one = array!(1.0f32);
        let normed = self.norm_final.forward(x)?;
        let modulated = ops::multiply(&normed, &ops::add(&one, &scale)?)?;
        self.linear.forward(&modulated)
    }
}

// ============================================================================
// Quantized Z-Image Transformer
// ============================================================================

/// Quantized Z-Image-Turbo Transformer
///
/// Uses 4-bit quantized weights for memory-efficient inference.
/// Memory usage: ~3GB instead of ~12GB with f32 weights.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct ZImageTransformerQuantized {
    pub config: ZImageConfig,
    pub rope_inv_freqs: [Array; 3],

    #[param]
    pub t_embedder: QuantizedTimestepEmbedder,
    #[param]
    pub x_embedder: QuantizedLinear,
    #[param]
    pub cap_norm: RmsNorm,
    #[param]
    pub cap_linear: QuantizedLinear,

    pub x_pad_token: Array,
    pub cap_pad_token: Array,

    #[param]
    pub noise_refiner: Vec<QuantizedTransformerBlock>,
    #[param]
    pub context_refiner: Vec<QuantizedTransformerBlock>,
    #[param]
    pub layers: Vec<QuantizedTransformerBlock>,

    #[param]
    pub final_layer: QuantizedFinalLayer,
}

impl ZImageTransformerQuantized {
    /// Compute RoPE frequencies for caching
    pub fn compute_rope(
        &self,
        x_pos: &Array,
        cap_pos: &Array,
    ) -> Result<(Array, Array), Exception> {
        let unified_pos = ops::concatenate_axis(&[x_pos, cap_pos], 1)?;
        compute_rope_3axis_cached(&unified_pos, &self.rope_inv_freqs)
    }

    /// Forward pass with pre-computed RoPE
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

        // Timestep embedding
        let t_scaled = ops::multiply(t, &array!(self.config.t_scale))?;
        let temb = self.t_embedder.forward(&t_scaled)?;

        // Embed image patches
        let mut x = self.x_embedder.forward(x)?;
        if let Some(mask) = x_mask {
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

        // Split RoPE
        let x_cos = cos.index((.., ..x_seq, .., ..));
        let x_sin = sin.index((.., ..x_seq, .., ..));
        let cap_cos = cos.index((.., x_seq.., .., ..));
        let cap_sin = sin.index((.., x_seq.., .., ..));

        // Noise Refiner
        for block in self.noise_refiner.iter_mut() {
            x = block.forward(&x, &x_cos, &x_sin, Some(&temb), None)?;
        }

        // Context Refiner
        for block in self.context_refiner.iter_mut() {
            cap = block.forward(&cap, &cap_cos, &cap_sin, None, None)?;
        }

        // Concatenate for joint processing
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
// Weight Loading
// ============================================================================

/// Create a QuantizedLinear from pre-quantized weights
///
/// # Arguments
/// * `weights` - HashMap containing the layer's weights
/// * `prefix` - Key prefix (e.g., "layers.0.attention.to_q")
/// * `group_size` - Quantization group size (32 for Z-Image)
/// * `bits` - Bits per weight (4 for Z-Image)
/// * `has_bias` - Whether the layer has a linear bias term
pub fn create_quantized_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
    has_bias: bool,
) -> Result<QuantizedLinear, Exception> {
    let weight_key = format!("{}.weight", prefix);
    let scales_key = format!("{}.scales", prefix);
    let biases_key = format!("{}.biases", prefix);
    let bias_key = format!("{}.bias", prefix);

    let weight = weights.get(&weight_key)
        .unwrap_or_else(|| panic!("Missing weight: {}", weight_key))
        .clone();
    let scales = weights.get(&scales_key)
        .unwrap_or_else(|| panic!("Missing scales: {}", scales_key))
        .clone();
    let biases = weights.get(&biases_key)
        .unwrap_or_else(|| panic!("Missing biases: {}", biases_key))
        .clone();

    let bias = if has_bias {
        weights.get(&bias_key).cloned()
    } else {
        None
    };

    let inner = Linear {
        weight: Param::new(weight),
        bias: Param::new(bias),
    };

    Ok(QuantizedLinear {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    })
}

/// Create a QuantizedAttention from pre-quantized weights
pub fn create_quantized_attention(
    weights: &HashMap<String, Array>,
    prefix: &str,
    config: &ZImageConfig,
) -> Result<QuantizedAttention, Exception> {
    let head_dim = config.head_dim();
    let scale = (head_dim as f32).powf(-0.5);

    Ok(QuantizedAttention {
        n_heads: config.n_heads,
        n_kv_heads: config.n_kv_heads,
        head_dim,
        scale,
        to_q: create_quantized_linear(weights, &format!("{}.to_q", prefix), 32, 4, false)?,
        to_k: create_quantized_linear(weights, &format!("{}.to_k", prefix), 32, 4, false)?,
        to_v: create_quantized_linear(weights, &format!("{}.to_v", prefix), 32, 4, false)?,
        to_out: create_quantized_linear(weights, &format!("{}.to_out", prefix), 32, 4, false)?,
        norm_q: nn::RmsNormBuilder::new(head_dim).eps(config.norm_eps).build()?,
        norm_k: nn::RmsNormBuilder::new(head_dim).eps(config.norm_eps).build()?,
    })
}

/// Create a QuantizedFeedForward from pre-quantized weights
pub fn create_quantized_feed_forward(
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<QuantizedFeedForward, Exception> {
    Ok(QuantizedFeedForward {
        w1: create_quantized_linear(weights, &format!("{}.w1", prefix), 32, 4, false)?,
        w2: create_quantized_linear(weights, &format!("{}.w2", prefix), 32, 4, false)?,
        w3: create_quantized_linear(weights, &format!("{}.w3", prefix), 32, 4, false)?,
    })
}

/// Create a QuantizedTransformerBlock from pre-quantized weights
pub fn create_quantized_block(
    weights: &HashMap<String, Array>,
    prefix: &str,
    config: &ZImageConfig,
    has_modulation: bool,
) -> Result<QuantizedTransformerBlock, Exception> {
    let dim = config.dim;

    let attention = create_quantized_attention(weights, &format!("{}.attention", prefix), config)?;
    let feed_forward = create_quantized_feed_forward(weights, &format!("{}.feed_forward", prefix))?;

    // Load RmsNorm weights
    let load_rms_norm = |name: &str| -> Result<RmsNorm, Exception> {
        let key = format!("{}.{}.weight", prefix, name);
        let weight = weights.get(&key)
            .unwrap_or_else(|| panic!("Missing norm weight: {}", key))
            .clone();
        let mut norm = nn::RmsNormBuilder::new(dim).eps(config.norm_eps).build()?;
        norm.weight = Param::new(weight);
        Ok(norm)
    };

    // Load norm_q and norm_k weights into attention
    let norm_q_key = format!("{}.attention.norm_q.weight", prefix);
    let norm_k_key = format!("{}.attention.norm_k.weight", prefix);

    let mut attention = attention;
    if let Some(w) = weights.get(&norm_q_key) {
        attention.norm_q.weight = Param::new(w.clone());
    }
    if let Some(w) = weights.get(&norm_k_key) {
        attention.norm_k.weight = Param::new(w.clone());
    }

    let ada_ln_modulation = if has_modulation {
        Some(create_quantized_linear(weights, &format!("{}.ada_ln_modulation", prefix), 32, 4, true)?)
    } else {
        None
    };

    Ok(QuantizedTransformerBlock {
        dim,
        has_modulation,
        attention,
        feed_forward,
        attention_norm1: load_rms_norm("attention_norm1")?,
        attention_norm2: load_rms_norm("attention_norm2")?,
        ffn_norm1: load_rms_norm("ffn_norm1")?,
        ffn_norm2: load_rms_norm("ffn_norm2")?,
        ada_ln_modulation,
    })
}

/// Create a QuantizedTimestepEmbedder from pre-quantized weights
pub fn create_quantized_timestep_embedder(
    weights: &HashMap<String, Array>,
) -> Result<QuantizedTimestepEmbedder, Exception> {
    let half = TIMESTEP_EMBED_DIM / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-10000.0f32.ln() * (i as f32) / (half as f32)).exp())
        .collect();
    let cached_freqs = Array::from_slice(&freqs, &[1, half]);

    Ok(QuantizedTimestepEmbedder {
        frequency_embedding_size: TIMESTEP_EMBED_DIM,
        cached_freqs,
        linear1: create_quantized_linear(weights, "t_embedder.linear1", 32, 4, true)?,
        linear2: create_quantized_linear(weights, "t_embedder.linear2", 32, 4, true)?,
    })
}

/// Create a QuantizedFinalLayer from pre-quantized weights
pub fn create_quantized_final_layer(
    weights: &HashMap<String, Array>,
    config: &ZImageConfig,
) -> Result<QuantizedFinalLayer, Exception> {
    let dim = config.dim;
    let patch_channels = config.in_channels * config.patch_size * config.patch_size;

    Ok(QuantizedFinalLayer {
        dim,
        out_channels: patch_channels,
        norm_final: nn::LayerNormBuilder::new(dim).affine(false).eps(1e-6).build()?,
        linear: create_quantized_linear(weights, "final_layer.linear", 32, 4, true)?,
        ada_ln_modulation: create_quantized_linear(weights, "final_layer.ada_ln_modulation", 32, 4, true)?,
    })
}

/// Load a complete quantized Z-Image transformer from MLX weights
///
/// # Arguments
/// * `weights` - HashMap of weight name -> Array (from safetensors)
/// * `config` - Model configuration
pub fn load_quantized_zimage_transformer(
    weights: HashMap<String, Array>,
    config: ZImageConfig,
) -> Result<ZImageTransformerQuantized, Exception> {
    // Sanitize weight keys
    let weights = sanitize_quantized_weights(weights);

    // Pre-compute RoPE frequencies
    let rope_inv_freqs = precompute_rope_inv_freqs(&config.axes_dims, config.rope_theta);

    // Load components
    let t_embedder = create_quantized_timestep_embedder(&weights)?;
    let x_embedder = create_quantized_linear(&weights, "x_embedder", 32, 4, true)?;

    // Caption embedder
    let cap_norm_key = "cap_norm.weight";
    let cap_norm_weight = weights.get(cap_norm_key)
        .expect("Missing cap_norm.weight")
        .clone();
    let mut cap_norm = nn::RmsNormBuilder::new(config.cap_feat_dim).eps(config.norm_eps).build()?;
    cap_norm.weight = Param::new(cap_norm_weight);

    let cap_linear = create_quantized_linear(&weights, "cap_linear", 32, 4, true)?;

    // Pad tokens
    let x_pad_token = weights.get("x_pad_token")
        .cloned()
        .unwrap_or_else(|| Array::zeros::<f32>(&[1, config.dim]).unwrap());
    let cap_pad_token = weights.get("cap_pad_token")
        .cloned()
        .unwrap_or_else(|| Array::zeros::<f32>(&[1, config.dim]).unwrap());

    // Load transformer blocks
    let mut noise_refiner = Vec::with_capacity(config.n_refiner_layers as usize);
    for i in 0..config.n_refiner_layers {
        let prefix = format!("noise_refiner.{}", i);
        noise_refiner.push(create_quantized_block(&weights, &prefix, &config, true)?);
    }

    let mut context_refiner = Vec::with_capacity(config.n_refiner_layers as usize);
    for i in 0..config.n_refiner_layers {
        let prefix = format!("context_refiner.{}", i);
        context_refiner.push(create_quantized_block(&weights, &prefix, &config, false)?);
    }

    let mut layers = Vec::with_capacity(config.n_layers as usize);
    for i in 0..config.n_layers {
        let prefix = format!("layers.{}", i);
        layers.push(create_quantized_block(&weights, &prefix, &config, true)?);
    }

    let final_layer = create_quantized_final_layer(&weights, &config)?;

    Ok(ZImageTransformerQuantized {
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

/// Sanitize MLX quantized weight keys to our format
fn sanitize_quantized_weights(
    weights: HashMap<String, Array>,
) -> HashMap<String, Array> {
    let mut sanitized = HashMap::new();

    for (key, value) in weights {
        let new_key = key
            // Caption embedder
            .replace("cap_embedder.layers.0.", "cap_norm.")
            .replace("cap_embedder.layers.1.", "cap_linear.")
            // Final layer adaLN
            .replace("final_layer.adaLN_modulation.layers.1.", "final_layer.ada_ln_modulation.")
            // Block adaLN modulation
            .replace(".adaLN_modulation.", ".ada_ln_modulation.");

        sanitized.insert(new_key, value);
    }

    sanitized
}
