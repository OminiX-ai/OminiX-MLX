//! Full Precision Qwen-Image Transformer
//!
//! Matches the architecture of Qwen/Qwen-Image (HuggingFace diffusers format)
//! Uses joint attention with separate image/text pathways.

use std::collections::HashMap;

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, Param};
use mlx_rs::nn::{Linear, LinearBuilder, RmsNorm, RmsNormBuilder, silu, gelu_approximate};
use mlx_rs::ops::{self, indexing::IndexOp};
use mlx_rs::fast;
use mlx_rs::Array;

// For memory cache clearing
extern crate mlx_sys;

// fused_modulate kernel available but manual implementation is faster
// due to MLX's efficient lazy evaluation (see USE_FUSED_MODULATE flag)

/// Configuration for full precision Qwen-Image Transformer
#[derive(Debug, Clone)]
pub struct QwenFullConfig {
    pub in_channels: i32,          // 64 (patch_size^2 * out_channels)
    pub out_channels: i32,         // 16 (latent channels)
    pub num_layers: i32,           // 60
    pub attention_head_dim: i32,   // 128
    pub num_attention_heads: i32,  // 24
    pub joint_attention_dim: i32,  // 3584 (text encoder dim)
    pub patch_size: i32,           // 2
}

impl Default for QwenFullConfig {
    fn default() -> Self {
        Self {
            in_channels: 64,
            out_channels: 16,
            num_layers: 60,
            attention_head_dim: 128,
            num_attention_heads: 24,
            joint_attention_dim: 3584,
            patch_size: 2,
        }
    }
}

impl QwenFullConfig {
    pub fn inner_dim(&self) -> i32 {
        self.num_attention_heads * self.attention_head_dim // 3072
    }
}

// ============================================================================
// GELU MLP (GELU-approximate activation)
// ============================================================================

/// GELU-approximate Feed Forward network (matches HuggingFace FeedForward)
#[derive(Debug, Clone, ModuleParameters)]
pub struct GeluMLP {
    #[param]
    pub proj_in: Linear,   // net.0.proj - projects to hidden_dim
    #[param]
    pub proj_out: Linear,  // net.2 - projects back to dim
}

impl GeluMLP {
    pub fn new(dim: i32) -> Result<Self, Exception> {
        let hidden_dim = dim * 4;  // 12288 for dim=3072
        Ok(Self {
            proj_in: LinearBuilder::new(dim, hidden_dim).bias(true).build()?,
            proj_out: LinearBuilder::new(hidden_dim, dim).bias(true).build()?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let hidden = self.proj_in.forward(x)?;
        // Use MLX's optimized compiled GELU-approximate
        let activated = gelu_approximate(&hidden)?;
        self.proj_out.forward(&activated)
    }
}

// ============================================================================
// QK Normalization (per-head RMSNorm)
// ============================================================================

/// Per-head RMSNorm for Q/K normalization
#[derive(Debug, Clone, ModuleParameters)]
pub struct QKNorm {
    #[param]
    pub weight: Param<Array>,
    head_dim: i32,
    eps: f32,
}

impl QKNorm {
    pub fn new(head_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::new(Array::ones::<f32>(&[head_dim])?),
            head_dim,
            eps: 1e-6,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        // x: [batch, seq, heads, head_dim]
        // Use MLX's optimized fast RMS norm
        fast::rms_norm(x, &*self.weight, self.eps)
    }
}

// ============================================================================
// Joint Attention
// ============================================================================

/// Joint Attention with separate image/text pathways
#[derive(Debug, Clone, ModuleParameters)]
pub struct JointAttention {
    // Image projections
    #[param]
    pub to_q: Linear,
    #[param]
    pub to_k: Linear,
    #[param]
    pub to_v: Linear,
    #[param]
    pub to_out: Linear,

    // Text projections
    #[param]
    pub add_q_proj: Linear,
    #[param]
    pub add_k_proj: Linear,
    #[param]
    pub add_v_proj: Linear,
    #[param]
    pub to_add_out: Linear,

    // Q/K normalization
    #[param]
    pub norm_q: QKNorm,
    #[param]
    pub norm_k: QKNorm,
    #[param]
    pub norm_added_q: QKNorm,
    #[param]
    pub norm_added_k: QKNorm,

    num_heads: i32,
    head_dim: i32,
}

impl JointAttention {
    pub fn new(dim: i32, num_heads: i32, head_dim: i32) -> Result<Self, Exception> {
        let inner_dim = num_heads * head_dim;
        Ok(Self {
            // Image
            to_q: LinearBuilder::new(dim, inner_dim).bias(true).build()?,
            to_k: LinearBuilder::new(dim, inner_dim).bias(true).build()?,
            to_v: LinearBuilder::new(dim, inner_dim).bias(true).build()?,
            to_out: LinearBuilder::new(inner_dim, dim).bias(true).build()?,
            // Text
            add_q_proj: LinearBuilder::new(dim, inner_dim).bias(true).build()?,
            add_k_proj: LinearBuilder::new(dim, inner_dim).bias(true).build()?,
            add_v_proj: LinearBuilder::new(dim, inner_dim).bias(true).build()?,
            to_add_out: LinearBuilder::new(inner_dim, dim).bias(true).build()?,
            // QK norm
            norm_q: QKNorm::new(head_dim)?,
            norm_k: QKNorm::new(head_dim)?,
            norm_added_q: QKNorm::new(head_dim)?,
            norm_added_k: QKNorm::new(head_dim)?,

            num_heads,
            head_dim,
        })
    }

    pub fn forward(
        &mut self,
        img: &Array,           // [batch, img_seq, dim]
        txt: &Array,           // [batch, txt_seq, dim]
        img_rope: Option<(&Array, &Array)>,
        txt_rope: Option<(&Array, &Array)>,
    ) -> Result<(Array, Array), Exception> {
        let batch = img.shape()[0];
        let img_seq = img.shape()[1];
        let txt_seq = txt.shape()[1];

        // Image Q/K/V
        let img_q = self.to_q.forward(img)?;
        let img_k = self.to_k.forward(img)?;
        let img_v = self.to_v.forward(img)?;

        // Text Q/K/V
        let txt_q = self.add_q_proj.forward(txt)?;
        let txt_k = self.add_k_proj.forward(txt)?;
        let txt_v = self.add_v_proj.forward(txt)?;

        // Reshape to [batch, seq, heads, head_dim]
        let img_q = img_q.reshape(&[batch, img_seq, self.num_heads, self.head_dim])?;
        let img_k = img_k.reshape(&[batch, img_seq, self.num_heads, self.head_dim])?;
        let img_v = img_v.reshape(&[batch, img_seq, self.num_heads, self.head_dim])?;

        let txt_q = txt_q.reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?;
        let txt_k = txt_k.reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?;
        let txt_v = txt_v.reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?;

        // Apply QK normalization
        let img_q = self.norm_q.forward(&img_q)?;
        let img_k = self.norm_k.forward(&img_k)?;
        let txt_q = self.norm_added_q.forward(&txt_q)?;
        let txt_k = self.norm_added_k.forward(&txt_k)?;

        // Apply RoPE
        let img_q = if let Some((cos, sin)) = img_rope {
            apply_rope(&img_q, cos, sin)?
        } else { img_q };
        let img_k = if let Some((cos, sin)) = img_rope {
            apply_rope(&img_k, cos, sin)?
        } else { img_k };
        let txt_q = if let Some((cos, sin)) = txt_rope {
            apply_rope(&txt_q, cos, sin)?
        } else { txt_q };
        let txt_k = if let Some((cos, sin)) = txt_rope {
            apply_rope(&txt_k, cos, sin)?
        } else { txt_k };

        // Concatenate image and text for joint attention
        // Q: [batch, img_seq + txt_seq, heads, head_dim]
        let q = ops::concatenate_axis(&[&txt_q, &img_q], 1)?;
        let k = ops::concatenate_axis(&[&txt_k, &img_k], 1)?;
        let v = ops::concatenate_axis(&[&txt_v, &img_v], 1)?;

        // Transpose to [batch, heads, seq, head_dim] for SDPA
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Use MLX's optimized scaled dot-product attention (Metal kernel)
        let scale = (self.head_dim as f32).sqrt().recip();
        let out = fast::scaled_dot_product_attention(&q, &k, &v, scale, None)?;

        // Transpose back to [batch, seq, heads, head_dim]
        let out = out.transpose_axes(&[0, 2, 1, 3])?;
        let out = out.reshape(&[batch, txt_seq + img_seq, self.num_heads * self.head_dim])?;

        // Split back into text and image
        let txt_out = out.index((.., ..txt_seq, ..));
        let img_out = out.index((.., txt_seq.., ..));

        // Apply output projections
        let txt_out = self.to_add_out.forward(&txt_out)?;
        let img_out = self.to_out.forward(&img_out)?;

        Ok((img_out, txt_out))
    }
}

/// Apply rotary position embedding (Qwen complex-valued style)
/// Qwen uses interleaved pairs: [real, imag, real, imag, ...]
/// This matches use_real=False in diffusers' apply_rotary_emb_qwen
fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array, Exception> {
    // x: [batch, seq, heads, head_dim]
    // cos, sin: [seq, head_dim/2] - one value per pair
    let shape = x.shape();
    let batch = shape[0];
    let seq_len = shape[1];
    let heads = shape[2];
    let head_dim = shape[3];
    let half_dim = head_dim / 2;

    // Get cos/sin for this sequence length
    let cos = cos.index((..seq_len, ..half_dim));
    let sin = sin.index((..seq_len, ..half_dim));

    // Reshape x to expose pairs: [batch, seq, heads, half_dim, 2]
    let x_pairs = x.reshape(&[batch, seq_len, heads, half_dim, 2])?;

    // Extract real and imaginary parts
    let x_real = x_pairs.index((.., .., .., .., 0));  // [batch, seq, heads, half_dim]
    let x_imag = x_pairs.index((.., .., .., .., 1));  // [batch, seq, heads, half_dim]

    // Reshape cos/sin for broadcasting: [1, seq, 1, half_dim]
    let cos = cos.reshape(&[1, seq_len, 1, half_dim])?;
    let sin = sin.reshape(&[1, seq_len, 1, half_dim])?;

    // Complex multiplication: (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
    let out_real = ops::subtract(
        &ops::multiply(&x_real, &cos)?,
        &ops::multiply(&x_imag, &sin)?,
    )?;
    let out_imag = ops::add(
        &ops::multiply(&x_real, &sin)?,
        &ops::multiply(&x_imag, &cos)?,
    )?;

    // Stack and reshape back: [batch, seq, heads, half_dim, 2] -> [batch, seq, heads, head_dim]
    let out_pairs = ops::stack_axis(&[&out_real, &out_imag], -1)?;
    out_pairs.reshape(&[batch, seq_len, heads, head_dim])
}

// ============================================================================
// AdaLayerNorm Modulation
// ============================================================================

/// AdaLayerNorm modulation (outputs shift, scale, gate for attention and MLP)
#[derive(Debug, Clone, ModuleParameters)]
pub struct AdaLayerNormMod {
    #[param]
    pub linear: Linear,  // mod.1 - outputs 6 values (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
}

impl AdaLayerNormMod {
    pub fn new(dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            linear: LinearBuilder::new(dim, dim * 6).bias(true).build()?,
        })
    }

    pub fn forward(&mut self, temb: &Array) -> Result<(Array, Array, Array, Array, Array, Array), Exception> {
        // Order: linear(silu(temb)) - matches diffusers AdaLayerNormZero
        let out = self.linear.forward(&silu(temb)?)?;
        let chunks = ops::split(&out, 6, -1)?;
        Ok((
            chunks[0].clone(),  // shift_attn
            chunks[1].clone(),  // scale_attn
            chunks[2].clone(),  // gate_attn
            chunks[3].clone(),  // shift_mlp
            chunks[4].clone(),  // scale_mlp
            chunks[5].clone(),  // gate_mlp
        ))
    }
}

// ============================================================================
// Transformer Block
// ============================================================================

/// Single transformer block with joint attention
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenFullBlock {
    #[param]
    pub attn: JointAttention,
    #[param]
    pub img_mlp: GeluMLP,
    #[param]
    pub txt_mlp: GeluMLP,
    #[param]
    pub img_mod: AdaLayerNormMod,
    #[param]
    pub txt_mod: AdaLayerNormMod,

    inner_dim: i32,
}

impl QwenFullBlock {
    pub fn new(dim: i32, num_heads: i32, head_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            attn: JointAttention::new(dim, num_heads, head_dim)?,
            img_mlp: GeluMLP::new(dim)?,
            txt_mlp: GeluMLP::new(dim)?,
            img_mod: AdaLayerNormMod::new(dim)?,
            txt_mod: AdaLayerNormMod::new(dim)?,
            inner_dim: dim,
        })
    }

    pub fn forward(
        &mut self,
        img: &Array,
        txt: &Array,
        temb: &Array,
        img_rope: Option<(&Array, &Array)>,
        txt_rope: Option<(&Array, &Array)>,
    ) -> Result<(Array, Array), Exception> {
        // Get modulation parameters
        let (img_shift_attn, img_scale_attn, img_gate_attn, img_shift_mlp, img_scale_mlp, img_gate_mlp) =
            self.img_mod.forward(temb)?;
        let (txt_shift_attn, txt_scale_attn, txt_gate_attn, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp) =
            self.txt_mod.forward(temb)?;

        // Pre-attention modulation (no separate norm - modulation includes implicit norm)
        let img_modulated = modulate(&img, &img_shift_attn, &img_scale_attn)?;
        let txt_modulated = modulate(&txt, &txt_shift_attn, &txt_scale_attn)?;

        // Joint attention
        let (img_attn, txt_attn) = self.attn.forward(&img_modulated, &txt_modulated, img_rope, txt_rope)?;

        // Residual with gate
        let img = ops::add(img, &ops::multiply(&img_attn, &img_gate_attn)?)?;
        let txt = ops::add(txt, &ops::multiply(&txt_attn, &txt_gate_attn)?)?;

        // Pre-MLP modulation
        let img_modulated = modulate(&img, &img_shift_mlp, &img_scale_mlp)?;
        let txt_modulated = modulate(&txt, &txt_shift_mlp, &txt_scale_mlp)?;

        // MLPs
        let img_mlp_out = self.img_mlp.forward(&img_modulated)?;
        let txt_mlp_out = self.txt_mlp.forward(&txt_modulated)?;

        // Residual with gate
        let img = ops::add(&img, &ops::multiply(&img_mlp_out, &img_gate_mlp)?)?;
        let txt = ops::add(&txt, &ops::multiply(&txt_mlp_out, &txt_gate_mlp)?)?;

        Ok((img, txt))
    }
}

/// Use fused Metal kernel for modulation
/// Set to false for better performance (MLX's lazy evaluation is already efficient)
/// The fused kernel is available and working if needed for specific use cases
const USE_FUSED_MODULATE: bool = false;

/// Apply layer norm (no learnable params) then modulation: (1 + scale) * LayerNorm(x) + shift
fn modulate(x: &Array, shift: &Array, scale: &Array) -> Result<Array, Exception> {
    modulate_manual(x, shift, scale)
}

/// Manual modulate implementation
/// MLX lazy evaluation handles Array::from_f32 efficiently - no GPU allocation until eval()
fn modulate_manual(x: &Array, shift: &Array, scale: &Array) -> Result<Array, Exception> {
    // LayerNorm without learnable parameters (elementwise_affine=False)
    let mean = ops::mean_axis(x, -1, true)?;
    let x_centered = ops::subtract(x, &mean)?;
    let var = ops::mean_axis(&ops::multiply(&x_centered, &x_centered)?, -1, true)?;
    let normalized = ops::divide(&x_centered, &ops::sqrt(&ops::add(&var, &Array::from_f32(1e-6))?)?)?;

    // Apply modulation: (1 + scale) * normalized + shift
    let scaled = ops::multiply(&normalized, &ops::add(scale, &Array::from_f32(1.0))?)?;
    ops::add(&scaled, shift)
}

// ============================================================================
// Timestep Embedding
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct TimestepEmbedder {
    /// Pre-computed sinusoidal frequencies (cached for performance)
    /// This eliminates redundant computation across 40 calls per generation.
    pub cached_freqs: Array,

    #[param]
    pub linear_1: Linear,
    #[param]
    pub linear_2: Linear,
}

impl TimestepEmbedder {
    pub fn new(dim: i32) -> Result<Self, Exception> {
        // Pre-compute frequencies ONCE at initialization (from zimage-mlx pattern)
        let half_dim = 128; // 256 / 2
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| (-(i as f32) * (10000.0f32.ln()) / half_dim as f32).exp())
            .collect();
        let cached_freqs = Array::from_slice(&freqs, &[1, half_dim]);

        Ok(Self {
            cached_freqs,
            linear_1: LinearBuilder::new(256, dim).bias(true).build()?,
            linear_2: LinearBuilder::new(dim, dim).bias(true).build()?,
        })
    }

    pub fn forward(&mut self, t: &Array) -> Result<Array, Exception> {
        // Use cached frequencies instead of recomputing
        // t is in [0, 1], scale to [0, 1000] for embedding
        let t_scaled = ops::multiply(t, &Array::from_f32(1000.0))?;
        let t_expanded = t_scaled.reshape(&[-1, 1])?;
        let args = ops::multiply(&t_expanded, &self.cached_freqs)?;

        let cos = ops::cos(&args)?;
        let sin = ops::sin(&args)?;
        let emb = ops::concatenate_axis(&[&cos, &sin], -1)?;

        // MLP with SiLU activation
        let h = silu(&self.linear_1.forward(&emb)?)?;
        self.linear_2.forward(&h)
    }
}

// ============================================================================
// Final Norm with Linear
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct FinalNorm {
    #[param]
    pub linear: Linear,
    eps: f32,
}

impl FinalNorm {
    pub fn new(dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            linear: LinearBuilder::new(dim, dim * 2).bias(true).build()?,  // shift and scale
            eps: 1e-6,
        })
    }

    pub fn forward(&mut self, x: &Array, temb: &Array) -> Result<Array, Exception> {
        // Get scale and shift from temb
        // Order: linear(silu(temb)) - matches diffusers
        let mod_out = self.linear.forward(&silu(temb)?)?;
        let chunks = ops::split(&mod_out, 2, -1)?;
        // First half = scale, second half = shift (matching quantized version)
        let scale = &chunks[0];
        let shift = &chunks[1];

        // RMSNorm
        let x_sq = ops::multiply(x, x)?;
        let mean_sq = ops::mean_axis(&x_sq, -1, true)?;
        let eps = Array::from_f32(self.eps);
        let rms = ops::sqrt(&ops::add(&mean_sq, &eps)?)?;
        let normalized = ops::divide(x, &rms)?;

        // Modulate
        let one = Array::from_f32(1.0);
        let scaled = ops::multiply(&normalized, &ops::add(scale, &one)?)?;
        ops::add(&scaled, shift)
    }
}

// ============================================================================
// Full Transformer
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenFullTransformer {
    #[param]
    pub img_in: Linear,
    #[param]
    pub txt_in: Linear,
    #[param]
    pub txt_norm: RmsNorm,
    #[param]
    pub time_text_embed: TimestepEmbedder,
    #[param]
    pub blocks: Vec<QwenFullBlock>,
    #[param]
    pub norm_out: FinalNorm,
    #[param]
    pub proj_out: Linear,

    config: QwenFullConfig,
}

impl QwenFullTransformer {
    pub fn new(config: QwenFullConfig) -> Result<Self, Exception> {
        let inner_dim = config.inner_dim();

        let mut blocks = Vec::with_capacity(config.num_layers as usize);
        for _ in 0..config.num_layers {
            blocks.push(QwenFullBlock::new(
                inner_dim,
                config.num_attention_heads,
                config.attention_head_dim,
            )?);
        }

        Ok(Self {
            img_in: LinearBuilder::new(config.in_channels, inner_dim).bias(true).build()?,
            txt_in: LinearBuilder::new(config.joint_attention_dim, inner_dim).bias(true).build()?,
            // txt_norm is applied to encoder_hidden_states BEFORE txt_in
            txt_norm: RmsNormBuilder::new(config.joint_attention_dim).build()?,
            time_text_embed: TimestepEmbedder::new(inner_dim)?,
            blocks,
            norm_out: FinalNorm::new(inner_dim)?,
            proj_out: LinearBuilder::new(inner_dim, config.in_channels).bias(true).build()?,
            config,
        })
    }

    pub fn forward(
        &mut self,
        img: &Array,           // [batch, img_seq, 64]
        txt: &Array,           // [batch, txt_seq, 3584]
        timestep: &Array,      // [batch] in [0, 1]
        img_rope: Option<(&Array, &Array)>,
        txt_rope: Option<(&Array, &Array)>,
    ) -> Result<Array, Exception> {
        // Project inputs
        // Note: txt_norm is applied BEFORE txt_in (matches diffusers)
        let mut img = self.img_in.forward(img)?;
        let txt_normed = self.txt_norm.forward(txt)?;
        let mut txt = self.txt_in.forward(&txt_normed)?;

        // Time embedding
        let temb = self.time_text_embed.forward(timestep)?;
        let temb = temb.reshape(&[temb.shape()[0], 1, -1])?;

        // Process through all 60 blocks
        for block in self.blocks.iter_mut() {
            let (new_img, new_txt) = block.forward(&img, &txt, &temb, img_rope, txt_rope)?;
            img = new_img;
            txt = new_txt;
        }

        // Final norm and projection (image only)
        // OPTIMIZATION: Upcast to FP32 before norm_out for numerical stability
        // DiT activations can reach extreme values which causes precision issues
        let input_dtype = img.dtype();
        let img = img.as_dtype(mlx_rs::Dtype::Float32)?;
        let img = self.norm_out.forward(&img, &temb)?;
        let img = img.as_dtype(input_dtype)?;
        self.proj_out.forward(&img)
    }
}

// ============================================================================
// Weight Loading
// ============================================================================

/// Load full precision transformer weights from HuggingFace format
pub fn load_full_precision_weights(
    transformer: &mut QwenFullTransformer,
    weights: HashMap<String, Array>,
) -> Result<(), Box<dyn std::error::Error>> {
    for (name, weight) in weights {
        load_weight(transformer, &name, weight)?;
    }
    Ok(())
}

fn load_weight(
    transformer: &mut QwenFullTransformer,
    name: &str,
    weight: Array,
) -> Result<(), Box<dyn std::error::Error>> {
    let parts: Vec<&str> = name.split('.').collect();

    match parts[0] {
        "img_in" => {
            match parts[1] {
                "weight" => transformer.img_in.weight = Param::new(weight),
                "bias" => transformer.img_in.bias = Param::new(Some(weight)),
                _ => {}
            }
        }
        "txt_in" => {
            match parts[1] {
                "weight" => transformer.txt_in.weight = Param::new(weight),
                "bias" => transformer.txt_in.bias = Param::new(Some(weight)),
                _ => {}
            }
        }
        "txt_norm" => {
            if parts[1] == "weight" {
                transformer.txt_norm.weight = Param::new(weight);
            }
        }
        "time_text_embed" => {
            if parts[1] == "timestep_embedder" {
                match parts[2] {
                    "linear_1" => match parts[3] {
                        "weight" => transformer.time_text_embed.linear_1.weight = Param::new(weight),
                        "bias" => transformer.time_text_embed.linear_1.bias = Param::new(Some(weight)),
                        _ => {}
                    },
                    "linear_2" => match parts[3] {
                        "weight" => transformer.time_text_embed.linear_2.weight = Param::new(weight),
                        "bias" => transformer.time_text_embed.linear_2.bias = Param::new(Some(weight)),
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
        "transformer_blocks" => {
            if let Ok(idx) = parts[1].parse::<usize>() {
                if idx < transformer.blocks.len() {
                    load_block_weight(&mut transformer.blocks[idx], &parts[2..], weight)?;
                }
            }
        }
        "norm_out" => {
            if parts[1] == "linear" {
                match parts[2] {
                    "weight" => transformer.norm_out.linear.weight = Param::new(weight),
                    "bias" => transformer.norm_out.linear.bias = Param::new(Some(weight)),
                    _ => {}
                }
            }
        }
        "proj_out" => {
            match parts[1] {
                "weight" => transformer.proj_out.weight = Param::new(weight),
                "bias" => transformer.proj_out.bias = Param::new(Some(weight)),
                _ => {}
            }
        }
        _ => {
            // Unknown weight, skip
        }
    }

    Ok(())
}

fn load_block_weight(
    block: &mut QwenFullBlock,
    parts: &[&str],
    weight: Array,
) -> Result<(), Box<dyn std::error::Error>> {
    match parts[0] {
        "attn" => {
            match parts[1] {
                "to_q" => match parts[2] {
                    "weight" => block.attn.to_q.weight = Param::new(weight),
                    "bias" => block.attn.to_q.bias = Param::new(Some(weight)),
                    _ => {}
                },
                "to_k" => match parts[2] {
                    "weight" => block.attn.to_k.weight = Param::new(weight),
                    "bias" => block.attn.to_k.bias = Param::new(Some(weight)),
                    _ => {}
                },
                "to_v" => match parts[2] {
                    "weight" => block.attn.to_v.weight = Param::new(weight),
                    "bias" => block.attn.to_v.bias = Param::new(Some(weight)),
                    _ => {}
                },
                "to_out" => {
                    // to_out.0.weight/bias
                    if parts[2] == "0" {
                        match parts[3] {
                            "weight" => block.attn.to_out.weight = Param::new(weight),
                            "bias" => block.attn.to_out.bias = Param::new(Some(weight)),
                            _ => {}
                        }
                    }
                }
                "add_q_proj" => match parts[2] {
                    "weight" => block.attn.add_q_proj.weight = Param::new(weight),
                    "bias" => block.attn.add_q_proj.bias = Param::new(Some(weight)),
                    _ => {}
                },
                "add_k_proj" => match parts[2] {
                    "weight" => block.attn.add_k_proj.weight = Param::new(weight),
                    "bias" => block.attn.add_k_proj.bias = Param::new(Some(weight)),
                    _ => {}
                },
                "add_v_proj" => match parts[2] {
                    "weight" => block.attn.add_v_proj.weight = Param::new(weight),
                    "bias" => block.attn.add_v_proj.bias = Param::new(Some(weight)),
                    _ => {}
                },
                "to_add_out" => match parts[2] {
                    "weight" => block.attn.to_add_out.weight = Param::new(weight),
                    "bias" => block.attn.to_add_out.bias = Param::new(Some(weight)),
                    _ => {}
                },
                "norm_q" => {
                    if parts[2] == "weight" {
                        block.attn.norm_q.weight = Param::new(weight);
                    }
                }
                "norm_k" => {
                    if parts[2] == "weight" {
                        block.attn.norm_k.weight = Param::new(weight);
                    }
                }
                "norm_added_q" => {
                    if parts[2] == "weight" {
                        block.attn.norm_added_q.weight = Param::new(weight);
                    }
                }
                "norm_added_k" => {
                    if parts[2] == "weight" {
                        block.attn.norm_added_k.weight = Param::new(weight);
                    }
                }
                _ => {}
            }
        }
        "img_mlp" => {
            // img_mlp.net.0.proj.weight/bias, img_mlp.net.2.weight/bias
            if parts[1] == "net" {
                match parts[2] {
                    "0" => {
                        if parts[3] == "proj" {
                            match parts[4] {
                                "weight" => block.img_mlp.proj_in.weight = Param::new(weight),
                                "bias" => block.img_mlp.proj_in.bias = Param::new(Some(weight)),
                                _ => {}
                            }
                        }
                    }
                    "2" => match parts[3] {
                        "weight" => block.img_mlp.proj_out.weight = Param::new(weight),
                        "bias" => block.img_mlp.proj_out.bias = Param::new(Some(weight)),
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
        "txt_mlp" => {
            if parts[1] == "net" {
                match parts[2] {
                    "0" => {
                        if parts[3] == "proj" {
                            match parts[4] {
                                "weight" => block.txt_mlp.proj_in.weight = Param::new(weight),
                                "bias" => block.txt_mlp.proj_in.bias = Param::new(Some(weight)),
                                _ => {}
                            }
                        }
                    }
                    "2" => match parts[3] {
                        "weight" => block.txt_mlp.proj_out.weight = Param::new(weight),
                        "bias" => block.txt_mlp.proj_out.bias = Param::new(Some(weight)),
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
        "img_mod" => {
            // img_mod.1.weight/bias
            if parts[1] == "1" {
                match parts[2] {
                    "weight" => block.img_mod.linear.weight = Param::new(weight),
                    "bias" => block.img_mod.linear.bias = Param::new(Some(weight)),
                    _ => {}
                }
            }
        }
        "txt_mod" => {
            if parts[1] == "1" {
                match parts[2] {
                    "weight" => block.txt_mod.linear.weight = Param::new(weight),
                    "bias" => block.txt_mod.linear.bias = Param::new(Some(weight)),
                    _ => {}
                }
            }
        }
        _ => {}
    }

    Ok(())
}
