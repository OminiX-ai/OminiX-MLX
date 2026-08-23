//! Quantized Qwen-Image Transformer
//!
//! Matches the weight structure of mlx-community/Qwen-Image-2512-4bit

use std::collections::HashMap;
use std::rc::Rc;

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::fast::{self, ScaledDotProductAttentionMask};
use mlx_rs::module::{Module, ModuleParameters, Param};
use mlx_rs::nn::{RmsNorm, RmsNormBuilder};
use mlx_rs::nn::{QuantizedLinear, QuantizedLinearBuilder};
use mlx_rs::ops::{self, indexing::IndexOp};
use mlx_rs::{Array, Dtype};

/// Configuration for Qwen-Image Transformer
#[derive(Debug, Clone)]
pub struct QwenConfig {
    pub in_channels: i32,          // 64 (patch_size^2 * latent_channels)
    pub out_channels: i32,         // 16 (latent_channels)
    pub num_layers: i32,           // 60
    pub attention_head_dim: i32,   // 128
    pub num_attention_heads: i32,  // 24
    pub joint_attention_dim: i32,  // 3584
    pub patch_size: i32,           // 2
    pub quantization_bits: i32,    // 4 or 8
    pub quantization_group_size: i32, // 64
}

impl Default for QwenConfig {
    fn default() -> Self {
        Self {
            in_channels: 64,
            out_channels: 16,
            num_layers: 60,
            attention_head_dim: 128,
            num_attention_heads: 24,
            joint_attention_dim: 3584,
            patch_size: 2,
            quantization_bits: 4,
            quantization_group_size: 64,
        }
    }
}

impl QwenConfig {
    pub fn with_8bit() -> Self {
        Self {
            quantization_bits: 8,
            quantization_group_size: 64,
            ..Default::default()
        }
    }
}

impl QwenConfig {
    pub fn inner_dim(&self) -> i32 {
        self.num_attention_heads * self.attention_head_dim // 3072
    }

    /// Create a QuantizedLinear with the config's quantization settings
    pub fn quantized_linear(&self, input_dims: i32, output_dims: i32) -> Result<QuantizedLinear, Exception> {
        QuantizedLinearBuilder::new(input_dims, output_dims)
            .group_size(self.quantization_group_size)
            .bits(self.quantization_bits)
            .build()
    }
}

/// Quantized Feed Forward network
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenFeedForward {
    #[param]
    pub mlp_in: QuantizedLinear,  // GELU gate projection
    #[param]
    pub mlp_out: QuantizedLinear, // Output projection
}

impl QwenFeedForward {
    pub fn new(dim: i32, config: &QwenConfig) -> Result<Self, Exception> {
        let hidden_dim = dim * 4; // 12288
        Ok(Self {
            mlp_in: config.quantized_linear(dim, hidden_dim)?,
            mlp_out: config.quantized_linear(hidden_dim, dim)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let hidden = self.mlp_in.forward(x)?;
        let hidden = mlx_rs::nn::gelu_approximate(&hidden)?;
        self.mlp_out.forward(&hidden)
    }
}

/// Quantized Attention
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenAttention {
    pub dim: i32,
    pub num_heads: i32,
    pub head_dim: i32,

    // Image projections (quantized)
    #[param]
    pub to_q: QuantizedLinear,
    #[param]
    pub to_k: QuantizedLinear,
    #[param]
    pub to_v: QuantizedLinear,

    // Text projections (quantized)
    #[param]
    pub add_q_proj: QuantizedLinear,
    #[param]
    pub add_k_proj: QuantizedLinear,
    #[param]
    pub add_v_proj: QuantizedLinear,

    // RMSNorm (not quantized)
    #[param]
    pub norm_q: RmsNorm,
    #[param]
    pub norm_k: RmsNorm,
    #[param]
    pub norm_added_q: RmsNorm,
    #[param]
    pub norm_added_k: RmsNorm,

    // Output projections (quantized)
    #[param]
    pub attn_to_out: Vec<QuantizedLinear>, // Single element list to match weight name "attn_to_out.0"
    #[param]
    pub to_add_out: QuantizedLinear,
}

impl QwenAttention {
    pub fn new(dim: i32, num_heads: i32, head_dim: i32, config: &QwenConfig) -> Result<Self, Exception> {
        Ok(Self {
            dim,
            num_heads,
            head_dim,
            to_q: config.quantized_linear(dim, dim)?,
            to_k: config.quantized_linear(dim, dim)?,
            to_v: config.quantized_linear(dim, dim)?,
            add_q_proj: config.quantized_linear(dim, dim)?,
            add_k_proj: config.quantized_linear(dim, dim)?,
            add_v_proj: config.quantized_linear(dim, dim)?,
            norm_q: RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            norm_k: RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            norm_added_q: RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            norm_added_k: RmsNormBuilder::new(head_dim).eps(1e-6).build()?,
            attn_to_out: vec![config.quantized_linear(dim, dim)?],
            to_add_out: config.quantized_linear(dim, dim)?,
        })
    }

    pub fn forward(
        &mut self,
        img_modulated: &Array,
        txt_modulated: &Array,
        img_rotary_emb: Option<(&Array, &Array)>,
        txt_rotary_emb: Option<(&Array, &Array)>,
        encoder_hidden_states_mask: Option<&Array>,  // [B, txt_seq] with 1 for real, 0 for padding
    ) -> Result<(Array, Array), Exception> {
        let batch = img_modulated.dim(0);

        // Image projections
        let mut img_q = self.to_q.forward(img_modulated)?;
        let mut img_k = self.to_k.forward(img_modulated)?;
        let img_v = self.to_v.forward(img_modulated)?;

        // Text projections
        let mut txt_q = self.add_q_proj.forward(txt_modulated)?;
        let mut txt_k = self.add_k_proj.forward(txt_modulated)?;
        let txt_v = self.add_v_proj.forward(txt_modulated)?;

        // Reshape to [B, seq, heads, head_dim]
        let img_seq = img_q.dim(1);
        let txt_seq = txt_q.dim(1);

        img_q = img_q.reshape(&[batch, img_seq, self.num_heads, self.head_dim])?;
        img_k = img_k.reshape(&[batch, img_seq, self.num_heads, self.head_dim])?;
        let img_v = img_v.reshape(&[batch, img_seq, self.num_heads, self.head_dim])?;

        txt_q = txt_q.reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?;
        txt_k = txt_k.reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?;
        let txt_v = txt_v.reshape(&[batch, txt_seq, self.num_heads, self.head_dim])?;

        // Apply RMSNorm
        img_q = self.norm_q.forward(&img_q)?;
        img_k = self.norm_k.forward(&img_k)?;
        txt_q = self.norm_added_q.forward(&txt_q)?;
        txt_k = self.norm_added_k.forward(&txt_k)?;

        // Apply RoPE if provided
        if let Some((cos, sin)) = img_rotary_emb {
            img_q = apply_rope_qwen(&img_q, cos, sin)?;
            img_k = apply_rope_qwen(&img_k, cos, sin)?;
        }
        if let Some((cos, sin)) = txt_rotary_emb {
            txt_q = apply_rope_qwen(&txt_q, cos, sin)?;
            txt_k = apply_rope_qwen(&txt_k, cos, sin)?;
        }

        // Concatenate for joint attention
        let joint_q = ops::concatenate_axis(&[&txt_q, &img_q], 1)?;
        let joint_k = ops::concatenate_axis(&[&txt_k, &img_k], 1)?;
        let joint_v = ops::concatenate_axis(&[&txt_v, &img_v], 1)?;

        // Scaled dot-product attention
        // Transpose to [B, heads, seq, head_dim]
        let q = joint_q.transpose_axes(&[0, 2, 1, 3])?;
        let k = joint_k.transpose_axes(&[0, 2, 1, 3])?;
        let v = joint_v.transpose_axes(&[0, 2, 1, 3])?;

        // P1.1: fused SDPA — one kernel instead of matmul/scale/softmax/matmul,
        // and no materialized K^T transpose.
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Build additive mask if provided: 0 for real tokens, -1e9 for padding
        let out = if let Some(mask) = encoder_hidden_states_mask {
            let img_seq = img_modulated.dim(1);
            let ones_img = Array::ones::<f32>(&[batch, img_seq])?;
            let joint_mask = ops::concatenate_axis(&[mask, &ones_img], 1)?;
            let additive_mask = ops::multiply(
                &ops::subtract(&Array::from_f32(1.0), &joint_mask)?,
                &Array::from_f32(-1e9),
            )?;
            let additive_mask = additive_mask.reshape(&[batch, 1, 1, txt_seq + img_seq])?;
            fast::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                scale,
                ScaledDotProductAttentionMask::Array(&additive_mask),
            )?
        } else {
            fast::scaled_dot_product_attention(&q, &k, &v, scale, None)?
        };

        // Transpose back and reshape
        let out = out.transpose_axes(&[0, 2, 1, 3])?;
        let joint_seq = out.dim(1);
        let out = out.reshape(&[batch, joint_seq, self.dim])?;

        // Split output
        let txt_out = out.index((.., ..txt_seq, ..));
        let img_out = out.index((.., txt_seq.., ..));

        // Output projections
        let img_out = self.attn_to_out[0].forward(&img_out)?;
        let txt_out = self.to_add_out.forward(&txt_out)?;

        Ok((img_out, txt_out))
    }
}

/// Quantized Transformer Block
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTransformerBlock {
    pub dim: i32,

    // Image modulation
    #[param]
    pub img_mod_linear: QuantizedLinear,

    // Text modulation
    #[param]
    pub txt_mod_linear: QuantizedLinear,

    // Attention
    #[param]
    pub attn: QwenAttention,

    // Image FFN
    #[param]
    pub img_ff: QwenFeedForward,

    // Text FFN
    #[param]
    pub txt_ff: QwenFeedForward,
}

impl QwenTransformerBlock {
    pub fn new(dim: i32, num_heads: i32, head_dim: i32, config: &QwenConfig) -> Result<Self, Exception> {
        Ok(Self {
            dim,
            img_mod_linear: config.quantized_linear(dim, dim * 6)?,
            txt_mod_linear: config.quantized_linear(dim, dim * 6)?,
            attn: QwenAttention::new(dim, num_heads, head_dim, config)?,
            img_ff: QwenFeedForward::new(dim, config)?,
            txt_ff: QwenFeedForward::new(dim, config)?,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,      // Image hidden states
        encoder_hidden_states: &Array, // Text hidden states
        text_embeddings: &Array,    // Time embeddings
        img_rotary_emb: Option<(&Array, &Array)>,
        txt_rotary_emb: Option<(&Array, &Array)>,
        encoder_hidden_states_mask: Option<&Array>,
    ) -> Result<(Array, Array), Exception> {
        // Image modulation
        let img_silu = mlx_rs::nn::silu(text_embeddings)?;
        let img_mod_params = self.img_mod_linear.forward(&img_silu)?;
        let (img_mod1, img_mod2) = split_half(&img_mod_params)?;

        // Text modulation
        let txt_silu = mlx_rs::nn::silu(text_embeddings)?;
        let txt_mod_params = self.txt_mod_linear.forward(&txt_silu)?;
        let (txt_mod1, txt_mod2) = split_half(&txt_mod_params)?;

        // Apply LayerNorm and modulation
        let img_normed = layer_norm(hidden_states, 1e-6)?;
        let (img_modulated, img_gate1) = modulate(&img_normed, &img_mod1)?;
        let txt_normed = layer_norm(encoder_hidden_states, 1e-6)?;
        let (txt_modulated, txt_gate1) = modulate(&txt_normed, &txt_mod1)?;

        // Joint attention
        let (img_attn_out, txt_attn_out) = self.attn.forward(
            &img_modulated,
            &txt_modulated,
            img_rotary_emb,
            txt_rotary_emb,
            encoder_hidden_states_mask,
        )?;

        // Image: gate + residual
        let img_gate1_exp = img_gate1.expand_dims(1)?;
        let hidden_states = ops::add(hidden_states, &ops::multiply(&img_gate1_exp, &img_attn_out)?)?;

        // Text: gate + residual
        let txt_gate1_exp = txt_gate1.expand_dims(1)?;
        let encoder_hidden_states = ops::add(encoder_hidden_states, &ops::multiply(&txt_gate1_exp, &txt_attn_out)?)?;

        // Image FFN with mod2
        let img_normed2 = layer_norm(&hidden_states, 1e-6)?;
        let (img_modulated2, img_gate2) = modulate(&img_normed2, &img_mod2)?;
        let img_mlp_out = self.img_ff.forward(&img_modulated2)?;
        let img_gate2_exp = img_gate2.expand_dims(1)?;
        let hidden_states = ops::add(&hidden_states, &ops::multiply(&img_gate2_exp, &img_mlp_out)?)?;

        // Text FFN with mod2
        let txt_normed2 = layer_norm(&encoder_hidden_states, 1e-6)?;
        let (txt_modulated2, txt_gate2) = modulate(&txt_normed2, &txt_mod2)?;
        let txt_mlp_out = self.txt_ff.forward(&txt_modulated2)?;
        let txt_gate2_exp = txt_gate2.expand_dims(1)?;
        let encoder_hidden_states = ops::add(&encoder_hidden_states, &ops::multiply(&txt_gate2_exp, &txt_mlp_out)?)?;

        Ok((encoder_hidden_states, hidden_states))
    }

    /// Edit-mode forward: dual time-embedding with per-token modulation blending.
    /// temb: [2, dim] — row 0 = real timestep embedding, row 1 = zero timestep embedding
    /// modulate_index: [total_img_seq] — 0.0 for main tokens, 1.0 for ref tokens
    /// encoder_hidden_states_mask: optional [B, txt_seq] for batched CFG padding
    pub fn forward_edit(
        &mut self,
        hidden_states: &Array,
        encoder_hidden_states: &Array,
        temb: &Array,
        img_rotary_emb: (&Array, &Array),
        txt_rotary_emb: (&Array, &Array),
        modulate_index: &Array,
        encoder_hidden_states_mask: Option<&Array>,
    ) -> Result<(Array, Array), Exception> {
        let dim = self.dim;

        // Image modulation: project dual temb through img_mod_linear
        let img_silu = mlx_rs::nn::silu(temb)?; // [2, dim]
        let img_mod_params = self.img_mod_linear.forward(&img_silu)?; // [2, 6*dim]
        let (img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2) =
            prepare_img_mod_edit(&img_mod_params, modulate_index, dim)?;
        // Each is [1, total_img_seq, dim]

        // Text modulation: use real temb only (first row)
        let real_temb = temb.index((0..1, ..)); // [1, dim]
        let txt_silu = mlx_rs::nn::silu(&real_temb)?;
        let txt_mod_params = self.txt_mod_linear.forward(&txt_silu)?; // [1, 6*dim]
        let txt_shift1 = txt_mod_params.index((.., ..dim));
        let txt_scale1 = txt_mod_params.index((.., dim..dim * 2));
        let txt_gate1 = txt_mod_params.index((.., dim * 2..dim * 3));
        let txt_shift2 = txt_mod_params.index((.., dim * 3..dim * 4));
        let txt_scale2 = txt_mod_params.index((.., dim * 4..dim * 5));
        let txt_gate2 = txt_mod_params.index((.., dim * 5..));

        // LayerNorm + modulation
        let img_normed = layer_norm(hidden_states, 1e-6)?;
        let txt_normed = layer_norm(encoder_hidden_states, 1e-6)?;
        let img_modulated = modulate_flex(&img_normed, &img_shift1, &img_scale1)?;
        let txt_modulated = modulate_2d(&txt_normed, &txt_shift1, &txt_scale1)?;

        // Joint attention (unchanged — just operates on larger img seq)
        let (img_attn_out, txt_attn_out) = self.attn.forward(
            &img_modulated,
            &txt_modulated,
            Some((&img_rotary_emb.0, &img_rotary_emb.1)),
            Some((&txt_rotary_emb.0, &txt_rotary_emb.1)),
            encoder_hidden_states_mask,
        )?;

        // Image: gate_flex + residual
        let hidden_states = ops::add(hidden_states, &gate_flex(&img_gate1, &img_attn_out)?)?;

        // Text: standard gate + residual
        let txt_gate1_exp = txt_gate1.expand_dims(1)?;
        let encoder_hidden_states = ops::add(
            encoder_hidden_states,
            &ops::multiply(&txt_gate1_exp, &txt_attn_out)?,
        )?;

        // Image FFN with mod2
        let img_normed2 = layer_norm(&hidden_states, 1e-6)?;
        let img_mod2_input = modulate_flex(&img_normed2, &img_shift2, &img_scale2)?;
        let img_mlp_out = self.img_ff.forward(&img_mod2_input)?;
        let hidden_states = ops::add(&hidden_states, &gate_flex(&img_gate2, &img_mlp_out)?)?;

        // Text FFN with mod2 (standard 2D modulation)
        let txt_normed2 = layer_norm(&encoder_hidden_states, 1e-6)?;
        let txt_mod2_input = modulate_2d(&txt_normed2, &txt_shift2, &txt_scale2)?;
        let txt_mlp_out = self.txt_ff.forward(&txt_mod2_input)?;
        let txt_gate2_exp = txt_gate2.expand_dims(1)?;
        let encoder_hidden_states = ops::add(
            &encoder_hidden_states,
            &ops::multiply(&txt_gate2_exp, &txt_mlp_out)?,
        )?;

        Ok((encoder_hidden_states, hidden_states))
    }
}

/// Timestep Embedder
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTimestepEmbedder {
    #[param]
    pub linear_1: QuantizedLinear,
    #[param]
    pub linear_2: QuantizedLinear,
}

impl QwenTimestepEmbedder {
    pub fn new(timestep_dim: i32, inner_dim: i32, config: &QwenConfig) -> Result<Self, Exception> {
        Ok(Self {
            linear_1: config.quantized_linear(timestep_dim, inner_dim)?,
            linear_2: config.quantized_linear(inner_dim, inner_dim)?,
        })
    }

    pub fn forward(&mut self, t: &Array) -> Result<Array, Exception> {
        let emb = get_timestep_embedding(t, 256)?;
        let emb = self.linear_1.forward(&emb)?;
        let emb = mlx_rs::nn::silu(&emb)?;
        self.linear_2.forward(&emb)
    }
}

/// Time-Text Embed
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTimeTextEmbed {
    #[param]
    pub timestep_embedder: QwenTimestepEmbedder,
}

impl QwenTimeTextEmbed {
    pub fn new(timestep_dim: i32, inner_dim: i32, config: &QwenConfig) -> Result<Self, Exception> {
        Ok(Self {
            timestep_embedder: QwenTimestepEmbedder::new(timestep_dim, inner_dim, config)?,
        })
    }

    pub fn forward(&mut self, timestep: &Array, _hidden_states: &Array) -> Result<Array, Exception> {
        self.timestep_embedder.forward(timestep)
    }
}

/// AdaLayerNorm for output
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenAdaLayerNormOut {
    #[param]
    pub linear: QuantizedLinear,
}

impl QwenAdaLayerNormOut {
    pub fn new(inner_dim: i32, config: &QwenConfig) -> Result<Self, Exception> {
        Ok(Self {
            linear: config.quantized_linear(inner_dim, inner_dim * 2)?,
        })
    }

    pub fn forward(&mut self, x: &Array, temb: &Array) -> Result<Array, Exception> {
        let emb = mlx_rs::nn::silu(temb)?;
        let emb = self.linear.forward(&emb)?;

        // Split into scale and shift (scale is first half, shift is second half - matching mflux)
        let half = emb.dim(-1) / 2;
        let scale = emb.index((.., ..half)).expand_dims(1)?;   // First half = scale
        let shift = emb.index((.., half..)).expand_dims(1)?;   // Second half = shift

        let normed = layer_norm(x, 1e-6)?;
        let one = Array::from_f32(1.0);
        let scale_factor = ops::add(&one, &scale)?;
        ops::add(&ops::multiply(&normed, &scale_factor)?, &shift)
    }
}

/// RMS Norm for text input
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenTransformerRMSNorm {
    #[param]
    pub weight: Param<Array>,
    pub eps: f32,
}

impl QwenTransformerRMSNorm {
    pub fn new(dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::new(Array::ones::<f32>(&[dim])?),
            eps: 1e-6,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let input_dtype = x.dtype();

        // Compute variance in float32 for numerical stability
        let x_f32 = x.as_dtype(mlx_rs::Dtype::Float32)?;
        let variance = ops::mean_axes(&ops::square(&x_f32)?, &[-1], true)?;
        let eps = Array::from_f32(self.eps);

        // Use rsqrt and multiply with ORIGINAL x (not x_f32) to match mflux
        let rsqrt_var = ops::rsqrt(&ops::add(&variance, &eps)?)?;
        let mut hidden_states = ops::multiply(x, &rsqrt_var)?;

        // Handle weight dtype conversion to match mflux
        let weight_dtype = self.weight.dtype();
        if weight_dtype == mlx_rs::Dtype::Bfloat16 || weight_dtype == mlx_rs::Dtype::Float16 {
            hidden_states = hidden_states.as_dtype(weight_dtype)?;
        }
        hidden_states = ops::multiply(&hidden_states, &self.weight)?;

        // Cast back to input dtype if needed
        if hidden_states.dtype() != input_dtype {
            hidden_states = hidden_states.as_dtype(input_dtype)?;
        }

        Ok(hidden_states)
    }
}

/// Main Quantized Transformer
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenQuantizedTransformer {
    pub config: QwenConfig,

    #[param]
    pub img_in: QuantizedLinear,

    #[param]
    pub txt_norm: QwenTransformerRMSNorm,

    #[param]
    pub txt_in: QuantizedLinear,

    #[param]
    pub time_text_embed: QwenTimeTextEmbed,

    #[param]
    pub transformer_blocks: Vec<QwenTransformerBlock>,

    #[param]
    pub norm_out: QwenAdaLayerNormOut,

    #[param]
    pub proj_out: QuantizedLinear,
}

impl QwenQuantizedTransformer {
    pub fn new(config: QwenConfig) -> Result<Self, Exception> {
        let inner_dim = config.inner_dim();
        let output_dim = config.patch_size * config.patch_size * config.out_channels;

        let mut transformer_blocks = Vec::with_capacity(config.num_layers as usize);
        for _ in 0..config.num_layers {
            transformer_blocks.push(QwenTransformerBlock::new(
                inner_dim,
                config.num_attention_heads,
                config.attention_head_dim,
                &config,
            )?);
        }

        Ok(Self {
            img_in: config.quantized_linear(config.in_channels, inner_dim)?,
            txt_norm: QwenTransformerRMSNorm::new(config.joint_attention_dim)?,
            txt_in: config.quantized_linear(config.joint_attention_dim, inner_dim)?,
            time_text_embed: QwenTimeTextEmbed::new(256, inner_dim, &config)?,
            transformer_blocks,
            norm_out: QwenAdaLayerNormOut::new(inner_dim, &config)?,
            proj_out: config.quantized_linear(inner_dim, output_dim)?,
            config,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Array,          // [B, seq, in_channels]
        encoder_hidden_states: &Array,  // [B, txt_seq, joint_attention_dim]
        timestep: &Array,               // [B]
        img_rotary_emb: Option<(&Array, &Array)>,
        txt_rotary_emb: Option<(&Array, &Array)>,
        encoder_hidden_states_mask: Option<&Array>,  // [B, txt_seq] with 1 for real, 0 for padding
    ) -> Result<Array, Exception> {
        // P0: use bfloat16 for residual stream to halve memory bandwidth
        let residual_dtype = Dtype::Bfloat16;

        // Project image patches
        let mut hidden_states = self.img_in.forward(hidden_states)?;
        hidden_states = hidden_states.as_dtype(residual_dtype)?;

        // Normalize and project text
        let encoder_hidden_states = self.txt_norm.forward(encoder_hidden_states)?;
        let mut encoder_hidden_states = self.txt_in.forward(&encoder_hidden_states)?;
        encoder_hidden_states = encoder_hidden_states.as_dtype(residual_dtype)?;

        // Time embedding
        let text_embeddings = self.time_text_embed.forward(timestep, &hidden_states)?;
        let text_embeddings = text_embeddings.as_dtype(residual_dtype)?;

        // Apply transformer blocks
        for block in self.transformer_blocks.iter_mut() {
            let (enc, hid) = block.forward(
                &hidden_states,
                &encoder_hidden_states,
                &text_embeddings,
                img_rotary_emb,
                txt_rotary_emb,
                encoder_hidden_states_mask,
            )?;
            encoder_hidden_states = enc;
            hidden_states = hid;
        }

        // Final norm and projection — upcast to FP32 for numerical stability
        let hidden_states = hidden_states.as_dtype(Dtype::Float32)?;
        let hidden_states = self.norm_out.forward(&hidden_states, &text_embeddings)?;
        let hidden_states = hidden_states.as_dtype(residual_dtype)?;

        let result = self.proj_out.forward(&hidden_states)?;
        result.as_dtype(residual_dtype)
    }

    /// Edit-mode forward: concatenates ref patches, uses dual time embedding,
    /// per-token modulation blending, and slices output to main tokens.
    ///
    /// Supports batch > 1 for CFG batching (cond+uncond as batch=2).
    /// ref_latent_patches: packed ref patches, each [1, ref_seq_i, in_channels]
    /// img_rotary_emb: pre-built edit RoPE for all image tokens (main+refs)
    /// txt_rotary_emb: pre-built text RoPE
    /// encoder_hidden_states_mask: optional [B, txt_seq] with 1=real, 0=pad for batched CFG
    pub fn forward_edit(
        &mut self,
        hidden_states: &Array,
        encoder_hidden_states: &Array,
        timestep: &Array,
        ref_latent_patches: &[&Array],
        img_rotary_emb: (&Array, &Array),
        txt_rotary_emb: (&Array, &Array),
        encoder_hidden_states_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let residual_dtype = Dtype::Bfloat16;
        let batch = hidden_states.dim(0);

        // 1. Project main patches through img_in
        let mut hidden_states = self.img_in.forward(hidden_states)?;
        hidden_states = hidden_states.as_dtype(residual_dtype)?;
        let main_img_seq = hidden_states.dim(1);

        // 2. Project and concat ref patches through same img_in
        for ref_patch in ref_latent_patches {
            let ref_embed = self.img_in.forward(ref_patch)?.as_dtype(residual_dtype)?;
            let ref_embed = if ref_embed.dim(0) != batch {
                ops::broadcast_to(&ref_embed, &[batch, ref_embed.dim(1), ref_embed.dim(2)])?
            } else {
                ref_embed
            };
            hidden_states = ops::concatenate_axis(&[&hidden_states, &ref_embed], 1)?;
        }

        // 3. Normalize and project text
        let encoder_hidden_states = self.txt_norm.forward(encoder_hidden_states)?;
        let mut encoder_hidden_states = self.txt_in.forward(&encoder_hidden_states)?;
        encoder_hidden_states = encoder_hidden_states.as_dtype(residual_dtype)?;

        // 4. Dual time embedding: real + zero
        let real_temb = self.time_text_embed.forward(timestep, &hidden_states)?
            .as_dtype(residual_dtype)?;
        let zero_timestep = Array::zeros::<f32>(timestep.shape())?;
        let zero_temb = self.time_text_embed.forward(&zero_timestep, &hidden_states)?
            .as_dtype(residual_dtype)?;
        let temb = ops::concatenate_axis(&[&real_temb, &zero_temb], 0)?; // [2, inner_dim]

        // 5. Build modulate_index: 0 for main tokens, 1 for ref tokens
        let total_img_seq = hidden_states.dim(1);
        let num_ref_tokens = total_img_seq - main_img_seq;
        let mut mod_idx_vals = vec![0.0f32; main_img_seq as usize];
        mod_idx_vals.extend(vec![1.0f32; num_ref_tokens as usize]);
        let modulate_index = Array::from_slice(&mod_idx_vals, &[total_img_seq]);

        // 6. Apply transformer blocks with edit mode
        for block in &mut self.transformer_blocks {
            let (enc, hid) = block.forward_edit(
                &hidden_states,
                &encoder_hidden_states,
                &temb,
                (&img_rotary_emb.0, &img_rotary_emb.1),
                (&txt_rotary_emb.0, &txt_rotary_emb.1),
                &modulate_index,
                encoder_hidden_states_mask,
            )?;
            encoder_hidden_states = enc;
            hidden_states = hid;
        }

        // 7. Slice to main tokens only (discard ref tokens)
        let hidden_states = hidden_states.index((.., ..main_img_seq, ..));

        // 8. Final norm and projection — upcast to FP32 for numerical stability
        let hidden_states = hidden_states.as_dtype(Dtype::Float32)?;
        let hidden_states = self.norm_out.forward(&hidden_states, &real_temb)?;
        let hidden_states = hidden_states.as_dtype(residual_dtype)?;

        self.proj_out.forward(&hidden_states)
    }
}

// Helper functions

fn split_half(x: &Array) -> Result<(Array, Array), Exception> {
    let half = x.dim(-1) / 2;
    let first = x.index((.., ..half));
    let second = x.index((.., half..));
    Ok((first, second))
}

/// LayerNorm without learnable weights (for pre-modulation normalization).
/// Computes in float32 for numerical stability, returns in input dtype.
fn layer_norm(x: &Array, eps: f32) -> Result<Array, Exception> {
    let input_dtype = x.dtype();
    let x_f32 = x.as_dtype(Dtype::Float32)?;
    let mean = ops::mean_axes(&x_f32, &[-1], true)?;
    let x_centered = ops::subtract(&x_f32, &mean)?;
    let variance = ops::mean_axes(&ops::square(&x_centered)?, &[-1], true)?;
    let rsqrt_var = ops::rsqrt(&ops::add(&variance, &Array::from_f32(eps))?)?;
    let y = ops::multiply(&x_centered, &rsqrt_var)?;
    y.as_dtype(input_dtype)
}

/// Clip values to prevent numerical explosion (like FLUX-klein's ±65504)
fn clip_values(x: &Array) -> Result<Array, Exception> {
    let min_val = Array::from_f32(-65504.0);
    let max_val = Array::from_f32(65504.0);
    let clipped = ops::maximum(x, &min_val)?;
    ops::minimum(&clipped, &max_val)
}

fn modulate(x: &Array, mod_params: &Array) -> Result<(Array, Array), Exception> {
    let dim = mod_params.dim(-1) / 3;
    let shift = mod_params.index((.., ..dim)).expand_dims(1)?;
    let scale = mod_params.index((.., dim..dim * 2)).expand_dims(1)?;
    let gate = mod_params.index((.., dim * 2..));

    let one = Array::from_f32(1.0);
    let scale_factor = ops::add(&one, &scale)?;
    let modulated = ops::add(&ops::multiply(x, &scale_factor)?, &shift)?;

    Ok((modulated, gate))
}

fn apply_rope_qwen(x: &Array, cos: &Array, sin: &Array) -> Result<Array, Exception> {
    // x: [B, seq, heads, head_dim]
    // cos, sin: [seq, head_dim/2]
    let x_f32 = x.as_dtype(mlx_rs::Dtype::Float32)?;

    // Reshape to pairs: [B, seq, heads, head_dim/2, 2]
    let shape = x.shape();
    let new_shape = [shape[0], shape[1], shape[2], shape[3] / 2, 2];
    let x_pairs = x_f32.reshape(&new_shape)?;

    let x_real = x_pairs.index((.., .., .., .., 0));
    let x_imag = x_pairs.index((.., .., .., .., 1));

    // Expand cos/sin: [1, seq, 1, head_dim/2]
    let cos_exp = cos.expand_dims(0)?.expand_dims(2)?;
    let sin_exp = sin.expand_dims(0)?.expand_dims(2)?;

    // Apply rotation
    let out_real = ops::subtract(&ops::multiply(&x_real, &cos_exp)?, &ops::multiply(&x_imag, &sin_exp)?)?;
    let out_imag = ops::add(&ops::multiply(&x_real, &sin_exp)?, &ops::multiply(&x_imag, &cos_exp)?)?;

    // Stack and reshape back
    let out_pairs = ops::stack_axis(&[&out_real, &out_imag], -1)?;
    let out = out_pairs.reshape(shape)?;

    out.as_dtype(x.dtype())
}

// ─── Edit mode helpers ───────────────────────────────────────────────────────

/// Blend image modulation parameters per-token using modulate_index.
/// img_mod_params: [2, 6*dim] (row 0 = real, row 1 = zero)
/// modulate_index: [total_img_seq] (0.0 for main, 1.0 for ref)
/// Returns 6 arrays each [1, total_img_seq, dim]: shift1, scale1, gate1, shift2, scale2, gate2
fn prepare_img_mod_edit(
    img_mod_params: &Array,
    modulate_index: &Array,
    dim: i32,
) -> Result<(Array, Array, Array, Array, Array, Array), Exception> {
    let d = dim;
    // Split into 6 parts of [2, dim]
    let parts: [Array; 6] = [
        img_mod_params.index((.., ..d)),
        img_mod_params.index((.., d..d * 2)),
        img_mod_params.index((.., d * 2..d * 3)),
        img_mod_params.index((.., d * 3..d * 4)),
        img_mod_params.index((.., d * 4..d * 5)),
        img_mod_params.index((.., d * 5..)),
    ];

    // modulate_index: [seq] -> [1, seq, 1]
    let idx = modulate_index.reshape(&[1, -1, 1])?;
    let one = Array::from_f32(1.0);
    let one_minus_idx = ops::subtract(&one, &idx)?;

    let mut blended = Vec::with_capacity(6);
    for p in &parts {
        // p: [2, dim] -> real = p[0:1], zero = p[1:2]
        let real = p.index((0..1, ..)).expand_dims(1)?; // [1, 1, dim]
        let zero = p.index((1..2, ..)).expand_dims(1)?; // [1, 1, dim]
        let b = ops::add(
            &ops::multiply(&one_minus_idx, &real)?,
            &ops::multiply(&idx, &zero)?,
        )?; // [1, seq, dim]
        blended.push(b);
    }

    Ok((
        blended[0].clone(), blended[1].clone(), blended[2].clone(),
        blended[3].clone(), blended[4].clone(), blended[5].clone(),
    ))
}

/// Apply modulation with 3D shift/scale (edit mode: per-token blending)
fn modulate_flex(x: &Array, shift: &Array, scale: &Array) -> Result<Array, Exception> {
    // x: [1, seq, dim], shift/scale: [1, seq, dim] — already expanded
    let one = Array::from_f32(1.0);
    let scale_factor = ops::add(&one, scale)?;
    ops::add(&ops::multiply(x, &scale_factor)?, shift)
}

/// Apply gating with 3D gate (edit mode: per-token)
fn gate_flex(gate: &Array, y: &Array) -> Result<Array, Exception> {
    ops::multiply(gate, y)
}

/// Apply modulation with 2D shift/scale (standard: broadcast over seq)
fn modulate_2d(x: &Array, shift: &Array, scale: &Array) -> Result<Array, Exception> {
    let shift_exp = shift.expand_dims(1)?;
    let scale_exp = scale.expand_dims(1)?;
    let one = Array::from_f32(1.0);
    let scale_factor = ops::add(&one, &scale_exp)?;
    ops::add(&ops::multiply(x, &scale_factor)?, &shift_exp)
}

/// Centered position indices: [-ceil(n/2), ..., -1, 0, 1, ..., floor(n/2)-1]
fn centered_positions_vec(length: i32) -> Vec<f32> {
    let half = length / 2;
    let start = -(length - half);
    let mut positions = Vec::with_capacity(length as usize);
    for i in start..0 {
        positions.push(i as f32);
    }
    for i in 0..half {
        positions.push(i as f32);
    }
    positions
}

/// 1D RoPE frequencies: positions -> (cos, sin), each [seq, dim/2]
fn rope_frequencies_1d(positions: &Array, dim: i32, theta: f32) -> Result<(Array, Array), Exception> {
    let half_dim = dim / 2;
    let omega: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
        .collect();
    let omega = Array::from_slice(&omega, &[half_dim]);

    let pos_exp = positions.expand_dims(-1)?;  // [seq, 1]
    let omega_exp = omega.expand_dims(0)?;     // [1, half_dim]
    let angles = ops::multiply(&pos_exp, &omega_exp)?; // [seq, half_dim]

    Ok((ops::cos(&angles)?, ops::sin(&angles)?))
}

/// Build 3-axis RoPE for image-edit: main image + reference images + text.
/// img_shape: (frame, patch_h, patch_w) in patchified space
/// ref_shapes: per-ref (frame, patch_h, patch_w)
/// Returns ((img_cos, img_sin), (txt_cos, txt_sin))
pub fn build_edit_rope(
    img_shape: (i32, i32, i32),
    ref_shapes: &[(i32, i32, i32)],
    txt_seq_len: i32,
    theta: f32,
    axes_dims: [i32; 3],
) -> Result<((Array, Array), (Array, Array)), Exception> {
    let mut all_shapes = vec![img_shape];
    all_shapes.extend_from_slice(ref_shapes);

    let mut frame_positions: Vec<f32> = Vec::new();
    let mut row_positions: Vec<f32> = Vec::new();
    let mut col_positions: Vec<f32> = Vec::new();
    let mut max_vid_index: i32 = 0;

    for (idx, &(frame, height, width)) in all_shapes.iter().enumerate() {
        let h_centered = centered_positions_vec(height);
        let w_centered = centered_positions_vec(width);

        for f in 0..frame {
            let frame_val = (idx as i32 + f) as f32;
            for h in &h_centered {
                for w in &w_centered {
                    frame_positions.push(frame_val);
                    row_positions.push(*h);
                    col_positions.push(*w);
                }
            }
        }
        max_vid_index = max_vid_index.max(height / 2).max(width / 2);
    }

    let total_seq = frame_positions.len() as i32;
    let frame_pos = Array::from_slice(&frame_positions, &[total_seq]);
    let row_pos = Array::from_slice(&row_positions, &[total_seq]);
    let col_pos = Array::from_slice(&col_positions, &[total_seq]);

    let (f_cos, f_sin) = rope_frequencies_1d(&frame_pos, axes_dims[0], theta)?;
    let (h_cos, h_sin) = rope_frequencies_1d(&row_pos, axes_dims[1], theta)?;
    let (w_cos, w_sin) = rope_frequencies_1d(&col_pos, axes_dims[2], theta)?;

    let img_cos = ops::concatenate_axis(&[&f_cos, &h_cos, &w_cos], -1)?;
    let img_sin = ops::concatenate_axis(&[&f_sin, &h_sin, &w_sin], -1)?;

    // Text positions start at max_vid_index offset
    let txt_positions: Vec<f32> = (max_vid_index..max_vid_index + txt_seq_len)
        .map(|i| i as f32)
        .collect();
    let txt_pos = Array::from_slice(&txt_positions, &[txt_seq_len]);

    let mut txt_cos_parts = Vec::new();
    let mut txt_sin_parts = Vec::new();
    for &dim in &axes_dims {
        let (tc, ts) = rope_frequencies_1d(&txt_pos, dim, theta)?;
        txt_cos_parts.push(tc);
        txt_sin_parts.push(ts);
    }
    let txt_cos_refs: Vec<&Array> = txt_cos_parts.iter().collect();
    let txt_sin_refs: Vec<&Array> = txt_sin_parts.iter().collect();
    let txt_cos = ops::concatenate_axis(&txt_cos_refs, -1)?;
    let txt_sin = ops::concatenate_axis(&txt_sin_refs, -1)?;

    Ok(((img_cos, img_sin), (txt_cos, txt_sin)))
}

fn get_timestep_embedding(t: &Array, dim: i32) -> Result<Array, Exception> {
    // Sinusoidal timestep embeddings (matching diffusers Timesteps)
    // Parameters: flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000, max_period=10000
    let half = dim / 2;
    let freq_seq = Array::from_iter((0..half).map(|i| i as f32), &[half]);
    // exponent = -log(max_period) * arange(0, half) / (half - downscale_freq_shift)
    // With downscale_freq_shift=0: exponent = -log(10000) * i / half
    let log_timescale = (10000.0f32).ln() / half as f32;
    let freqs = ops::exp(&ops::multiply(&freq_seq, &Array::from_f32(-log_timescale))?)?;

    // t: [B] -> [B, 1]
    let t_exp = t.expand_dims(1)?;
    // freqs: [half] -> [1, half]
    let freqs_exp = freqs.expand_dims(0)?;

    // Scale timestep by 1000 (matching diffusers Timesteps scale parameter)
    let t_scaled = ops::multiply(&t_exp, &Array::from_f32(1000.0))?;

    let args = ops::multiply(&t_scaled, &freqs_exp)?;
    let sin_emb = ops::sin(&args)?;
    let cos_emb = ops::cos(&args)?;

    // flip_sin_to_cos=True: output order is [cos, sin] not [sin, cos]
    ops::concatenate_axis(&[&cos_emb, &sin_emb], -1)
}

/// Load weights from HashMap into the model
///
/// Transforms weight keys to match mlx-rs QuantizedLinear structure:
/// - `xxx.weight` -> `xxx.inner.weight` (for quantized weights)
/// - `xxx.bias` -> `xxx.inner.bias` (for quantized linear output bias)
/// (but keeps `xxx.scales` and `xxx.biases` as-is since they match)
pub fn load_transformer_weights(
    model: &mut QwenQuantizedTransformer,
    weights: HashMap<String, Array>,
) -> Result<(), Exception> {
    // First pass: identify which paths have quantized weights (uint32)
    let quantized_paths: std::collections::HashSet<String> = weights.iter()
        .filter_map(|(k, v)| {
            if k.ends_with(".weight") && v.dtype() == mlx_rs::Dtype::Uint32 {
                // Extract the path prefix (everything before .weight)
                Some(k.trim_end_matches(".weight").to_string())
            } else {
                None
            }
        })
        .collect();

    let mut transformed_weights: HashMap<Rc<str>, Array> = HashMap::new();

    for (key, value) in weights {
        let new_key = if key.ends_with(".weight") {
            let path = key.trim_end_matches(".weight");
            if quantized_paths.contains(path) {
                // Quantized weight -> inner.weight
                let k = format!("{}.inner.weight", path);
                if key.contains("timestep") {
                    eprintln!("[WEIGHT LOAD] {} -> {}", key, k);
                }
                k
            } else {
                key
            }
        } else if key.ends_with(".bias") && !key.ends_with(".biases") {
            let path = key.trim_end_matches(".bias");
            if quantized_paths.contains(path) {
                // Quantized linear's output bias -> inner.bias
                let k = format!("{}.inner.bias", path);
                if key.contains("timestep") {
                    eprintln!("[WEIGHT LOAD] {} -> {}", key, k);
                }
                k
            } else {
                key
            }
        } else {
            // scales, biases, or other keys - keep as-is
            if key.contains("timestep") {
                eprintln!("[WEIGHT LOAD] {} (unchanged)", key);
            }
            key
        };

        transformed_weights.insert(Rc::from(new_key.as_str()), value);
    }

    model.update_flattened(transformed_weights);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = QwenConfig::default();
        assert_eq!(config.inner_dim(), 3072);
    }

    #[test]
    fn test_timestep_embedding() {
        let t = Array::from_slice(&[0.5f32], &[1]);
        let emb = get_timestep_embedding(&t, 256).unwrap();
        assert_eq!(emb.shape(), &[1, 256]);
    }
}
