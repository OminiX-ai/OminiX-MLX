//! Quantized Qwen-Image Transformer
//!
//! Matches the weight structure of mlx-community/Qwen-Image-2512-4bit

use std::collections::HashMap;
use std::rc::Rc;

use mlx_macros::ModuleParameters;
use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, ModuleParameters, Param};
use mlx_rs::nn::{RmsNorm, RmsNormBuilder};
use mlx_rs::nn::{QuantizedLinear, QuantizedLinearBuilder};
use mlx_rs::ops::{self, indexing::IndexOp};
use mlx_rs::Array;

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

        // Debug FFN
        static DEBUG_FFN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let debug_ffn = !DEBUG_FFN.swap(true, std::sync::atomic::Ordering::SeqCst);
        if debug_ffn {
            mlx_rs::transforms::eval([x, &hidden]).ok();
            eprintln!("[DEBUG FFN] input: [{:.2}, {:.2}], after mlp_in: [{:.2}, {:.2}]",
                x.min(None).unwrap().item::<f32>(),
                x.max(None).unwrap().item::<f32>(),
                hidden.min(None).unwrap().item::<f32>(),
                hidden.max(None).unwrap().item::<f32>());
        }

        let hidden = mlx_rs::nn::gelu_approximate(&hidden)?;

        if debug_ffn {
            mlx_rs::transforms::eval([&hidden]).ok();
            eprintln!("[DEBUG FFN] after gelu: [{:.2}, {:.2}]",
                hidden.min(None).unwrap().item::<f32>(),
                hidden.max(None).unwrap().item::<f32>());
        }

        let output = self.mlp_out.forward(&hidden)?;

        if debug_ffn {
            mlx_rs::transforms::eval([&output]).ok();
            eprintln!("[DEBUG FFN] after mlp_out: [{:.2}, {:.2}]",
                output.min(None).unwrap().item::<f32>(),
                output.max(None).unwrap().item::<f32>());
        }

        Ok(output)
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

        // Debug Q/K before RMSNorm
        static DEBUG_BEFORE_NORM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if false {
            mlx_rs::transforms::eval([&img_q, &txt_q]).ok();
            let img_q_norm = mlx_rs::ops::sqrt(&mlx_rs::ops::sum_axis(&mlx_rs::ops::multiply(&img_q, &img_q).unwrap(), -1, false).unwrap()).unwrap();
            let txt_q_norm = mlx_rs::ops::sqrt(&mlx_rs::ops::sum_axis(&mlx_rs::ops::multiply(&txt_q, &txt_q).unwrap(), -1, false).unwrap()).unwrap();
            eprintln!("[DEBUG BEFORE NORM] img_q L2 norm: mean={:.4}, txt_q L2 norm: mean={:.4}",
                img_q_norm.mean(None).unwrap().item::<f32>(),
                txt_q_norm.mean(None).unwrap().item::<f32>());

            // Debug norm weights
            let norm_q_weight = &self.norm_q.weight;
            let norm_added_q_weight = &self.norm_added_q.weight;
            mlx_rs::transforms::eval([norm_q_weight.as_ref(), norm_added_q_weight.as_ref()]).ok();
            eprintln!("[DEBUG NORM WEIGHTS] norm_q.weight: shape={:?}, min={:.4}, max={:.4}, mean={:.4}",
                norm_q_weight.shape(),
                norm_q_weight.min(None).unwrap().item::<f32>(),
                norm_q_weight.max(None).unwrap().item::<f32>(),
                norm_q_weight.mean(None).unwrap().item::<f32>());
            eprintln!("[DEBUG NORM WEIGHTS] norm_added_q.weight: shape={:?}, min={:.4}, max={:.4}, mean={:.4}",
                norm_added_q_weight.shape(),
                norm_added_q_weight.min(None).unwrap().item::<f32>(),
                norm_added_q_weight.max(None).unwrap().item::<f32>(),
                norm_added_q_weight.mean(None).unwrap().item::<f32>());
        }

        // Apply RMSNorm
        img_q = self.norm_q.forward(&img_q)?;
        img_k = self.norm_k.forward(&img_k)?;
        txt_q = self.norm_added_q.forward(&txt_q)?;
        txt_k = self.norm_added_k.forward(&txt_k)?;

        // Debug Q/K after norm but before RoPE
        static DEBUG_NORM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if false {
            mlx_rs::transforms::eval([&img_q, &txt_q]).ok();
            let img_q_norm = mlx_rs::ops::sqrt(&mlx_rs::ops::sum_axis(&mlx_rs::ops::multiply(&img_q, &img_q).unwrap(), -1, false).unwrap()).unwrap();
            let txt_q_norm = mlx_rs::ops::sqrt(&mlx_rs::ops::sum_axis(&mlx_rs::ops::multiply(&txt_q, &txt_q).unwrap(), -1, false).unwrap()).unwrap();
            eprintln!("[DEBUG AFTER NORM] img_q norm: mean={:.4}, txt_q norm: mean={:.4}",
                img_q_norm.mean(None).unwrap().item::<f32>(),
                txt_q_norm.mean(None).unwrap().item::<f32>());
        }

        // Apply RoPE if provided
        if let Some((cos, sin)) = img_rotary_emb {
            img_q = apply_rope_qwen(&img_q, cos, sin)?;
            img_k = apply_rope_qwen(&img_k, cos, sin)?;
        }
        if let Some((cos, sin)) = txt_rotary_emb {
            txt_q = apply_rope_qwen(&txt_q, cos, sin)?;
            txt_k = apply_rope_qwen(&txt_k, cos, sin)?;
        }

        // Debug Q/K magnitudes
        static DEBUG_QK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if false {
            mlx_rs::transforms::eval([&img_q, &txt_q, &img_k, &txt_k]).ok();
            // Compute L2 norm per token
            let img_q_norm = mlx_rs::ops::sqrt(&mlx_rs::ops::sum_axis(&mlx_rs::ops::multiply(&img_q, &img_q).unwrap(), -1, false).unwrap()).unwrap();
            let txt_q_norm = mlx_rs::ops::sqrt(&mlx_rs::ops::sum_axis(&mlx_rs::ops::multiply(&txt_q, &txt_q).unwrap(), -1, false).unwrap()).unwrap();
            let img_k_norm = mlx_rs::ops::sqrt(&mlx_rs::ops::sum_axis(&mlx_rs::ops::multiply(&img_k, &img_k).unwrap(), -1, false).unwrap()).unwrap();
            let txt_k_norm = mlx_rs::ops::sqrt(&mlx_rs::ops::sum_axis(&mlx_rs::ops::multiply(&txt_k, &txt_k).unwrap(), -1, false).unwrap()).unwrap();
            eprintln!("[DEBUG Q/K] img_q norm: mean={:.4}, img_k norm: mean={:.4}",
                img_q_norm.mean(None).unwrap().item::<f32>(),
                img_k_norm.mean(None).unwrap().item::<f32>());
            eprintln!("[DEBUG Q/K] txt_q norm: mean={:.4}, txt_k norm: mean={:.4}",
                txt_q_norm.mean(None).unwrap().item::<f32>(),
                txt_k_norm.mean(None).unwrap().item::<f32>());
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

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn_scores = ops::matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?;
        let mut attn_scores = ops::multiply(&attn_scores, &Array::from_f32(scale))?;

        // Apply attention mask if provided
        if let Some(mask) = encoder_hidden_states_mask {
            let img_seq = img_modulated.dim(1);
            let ones_img = Array::ones::<f32>(&[batch, img_seq])?;
            let joint_mask = ops::concatenate_axis(&[mask, &ones_img], 1)?;
            // Convert to additive mask: 0 for real tokens, -1e9 for padding
            let additive_mask = ops::multiply(
                &ops::subtract(&Array::from_f32(1.0), &joint_mask)?,
                &Array::from_f32(-1e9),
            )?;
            let additive_mask = additive_mask.reshape(&[batch, 1, 1, txt_seq + img_seq])?;
            attn_scores = ops::add(&attn_scores, &additive_mask)?;
        }

        let attn = mlx_rs::ops::softmax_axis(&attn_scores, -1, None)?;
        let out = ops::matmul(&attn, &v)?;

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
        encoder_hidden_states_mask: Option<&Array>,  // [B, txt_seq] attention mask
    ) -> Result<(Array, Array), Exception> {
        // Debug hidden_states at start of block
        static DEBUG_BLOCK_INPUT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_BLOCK_INPUT.swap(true, std::sync::atomic::Ordering::SeqCst) {
            mlx_rs::transforms::eval([hidden_states]).ok();
            eprintln!("[DEBUG BLOCK_INPUT] hidden_states: shape={:?}, range=[{:.2}, {:.2}], mean={:.4}",
                hidden_states.shape(),
                hidden_states.min(None).unwrap().item::<f32>(),
                hidden_states.max(None).unwrap().item::<f32>(),
                hidden_states.mean(None).unwrap().item::<f32>());
        }

        // Debug timestep embeddings on first call
        static DEBUG_TEMB: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if false {
            mlx_rs::transforms::eval([text_embeddings]).ok();
            eprintln!("[DEBUG TEMB] shape: {:?}, min={:.4}, max={:.4}, mean={:.4}",
                text_embeddings.shape(),
                text_embeddings.min(None).unwrap().item::<f32>(),
                text_embeddings.max(None).unwrap().item::<f32>(),
                text_embeddings.mean(None).unwrap().item::<f32>());
        }

        // Image modulation
        let img_silu = mlx_rs::nn::silu(text_embeddings)?;
        let img_mod_params = self.img_mod_linear.forward(&img_silu)?;

        // Debug mod_params on first call
        static DEBUG_MOD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let debug_mod = !DEBUG_MOD.swap(true, std::sync::atomic::Ordering::SeqCst);
        if debug_mod {
            mlx_rs::transforms::eval([&img_mod_params]).ok();
            eprintln!("[DEBUG IMG_MOD_PARAMS] range: [{:.2}, {:.2}]",
                img_mod_params.min(None).unwrap().item::<f32>(),
                img_mod_params.max(None).unwrap().item::<f32>());
        }

        let (img_mod1, img_mod2) = split_half(&img_mod_params)?;

        // Text modulation
        let txt_silu = mlx_rs::nn::silu(text_embeddings)?;
        let txt_mod_params = self.txt_mod_linear.forward(&txt_silu)?;
        let (txt_mod1, txt_mod2) = split_half(&txt_mod_params)?;

        // Apply LayerNorm and modulation to image
        let img_normed = layer_norm(hidden_states, 1e-6)?;

        // Debug img_normed
        static DEBUG_IMG_NORMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_IMG_NORMED.swap(true, std::sync::atomic::Ordering::SeqCst) {
            mlx_rs::transforms::eval([&img_normed]).ok();
            eprintln!("[DEBUG IMG_NORMED] range=[{:.2}, {:.2}], mean={:.4}",
                img_normed.min(None).unwrap().item::<f32>(),
                img_normed.max(None).unwrap().item::<f32>(),
                img_normed.mean(None).unwrap().item::<f32>());
        }

        let (img_modulated, img_gate1) = modulate(&img_normed, &img_mod1)?;

        // Debug img_modulated
        static DEBUG_IMG_MODULATED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_IMG_MODULATED.swap(true, std::sync::atomic::Ordering::SeqCst) {
            mlx_rs::transforms::eval([&img_modulated]).ok();
            eprintln!("[DEBUG IMG_MODULATED] range=[{:.2}, {:.2}], mean={:.4}",
                img_modulated.min(None).unwrap().item::<f32>(),
                img_modulated.max(None).unwrap().item::<f32>(),
                img_modulated.mean(None).unwrap().item::<f32>());
        }

        // Apply LayerNorm and modulation to text
        let txt_normed = layer_norm(encoder_hidden_states, 1e-6)?;
        let (txt_modulated, txt_gate1) = modulate(&txt_normed, &txt_mod1)?;

        // Debug gates on first call
        static DEBUG_GATE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if false {
            mlx_rs::transforms::eval([&img_gate1, &txt_gate1]).ok();
            eprintln!("[DEBUG GATE] img_gate1: min={:.4}, max={:.4}, mean={:.4}",
                img_gate1.min(None).unwrap().item::<f32>(),
                img_gate1.max(None).unwrap().item::<f32>(),
                img_gate1.mean(None).unwrap().item::<f32>());
            eprintln!("[DEBUG GATE] txt_gate1: min={:.4}, max={:.4}, mean={:.4}",
                txt_gate1.min(None).unwrap().item::<f32>(),
                txt_gate1.max(None).unwrap().item::<f32>(),
                txt_gate1.mean(None).unwrap().item::<f32>());
        }

        // Joint attention
        let (img_attn_out, txt_attn_out) = self.attn.forward(
            &img_modulated,
            &txt_modulated,
            img_rotary_emb,
            txt_rotary_emb,
            encoder_hidden_states_mask,
        )?;

        // Debug attention output before gating
        static DEBUG_ATTN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let debug_attn = !DEBUG_ATTN.swap(true, std::sync::atomic::Ordering::SeqCst);
        if debug_attn {
            mlx_rs::transforms::eval([&img_attn_out, &txt_attn_out]).ok();
            eprintln!("[DEBUG ATTN_OUT] img: [{:.2}, {:.2}], txt: [{:.2}, {:.2}]",
                img_attn_out.min(None).unwrap().item::<f32>(),
                img_attn_out.max(None).unwrap().item::<f32>(),
                txt_attn_out.min(None).unwrap().item::<f32>(),
                txt_attn_out.max(None).unwrap().item::<f32>());
        }

        // Image: gate + residual (no clipping)
        let img_gate1_exp = img_gate1.expand_dims(1)?;
        let hidden_states = ops::add(hidden_states, &ops::multiply(&img_gate1_exp, &img_attn_out)?)?;

        // Text: gate + residual (no clipping)
        let txt_gate1_exp = txt_gate1.expand_dims(1)?;
        let encoder_hidden_states = ops::add(encoder_hidden_states, &ops::multiply(&txt_gate1_exp, &txt_attn_out)?)?;

        if debug_attn {
            mlx_rs::transforms::eval([&hidden_states, &encoder_hidden_states]).ok();
            eprintln!("[DEBUG AFTER_ATTN] img: [{:.2}, {:.2}], txt: [{:.2}, {:.2}]",
                hidden_states.min(None).unwrap().item::<f32>(),
                hidden_states.max(None).unwrap().item::<f32>(),
                encoder_hidden_states.min(None).unwrap().item::<f32>(),
                encoder_hidden_states.max(None).unwrap().item::<f32>());
        }

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

        if debug_attn {
            mlx_rs::transforms::eval([&hidden_states, &encoder_hidden_states]).ok();
            eprintln!("[DEBUG AFTER_FFN] img: [{:.2}, {:.2}], txt: [{:.2}, {:.2}]",
                hidden_states.min(None).unwrap().item::<f32>(),
                hidden_states.max(None).unwrap().item::<f32>(),
                encoder_hidden_states.min(None).unwrap().item::<f32>(),
                encoder_hidden_states.max(None).unwrap().item::<f32>());
        }

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
        // Debug input t
        static DEBUG_T: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if false {
            mlx_rs::transforms::eval([t]).ok();
            eprintln!("[DEBUG TS] input t value: {:.4}", t.item::<f32>());
        }

        // Get sinusoidal embeddings
        let emb = get_timestep_embedding(t, 256)?;

        // Debug timestep embedding pipeline
        static DEBUG_TS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let debug = !DEBUG_TS.swap(true, std::sync::atomic::Ordering::SeqCst);
        if debug {
            mlx_rs::transforms::eval([&emb]).ok();
            eprintln!("[DEBUG TS] sinusoidal emb: min={:.4}, max={:.4}, mean={:.4}",
                emb.min(None).unwrap().item::<f32>(),
                emb.max(None).unwrap().item::<f32>(),
                emb.mean(None).unwrap().item::<f32>());
        }

        let emb = self.linear_1.forward(&emb)?;
        if debug {
            mlx_rs::transforms::eval([&emb]).ok();
            eprintln!("[DEBUG TS] after linear_1: min={:.4}, max={:.4}, mean={:.4}",
                emb.min(None).unwrap().item::<f32>(),
                emb.max(None).unwrap().item::<f32>(),
                emb.mean(None).unwrap().item::<f32>());
        }

        let emb = mlx_rs::nn::silu(&emb)?;
        if debug {
            mlx_rs::transforms::eval([&emb]).ok();
            eprintln!("[DEBUG TS] after silu: min={:.4}, max={:.4}, mean={:.4}",
                emb.min(None).unwrap().item::<f32>(),
                emb.max(None).unwrap().item::<f32>(),
                emb.mean(None).unwrap().item::<f32>());

            // Debug linear_2 scales/biases (dequantized output)
            if let Ok(test_in) = Array::ones::<f32>(&[1, 3072]) {
                if let Ok(test_out) = self.linear_2.forward(&test_in) {
                    mlx_rs::transforms::eval([&test_out]).ok();
                    eprintln!("[DEBUG TS] linear_2 with ones input: min={:.4}, max={:.4}",
                        test_out.min(None).unwrap().item::<f32>(),
                        test_out.max(None).unwrap().item::<f32>());
                }
            }
        }

        let result = self.linear_2.forward(&emb)?;
        if debug {
            mlx_rs::transforms::eval([&result]).ok();
            eprintln!("[DEBUG TS] after linear_2: min={:.4}, max={:.4}, mean={:.4}",
                result.min(None).unwrap().item::<f32>(),
                result.max(None).unwrap().item::<f32>(),
                result.mean(None).unwrap().item::<f32>());
        }
        Ok(result)
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
        // Project image patches
        let mut hidden_states = self.img_in.forward(hidden_states)?;

        // Debug: check hidden_states after img_in projection
        static DEBUG_IMG_IN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_IMG_IN.swap(true, std::sync::atomic::Ordering::SeqCst) {
            mlx_rs::transforms::eval([&hidden_states]).ok();
            eprintln!("[DEBUG IMG_IN] After img_in: shape={:?}, range=[{:.2}, {:.2}], mean={:.4}",
                hidden_states.shape(),
                hidden_states.min(None).unwrap().item::<f32>(),
                hidden_states.max(None).unwrap().item::<f32>(),
                hidden_states.mean(None).unwrap().item::<f32>());
        }

        // Debug: check text embedding BEFORE normalization (first call only)
        static DEBUG_TXT_RAW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if false {
            mlx_rs::transforms::eval([encoder_hidden_states]).ok();
            eprintln!("[DEBUG TXT_RAW] Input text shape: {:?}, dtype: {:?}",
                encoder_hidden_states.shape(), encoder_hidden_states.dtype());
            eprintln!("[DEBUG TXT_RAW] range: [{:.4}, {:.4}], mean: {:.4}",
                encoder_hidden_states.min(None).unwrap().item::<f32>(),
                encoder_hidden_states.max(None).unwrap().item::<f32>(),
                encoder_hidden_states.mean(None).unwrap().item::<f32>());

            // Debug txt_norm weight
            mlx_rs::transforms::eval([self.txt_norm.weight.as_ref()]).ok();
            eprintln!("[DEBUG TXT_NORM] weight shape: {:?}, dtype: {:?}, range: [{:.4}, {:.4}], mean: {:.4}",
                self.txt_norm.weight.shape(), self.txt_norm.weight.dtype(),
                self.txt_norm.weight.min(None).unwrap().item::<f32>(),
                self.txt_norm.weight.max(None).unwrap().item::<f32>(),
                self.txt_norm.weight.mean(None).unwrap().item::<f32>());
        }

        // Normalize and project text
        let encoder_hidden_states = self.txt_norm.forward(encoder_hidden_states)?;

        // Debug: check text embedding AFTER txt_norm but BEFORE txt_in (first call only)
        static DEBUG_TXT_NORMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if false {
            mlx_rs::transforms::eval([&encoder_hidden_states]).ok();
            eprintln!("[DEBUG TXT_NORMED] After txt_norm shape: {:?}, dtype: {:?}",
                encoder_hidden_states.shape(), encoder_hidden_states.dtype());
            eprintln!("[DEBUG TXT_NORMED] range: [{:.4}, {:.4}], mean: {:.4}",
                encoder_hidden_states.min(None).unwrap().item::<f32>(),
                encoder_hidden_states.max(None).unwrap().item::<f32>(),
                encoder_hidden_states.mean(None).unwrap().item::<f32>());
        }

        let mut encoder_hidden_states = self.txt_in.forward(&encoder_hidden_states)?;

        // Debug: check text embedding after projection (first call only)
        static DEBUG_TXT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if false {
            mlx_rs::transforms::eval([&encoder_hidden_states]).ok();
            eprintln!("[DEBUG TXT_IN] Projected text shape: {:?}", encoder_hidden_states.shape());
            eprintln!("[DEBUG TXT_IN] range: [{:.4}, {:.4}], mean: {:.4}",
                encoder_hidden_states.min(None).unwrap().item::<f32>(),
                encoder_hidden_states.max(None).unwrap().item::<f32>(),
                encoder_hidden_states.mean(None).unwrap().item::<f32>());
        }

        // Time embedding
        let text_embeddings = self.time_text_embed.forward(timestep, &hidden_states)?;

        // Apply transformer blocks
        for (idx, block) in self.transformer_blocks.iter_mut().enumerate() {
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

            // Debug after block 0
            static DEBUG_BLOCK0: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if idx == 0 && !DEBUG_BLOCK0.swap(true, std::sync::atomic::Ordering::SeqCst) {
                mlx_rs::transforms::eval([&hidden_states, &encoder_hidden_states]).ok();
                eprintln!("[DEBUG BLOCK0] After block 0:");
                eprintln!("  hidden_states: [{:.3}, {:.3}], mean={:.4}",
                    hidden_states.min(None).unwrap().item::<f32>(),
                    hidden_states.max(None).unwrap().item::<f32>(),
                    hidden_states.mean(None).unwrap().item::<f32>());
                eprintln!("  encoder_hidden_states: [{:.3}, {:.3}]",
                    encoder_hidden_states.min(None).unwrap().item::<f32>(),
                    encoder_hidden_states.max(None).unwrap().item::<f32>());
            }
        }

        // Debug before norm_out
        static DEBUG_PRE_NORM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_PRE_NORM.swap(true, std::sync::atomic::Ordering::SeqCst) {
            mlx_rs::transforms::eval([&hidden_states]).ok();
            eprintln!("[DEBUG PRE_NORM_OUT] After all 60 blocks:");
            eprintln!("  hidden_states: [{:.3}, {:.3}], mean={:.4}",
                hidden_states.min(None).unwrap().item::<f32>(),
                hidden_states.max(None).unwrap().item::<f32>(),
                hidden_states.mean(None).unwrap().item::<f32>());
        }

        // Final norm and projection
        let hidden_states = self.norm_out.forward(&hidden_states, &text_embeddings)?;

        // Debug after norm_out
        static DEBUG_POST_NORM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_POST_NORM.swap(true, std::sync::atomic::Ordering::SeqCst) {
            mlx_rs::transforms::eval([&hidden_states]).ok();
            eprintln!("[DEBUG POST_NORM_OUT] After norm_out: [{:.3}, {:.3}], mean={:.4}",
                hidden_states.min(None).unwrap().item::<f32>(),
                hidden_states.max(None).unwrap().item::<f32>(),
                hidden_states.mean(None).unwrap().item::<f32>());
        }

        let result = self.proj_out.forward(&hidden_states)?;

        // Debug final output
        static DEBUG_FINAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !DEBUG_FINAL.swap(true, std::sync::atomic::Ordering::SeqCst) {
            mlx_rs::transforms::eval([&result]).ok();
            eprintln!("[DEBUG PROJ_OUT] Final output: [{:.3}, {:.3}], mean={:.4}",
                result.min(None).unwrap().item::<f32>(),
                result.max(None).unwrap().item::<f32>(),
                result.mean(None).unwrap().item::<f32>());
        }

        Ok(result)
    }
}

// Helper functions

fn split_half(x: &Array) -> Result<(Array, Array), Exception> {
    let half = x.dim(-1) / 2;
    let first = x.index((.., ..half));
    let second = x.index((.., half..));
    Ok((first, second))
}

/// RMSNorm without learnable weights (for pre-modulation normalization)
/// Qwen models use RMSNorm, not LayerNorm!
fn layer_norm(x: &Array, eps: f32) -> Result<Array, Exception> {
    // LayerNorm (affine=false): (x - mean) / sqrt(var + eps)
    // The reference implementation uses LayerNorm, NOT RMSNorm!
    let mean = ops::mean_axes(x, &[-1], true)?;
    let x_centered = ops::subtract(x, &mean)?;
    let variance = ops::mean_axes(&ops::square(&x_centered)?, &[-1], true)?;
    let rsqrt_var = ops::rsqrt(&ops::add(&variance, &Array::from_f32(eps))?)?;
    ops::multiply(&x_centered, &rsqrt_var)
}

/// Clip values to prevent numerical explosion (like FLUX-klein's ±65504)
fn clip_values(x: &Array) -> Result<Array, Exception> {
    let min_val = Array::from_f32(-65504.0);
    let max_val = Array::from_f32(65504.0);
    let clipped = ops::maximum(x, &min_val)?;
    ops::minimum(&clipped, &max_val)
}

fn modulate(x: &Array, mod_params: &Array) -> Result<(Array, Array), Exception> {
    // mod_params: [B, dim*3] -> shift, scale, gate each [B, dim]
    let dim = mod_params.dim(-1) / 3;
    let shift = mod_params.index((.., ..dim)).expand_dims(1)?;
    let scale = mod_params.index((.., dim..dim * 2)).expand_dims(1)?;
    let gate = mod_params.index((.., dim * 2..));

    // Debug scale/shift values on first call
    static DEBUG_MODULATE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !DEBUG_MODULATE.swap(true, std::sync::atomic::Ordering::SeqCst) {
        mlx_rs::transforms::eval([&scale, &shift, &gate]).ok();
        eprintln!("[DEBUG MODULATE] scale: [{:.2}, {:.2}], shift: [{:.2}, {:.2}], gate: [{:.2}, {:.2}]",
            scale.min(None).unwrap().item::<f32>(),
            scale.max(None).unwrap().item::<f32>(),
            shift.min(None).unwrap().item::<f32>(),
            shift.max(None).unwrap().item::<f32>(),
            gate.min(None).unwrap().item::<f32>(),
            gate.max(None).unwrap().item::<f32>());
    }

    // (1 + scale) * x + shift
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
