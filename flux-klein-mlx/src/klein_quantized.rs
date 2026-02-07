//! INT8 Quantized FLUX.2-klein-4B Model Implementation
//!
//! This module provides INT8 quantized versions of the FLUX.2-klein model
//! for reduced memory usage (~2x) and potentially faster inference on
//! hardware with efficient INT8 support.
//!
//! # Usage
//!
//! ```rust,ignore
//! // Load f32 model first
//! let mut flux = FluxKlein::new(params)?;
//! flux.update_flattened(weights);
//!
//! // Convert to INT8 quantized
//! let quantized_flux = QuantizedFluxKlein::from_unquantized(flux, 64, 8)?;
//! ```

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    module::{Module, Param},
    nn::{LayerNorm, LayerNormBuilder, Linear, RmsNorm, QuantizedLinear},
    ops,
    ops::indexing::IndexOp,
    Array,
};
use mlx_macros::ModuleParameters;

use crate::klein_model::{
    FluxKlein, FluxKleinParams, KleinDoubleBlock, KleinSingleBlock, SharedModulation,
    apply_rope, compute_rope_freqs,
};
use crate::layers::timestep_embedding;

// ============================================================================
// Quantized Shared Modulation
// ============================================================================

/// Quantized shared modulation layer
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct QuantizedSharedModulation {
    #[param]
    pub linear: QuantizedLinear,
    pub num_params: i32,
}

impl QuantizedSharedModulation {
    /// Create from unquantized SharedModulation
    pub fn from_unquantized(
        mod_layer: SharedModulation,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, Exception> {
        Ok(Self {
            linear: QuantizedLinear::try_from_linear(mod_layer.linear, group_size, bits)?,
            num_params: mod_layer.num_params,
        })
    }

    pub fn forward(&mut self, vec: &Array) -> Result<Vec<Array>, Exception> {
        let x = mlx_rs::nn::silu(vec)?;
        let out = self.linear.forward(&x)?;
        Ok(out.split(self.num_params, -1)?)
    }
}

// ============================================================================
// Quantized Klein Double Stream Block
// ============================================================================

/// Quantized double stream block for FLUX.2-klein
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct QuantizedKleinDoubleBlock {
    pub hidden_size: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    pub mlp_hidden: i32,

    // Pre-modulation normalization (not quantized - LayerNorm is cheap)
    #[param]
    pub img_norm1: LayerNorm,
    #[param]
    pub img_norm2: LayerNorm,
    #[param]
    pub txt_norm1: LayerNorm,
    #[param]
    pub txt_norm2: LayerNorm,

    // Post-residual normalization (not quantized)
    #[param]
    pub txt_post_attn_norm: RmsNorm,
    #[param]
    pub txt_post_mlp_norm: RmsNorm,

    // Image stream attention (quantized)
    #[param]
    pub img_to_q: QuantizedLinear,
    #[param]
    pub img_to_k: QuantizedLinear,
    #[param]
    pub img_to_v: QuantizedLinear,
    #[param]
    pub img_norm_q: RmsNorm,
    #[param]
    pub img_norm_k: RmsNorm,
    #[param]
    pub img_to_out: QuantizedLinear,

    // Text stream attention (quantized)
    #[param]
    pub txt_to_q: QuantizedLinear,
    #[param]
    pub txt_to_k: QuantizedLinear,
    #[param]
    pub txt_to_v: QuantizedLinear,
    #[param]
    pub txt_norm_q: RmsNorm,
    #[param]
    pub txt_norm_k: RmsNorm,
    #[param]
    pub txt_to_out: QuantizedLinear,

    // Image MLP (quantized)
    #[param]
    pub img_mlp_in: QuantizedLinear,
    #[param]
    pub img_mlp_out: QuantizedLinear,

    // Text MLP (quantized)
    #[param]
    pub txt_mlp_in: QuantizedLinear,
    #[param]
    pub txt_mlp_out: QuantizedLinear,
}

impl QuantizedKleinDoubleBlock {
    /// Create from unquantized KleinDoubleBlock
    pub fn from_unquantized(
        block: KleinDoubleBlock,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, Exception> {
        Ok(Self {
            hidden_size: block.hidden_size,
            num_heads: block.num_heads,
            head_dim: block.head_dim,
            mlp_hidden: block.mlp_hidden,

            // Normalization layers (keep as-is)
            img_norm1: block.img_norm1,
            img_norm2: block.img_norm2,
            txt_norm1: block.txt_norm1,
            txt_norm2: block.txt_norm2,
            txt_post_attn_norm: block.txt_post_attn_norm,
            txt_post_mlp_norm: block.txt_post_mlp_norm,

            // Quantize image attention
            img_to_q: QuantizedLinear::try_from_linear(block.img_to_q, group_size, bits)?,
            img_to_k: QuantizedLinear::try_from_linear(block.img_to_k, group_size, bits)?,
            img_to_v: QuantizedLinear::try_from_linear(block.img_to_v, group_size, bits)?,
            img_norm_q: block.img_norm_q,
            img_norm_k: block.img_norm_k,
            img_to_out: QuantizedLinear::try_from_linear(block.img_to_out, group_size, bits)?,

            // Quantize text attention
            txt_to_q: QuantizedLinear::try_from_linear(block.txt_to_q, group_size, bits)?,
            txt_to_k: QuantizedLinear::try_from_linear(block.txt_to_k, group_size, bits)?,
            txt_to_v: QuantizedLinear::try_from_linear(block.txt_to_v, group_size, bits)?,
            txt_norm_q: block.txt_norm_q,
            txt_norm_k: block.txt_norm_k,
            txt_to_out: QuantizedLinear::try_from_linear(block.txt_to_out, group_size, bits)?,

            // Quantize MLPs
            img_mlp_in: QuantizedLinear::try_from_linear(block.img_mlp_in, group_size, bits)?,
            img_mlp_out: QuantizedLinear::try_from_linear(block.img_mlp_out, group_size, bits)?,
            txt_mlp_in: QuantizedLinear::try_from_linear(block.txt_mlp_in, group_size, bits)?,
            txt_mlp_out: QuantizedLinear::try_from_linear(block.txt_mlp_out, group_size, bits)?,
        })
    }

    /// Forward pass (identical logic to unquantized version)
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

        // Unpack modulation
        let (img_shift1, img_scale1, img_gate1) = (&img_mod[0], &img_mod[1], &img_mod[2]);
        let (img_shift2, img_scale2, img_gate2) = (&img_mod[3], &img_mod[4], &img_mod[5]);
        let (txt_shift1, txt_scale1, txt_gate1) = (&txt_mod[0], &txt_mod[1], &txt_mod[2]);
        let (txt_shift2, txt_scale2, txt_gate2) = (&txt_mod[3], &txt_mod[4], &txt_mod[5]);

        // Apply pre-normalization THEN modulation
        let img_normed = self.img_norm1.forward(img)?;
        let txt_normed = self.txt_norm1.forward(txt)?;
        let img_mod1 = modulate(&img_normed, img_shift1, img_scale1)?;
        let txt_mod1 = modulate(&txt_normed, txt_shift1, txt_scale1)?;

        // ========== JOINT ATTENTION ==========
        let img_q = self.img_to_q.forward(&img_mod1)?;
        let img_k = self.img_to_k.forward(&img_mod1)?;
        let img_v = self.img_to_v.forward(&img_mod1)?;

        let txt_q = self.txt_to_q.forward(&txt_mod1)?;
        let txt_k = self.txt_to_k.forward(&txt_mod1)?;
        let txt_v = self.txt_to_v.forward(&txt_mod1)?;

        // Reshape for multi-head attention
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

        // Apply RoPE
        let txt_rope_cos = rope_cos.index((.., ..txt_seq, ..));
        let txt_rope_sin = rope_sin.index((.., ..txt_seq, ..));
        let img_rope_cos = rope_cos.index((.., txt_seq.., ..));
        let img_rope_sin = rope_sin.index((.., txt_seq.., ..));

        let img_q = apply_rope(&img_q, &img_rope_cos, &img_rope_sin)?;
        let img_k = apply_rope(&img_k, &img_rope_cos, &img_rope_sin)?;
        let txt_q = apply_rope(&txt_q, &txt_rope_cos, &txt_rope_sin)?;
        let txt_k = apply_rope(&txt_k, &txt_rope_cos, &txt_rope_sin)?;

        // Concatenate K and V for joint attention
        let combined_k = ops::concatenate_axis(&[&txt_k, &img_k], 1)?;
        let combined_v = ops::concatenate_axis(&[&txt_v, &img_v], 1)?;

        // Transpose for attention
        let img_q = img_q.transpose_axes(&[0, 2, 1, 3])?;
        let txt_q = txt_q.transpose_axes(&[0, 2, 1, 3])?;
        let combined_k = combined_k.transpose_axes(&[0, 2, 1, 3])?;
        let combined_v = combined_v.transpose_axes(&[0, 2, 1, 3])?;

        let scale = (self.head_dim as f32).sqrt();

        // Image attention
        let img_attn = ops::matmul(&img_q, &combined_k.transpose_axes(&[0, 1, 3, 2])?)?;
        let img_attn = ops::divide(&img_attn, &array!(scale))?;
        let img_attn = ops::softmax_axis(&img_attn, -1, None)?;
        let img_attn_out = ops::matmul(&img_attn, &combined_v)?;

        // Text attention
        let txt_attn = ops::matmul(&txt_q, &combined_k.transpose_axes(&[0, 1, 3, 2])?)?;
        let txt_attn = ops::divide(&txt_attn, &array!(scale))?;
        let txt_attn = ops::softmax_axis(&txt_attn, -1, None)?;
        let txt_attn_out = ops::matmul(&txt_attn, &combined_v)?;

        // Transpose back and reshape
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

        // SwiGLU MLP for image
        let img_proj = self.img_mlp_in.forward(&img_mlp_in)?;
        let img_splits = img_proj.split_axis(&[self.mlp_hidden], -1)?;
        let img_swiglu_out = ops::multiply(&mlx_rs::nn::silu(&img_splits[0])?, &img_splits[1])?;
        let img_mlp_out = self.img_mlp_out.forward(&img_swiglu_out)?;

        // SwiGLU MLP for text
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
// Quantized Klein Single Stream Block
// ============================================================================

/// Quantized single stream block for FLUX.2-klein
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct QuantizedKleinSingleBlock {
    pub hidden_size: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    pub mlp_hidden: i32,

    #[param]
    pub norm: LayerNorm,
    #[param]
    pub to_qkv_mlp: QuantizedLinear,
    #[param]
    pub to_out: QuantizedLinear,
    #[param]
    pub norm_q: RmsNorm,
    #[param]
    pub norm_k: RmsNorm,
    #[param]
    pub post_norm: RmsNorm,
}

impl QuantizedKleinSingleBlock {
    /// Create from unquantized KleinSingleBlock
    pub fn from_unquantized(
        block: KleinSingleBlock,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, Exception> {
        Ok(Self {
            hidden_size: block.hidden_size,
            num_heads: block.num_heads,
            head_dim: block.head_dim,
            mlp_hidden: block.mlp_hidden,

            norm: block.norm,
            to_qkv_mlp: QuantizedLinear::try_from_linear(block.to_qkv_mlp, group_size, bits)?,
            to_out: QuantizedLinear::try_from_linear(block.to_out, group_size, bits)?,
            norm_q: block.norm_q,
            norm_k: block.norm_k,
            post_norm: block.post_norm,
        })
    }

    /// Forward pass
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

        let (shift, scale, gate_val) = (&mod_params[0], &mod_params[1], &mod_params[2]);

        // Apply pre-norm THEN modulation
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

        // Reshape Q, K, V
        let q = q.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;

        // Apply QK normalization
        let q = self.norm_q.forward(&q)?;
        let k = self.norm_k.forward(&k)?;

        // Apply RoPE
        let q = apply_rope(&q, rope_cos, rope_sin)?;
        let k = apply_rope(&k, rope_cos, rope_sin)?;

        // Attention
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let scale_factor = (self.head_dim as f32).sqrt();
        let attn = ops::matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?;
        let attn = ops::divide(&attn, &array!(scale_factor))?;
        let attn = ops::softmax_axis(&attn, -1, None)?;
        let attn_out = ops::matmul(&attn, &v)?;

        // Reshape back
        let attn_out = attn_out.transpose_axes(&[0, 2, 1, 3])?;
        let attn_out = attn_out.reshape(&[batch, seq_len, -1])?;

        // SwiGLU MLP
        let mlp_out = ops::multiply(&mlx_rs::nn::silu(mlp_gate)?, mlp_up)?;

        // Concatenate and project
        let combined = ops::concatenate_axis(&[attn_out, mlp_out], -1)?;
        let out = self.to_out.forward(&combined)?;

        // Apply gate and residual
        ops::add(x, &gate(&out, gate_val)?)
    }
}

// ============================================================================
// Quantized FLUX.2-klein Model
// ============================================================================

/// INT8 Quantized FLUX.2-klein transformer
///
/// This is the quantized version of FluxKlein with all Linear layers
/// converted to QuantizedLinear for reduced memory usage.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct QuantizedFluxKlein {
    pub params: FluxKleinParams,

    // Input embeddings (quantized)
    #[param]
    pub x_embedder: QuantizedLinear,
    #[param]
    pub context_embedder: QuantizedLinear,

    #[param]
    pub txt_norm: RmsNorm,

    // Time embedding (quantized)
    #[param]
    pub time_embed_1: QuantizedLinear,
    #[param]
    pub time_embed_2: QuantizedLinear,

    // Shared modulation (quantized)
    #[param]
    pub double_mod_img: QuantizedSharedModulation,
    #[param]
    pub double_mod_txt: QuantizedSharedModulation,
    #[param]
    pub single_mod: QuantizedSharedModulation,

    // Transformer blocks (quantized)
    #[param]
    pub double_blocks: Vec<QuantizedKleinDoubleBlock>,
    #[param]
    pub single_blocks: Vec<QuantizedKleinSingleBlock>,

    // Final layer (quantized)
    #[param]
    pub final_norm: RmsNorm,
    #[param]
    pub norm_out: QuantizedLinear,
    #[param]
    pub proj_out: QuantizedLinear,
}

impl QuantizedFluxKlein {
    /// Convert an unquantized FluxKlein to quantized version
    ///
    /// # Arguments
    /// * `model` - The unquantized FluxKlein model with loaded weights
    /// * `group_size` - Quantization group size (default: 64)
    /// * `bits` - Quantization bits (8 for INT8)
    ///
    /// # Example
    /// ```rust,ignore
    /// let quantized = QuantizedFluxKlein::from_unquantized(flux, 64, 8)?;
    /// ```
    pub fn from_unquantized(
        model: FluxKlein,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, Exception> {
        println!("  Quantizing transformer to INT{} (group_size={})...", bits, group_size);

        // Quantize double blocks
        let double_blocks: Result<Vec<_>, _> = model
            .double_blocks
            .into_iter()
            .enumerate()
            .map(|(i, block)| {
                if i == 0 {
                    println!("    Quantizing {} double blocks...", model.params.depth);
                }
                QuantizedKleinDoubleBlock::from_unquantized(block, group_size, bits)
            })
            .collect();
        let double_blocks = double_blocks?;

        // Quantize single blocks
        let single_blocks: Result<Vec<_>, _> = model
            .single_blocks
            .into_iter()
            .enumerate()
            .map(|(i, block)| {
                if i == 0 {
                    println!("    Quantizing {} single blocks...", model.params.depth_single);
                }
                QuantizedKleinSingleBlock::from_unquantized(block, group_size, bits)
            })
            .collect();
        let single_blocks = single_blocks?;

        println!("    Quantizing embeddings and output layers...");

        Ok(Self {
            params: model.params,

            x_embedder: QuantizedLinear::try_from_linear(model.x_embedder, group_size, bits)?,
            context_embedder: QuantizedLinear::try_from_linear(model.context_embedder, group_size, bits)?,
            txt_norm: model.txt_norm,

            time_embed_1: QuantizedLinear::try_from_linear(model.time_embed_1, group_size, bits)?,
            time_embed_2: QuantizedLinear::try_from_linear(model.time_embed_2, group_size, bits)?,

            double_mod_img: QuantizedSharedModulation::from_unquantized(model.double_mod_img, group_size, bits)?,
            double_mod_txt: QuantizedSharedModulation::from_unquantized(model.double_mod_txt, group_size, bits)?,
            single_mod: QuantizedSharedModulation::from_unquantized(model.single_mod, group_size, bits)?,

            double_blocks,
            single_blocks,

            final_norm: model.final_norm,
            norm_out: QuantizedLinear::try_from_linear(model.norm_out, group_size, bits)?,
            proj_out: QuantizedLinear::try_from_linear(model.proj_out, group_size, bits)?,
        })
    }

    /// Compute RoPE frequencies for caching (same as unquantized)
    pub fn compute_rope(
        txt_ids: &Array,
        img_ids: &Array,
    ) -> Result<(Array, Array), Exception> {
        let all_ids = ops::concatenate_axis(&[txt_ids, img_ids], 1)?;
        compute_rope_freqs(&all_ids, &[32, 32, 32, 32], 2000.0)
    }

    /// Forward pass with pre-computed RoPE
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
            let (new_img, new_txt) = block.forward(&img, &txt, &img_mod, &txt_mod, rope_cos, rope_sin)?;
            img = new_img;
            txt = new_txt;
        }

        // Concatenate streams for single blocks
        let mut x = ops::concatenate_axis(&[txt, img], 1)?;

        // Single stream blocks
        for block in self.single_blocks.iter_mut() {
            x = block.forward(&x, &single_mod, rope_cos, rope_sin)?;
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

    /// Forward pass (convenience wrapper)
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
}

// ============================================================================
// Direct Loading from Pre-Quantized Weights
// ============================================================================

/// Create a QuantizedLinear from pre-quantized weight/scales/biases arrays
///
/// Expects the weights HashMap to contain:
/// - `{prefix}.weight` - U32 packed quantized weights
/// - `{prefix}.scales` - quantization scales
/// - `{prefix}.biases` - quantization biases
pub fn create_quantized_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<QuantizedLinear, Exception> {
    let weight_key = format!("{}.weight", prefix);
    let scales_key = format!("{}.scales", prefix);
    let biases_key = format!("{}.biases", prefix);

    let weight = weights.get(&weight_key)
        .unwrap_or_else(|| panic!("Missing quantized weight: {}", weight_key))
        .clone();
    let scales = weights.get(&scales_key)
        .unwrap_or_else(|| panic!("Missing quantized scales: {}", scales_key))
        .clone();
    let biases = weights.get(&biases_key)
        .unwrap_or_else(|| panic!("Missing quantized biases: {}", biases_key))
        .clone();

    let inner = Linear {
        weight: Param::new(weight),
        bias: Param::new(None),
    };

    Ok(QuantizedLinear {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    })
}

/// Load an RmsNorm from weights
fn load_rms_norm(
    weights: &HashMap<String, Array>,
    key: &str,
    dim: i32,
) -> Result<RmsNorm, Exception> {
    let weight = weights.get(key)
        .unwrap_or_else(|| panic!("Missing norm weight: {}", key))
        .clone();
    let mut norm = RmsNorm::new(dim)?;
    norm.weight = Param::new(weight);
    Ok(norm)
}

/// Load a complete quantized FLUX.2-klein transformer from pre-quantized weights
///
/// The weights should already be sanitized (using `sanitize_klein_quantized_weights()`
/// or `sanitize_klein_model_weights()` for the base keys).
///
/// # Arguments
/// * `weights` - HashMap of sanitized weight name -> Array
/// * `group_size` - Quantization group size (typically 64)
/// * `bits` - Quantization bits (typically 8)
pub fn load_quantized_flux_klein(
    weights: HashMap<String, Array>,
    group_size: i32,
    bits: i32,
) -> Result<QuantizedFluxKlein, Exception> {
    let params = FluxKleinParams::default();
    let hidden = params.hidden_size;
    let head_dim = params.head_dim;

    // Input embeddings
    let x_embedder = create_quantized_linear(&weights, "x_embedder", group_size, bits)?;
    let context_embedder = create_quantized_linear(&weights, "context_embedder", group_size, bits)?;

    // txt_norm (RmsNorm, default-initialized — not present in diffusers weights)
    let txt_norm = RmsNorm::new(hidden)?;

    // Time embedding
    let time_embed_1 = create_quantized_linear(&weights, "time_embed_1", group_size, bits)?;
    let time_embed_2 = create_quantized_linear(&weights, "time_embed_2", group_size, bits)?;

    // Shared modulation layers
    let double_mod_img = QuantizedSharedModulation {
        linear: create_quantized_linear(&weights, "double_mod_img.linear", group_size, bits)?,
        num_params: 6,
    };
    let double_mod_txt = QuantizedSharedModulation {
        linear: create_quantized_linear(&weights, "double_mod_txt.linear", group_size, bits)?,
        num_params: 6,
    };
    let single_mod = QuantizedSharedModulation {
        linear: create_quantized_linear(&weights, "single_mod.linear", group_size, bits)?,
        num_params: 3,
    };

    // Double blocks
    let mut double_blocks = Vec::with_capacity(params.depth as usize);
    for i in 0..params.depth as usize {
        let prefix = format!("double_blocks.{}", i);
        let block = QuantizedKleinDoubleBlock {
            hidden_size: hidden,
            num_heads: params.num_heads,
            head_dim,
            mlp_hidden: params.mlp_hidden,

            // LayerNorm (not quantized, no learnable affine)
            img_norm1: LayerNormBuilder::new(hidden).affine(false).eps(1e-6).build()?,
            img_norm2: LayerNormBuilder::new(hidden).affine(false).eps(1e-6).build()?,
            txt_norm1: LayerNormBuilder::new(hidden).affine(false).eps(1e-6).build()?,
            txt_norm2: LayerNormBuilder::new(hidden).affine(false).eps(1e-6).build()?,

            // Post-residual norms (default-initialized — not present in diffusers weights)
            txt_post_attn_norm: RmsNorm::new(hidden)?,
            txt_post_mlp_norm: RmsNorm::new(hidden)?,

            // Image attention
            img_to_q: create_quantized_linear(&weights, &format!("{}.img_to_q", prefix), group_size, bits)?,
            img_to_k: create_quantized_linear(&weights, &format!("{}.img_to_k", prefix), group_size, bits)?,
            img_to_v: create_quantized_linear(&weights, &format!("{}.img_to_v", prefix), group_size, bits)?,
            img_norm_q: load_rms_norm(&weights, &format!("{}.img_norm_q.weight", prefix), head_dim)?,
            img_norm_k: load_rms_norm(&weights, &format!("{}.img_norm_k.weight", prefix), head_dim)?,
            img_to_out: create_quantized_linear(&weights, &format!("{}.img_to_out", prefix), group_size, bits)?,

            // Text attention
            txt_to_q: create_quantized_linear(&weights, &format!("{}.txt_to_q", prefix), group_size, bits)?,
            txt_to_k: create_quantized_linear(&weights, &format!("{}.txt_to_k", prefix), group_size, bits)?,
            txt_to_v: create_quantized_linear(&weights, &format!("{}.txt_to_v", prefix), group_size, bits)?,
            txt_norm_q: load_rms_norm(&weights, &format!("{}.txt_norm_q.weight", prefix), head_dim)?,
            txt_norm_k: load_rms_norm(&weights, &format!("{}.txt_norm_k.weight", prefix), head_dim)?,
            txt_to_out: create_quantized_linear(&weights, &format!("{}.txt_to_out", prefix), group_size, bits)?,

            // MLPs
            img_mlp_in: create_quantized_linear(&weights, &format!("{}.img_mlp_in", prefix), group_size, bits)?,
            img_mlp_out: create_quantized_linear(&weights, &format!("{}.img_mlp_out", prefix), group_size, bits)?,
            txt_mlp_in: create_quantized_linear(&weights, &format!("{}.txt_mlp_in", prefix), group_size, bits)?,
            txt_mlp_out: create_quantized_linear(&weights, &format!("{}.txt_mlp_out", prefix), group_size, bits)?,
        };
        double_blocks.push(block);
    }

    // Single blocks
    let mut single_blocks = Vec::with_capacity(params.depth_single as usize);
    for i in 0..params.depth_single as usize {
        let prefix = format!("single_blocks.{}", i);
        let block = QuantizedKleinSingleBlock {
            hidden_size: hidden,
            num_heads: params.num_heads,
            head_dim,
            mlp_hidden: params.mlp_hidden,

            norm: LayerNormBuilder::new(hidden).affine(false).eps(1e-6).build()?,
            to_qkv_mlp: create_quantized_linear(&weights, &format!("{}.to_qkv_mlp", prefix), group_size, bits)?,
            to_out: create_quantized_linear(&weights, &format!("{}.to_out", prefix), group_size, bits)?,
            norm_q: load_rms_norm(&weights, &format!("{}.norm_q.weight", prefix), head_dim)?,
            norm_k: load_rms_norm(&weights, &format!("{}.norm_k.weight", prefix), head_dim)?,
            post_norm: RmsNorm::new(hidden)?,  // default-initialized — not present in diffusers weights
        };
        single_blocks.push(block);
    }

    // Final layer
    let final_norm = RmsNorm::new(hidden)?;  // default-initialized — not present in diffusers weights
    let norm_out = create_quantized_linear(&weights, "norm_out", group_size, bits)?;
    let proj_out = create_quantized_linear(&weights, "proj_out", group_size, bits)?;

    Ok(QuantizedFluxKlein {
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

/// Quantize bf16/f32 FLUX.2-klein weights and save as pre-quantized safetensors
///
/// This is a one-time operation. Run on a machine with enough RAM to hold the
/// full bf16 model (~30GB peak). The output file will be ~4-5GB.
///
/// # Arguments
/// * `input_path` - Path to the bf16 transformer safetensors file
/// * `output_path` - Path to save the quantized safetensors file (must end in .safetensors)
/// * `group_size` - Quantization group size (default: 64)
/// * `bits` - Quantization bits (default: 8)
pub fn quantize_and_save_flux_klein(
    input_path: &Path,
    output_path: &Path,
    group_size: i32,
    bits: i32,
) -> Result<(), Exception> {
    println!("Loading bf16 weights from {:?}...", input_path);
    let raw_weights = crate::weights::load_safetensors(input_path)
        .map_err(|e| Exception::custom(format!("Failed to load weights: {}", e)))?;

    println!("Sanitizing {} weight keys...", raw_weights.len());
    let weights = crate::weights::sanitize_klein_model_weights(raw_weights);

    println!("Quantizing {} weights to {}bit (group_size={})...", weights.len(), bits, group_size);
    let mut quantized_tensors: HashMap<String, Array> = HashMap::new();

    // Determine if a key is a norm weight (should NOT be quantized)
    let is_norm_weight = |key: &str| -> bool {
        if !key.ends_with(".weight") {
            return false;
        }
        // Block norms: img_norm_q, img_norm_k, txt_norm_q, txt_norm_k, norm_q, norm_k
        // These are small RmsNorm weights that exist in the diffusers model
        key.contains("_norm_q.") || key.contains("_norm_k.") ||
        key.contains(".norm_q.") || key.contains(".norm_k.")
    };

    for (key, value) in &weights {
        if is_norm_weight(key) {
            // Keep norm weights as f32
            let v32 = value.as_type::<f32>().unwrap_or_else(|_| value.clone());
            quantized_tensors.insert(key.clone(), v32);
        } else if key.ends_with(".weight") {
            // Quantize linear weights
            let v32 = value.as_type::<f32>().unwrap_or_else(|_| value.clone());
            let (q_weight, scales, biases) = ops::quantize(&v32, group_size, bits, None::<&str>)?;
            q_weight.eval()?;
            scales.eval()?;
            biases.eval()?;

            let base = &key[..key.len() - 7]; // strip ".weight"
            quantized_tensors.insert(key.clone(), q_weight);
            quantized_tensors.insert(format!("{}.scales", base), scales);
            quantized_tensors.insert(format!("{}.biases", base), biases);
        }
    }

    // Save using MLX's native safetensors saving
    println!("Saving {} tensors to {:?}...", quantized_tensors.len(), output_path);

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| Exception::custom(format!("Failed to create directory: {}", e)))?;
    }

    Array::save_safetensors(quantized_tensors, None::<&HashMap<String, String>>, output_path)
        .map_err(|e| Exception::custom(format!("Failed to save safetensors: {}", e)))?;

    println!("Quantized weights saved to {:?}", output_path);

    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Apply modulation: (1 + scale) * x + shift
fn modulate(x: &Array, shift: &Array, scale: &Array) -> Result<Array, Exception> {
    let shift = shift.reshape(&[shift.dim(0), 1, shift.dim(-1)])?;
    let scale = scale.reshape(&[scale.dim(0), 1, scale.dim(-1)])?;

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
