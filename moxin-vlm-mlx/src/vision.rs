//! DINOv2 and SigLIP Vision Transformer encoders.
//!
//! Implements the dual-backbone vision encoder from Prismatic VLMs:
//! - DINOv2 ViT-Large/14 (24 layers, 1024-dim, with registers + LayerScale)
//! - SigLIP ViT-SO400M/14 (27 layers, 1152-dim)

use std::collections::HashMap;

use mlx_rs::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::indexing::IndexOp,
    quantization::{MaybeQuantized, Quantizable},
    Array,
};

use crate::error::Error;

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone)]
pub struct ViTConfig {
    pub embed_dim: i32,
    pub depth: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    pub intermediate_dim: i32,
    pub has_cls_token: bool,
    pub num_registers: i32,
    pub has_layer_scale: bool,
    pub patch_size: i32,
    pub image_size: i32,
}

impl ViTConfig {
    /// DINOv2 ViT-Large/14 with 4 register tokens
    pub fn dinov2_large() -> Self {
        Self {
            embed_dim: 1024,
            depth: 24,
            num_heads: 16,
            head_dim: 64,
            intermediate_dim: 4096,
            has_cls_token: true,
            num_registers: 4,
            has_layer_scale: true,
            patch_size: 14,
            image_size: 224,
        }
    }

    /// SigLIP ViT-SO400M/14
    pub fn siglip_so400m() -> Self {
        Self {
            embed_dim: 1152,
            depth: 27,
            num_heads: 16,
            head_dim: 72,
            intermediate_dim: 4304,
            has_cls_token: false,
            num_registers: 0,
            has_layer_scale: false,
            patch_size: 14,
            image_size: 224,
        }
    }

    pub fn num_patches(&self) -> i32 {
        (self.image_size / self.patch_size) * (self.image_size / self.patch_size)
    }
}

// ============================================================================
// ViT Attention (bidirectional, no RoPE, no KV cache)
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct ViTAttention {
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub out_proj: MaybeQuantized<nn::Linear>,
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
}

impl Module<&Array> for ViTAttention {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let b = shape[0];
        let n = shape[1];

        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[b, n, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[b, n, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[b, n, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Bidirectional SDPA (no causal mask)
        let output = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, None)?;

        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, n, -1])?;

        self.out_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.out_proj.training_mode(mode);
    }
}

// ============================================================================
// ViT MLP (GELU activation)
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct ViTMlp {
    #[param]
    pub fc1: MaybeQuantized<nn::Linear>,
    #[param]
    pub fc2: MaybeQuantized<nn::Linear>,
}

impl Module<&Array> for ViTMlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let h = self.fc1.forward(x)?;
        let h = nn::gelu(h)?;
        self.fc2.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.fc1.training_mode(mode);
        self.fc2.training_mode(mode);
    }
}

// ============================================================================
// LayerScale (element-wise scaling, used by DINOv2)
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct LayerScale {
    #[param]
    pub gamma: Param<Array>,
}

impl Module<&Array> for LayerScale {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        x.multiply(self.gamma.as_ref())
    }

    fn training_mode(&mut self, _: bool) {}
}

// ============================================================================
// ViT Transformer Block
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct ViTBlock {
    #[param]
    pub norm1: nn::LayerNorm,
    #[param]
    pub attn: ViTAttention,
    #[param]
    pub ls1: Option<LayerScale>,
    #[param]
    pub norm2: nn::LayerNorm,
    #[param]
    pub mlp: ViTMlp,
    #[param]
    pub ls2: Option<LayerScale>,
}

impl Module<&Array> for ViTBlock {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let normed = self.norm1.forward(x)?;
        let mut attn_out = self.attn.forward(&normed)?;
        if let Some(ls) = &mut self.ls1 {
            attn_out = ls.forward(&attn_out)?;
        }
        let h = x.add(&attn_out)?;

        let normed = self.norm2.forward(&h)?;
        let mut mlp_out = self.mlp.forward(&normed)?;
        if let Some(ls) = &mut self.ls2 {
            mlp_out = ls.forward(&mlp_out)?;
        }
        h.add(&mlp_out)
    }

    fn training_mode(&mut self, mode: bool) {
        self.norm1.training_mode(mode);
        self.attn.training_mode(mode);
        self.norm2.training_mode(mode);
        self.mlp.training_mode(mode);
    }
}

// ============================================================================
// ViT Encoder
// ============================================================================

#[derive(Debug)]
pub struct ViTEncoder {
    pub config: ViTConfig,
    pub patch_embed: nn::Conv2d,
    pub cls_token: Option<Param<Array>>,
    pub reg_tokens: Option<Param<Array>>,
    pub pos_embed: Param<Array>,
    pub blocks: Vec<ViTBlock>,
    pub norm: nn::LayerNorm,
}

impl ViTEncoder {
    /// Forward pass: image [B, H, W, 3] (NHWC) -> patch features [B, num_patches, embed_dim]
    ///
    /// Extracts features from the second-to-last transformer block.
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let b = x.shape()[0];
        let dim = self.config.embed_dim;
        let num_patches = self.config.num_patches();

        // Patch embedding: [B, H, W, 3] -> [B, H/P, W/P, dim] -> [B, num_patches, dim]
        let mut h = self.patch_embed.forward(x)?;
        h = h.reshape(&[b, num_patches, dim])?;

        // Position embeddings may or may not include CLS position.
        // If pos_embed has num_patches tokens, add before CLS; if num_patches+1, add after CLS.
        let pos_len = self.pos_embed.as_ref().shape()[1];
        let add_pos_before_cls = pos_len == num_patches;

        if add_pos_before_cls {
            h = h.add(self.pos_embed.as_ref())?;
        }

        // Prepend CLS token
        if let Some(cls) = &self.cls_token {
            let cls_expanded = mlx_rs::ops::broadcast_to(cls.as_ref(), &[b, 1, dim])?;
            h = mlx_rs::ops::concatenate_axis(&[&cls_expanded, &h], 1)?;
        }

        // Add position embeddings (if they include CLS position)
        if !add_pos_before_cls {
            h = h.add(self.pos_embed.as_ref())?;
        }

        // Insert register tokens after CLS (before patches)
        if let Some(reg) = &self.reg_tokens {
            let n_reg = self.config.num_registers;
            let reg_expanded = mlx_rs::ops::broadcast_to(reg.as_ref(), &[b, n_reg, dim])?;
            let cls_part = h.index((.., ..1, ..));
            let patch_part = h.index((.., 1.., ..));
            h = mlx_rs::ops::concatenate_axis(&[&cls_part, &reg_expanded, &patch_part], 1)?;
        }

        // Run blocks up to second-to-last (feature extraction layer)
        let extraction_idx = (self.config.depth - 2) as usize;
        for block in self.blocks.iter_mut().take(extraction_idx + 1) {
            h = block.forward(&h)?;
        }

        // Strip CLS + register tokens, return only patch features
        if self.config.has_cls_token {
            let skip = 1 + self.config.num_registers;
            Ok(h.index((.., skip.., ..)))
        } else {
            Ok(h)
        }
    }

    /// Quantize all linear layers in this encoder.
    pub fn quantize(mut self, group_size: i32, bits: i32) -> std::result::Result<Self, Exception> {
        self.blocks = self
            .blocks
            .into_iter()
            .map(|block| quantize_vit_block(block, group_size, bits))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(self)
    }
}

fn quantize_mq(
    mq: MaybeQuantized<nn::Linear>,
    group_size: i32,
    bits: i32,
) -> std::result::Result<MaybeQuantized<nn::Linear>, Exception> {
    mq.try_into_quantized(group_size, bits)
}

fn quantize_vit_block(
    block: ViTBlock,
    group_size: i32,
    bits: i32,
) -> std::result::Result<ViTBlock, Exception> {
    Ok(ViTBlock {
        attn: ViTAttention {
            q_proj: quantize_mq(block.attn.q_proj, group_size, bits)?,
            k_proj: quantize_mq(block.attn.k_proj, group_size, bits)?,
            v_proj: quantize_mq(block.attn.v_proj, group_size, bits)?,
            out_proj: quantize_mq(block.attn.out_proj, group_size, bits)?,
            num_heads: block.attn.num_heads,
            head_dim: block.attn.head_dim,
            scale: block.attn.scale,
        },
        mlp: ViTMlp {
            fc1: quantize_mq(block.mlp.fc1, group_size, bits)?,
            fc2: quantize_mq(block.mlp.fc2, group_size, bits)?,
        },
        norm1: block.norm1,
        ls1: block.ls1,
        norm2: block.norm2,
        ls2: block.ls2,
    })
}

// ============================================================================
// Weight loading
// ============================================================================

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

fn get_weight_opt(weights: &HashMap<String, Array>, key: &str) -> Option<Array> {
    weights.get(key).cloned()
}

fn make_linear(weight: Array, bias: Option<Array>) -> MaybeQuantized<nn::Linear> {
    MaybeQuantized::new(nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(bias),
    })
}

/// Load a ViT encoder from weight map.
pub fn load_vit_encoder(
    weights: &HashMap<String, Array>,
    prefix: &str,
    config: ViTConfig,
) -> Result<ViTEncoder, Error> {
    // Patch embedding Conv2d: transpose PyTorch [O,I,H,W] -> MLX [O,H,W,I]
    let conv_w = get_weight(weights, &format!("{}.patch_embed.proj.weight", prefix))?;
    let conv_w = conv_w.transpose_axes(&[0, 2, 3, 1])?;
    let conv_b = get_weight_opt(weights, &format!("{}.patch_embed.proj.bias", prefix));

    let patch_embed = nn::Conv2d {
        weight: Param::new(conv_w),
        bias: Param::new(conv_b),
        stride: (config.patch_size, config.patch_size),
        padding: (0, 0),
        dilation: (1, 1),
        groups: 1,
    };

    // CLS token
    let cls_token = if config.has_cls_token {
        Some(Param::new(get_weight(
            weights,
            &format!("{}.cls_token", prefix),
        )?))
    } else {
        None
    };

    // Register tokens (DINOv2) - TIMM uses "reg_token" (singular)
    let reg_tokens = if config.num_registers > 0 {
        let key = format!("{}.reg_token", prefix);
        let alt_key = format!("{}.register_tokens", prefix);
        let token = get_weight_opt(weights, &key).or_else(|| get_weight_opt(weights, &alt_key));
        token.map(Param::new)
    } else {
        None
    };

    // Position embeddings
    let pos_embed = Param::new(get_weight(
        weights,
        &format!("{}.pos_embed", prefix),
    )?);

    // Transformer blocks
    let mut blocks = Vec::with_capacity(config.depth as usize);
    for i in 0..config.depth {
        let bp = format!("{}.blocks.{}", prefix, i);
        blocks.push(load_vit_block(weights, &bp, &config)?);
    }

    // Final LayerNorm
    let norm = nn::LayerNorm {
        dimensions: config.embed_dim,
        eps: 1e-6,
        weight: Param::new(Some(get_weight(
            weights,
            &format!("{}.norm.weight", prefix),
        )?)),
        bias: Param::new(get_weight_opt(
            weights,
            &format!("{}.norm.bias", prefix),
        )),
    };

    Ok(ViTEncoder {
        config,
        patch_embed,
        cls_token,
        reg_tokens,
        pos_embed,
        blocks,
        norm,
    })
}

fn load_vit_block(
    weights: &HashMap<String, Array>,
    prefix: &str,
    config: &ViTConfig,
) -> Result<ViTBlock, Error> {
    let norm1 = nn::LayerNorm {
        dimensions: config.embed_dim,
        eps: 1e-6,
        weight: Param::new(Some(get_weight(weights, &format!("{}.norm1.weight", prefix))?)),
        bias: Param::new(get_weight_opt(weights, &format!("{}.norm1.bias", prefix))),
    };
    let norm2 = nn::LayerNorm {
        dimensions: config.embed_dim,
        eps: 1e-6,
        weight: Param::new(Some(get_weight(weights, &format!("{}.norm2.weight", prefix))?)),
        bias: Param::new(get_weight_opt(weights, &format!("{}.norm2.bias", prefix))),
    };

    let attn = load_vit_attention(weights, &format!("{}.attn", prefix), config)?;

    let ls1 = if config.has_layer_scale {
        // Try both key names: "gamma" (TIMM) and "scale_factor" (HF-converted)
        let gamma = get_weight_opt(weights, &format!("{}.ls1.gamma", prefix))
            .or_else(|| get_weight_opt(weights, &format!("{}.ls1.scale_factor", prefix)))
            .ok_or_else(|| crate::error::Error::WeightNotFound(format!("{}.ls1.gamma/scale_factor", prefix)))?;
        Some(LayerScale {
            gamma: Param::new(gamma),
        })
    } else {
        None
    };
    let ls2 = if config.has_layer_scale {
        let gamma = get_weight_opt(weights, &format!("{}.ls2.gamma", prefix))
            .or_else(|| get_weight_opt(weights, &format!("{}.ls2.scale_factor", prefix)))
            .ok_or_else(|| crate::error::Error::WeightNotFound(format!("{}.ls2.gamma/scale_factor", prefix)))?;
        Some(LayerScale {
            gamma: Param::new(gamma),
        })
    } else {
        None
    };

    let mlp = ViTMlp {
        fc1: make_linear(
            get_weight(weights, &format!("{}.mlp.fc1.weight", prefix))?,
            get_weight_opt(weights, &format!("{}.mlp.fc1.bias", prefix)),
        ),
        fc2: make_linear(
            get_weight(weights, &format!("{}.mlp.fc2.weight", prefix))?,
            get_weight_opt(weights, &format!("{}.mlp.fc2.bias", prefix)),
        ),
    };

    Ok(ViTBlock {
        norm1,
        attn,
        ls1,
        norm2,
        mlp,
        ls2,
    })
}

fn load_vit_attention(
    weights: &HashMap<String, Array>,
    prefix: &str,
    config: &ViTConfig,
) -> Result<ViTAttention, Error> {
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;

    // Combined QKV weight: [3*dim, dim] -> split into 3 x [dim, dim]
    let qkv_weight = get_weight(weights, &format!("{}.qkv.weight", prefix))?;
    let qkv_bias = get_weight_opt(weights, &format!("{}.qkv.bias", prefix));

    let w_parts = qkv_weight.split(3, 0)?;
    let b_parts = match &qkv_bias {
        Some(b) => {
            let parts = b.split(3, 0)?;
            vec![
                Some(parts[0].clone()),
                Some(parts[1].clone()),
                Some(parts[2].clone()),
            ]
        }
        None => vec![None, None, None],
    };

    Ok(ViTAttention {
        q_proj: make_linear(w_parts[0].clone(), b_parts[0].clone()),
        k_proj: make_linear(w_parts[1].clone(), b_parts[1].clone()),
        v_proj: make_linear(w_parts[2].clone(), b_parts[2].clone()),
        out_proj: make_linear(
            get_weight(weights, &format!("{}.proj.weight", prefix))?,
            get_weight_opt(weights, &format!("{}.proj.bias", prefix)),
        ),
        num_heads,
        head_dim,
        scale: (head_dim as f32).powf(-0.5),
    })
}
