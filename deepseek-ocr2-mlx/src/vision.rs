//! SAM ViT-B Image Encoder for DeepSeek-OCR-2.
//!
//! Implements a SAM-style ViT-B/16 encoder with:
//! - 12 transformer blocks, embed_dim=768, patch_size=16
//! - Window attention (window_size=14, global at indices 2,5,8,11)
//! - Decomposed relative position embeddings
//! - Neck + downsampling to 896-dim output

use std::collections::HashMap;

use mlx_rs::{
    array,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp},
    Array,
};

use crate::error::Error;

// ============================================================================
// PatchEmbed
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct PatchEmbed {
    #[param]
    pub proj: nn::Conv2d,
}

impl Module<&Array> for PatchEmbed {
    type Output = Array;
    type Error = Exception;
    fn training_mode(&mut self, _: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Exception> {
        // x: [B, H, W, 3] (MLX channels-last) -> [B, H/16, W/16, 768]
        self.proj.forward(x)
    }
}

// ============================================================================
// MLPBlock (GELU activation)
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct MLPBlock {
    #[param]
    pub lin1: nn::Linear,
    #[param]
    pub lin2: nn::Linear,
}

impl Module<&Array> for MLPBlock {
    type Output = Array;
    type Error = Exception;
    fn training_mode(&mut self, _: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Exception> {
        let h = self.lin1.forward(x)?;
        let h = nn::gelu_approximate(&h)?;
        self.lin2.forward(&h)
    }
}

// ============================================================================
// Attention with decomposed relative position embeddings
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct SamAttention {
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub use_rel_pos: bool,

    #[param]
    pub qkv: nn::Linear,
    #[param]
    pub proj: nn::Linear,
    #[param]
    pub rel_pos_h: Param<Array>, // may be empty [0] if not using rel pos
    #[param]
    pub rel_pos_w: Param<Array>,
}

impl SamAttention {
    /// Forward on spatial input x: [B, H, W, C]
    pub fn forward_spatial(&mut self, x: &Array) -> std::result::Result<Array, Exception> {
        let shape = x.shape();
        let b = shape[0] as i32;
        let h = shape[1] as i32;
        let w = shape[2] as i32;
        let hw = h * w;

        // QKV projection: [B, H, W, 3*C]
        let qkv = self.qkv.forward(x)?;
        // Reshape to [B, H*W, 3, num_heads, head_dim] then permute
        let qkv = qkv
            .reshape(&[b, hw, 3, self.num_heads, self.head_dim])?
            .transpose_axes(&[2, 0, 3, 1, 4])?; // [3, B, nH, H*W, hd]

        let q = qkv.index(0);
        let k = qkv.index(1);
        let v = qkv.index(2);

        // Attention: use SDPA (no mask for bidirectional)
        // q, k, v: [B, nH, H*W, hd]
        let attn = if self.use_rel_pos && (*self.rel_pos_h).shape()[0] > 0 {
            // Compute attention with relative position bias
            let scores = q.matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?.multiply(array!(self.scale))?;

            // Add decomposed relative position bias
            let bias = self.compute_rel_pos_bias(&q, h, w)?;
            let scores = scores.add(&bias)?;

            let attn_weights = ops::softmax_axis(&scores, -1, true)?;
            attn_weights.matmul(&v)?
        } else {
            // Simple SDPA without bias
            mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, None)?
        };

        // [B, nH, H*W, hd] -> [B, H, W, C]
        let out = attn
            .transpose_axes(&[0, 2, 1, 3])? // [B, H*W, nH, hd]
            .reshape(&[b, h, w, -1])?;

        self.proj.forward(&out)
    }

    /// Compute relative position bias for attention scores.
    fn compute_rel_pos_bias(&self, q: &Array, h: i32, w: i32) -> std::result::Result<Array, Exception> {
        let shape = q.shape();
        let b = shape[0] as i32;
        let nh = shape[1] as i32;
        let dim = shape[3] as i32;
        let bnh = b * nh;

        let rh = get_rel_pos_1d(h, h, &*self.rel_pos_h)?; // [H, H, hd]
        let rw = get_rel_pos_1d(w, w, &*self.rel_pos_w)?; // [W, W, hd]

        let r_q = q.reshape(&[bnh, h, w, dim])?;

        // rel_h: [bnh, h, w, c] @ [h, c, h] -> need matmul per h
        // Reshape for batch matmul: [bnh*w, h, c] @ [h, c]^T -> [bnh*w, h, h]
        // Then reshape to [bnh, h, w, h]
        let rh_t = rh.transpose_axes(&[0, 2, 1])?; // [H, hd, H]

        // einsum "bhwc,hkc->bhwk" = for each (b,h_pos,w_pos), dot with Rh[h_pos]
        // Simplified: reshape r_q to [bnh, h*w, dim], matmul with Rh^T [h, dim, h] -> need broadcasting
        // Actually: r_q[:, i, :, :] @ Rh[i, :, :]^T for each i
        // Use batch approach: [bnh*h, w, dim] @ [h, dim, h] with index=h
        // Simpler: just use the full matmul [bnh, h, w, dim] @ [1, h, dim, h] with broadcast
        let rh_expanded = rh_t.reshape(&[1, h, dim, h])?;
        let rel_h = r_q.matmul(&rh_expanded)?; // [bnh, h, w, h]

        // rel_w: similar for w dimension
        // r_q transposed: [bnh, w, h, dim]
        let r_q_t = r_q.transpose_axes(&[0, 2, 1, 3])?; // [bnh, w, h, dim]
        let rw_t = rw.transpose_axes(&[0, 2, 1])?; // [W, hd, W]
        let rw_expanded = rw_t.reshape(&[1, w, dim, w])?;
        let rel_w_raw = r_q_t.matmul(&rw_expanded)?; // [bnh, w, h, w]
        let rel_w = rel_w_raw.transpose_axes(&[0, 2, 1, 3])?; // [bnh, h, w, w]

        // Combine: [bnh, h, w, h, 1] + [bnh, h, w, 1, w] -> [bnh, h, w, h, w]
        let rel_h = rel_h.reshape(&[bnh, h * w, h, 1])?;
        let rel_w = rel_w.reshape(&[bnh, h * w, 1, w])?;
        let bias = rel_h.add(&rel_w)?; // [bnh, h*w, h, w]
        let bias = bias.reshape(&[b, nh, h * w, h * w])?;

        Ok(bias)
    }
}

/// Get relative position embeddings, interpolating if needed.
fn get_rel_pos_1d(q_size: i32, k_size: i32, rel_pos: &Array) -> std::result::Result<Array, Exception> {
    let max_rel_dist = 2 * q_size.max(k_size) - 1;
    let rel_pos_len = rel_pos.shape()[0] as i32;

    let rel_pos = if rel_pos_len != max_rel_dist {
        // Nearest-neighbor interpolation
        let scale = rel_pos_len as f32 / max_rel_dist as f32;
        let indices: Vec<i32> = (0..max_rel_dist)
            .map(|i| ((i as f32 * scale) as i32).min(rel_pos_len - 1))
            .collect();
        let idx_arr = Array::from_slice(&indices, &[max_rel_dist]);
        ops::indexing::take_axis(rel_pos, &idx_arr, 0)?
    } else {
        rel_pos.clone()
    };

    // Compute relative coordinate indices
    let q_max = q_size.max(k_size);
    let q_scale = (q_max as f32 / q_size as f32).max(1.0);
    let k_scale = (q_max as f32 / k_size as f32).max(1.0);
    let offset = ((k_size - 1) as f32 * k_scale) as i32;

    let mut coord_indices = Vec::with_capacity((q_size * k_size) as usize);
    for qi in 0..q_size {
        for ki in 0..k_size {
            let q_coord = (qi as f32 * q_scale) as i32;
            let k_coord = (ki as f32 * k_scale) as i32;
            coord_indices.push(q_coord - k_coord + offset);
        }
    }

    let idx = Array::from_slice(&coord_indices, &[q_size * k_size]);
    let gathered = ops::indexing::take_axis(&rel_pos, &idx, 0)?;
    let head_dim = rel_pos.shape()[1] as i32;
    gathered.reshape(&[q_size, k_size, head_dim])
}

// ============================================================================
// SAM Block with window attention
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct SamBlock {
    pub window_size: i32,

    #[param]
    pub norm1: nn::LayerNorm,
    #[param]
    pub attn: SamAttention,
    #[param]
    pub norm2: nn::LayerNorm,
    #[param]
    pub mlp: MLPBlock,
}

impl Module<&Array> for SamBlock {
    type Output = Array;
    type Error = Exception;
    fn training_mode(&mut self, _: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Exception> {
        let shortcut = x.clone();
        let h = self.norm1.forward(x)?;

        let (h, pad_hw) = if self.window_size > 0 {
            window_partition(&h, self.window_size)?
        } else {
            (h, (x.shape()[1] as i32, x.shape()[2] as i32))
        };

        let h = self.attn.forward_spatial(&h)?;

        let h = if self.window_size > 0 {
            let orig_h = shortcut.shape()[1] as i32;
            let orig_w = shortcut.shape()[2] as i32;
            window_unpartition(&h, self.window_size, pad_hw, (orig_h, orig_w))?
        } else {
            h
        };

        let x = shortcut.add(&h)?;
        let normed = self.norm2.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.add(&mlp_out)
    }
}

/// Partition into non-overlapping windows.
fn window_partition(x: &Array, ws: i32) -> std::result::Result<(Array, (i32, i32)), Exception> {
    let shape = x.shape();
    let b = shape[0] as i32;
    let h = shape[1] as i32;
    let w = shape[2] as i32;
    let c = shape[3] as i32;

    let pad_h = (ws - h % ws) % ws;
    let pad_w = (ws - w % ws) % ws;

    let x = if pad_h > 0 || pad_w > 0 {
        let widths = [(0, 0), (0, pad_h), (0, pad_w), (0, 0)];
        ops::pad(x, &widths, None::<Array>, None::<ops::PadMode>)?
    } else {
        x.clone()
    };

    let hp = h + pad_h;
    let wp = w + pad_w;
    let nh = hp / ws;
    let nw = wp / ws;

    let x = x
        .reshape(&[b, nh, ws, nw, ws, c])?
        .transpose_axes(&[0, 1, 3, 2, 4, 5])?
        .reshape(&[b * nh * nw, ws, ws, c])?;

    Ok((x, (hp, wp)))
}

/// Reverse window partition.
fn window_unpartition(
    windows: &Array,
    ws: i32,
    pad_hw: (i32, i32),
    orig_hw: (i32, i32),
) -> std::result::Result<Array, Exception> {
    let (hp, wp) = pad_hw;
    let (h, w) = orig_hw;
    let c = windows.shape()[3] as i32;
    let num_windows = hp * wp / ws / ws;
    let b = windows.shape()[0] as i32 / num_windows;
    let nh = hp / ws;
    let nw = wp / ws;

    let x = windows
        .reshape(&[b, nh, nw, ws, ws, c])?
        .transpose_axes(&[0, 1, 3, 2, 4, 5])?
        .reshape(&[b, hp, wp, c])?;

    if hp > h || wp > w {
        Ok(x.index((.., ..h, ..w, ..)))
    } else {
        Ok(x)
    }
}

// ============================================================================
// ImageEncoderViT (SAM ViT-B)
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct ImageEncoderViT {
    #[param]
    pub patch_embed: PatchEmbed,
    #[param]
    pub pos_embed: Param<Array>, // [1, H, W, C]
    #[param]
    pub blocks: Vec<SamBlock>,
    // Neck: conv1(768->256,k=1) -> ln2d -> conv2(256->256,k=3) -> ln2d
    #[param]
    pub neck_0: nn::Conv2d,
    #[param]
    pub neck_0_ln_weight: Param<Array>,
    #[param]
    pub neck_0_ln_bias: Param<Array>,
    #[param]
    pub neck_1: nn::Conv2d,
    #[param]
    pub neck_1_ln_weight: Param<Array>,
    #[param]
    pub neck_1_ln_bias: Param<Array>,
    // Downsample
    #[param]
    pub net_2: nn::Conv2d,
    #[param]
    pub net_3: nn::Conv2d,
}

impl ImageEncoderViT {
    /// LayerNorm2d: normalize over channels in NCHW format
    fn layer_norm_2d(
        x: &Array,
        weight: &Array,
        bias: &Array,
        eps: f32,
    ) -> std::result::Result<Array, Exception> {
        // x: [B, C, H, W]
        let u = x.mean_axis(1, true)?;
        let centered = x.subtract(&u)?;
        let var = centered.square()?.mean_axis(1, true)?;
        let x_norm = centered.divide(&var.add(array!(eps))?.sqrt()?)?;
        let w = weight.reshape(&[1, -1, 1, 1])?;
        let b = bias.reshape(&[1, -1, 1, 1])?;
        x_norm.multiply(&w)?.add(&b)
    }
}

impl Module<&Array> for ImageEncoderViT {
    type Output = Array;
    type Error = Exception;
    fn training_mode(&mut self, _: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Exception> {
        // x: [B, H, W, 3] channels-last
        let mut h = self.patch_embed.forward(x)?;

        // Add position embeddings (interpolate if sizes differ)
        let pe_h = (*self.pos_embed).shape()[1] as i32;
        let cur_h = h.shape()[1] as i32;
        if pe_h == cur_h {
            h = h.add(&*self.pos_embed)?;
        } else {
            // Simple nearest-neighbor interpolation of pos_embed
            let pe = nearest_interpolate_2d(&*self.pos_embed, cur_h, cur_h)?;
            h = h.add(&pe)?;
        }

        // Transformer blocks
        for block in &mut self.blocks {
            h = block.forward(&h)?;
        }

        // Neck: [B, H', W', 768] -> [B, 768, H', W'] (channels-first for conv)
        let h = h.transpose_axes(&[0, 3, 1, 2])?;

        // Conv1 (k=1, no bias): need channels-last for MLX Conv2d
        let h = h.transpose_axes(&[0, 2, 3, 1])?;
        let h = self.neck_0.forward(&h)?;
        let h = h.transpose_axes(&[0, 3, 1, 2])?;
        let h = Self::layer_norm_2d(&h, &*self.neck_0_ln_weight, &*self.neck_0_ln_bias, 1e-6)?;

        // Conv2 (k=3, p=1, no bias)
        let h = h.transpose_axes(&[0, 2, 3, 1])?;
        let h = self.neck_1.forward(&h)?;
        let h = h.transpose_axes(&[0, 3, 1, 2])?;
        let h = Self::layer_norm_2d(&h, &*self.neck_1_ln_weight, &*self.neck_1_ln_bias, 1e-6)?;

        // net_2: Conv2d(256->512, k=3, s=2, p=1)
        let h = h.transpose_axes(&[0, 2, 3, 1])?;
        let h = self.net_2.forward(&h)?;

        // net_3: Conv2d(512->896, k=3, s=2, p=1)
        let h = h.transpose_axes(&[0, 3, 1, 2])?;
        let h = h.transpose_axes(&[0, 2, 3, 1])?;
        let h = self.net_3.forward(&h)?;

        // Output: [B, H'', W'', 896] channels-last
        Ok(h)
    }
}

/// Nearest-neighbor 2D interpolation for position embeddings.
fn nearest_interpolate_2d(pe: &Array, tgt_h: i32, tgt_w: i32) -> std::result::Result<Array, Exception> {
    let src_h = pe.shape()[1] as i32;
    let src_w = pe.shape()[2] as i32;
    let scale_h = src_h as f32 / tgt_h as f32;
    let scale_w = src_w as f32 / tgt_w as f32;

    let h_indices: Vec<i32> = (0..tgt_h)
        .map(|i| ((i as f32 * scale_h) as i32).min(src_h - 1))
        .collect();
    let w_indices: Vec<i32> = (0..tgt_w)
        .map(|i| ((i as f32 * scale_w) as i32).min(src_w - 1))
        .collect();

    let h_idx = Array::from_slice(&h_indices, &[tgt_h]);
    let w_idx = Array::from_slice(&w_indices, &[tgt_w]);

    // pe: [1, src_h, src_w, C] -> take along h axis first, then w
    let result = ops::indexing::take_axis(pe, &h_idx, 1)?;
    ops::indexing::take_axis(&result, &w_idx, 2)
}

// ============================================================================
// Weight loading
// ============================================================================

pub fn load_sam_encoder(
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<ImageEncoderViT, Error> {
    let get = |key: &str| -> Result<Array, Error> {
        weights
            .get(key)
            .cloned()
            .ok_or_else(|| Error::Model(format!("SAM weight not found: {}", key)))
    };

    // Transpose Conv2d weights from PyTorch [O,I,H,W] to MLX [O,H,W,I]
    let conv2d_weight = |key: &str| -> Result<Array, Error> {
        // Weights already in MLX format (C_out, kH, kW, C_in), no transpose needed
        get(key)
    };

    // PatchEmbed: Conv2d(3, 768, k=16, s=16)
    let patch_embed = PatchEmbed {
        proj: nn::Conv2d {
            weight: Param::new(conv2d_weight(&format!("{}.patch_embed.proj.weight", prefix))?),
            bias: Param::new(Some(get(&format!("{}.patch_embed.proj.bias", prefix))?)),
            stride: (16, 16),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
        },
    };

    let pos_embed = get(&format!("{}.pos_embed", prefix))?;

    // Blocks
    let global_attn_indexes = [2, 5, 8, 11];
    let window_size = 14;
    let num_heads = 12;
    let embed_dim = 768;
    let head_dim = embed_dim / num_heads;

    let mut blocks = Vec::new();
    for i in 0..12i32 {
        let bp = format!("{}.blocks.{}", prefix, i);
        let is_global = global_attn_indexes.contains(&i);
        let ws = if is_global { 0 } else { window_size };

        let rel_pos_h = get(&format!("{}.attn.rel_pos_h", bp))?;
        let rel_pos_w = get(&format!("{}.attn.rel_pos_w", bp))?;

        let attn = SamAttention {
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            use_rel_pos: true,
            qkv: nn::Linear {
                weight: Param::new(get(&format!("{}.attn.qkv.weight", bp))?),
                bias: Param::new(Some(get(&format!("{}.attn.qkv.bias", bp))?)),
            },
            proj: nn::Linear {
                weight: Param::new(get(&format!("{}.attn.proj.weight", bp))?),
                bias: Param::new(Some(get(&format!("{}.attn.proj.bias", bp))?)),
            },
            rel_pos_h: Param::new(rel_pos_h),
            rel_pos_w: Param::new(rel_pos_w),
        };

        let mlp = MLPBlock {
            lin1: nn::Linear {
                weight: Param::new(get(&format!("{}.mlp.lin1.weight", bp))?),
                bias: Param::new(Some(get(&format!("{}.mlp.lin1.bias", bp))?)),
            },
            lin2: nn::Linear {
                weight: Param::new(get(&format!("{}.mlp.lin2.weight", bp))?),
                bias: Param::new(Some(get(&format!("{}.mlp.lin2.bias", bp))?)),
            },
        };

        blocks.push(SamBlock {
            window_size: ws,
            norm1: nn::LayerNorm {
                weight: Param::new(Some(get(&format!("{}.norm1.weight", bp))?)),
                bias: Param::new(Some(get(&format!("{}.norm1.bias", bp))?)),
                eps: 1e-6,
                dimensions: embed_dim,
            },
            attn,
            norm2: nn::LayerNorm {
                weight: Param::new(Some(get(&format!("{}.norm2.weight", bp))?)),
                bias: Param::new(Some(get(&format!("{}.norm2.bias", bp))?)),
                eps: 1e-6,
                dimensions: embed_dim,
            },
            mlp,
        });
    }

    // Neck convolutions and LayerNorm2d weights
    // neck.0: Conv2d(768, 256, k=1)
    let neck_0 = nn::Conv2d {
        weight: Param::new(conv2d_weight(&format!("{}.neck.0.weight", prefix))?),
        bias: Param::new(None),
        stride: (1, 1),
        padding: (0, 0),
        dilation: (1, 1),
        groups: 1,
    };
    let neck_0_ln_weight = get(&format!("{}.neck.1.weight", prefix))?;
    let neck_0_ln_bias = get(&format!("{}.neck.1.bias", prefix))?;

    // neck.2: Conv2d(256, 256, k=3, p=1)
    let neck_1 = nn::Conv2d {
        weight: Param::new(conv2d_weight(&format!("{}.neck.2.weight", prefix))?),
        bias: Param::new(None),
        stride: (1, 1),
        padding: (1, 1),
        dilation: (1, 1),
        groups: 1,
    };
    let neck_1_ln_weight = get(&format!("{}.neck.3.weight", prefix))?;
    let neck_1_ln_bias = get(&format!("{}.neck.3.bias", prefix))?;

    // net_2: Conv2d(256, 512, k=3, s=2, p=1)
    let net_2 = nn::Conv2d {
        weight: Param::new(conv2d_weight(&format!("{}.net_2.weight", prefix))?),
        bias: Param::new(None),
        stride: (2, 2),
        padding: (1, 1),
        dilation: (1, 1),
        groups: 1,
    };
    // net_3: Conv2d(512, 896, k=3, s=2, p=1)
    let net_3 = nn::Conv2d {
        weight: Param::new(conv2d_weight(&format!("{}.net_3.weight", prefix))?),
        bias: Param::new(None),
        stride: (2, 2),
        padding: (1, 1),
        dilation: (1, 1),
        groups: 1,
    };

    Ok(ImageEncoderViT {
        patch_embed,
        pos_embed: Param::new(pos_embed),
        blocks,
        neck_0,
        neck_0_ln_weight: Param::new(neck_0_ln_weight),
        neck_0_ln_bias: Param::new(neck_0_ln_bias),
        neck_1,
        neck_1_ln_weight: Param::new(neck_1_ln_weight),
        neck_1_ln_bias: Param::new(neck_1_ln_bias),
        net_2,
        net_3,
    })
}
