//! Qwen3.5/3.8 image tower. Inputs are processor-normalized, merge-ordered patches.
//! Uses the checkpoint's full-precision vision weights and its spatial RoPE.
//! Reference: mlx-vlm 0.8.0 qwen3_vl/vision.py (MIT); no DeepStack in Qwen3.5.

use crate::{attention::apply_rotary, model::load_all_weights};
use mlx_rs::{
    error::Exception,
    module::{Module, Param},
    nn,
    ops::indexing::IndexOp,
    Array,
};
use mlx_rs_core::error::Error;
use serde::Deserialize;
use std::{collections::HashMap, path::Path};

#[derive(Clone, Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: i32,
    pub num_heads: i32,
    pub patch_size: i32,
    pub temporal_patch_size: i32,
    pub spatial_merge_size: i32,
    pub in_channels: i32,
    pub out_hidden_size: i32,
    pub deepstack_visual_indexes: Vec<usize>,
}

fn get(w: &HashMap<String, Array>, name: &str) -> Result<Array, Error> {
    let value = w
        .get(name)
        .ok_or_else(|| Error::WeightNotFound(name.into()))?;
    if value.dtype() == mlx_rs::Dtype::Uint32 {
        return Err(Error::Model(
            "This vision path requires unquantized vision weights".into(),
        ));
    }
    // Retain checkpoint values, but compute the small image tower in float32.
    // BF16 attention rounding compounds strongly across its 27 layers and can
    // vary across Metal kernel versions; the language model remains INT4/BF16.
    Ok(value.as_dtype(mlx_rs::Dtype::Float32)?)
}
fn linear(w: &HashMap<String, Array>, name: &str) -> Result<nn::Linear, Error> {
    let weight = get(w, &format!("{name}.weight"))?;
    if weight.dtype() == mlx_rs::Dtype::Uint32 {
        return Err(Error::Model(
            "This vision path requires unquantized vision weights".into(),
        ));
    }
    Ok(nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(Some(get(w, &format!("{name}.bias"))?)),
    })
}
struct Norm {
    weight: Array,
    bias: Array,
}
impl Norm {
    fn load(w: &HashMap<String, Array>, p: &str) -> Result<Self, Error> {
        Ok(Self {
            weight: get(w, &format!("{p}.weight"))?,
            bias: get(w, &format!("{p}.bias"))?,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        mlx_rs::fast::layer_norm(x, Some(&self.weight), Some(&self.bias), 1e-6)
    }
}
struct Block {
    norm1: Norm,
    norm2: Norm,
    qkv: nn::Linear,
    proj: nn::Linear,
    fc1: nn::Linear,
    fc2: nn::Linear,
}
impl Block {
    fn forward(
        &mut self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        heads: i32,
    ) -> Result<Array, Exception> {
        let n = x.shape()[0];
        let dim = x.shape()[1] / heads;
        let norm1 = self.norm1.forward(x)?;
        let qkv = self
            .qkv
            .forward(&norm1)?
            .reshape(&[n, 3, heads, dim])?
            .transpose_axes(&[1, 2, 0, 3])?;
        let q = apply_rotary(&qkv.index((0..1, .., .., ..)), cos, sin)?;
        let k = apply_rotary(&qkv.index((1..2, .., .., ..)), cos, sin)?;
        let v = qkv.index((2..3, .., .., ..));
        // MLX's fused kernel supports 64/80/128-wide heads. Qwen's 72-wide
        // vision heads need zero padding, matching mlx-vlm ensure_fused_sdpa.
        // The fallback's BF16 attention math produces materially different values.
        let padded_dim = [64, 80, 128].into_iter().find(|&d| d >= dim).unwrap_or(dim);
        let pad = |a: Array| -> Result<Array, Exception> {
            if padded_dim == dim {
                return Ok(a);
            }
            let zeros =
                Array::zeros::<f32>(&[1, heads, n, padded_dim - dim])?.as_dtype(a.dtype())?;
            mlx_rs::ops::concatenate_axis(&[&a, &zeros], -1)
        };
        let attn = mlx_rs::fast::scaled_dot_product_attention(
            pad(q)?,
            pad(k)?,
            pad(v)?,
            (dim as f64).sqrt().recip() as f32,
            None,
        )?
        .index((.., .., .., ..dim))
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[n, -1])?;
        let proj = self.proj.forward(&attn)?;
        let h = x.add(proj)?;
        let normed = self.norm2.forward(&h)?;
        let fc1 = self.fc1.forward(&normed)?;
        let activated = nn::gelu_approximate(&fc1)?;
        h.add(self.fc2.forward(&activated)?)
    }
}

pub struct VisionModel {
    pub config: VisionConfig,
    patch: nn::Conv3d,
    position: nn::Embedding,
    blocks: Vec<Block>,
    norm: Norm,
    fc1: nn::Linear,
    fc2: nn::Linear,
}
impl VisionModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path.join("config.json"))?)?;
        let config: VisionConfig = serde_json::from_value(json["vision_config"].clone())?;
        if !config.deepstack_visual_indexes.is_empty()
            || config.num_heads <= 0
            || config.hidden_size <= 0
            || config.hidden_size % config.num_heads != 0
            || config.spatial_merge_size <= 0
            || config.patch_size <= 0
            || config.temporal_patch_size != 2
            || config.in_channels != 3
        {
            return Err(Error::Model(
                "Expected Qwen3.5 vision configuration without DeepStack".into(),
            ));
        }
        let w = load_all_weights(path)?;
        let p = "vision_tower";
        let mut patch = get(&w, &format!("{p}.patch_embed.proj.weight"))?;
        if patch.shape()[1] == config.in_channels && patch.shape()[4] != config.in_channels {
            patch = patch.transpose_axes(&[0, 2, 3, 4, 1])?;
        }
        let mut blocks = Vec::new();
        for i in 0..config.depth {
            let b = format!("{p}.blocks.{i}");
            blocks.push(Block {
                norm1: Norm::load(&w, &format!("{b}.norm1"))?,
                norm2: Norm::load(&w, &format!("{b}.norm2"))?,
                qkv: linear(&w, &format!("{b}.attn.qkv"))?,
                proj: linear(&w, &format!("{b}.attn.proj"))?,
                fc1: linear(&w, &format!("{b}.mlp.linear_fc1"))?,
                fc2: linear(&w, &format!("{b}.mlp.linear_fc2"))?,
            });
        }
        Ok(Self {
            patch: nn::Conv3d {
                weight: Param::new(patch),
                bias: Param::new(Some(get(&w, &format!("{p}.patch_embed.proj.bias"))?)),
                stride: (
                    config.temporal_patch_size,
                    config.patch_size,
                    config.patch_size,
                ),
                padding: (0, 0, 0),
                dilation: (1, 1, 1),
                groups: 1,
            },
            position: nn::Embedding {
                weight: Param::new(get(&w, &format!("{p}.pos_embed.weight"))?),
            },
            blocks,
            norm: Norm::load(&w, &format!("{p}.merger.norm"))?,
            fc1: linear(&w, &format!("{p}.merger.linear_fc1"))?,
            fc2: linear(&w, &format!("{p}.merger.linear_fc2"))?,
            config,
        })
    }

    /// One image only: independent attention per image prevents cross-camera leakage.
    /// Indices/weights describe bilinear learned-position interpolation in patch order.
    pub fn forward(
        &mut self,
        pixels: &Array,
        indices: &Array,
        weights: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array, Exception> {
        let c = &self.config;
        if pixels.ndim() != 2 {
            return Err(Exception::custom("Expected rank-two normalized patches"));
        }
        let n = pixels.shape()[0];
        let area = c.spatial_merge_size * c.spatial_merge_size;
        if n == 0
            || pixels.shape()[1]
                != c.in_channels * c.temporal_patch_size * c.patch_size * c.patch_size
            || n % area != 0
            || indices.shape() != [4, n]
            || weights.shape() != [4, n]
        {
            return Err(Exception::custom(
                "Invalid normalized image patches/position interpolation",
            ));
        }
        let x = pixels
            .as_dtype(self.patch.weight.dtype())?
            .reshape(&[
                n,
                c.in_channels,
                c.temporal_patch_size,
                c.patch_size,
                c.patch_size,
            ])?
            .transpose_axes(&[0, 2, 3, 4, 1])?;
        let mut h = self.patch.forward(&x)?.reshape(&[n, c.hidden_size])?;
        let pos = self.position.forward(indices)?;
        let weighted = pos.multiply(weights.as_dtype(pos.dtype())?.reshape(&[4, n, 1])?)?;
        let pos = weighted
            .index(0)
            .add(weighted.index(1))?
            .add(weighted.index(2))?
            .add(weighted.index(3))?;
        h = h.add(pos)?;
        for block in &mut self.blocks {
            h = block.forward(&h, cos, sin, c.num_heads)?;
        }
        let h = self
            .norm
            .forward(&h)?
            .reshape(&[n / area, area * c.hidden_size])?;
        self.fc2.forward(&nn::gelu(self.fc1.forward(&h)?)?)
    }
}
