use anyhow::{ensure, Context, Result};
use mlx_rs::{fast, nn, ops, Array};

use crate::weights::{Linear, Weights};
use crate::VisionConfig;

struct LayerNorm {
    weight: Array,
    bias: Array,
    eps: f32,
}

impl LayerNorm {
    fn load(weights: &mut Weights, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: weights.take(&format!("{prefix}.weight"))?,
            bias: weights.take(&format!("{prefix}.bias"))?,
            eps,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(fast::layer_norm(x, &self.weight, &self.bias, self.eps)?)
    }
}

struct Rotary {
    cos: Array,
    sin: Array,
}

impl Rotary {
    fn new(h: i32, w: i32, head_dim: i32) -> Result<Self> {
        // The first quarter of a head uses row coordinates and the second
        // uses columns. The reference repeats both for the NeoX rotation.
        let quarter = head_dim / 4;
        let inv_freq: Vec<_> = (0..quarter)
            .map(|i| 1.0 / 10000_f32.powf(i as f32 / quarter as f32))
            .collect();
        let mut angles = Vec::new();
        for row in 0..h {
            for col in 0..w {
                for coord in [row, col] {
                    angles.extend(inv_freq.iter().map(|freq| coord as f32 * freq));
                }
            }
        }
        let angles = Array::from_slice(&angles, &[1, 1, h * w, head_dim / 2]);
        let angles = ops::concatenate_axis(&[&angles, &angles], -1)?;
        Ok(Self {
            cos: angles.cos()?,
            sin: angles.sin()?,
        })
    }

    fn apply(&self, x: &Array) -> Result<Array> {
        let halves = x.split(2, -1)?;
        let rotated = ops::concatenate_axis(&[&halves[1].negative()?, &halves[0]], -1)?;
        let x_rot = x.multiply(&self.cos)?.add(rotated.multiply(&self.sin)?)?;
        Ok(x_rot.as_dtype(x.dtype())?)
    }
}

struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    out: Linear,
    fc1: Linear,
    fc2: Linear,
}

impl Block {
    fn load(weights: &mut Weights, index: usize) -> Result<Self> {
        let prefix = format!("visual.vision_model.encoder.layers.{index}");
        let attn = format!("{prefix}.self_attn");
        // The official checkpoint stores Q, K and V separately.
        let mut matrices = Vec::with_capacity(3);
        let mut biases = Vec::with_capacity(3);
        for proj in ["q_proj", "k_proj", "v_proj"] {
            matrices.push(weights.take(&format!("{attn}.{proj}.weight"))?);
            biases.push(weights.take(&format!("{attn}.{proj}.bias"))?);
        }
        let qkv = Linear {
            weight: ops::concatenate_axis(&matrices, 0)?,
            bias: Some(ops::concatenate_axis(&biases, 0)?),
        };
        Ok(Self {
            norm1: LayerNorm::load(weights, &format!("{prefix}.layer_norm1"), 1e-6)?,
            norm2: LayerNorm::load(weights, &format!("{prefix}.layer_norm2"), 1e-6)?,
            qkv,
            out: Linear::load(weights, &format!("{attn}.out_proj"), true)?,
            fc1: Linear::load(weights, &format!("{prefix}.mlp.fc1"), true)?,
            fc2: Linear::load(weights, &format!("{prefix}.mlp.fc2"), true)?,
        })
    }

    fn forward(&self, x: &Array, heads: i32, rotary: &Rotary) -> Result<Array> {
        let n = x.shape()[0];
        let hidden = x.shape()[1];
        let head_dim = hidden / heads;
        let qkv = self.qkv.forward(&self.norm1.forward(x)?)?.split(3, -1)?;
        let as_heads = |v: &Array| {
            v.reshape(&[1, n, heads, head_dim])?
                .transpose_axes(&[0, 2, 1, 3])
        };
        let q = rotary.apply(&as_heads(&qkv[0])?)?;
        let k = rotary.apply(&as_heads(&qkv[1])?)?;
        let v = as_heads(&qkv[2])?;
        let attn =
            fast::scaled_dot_product_attention(q, k, v, (head_dim as f32).sqrt().recip(), None)?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[n, hidden])?;
        let x = x.add(self.out.forward(&attn)?)?;
        let ff = self.fc1.forward(&self.norm2.forward(&x)?)?;
        let ff = nn::gelu_approximate(&ff)?.as_dtype(ff.dtype())?;
        Ok(x.add(self.fc2.forward(&ff)?)?)
    }
}

// Half-pixel coordinates for bilinear interpolation of the learned position table.
fn interpolation_axis(size: i32, target: i32) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let mut lo = Vec::with_capacity(target as usize);
    let mut hi = Vec::with_capacity(target as usize);
    let mut weights = Vec::with_capacity(target as usize);
    for i in 0..target {
        let point = if target == 1 {
            0.0
        } else {
            (i as f32 + 0.5) * size as f32 / target as f32 - 0.5
        };
        let left = point.floor() as i32;
        let lower = left.clamp(0, size - 1);
        lo.push(lower);
        hi.push((left + 1).clamp(0, size - 1));
        weights.push(point - lower as f32);
    }
    (lo, hi, weights)
}

pub(crate) struct Vision {
    patch_weight: Array,
    patch_bias: Array,
    positions: Array,
    position_side: i32,
    patch: i32,
    merge: i32,
    hidden: i32,
    heads: i32,
    blocks: Vec<Block>,
    post_norm: LayerNorm,
    pre_norm: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl Vision {
    pub(crate) fn load(weights: &mut Weights, cfg: &VisionConfig) -> Result<Self> {
        ensure!(
            cfg.patch_size > 0 && cfg.spatial_merge_size > 0 && cfg.image_size >= cfg.patch_size,
            "invalid vision patch configuration"
        );
        ensure!(
            cfg.hidden_size > 0
                && cfg.num_attention_heads > 0
                && cfg.hidden_size % cfg.num_attention_heads == 0
                && (cfg.hidden_size / cfg.num_attention_heads) % 4 == 0,
            "vision head width must be a positive multiple of four"
        );
        let prefix = "visual.vision_model.embeddings";
        let patch_weight = weights.take(&format!("{prefix}.patch_embedding.weight"))?;
        ensure!(
            patch_weight.shape() == [cfg.hidden_size, 3, cfg.patch_size, cfg.patch_size],
            "invalid PaddleOCR patch kernel shape {:?}",
            patch_weight.shape()
        );
        let positions = weights.take(&format!("{prefix}.position_embedding.weight"))?;
        let position_side = cfg.image_size / cfg.patch_size;
        let position_count = position_side
            .checked_mul(position_side)
            .context("vision position grid overflow")?;
        ensure!(
            positions.shape() == [position_count, cfg.hidden_size],
            "invalid vision position table shape {:?}",
            positions.shape()
        );
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(weights, i))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_weight: patch_weight.transpose_axes(&[0, 2, 3, 1])?,
            patch_bias: weights.take(&format!("{prefix}.patch_embedding.bias"))?,
            positions,
            position_side,
            patch: cfg.patch_size,
            merge: cfg.spatial_merge_size,
            hidden: cfg.hidden_size,
            heads: cfg.num_attention_heads,
            blocks,
            post_norm: LayerNorm::load(
                weights,
                "visual.vision_model.post_layernorm",
                cfg.layer_norm_eps,
            )?,
            pre_norm: LayerNorm::load(weights, "mlp_AR.pre_norm", 1e-6)?,
            linear1: Linear::load(weights, "mlp_AR.linear_1", true)?,
            linear2: Linear::load(weights, "mlp_AR.linear_2", true)?,
        })
    }

    fn interpolate_positions(&self, h: i32, w: i32) -> Result<Array> {
        if h == self.position_side && w == self.position_side {
            return Ok(self.positions.clone());
        }
        let (top, bottom, dy) = interpolation_axis(self.position_side, h);
        let (left, right, dx) = interpolation_axis(self.position_side, w);
        let gather = |rows: &[i32], cols: &[i32]| -> Result<Array> {
            let indices: Vec<_> = rows
                .iter()
                .flat_map(|&row| cols.iter().map(move |&col| row * self.position_side + col))
                .collect();
            Ok(self
                .positions
                .take_axis(Array::from_slice(&indices, &[h, w]), 0)?)
        };
        let dy = Array::from_slice(&dy, &[h, 1, 1]);
        let dx = Array::from_slice(&dx, &[1, w, 1]);
        let one = Array::from_f32(1.0);
        let top_weight = one.subtract(&dy)?;
        let left_weight = one.subtract(&dx)?;
        let tl = top_weight
            .multiply(&left_weight)?
            .multiply(gather(&top, &left)?)?;
        let tr = top_weight.multiply(&dx)?.multiply(gather(&top, &right)?)?;
        let bl = dy
            .multiply(&left_weight)?
            .multiply(gather(&bottom, &left)?)?;
        let br = dy.multiply(&dx)?.multiply(gather(&bottom, &right)?)?;
        Ok(tl
            .add(tr)?
            .add(bl)?
            .add(br)?
            .as_dtype(self.positions.dtype())?
            .reshape(&[h * w, self.hidden])?)
    }

    pub(crate) fn encode(&self, pixels: &Array, grid_h: i32, grid_w: i32) -> Result<Array> {
        ensure!(
            grid_h > 0 && grid_w > 0 && grid_h % self.merge == 0 && grid_w % self.merge == 0,
            "vision grid {grid_h}x{grid_w} must be positive and divisible by {}",
            self.merge
        );
        let h = grid_h
            .checked_mul(self.patch)
            .context("image height overflow")?;
        let w = grid_w
            .checked_mul(self.patch)
            .context("image width overflow")?;
        let n = grid_h
            .checked_mul(grid_w)
            .context("image patch count overflow")?;
        ensure!(
            pixels.shape() == [1, h, w, 3],
            "expected NHWC pixels [1,{h},{w},3], got {:?}",
            pixels.shape()
        );
        let pixels = pixels.as_dtype(self.patch_weight.dtype())?;
        let patches = ops::conv2d(
            &pixels,
            &self.patch_weight,
            (self.patch, self.patch),
            None,
            None,
            None,
        )?
        .add(&self.patch_bias)?
        .reshape(&[n, self.hidden])?;
        let mut x = patches.add(self.interpolate_positions(grid_h, grid_w)?)?;
        let rotary = Rotary::new(grid_h, grid_w, self.hidden / self.heads)?;
        for (i, block) in self.blocks.iter().enumerate() {
            x = block
                .forward(&x, self.heads, &rotary)
                .with_context(|| format!("vision encoder layer {i}"))?;
        }
        // Traverse merge blocks row-major, then traverse patches inside each block.
        let rows = grid_h / self.merge;
        let cols = grid_w / self.merge;
        let merged = self
            .pre_norm
            .forward(&self.post_norm.forward(&x)?)?
            .reshape(&[rows, self.merge, cols, self.merge, self.hidden])?
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[rows * cols, -1])?;
        let projected = self.linear1.forward(&merged)?;
        self.linear2
            .forward(&nn::gelu(&projected)?.as_dtype(projected.dtype())?)
    }
}
