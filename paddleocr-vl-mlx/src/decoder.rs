use crate::{
    weights::{Linear, Weights},
    Config,
};
use anyhow::Result;
use mlx_rs::{
    fast::{self, ScaledDotProductAttentionMask},
    nn,
    ops::{
        concatenate_axis,
        indexing::{Ellipsis, IndexOp},
    },
    Array, Dtype,
};
use mlx_rs_core::{KVCache, KeyValueCache};

struct Layer {
    input_norm: Array,
    post_norm: Array,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}

pub(crate) struct Decoder {
    pub embedding: Array,
    layers: Vec<Layer>,
    norm: Array,
    head: Linear,
    eps: f32,
    heads: i32,
    kv_heads: i32,
    dim: i32,
    theta: f32,
    sections: [usize; 3],
}

// A still image uses a fixed temporal index and a 2D grid of merged patches.
// Text following the image resumes after the maximum of the three axes.
pub(crate) fn positions(prefix: i32, h: i32, w: i32, suffix: i32) -> (Vec<[i32; 3]>, i32) {
    let mut pos: Vec<_> = (0..prefix).map(|p| [p; 3]).collect();
    for row in 0..h {
        for col in 0..w {
            pos.push([prefix, prefix + row, prefix + col]);
        }
    }
    let start = prefix + h.max(w);
    pos.extend((start..start + suffix).map(|p| [p; 3]));
    (pos, start + suffix)
}

enum Rotation {
    ImagePrompt { cos: Array, sin: Array },
    Text { offset: i32 },
}

impl Decoder {
    pub fn load(weights: &mut Weights, cfg: &Config) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let mut linear = |name: &str| Linear::load(weights, &format!("{p}.{name}"), false);
            layers.push(Layer {
                q: linear("self_attn.q_proj")?,
                k: linear("self_attn.k_proj")?,
                v: linear("self_attn.v_proj")?,
                o: linear("self_attn.o_proj")?,
                gate: linear("mlp.gate_proj")?,
                up: linear("mlp.up_proj")?,
                down: linear("mlp.down_proj")?,
                input_norm: weights.take(&format!("{p}.input_layernorm.weight"))?,
                post_norm: weights.take(&format!("{p}.post_attention_layernorm.weight"))?,
            });
        }
        Ok(Self {
            layers,
            embedding: weights.take("model.embed_tokens.weight")?,
            norm: weights.take("model.norm.weight")?,
            head: Linear::load(weights, "lm_head", false)?,
            eps: cfg.rms_norm_eps,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            dim: cfg.head_dim,
            theta: cfg.rope_theta,
            sections: cfg.rope_scaling.mrope_section,
        })
    }

    pub fn cache(&self) -> Vec<KVCache> {
        self.layers.iter().map(|_| KVCache::new()).collect()
    }

    pub fn embed(&self, ids: &[u32]) -> Result<Array> {
        Ok(self
            .embedding
            .take_axis(&Array::from_slice(ids, &[ids.len() as i32]), 0)?)
    }

    pub fn prefill(&self, embeds: Array, pos: &[[i32; 3]], cache: &mut [KVCache]) -> Result<Array> {
        let half = self.dim as usize / 2;
        let mut angles = Vec::with_capacity(pos.len() * half);
        for xyz in pos {
            for j in 0..half {
                let axis = if j < self.sections[0] {
                    0
                } else if j < self.sections[0] + self.sections[1] {
                    1
                } else {
                    2
                };
                angles.push(xyz[axis] as f32 * self.theta.powf(-2.0 * j as f32 / self.dim as f32));
            }
        }
        let half_angles = Array::from_slice(&angles, &[1, 1, pos.len() as i32, half as i32]);
        let angles = concatenate_axis(&[&half_angles, &half_angles], -1)?;
        let rot = Rotation::ImagePrompt {
            cos: angles.cos()?.as_dtype(embeds.dtype())?,
            sin: angles.sin()?.as_dtype(embeds.dtype())?,
        };
        self.forward(embeds, &rot, cache)
    }

    pub fn step(&self, token: &Array, offset: i32, cache: &mut [KVCache]) -> Result<Array> {
        self.forward(
            self.embedding.take_axis(token, 0)?,
            &Rotation::Text { offset },
            cache,
        )
    }

    fn rotate(&self, x: &Array, rotation: &Rotation) -> Result<Array> {
        Ok(match rotation {
            Rotation::Text { offset } => {
                fast::rope(x, self.dim, false, self.theta, 1.0, *offset, None)?
            }
            Rotation::ImagePrompt { cos, sin } => {
                let half = self.dim / 2;
                let x32 = x.as_dtype(Dtype::Float32)?;
                let rotated = concatenate_axis(
                    &[
                        x32.index((Ellipsis, half..)).negative()?,
                        x32.index((Ellipsis, ..half)),
                    ],
                    -1,
                )?;
                x32.multiply(cos)?
                    .add(&rotated.multiply(sin)?)?
                    .as_dtype(x.dtype())?
            }
        })
    }

    fn forward(&self, mut x: Array, rot: &Rotation, cache: &mut [KVCache]) -> Result<Array> {
        let len = x.shape()[0];
        for (layer, cache) in self.layers.iter().zip(cache) {
            let h = fast::rms_norm(&x, &layer.input_norm, self.eps)?;
            let heads = |proj: &Linear, n: i32| -> Result<Array> {
                Ok(proj
                    .forward(&h)?
                    .reshape(&[1, len, n, self.dim])?
                    .transpose_axes(&[0, 2, 1, 3])?)
            };
            let q = self.rotate(&heads(&layer.q, self.heads)?, rot)?;
            let k = self.rotate(&heads(&layer.k, self.kv_heads)?, rot)?;
            let (k, v) = cache.update_and_fetch(k, heads(&layer.v, self.kv_heads)?)?;
            let mask = (len > 1).then_some(ScaledDotProductAttentionMask::Causal);
            let a = fast::scaled_dot_product_attention(
                q,
                k,
                v,
                (self.dim as f32).sqrt().recip(),
                mask,
            )?;
            x = x.add(
                layer
                    .o
                    .forward(&a.transpose_axes(&[0, 2, 1, 3])?.reshape(&[len, -1])?)?,
            )?;
            let h = fast::rms_norm(&x, &layer.post_norm, self.eps)?;
            let gated = nn::silu(layer.gate.forward(&h)?)?.multiply(layer.up.forward(&h)?)?;
            x = x.add(layer.down.forward(&gated)?)?;
        }
        let last = fast::rms_norm(x.index((len - 1..len, ..)), &self.norm, self.eps)?;
        Ok(mlx_rs::ops::indexing::argmax_axis(
            self.head.forward(&last)?,
            -1,
            false,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::positions;
    #[test]
    fn image_and_following_text_have_distinct_positions() {
        let (pos, next) = positions(2, 2, 3, 2);
        assert_eq!(
            pos,
            vec![
                [0; 3],
                [1; 3],
                [2, 2, 2],
                [2, 2, 3],
                [2, 2, 4],
                [2, 3, 2],
                [2, 3, 3],
                [2, 3, 4],
                [5; 3],
                [6; 3]
            ]
        );
        assert_eq!(next, 7);
    }
}
