//! Qwen2 Decoder-as-Encoder (Visual Causal Flow) for DeepSeek-OCR-2.
//!
//! A 24-layer Qwen2 model used as a vision encoder with mixed attention:
//! - Image tokens: bidirectional attention (non-causal)
//! - Query tokens: causal attention (can attend to all image tokens + prior queries)
//!
//! Input: SAM features [B, HW, 896] concatenated with learnable query embeddings
//! Output: Query token features [B, num_queries, 896]

use std::collections::HashMap;

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp},
    Array, Dtype,
};

use crate::error::Error;

// ============================================================================
// Qwen2 Attention (standard GQA with RoPE)
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Qwen2Attention {
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub o_proj: nn::Linear,
    #[param]
    pub rope: nn::Rope,
}

impl Qwen2Attention {
    /// Forward with custom attention mask.
    /// x: [B, L, D], mask: [B, 1, L, L]
    pub fn forward_with_mask(
        &mut self,
        x: &Array,
        mask: &Array,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let b = shape[0] as i32;
        let l = shape[1] as i32;

        let q = self.q_proj.forward(x)?
            .reshape(&[b, l, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self.k_proj.forward(x)?
            .reshape(&[b, l, self.num_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self.v_proj.forward(x)?
            .reshape(&[b, l, self.num_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE (no cache offset since this is a single-pass encoder)
        let q_input = nn::RopeInputBuilder::new(&q).build()?;
        let q = self.rope.forward(q_input)?;
        let k_input = nn::RopeInputBuilder::new(&k).build()?;
        let k = self.rope.forward(k_input)?;

        // GQA: repeat KV heads if needed
        let k = if self.num_kv_heads < self.num_heads {
            let n_rep = self.num_heads / self.num_kv_heads;
            let _k_shape = k.shape();
            let k = k.reshape(&[b, self.num_kv_heads, 1, l, self.head_dim])?;
            let k = ops::broadcast_to(
                &k,
                &[b, self.num_kv_heads, n_rep, l, self.head_dim],
            )?;
            k.reshape(&[b, self.num_heads, l, self.head_dim])?
        } else {
            k
        };
        let v = if self.num_kv_heads < self.num_heads {
            let n_rep = self.num_heads / self.num_kv_heads;
            let v = v.reshape(&[b, self.num_kv_heads, 1, l, self.head_dim])?;
            let v = ops::broadcast_to(
                &v,
                &[b, self.num_kv_heads, n_rep, l, self.head_dim],
            )?;
            v.reshape(&[b, self.num_heads, l, self.head_dim])?
        } else {
            v
        };

        // Attention with custom mask
        let attn = q.matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?.multiply(mlx_rs::array!(self.scale))?;
        let attn = attn.add(mask)?;
        let attn = ops::softmax_axis(&attn, -1, true)?;
        let out = attn.matmul(&v)?;

        let out = out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, l, -1])?;

        self.o_proj.forward(&out)
    }
}

// ============================================================================
// Qwen2 MLP (SwiGLU)
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Qwen2MLP {
    #[param]
    pub gate_proj: nn::Linear,
    #[param]
    pub up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
}

impl Module<&Array> for Qwen2MLP {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, _: bool) {}

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = mlx_rs_core::fused_swiglu(&up, &gate)?;
        self.down_proj.forward(&activated)
    }
}

// ============================================================================
// Qwen2 Layer
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Qwen2Layer {
    #[param]
    pub self_attn: Qwen2Attention,
    #[param]
    pub mlp: Qwen2MLP,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl Qwen2Layer {
    pub fn forward_with_mask(
        &mut self,
        x: &Array,
        mask: &Array,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward_with_mask(&normed, mask)?;
        let h = x.add(&attn_out)?;

        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        h.add(&mlp_out)
    }
}

// ============================================================================
// Qwen2 Decoder-as-Encoder
// ============================================================================

#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Qwen2Encoder {
    #[param]
    pub layers: Vec<Qwen2Layer>,
    #[param]
    pub norm: nn::RmsNorm,
    #[param]
    pub query_768: Param<Array>,  // [144, 896] for 768px images
    #[param]
    pub query_1024: Param<Array>, // [256, 896] for 1024px images
}

impl Qwen2Encoder {
    /// Forward pass with visual causal flow.
    /// x: [B, C, H, W] from SAM encoder (channels-last: [B, H, W, 896])
    /// Returns: [B, num_queries, 896]
    pub fn forward_vision(&mut self, x: &Array) -> Result<Array, Exception> {
        // Flatten spatial dims: [B, H, W, C] -> [B, H*W, C]
        let shape = x.shape();
        let b = shape[0] as i32;
        let h = shape[1] as i32;
        let w = shape[2] as i32;
        let c = shape[3] as i32;
        let n_image = h * w;
        let x = x.reshape(&[b, n_image, c])?;

        // Select query based on spatial size
        let query = if n_image == 144 {
            &*self.query_768
        } else {
            &*self.query_1024
        };
        let n_query = query.shape()[0] as i32;

        // Expand query for batch: [n_query, C] -> [B, n_query, C]
        let query_batch = ops::broadcast_to(
            &query.reshape(&[1, n_query, c])?,
            &[b, n_query, c],
        )?;

        // Concatenate: [image_tokens | query_tokens]
        let x_combined = ops::concatenate_axis(
            &[&x, &query_batch],
            1,
        )?;

        // Build mixed attention mask:
        // - Image tokens (0..n_image): bidirectional among themselves
        // - Query tokens (n_image..total): causal + can attend to all image tokens
        let mask = build_visual_causal_mask(n_image, n_query, x_combined.dtype())?;
        // mask: [1, 1, total_len, total_len]

        let mut h = x_combined;
        for layer in &mut self.layers {
            h = layer.forward_with_mask(&h, &mask)?;
        }

        let h = self.norm.forward(&h)?;

        // Return only query tokens: [B, n_query, C]
        Ok(h.index((.., n_image.., ..)))
    }
}

/// Build mixed attention mask for visual causal flow.
/// Image tokens: bidirectional, Query tokens: causal + attend to all images.
/// Returns: [1, 1, total, total] mask with 0 for allowed and -inf for masked.
fn build_visual_causal_mask(
    n_image: i32,
    n_query: i32,
    dtype: Dtype,
) -> Result<Array, Exception> {
    let total = n_image + n_query;
    let min_val = if dtype == Dtype::Float32 {
        f32::NEG_INFINITY
    } else {
        -1e9 // Safe value for bf16/f16
    };

    // Build mask on CPU
    let mut mask_vec = vec![min_val; (total * total) as usize];

    // Image tokens: bidirectional (can attend to all image tokens)
    for i in 0..n_image {
        for j in 0..n_image {
            mask_vec[(i * total + j) as usize] = 0.0;
        }
    }

    // Query tokens: can attend to all image tokens + causal among queries
    for i in 0..n_query {
        let qi = n_image + i;
        // Attend to all image tokens
        for j in 0..n_image {
            mask_vec[(qi * total + j) as usize] = 0.0;
        }
        // Causal among query tokens
        for j in 0..=i {
            let qj = n_image + j;
            mask_vec[(qi * total + qj) as usize] = 0.0;
        }
    }

    let mask = Array::from_slice(&mask_vec, &[total, total]);
    // Reshape to [1, 1, total, total] for broadcasting
    mask.reshape(&[1, 1, total, total])
}

// ============================================================================
// Weight loading
// ============================================================================

pub fn load_qwen2_encoder(
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<Qwen2Encoder, Error> {
    let get = |key: &str| -> Result<Array, Error> {
        weights
            .get(key)
            .cloned()
            .ok_or_else(|| Error::Model(format!("Qwen2 weight not found: {}", key)))
    };

    let hidden_dim = 896;
    let num_heads = 14;
    let num_kv_heads = 2;
    let head_dim = hidden_dim / num_heads; // 64
    let _intermediate_size = 4864;
    let rope_theta = 1000000.0f32;
    let rms_eps = 1e-6f32;

    let mut layers = Vec::new();
    for i in 0..24 {
        let lp = format!("{}.layers.{}", prefix, i);

        let rope = nn::RopeBuilder::new(head_dim)
            .base(rope_theta)
            .traditional(false)
            .build()
            .map_err(|e| Error::Model(format!("RoPE build error: {:?}", e)))?;

        let attn = Qwen2Attention {
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            q_proj: nn::Linear {
                weight: Param::new(get(&format!("{}.self_attn.q_proj.weight", lp))?),
                bias: Param::new(Some(get(&format!("{}.self_attn.q_proj.bias", lp))?)),
            },
            k_proj: nn::Linear {
                weight: Param::new(get(&format!("{}.self_attn.k_proj.weight", lp))?),
                bias: Param::new(Some(get(&format!("{}.self_attn.k_proj.bias", lp))?)),
            },
            v_proj: nn::Linear {
                weight: Param::new(get(&format!("{}.self_attn.v_proj.weight", lp))?),
                bias: Param::new(Some(get(&format!("{}.self_attn.v_proj.bias", lp))?)),
            },
            o_proj: nn::Linear {
                weight: Param::new(get(&format!("{}.self_attn.o_proj.weight", lp))?),
                bias: Param::new(None),
            },
            rope,
        };

        let mlp = Qwen2MLP {
            gate_proj: nn::Linear {
                weight: Param::new(get(&format!("{}.mlp.gate_proj.weight", lp))?),
                bias: Param::new(None),
            },
            up_proj: nn::Linear {
                weight: Param::new(get(&format!("{}.mlp.up_proj.weight", lp))?),
                bias: Param::new(None),
            },
            down_proj: nn::Linear {
                weight: Param::new(get(&format!("{}.mlp.down_proj.weight", lp))?),
                bias: Param::new(None),
            },
        };

        let input_layernorm = nn::RmsNorm {
            weight: Param::new(get(&format!("{}.input_layernorm.weight", lp))?),
            eps: rms_eps,
        };

        let post_attention_layernorm = nn::RmsNorm {
            weight: Param::new(get(&format!(
                "{}.post_attention_layernorm.weight",
                lp
            ))?),
            eps: rms_eps,
        };

        layers.push(Qwen2Layer {
            self_attn: attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        });
    }

    let norm = nn::RmsNorm {
        weight: Param::new(get(&format!("{}.norm.weight", prefix))?),
        eps: rms_eps,
    };

    let query_768 = get(&format!("{}.query_768", prefix))?;
    let query_1024 = get(&format!("{}.query_1024", prefix))?;

    Ok(Qwen2Encoder {
        layers,
        norm,
        query_768: Param::new(query_768),
        query_1024: Param::new(query_1024),
    })
}
