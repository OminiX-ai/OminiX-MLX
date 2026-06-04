//! Gemma 4 attention (LayerKind-driven: sliding vs global).
//!
//! Replicates mlx-vlm `Attention.__call__` exactly:
//! - sliding layers: `head_dim=256`, `n_kv_heads=8`, standard full-dim RoPE
//!   (base=10000), with a `v_proj`.
//! - global/full layers: `head_dim=512`, `n_kv_heads=1`, ProportionalRoPE
//!   (base=1e6, partial_rotary_factor=0.25), `use_k_eq_v` so `values` come from
//!   the RAW `k_proj` output (BEFORE `k_norm`) and there is NO `v_proj`.
//!
//! `scale = 1.0` for both. GQA is handled inside SDPA.

use std::collections::HashMap;

use mlx_rs::{
    fast::rope,
    module::Module,
    ops::{concatenate_axis, indexing::IndexOp},
    Array,
};
use mlx_rs_core::{
    cache::{KVCache, KeyValueCache},
    utils::{scaled_dot_product_attention, SdpaMask},
};

use crate::config::{LayerKind, ModelArgs, QuantConfig, RopeType};
use crate::error::Result;
use crate::norm::GemmaRmsNorm;
use crate::weights::{get_weight, make_quantized_linear};

/// RoPE variant for one attention layer.
enum Rope {
    /// Standard full-dim RoPE (sliding layers). `dims=head_dim`, given `base`.
    Standard { dims: i32, base: f32 },
    /// Proportional RoPE (global layers). See `ProportionalRoPE` in mlx-vlm.
    Proportional {
        dims: i32,
        rotated_dims: i32,
        /// `freqs` for `mlx::fast::rope` (base^(+exp), denominator = full head_dim).
        freqs: Array,
    },
}

impl Rope {
    /// Build from a `RopeSpec` for the given `head_dim`.
    fn new(spec: &crate::config::RopeSpec, head_dim: i32) -> Self {
        match spec.rope_type {
            RopeType::Default => Rope::Standard { dims: head_dim, base: spec.theta },
            RopeType::Proportional => {
                // rope_angles = int(partial * dims // 2); rotated_dims = 2*rope_angles.
                let rope_angles = ((spec.partial_rotary_factor * head_dim as f32) as i32) / 2;
                let rotated_dims = 2 * rope_angles;
                // freqs[i] = base ** (arange(0, rotated_dims, 2) / head_dim)  (denominator = full head_dim)
                let n = (rotated_dims / 2) as usize;
                let mut f = Vec::with_capacity(n);
                for i in 0..n {
                    let exp = (2 * i) as f32 / head_dim as f32;
                    f.push(spec.theta.powf(exp));
                }
                let freqs = Array::from_slice(&f, &[n as i32]);
                Rope::Proportional { dims: head_dim, rotated_dims, freqs }
            }
        }
    }

    /// Apply RoPE to `x` of shape `[B, n_heads, L, head_dim]` at `offset`.
    fn forward_at(&self, x: &Array, offset: i32) -> Result<Array> {
        match self {
            Rope::Standard { dims, base } => {
                Ok(rope(x, *dims, false, Some(*base), 1.0, offset, None)?)
            }
            Rope::Proportional { dims, rotated_dims, freqs } => {
                let dims = *dims;
                let rd = *rotated_dims;
                let half = dims / 2;
                let rd_half = rd / 2;

                // tail is empty here (dims == head_dim), so head == x.
                let head = x.index((.., .., .., ..dims));
                let left = head.index((.., .., .., ..half));
                let right = head.index((.., .., .., half..));

                // rotated = concat(left[:rd/2], right[:rd/2])
                let left_front = left.index((.., .., .., ..rd_half));
                let right_front = right.index((.., .., .., ..rd_half));
                let rotated = concatenate_axis(&[&left_front, &right_front], -1)?;

                let rotated = rope(&rotated, rd, false, None, 1.0, offset, Some(freqs))?;

                // left  = concat(rotated[:rd/2], left[rd/2:])
                let new_left = concatenate_axis(
                    &[&rotated.index((.., .., .., ..rd_half)), &left.index((.., .., .., rd_half..))],
                    -1,
                )?;
                // right = concat(rotated[rd/2:], right[rd/2:])
                let new_right = concatenate_axis(
                    &[&rotated.index((.., .., .., rd_half..)), &right.index((.., .., .., rd_half..))],
                    -1,
                )?;
                Ok(concatenate_axis(&[&new_left, &new_right], -1)?)
            }
        }
    }
}

/// Gemma 4 self-attention for a single decoder layer.
pub struct Attention {
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    use_k_eq_v: bool,

    q_proj: mlx_rs::nn::QuantizedLinear,
    k_proj: mlx_rs::nn::QuantizedLinear,
    /// Absent for global (`use_k_eq_v`) layers.
    v_proj: Option<mlx_rs::nn::QuantizedLinear>,
    o_proj: mlx_rs::nn::QuantizedLinear,

    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
    v_norm: GemmaRmsNorm,

    rope: Rope,
}

impl Attention {
    /// Build attention for layer `layer_idx` from pre-loaded weights + config.
    pub fn from_weights(
        weights: &HashMap<String, Array>,
        args: &ModelArgs,
        quant: &QuantConfig,
        layer_idx: i32,
    ) -> Result<Self> {
        let kind = args.layer_types[layer_idx as usize];
        let is_sliding = kind == LayerKind::Sliding;

        let head_dim = if is_sliding { args.head_dim } else { args.global_head_dim };
        let n_heads = args.num_attention_heads;

        let use_k_eq_v = args.attention_k_eq_v && !is_sliding;
        let n_kv_heads = if use_k_eq_v {
            args.num_global_key_value_heads
        } else {
            args.num_key_value_heads
        };

        let base = format!("language_model.model.layers.{layer_idx}.self_attn");
        let load = |name: &str| -> Result<mlx_rs::nn::QuantizedLinear> {
            let prefix = format!("{base}.{name}");
            let (bits, group_size) = quant.quant_for(&prefix);
            make_quantized_linear(weights, &prefix, group_size, bits)
        };

        let q_proj = load("q_proj")?;
        let k_proj = load("k_proj")?;
        let v_proj = if use_k_eq_v { None } else { Some(load("v_proj")?) };
        let o_proj = load("o_proj")?;

        let q_norm = GemmaRmsNorm::from_weight(
            get_weight(weights, &format!("{base}.q_norm.weight"))?,
            args.rms_norm_eps,
        );
        let k_norm = GemmaRmsNorm::from_weight(
            get_weight(weights, &format!("{base}.k_norm.weight"))?,
            args.rms_norm_eps,
        );
        let v_norm = GemmaRmsNorm::new_no_scale(head_dim, args.rms_norm_eps);

        let rope_spec = if is_sliding { &args.rope_sliding } else { &args.rope_global };
        let rope = Rope::new(rope_spec, head_dim);

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale: 1.0,
            use_k_eq_v,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            rope,
        })
    }

    /// Forward (prefill). `x` is `[B, L, hidden]`. `mask` is passed straight to
    /// SDPA: either a bool `[L, L]` (true=visible, e.g. from
    /// `mask::full_causal_mask`) or a float additive `[L, L]` (0=keep, -inf=mask).
    /// The M0 parity example uses the float form to match the golden dump; real
    /// inference uses bool. Thin wrapper over [`Attention::attend`] with offset 0
    /// and no cache — bit-identical to the M1 prefill path.
    pub fn forward(&mut self, x: &Array, mask: &Array) -> Result<Array> {
        self.attend(x, Some(mask), None)
    }

    /// Cache-aware attention core. When `cache` is `None`, behaves exactly like
    /// the M1 prefill path (RoPE at offset 0, no cache). When `cache` is `Some`,
    /// RoPE is applied at `cache.offset()` to q and k, then k/v are appended to
    /// (and re-read from) the cache before SDPA — the canonical decode path.
    #[allow(non_snake_case)]
    pub fn attend(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array> {
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        // RoPE offset: cache's current length (0 for prefill / no cache).
        let offset = cache.as_ref().map(|c| c.offset()).unwrap_or(0);

        // queries = q_norm(q_proj(x).reshape(B,L,n_heads,head_dim))  [norm over last axis]
        let queries = self.q_proj.forward(x)?;
        let queries = queries.reshape(&[B, L, self.n_heads, self.head_dim])?;
        let queries = self.q_norm.forward(&queries)?;

        // keys = k_proj(x).reshape(B,L,n_kv_heads,head_dim)
        let keys = self.k_proj.forward(x)?;
        let keys = keys.reshape(&[B, L, self.n_kv_heads, self.head_dim])?;

        // values: k_eq_v -> RAW keys (before k_norm); else v_proj(x)
        let values = if self.use_k_eq_v {
            keys.clone()
        } else {
            // Invariant: constructor builds v_proj for every non-k_eq_v layer.
            // Return Err rather than panic to keep forward() on the Result contract.
            let v_proj = self.v_proj.as_mut().ok_or_else(|| {
                crate::error::Error::Model("v_proj missing on non-k_eq_v layer".into())
            })?;
            let v = v_proj.forward(x)?;
            v.reshape(&[B, L, self.n_kv_heads, self.head_dim])?
        };

        // keys = rope(k_norm(keys).transpose)
        let keys = self.k_norm.forward(&keys)?;
        let keys = keys.transpose_axes(&[0, 2, 1, 3])?;
        let keys = self.rope.forward_at(&keys, offset)?;

        // values = v_norm(values).transpose  (NO rope)
        let values = self.v_norm.forward(&values)?;
        let values = values.transpose_axes(&[0, 2, 1, 3])?;

        // queries = rope(queries.transpose)
        let queries = queries.transpose_axes(&[0, 2, 1, 3])?;
        let queries = self.rope.forward_at(&queries, offset)?;

        // Append to / re-read from the KV cache for decode; pass-through for prefill.
        let (keys, values) = match cache {
            Some(c) => c.update_and_fetch(keys, values)?,
            None => (keys, values),
        };

        let output = scaled_dot_product_attention::<KVCache>(
            queries,
            keys,
            values,
            None,
            self.scale,
            mask.map(SdpaMask::Array),
        )?;
        let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[B, L, -1])?;

        Ok(self.o_proj.forward(&output)?)
    }
}
