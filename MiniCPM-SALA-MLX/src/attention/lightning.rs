use std::cmp::min;

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    ops::indexing::IndexOp,
    quantization::MaybeQuantized,
    Array,
};

use crate::config::ModelArgs;

/// Default chunk size for chunked GLA prefill.
const DEFAULT_CHUNK_SIZE: i32 = 64;

/// Recurrent state cache for lightning (GLA) attention layers.
#[derive(Debug)]
pub struct LightningCache {
    /// Recurrent state: [B, n_heads, head_dim, head_dim]
    pub state: Option<Array>,
    pub n_heads: i32,
    pub head_dim: i32,
    pub offset: i32,
}

impl LightningCache {
    pub fn new(n_heads: i32, head_dim: i32) -> Self {
        Self {
            state: None,
            n_heads,
            head_dim,
            offset: 0,
        }
    }
}

// ============================================================================
// ALiBi Slopes
// ============================================================================

/// Build ALiBi slopes (negated) for GLA decay.
/// These are NOT learnable — they are derived from the number of heads.
fn build_alibi_slopes(n_heads: i32) -> Vec<f32> {
    fn get_slopes_power_of_2(n: i32) -> Vec<f32> {
        let start = 2.0_f32.powf(-(2.0_f32.powf(-((n as f32).log2() - 3.0))));
        let ratio = start;
        (0..n).map(|i| start * ratio.powi(i)).collect()
    }

    fn get_slopes(n: i32) -> Vec<f32> {
        let log2_n = (n as f32).log2();
        if (log2_n - log2_n.floor()).abs() < 1e-6 {
            get_slopes_power_of_2(n)
        } else {
            let closest_pow2 = 2_i32.pow(log2_n.floor() as u32);
            let mut slopes = get_slopes_power_of_2(closest_pow2);
            let extra = get_slopes(2 * closest_pow2);
            for i in (0..extra.len()).step_by(2) {
                if slopes.len() >= n as usize {
                    break;
                }
                slopes.push(extra[i]);
            }
            slopes.truncate(n as usize);
            slopes
        }
    }

    // Negate for decay
    get_slopes(n_heads).into_iter().map(|s| -s).collect()
}

/// Public wrapper for `build_alibi_slopes`, used by quantized loading.
pub fn build_alibi_slopes_pub(n_heads: i32) -> Vec<f32> {
    build_alibi_slopes(n_heads)
}

// ============================================================================
// Decay Tensor Builders (computed on CPU, cached for reuse)
// ============================================================================

/// Build intra-chunk causal decay mask: [1, H, C, C]
/// mask[h, i, j] = exp(slope_h * (i - j)) for j <= i, 0 otherwise
fn build_intra_decay_mask(c: i32, slopes: &[f32]) -> Array {
    let h = slopes.len();
    let c_us = c as usize;
    let mut data = vec![0.0f32; h * c_us * c_us];
    for head in 0..h {
        let s = slopes[head];
        for i in 0..c_us {
            for j in 0..=i {
                data[head * c_us * c_us + i * c_us + j] = (s * (i as f32 - j as f32)).exp();
            }
        }
    }
    Array::from_slice(&data, &[1, h as i32, c, c])
}

/// Build query decay for inter-chunk state lookup: [1, H, C, 1]
/// query_decay[h, t] = exp(slope_h * (t + 1)) for t = 0..C-1
fn build_query_decay(c: i32, slopes: &[f32]) -> Array {
    let h = slopes.len();
    let c_us = c as usize;
    let mut data = vec![0.0f32; h * c_us];
    for head in 0..h {
        let s = slopes[head];
        for t in 0..c_us {
            data[head * c_us + t] = (s * (t as f32 + 1.0)).exp();
        }
    }
    Array::from_slice(&data, &[1, h as i32, c, 1])
}

/// Build reverse decay for key weighting in state update: [1, H, C, 1]
/// reverse_decay[h, t] = exp(slope_h * (C - 1 - t)) for t = 0..C-1
fn build_reverse_decay(c: i32, slopes: &[f32]) -> Array {
    let h = slopes.len();
    let c_us = c as usize;
    let mut data = vec![0.0f32; h * c_us];
    for head in 0..h {
        let s = slopes[head];
        for t in 0..c_us {
            data[head * c_us + t] = (s * (c_us as f32 - 1.0 - t as f32)).exp();
        }
    }
    Array::from_slice(&data, &[1, h as i32, c, 1])
}

/// Build chunk decay factor for state propagation: [1, H, 1, 1]
/// chunk_decay[h] = exp(slope_h * C)
fn build_chunk_decay(c: i32, slopes: &[f32]) -> Array {
    let scaled: Vec<f32> = slopes.iter().map(|&s| (s * c as f32).exp()).collect();
    Array::from_slice(&scaled, &[1, slopes.len() as i32, 1, 1])
}

/// Build all four decay tensors for a given chunk size.
fn build_decay_tensors(c: i32, slopes: &[f32]) -> (Array, Array, Array, Array) {
    (
        build_intra_decay_mask(c, slopes),
        build_query_decay(c, slopes),
        build_reverse_decay(c, slopes),
        build_chunk_decay(c, slopes),
    )
}

/// Zero-pad a 4D tensor along axis 2 (sequence dimension).
/// Input: [B, H, L, D] → Output: [B, H, L + pad, D]
#[allow(non_snake_case)]
fn pad_seq_dim(x: &Array, pad: i32) -> Result<Array, Exception> {
    let shape = x.shape();
    let zeros = Array::zeros::<f32>(&[shape[0], shape[1], pad, shape[3]])?;
    let refs: Vec<&Array> = vec![x, &zeros];
    mlx_rs::ops::concatenate_axis(&refs, 2)
}

// ============================================================================
// Lightning Attention
// ============================================================================

/// Lightning attention using Gated Linear Attention (GLA) with recurrent state.
///
/// Uses chunked prefill for L > 1 (batched matmul within chunks of size C)
/// and single-step recurrence for decode (L = 1).
#[derive(Debug, ModuleParameters, Quantizable)]
#[module(root = mlx_rs)]
pub struct LightningAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub use_rope: bool,
    pub use_output_gate: bool,
    pub use_output_norm: bool,
    pub chunk_size: i32,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub q_norm: Option<nn::RmsNorm>,
    #[param]
    pub k_norm: Option<nn::RmsNorm>,
    #[param]
    pub o_norm: Option<nn::RmsNorm>,
    #[quantizable]
    #[param]
    pub z_proj: Option<MaybeQuantized<nn::Linear>>,
    #[param]
    pub rope: Option<nn::Rope>,

    /// ALiBi decay slopes (not a learned parameter)
    pub decay_slopes: Vec<f32>,
    /// Cached decay tensors for chunked prefill (lazily initialized)
    pub intra_decay_mask: Option<Array>,
    pub query_decay: Option<Array>,
    pub reverse_decay: Option<Array>,
    pub chunk_decay: Option<Array>,
}

impl LightningAttention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.lightning_num_heads();
        let n_kv_heads = args.lightning_num_kv_heads();
        let head_dim = args.lightning_head_dim();
        let scale = args.lightning_scale_value();
        let bias = args.attention_bias;

        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(bias)
            .build()?;

        let q_norm = if args.qk_norm {
            Some(
                nn::RmsNormBuilder::new(head_dim)
                    .eps(args.rms_norm_eps)
                    .build()?,
            )
        } else {
            None
        };
        let k_norm = if args.qk_norm {
            Some(
                nn::RmsNormBuilder::new(head_dim)
                    .eps(args.rms_norm_eps)
                    .build()?,
            )
        } else {
            None
        };

        let o_norm = if args.use_output_norm {
            Some(
                nn::RmsNormBuilder::new(n_heads * head_dim)
                    .eps(args.rms_norm_eps)
                    .build()?,
            )
        } else {
            None
        };

        let z_proj = if args.use_output_gate {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(dim, n_heads * head_dim)
                    .bias(bias)
                    .build()?,
            ))
        } else {
            None
        };

        let rope = if args.lightning_use_rope {
            Some(mlx_rs_core::utils::initialize_rope(
                head_dim,
                args.rope_theta,
                false,
                &None,
                args.max_position_embeddings,
            )?)
        } else {
            None
        };

        let decay_slopes = build_alibi_slopes(n_heads);

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            use_rope: args.lightning_use_rope,
            use_output_gate: args.use_output_gate,
            use_output_norm: args.use_output_norm,
            chunk_size: DEFAULT_CHUNK_SIZE,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            q_norm,
            k_norm,
            o_norm,
            z_proj,
            rope,
            decay_slopes,
            intra_decay_mask: None,
            query_decay: None,
            reverse_decay: None,
            chunk_decay: None,
        })
    }

    /// Ensure cached decay tensors are built for the configured chunk_size.
    fn ensure_decay_tensors(&mut self) {
        if self.intra_decay_mask.is_none() {
            let (mask, q_decay, r_decay, c_decay) =
                build_decay_tensors(self.chunk_size, &self.decay_slopes);
            self.intra_decay_mask = Some(mask);
            self.query_decay = Some(q_decay);
            self.reverse_decay = Some(r_decay);
            self.chunk_decay = Some(c_decay);
        }
    }

    /// Forward pass: chunked GLA for prefill (L>1), single recurrent step for decode (L=1).
    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        x: &Array,
        cache: &mut LightningCache,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape to [B, n_heads, L, head_dim]
        let mut queries = queries
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = keys
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // QK norm
        if let Some(qn) = &mut self.q_norm {
            queries = qn.forward(&queries)?;
        }
        if let Some(kn) = &mut self.k_norm {
            keys = kn.forward(&keys)?;
        }

        // RoPE with workaround for MLX fast::rope bug (B>1 && L=1 produces
        // different rotations for different batch elements). Fix: merge B into H
        // dimension so B=1, apply RoPE, then reshape back.
        if let Some(rope) = &mut self.rope {
            if B > 1 && L == 1 {
                let qh = queries.shape()[1]; // n_heads
                let kh = keys.shape()[1]; // n_kv_heads
                let d = queries.shape()[3]; // head_dim

                let q_flat = queries.reshape(&[1, B * qh, L, d])?;
                let q_input = nn::RopeInputBuilder::new(&q_flat)
                    .offset(cache.offset)
                    .build()?;
                queries = rope.forward(q_input)?.reshape(&[B, qh, L, d])?;

                let k_flat = keys.reshape(&[1, B * kh, L, d])?;
                let k_input = nn::RopeInputBuilder::new(&k_flat)
                    .offset(cache.offset)
                    .build()?;
                keys = rope.forward(k_input)?.reshape(&[B, kh, L, d])?;
            } else {
                let q_input = nn::RopeInputBuilder::new(&queries)
                    .offset(cache.offset)
                    .build()?;
                queries = rope.forward(q_input)?;
                let k_input = nn::RopeInputBuilder::new(&keys)
                    .offset(cache.offset)
                    .build()?;
                keys = rope.forward(k_input)?;
            }
        }

        // Repeat KV heads if GQA (lightning uses n_heads == n_kv_heads typically)
        let n_rep = self.n_heads / self.n_kv_heads;
        let keys = if n_rep > 1 {
            let ks = keys.shape().to_vec();
            let expanded = keys.reshape(&[ks[0], ks[1], 1, ks[2], ks[3]])?;
            let expanded =
                mlx_rs::ops::broadcast_to(&expanded, &[ks[0], ks[1], n_rep, ks[2], ks[3]])?;
            expanded.reshape(&[ks[0], ks[1] * n_rep, ks[2], ks[3]])?
        } else {
            keys
        };
        let values = if n_rep > 1 {
            let vs = values.shape().to_vec();
            let expanded = values.reshape(&[vs[0], vs[1], 1, vs[2], vs[3]])?;
            let expanded =
                mlx_rs::ops::broadcast_to(&expanded, &[vs[0], vs[1], n_rep, vs[2], vs[3]])?;
            expanded.reshape(&[vs[0], vs[1] * n_rep, vs[2], vs[3]])?
        } else {
            values
        };

        // Apply scale
        let queries = queries.multiply(array!(self.scale))?;

        // Dispatch: chunked prefill vs single-step decode
        let output = if L == 1 {
            // Decode: single recurrent step
            let decay = Array::from_slice(&self.decay_slopes, &[1, self.n_heads, 1, 1]);
            self.gla_recurrent_step(&queries, &keys, &values, &decay, cache)?
        } else {
            // Prefill: chunked GLA
            self.gla_chunked(&queries, &keys, &values, cache)?
        };

        // output: [B, n_heads, L, head_dim] -> [B, L, n_heads * head_dim]
        let mut output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?;

        // Output normalization
        if let Some(on) = &mut self.o_norm {
            output = on.forward(&output)?;
        }

        // Output gating: sigmoid(z_proj(x)) * output
        if let Some(zp) = &mut self.z_proj {
            let gate = nn::sigmoid(zp.forward(x)?)?;
            output = output.multiply(gate)?;
        }

        self.o_proj.forward(&output)
    }

    /// Single recurrent step for decode (L=1).
    /// state = exp(decay) * state + k^T @ v
    /// output = q @ state
    #[allow(non_snake_case)]
    fn gla_recurrent_step(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        decay: &Array,
        cache: &mut LightningCache,
    ) -> Result<Array, Exception> {
        let shape = q.shape();
        let B = shape[0];
        let H = shape[1];
        let D = shape[3];

        let decay_factor = mlx_rs::ops::exp(decay)?;

        let mut state = match &cache.state {
            Some(s) => s.clone(),
            None => Array::zeros::<f32>(&[B, H, D, D])?,
        };

        // k: [B, H, 1, D] -> k_t: [B, H, D, 1]
        let k_t = k.transpose_axes(&[0, 1, 3, 2])?;
        // kv: [B, H, D, D]
        let kv = mlx_rs::ops::matmul(&k_t, v)?;
        // state = decay * state + kv
        state = state.multiply(&decay_factor)?.add(kv)?;

        // output = q @ state: [B, H, 1, D]
        let output = mlx_rs::ops::matmul(q, &state)?;

        cache.state = Some(state);
        cache.offset += 1;

        Ok(output)
    }

    /// Chunked GLA prefill for L > 1.
    ///
    /// Splits Q/K/V into chunks of size C and processes each with:
    /// - Intra-chunk: quadratic attention with decay mask  (Q_c @ K_c^T) * mask @ V_c
    /// - Inter-chunk: query against accumulated state       Q_c_scaled @ state
    /// - State update: state = chunk_decay * state + K_weighted^T @ V_c
    #[allow(non_snake_case)]
    fn gla_chunked(
        &mut self,
        q: &Array,
        k: &Array,
        v: &Array,
        cache: &mut LightningCache,
    ) -> Result<Array, Exception> {
        let shape = q.shape();
        let B = shape[0];
        let H = shape[1];
        let L = shape[2];
        let D = shape[3];
        let C = self.chunk_size;

        // Ensure cached decay tensors exist
        self.ensure_decay_tensors();

        let mut state = match cache.state.take() {
            Some(s) => s,
            None => Array::zeros::<f32>(&[B, H, D, D])?,
        };

        let num_chunks = (L + C - 1) / C;
        let mut outputs = Vec::with_capacity(num_chunks as usize);

        // Cached decay tensors are always for full chunk size C
        let mask = self.intra_decay_mask.as_ref().unwrap().clone();
        let q_decay = self.query_decay.as_ref().unwrap().clone();
        let r_decay = self.reverse_decay.as_ref().unwrap().clone();
        let c_decay = self.chunk_decay.as_ref().unwrap().clone();

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * C;
            let end = min((chunk_idx + 1) * C, L);
            let actual_c = end - start;

            // Slice chunk: [B, H, actual_c, D]
            let q_c = q.index((.., .., start..end, ..));
            let k_c = k.index((.., .., start..end, ..));
            let v_c = v.index((.., .., start..end, ..));

            // For partial last chunk, zero-pad to full C so the fused kernel
            // can use fixed template parameters (avoids recompilation).
            let (q_pad, k_pad, v_pad) = if actual_c < C {
                (
                    pad_seq_dim(&q_c, C - actual_c)?,
                    pad_seq_dim(&k_c, C - actual_c)?,
                    pad_seq_dim(&v_c, C - actual_c)?,
                )
            } else {
                (q_c.clone(), k_c.clone(), v_c.clone())
            };

            // 1. Fused intra-chunk attention (was 4 ops, now 1 Metal kernel)
            let intra_out = crate::metal_kernels::fused_intra_chunk_attn(
                &q_pad, &k_pad, &v_pad, &mask, B, H, C, D,
            )?;

            // 2. Inter-chunk: Q_c_scaled @ state (standard MLX ops, already optimized)
            let q_scaled = q_pad.multiply(&q_decay)?;
            let inter_out = mlx_rs::ops::matmul(&q_scaled, &state)?;

            // 3. Combine and trim to actual_c if padded
            let combined = intra_out.add(inter_out)?;
            let out_c = if actual_c < C {
                combined.index((.., .., ..actual_c, ..))
            } else {
                combined
            };
            outputs.push(out_c);

            // 4. Fused state update (was 4 ops, now 1 Metal kernel)
            state = crate::metal_kernels::fused_state_update(
                &k_pad, &v_pad, &state, &r_decay, &c_decay, B, H, C, D,
            )?;
        }

        cache.state = Some(state);
        cache.offset += L as i32;

        // Concat chunk outputs: [B, H, L, D]
        if outputs.len() == 1 {
            Ok(outputs.into_iter().next().unwrap())
        } else {
            let refs: Vec<&Array> = outputs.iter().collect();
            mlx_rs::ops::concatenate_axis(&refs, 2)
        }
    }

    pub fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        if let Some(n) = &mut self.q_norm {
            n.training_mode(mode);
        }
        if let Some(n) = &mut self.k_norm {
            n.training_mode(mode);
        }
        if let Some(n) = &mut self.o_norm {
            n.training_mode(mode);
        }
        if let Some(z) = &mut self.z_proj {
            z.training_mode(mode);
        }
    }
}
