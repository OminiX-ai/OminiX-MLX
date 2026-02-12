use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    ops::indexing::IndexOp,
    Array,
};

use crate::config::ModelArgs;

/// Recurrent state cache for lightning (GLA) attention layers.
#[derive(Debug)]
pub struct LightningCache {
    /// Recurrent state: [1, n_heads, head_dim, head_dim]
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

/// Lightning attention using Gated Linear Attention (GLA) with recurrent state.
#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct LightningAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub use_rope: bool,
    pub use_output_gate: bool,
    pub use_output_norm: bool,

    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub o_proj: nn::Linear,
    #[param]
    pub q_norm: Option<nn::RmsNorm>,
    #[param]
    pub k_norm: Option<nn::RmsNorm>,
    #[param]
    pub o_norm: Option<nn::RmsNorm>,
    #[param]
    pub z_proj: Option<nn::Linear>,
    #[param]
    pub rope: Option<nn::Rope>,

    /// ALiBi decay slopes (not a learned parameter)
    pub decay_slopes: Vec<f32>,
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
            Some(
                nn::LinearBuilder::new(dim, n_heads * head_dim)
                    .bias(bias)
                    .build()?,
            )
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
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            o_norm,
            z_proj,
            rope,
            decay_slopes,
        })
    }

    /// Forward pass using naive GLA recurrent attention.
    /// For decode (L=1): single recurrent step.
    /// For prefill (L>1): loop over tokens.
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

        // RoPE
        if let Some(rope) = &mut self.rope {
            let q_input = nn::RopeInputBuilder::new(&queries)
                .offset(cache.offset)
                .build()?;
            queries = rope.forward(q_input)?;
            let k_input = nn::RopeInputBuilder::new(&keys)
                .offset(cache.offset)
                .build()?;
            keys = rope.forward(k_input)?;
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

        // Build decay tensor: [1, n_heads, 1, 1]
        let decay = Array::from_slice(&self.decay_slopes, &[1, self.n_heads, 1, 1]);

        // Run GLA recurrent: process all tokens
        let output = self.gla_recurrent(&queries, &keys, &values, &decay, cache)?;

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

    /// Naive GLA recurrent attention.
    /// state_{t+1} = exp(decay) * state_t + k_t^T @ v_t
    /// output_t = q_t @ state_t
    ///
    /// q, k, v: [B, n_heads, L, head_dim]
    /// decay: [1, n_heads, 1, 1]
    #[allow(non_snake_case)]
    fn gla_recurrent(
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
        let L = shape[2];
        let D = shape[3];

        // exp(decay) for state decay — decay_slopes are negative, so exp gives < 1
        let decay_factor = mlx_rs::ops::exp(decay)?;

        // Initialize or retrieve state: [B, H, D, D]
        let mut state = match &cache.state {
            Some(s) => s.clone(),
            None => Array::zeros::<f32>(&[B, H, D, D])?,
        };

        let mut outputs = Vec::with_capacity(L as usize);

        for t in 0..L {
            // q_t: [B, H, 1, D]
            let q_t = q.index((.., .., t..t + 1, ..));
            // k_t: [B, H, D, 1]
            let k_t = k.index((.., .., t..t + 1, ..)).transpose_axes(&[0, 1, 3, 2])?;
            // v_t: [B, H, 1, D]
            let v_t = v.index((.., .., t..t + 1, ..));

            // outer product: k_t^T @ v_t -> [B, H, D, D]
            let kv = mlx_rs::ops::matmul(&k_t, &v_t)?;

            // state = decay * state + kv
            state = state.multiply(&decay_factor)?.add(kv)?;

            // output_t = q_t @ state -> [B, H, 1, D]
            let o_t = mlx_rs::ops::matmul(&q_t, &state)?;
            outputs.push(o_t);
        }

        // Update cache
        cache.state = Some(state);
        cache.offset += L as i32;

        // Concat along seq dim: [B, H, L, D]
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
