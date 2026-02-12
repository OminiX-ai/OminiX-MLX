use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    Array,
};
use mlx_rs_core::{
    cache::{KVCache, KeyValueCache},
    utils::{scaled_dot_product_attention, SdpaMask},
};

use crate::config::ModelArgs;

/// Sparse attention layer using standard SDPA as fallback.
/// In the original model this is InfLLMv2; here we use dense attention
/// which is correct for context < dense_len (8192).
///
/// Sparse layers do NOT have q_norm/k_norm (those are lightning-only).
/// Sparse layers DO have o_gate for output gating.
#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct SparseAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub scale: f32,
    pub use_rope: bool,

    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub o_proj: nn::Linear,
    #[param]
    pub o_gate: Option<nn::Linear>,
    #[param]
    pub rope: Option<nn::Rope>,
}

impl SparseAttention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;
        let scale = (head_dim as f32).sqrt().recip();
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

        let o_gate = if args.attn_use_output_gate {
            Some(
                nn::LinearBuilder::new(dim, n_heads * head_dim)
                    .bias(bias)
                    .build()?,
            )
        } else {
            None
        };

        let rope = if args.attn_use_rope {
            Some(
                mlx_rs_core::utils::initialize_rope(
                    head_dim,
                    args.rope_theta,
                    false,
                    &None,
                    args.max_position_embeddings,
                )?,
            )
        } else {
            None
        };

        Ok(Self {
            n_heads,
            n_kv_heads,
            scale,
            use_rope: args.attn_use_rope,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            o_gate,
            rope,
        })
    }

    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KVCache,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape to [B, n_heads, L, head_dim] — no QK norm for sparse layers
        let mut queries = queries
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = keys
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE if configured (attn_use_rope=false for this model)
        if let Some(rope) = &mut self.rope {
            let q_input = nn::RopeInputBuilder::new(&queries)
                .offset(cache.offset())
                .build()?;
            queries = rope.forward(q_input)?;
            let k_input = nn::RopeInputBuilder::new(&keys)
                .offset(cache.offset())
                .build()?;
            keys = rope.forward(k_input)?;
        }

        // Update KV cache
        let (keys, values) = cache.update_and_fetch(keys, values)?;

        // SDPA
        let sdpa_mask = match mask {
            Some(m) => Some(SdpaMask::Array(m)),
            None if L > 1 => Some(SdpaMask::Causal),
            None => None,
        };

        let mut output = scaled_dot_product_attention::<KVCache>(
            queries, keys, values, None, self.scale, sdpa_mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        // Output gating: sigmoid(o_gate(x)) * output
        if let Some(o_gate) = &mut self.o_gate {
            let gate = nn::sigmoid(o_gate.forward(x)?)?;
            output = output.multiply(gate)?;
        }

        self.o_proj.forward(&output)
    }

    pub fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        if let Some(g) = &mut self.o_gate {
            g.training_mode(mode);
        }
    }
}
