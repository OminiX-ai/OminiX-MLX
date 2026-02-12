use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    ops::indexing::{IndexOp, take_axis},
    quantization::MaybeQuantized,
    Array,
};
use mlx_rs_core::{
    cache::KVCache,
    utils::{scaled_dot_product_attention, SdpaMask},
};

use crate::config::{ModelArgs, SparseConfig};

// ============================================================================
// SparseKVCache — stores full KV history for InfLLMv2 sparse attention
// ============================================================================

/// KV cache for sparse (InfLLMv2) attention layers.
///
/// Stores the full key/value history and provides methods for:
/// - Dense SDPA when total_len <= dense_len
/// - InfLLMv2 sparse attention when total_len > dense_len
#[derive(Debug)]
pub struct SparseKVCache {
    /// Full key history: [B, n_kv_heads, total_len, head_dim]
    pub keys: Option<Array>,
    /// Full value history: [B, n_kv_heads, total_len, head_dim]
    pub values: Option<Array>,
    pub offset: i32,
    pub config: SparseConfig,
}

impl SparseKVCache {
    pub fn new(config: SparseConfig) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            config,
        }
    }

    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// Append new keys and values to the cache.
    /// keys/values: [B, n_kv_heads, L, head_dim]
    pub fn append(&mut self, keys: Array, values: Array) -> Result<(), Exception> {
        let l = keys.shape()[2];

        self.keys = Some(match self.keys.take() {
            Some(existing) => {
                let refs: Vec<&Array> = vec![&existing, &keys];
                mlx_rs::ops::concatenate_axis(&refs, 2)?
            }
            None => keys,
        });

        self.values = Some(match self.values.take() {
            Some(existing) => {
                let refs: Vec<&Array> = vec![&existing, &values];
                mlx_rs::ops::concatenate_axis(&refs, 2)?
            }
            None => values,
        });

        self.offset += l;
        Ok(())
    }

    /// Get all cached keys: [B, n_kv_heads, total_len, head_dim]
    pub fn all_keys(&self) -> &Array {
        self.keys.as_ref().unwrap()
    }

    /// Get all cached values: [B, n_kv_heads, total_len, head_dim]
    pub fn all_values(&self) -> &Array {
        self.values.as_ref().unwrap()
    }
}

// ============================================================================
// InfLLMv2 Helper Functions
// ============================================================================

/// Compress keys by mean-pooling with non-overlapping windows.
/// Input: [B, H, L, D] → Output: [B, H, num_blocks, D]
/// where num_blocks = L / kernel_size (truncated).
#[allow(non_snake_case)]
fn compress_keys(keys: &Array, kernel_size: i32) -> Result<Array, Exception> {
    let shape = keys.shape();
    let B = shape[0];
    let H = shape[1];
    let L = shape[2];
    let D = shape[3];

    let num_blocks = L / kernel_size;
    let usable_len = num_blocks * kernel_size;

    // Truncate to usable length, reshape, mean-pool
    let truncated = keys.index((.., .., ..usable_len, ..));
    let reshaped = truncated.reshape(&[B, H, num_blocks, kernel_size, D])?;
    mlx_rs::ops::mean_axis(&reshaped, 3, None)
}

/// InfLLMv2 sparse attention for long contexts.
///
/// Two-stage algorithm:
/// 1. CompressK: Mean-pool keys from the "middle" region into block representatives
/// 2. Score queries against compressed keys, select top-K blocks
/// 3. Gather K,V from: init blocks + selected blocks + sliding window
/// 4. Run SDPA on gathered subset
#[allow(non_snake_case)]
fn infllmv2_attention(
    queries: &Array,
    cache: &SparseKVCache,
    n_heads: i32,
    n_kv_heads: i32,
    scale: f32,
) -> Result<Array, Exception> {
    let config = &cache.config;
    let ks = config.kernel_size;
    let total_len = cache.offset;
    let all_keys = cache.all_keys();
    let all_values = cache.all_values();

    let B = queries.shape()[0];
    let L_q = queries.shape()[2];
    let D = queries.shape()[3];

    // Token regions
    let init_end = config.init_blocks * config.block_size;
    let window_start = (total_len - config.window_size).max(init_end);
    let middle_start = init_end;
    let middle_end = window_start;
    let middle_len = middle_end - middle_start;

    if middle_len < ks {
        // Not enough middle tokens for sparse attention, fall back to dense
        let sdpa_mask = if L_q > 1 {
            Some(SdpaMask::Causal)
        } else {
            None
        };
        return scaled_dot_product_attention::<KVCache>(
            queries.clone(),
            all_keys.clone(),
            all_values.clone(),
            None,
            scale,
            sdpa_mask,
        );
    }

    // 1. Compress middle keys: mean pool with kernel_size
    let middle_keys = all_keys.index((.., .., middle_start..middle_end, ..));
    let compressed = compress_keys(&middle_keys, ks)?;
    // compressed: [B, H_kv, num_compressed, D]
    let num_compressed = compressed.shape()[2];

    // 2. Score queries against compressed keys (per KV head group)
    // Use first Q head per KV group for scoring
    let n_rep = n_heads / n_kv_heads;
    let q_for_scoring = if n_rep > 1 {
        // Reshape to [B, n_kv, n_rep, L_q, D], take first rep
        queries
            .reshape(&[B, n_kv_heads, n_rep, L_q, D])?
            .index((.., .., 0..1, .., ..))
            .reshape(&[B, n_kv_heads, L_q, D])?
    } else {
        queries.clone()
    };

    // scores = Q_kv @ compressed^T : [B, H_kv, L_q, num_compressed]
    let c_t = compressed.transpose_axes(&[0, 1, 3, 2])?;
    let scores = mlx_rs::ops::matmul(&q_for_scoring, &c_t)?;

    // 3. Top-K selection: argsort descending, take first topk indices
    let topk = config.topk.min(num_compressed);
    let neg_scores = scores.multiply(array!(-1.0f32))?;
    let sorted_idx = mlx_rs::ops::argsort_axis(&neg_scores, -1)?;
    let top_idx = sorted_idx.index((.., .., .., ..topk));
    // top_idx: [B, H_kv, L_q, topk]

    // Eval to materialize indices before CPU-side index building
    mlx_rs::transforms::eval([&top_idx])?;

    // 4. Gather K,V per KV head
    // For each KV head: build gather indices = init + selected blocks + window
    let init_idx = mlx_rs::ops::arange::<_, i32>(0, init_end, None)?;
    let window_idx = mlx_rs::ops::arange::<_, i32>(window_start, total_len, None)?;
    let offsets = mlx_rs::ops::arange::<_, i32>(0, ks, None)?; // [ks]

    // 4. Gather K,V per batch element and KV head
    let mut batch_gathered_keys = Vec::with_capacity(B as usize);
    let mut batch_gathered_values = Vec::with_capacity(B as usize);

    for b in 0..B {
        let mut gathered_keys_list = Vec::with_capacity(n_kv_heads as usize);
        let mut gathered_values_list = Vec::with_capacity(n_kv_heads as usize);

        for h in 0..n_kv_heads {
            // Block indices for this batch+head, last query position
            let q_pos = L_q - 1;
            let head_block_idx = top_idx
                .index((b, h, q_pos, ..))
                .as_type::<i32>()?; // [topk]

            // Convert block indices to token indices:
            // block b → tokens [middle_start + b * ks .. middle_start + (b+1) * ks]
            let block_starts = head_block_idx
                .multiply(array!(ks))?
                .add(array!(middle_start))?; // [topk]
            let starts_2d = block_starts.reshape(&[topk, 1])?;
            let offsets_2d = offsets.reshape(&[1, ks])?;
            let block_tokens = starts_2d.add(offsets_2d)?.reshape(&[-1])?; // [topk * ks]

            // Concatenate: init + block_tokens + window
            let idx_refs: Vec<&Array> = vec![&init_idx, &block_tokens, &window_idx];
            let gather_idx = mlx_rs::ops::concatenate(&idx_refs)?;

            // Gather keys/values for this batch+head: [total_len, D] → [gathered_len, D]
            let k_h = all_keys.index((b, h, .., ..)); // [total_len, D]
            let v_h = all_values.index((b, h, .., ..)); // [total_len, D]

            let gk = take_axis(&k_h, &gather_idx, 0)?;
            let gv = take_axis(&v_h, &gather_idx, 0)?;

            // [1, gathered_len, D]
            gathered_keys_list.push(gk.reshape(&[1, -1, D])?);
            gathered_values_list.push(gv.reshape(&[1, -1, D])?);
        }

        // Stack heads: [n_kv_heads, gathered_len, D] → [1, n_kv_heads, gathered_len, D]
        let gk_refs: Vec<&Array> = gathered_keys_list.iter().collect();
        let gv_refs: Vec<&Array> = gathered_values_list.iter().collect();
        let batch_k = mlx_rs::ops::concatenate_axis(&gk_refs, 0)?
            .reshape(&[1, n_kv_heads, -1, D])?;
        let batch_v = mlx_rs::ops::concatenate_axis(&gv_refs, 0)?
            .reshape(&[1, n_kv_heads, -1, D])?;

        batch_gathered_keys.push(batch_k);
        batch_gathered_values.push(batch_v);
    }

    // Stack batch: [B, n_kv_heads, gathered_len, D]
    let gathered_keys = if B == 1 {
        batch_gathered_keys.into_iter().next().unwrap()
    } else {
        let gk_refs: Vec<&Array> = batch_gathered_keys.iter().collect();
        mlx_rs::ops::concatenate_axis(&gk_refs, 0)?
    };
    let gathered_values = if B == 1 {
        batch_gathered_values.into_iter().next().unwrap()
    } else {
        let gv_refs: Vec<&Array> = batch_gathered_values.iter().collect();
        mlx_rs::ops::concatenate_axis(&gv_refs, 0)?
    };

    // 5. Run SDPA on gathered K,V (SDPA handles GQA broadcast internally)
    // No causal mask needed since we've already selected relevant blocks
    scaled_dot_product_attention::<KVCache>(
        queries.clone(),
        gathered_keys,
        gathered_values,
        None,
        scale,
        None,
    )
}

// ============================================================================
// SparseAttention
// ============================================================================

/// Sparse attention layer using standard SDPA for short contexts,
/// InfLLMv2 two-stage sparse attention for long contexts.
///
/// Sparse layers do NOT have q_norm/k_norm (those are lightning-only).
/// Sparse layers DO have o_gate for output gating.
#[derive(Debug, ModuleParameters, Quantizable)]
#[module(root = mlx_rs)]
pub struct SparseAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub scale: f32,
    pub use_rope: bool,

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
    #[quantizable]
    #[param]
    pub o_gate: Option<MaybeQuantized<nn::Linear>>,
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
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(dim, n_heads * head_dim)
                    .bias(bias)
                    .build()?,
            ))
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
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            o_gate,
            rope,
        })
    }

    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut SparseKVCache,
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

        // Append K,V to cache
        cache.append(keys, values)?;

        let total_len = cache.offset;
        let dense_len = cache.config.dense_len;

        if total_len <= dense_len {
            // Dense SDPA path for short contexts
            let all_keys = cache.all_keys().clone();
            let all_values = cache.all_values().clone();

            let sdpa_mask = match mask {
                Some(m) => Some(SdpaMask::Array(m)),
                None if L > 1 => Some(SdpaMask::Causal),
                None => None,
            };

            let mut output = scaled_dot_product_attention::<KVCache>(
                queries, all_keys, all_values, None, self.scale, sdpa_mask,
            )?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?;

            if let Some(o_gate) = &mut self.o_gate {
                let gate = nn::sigmoid(o_gate.forward(x)?)?;
                output = output.multiply(gate)?;
            }

            self.o_proj.forward(&output)
        } else {
            // InfLLMv2 sparse attention for long contexts
            let mut output = infllmv2_attention(
                &queries, cache, self.n_heads, self.n_kv_heads, self.scale,
            )?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?;

            if let Some(o_gate) = &mut self.o_gate {
                let gate = nn::sigmoid(o_gate.forward(x)?)?;
                output = output.multiply(gate)?;
            }

            self.o_proj.forward(&output)
        }
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
