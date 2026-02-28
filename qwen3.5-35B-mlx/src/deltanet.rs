use mlx_rs::{
    array,
    error::Exception,
    module::{Module, Param},
    nn,
    ops::{
        broadcast_to, concatenate_axis,
        indexing::IndexOp,
        zeros_dtype,
    },
    quantization::MaybeQuantized,
    transforms::async_eval,
    Array,
};

use crate::cache::RecurrentState;

/// Async eval interval for prefill recurrence (in time steps).
///
/// Controls graph depth vs pipeline stalls: smaller values reduce the lazy
/// computation graph depth (preventing compilation overhead on long sequences)
/// but add more async_eval dispatch overhead. Empirically optimal at 8 for
/// the 48-head, 128-dim state configuration. Benchmarks on 257 tokens:
/// interval 4 → 2.46s, 8 → 2.38s, 16 → 2.65s, none → 2.86s.
const EVAL_INTERVAL: i32 = 8;

/// Gated DeltaNet — linear attention with a fixed-size recurrent state.
///
/// Uses the delta rule to maintain a [num_v_heads, k_dim, v_dim] state matrix
/// that serves as a compressed memory, replacing the growing KV cache.
pub struct GatedDeltaNet {
    pub in_proj_qkv: MaybeQuantized<nn::Linear>, // hidden → key_dim*2 + value_dim
    pub in_proj_z: MaybeQuantized<nn::Linear>,    // hidden → value_dim (output gate)
    pub in_proj_a: MaybeQuantized<nn::Linear>,    // hidden → num_v_heads (alpha/decay)
    pub in_proj_b: MaybeQuantized<nn::Linear>,    // hidden → num_v_heads (beta/update)
    pub conv1d_weight: Param<Array>,              // [conv_dim, 1, kernel_size]
    pub a_log: Param<Array>,                      // [num_v_heads]
    pub dt_bias: Param<Array>,                    // [num_v_heads]
    pub norm: nn::RmsNorm,                        // weight shape [value_head_dim]
    pub out_proj: MaybeQuantized<nn::Linear>,     // value_dim → hidden

    // Dimensions
    pub num_k_heads: i32,
    pub num_v_heads: i32,
    pub key_head_dim: i32,
    pub value_head_dim: i32,
    pub key_dim: i32,   // num_k_heads * key_head_dim
    pub value_dim: i32, // num_v_heads * value_head_dim
    pub conv_dim: i32,  // key_dim * 2 + value_dim
    pub conv_kernel_size: i32,
}

/// L2 normalize the last dimension of an array.
fn l2_normalize(x: &Array, eps: f32) -> Result<Array, Exception> {
    let norm_sq = x.square()?.sum_axis(-1, true)?;
    let norm = norm_sq.add(array!(eps))?.sqrt()?;
    x.divide(&norm)
}

impl GatedDeltaNet {
    /// Process a single token through the DeltaNet layer (decode step).
    #[allow(non_snake_case)]
    pub fn forward_step(
        &mut self,
        x: &Array,
        cache: &mut RecurrentState,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let B = shape[0];
        // x: [B, 1, hidden]

        // 1. Project
        let qkv = self.in_proj_qkv.forward(x)?; // [B, 1, conv_dim]
        let z = self.in_proj_z.forward(x)?;      // [B, 1, value_dim]
        let a = self.in_proj_a.forward(x)?;      // [B, 1, num_v_heads]
        let b = self.in_proj_b.forward(x)?;      // [B, 1, num_v_heads]

        // 2. Causal Conv1d update
        // qkv: [B, 1, conv_dim] → [B, conv_dim, 1]
        let qkv_cf = qkv.transpose_axes(&[0, 2, 1])?;
        let qkv_after_conv = self.conv1d_step(&qkv_cf, cache)?; // [B, 1, conv_dim]

        // 3. Split into Q, K, V
        let q = qkv_after_conv.index((.., .., ..self.key_dim));
        let k = qkv_after_conv.index((.., .., self.key_dim..self.key_dim * 2));
        let v = qkv_after_conv.index((.., .., self.key_dim * 2..));

        // Reshape to heads: Q,K [B, 1, num_k_heads, key_head_dim], V [B, 1, num_v_heads, value_head_dim]
        let q = q.reshape(&[B, 1, self.num_k_heads, self.key_head_dim])?;
        let k = k.reshape(&[B, 1, self.num_k_heads, self.key_head_dim])?;
        let v = v.reshape(&[B, 1, self.num_v_heads, self.value_head_dim])?;

        // Reshape z: [B, 1, num_v_heads, value_head_dim]
        let z = z.reshape(&[B, 1, self.num_v_heads, self.value_head_dim])?;

        // 4. L2 normalize Q, K
        let q = l2_normalize(&q, 1e-6)?;
        let k = l2_normalize(&k, 1e-6)?;

        // 5. Scale Q
        let scale = 1.0 / (self.key_head_dim as f32).sqrt();
        let q = q.multiply(array!(scale))?;

        // 6. Expand Q, K from num_k_heads to num_v_heads via repeat_interleave
        let ratio = self.num_v_heads / self.num_k_heads;
        let q = self.repeat_interleave_heads(&q, ratio)?;
        let k = self.repeat_interleave_heads(&k, ratio)?;
        // Now Q, K: [B, 1, num_v_heads, key_head_dim]

        // 7. Compute gates
        let a_squeezed = a.reshape(&[B, self.num_v_heads])?; // [B, num_v_heads]
        let b_squeezed = b.reshape(&[B, self.num_v_heads])?;
        let beta = nn::sigmoid(b_squeezed)?; // [B, num_v_heads]
        let g = self.compute_decay(&a_squeezed)?; // [B, num_v_heads] (negative values)

        // 8. Squeeze seq dim: Q,K [B, num_v_heads, key_head_dim], V [B, num_v_heads, value_head_dim]
        let q = q.reshape(&[B, self.num_v_heads, self.key_head_dim])?;
        let k = k.reshape(&[B, self.num_v_heads, self.key_head_dim])?;
        let v = v.reshape(&[B, self.num_v_heads, self.value_head_dim])?;

        // 9. Recurrent state update
        let output = self.recurrent_step(&q, &k, &v, &g, &beta, cache)?;
        // output: [B, num_v_heads, value_head_dim]

        // 10. Gated RMSNorm: norm(output) * silu(z)
        let output = output.reshape(&[B, 1, self.num_v_heads, self.value_head_dim])?;
        let z_gate = nn::silu(z)?;
        let gated = self.norm.forward(&output)?.multiply(z_gate)?;
        let flat = gated.reshape(&[B, 1, self.value_dim])?;

        // 11. Output projection
        self.out_proj.forward(&flat)
    }

    /// Process a full sequence through the DeltaNet layer (prefill).
    ///
    /// Optimized sequential recurrence: projections are batched (parallel),
    /// then the delta-rule state update runs per-step with matmul-based
    /// ops and periodic async_eval to limit graph depth.
    #[allow(non_snake_case)]
    pub fn forward_prefill(
        &mut self,
        x: &Array,
        cache: &mut RecurrentState,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        // 1. Project full sequence (parallel)
        let qkv = self.in_proj_qkv.forward(x)?; // [B, L, conv_dim]
        let z = self.in_proj_z.forward(x)?;      // [B, L, value_dim]
        let a = self.in_proj_a.forward(x)?;      // [B, L, num_v_heads]
        let b = self.in_proj_b.forward(x)?;      // [B, L, num_v_heads]

        // 2. Causal Conv1d on full sequence (parallel)
        let qkv_cf = qkv.transpose_axes(&[0, 2, 1])?; // [B, conv_dim, L]
        let qkv_after_conv = self.conv1d_prefill(&qkv_cf, cache, B, L)?; // [B, L, conv_dim]

        // 3. Split into Q, K, V
        let q_flat = qkv_after_conv.index((.., .., ..self.key_dim));
        let k_flat = qkv_after_conv.index((.., .., self.key_dim..self.key_dim * 2));
        let v_flat = qkv_after_conv.index((.., .., self.key_dim * 2..));

        let q = q_flat.reshape(&[B, L, self.num_k_heads, self.key_head_dim])?;
        let k = k_flat.reshape(&[B, L, self.num_k_heads, self.key_head_dim])?;
        let v = v_flat.reshape(&[B, L, self.num_v_heads, self.value_head_dim])?;
        let z = z.reshape(&[B, L, self.num_v_heads, self.value_head_dim])?;

        // 4. L2 normalize Q, K
        let q = l2_normalize(&q, 1e-6)?;
        let k = l2_normalize(&k, 1e-6)?;

        // 5. Scale Q
        let scale = 1.0 / (self.key_head_dim as f32).sqrt();
        let q = q.multiply(array!(scale))?;

        // 6. Expand Q, K from num_k_heads to num_v_heads
        let ratio = self.num_v_heads / self.num_k_heads;
        let q = self.repeat_interleave_heads(&q, ratio)?;
        let k = self.repeat_interleave_heads(&k, ratio)?;

        // 7. Compute gates for all positions
        let beta = nn::sigmoid(b)?; // [B, L, num_v_heads]
        let g = self.compute_decay_batched(&a)?; // [B, L, num_v_heads]

        // ===== OPTIMIZED SEQUENTIAL RECURRENCE =====
        // Pre-cast to float32 and pre-transpose to [B, H, L, dim] ONCE
        // (eliminates 5 dtype casts + multiple reshapes per step)
        let H = self.num_v_heads;
        let K_dim = self.key_head_dim;
        let V_dim = self.value_head_dim;

        let q = q.as_dtype(mlx_rs::Dtype::Float32)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.as_dtype(mlx_rs::Dtype::Float32)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.as_dtype(mlx_rs::Dtype::Float32)?.transpose_axes(&[0, 2, 1, 3])?;
        let g = g.as_dtype(mlx_rs::Dtype::Float32)?.transpose_axes(&[0, 2, 1])?;
        let beta = beta.as_dtype(mlx_rs::Dtype::Float32)?.transpose_axes(&[0, 2, 1])?;
        // q, k: [B, H, L, K], v: [B, H, L, V], g, beta: [B, H, L]

        // 8. Initialize state
        let state = match cache.state.take() {
            Some(s) => s,
            None => zeros_dtype(&[B, H, K_dim, V_dim], mlx_rs::Dtype::Float32)?,
        };

        // 9. Sequential recurrent loop (optimized: column vectors + matmul)
        // Pre-compute all per-step values as column/row vectors
        let decay_all = g.exp()?.reshape(&[B, H, L, 1, 1])?;     // [B, H, L, 1, 1]
        let k_col_all = k.reshape(&[B, H, L, K_dim, 1])?;         // [B, H, L, K, 1]
        let q_col_all = q.reshape(&[B, H, L, K_dim, 1])?;         // [B, H, L, K, 1]
        let v_col_all = v.reshape(&[B, H, L, V_dim, 1])?;         // [B, H, L, V, 1]
        let beta_col_all = beta.reshape(&[B, H, L, 1, 1])?;       // [B, H, L, 1, 1]
        let k_row_all = k_col_all.transpose_axes(&[0, 1, 2, 4, 3])?; // [B, H, L, 1, K]

        // State in [B, H, V, K] layout for matmul
        let mut state_t = state.transpose_axes(&[0, 1, 3, 2])?;

        let mut outputs = Vec::with_capacity(L as usize);
        for t in 0..L {
            let decay_t = decay_all.index((.., .., t, .., ..));     // [B, H, 1, 1]
            let k_t = k_col_all.index((.., .., t, .., ..));         // [B, H, K, 1]
            let v_col = v_col_all.index((.., .., t, .., ..));       // [B, H, V, 1]
            let beta_t = beta_col_all.index((.., .., t, .., ..));   // [B, H, 1, 1]
            let q_t = q_col_all.index((.., .., t, .., ..));         // [B, H, K, 1]
            let k_row = k_row_all.index((.., .., t, .., ..));       // [B, H, 1, K]

            // 1. Decay (1 op)
            state_t = state_t.multiply(&decay_t)?;

            // 2. kv_mem = state_t @ k_t (1 matmul)
            let kv_mem = state_t.matmul(&k_t)?;

            // 3. delta = (v - kv_mem) * beta (2 ops)
            let delta = v_col.subtract(&kv_mem)?.multiply(&beta_t)?;

            // 4. state += delta @ k_row (1 matmul + 1 add)
            state_t = state_t.add(delta.matmul(&k_row)?)?;

            // 5. out = state_t @ q_t (1 matmul)
            outputs.push(state_t.matmul(&q_t)?);

            // Async eval periodically to limit graph depth
            if L > EVAL_INTERVAL && (t + 1) % EVAL_INTERVAL == 0 {
                async_eval([&state_t])?;
            }
        }

        // Convert state_t back to [B, H, K, V] for cache
        let state = state_t.transpose_axes(&[0, 1, 3, 2])?;

        cache.state = Some(state);
        cache.step += L;

        // 10. Stack outputs: [B, H, L, V, 1] → [B, L, H, V]
        let output = mlx_rs::ops::stack_axis(&outputs, 2)?;
        let output = output.reshape(&[B, H, L, V_dim])?;
        let output = output.transpose_axes(&[0, 2, 1, 3])?;

        // 16. Gated RMSNorm + output projection
        let normed = self.norm.forward(&output)?;
        let z_gate = nn::silu(z)?;
        let gated = normed.multiply(z_gate)?;
        let flat = gated.reshape(&[B, L, self.value_dim])?;

        self.out_proj.forward(&flat)
    }

    /// Causal conv1d for a single timestep (decode).
    fn conv1d_step(
        &self,
        qkv_cf: &Array, // [B, conv_dim, 1]
        cache: &mut RecurrentState,
    ) -> Result<Array, Exception> {
        let kernel_size = self.conv_kernel_size;

        // Get or initialize conv state
        let conv_state = match cache.conv_state.take() {
            Some(state) => state,
            None => {
                let shape = qkv_cf.shape();
                zeros_dtype(&[shape[0], self.conv_dim, kernel_size - 1], qkv_cf.dtype())?
            }
        };

        // Concatenate: [B, conv_dim, kernel_size-1] + [B, conv_dim, 1] → [B, conv_dim, kernel_size]
        let combined = concatenate_axis(&[&conv_state, qkv_cf], -1)?;

        // Update conv state: last kernel_size-1 elements
        cache.conv_state = Some(combined.index((.., .., 1..)));

        // Depthwise conv: weight [conv_dim, 1, kernel_size] → [conv_dim, kernel_size]
        let w = self.conv1d_weight.as_ref().reshape(&[self.conv_dim, kernel_size])?;
        // combined [B, conv_dim, kernel_size] * w [conv_dim, kernel_size] → sum over last dim
        let out = combined.multiply(&w)?.sum_axis(-1, false)?; // [B, conv_dim]
        let out = nn::silu(out)?;

        // [B, conv_dim] → [B, 1, conv_dim]
        out.reshape(&[-1, 1, self.conv_dim])
    }

    /// Causal conv1d for full sequence (prefill).
    #[allow(non_snake_case)]
    fn conv1d_prefill(
        &self,
        qkv_cf: &Array, // [B, conv_dim, L]
        cache: &mut RecurrentState,
        B: i32,
        L: i32,
    ) -> Result<Array, Exception> {
        let kernel_size = self.conv_kernel_size;

        // Pad left with zeros: [B, conv_dim, kernel_size-1 + L]
        let zero_pad = zeros_dtype(
            &[B, self.conv_dim, kernel_size - 1],
            qkv_cf.dtype(),
        )?;
        let padded = concatenate_axis(&[&zero_pad, qkv_cf], -1)?;

        // Save conv state: last kernel_size-1 elements of raw qkv
        cache.conv_state = Some(qkv_cf.index((.., .., -(kernel_size - 1)..)));

        // Apply depthwise conv using kernel tap loop (kernel_size=4 iterations)
        let w = self.conv1d_weight.as_ref().reshape(&[self.conv_dim, kernel_size])?;
        let mut result = zeros_dtype(&[B, self.conv_dim, L], qkv_cf.dtype())?;
        for tap in 0..kernel_size {
            let window = padded.index((.., .., tap..tap + L));
            let wk = w.index((.., tap..tap + 1)); // [conv_dim, 1]
            result = result.add(window.multiply(&wk)?)?;
        }
        let result = nn::silu(result)?;

        // [B, conv_dim, L] → [B, L, conv_dim]
        result.transpose_axes(&[0, 2, 1])
    }

    /// Single recurrent step of the delta rule.
    ///
    /// Updates the state matrix and returns the query output.
    #[allow(non_snake_case)]
    fn recurrent_step(
        &self,
        q: &Array,    // [B, num_v_heads, key_head_dim]
        k: &Array,    // [B, num_v_heads, key_head_dim]
        v: &Array,    // [B, num_v_heads, value_head_dim]
        g: &Array,    // [B, num_v_heads] (decay, negative values)
        beta: &Array, // [B, num_v_heads] (update gate, 0-1)
        cache: &mut RecurrentState,
    ) -> Result<Array, Exception> {
        let B = q.shape()[0];

        // Get or initialize state: [B, num_v_heads, key_head_dim, value_head_dim]
        let mut state = match cache.state.take() {
            Some(s) => s,
            None => zeros_dtype(
                &[B, self.num_v_heads, self.key_head_dim, self.value_head_dim],
                mlx_rs::Dtype::Float32,
            )?,
        };

        // Cast to float32 for numerical stability
        let q = q.as_dtype(mlx_rs::Dtype::Float32)?;
        let k = k.as_dtype(mlx_rs::Dtype::Float32)?;
        let v = v.as_dtype(mlx_rs::Dtype::Float32)?;
        let g = g.as_dtype(mlx_rs::Dtype::Float32)?;
        let beta = beta.as_dtype(mlx_rs::Dtype::Float32)?;

        // Decay: state *= exp(g)
        // g: [B, num_v_heads] → [B, num_v_heads, 1, 1]
        let decay = g.exp()?.reshape(&[B, self.num_v_heads, 1, 1])?;
        state = state.multiply(&decay)?;

        // kv_mem = einsum('bhkv,bhk->bhv', state, k)
        // k: [B, H, K] → [B, H, K, 1]
        let k_expanded = k.reshape(&[B, self.num_v_heads, self.key_head_dim, 1])?;
        let kv_mem = state.multiply(&k_expanded)?.sum_axis(-2, false)?; // [B, H, V]

        // delta = (v - kv_mem) * beta
        // beta: [B, H] → [B, H, 1]
        let beta_expanded = beta.reshape(&[B, self.num_v_heads, 1])?;
        let delta = v.subtract(&kv_mem)?.multiply(&beta_expanded)?; // [B, H, V]

        // state += outer(k, delta)
        // k: [B, H, K, 1], delta: [B, H, 1, V]
        let delta_row = delta.reshape(&[B, self.num_v_heads, 1, self.value_head_dim])?;
        let outer = k_expanded.multiply(&delta_row)?; // [B, H, K, V]
        state = state.add(outer)?;

        // output = einsum('bhkv,bhk->bhv', state, q)
        let q_expanded = q.reshape(&[B, self.num_v_heads, self.key_head_dim, 1])?;
        let output = state.multiply(&q_expanded)?.sum_axis(-2, false)?; // [B, H, V]

        // Save state
        cache.state = Some(state);
        cache.step += 1;

        Ok(output)
    }

    /// Compute decay gate: g = -exp(A_log) * softplus(a + dt_bias)
    fn compute_decay(&self, a: &Array) -> Result<Array, Exception> {
        let a = a.as_dtype(mlx_rs::Dtype::Float32)?;
        let a_plus_bias = a.add(self.dt_bias.as_ref())?;
        let sp = nn::softplus(a_plus_bias)?;
        let neg_exp_a_log = self.a_log.as_ref().as_dtype(mlx_rs::Dtype::Float32)?.exp()?.negative()?;
        neg_exp_a_log.multiply(sp)
    }

    /// Compute decay gate for batched positions: [B, L, num_v_heads]
    fn compute_decay_batched(&self, a: &Array) -> Result<Array, Exception> {
        let a = a.as_dtype(mlx_rs::Dtype::Float32)?;
        let a_plus_bias = a.add(self.dt_bias.as_ref())?;
        let sp = nn::softplus(a_plus_bias)?;
        let neg_exp_a_log = self.a_log.as_ref().as_dtype(mlx_rs::Dtype::Float32)?.exp()?.negative()?;
        neg_exp_a_log.multiply(sp)
    }

    /// Expand heads from num_k_heads to num_v_heads via repeat_interleave.
    ///
    /// Input: [B, ..., num_k_heads, head_dim]
    /// Output: [B, ..., num_v_heads, head_dim]
    fn repeat_interleave_heads(
        &self,
        x: &Array,
        ratio: i32,
    ) -> Result<Array, Exception> {
        if ratio == 1 {
            return Ok(x.clone());
        }
        let shape = x.shape();
        let ndim = shape.len();
        // The heads dim is at ndim-2
        let heads = shape[ndim - 2];

        // Insert a new dim after heads: [..., num_k_heads, 1, head_dim]
        let mut new_shape: Vec<i32> = shape.to_vec();
        new_shape.insert(ndim - 1, 1);
        let x = x.reshape(&new_shape)?;

        // Broadcast: [..., num_k_heads, ratio, head_dim]
        let mut broadcast_shape = new_shape.clone();
        broadcast_shape[ndim - 1] = ratio;
        let x = broadcast_to(&x, &broadcast_shape)?;

        // Reshape: [..., num_k_heads * ratio, head_dim]
        let mut final_shape: Vec<i32> = shape.to_vec();
        final_shape[ndim - 2] = heads * ratio;
        x.reshape(&final_shape)
    }
}
