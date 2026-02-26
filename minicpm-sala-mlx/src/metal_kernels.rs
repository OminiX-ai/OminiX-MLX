//! Custom Metal kernels for fused GLA (Gated Linear Attention) operations.
//!
//! Provides:
//! - fused_intra_chunk_attn: Fuses Q@K^T * decay_mask @ V into a single kernel
//! - fused_state_update: Fuses K*reverse_decay, K_w^T@V, chunk_decay*state+kv

use mlx_rs::{Array, error::Exception};
use std::ffi::CString;
use std::sync::OnceLock;

// ============================================================================
// Kernel 1: Fused Intra-Chunk Attention with Decay Mask
// ============================================================================
//
// Computes: out[b,h,i,d] = sum_{j<=i} decay_mask[h,i,j] * dot(Q[b,h,i,:], K[b,h,j,:]) * V[b,h,j,d]
//
// This fuses 4 separate ops (transpose, Q@K^T, *mask, @V) into 1 kernel.
// Never materializes the full C×C scores matrix in device memory.
//
// Thread layout: one threadgroup per (b, h, i) triple, 256 threads per group.
// Shared memory: ~1.75KB per threadgroup (well within 32KB limit).

const INTRA_CHUNK_ATTN_SOURCE: &str = r#"
    uint pos = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    constexpr uint THREADS = 256;

    uint b = pos / (H * C);
    uint h = (pos / C) % H;
    uint i = pos % C;

    // Shared memory for Q row and computed scores
    threadgroup T shared_q[D];
    threadgroup T shared_scores[C];
    threadgroup T reduction[THREADS];

    // Phase 1: Cooperatively load Q[b,h,i,:] into shared memory
    uint q_base = ((b * H + h) * C + i) * D;
    for (uint d = tid; d < D; d += THREADS) {
        shared_q[d] = q[q_base + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute masked attention scores
    // Assign THREADS/C = 4 threads per j value (256/64=4)
    // Each group of 4 computes dot(Q[i], K[j]) by splitting D=128 into 4 chunks of 32
    constexpr uint THREADS_PER_J = THREADS / C;
    uint j_idx = tid / THREADS_PER_J;
    uint lane = tid % THREADS_PER_J;

    T partial_dot = T(0);
    if (j_idx <= i && j_idx < C) {
        uint k_base = ((b * H + h) * C + j_idx) * D;
        uint chunk_size = D / THREADS_PER_J;
        uint d_start = lane * chunk_size;
        uint d_end = d_start + chunk_size;
        for (uint d = d_start; d < d_end; d++) {
            partial_dot += shared_q[d] * k[k_base + d];
        }
    }

    // Store partial results for reduction
    reduction[tid] = partial_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree-reduce within each group of THREADS_PER_J=4 threads
    if (THREADS_PER_J >= 4 && (lane < 2)) {
        reduction[tid] += reduction[tid + 2];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        T score = reduction[tid] + reduction[tid + 1];
        if (j_idx <= i && j_idx < C) {
            // Apply decay mask: mask layout is [1, H, C, C] row-major
            uint mask_idx = (h * C + i) * C + j_idx;
            score *= decay_mask[mask_idx];
        } else {
            score = T(0);
        }
        shared_scores[j_idx] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Compute output[b,h,i,d] = sum_j scores[j] * V[b,h,j,d]
    // Each thread handles ceil(D/THREADS) output dimensions
    uint out_base = ((b * H + h) * C + i) * D;
    for (uint d = tid; d < D; d += THREADS) {
        T accum = T(0);
        for (uint j = 0; j <= i && j < C; j++) {
            uint v_idx = ((b * H + h) * C + j) * D + d;
            accum += shared_scores[j] * v[v_idx];
        }
        out[out_base + d] = accum;
    }
"#;

// ============================================================================
// Kernel 2: Fused State Update
// ============================================================================
//
// Computes: state_out[b,h,d_out,d_in] = chunk_decay[h] * state_in[b,h,d_out,d_in]
//           + sum_{t=0..C-1} (K[b,h,t,d_out] * reverse_decay[h,t]) * V[b,h,t,d_in]
//
// This fuses 4 ops (K*reverse_decay, transpose, K_w^T@V, scale+add) into 1 kernel.
//
// Thread layout: one threadgroup per (b, h, d_out) triple, 256 threads per group.
// Shared memory: 256 bytes per threadgroup.

const STATE_UPDATE_SOURCE: &str = r#"
    uint pos = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    constexpr uint THREADS = 256;

    uint b = pos / (H * D);
    uint h = (pos / D) % H;
    uint d_out = pos % D;

    // Load K[b,h,:,d_out] * reverse_decay[h,:] into shared memory
    threadgroup T shared_kw[C];
    if (tid < C) {
        uint k_idx = ((b * H + h) * C + tid) * D + d_out;
        // reverse_decay layout: [1, H, C, 1] row-major -> index h*C + t
        uint rd_idx = h * C + tid;
        shared_kw[tid] = k[k_idx] * reverse_decay[rd_idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // chunk_decay layout: [1, H, 1, 1] row-major -> index h
    T cd = chunk_decay[h];

    // Each thread computes one or more d_in elements
    for (uint d_in = tid; d_in < D; d_in += THREADS) {
        T accum = T(0);
        for (uint t = 0; t < C; t++) {
            uint v_idx = ((b * H + h) * C + t) * D + d_in;
            accum += shared_kw[t] * v[v_idx];
        }
        // state layout: [B, H, D, D] row-major
        uint state_idx = ((b * H + h) * D + d_out) * D + d_in;
        state_out[state_idx] = cd * state_in[state_idx] + accum;
    }
"#;

// ============================================================================
// Kernel 3: Fused GLA Decode (single recurrent step)
// ============================================================================
//
// NOTE: This kernel is retained for reference but NOT currently used.
// Benchmarking showed ~0% gain because decode is memory-bandwidth limited by
// weight loading (~6.8GB/token). The stride-D column reads for q@state also
// offset kernel dispatch savings. To re-enable, replace gla_recurrent_step in
// lightning.rs with a call to fused_gla_decode().
//
// Fuses 5 ops into 1 kernel for the decode path (L=1):
//   1. exp(decay)
//   2. k^T  (implicit — handled by indexing)
//   3. k^T @ v  (outer product)
//   4. decay * state + kv  (state update)
//   5. q @ state  (output)
//
// Reads state once instead of twice (saves ~2MB bandwidth per layer).
//
// Inputs:  q [B,H,1,D], k [B,H,1,D], v [B,H,1,D], decay [1,H,1,1], state_in [B,H,D,D]
// Outputs: out [B,H,1,D], state_out [B,H,D,D]
//
// Thread layout: one threadgroup per (b, h, d_out) triple = B×H×D threadgroups, 256 threads each.

#[allow(dead_code)]
const DECODE_GLA_SOURCE: &str = r#"
    uint pos = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    constexpr uint THREADS = 256;

    uint b = pos / (H * D);
    uint h = (pos / D) % H;
    uint d_out = pos % D;

    // exp(decay) for this head — decay layout: [1, H, 1, 1] flattened
    T decay_val = metal::exp(decay[h]);
    // k[b,h,0,d_out] and v[b,h,0,d_out] — [B,H,1,D] flattened
    T k_val = k[(b * H + h) * D + d_out];
    T v_out = v[(b * H + h) * D + d_out];

    // Phase 1: Update state row d_out + compute dot(q, k)
    //   state_out[d_out, d_in] = decay * state_in[d_out, d_in] + k[d_out] * v[d_in]
    //   qk = sum_i q[i] * k[i]
    T qk_accum = T(0);
    for (uint d_in = tid; d_in < D; d_in += THREADS) {
        uint row_idx = ((b * H + h) * D + d_out) * D + d_in;
        T v_val = v[(b * H + h) * D + d_in];
        state_out[row_idx] = decay_val * state_in[row_idx] + k_val * v_val;

        T q_val = q[(b * H + h) * D + d_in];
        T k_in = k[(b * H + h) * D + d_in];
        qk_accum += q_val * k_in;
    }

    // Phase 2: Compute (q @ state_in)[d_out] = sum_i q[i] * state_in[i, d_out]
    //   Uses column d_out of old state (stride-D access pattern)
    T qs_accum = T(0);
    for (uint d_in = tid; d_in < D; d_in += THREADS) {
        uint col_idx = ((b * H + h) * D + d_in) * D + d_out;
        T q_val = q[(b * H + h) * D + d_in];
        qs_accum += q_val * state_in[col_idx];
    }

    // Phase 3: Parallel reduction of both accumulators
    threadgroup T shared_qk[THREADS];
    threadgroup T shared_qs[THREADS];
    shared_qk[tid] = qk_accum;
    shared_qs[tid] = qs_accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_qk[tid] += shared_qk[tid + stride];
            shared_qs[tid] += shared_qs[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        // out[d_out] = decay * (q @ state_in)[d_out] + v[d_out] * dot(q, k)
        // Derived from: out = q @ new_state = q @ (decay*state_in + k^T@v)
        out[(b * H + h) * D + d_out] = decay_val * shared_qs[0] + v_out * shared_qk[0];
    }
"#;

// ============================================================================
// Kernel management
// ============================================================================

struct MetalKernel {
    kernel: mlx_sys::mlx_fast_metal_kernel,
    input_names: mlx_sys::mlx_vector_string,
    output_names: mlx_sys::mlx_vector_string,
}

unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_free(self.kernel);
            mlx_sys::mlx_vector_string_free(self.input_names);
            mlx_sys::mlx_vector_string_free(self.output_names);
        }
    }
}

static INTRA_CHUNK_KERNEL: OnceLock<MetalKernel> = OnceLock::new();
static STATE_UPDATE_KERNEL: OnceLock<MetalKernel> = OnceLock::new();
#[allow(dead_code)]
static DECODE_GLA_KERNEL: OnceLock<MetalKernel> = OnceLock::new();

fn create_intra_chunk_kernel() -> MetalKernel {
    unsafe {
        let names: Vec<CString> = ["q", "k", "v", "decay_mask", "out"]
            .iter()
            .map(|n| CString::new(*n).unwrap())
            .collect();

        let input_names = mlx_sys::mlx_vector_string_new();
        for n in &names[..4] {
            mlx_sys::mlx_vector_string_append_value(input_names, n.as_ptr());
        }

        let output_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(output_names, names[4].as_ptr());

        let source = CString::new(INTRA_CHUNK_ATTN_SOURCE).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("fused_intra_chunk_attn").unwrap();

        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            input_names,
            output_names,
            source.as_ptr(),
            header.as_ptr(),
            true,  // ensure_row_contiguous
            false, // atomic_outputs
        );

        MetalKernel { kernel, input_names, output_names }
    }
}

fn create_state_update_kernel() -> MetalKernel {
    unsafe {
        let names: Vec<CString> =
            ["k", "v", "state_in", "reverse_decay", "chunk_decay", "state_out"]
                .iter()
                .map(|n| CString::new(*n).unwrap())
                .collect();

        let input_names = mlx_sys::mlx_vector_string_new();
        for n in &names[..5] {
            mlx_sys::mlx_vector_string_append_value(input_names, n.as_ptr());
        }

        let output_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(output_names, names[5].as_ptr());

        let source = CString::new(STATE_UPDATE_SOURCE).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("fused_state_update").unwrap();

        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            input_names,
            output_names,
            source.as_ptr(),
            header.as_ptr(),
            true,  // ensure_row_contiguous
            false, // atomic_outputs
        );

        MetalKernel { kernel, input_names, output_names }
    }
}

#[allow(dead_code)]
fn create_decode_gla_kernel() -> MetalKernel {
    unsafe {
        let names: Vec<CString> =
            ["q", "k", "v", "decay", "state_in", "out", "state_out"]
                .iter()
                .map(|n| CString::new(*n).unwrap())
                .collect();

        let input_names = mlx_sys::mlx_vector_string_new();
        for n in &names[..5] {
            mlx_sys::mlx_vector_string_append_value(input_names, n.as_ptr());
        }

        let output_names = mlx_sys::mlx_vector_string_new();
        for n in &names[5..] {
            mlx_sys::mlx_vector_string_append_value(output_names, n.as_ptr());
        }

        let source = CString::new(DECODE_GLA_SOURCE).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("fused_gla_decode").unwrap();

        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            input_names,
            output_names,
            source.as_ptr(),
            header.as_ptr(),
            true,  // ensure_row_contiguous
            false, // atomic_outputs
        );

        MetalKernel { kernel, input_names, output_names }
    }
}

// ============================================================================
// Public dispatch functions
// ============================================================================

/// Fused intra-chunk attention with decay masking.
///
/// Computes: `(Q @ K^T) * decay_mask @ V` in a single Metal kernel.
/// Never materializes the full C×C scores matrix in device memory.
///
/// # Arguments
/// * `q` - Queries [B, H, C, D]
/// * `k` - Keys [B, H, C, D]
/// * `v` - Values [B, H, C, D]
/// * `decay_mask` - Causal decay mask [1, H, C, C]
///
/// # Returns
/// Output [B, H, C, D]
#[allow(non_snake_case)]
pub fn fused_intra_chunk_attn(
    q: &Array,
    k: &Array,
    v: &Array,
    decay_mask: &Array,
    B: i32,
    H: i32,
    C: i32,
    D: i32,
) -> Result<Array, Exception> {
    let kernel = INTRA_CHUNK_KERNEL.get_or_init(create_intra_chunk_kernel);

    // Cast all inputs to float32 for uniform dtype in the kernel.
    // The decay_mask is always float32; Q/K/V may be float16 from quantized layers.
    let q = q.as_type::<f32>()?;
    let k = k.as_type::<f32>()?;
    let v = v.as_type::<f32>()?;
    let decay_mask = decay_mask.as_type::<f32>()?;
    let dtype: u32 = mlx_rs::Dtype::Float32.into();

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        // Template arguments
        let t_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, t_name.as_ptr(), dtype,
        );
        let h_name = CString::new("H").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, h_name.as_ptr(), H);
        let c_name = CString::new("C").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c_name.as_ptr(), C);
        let d_name = CString::new("D").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, d_name.as_ptr(), D);

        // Grid: B*H*C threadgroups, 256 threads each
        let num_threadgroups = B * H * C;
        let total_threads = num_threadgroups * 256;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, total_threads, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1);

        // Output shape: [B, H, C, D]
        let out_shape = [B, H, C, D];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            out_shape.as_ptr(),
            4,
            dtype,
        );

        // Input arrays
        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, q.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, k.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, v.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, decay_mask.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs,
            kernel.kernel,
            inputs,
            config,
            stream,
        );

        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom(
                "fused_intra_chunk_attn Metal kernel execution failed",
            ));
        }

        let mut result = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut result, outputs, 0);

        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);

        Ok(Array::from_ptr(result))
    }
}

/// Fused state update for GLA chunked prefill.
///
/// Computes: `state_out = chunk_decay * state_in + (K * reverse_decay)^T @ V`
/// in a single Metal kernel.
///
/// # Arguments
/// * `k` - Keys [B, H, C, D]
/// * `v` - Values [B, H, C, D]
/// * `state_in` - Previous state [B, H, D, D]
/// * `reverse_decay` - Reverse decay weights [1, H, C, 1]
/// * `chunk_decay` - Chunk decay factor [1, H, 1, 1]
///
/// # Returns
/// Updated state [B, H, D, D]
#[allow(non_snake_case)]
pub fn fused_state_update(
    k: &Array,
    v: &Array,
    state_in: &Array,
    reverse_decay: &Array,
    chunk_decay: &Array,
    B: i32,
    H: i32,
    C: i32,
    D: i32,
) -> Result<Array, Exception> {
    let kernel = STATE_UPDATE_KERNEL.get_or_init(create_state_update_kernel);

    // Cast all inputs to float32 for uniform dtype in the kernel.
    let k = k.as_type::<f32>()?;
    let v = v.as_type::<f32>()?;
    let state_in = state_in.as_type::<f32>()?;
    let reverse_decay = reverse_decay.as_type::<f32>()?;
    let chunk_decay = chunk_decay.as_type::<f32>()?;
    let dtype: u32 = mlx_rs::Dtype::Float32.into();

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        // Template arguments
        let t_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, t_name.as_ptr(), dtype,
        );
        let h_name = CString::new("H").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, h_name.as_ptr(), H);
        let c_name = CString::new("C").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, c_name.as_ptr(), C);
        let d_name = CString::new("D").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, d_name.as_ptr(), D);

        // Grid: B*H*D threadgroups, 256 threads each
        let num_threadgroups = B * H * D;
        let total_threads = num_threadgroups * 256;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, total_threads, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1);

        // Output shape: [B, H, D, D]
        let out_shape = [B, H, D, D];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            out_shape.as_ptr(),
            4,
            dtype,
        );

        // Input arrays
        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, k.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, v.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, state_in.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, reverse_decay.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, chunk_decay.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs,
            kernel.kernel,
            inputs,
            config,
            stream,
        );

        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom(
                "fused_state_update Metal kernel execution failed",
            ));
        }

        let mut result = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut result, outputs, 0);

        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);

        Ok(Array::from_ptr(result))
    }
}

/// Fused GLA decode step (single recurrent step, L=1).
///
/// **NOTE**: Currently unused — retained for future reference. See kernel 3 comments above.
///
/// Fuses 5 ops into 1 kernel: exp(decay), k^T@v outer product, decay*state+kv, q@state.
/// Reads state once instead of twice, saving ~2MB bandwidth per layer.
///
/// # Arguments
/// * `q` - Queries [B, H, 1, D]
/// * `k` - Keys [B, H, 1, D]
/// * `v` - Values [B, H, 1, D]
/// * `decay` - Raw ALiBi slopes [1, H, 1, 1] (kernel computes exp internally)
/// * `state_in` - Previous recurrent state [B, H, D, D]
///
/// # Returns
/// Tuple of (output [B, H, 1, D], new_state [B, H, D, D])
#[allow(dead_code)]
#[allow(non_snake_case)]
pub fn fused_gla_decode(
    q: &Array,
    k: &Array,
    v: &Array,
    decay: &Array,
    state_in: &Array,
    B: i32,
    H: i32,
    D: i32,
) -> Result<(Array, Array), Exception> {
    let kernel = DECODE_GLA_KERNEL.get_or_init(create_decode_gla_kernel);

    // Cast all inputs to float32 for uniform dtype in the kernel.
    let q = q.as_type::<f32>()?;
    let k = k.as_type::<f32>()?;
    let v = v.as_type::<f32>()?;
    let decay = decay.as_type::<f32>()?;
    let state_in = state_in.as_type::<f32>()?;
    let dtype: u32 = mlx_rs::Dtype::Float32.into();

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        // Template arguments: T (dtype), H (heads), D (head_dim)
        let t_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, t_name.as_ptr(), dtype,
        );
        let h_name = CString::new("H").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, h_name.as_ptr(), H);
        let d_name = CString::new("D").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, d_name.as_ptr(), D);

        // Grid: B*H*D threadgroups, 256 threads each
        let num_threadgroups = B * H * D;
        let total_threads = num_threadgroups * 256;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, total_threads, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1);

        // Output 1: out [B, H, 1, D]
        let out_shape = [B, H, 1, D];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            out_shape.as_ptr(),
            4,
            dtype,
        );

        // Output 2: state_out [B, H, D, D]
        let state_shape = [B, H, D, D];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            state_shape.as_ptr(),
            4,
            dtype,
        );

        // Input arrays
        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, q.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, k.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, v.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, decay.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, state_in.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs,
            kernel.kernel,
            inputs,
            config,
            stream,
        );

        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom(
                "fused_gla_decode Metal kernel execution failed",
            ));
        }

        // Extract both outputs
        let mut result_out = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut result_out, outputs, 0);
        let mut result_state = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut result_state, outputs, 1);

        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);

        Ok((Array::from_ptr(result_out), Array::from_ptr(result_state)))
    }
}
