//! Custom Metal kernels for fused operations
//!
//! Provides:
//! - fused_swiglu: 10-12x faster than separate silu + multiply (for MoE models)
//! - fused_modulate: Fused LayerNorm + modulation for DiT transformers

use mlx_rs::{Array, error::Exception};
use std::ffi::CString;
use std::sync::OnceLock;

const SWIGLU_KERNEL_SOURCE: &str = r#"
    uint elem = thread_position_in_grid.x;
    T gate_val = gate[elem];
    T x_val = x[elem];
    // silu(gate) = gate / (1 + exp(-gate))
    T silu_gate = gate_val / (T(1) + metal::exp(-gate_val));
    out[elem] = silu_gate * x_val;
"#;

// Fused LayerNorm + Modulation kernel for DiT transformers
// Computes: (1 + scale) * LayerNorm(x) + shift
// where LayerNorm has no learnable parameters (elementwise_affine=False)
//
// This kernel uses parallel reduction within each threadgroup to compute
// mean and variance efficiently.
//
// IMPORTANT: Always launch exactly 256 threads per threadgroup for correct reduction.
const MODULATE_KERNEL_SOURCE: &str = r#"
    // Each threadgroup handles one row (one position in the sequence)
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    constexpr uint THREADS = 256;

    // Shared memory for parallel reduction
    threadgroup T shared_sum[256];
    threadgroup T shared_sum_sq[256];

    // Initialize shared memory to 0 (all threads do this)
    shared_sum[tid] = T(0);
    shared_sum_sq[tid] = T(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread accumulates partial sums over its portion of the row
    T local_sum = T(0);
    T local_sum_sq = T(0);

    uint base = row * dim;
    for (uint i = tid; i < dim; i += THREADS) {
        T val = x[base + i];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Store to shared memory
    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction - fully unrolled for 256 threads
    if (tid < 128) { shared_sum[tid] += shared_sum[tid + 128]; shared_sum_sq[tid] += shared_sum_sq[tid + 128]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) { shared_sum[tid] += shared_sum[tid + 64]; shared_sum_sq[tid] += shared_sum_sq[tid + 64]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) { shared_sum[tid] += shared_sum[tid + 32]; shared_sum_sq[tid] += shared_sum_sq[tid + 32]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) { shared_sum[tid] += shared_sum[tid + 16]; shared_sum_sq[tid] += shared_sum_sq[tid + 16]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) { shared_sum[tid] += shared_sum[tid + 8]; shared_sum_sq[tid] += shared_sum_sq[tid + 8]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4) { shared_sum[tid] += shared_sum[tid + 4]; shared_sum_sq[tid] += shared_sum_sq[tid + 4]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2) { shared_sum[tid] += shared_sum[tid + 2]; shared_sum_sq[tid] += shared_sum_sq[tid + 2]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { shared_sum[0] += shared_sum[1]; shared_sum_sq[0] += shared_sum_sq[1]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ALL threads read the final sums and compute mean/inv_std locally
    // (avoids issues with scalar threadgroup variable broadcast)
    T sum_val = shared_sum[0];
    T sum_sq_val = shared_sum_sq[0];
    T mean = sum_val / T(dim);
    T var = sum_sq_val / T(dim) - mean * mean;
    // Clamp variance to avoid NaN from numerical precision issues
    var = max(var, T(0));
    T inv_std = rsqrt(var + T(1e-6));

    // Apply normalization and modulation: (1 + scale) * normalized + shift
    for (uint i = tid; i < dim; i += THREADS) {
        T normalized = (x[base + i] - mean) * inv_std;
        T scale_val = scale[i];
        T shift_val = shift[i];
        out[base + i] = (T(1) + scale_val) * normalized + shift_val;
    }
"#;

// Flash attention kernel using online softmax (no attention matrix materialization).
// Handles both decode (Q=1) and prefill (Q>1) with causal masking and GQA.
//
// Inputs: q [num_q_heads, seq_q, head_dim], k [num_kv_heads, seq_kv, head_dim],
//         v [num_kv_heads, seq_kv, head_dim]
// Output: out [num_q_heads, seq_q, head_dim]
//
// Grid: (num_q_heads * seq_q) threadgroups, head_dim threads each.
// Each threadgroup computes one output vector using cooperative dot product
// reduction and per-thread online softmax accumulation.
const FLASH_ATTENTION_KERNEL_SOURCE: &str = r#"
    // Decode scale from bit-cast integer template arg
    int _sb = scale_bits;
    float scale = as_type<float>(_sb);

    uint tg_idx = threadgroup_position_in_grid.x;
    uint head_q = tg_idx / seq_q;
    uint q_pos = tg_idx % seq_q;
    uint d = thread_position_in_threadgroup.x;

    // GQA: map query head to corresponding KV head
    uint head_kv = head_q / gqa_factor;

    // Load Q[head_q, q_pos, :] into shared memory (all threads cooperate)
    threadgroup float shared_q[head_dim];
    shared_q[d] = float(q[head_q * seq_q * head_dim + q_pos * head_dim + d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Shared memory for dot product reduction
    threadgroup float shared_dot[head_dim];

    // Determine max KV position for causal masking
    uint max_kv_pos = seq_kv;
    if (causal) {
        uint absolute_q_pos = q_pos + kv_offset;
        max_kv_pos = min(absolute_q_pos + 1, (uint)seq_kv);
    }

    // Online softmax accumulators (per thread, for output dimension d)
    float m = -1e38;
    float l = 0.0;
    float acc = 0.0;

    uint kv_base = head_kv * seq_kv * head_dim;

    for (uint kv = 0; kv < max_kv_pos; kv++) {
        uint kv_off = kv_base + kv * head_dim;

        // Cooperative dot product: Q[q_pos] . K[kv]
        shared_dot[d] = shared_q[d] * float(k[kv_off + d]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tree reduction for dot product (power-of-2 head_dim)
        for (uint stride = head_dim / 2; stride > 0; stride >>= 1) {
            if (d < stride) {
                shared_dot[d] += shared_dot[d + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float score = shared_dot[0] * scale;

        // Online softmax update
        float new_m = max(m, score);
        float correction = exp(m - new_m);
        float weight = exp(score - new_m);
        l = l * correction + weight;
        acc = acc * correction + weight * float(v[kv_off + d]);
        m = new_m;
    }

    // Write normalized output
    if (l > 0.0) {
        out[head_q * seq_q * head_dim + q_pos * head_dim + d] = T(acc / l);
    } else {
        out[head_q * seq_q * head_dim + q_pos * head_dim + d] = T(0);
    }
"#;

static SWIGLU_KERNEL: OnceLock<MetalKernel> = OnceLock::new();
static MODULATE_KERNEL: OnceLock<MetalKernel> = OnceLock::new();
static FLASH_ATTN_KERNEL: OnceLock<MetalKernel> = OnceLock::new();

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

fn create_swiglu_kernel() -> MetalKernel {
    unsafe {
        let x_name = CString::new("x").unwrap();
        let gate_name = CString::new("gate").unwrap();
        let out_name = CString::new("out").unwrap();

        let input_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(input_names, x_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, gate_name.as_ptr());

        let output_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(output_names, out_name.as_ptr());

        let source = CString::new(SWIGLU_KERNEL_SOURCE).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("fused_swiglu").unwrap();

        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            input_names,
            output_names,
            source.as_ptr(),
            header.as_ptr(),
            true,
            false,
        );

        MetalKernel { kernel, input_names, output_names }
    }
}

fn create_modulate_kernel() -> MetalKernel {
    unsafe {
        let x_name = CString::new("x").unwrap();
        let scale_name = CString::new("scale").unwrap();
        let shift_name = CString::new("shift").unwrap();
        let out_name = CString::new("out").unwrap();

        let input_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(input_names, x_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, scale_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, shift_name.as_ptr());

        let output_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(output_names, out_name.as_ptr());

        let source = CString::new(MODULATE_KERNEL_SOURCE).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("fused_modulate").unwrap();

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

/// Fused SwiGLU activation using custom Metal kernel
///
/// Computes: silu(gate) * x = (gate / (1 + exp(-gate))) * x
///
/// This is ~10-12x faster than separate silu() + multiply() calls.
/// Critical for MoE models which have many SwiGLU calls per forward pass.
pub fn fused_swiglu(x: &Array, gate: &Array) -> Result<Array, Exception> {
    let kernel = SWIGLU_KERNEL.get_or_init(create_swiglu_kernel);

    let shape = x.shape();
    let total_elements: usize = shape.iter().map(|&s| s as usize).product();
    let dtype: u32 = x.dtype().into();

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        let type_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, type_name.as_ptr(), dtype);

        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, total_elements as i32, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1);

        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, shape_i32.as_ptr(), shape.len(), dtype);

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, x.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, gate.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream);

        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("Metal kernel execution failed"));
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

/// Fused LayerNorm + Modulation using custom Metal kernel
///
/// Computes: (1 + scale) * LayerNorm(x) + shift
/// where LayerNorm has no learnable parameters (elementwise_affine=False)
///
/// This fuses 7+ operations into a single Metal kernel:
/// - mean computation
/// - variance computation
/// - normalization
/// - scale application (1 + scale)
/// - shift application
///
/// Critical for DiT (Diffusion Transformer) models which call modulate
/// 4x per block × 60 blocks × 40 forward passes = 9600 times per generation.
///
/// # Arguments
/// * `x` - Input tensor of shape [batch, seq, dim] or [seq, dim]
/// * `shift` - Shift tensor, will be flattened to [dim]
/// * `scale` - Scale tensor, will be flattened to [dim]
///
/// # Returns
/// Output tensor of same shape as `x`
pub fn fused_modulate(x: &Array, shift: &Array, scale: &Array) -> Result<Array, Exception> {
    let kernel = MODULATE_KERNEL.get_or_init(create_modulate_kernel);

    let shape = x.shape();
    if shape.len() < 2 {
        return Err(Exception::custom("fused_modulate requires at least 2D input"));
    }

    let dim = shape[shape.len() - 1] as i32;
    let num_rows: i32 = shape.iter().take(shape.len() - 1).map(|&s| s as i32).product();
    let dtype: u32 = x.dtype().into();

    // Ensure shift and scale are contiguous [dim] arrays
    // Use flatten to handle any input shape
    let shift_flat = shift.flatten(None, None)?;
    let scale_flat = scale.flatten(None, None)?;

    // Verify dimensions match
    if shift_flat.shape()[0] != dim || scale_flat.shape()[0] != dim {
        return Err(Exception::custom(format!(
            "fused_modulate: shift/scale dim {} doesn't match x dim {}",
            shift_flat.shape()[0], dim
        )));
    }

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        // Template argument for dtype
        let type_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, type_name.as_ptr(), dtype);

        // Constant argument for dimension size
        let dim_name = CString::new("dim").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config, dim_name.as_ptr(), dim);

        // Grid: total threads = num_rows * 256 (so we get exactly num_rows threadgroups)
        // Threadgroup: 256 threads per group for parallel reduction
        // This gives threadgroup_position_in_grid.x ranging from 0 to num_rows-1
        let total_threads = num_rows * 256;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, total_threads, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1);

        // Output shape same as input
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, shape_i32.as_ptr(), shape.len(), dtype);

        // Input arrays
        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, x.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, scale_flat.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, shift_flat.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream);

        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("fused_modulate Metal kernel execution failed"));
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

fn create_flash_attn_kernel() -> MetalKernel {
    unsafe {
        let q_name = CString::new("q").unwrap();
        let k_name = CString::new("k").unwrap();
        let v_name = CString::new("v").unwrap();
        let out_name = CString::new("out").unwrap();

        let input_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(input_names, q_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, k_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, v_name.as_ptr());

        let output_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(output_names, out_name.as_ptr());

        let source = CString::new(FLASH_ATTENTION_KERNEL_SOURCE).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("flash_attention").unwrap();

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

/// Flash attention using a custom Metal kernel with online softmax.
///
/// Computes scaled dot-product attention without materializing the full
/// attention matrix. Supports GQA (grouped query attention) and causal masking.
///
/// Inputs must be 3D with batch*heads merged into the first dimension:
/// * `queries` - `[num_q_heads, seq_q, head_dim]`
/// * `keys` - `[num_kv_heads, seq_kv, head_dim]`
/// * `values` - `[num_kv_heads, seq_kv, head_dim]`
///
/// `head_dim` must be a power of 2 (e.g., 64, 128).
pub fn flash_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    causal: bool,
    kv_offset: i32,
) -> Result<Array, Exception> {
    let kernel = FLASH_ATTN_KERNEL.get_or_init(create_flash_attn_kernel);

    let q_shape = queries.shape();
    let k_shape = keys.shape();

    if q_shape.len() != 3 || k_shape.len() != 3 {
        return Err(Exception::custom(
            "flash_attention: expected 3D inputs [heads, seq, head_dim]",
        ));
    }

    let num_q_heads = q_shape[0] as i32;
    let seq_q = q_shape[1] as i32;
    let head_dim = q_shape[2] as i32;
    let num_kv_heads = k_shape[0] as i32;
    let seq_kv = k_shape[1] as i32;

    if num_q_heads % num_kv_heads != 0 {
        return Err(Exception::custom(
            "flash_attention: num_q_heads must be divisible by num_kv_heads",
        ));
    }
    let gqa_factor = num_q_heads / num_kv_heads;

    if head_dim & (head_dim - 1) != 0 {
        return Err(Exception::custom(
            "flash_attention: head_dim must be a power of 2",
        ));
    }

    let dtype: u32 = queries.dtype().into();
    let out_shape = [num_q_heads, seq_q, head_dim];

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        let type_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, type_name.as_ptr(), dtype);

        let hd_name = CString::new("head_dim").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config, hd_name.as_ptr(), head_dim);

        let sq_name = CString::new("seq_q").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config, sq_name.as_ptr(), seq_q);

        let skv_name = CString::new("seq_kv").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config, skv_name.as_ptr(), seq_kv);

        let gqa_name = CString::new("gqa_factor").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config, gqa_name.as_ptr(), gqa_factor);

        // Encode float scale as int bits (no float template args in MLX custom kernels)
        let scale_bits: i32 = f32::to_bits(scale) as i32;
        let scale_name = CString::new("scale_bits").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config, scale_name.as_ptr(), scale_bits);

        let causal_name = CString::new("causal").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
            config, causal_name.as_ptr(), causal);

        let offset_name = CString::new("kv_offset").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config, offset_name.as_ptr(), kv_offset);

        // Grid: one threadgroup per (head, query_pos). head_dim threads per group.
        let num_threadgroups = num_q_heads * seq_q;
        let total_threads = num_threadgroups * head_dim;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, total_threads, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, head_dim, 1, 1);

        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, out_shape.as_ptr(), out_shape.len(), dtype);

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, queries.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, keys.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, values.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream);

        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("flash_attention Metal kernel failed"));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_attention_decode() {
        // Simulates Qwen3-TTS decode: 16 Q heads, 8 KV heads, Q=1, KV=4, head_dim=64
        let q = Array::ones::<f32>(&[16, 1, 64]).unwrap();
        let k = Array::ones::<f32>(&[8, 4, 64]).unwrap();
        let v_data: Vec<f32> = (0..2048).map(|i| (i % 64) as f32 / 64.0).collect();
        let v = Array::from_slice(&v_data, &[8, 4, 64]);

        let scale = 1.0 / (64.0f32).sqrt();
        let result = flash_attention(&q, &k, &v, scale, false, 0).unwrap();
        result.eval().unwrap();

        assert_eq!(result.shape(), &[16, 1, 64]);
        let out = result.as_slice::<f32>();
        assert!(!out.iter().any(|x| x.is_nan()), "output contains NaN");
        assert!(out.iter().any(|x| *x != 0.0), "output is all zeros");
    }

    #[test]
    fn test_flash_attention_causal_prefill() {
        // Prefill: 4 Q heads, 2 KV heads, seq_q=3, seq_kv=3, head_dim=64
        let q = Array::ones::<f32>(&[4, 3, 64]).unwrap();
        let k = Array::ones::<f32>(&[2, 3, 64]).unwrap();
        let v = Array::ones::<f32>(&[2, 3, 64]).unwrap();

        let scale = 1.0 / (64.0f32).sqrt();
        let result = flash_attention(&q, &k, &v, scale, true, 0).unwrap();
        result.eval().unwrap();

        assert_eq!(result.shape(), &[4, 3, 64]);
        let out = result.as_slice::<f32>();
        assert!(!out.iter().any(|x| x.is_nan()), "causal output has NaN");
    }
}
