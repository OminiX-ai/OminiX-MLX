//! Custom Metal kernels for fused operations in Qwen3-TTS.
//!
//! Provides:
//! - fused_snake_beta: Fused SnakeBeta activation for the speech tokenizer decoder.
//!   Computes x + sin²(α·x) / (β + ε) in a single kernel instead of 8 separate ops.
//! - fused_residual_rmsnorm: Fused residual add + RmsNorm for transformer blocks.
//!   Computes h = x + residual, normed = rmsnorm(h, weight) in a single kernel.

use mlx_rs::{Array, error::Exception};
use std::ffi::CString;
use std::sync::OnceLock;

// SnakeBeta activation: x + sin²(alpha * x) / (beta + 1e-9)
// alpha and beta are PRE-EXPONENTIATED (exp already applied at load time).
// `dim` is the channel dimension — each thread computes elem % dim to index alpha/beta.
const SNAKE_BETA_KERNEL_SOURCE: &str = r#"
    uint elem = thread_position_in_grid.x;
    uint c = elem % dim;
    T x_val = x[elem];
    T a = alpha[c];
    T b = beta[c];
    T s = metal::sin(x_val * a);
    out[elem] = x_val + s * s / (b + T(1e-9));
"#;

// Fused residual + RmsNorm kernel.
// Computes: h = x + residual, normed = h * weight * rsqrt(mean(h^2) + 1e-6)
// Two outputs: h_out (un-normalized, needed as next residual) and normed_out.
// Each threadgroup processes one row. Uses parallel reduction for mean-of-squares.
const RESIDUAL_RMSNORM_KERNEL_SOURCE: &str = r#"
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    constexpr uint THREADS = 256;

    // Use float for accumulation to avoid bfloat16 precision loss
    threadgroup float shared_sum_sq[256];
    shared_sum_sq[tid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum_sq = 0.0f;
    uint base = row * dim;
    for (uint i = tid; i < dim; i += THREADS) {
        float h_val = float(x[base + i]) + float(residual[base + i]);
        h_out[base + i] = T(h_val);
        local_sum_sq += h_val * h_val;
    }

    shared_sum_sq[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 128) { shared_sum_sq[tid] += shared_sum_sq[tid + 128]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) { shared_sum_sq[tid] += shared_sum_sq[tid + 64]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) { shared_sum_sq[tid] += shared_sum_sq[tid + 32]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) { shared_sum_sq[tid] += shared_sum_sq[tid + 16]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) { shared_sum_sq[tid] += shared_sum_sq[tid + 8]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4) { shared_sum_sq[tid] += shared_sum_sq[tid + 4]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2) { shared_sum_sq[tid] += shared_sum_sq[tid + 2]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { shared_sum_sq[0] += shared_sum_sq[1]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = metal::rsqrt(shared_sum_sq[0] / float(dim) + 1e-6f);

    for (uint i = tid; i < dim; i += THREADS) {
        normed_out[base + i] = T(float(h_out[base + i]) * float(weight[i]) * rms);
    }
"#;

static SNAKE_BETA_KERNEL: OnceLock<MetalKernel> = OnceLock::new();
static RESIDUAL_RMSNORM_KERNEL: OnceLock<MetalKernel> = OnceLock::new();

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

fn create_snake_beta_kernel() -> MetalKernel {
    unsafe {
        let x_name = CString::new("x").unwrap();
        let alpha_name = CString::new("alpha").unwrap();
        let beta_name = CString::new("beta").unwrap();
        let out_name = CString::new("out").unwrap();

        let input_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(input_names, x_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, alpha_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, beta_name.as_ptr());

        let output_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(output_names, out_name.as_ptr());

        let source = CString::new(SNAKE_BETA_KERNEL_SOURCE).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("fused_snake_beta").unwrap();

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

/// Fused SnakeBeta activation using a custom Metal kernel.
///
/// Computes: `x + sin²(alpha * x) / (beta + 1e-9)`
///
/// where `alpha` and `beta` are PRE-EXPONENTIATED (exp() already applied).
/// This fuses 8 separate operations into a single GPU kernel dispatch.
///
/// # Arguments
/// * `x` - Input tensor [B, T, C]
/// * `alpha_exp` - Pre-exponentiated alpha, flattened to [C]
/// * `beta_exp` - Pre-exponentiated beta, flattened to [C]
pub fn fused_snake_beta(x: &Array, alpha_exp: &Array, beta_exp: &Array) -> Result<Array, Exception> {
    let kernel = SNAKE_BETA_KERNEL.get_or_init(create_snake_beta_kernel);

    let shape = x.shape();
    if shape.len() < 2 {
        return Err(Exception::custom("fused_snake_beta requires at least 2D input"));
    }

    let dim = shape[shape.len() - 1] as i32;
    let total_elements: usize = shape.iter().map(|&s| s as usize).product();
    let dtype: u32 = x.dtype().into();

    // Flatten alpha/beta to [C] for contiguous access
    let alpha_flat = alpha_exp.flatten(None, None)?;
    let beta_flat = beta_exp.flatten(None, None)?;

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        let type_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, type_name.as_ptr(), dtype);

        let dim_name = CString::new("dim").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config, dim_name.as_ptr(), dim);

        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, total_elements as i32, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1);

        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, shape_i32.as_ptr(), shape.len(), dtype);

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, x.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, alpha_flat.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, beta_flat.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream);

        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("fused_snake_beta Metal kernel execution failed"));
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

fn create_residual_rmsnorm_kernel() -> MetalKernel {
    unsafe {
        let x_name = CString::new("x").unwrap();
        let residual_name = CString::new("residual").unwrap();
        let weight_name = CString::new("weight").unwrap();
        let h_out_name = CString::new("h_out").unwrap();
        let normed_out_name = CString::new("normed_out").unwrap();

        let input_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(input_names, x_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, residual_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, weight_name.as_ptr());

        let output_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(output_names, h_out_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(output_names, normed_out_name.as_ptr());

        let source = CString::new(RESIDUAL_RMSNORM_KERNEL_SOURCE).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("fused_residual_rmsnorm").unwrap();

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

/// Fused residual addition + RmsNorm using a custom Metal kernel.
///
/// Computes: `h = x + residual`, `normed = h * weight * rsqrt(mean(h²) + 1e-6)`
///
/// Returns `(h, normed)` — both needed by transformer blocks (h for next residual, normed for MLP/attention).
/// Fuses 2 separate operations (add + rmsnorm) into a single GPU kernel dispatch,
/// eliminating one full read+write pass over the hidden state.
///
/// # Arguments
/// * `x` - Attention/MLP output [B, T, D]
/// * `residual` - Residual connection input [B, T, D]
/// * `weight` - RmsNorm weight [D]
pub fn fused_residual_rmsnorm(
    x: &Array,
    residual: &Array,
    weight: &Array,
) -> Result<(Array, Array), Exception> {
    let kernel = RESIDUAL_RMSNORM_KERNEL.get_or_init(create_residual_rmsnorm_kernel);

    let shape = residual.shape();
    if shape.len() < 2 {
        return Err(Exception::custom("fused_residual_rmsnorm requires at least 2D input"));
    }

    let dim = shape[shape.len() - 1] as i32;
    let num_rows: i32 = shape.iter().take(shape.len() - 1).map(|&s| s as i32).product();
    let dtype: u32 = residual.dtype().into();

    let weight_flat = weight.flatten(None, None)?;

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        let type_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, type_name.as_ptr(), dtype);

        let dim_name = CString::new("dim").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config, dim_name.as_ptr(), dim);

        // Grid: num_rows threadgroups × 256 threads each
        let total_threads = num_rows * 256;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, total_threads, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1);

        // Two outputs, both same shape as input
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, shape_i32.as_ptr(), shape.len(), dtype);
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, shape_i32.as_ptr(), shape.len(), dtype);

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, x.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, residual.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, weight_flat.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream);

        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("fused_residual_rmsnorm Metal kernel execution failed"));
        }

        let mut h_result = mlx_sys::mlx_array_new();
        let mut normed_result = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut h_result, outputs, 0);
        mlx_sys::mlx_vector_array_get(&mut normed_result, outputs, 1);

        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);

        Ok((Array::from_ptr(h_result), Array::from_ptr(normed_result)))
    }
}
