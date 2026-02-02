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

static SWIGLU_KERNEL: OnceLock<MetalKernel> = OnceLock::new();
static MODULATE_KERNEL: OnceLock<MetalKernel> = OnceLock::new();

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
