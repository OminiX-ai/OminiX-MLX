//! Minimal Metal Flash Attention test
//!
//! Run with:
//!   DYLD_LIBRARY_PATH=~/home/OminiX-MLX/universal-metal-flash-attention/.build/release \
//!   cargo run --example mfa_test

use std::ffi::c_void;
use std::ptr;

// FFI bindings
#[repr(C)]
#[derive(Clone, Copy)]
struct MfaContext(*mut c_void);

#[repr(C)]
#[derive(Clone, Copy)]
struct MfaBuffer(*mut c_void);

const MFA_SUCCESS: i32 = 0;
const MFA_PRECISION_FP32: i32 = 2;
const MFA_MASK_TYPE_NONE: i32 = 0;
const MFA_MASK_SCALAR_BYTE: i32 = 0;

#[link(name = "MFAFFI")]
extern "C" {
    fn mfa_is_device_supported() -> bool;
    fn mfa_get_version(major: *mut i32, minor: *mut i32, patch: *mut i32);
    fn mfa_create_context(context: *mut MfaContext) -> i32;
    fn mfa_destroy_context(context: MfaContext);
    fn mfa_create_buffer(context: MfaContext, size_bytes: usize, buffer: *mut MfaBuffer) -> i32;
    fn mfa_buffer_contents(buffer: MfaBuffer) -> *mut c_void;
    fn mfa_destroy_buffer(buffer: MfaBuffer);
    fn mfa_attention_forward(
        context: MfaContext,
        q: MfaBuffer, k: MfaBuffer, v: MfaBuffer, out: MfaBuffer,
        batch_size: u32, seq_len_q: u32, seq_len_kv: u32,
        num_heads: u32, head_dim: u16,
        softmax_scale: f32, causal: bool,
        input_precision: i32,
        intermediate_precision: i32,
        output_precision: i32,
        transpose_q: bool, transpose_k: bool, transpose_v: bool, transpose_o: bool,
        mask_ptr: *const c_void, mask_size_bytes: usize,
        mask_shape: *const i64, mask_strides: *const i64, mask_ndim: u32,
        mask_type: i32, mask_scalar_type: i32,
    ) -> i32;
}

fn main() {
    println!("=== Metal Flash Attention Test ===\n");

    // Check device support
    let supported = unsafe { mfa_is_device_supported() };
    println!("Device supported: {}", supported);
    if !supported {
        println!("ERROR: Metal Flash Attention not supported on this device");
        return;
    }

    // Get version
    let mut major = 0i32;
    let mut minor = 0i32;
    let mut patch = 0i32;
    unsafe { mfa_get_version(&mut major, &mut minor, &mut patch) };
    println!("MFA Version: {}.{}.{}", major, minor, patch);

    // Create context
    let mut context = MfaContext(ptr::null_mut());
    let err = unsafe { mfa_create_context(&mut context) };
    if err != MFA_SUCCESS {
        println!("ERROR: Failed to create context, error code: {}", err);
        return;
    }
    println!("Context created successfully");

    // Test parameters (small for testing)
    let batch_size = 1u32;
    let seq_len = 64u32;  // Small sequence for test
    let num_heads = 4u32;
    let head_dim = 64u16;
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();

    // Buffer size: batch * seq * heads * head_dim * sizeof(f32)
    let buffer_size = (batch_size * seq_len * num_heads * head_dim as u32 * 4) as usize;
    println!("\nTest parameters:");
    println!("  batch_size: {}", batch_size);
    println!("  seq_len: {}", seq_len);
    println!("  num_heads: {}", num_heads);
    println!("  head_dim: {}", head_dim);
    println!("  buffer_size: {} bytes", buffer_size);

    // Create buffers
    let mut q_buf = MfaBuffer(ptr::null_mut());
    let mut k_buf = MfaBuffer(ptr::null_mut());
    let mut v_buf = MfaBuffer(ptr::null_mut());
    let mut out_buf = MfaBuffer(ptr::null_mut());

    unsafe {
        if mfa_create_buffer(context, buffer_size, &mut q_buf) != MFA_SUCCESS {
            println!("ERROR: Failed to create Q buffer");
            mfa_destroy_context(context);
            return;
        }
        if mfa_create_buffer(context, buffer_size, &mut k_buf) != MFA_SUCCESS {
            println!("ERROR: Failed to create K buffer");
            mfa_destroy_buffer(q_buf);
            mfa_destroy_context(context);
            return;
        }
        if mfa_create_buffer(context, buffer_size, &mut v_buf) != MFA_SUCCESS {
            println!("ERROR: Failed to create V buffer");
            mfa_destroy_buffer(q_buf);
            mfa_destroy_buffer(k_buf);
            mfa_destroy_context(context);
            return;
        }
        if mfa_create_buffer(context, buffer_size, &mut out_buf) != MFA_SUCCESS {
            println!("ERROR: Failed to create output buffer");
            mfa_destroy_buffer(q_buf);
            mfa_destroy_buffer(k_buf);
            mfa_destroy_buffer(v_buf);
            mfa_destroy_context(context);
            return;
        }
    }
    println!("Buffers created successfully");

    // Initialize Q, K, V with test data
    unsafe {
        let q_ptr = mfa_buffer_contents(q_buf) as *mut f32;
        let k_ptr = mfa_buffer_contents(k_buf) as *mut f32;
        let v_ptr = mfa_buffer_contents(v_buf) as *mut f32;

        let num_elements = buffer_size / 4;
        for i in 0..num_elements {
            *q_ptr.add(i) = 0.1;
            *k_ptr.add(i) = 0.1;
            *v_ptr.add(i) = 0.1;
        }
    }
    println!("Buffers initialized with test data");

    // Run attention forward
    println!("\nRunning attention forward...");
    let start = std::time::Instant::now();

    let err = unsafe {
        mfa_attention_forward(
            context,
            q_buf, k_buf, v_buf, out_buf,
            batch_size, seq_len, seq_len,
            num_heads, head_dim,
            softmax_scale, false,  // not causal
            MFA_PRECISION_FP32, MFA_PRECISION_FP32, MFA_PRECISION_FP32,
            false, false, false, false,  // no transpose
            ptr::null(), 0,  // no mask
            ptr::null(), ptr::null(), 0,
            MFA_MASK_TYPE_NONE, MFA_MASK_SCALAR_BYTE,
        )
    };

    let elapsed = start.elapsed();

    if err != MFA_SUCCESS {
        println!("ERROR: Attention forward failed, error code: {}", err);
    } else {
        println!("Attention forward completed in {:?}", elapsed);

        // Check output
        unsafe {
            let out_ptr = mfa_buffer_contents(out_buf) as *const f32;
            let first_val = *out_ptr;
            let last_val = *out_ptr.add(buffer_size / 4 - 1);
            println!("Output[0]: {}", first_val);
            println!("Output[last]: {}", last_val);
        }
    }

    // Cleanup
    unsafe {
        mfa_destroy_buffer(q_buf);
        mfa_destroy_buffer(k_buf);
        mfa_destroy_buffer(v_buf);
        mfa_destroy_buffer(out_buf);
        mfa_destroy_context(context);
    }
    println!("\nCleanup complete");
    println!("\n=== Test Complete ===");
}
