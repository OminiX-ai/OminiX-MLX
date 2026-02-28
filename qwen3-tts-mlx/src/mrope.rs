//! Multimodal Rotary Position Embedding (MRoPE) for Qwen3-TTS.
//!
//! For TTS inference, only the temporal dimension advances (height=0, width=0).
//! Uses interleaved (traditional) rotation as specified by the model config.

use mlx_rs::{
    array,
    error::Exception,
    ops::{arange, concatenate_axis, indexing::NewAxis, ones, zeros},
    Array, Dtype,
};

/// Apply MRoPE to queries or keys for TTS (temporal-only).
///
/// Uses interleaved rotation: consecutive pairs (0,1), (2,3), etc.
/// This matches HuggingFace's `interleaved=True` / MLX's `traditional=True`.
///
/// - `x`: shape [B, num_heads, L, head_dim]
/// - `offset`: temporal position offset
/// - `temporal_section`: number of frequency pairs for temporal dim (e.g. 24)
/// - `base`: RoPE base frequency (e.g. 1_000_000)
/// - `head_dim`: full head dimension (e.g. 128)
pub fn apply_mrope_tts(
    x: &Array,
    offset: i32,
    temporal_section: i32,
    base: f32,
    head_dim: i32,
) -> Result<Array, Exception> {
    let half_dim = head_dim / 2;
    let l = x.dim(2) as i32;

    // Compute inverse frequencies for the temporal section:
    // inv_freq[i] = 1 / (base^(2*i / head_dim))  for i = 0..temporal_section
    let indices = arange::<_, f32>(0, temporal_section, None)?;
    let exponents = indices.multiply(array!(2.0f32 / head_dim as f32))?;
    let inv_freq = array!(base).power(&exponents)?.reciprocal()?;

    // Positions: [L]
    let positions = arange::<_, f32>(offset, offset + l, None)?;

    // Angles: [L, temporal_section] via broadcasting multiply
    let angles = positions
        .index((.., NewAxis))
        .multiply(inv_freq.index(NewAxis))?;

    // cos/sin for temporal section: [1, 1, L, temporal_section]
    let cos_temporal = angles
        .cos()?
        .reshape(&[1, 1, l, temporal_section])?;
    let sin_temporal = angles
        .sin()?
        .reshape(&[1, 1, l, temporal_section])?;

    // Pad to half_dim: temporal gets real rotation, rest gets identity (cos=1, sin=0)
    let pad_size = half_dim - temporal_section;
    let cos_half;
    let sin_half;
    if pad_size > 0 {
        let ones_pad = ones::<f32>(&[1, 1, l, pad_size])?;
        let zeros_pad = zeros::<f32>(&[1, 1, l, pad_size])?;
        cos_half = concatenate_axis(&[&cos_temporal, &ones_pad], -1)?;
        sin_half = concatenate_axis(&[&sin_temporal, &zeros_pad], -1)?;
    } else {
        cos_half = cos_temporal;
        sin_half = sin_temporal;
    };

    // --- Interleaved cos/sin expansion ---
    // cos_half: [1, 1, L, half_dim]
    // Need: [1, 1, L, head_dim] with pattern [c0, c0, c1, c1, ...]
    let cos_half_exp = cos_half.reshape(&[1, 1, l, half_dim, 1])?;
    let cos_full = concatenate_axis(&[&cos_half_exp, &cos_half_exp], -1)?
        .reshape(&[1, 1, l, head_dim])?;
    let sin_half_exp = sin_half.reshape(&[1, 1, l, half_dim, 1])?;
    let sin_full = concatenate_axis(&[&sin_half_exp, &sin_half_exp], -1)?
        .reshape(&[1, 1, l, head_dim])?;

    // --- Interleaved (traditional) rotation ---
    // For each consecutive pair (x[2i], x[2i+1]):
    //   rotated[2i]   = -x[2i+1]
    //   rotated[2i+1] = x[2i]
    let x = x.as_dtype(Dtype::Float32)?;
    let orig_shape: Vec<i32> = x.shape().iter().map(|&s| s as i32).collect();

    // Flatten all dims except last, then split into pairs
    let flat = x.reshape(&[-1, head_dim])?;
    let pairs = flat.reshape(&[-1, half_dim, 2])?;

    use mlx_rs::ops::indexing::IndexOp;
    let x_even = pairs.index((.., .., 0)); // [N, half_dim]  (x[2i])
    let x_odd = pairs.index((.., .., 1));  // [N, half_dim]  (x[2i+1])

    // Build rotated: [-x_odd, x_even] interleaved back to pairs
    let neg_x_odd = x_odd.negative()?;
    let neg_x_odd_exp = neg_x_odd.reshape(&[-1, half_dim, 1])?;
    let x_even_exp = x_even.reshape(&[-1, half_dim, 1])?;
    let x_rotated = concatenate_axis(&[&neg_x_odd_exp, &x_even_exp], -1)?
        .reshape(&[-1, head_dim])?
        .reshape(&orig_shape)?;

    // result = x * cos + rotated * sin
    x.multiply(&cos_full)?.add(x_rotated.multiply(&sin_full)?)
}
