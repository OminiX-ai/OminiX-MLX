//! Gemma 4 dual RoPE.
//! - sliding layers: standard full-dim RoPE (use mlx_rs::nn::Rope, dims=head_dim) — added later.
//! - global layers: proportional RoPE — frequency denominator is the FULL head_dim;
//!   only the first rotary_dim dims are rotated, the rest are identity (freq=0).

/// Returns a length `head_dim/2` inv_freq vector: the first `rotary_dim/2` entries are
/// `theta^(-(2*i / head_dim))`, the remaining entries are 0.0 (identity / unrotated).
///
/// SIGN CONVENTION — read before reusing: this returns the *true* inv_freq
/// (NEGATIVE exponent). `mlx_rs::fast::rope`'s `freqs` argument wants the
/// POSITIVE-exponent form (`theta^(+2i/head_dim)`) because the kernel takes its
/// reciprocal internally. `attention.rs::Rope::Proportional` therefore builds the
/// positive form inline — do NOT feed this function's output directly to
/// `fast::rope` (you'd double-invert). This fn documents the formula and is
/// unit-tested; the production rotation lives in attention.rs.
pub fn proportional_inv_freq(head_dim: i32, rotary_dim: i32, theta: f32) -> Vec<f32> {
    let half = (head_dim / 2) as usize;
    let rot_half = (rotary_dim / 2) as usize;
    let mut out = vec![0.0f32; half];
    for i in 0..rot_half {
        let exponent = (2 * i) as f32 / head_dim as f32;
        out[i] = theta.powf(-exponent);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proportional_inv_freq_uses_head_dim_denominator() {
        // head_dim=8, partial=0.5 -> rotary_dim=4 -> 2 nonzero freqs, 2 identity (0)
        let f = proportional_inv_freq(8, 4, 10000.0);
        assert_eq!(f.len(), 4); // head_dim/2
        let expect0 = (10000f32).powf(-(0.0/8.0)); // = 1.0
        let expect1 = (10000f32).powf(-(2.0/8.0));
        assert!((f[0]-expect0).abs() < 1e-5, "{f:?}");
        assert!((f[1]-expect1).abs() < 1e-5, "{f:?}");
        assert_eq!(f[2], 0.0);
        assert_eq!(f[3], 0.0);
    }
}
