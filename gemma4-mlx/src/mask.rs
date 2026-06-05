//! Per-layer-type attention masks (bool semantics: true = visible).
use mlx_rs::Array;
use mlx_rs::error::Exception;
use mlx_rs_core::utils::create_causal_mask;

/// Full causal bool mask, shape [N, offset+N].
pub fn full_causal_mask(n: i32, offset: i32) -> Result<Array, Exception> {
    create_causal_mask(n, Some(offset), None, None)
}

/// Sliding-window causal bool mask: causal AND restricted to a band of `window + 1`
/// visible columns per row (the current key plus the `window` prior keys, i.e.
/// `q - window <= k <= q`). Shape [N, offset+N].
pub fn sliding_window_mask(n: i32, offset: i32, window: i32) -> Result<Array, Exception> {
    create_causal_mask(n, Some(offset), Some(window), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_mask_is_causal_band() {
        // N=4, window=1 -> visible k in [q-1, q]; bool true=visible (4x4 row-major)
        let m = sliding_window_mask(4, 0, 1).unwrap();
        let v: Vec<bool> = m.as_slice::<bool>().to_vec();
        // row q occupies v[q*4 .. q*4+4]. window=1 => visible band {q-1, q}.
        assert_eq!(&v[0..4],   &[true, false, false, false]);  // q=0 sees k0
        assert_eq!(&v[4..8],   &[true, true, false, false]);   // q=1 sees k0,k1 (lower bound inclusive)
        assert_eq!(&v[8..12],  &[false, true, true, false]);   // q=2 sees k1,k2
        assert_eq!(&v[12..16], &[false, false, true, true]);   // q=3 sees k2,k3 (trailing edge)
    }

    #[test]
    fn full_mask_is_causal() {
        let m = full_causal_mask(3, 0).unwrap();
        let v: Vec<bool> = m.as_slice::<bool>().to_vec();
        assert_eq!(&v[0..3], &[true, false, false]);  // q=0 sees only k0
        assert_eq!(&v[6..9], &[true, true, true]);    // q=2 sees k0,k1,k2
    }
}
