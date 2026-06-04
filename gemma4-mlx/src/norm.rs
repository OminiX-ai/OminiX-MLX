//! Gemma 4 RMSNorm: standard gamma (weight*norm) or no-scale (pure norm, for v_norm).
//!
//! TODO(parity, M1): HF `Gemma4RMSNorm` upcasts to f32 (`_norm(x.float())`) before
//! computing variance, then casts back. We compute in the input dtype — fine for f32
//! (the M0 reference dump runs in f32); if bf16-activation parity drifts vs HF, upcast here.

use mlx_rs::{Array, error::Exception, ops::rsqrt};

/// Gemma 4 RMSNorm.
///
/// - **Standard form** (`from_weight`): `out = weight * x / sqrt(mean(x^2, axis=-1) + eps)`
///   where `weight` is plain gamma (initialized to ones externally — NOT `1 + weight`).
/// - **No-scale form** (`new_no_scale`): `out = x / sqrt(mean(x^2, axis=-1) + eps)`
///   used for the attention `v_norm`.
pub struct GemmaRmsNorm {
    weight: Option<Array>,
    eps: f32,
}

impl GemmaRmsNorm {
    /// Create a no-scale (pure) RMSNorm — no learnable weight.
    /// `_dim` is accepted for API symmetry but unused (no weight allocation needed).
    pub fn new_no_scale(_dim: i32, eps: f32) -> Self {
        Self { weight: None, eps }
    }

    /// Create a standard-gamma RMSNorm from an existing weight array.
    pub fn from_weight(weight: Array, eps: f32) -> Self {
        Self { weight: Some(weight), eps }
    }

    /// Forward pass.
    ///
    /// Computes `normed = x * rsqrt(mean(x^2, axis=-1, keepdims=true) + eps)`
    /// then multiplies by `weight` if present.
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        // mean(x^2, axis=-1, keepdims=true)
        let x2 = x.square()?;
        let var = x2.mean_axes(&[-1], true)?;
        // rsqrt(var + eps)
        let scale = rsqrt(&var.add(Array::from_f32(self.eps))?)?;
        let normed = x.multiply(&scale)?;
        match &self.weight {
            Some(w) => normed.multiply(w),
            None => Ok(normed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    #[test]
    fn no_scale_is_pure_rmsnorm() {
        // x=[3,4] -> rms=sqrt((9+16)/2)=sqrt(12.5); normed = x/rms
        let x = Array::from_slice(&[3.0f32, 4.0], &[1, 2]);
        let n = GemmaRmsNorm::new_no_scale(2, 0.0);
        let y = n.forward(&x).unwrap();
        let v: Vec<f32> = y.as_slice::<f32>().to_vec();
        let rms = (12.5f32).sqrt();
        assert!((v[0] - 3.0 / rms).abs() < 1e-4, "{v:?}");
        assert!((v[1] - 4.0 / rms).abs() < 1e-4, "{v:?}");
    }

    #[test]
    fn with_scale_multiplies_weight() {
        let x = Array::from_slice(&[3.0f32, 4.0], &[1, 2]);
        let w = Array::from_slice(&[2.0f32, 0.5], &[2]);
        let n = GemmaRmsNorm::from_weight(w, 0.0);
        let y = n.forward(&x).unwrap();
        let v: Vec<f32> = y.as_slice::<f32>().to_vec();
        let rms = (12.5f32).sqrt();
        assert!((v[0] - 2.0 * (3.0 / rms)).abs() < 1e-4, "{v:?}");
        assert!((v[1] - 0.5 * (4.0 / rms)).abs() < 1e-4, "{v:?}");
    }
}
