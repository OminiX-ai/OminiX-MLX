use mlx_rs::{array, error::Exception, ops::indexing::IndexOp, Array};

/// Build a pre-computed suppression mask as an Array.
/// Shape [vocab_size] with 0.0 for allowed tokens and -inf for suppressed tokens.
/// Suppresses [2048, 3072) except codec_eos.
pub fn build_suppression_mask(vocab_size: usize, eos_token: u32) -> Array {
    let mut mask = vec![0.0f32; vocab_size];
    for i in 2048..vocab_size.min(3072) {
        if i as u32 != eos_token {
            mask[i] = f32::NEG_INFINITY;
        }
    }
    Array::from_slice(&mask, &[vocab_size as i32])
}

/// Build an EOS suppression mask (for min_new_tokens enforcement).
/// Shape [vocab_size] with 0.0 for all tokens except -inf at eos_token.
pub fn build_eos_suppression_mask(vocab_size: usize, eos_token: u32) -> Array {
    let mut mask = vec![0.0f32; vocab_size];
    if (eos_token as usize) < vocab_size {
        mask[eos_token as usize] = f32::NEG_INFINITY;
    }
    Array::from_slice(&mask, &[vocab_size as i32])
}

/// Build an EOS unit mask for dynamic logit steering.
/// Shape [vocab_size] with 1.0 at eos_token and 0.0 elsewhere.
/// Multiply by a bias value and add to logits to encourage/suppress EOS.
pub fn build_eos_unit_mask(vocab_size: usize, eos_token: u32) -> Array {
    let mut mask = vec![0.0f32; vocab_size];
    if (eos_token as usize) < vocab_size {
        mask[eos_token as usize] = 1.0;
    }
    Array::from_slice(&mask, &[vocab_size as i32])
}

/// Compute EOS logit steering bias for a given generation step.
///
/// Based on the "Segment-Aware Conditioning" approach (arxiv 2601.03170):
/// - Before 60% of target: suppress EOS (negative bias)
/// - 60%-100% of target: linear ramp from suppression to neutral
/// - 100%-140% of target: linear ramp from neutral to encouragement
/// - After 140% of target: strong EOS encouragement
///
/// Returns a bias value to multiply with the EOS unit mask.
pub fn compute_eos_steering_bias(step: usize, target_frames: usize, speed_factor: f32) -> f32 {
    if (speed_factor - 1.0).abs() < 0.01 || target_frames == 0 {
        return 0.0;
    }

    let t = step as f32;
    let target = target_frames as f32;

    // The model's EOS logit is typically very negative (-20 to -40) during
    // active speech, so we need strong biases to have any effect.
    let suppress_strength = -30.0; // suppress EOS before target
    let encourage_strength = 40.0; // encourage EOS after target

    let phase_start = 0.6 * target; // start ramping from suppression
    let phase_end = 1.4 * target; // max encouragement

    if t < phase_start {
        // Strong EOS suppression — don't stop early
        suppress_strength
    } else if t < target {
        // Linear ramp: suppress → neutral
        let progress = (t - phase_start) / (target - phase_start);
        suppress_strength * (1.0 - progress)
    } else if t < phase_end {
        // Linear ramp: neutral → encourage
        let progress = (t - target) / (phase_end - target);
        encourage_strength * progress
    } else {
        // Strong EOS encouragement — time to stop
        encourage_strength
    }
}

/// PRNG state for seeded sampling. Splits key after each sample.
pub struct SamplingKey {
    key: Array,
}

impl SamplingKey {
    /// Create a new sampling key from a seed.
    pub fn new(seed: u64) -> Result<Self, Exception> {
        let key = mlx_rs::random::key(seed)?;
        Ok(Self { key })
    }

    /// Sample from categorical distribution using this key, then advance state.
    pub fn categorical(&mut self, logits: &Array) -> Result<Array, Exception> {
        let (k1, k2) = mlx_rs::random::split(&self.key, 2)?;
        mlx_rs::transforms::eval([&k1, &k2])?;
        let token = mlx_rs::random::categorical(logits, None, None, &k1)?;
        self.key = k2;
        Ok(token)
    }
}

/// GPU-resident repetition penalty mask.
/// Tracks which tokens have been generated and applies penalty without CPU roundtrips.
pub struct RepetitionPenaltyMask {
    /// Boolean mask [vocab_size]: true where token has been generated
    mask: Array,
    /// Pre-computed index array [0, 1, 2, ..., vocab-1] for one_hot creation
    indices: Array,
    /// The penalty value
    penalty: f32,
}

impl RepetitionPenaltyMask {
    /// Create a new penalty mask for the given vocab size and penalty factor.
    pub fn new(vocab_size: usize, penalty: f32) -> Result<Self, Exception> {
        let mask = Array::zeros::<f32>(&[vocab_size as i32])?;
        let indices = Array::arange::<_, i32>(None, vocab_size as i32, None)?;
        Ok(Self { mask, indices, penalty })
    }

    /// Record that a token was generated (updates the mask on GPU).
    pub fn record_token(&mut self, token: u32) -> Result<(), Exception> {
        let token_arr = Array::from_int(token as i32);
        let one_hot = self.indices.eq(&token_arr)?.as_dtype(mlx_rs::Dtype::Float32)?;
        self.mask = mlx_rs::ops::maximum(&self.mask, &one_hot)?;
        Ok(())
    }

    /// Apply repetition penalty to logits (all GPU ops, no CPU transfer).
    /// For tokens in the mask: positive logits are divided by penalty, negative logits are multiplied.
    pub fn apply(&self, logits: &Array) -> Result<Array, Exception> {
        if self.penalty == 1.0 {
            return Ok(logits.clone());
        }
        let penalty = array!(self.penalty);
        let zero = array!(0.0f32);

        // logits > 0 && mask > 0 → logits / penalty
        // logits <= 0 && mask > 0 → logits * penalty
        // mask == 0 → logits unchanged
        let positive = logits.gt(&zero)?;
        let in_mask = self.mask.gt(&zero)?;

        let divided = logits.divide(&penalty)?;
        let multiplied = logits.multiply(&penalty)?;

        // where(mask > 0, where(logits > 0, divided, multiplied), logits)
        let penalized = mlx_rs::ops::r#where(&positive, &divided, &multiplied)?;
        mlx_rs::ops::r#where(&in_mask, &penalized, logits)
    }
}

/// Sample a token from logits with temperature, top-k, top-p, repetition penalty,
/// and control token suppression.
/// If `rng_key` is Some, uses seeded sampling; otherwise uses global RNG.
pub fn sample_logits(
    logits: &Array,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    repetition_penalty: f32,
    generated_tokens: &[u32],
    rng_key: Option<&mut SamplingKey>,
) -> Result<u32, Exception> {
    sample_logits_with_mask(logits, temperature, top_k, top_p, repetition_penalty, generated_tokens, rng_key, None, None)
}

/// Full-featured sampling with pre-built suppression mask and GPU penalty mask.
pub fn sample_logits_with_mask(
    logits: &Array,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    repetition_penalty: f32,
    generated_tokens: &[u32],
    rng_key: Option<&mut SamplingKey>,
    suppress_mask: Option<&Array>,
    penalty_mask: Option<&RepetitionPenaltyMask>,
) -> Result<u32, Exception> {
    // logits shape: [1, 1, vocab] or [1, vocab] or [vocab]
    // Ensure Float32 — quantized codec_head may produce BFloat16
    let logits_f32 = logits.as_dtype(mlx_rs::Dtype::Float32)?;
    let mut logits = if logits_f32.ndim() == 3 {
        logits_f32.index((0, -1, ..))
    } else if logits_f32.ndim() == 2 {
        logits_f32.index((0, ..))
    } else {
        logits_f32
    };

    // Apply suppression mask (GPU addition, no CPU roundtrip)
    if let Some(mask) = suppress_mask {
        logits = logits.add(mask)?;
    }

    // Apply repetition penalty (GPU path if mask provided, CPU fallback otherwise)
    if let Some(pm) = penalty_mask {
        logits = pm.apply(&logits)?;
    } else if repetition_penalty != 1.0 && !generated_tokens.is_empty() {
        mlx_rs::transforms::eval(std::iter::once(&logits))?;
        logits = apply_repetition_penalty(&logits, generated_tokens, repetition_penalty)?;
    }

    if temperature == 0.0 {
        // Greedy
        let token = mlx_rs::ops::indexing::argmax_axis(&logits, -1, None)?;
        mlx_rs::transforms::eval(std::iter::once(&token))?;
        return Ok(token.item::<u32>());
    }

    // Temperature scaling
    logits = logits.multiply(array!(1.0f32 / temperature))?;

    // Top-k filtering (GPU-resident, no CPU roundtrip)
    if top_k > 0 {
        logits = apply_top_k(&logits, top_k)?;
    }

    // Top-p (nucleus) filtering (GPU-resident, no CPU roundtrip)
    if top_p > 0.0 && top_p < 1.0 {
        logits = apply_top_p(&logits, top_p)?;
    }

    // Sample from categorical distribution
    let token = if let Some(key) = rng_key {
        key.categorical(&logits)?
    } else {
        mlx_rs::random::categorical(&logits, None, None, None)?
    };
    mlx_rs::transforms::eval(std::iter::once(&token))?;
    Ok(token.item::<u32>())
}

fn apply_repetition_penalty(
    logits: &Array,
    tokens: &[u32],
    penalty: f32,
) -> Result<Array, Exception> {
    let mut logits_vec: Vec<f32> = logits.as_slice::<f32>().to_vec();
    for &tok in tokens {
        let idx = tok as usize;
        if idx < logits_vec.len() {
            if logits_vec[idx] > 0.0 {
                logits_vec[idx] /= penalty;
            } else {
                logits_vec[idx] *= penalty;
            }
        }
    }
    let vocab = logits_vec.len() as i32;
    Ok(Array::from_slice(&logits_vec, &[vocab]))
}

/// GPU-resident top-k: keep only the k largest logits, mask rest to -inf.
fn apply_top_k(logits: &Array, k: i32) -> Result<Array, Exception> {
    // topk returns the k largest values (unsorted)
    let top_values = mlx_rs::ops::indexing::topk(logits, k)?;
    // Threshold = minimum of the top-k values
    let threshold = top_values.min_axis(-1, true)?;
    // Mask everything below threshold
    let below = logits.lt(&threshold)?;
    mlx_rs::ops::r#where(&below, &array!(f32::NEG_INFINITY), logits)
}

/// GPU-resident top-p (nucleus) sampling: keep smallest set of tokens with cumulative probability >= p.
fn apply_top_p(logits: &Array, p: f32) -> Result<Array, Exception> {
    // Sort logits descending (negate → ascending sort → negate back)
    let sorted = mlx_rs::ops::sort_axis(&logits.negative()?, -1)?.negative()?;

    // Softmax on sorted logits → probabilities in descending order
    let sorted_probs = mlx_rs::ops::softmax_axis(&sorted, -1, None::<bool>)?;

    // Cumulative sum of sorted probabilities
    let cum_probs = mlx_rs::ops::cumsum(&sorted_probs, Some(-1), None, None)?;

    // Mask: remove tokens where cumsum (before this token) >= p.
    // shifted_cum = cumsum - current_prob gives the cumsum BEFORE this token.
    // Keep tokens where shifted_cum < p (includes the token that crosses p).
    let shifted_cum = cum_probs.subtract(&sorted_probs)?;
    let to_remove = shifted_cum.ge(&array!(p))?;

    // Find threshold: the minimum logit value among kept tokens in sorted space.
    // Replace removed tokens with +MAX so they don't affect the min.
    let kept_sorted = mlx_rs::ops::r#where(&to_remove, &array!(f32::MAX), &sorted)?;
    let threshold = kept_sorted.min_axis(-1, true)?;

    // Apply threshold to original (unsorted) logits
    let below = logits.lt(&threshold)?;
    mlx_rs::ops::r#where(&below, &array!(f32::NEG_INFINITY), logits)
}
