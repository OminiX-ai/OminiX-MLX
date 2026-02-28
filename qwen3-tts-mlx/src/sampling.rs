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

    // Top-k filtering
    if top_k > 0 {
        mlx_rs::transforms::eval(std::iter::once(&logits))?;
        logits = apply_top_k(&logits, top_k)?;
    }

    // Top-p (nucleus) filtering
    if top_p > 0.0 && top_p < 1.0 {
        mlx_rs::transforms::eval(std::iter::once(&logits))?;
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

fn apply_top_k(logits: &Array, k: i32) -> Result<Array, Exception> {
    let logits_vec: Vec<f32> = logits.as_slice::<f32>().to_vec();
    let vocab_size = logits_vec.len();
    if k as usize >= vocab_size {
        return Ok(logits.clone());
    }

    // Find threshold: k-th largest value
    let mut sorted = logits_vec.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k as usize];

    // Mask out values below threshold
    let masked: Vec<f32> = logits_vec
        .iter()
        .map(|&v| if v < threshold { f32::NEG_INFINITY } else { v })
        .collect();
    Ok(Array::from_slice(&masked, &[vocab_size as i32]))
}

/// Top-p (nucleus) sampling: keep smallest set of tokens with cumulative probability >= p.
fn apply_top_p(logits: &Array, p: f32) -> Result<Array, Exception> {
    let logits_vec: Vec<f32> = logits.as_slice::<f32>().to_vec();
    let vocab_size = logits_vec.len();

    // Sort indices by descending logit
    let mut indices: Vec<usize> = (0..vocab_size).collect();
    indices.sort_by(|&a, &b| {
        logits_vec[b]
            .partial_cmp(&logits_vec[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Compute softmax on sorted logits
    let max_logit = logits_vec[indices[0]];
    let sorted_exp: Vec<f32> = indices
        .iter()
        .map(|&i| (logits_vec[i] - max_logit).exp())
        .collect();
    let sum_exp: f32 = sorted_exp.iter().sum();
    let sorted_probs: Vec<f32> = sorted_exp.iter().map(|&e| e / sum_exp).collect();

    // Find cutoff: cumulative probability exceeds p
    let mut cum_prob = 0.0f32;
    let mut cutoff_idx = vocab_size;
    for (i, &prob) in sorted_probs.iter().enumerate() {
        cum_prob += prob;
        if cum_prob >= p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Always keep at least 1 token
    cutoff_idx = cutoff_idx.max(1);

    // Build mask: keep tokens in top-p set, set others to -inf
    let mut keep = vec![false; vocab_size];
    for &idx in &indices[..cutoff_idx] {
        keep[idx] = true;
    }
    let masked: Vec<f32> = logits_vec
        .iter()
        .enumerate()
        .map(|(i, &v)| if keep[i] { v } else { f32::NEG_INFINITY })
        .collect();
    Ok(Array::from_slice(&masked, &[vocab_size as i32]))
}
