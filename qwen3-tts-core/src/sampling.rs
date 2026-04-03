//! CPU-only sampling for TTS codec generation.
//!
//! TTS vocab is only 3072 tokens — CPU sampling takes microseconds.
//! This avoids backend-specific GPU sampling dependencies.

use rand::Rng;

/// Sample a token from logits using temperature, top-k, top-p.
pub fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: i32,
    top_p: f32,
    do_sample: bool,
    rng: &mut impl Rng,
) -> u32 {
    if !do_sample {
        return argmax(logits);
    }

    let mut probs: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();

    // Temperature scaling
    if temperature > 0.0 && temperature != 1.0 {
        for p in &mut probs {
            p.1 /= temperature;
        }
    }

    // Top-k filtering
    if top_k > 0 && (top_k as usize) < probs.len() {
        probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        probs.truncate(top_k as usize);
    }

    // Softmax
    let max_logit = probs.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for p in &mut probs {
        p.1 = (p.1 - max_logit).exp();
        sum += p.1;
    }
    for p in &mut probs {
        p.1 /= sum;
    }

    // Top-p (nucleus) filtering
    if top_p > 0.0 && top_p < 1.0 {
        probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumsum = 0.0;
        let mut cutoff = probs.len();
        for (i, p) in probs.iter().enumerate() {
            cumsum += p.1;
            if cumsum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);

        // Re-normalize
        let sum: f32 = probs.iter().map(|p| p.1).sum();
        for p in &mut probs {
            p.1 /= sum;
        }
    }

    // Multinomial sampling
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for &(idx, prob) in &probs {
        cumsum += prob;
        if r < cumsum {
            return idx as u32;
        }
    }

    // Fallback to last
    probs.last().map(|p| p.0 as u32).unwrap_or(0)
}

/// Greedy argmax.
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Apply repetition penalty to logits in-place.
pub fn apply_repetition_penalty(logits: &mut [f32], prev_tokens: &[u32], penalty: f32) {
    if penalty <= 1.0 {
        return;
    }
    for &tok in prev_tokens {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Suppress special tokens (>= 2048 except EOS) by setting logits to -inf.
pub fn suppress_special_tokens(logits: &mut [f32], eos_token: u32) {
    for (i, logit) in logits.iter_mut().enumerate() {
        let id = i as u32;
        if id >= 2048 && id != eos_token {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Compute EOS steering bias for speed control.
///
/// Ramps from negative (suppress EOS early) to positive (encourage EOS late)
/// based on the ratio of generated frames to expected frames.
pub fn compute_eos_steering_bias(
    gen_frames: usize,
    expected_frames: usize,
    speed_factor: f32,
) -> f32 {
    if speed_factor >= 1.0 || expected_frames == 0 {
        return 0.0;
    }

    let target = expected_frames as f32 / speed_factor;
    let ratio = gen_frames as f32 / target;

    // Ramp: suppress EOS below 60% of target, encourage above 140%
    if ratio < 0.6 {
        -10.0 // strongly suppress EOS
    } else if ratio < 1.0 {
        // Linear ramp from -10 to 0 between 60%-100%
        -10.0 * (1.0 - (ratio - 0.6) / 0.4)
    } else if ratio < 1.4 {
        // Linear ramp from 0 to 10 between 100%-140%
        10.0 * (ratio - 1.0) / 0.4
    } else {
        10.0 // strongly encourage EOS
    }
}
