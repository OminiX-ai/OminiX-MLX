//! Sampling utilities for autoregressive token generation
//!
//! Provides configurable sampling strategies including top-k, top-p (nucleus),
//! temperature scaling, and repetition penalty. Matches Python GPT-SoVITS
//! sampling order exactly.

use std::collections::HashSet;

use mlx_rs::{Array, ops, random, transforms::eval};

use crate::error::Error;

/// Configuration for token sampling
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Number of top tokens to consider (0 or negative = disabled)
    pub top_k: i32,
    /// Nucleus sampling threshold (1.0 = disabled)
    pub top_p: f32,
    /// Temperature for softmax (1.0 = no scaling)
    pub temperature: f32,
    /// Penalty for repeating tokens (1.0 = disabled)
    pub repetition_penalty: f32,
    /// EOS token ID (typically 1024 for GPT-SoVITS)
    pub eos_token: i32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            top_k: 15,
            top_p: 1.0,
            temperature: 1.0,
            repetition_penalty: 1.35,
            eos_token: 1024,
        }
    }
}

/// Stateful sampler that tracks previous tokens for repetition penalty
pub struct Sampler {
    config: SamplingConfig,
    previous_tokens: Vec<i32>,
}

impl Sampler {
    /// Create a new sampler with the given configuration
    pub fn new(config: SamplingConfig) -> Self {
        Self {
            config,
            previous_tokens: Vec::new(),
        }
    }

    /// Create a sampler with default configuration
    pub fn with_defaults() -> Self {
        Self::new(SamplingConfig::default())
    }

    /// Reset the sampler state (clear previous tokens)
    pub fn reset(&mut self) {
        self.previous_tokens.clear();
    }

    /// Get the current configuration
    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }

    /// Get mutable reference to configuration
    pub fn config_mut(&mut self) -> &mut SamplingConfig {
        &mut self.config
    }

    /// Get the previous tokens
    pub fn previous_tokens(&self) -> &[i32] {
        &self.previous_tokens
    }

    /// Sample from logits, returning (sampled_token, argmax_token)
    ///
    /// This applies the full sampling pipeline:
    /// 1. Repetition penalty
    /// 2. Top-p filtering (nucleus sampling)
    /// 3. Temperature scaling
    /// 4. Top-k filtering
    /// 5. Softmax
    /// 6. Categorical sampling
    pub fn sample(&mut self, logits: &Array) -> Result<(i32, i32), Error> {
        self.sample_internal(logits, false)
    }

    /// Sample with EOS token masked out (for early generation steps)
    ///
    /// Python masks EOS during the first ~11 tokens to prevent early stopping.
    pub fn sample_with_eos_mask(&mut self, logits: &Array) -> Result<(i32, i32), Error> {
        self.sample_internal(logits, true)
    }

    /// Add a token to the history (call after accepting a sampled token)
    pub fn add_token(&mut self, token: i32) {
        self.previous_tokens.push(token);
    }

    /// Internal sampling implementation
    fn sample_internal(&mut self, logits: &Array, mask_eos: bool) -> Result<(i32, i32), Error> {
        let mut logits_vec: Vec<f32> = logits
            .flatten(None, None)
            .map_err(|e| Error::Message(e.to_string()))?
            .as_slice()
            .to_vec();

        // Mask EOS token during early generation BEFORE computing argmax
        // Python does: if(idx<11): logits = logits[:, :-1]  (masks EOS before argmax check)
        if mask_eos && (self.config.eos_token as usize) < logits_vec.len() {
            logits_vec[self.config.eos_token as usize] = f32::NEG_INFINITY;
        }

        // Get argmax AFTER EOS masking - this matches Python's behavior
        let argmax_token = argmax(&logits_vec);

        // 1. Apply repetition penalty
        self.apply_repetition_penalty(&mut logits_vec);

        // 2. Top-p filtering (on logits, before temperature)
        if self.config.top_p < 1.0 && self.config.top_p > 0.0 {
            apply_top_p_filter(&mut logits_vec, self.config.top_p);
        }

        // 3. Temperature scaling
        if self.config.temperature != 1.0 {
            let t = self.config.temperature.max(1e-5);
            for v in logits_vec.iter_mut() {
                *v /= t;
            }
        }

        // 4. Top-k filtering
        let effective_k = if self.config.top_k <= 0 {
            logits_vec.len()
        } else {
            self.config.top_k as usize
        };
        if effective_k < logits_vec.len() {
            apply_top_k_filter(&mut logits_vec, effective_k);
        }

        // 5. Softmax
        let probs = softmax(&logits_vec);

        // 6. Categorical sample
        let sampled_token = categorical_sample(&probs)?;

        Ok((sampled_token, argmax_token))
    }

    /// Apply repetition penalty to logits
    fn apply_repetition_penalty(&self, logits: &mut [f32]) {
        if self.config.repetition_penalty == 1.0 || self.previous_tokens.is_empty() {
            return;
        }

        let used_tokens: HashSet<i32> = self.previous_tokens.iter().cloned().collect();

        for &token in &used_tokens {
            if token >= 0 && (token as usize) < logits.len() {
                let score = logits[token as usize];
                // Penalize: if score < 0, multiply by penalty; if score > 0, divide by penalty
                logits[token as usize] = if score < 0.0 {
                    score * self.config.repetition_penalty
                } else {
                    score / self.config.repetition_penalty
                };
            }
        }
    }
}

/// Apply top-p (nucleus) filtering to logits
fn apply_top_p_filter(logits: &mut [f32], top_p: f32) {
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Compute cumulative softmax probs on sorted logits
    let max_val = indexed[0].1;
    let sorted_probs: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_val).exp()).collect();
    let sum: f32 = sorted_probs.iter().sum();

    let mut cumsum = 0.0f32;
    let mut remove_set = HashSet::new();

    for (i, &(orig_idx, _)) in indexed.iter().enumerate() {
        cumsum += sorted_probs[i] / sum;
        if cumsum > top_p {
            remove_set.insert(orig_idx);
        }
    }

    for idx in remove_set {
        logits[idx] = f32::NEG_INFINITY;
    }
}

/// Apply top-k filtering to logits
fn apply_top_k_filter(logits: &mut [f32], k: usize) {
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let pivot = indexed[k.min(indexed.len()) - 1].1;

    for v in logits.iter_mut() {
        if *v < pivot {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Compute softmax of logits
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|v| (v - max_logit).exp()).sum();
    logits
        .iter()
        .map(|v| (v - max_logit).exp() / exp_sum)
        .collect()
}

/// Find argmax of a slice
fn argmax(values: &[f32]) -> i32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

/// Sample from a probability distribution using MLX random
fn categorical_sample(probs: &[f32]) -> Result<i32, Error> {
    // Collect valid (non-zero) probabilities
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_items: Vec<(usize, f32)> = indexed.into_iter().filter(|(_, p)| *p > 0.0).collect();

    if top_items.is_empty() {
        return Ok(0);
    }

    let total: f32 = top_items.iter().map(|(_, p)| p).sum();
    let normalized: Vec<f32> = top_items.iter().map(|(_, p)| p / total).collect();

    // Use MLX random for reproducibility
    let rand_arr =
        random::uniform::<f32, f32>(0.0, 1.0, &[], None).map_err(|e| Error::Message(e.to_string()))?;
    eval([&rand_arr]).map_err(|e| Error::Message(e.to_string()))?;
    let r: f32 = rand_arr.item();

    let mut cumsum = 0.0f32;
    for (i, p) in normalized.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return Ok(top_items[i].0 as i32);
        }
    }

    Ok(top_items[0].0 as i32)
}

/// Detect n-gram repetition in token sequence
///
/// Returns true if the last n tokens appear at least `min_count` times in the sequence.
pub fn detect_repetition(tokens: &[i32], n: usize, min_count: usize) -> bool {
    if tokens.len() < n * 2 {
        return false;
    }
    let last_n: Vec<i32> = tokens[tokens.len() - n..].to_vec();
    tokens
        .windows(n)
        .filter(|w| *w == last_n.as_slice())
        .count()
        >= min_count
}

/// Simple top-k sampling without repetition penalty (for simpler use cases)
pub fn sample_top_k_simple(logits: &Array, top_k: i32, temperature: f32) -> Result<i32, Error> {
    let scaled = if temperature != 1.0 {
        logits
            .divide(mlx_rs::array!(temperature))
            .map_err(|e| Error::Message(e.to_string()))?
    } else {
        logits.clone()
    };
    eval([&scaled]).map_err(|e| Error::Message(e.to_string()))?;

    let flat_logits = scaled
        .flatten(None, None)
        .map_err(|e| Error::Message(e.to_string()))?;
    eval([&flat_logits]).map_err(|e| Error::Message(e.to_string()))?;

    let probs = ops::softmax_axis(&flat_logits, -1, None).map_err(|e| Error::Message(e.to_string()))?;
    eval([&probs]).map_err(|e| Error::Message(e.to_string()))?;

    let prob_vec: Vec<f32> = probs.as_slice().to_vec();

    let mut indexed: Vec<(usize, f32)> = prob_vec.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let effective_k = if top_k <= 0 {
        indexed.len()
    } else {
        top_k as usize
    };
    let top_k_items: Vec<(usize, f32)> = indexed.into_iter().take(effective_k).collect();

    let total: f32 = top_k_items.iter().map(|(_, p)| p).sum();
    let normalized: Vec<f32> = top_k_items.iter().map(|(_, p)| p / total).collect();

    let rand_arr =
        random::uniform::<f32, f32>(0.0, 1.0, &[], None).map_err(|e| Error::Message(e.to_string()))?;
    eval([&rand_arr]).map_err(|e| Error::Message(e.to_string()))?;
    let r: f32 = rand_arr.item();

    let mut cumsum = 0.0f32;
    for (i, p) in normalized.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return Ok(top_k_items[i].0 as i32);
        }
    }

    Ok(top_k_items[0].0 as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_repetition() {
        assert!(!detect_repetition(&[1, 2, 3], 2, 2));
        assert!(detect_repetition(&[1, 2, 1, 2], 2, 2));
        assert!(!detect_repetition(&[1, 2, 3, 4], 2, 2));
        assert!(detect_repetition(&[1, 2, 3, 1, 2, 3], 3, 2));
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[1.0, 2.0, 5.0]), 2);
    }

    #[test]
    fn test_sampling_config_default() {
        let config = SamplingConfig::default();
        assert_eq!(config.top_k, 15);
        assert_eq!(config.top_p, 1.0);
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.repetition_penalty, 1.35);
        assert_eq!(config.eos_token, 1024);
    }
}
