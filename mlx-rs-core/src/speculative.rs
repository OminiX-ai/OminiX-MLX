//! Speculative decoding for faster inference
//!
//! Speculative decoding uses a smaller "draft" model to generate candidate tokens,
//! which are then verified by the larger "target" model in parallel. This can
//! significantly speed up inference when the draft model has high acceptance rate.

use mlx_rs::{
    argmax_axis, array, categorical,
    error::Exception,
    ops::indexing::{IndexOp, NewAxis},
    ops::logsumexp_axis,
    transforms::{eval, async_eval},
    Array,
};

use crate::cache::KeyValueCache;

/// Result of speculative decoding step
pub struct SpeculativeToken {
    /// The generated token
    pub token: Array,
    /// Log probabilities
    pub logprobs: Array,
    /// Whether this token was accepted from draft model (true) or from target model (false)
    pub from_draft: bool,
}

/// Speculative decoding generator
///
/// Uses a draft model to speculate multiple tokens ahead, then verifies them
/// with the target model in a single forward pass.
pub struct SpeculativeGenerate<'a, M, D, C>
where
    M: SpeculativeModel,
    D: SpeculativeModel,
    C: KeyValueCache + Default,
{
    /// Target (large) model
    target_model: &'a mut M,
    /// Draft (small) model
    draft_model: &'a mut D,
    /// KV cache for target model
    target_cache: &'a mut Vec<Option<C>>,
    /// KV cache for draft model
    draft_cache: &'a mut Vec<Option<C>>,
    /// Number of draft tokens to generate per step
    num_draft_tokens: usize,
    /// Temperature for sampling
    temperature: f32,
    /// Current state
    state: SpeculativeState<'a>,
    /// Token count
    token_count: usize,
    /// Pending tokens from verification (when draft tokens are accepted)
    pending_tokens: Vec<SpeculativeToken>,
}

enum SpeculativeState<'a> {
    /// Initial state - need to process prompt
    Prefill { prompt: &'a Array },
    /// Main generation loop
    Generate { last_token: Array },
}

/// Trait for models that support speculative decoding
pub trait SpeculativeModel {
    /// Forward pass returning logits
    fn forward_speculative<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception>;
}

impl<'a, M, D, C> SpeculativeGenerate<'a, M, D, C>
where
    M: SpeculativeModel,
    D: SpeculativeModel,
    C: KeyValueCache + Default,
{
    pub fn new(
        target_model: &'a mut M,
        draft_model: &'a mut D,
        target_cache: &'a mut Vec<Option<C>>,
        draft_cache: &'a mut Vec<Option<C>>,
        num_draft_tokens: usize,
        temperature: f32,
        prompt: &'a Array,
    ) -> Self {
        Self {
            target_model,
            draft_model,
            target_cache,
            draft_cache,
            num_draft_tokens,
            temperature,
            state: SpeculativeState::Prefill { prompt },
            token_count: 0,
            pending_tokens: Vec::new(),
        }
    }

    /// Sample a token from logits
    fn sample(&self, logits: &Array) -> Result<Array, Exception> {
        if self.temperature == 0.0 {
            argmax_axis!(logits, -1).map_err(Into::into)
        } else {
            let scaled = logits.multiply(array!(1.0 / self.temperature))?;
            categorical!(scaled).map_err(Into::into)
        }
    }

    /// Generate draft tokens using the draft model
    fn generate_draft_tokens(&mut self, start_token: &Array) -> Result<Vec<Array>, Exception> {
        let mut tokens = Vec::with_capacity(self.num_draft_tokens);
        let mut current = start_token.clone();

        for _ in 0..self.num_draft_tokens {
            let input = current.index((.., NewAxis));
            let logits = self.draft_model.forward_speculative(&input, self.draft_cache)?;
            let token = self.sample(&logits)?;
            let _ = async_eval([&token]);
            tokens.push(token.clone());
            current = token;
        }

        Ok(tokens)
    }

    /// Verify draft tokens with target model
    /// Returns (accepted_count, all_tokens, all_logprobs)
    fn verify_draft_tokens(
        &mut self,
        input_tokens: &Array,
    ) -> Result<(usize, Vec<Array>, Vec<Array>), Exception> {
        // Forward all tokens through target model at once
        let logits = self.target_model.forward_speculative(input_tokens, self.target_cache)?;

        // Get the number of positions (draft tokens + 1 for verification)
        let seq_len = input_tokens.shape()[1] as usize;

        let mut tokens = Vec::with_capacity(seq_len);
        let mut logprobs = Vec::with_capacity(seq_len);

        // Sample from each position's logits
        for i in 0..seq_len {
            let pos_logits = logits.index((.., i as i32, ..));
            let token = self.sample(&pos_logits)?;

            // Compute log probabilities
            let log_sum_exp = logsumexp_axis(&pos_logits, -1, true)?;
            let lp = pos_logits.subtract(&log_sum_exp)?;

            tokens.push(token);
            logprobs.push(lp);
        }

        eval(&tokens)?;

        Ok((seq_len, tokens, logprobs))
    }

    /// Trim cache by removing the last n entries
    #[allow(dead_code)]
    fn trim_cache<Cache: KeyValueCache>(cache: &mut Vec<Option<Cache>>, n: i32) {
        // This is a simplified version - full implementation would need
        // proper cache trimming support in KeyValueCache trait
        let _ = (cache, n);
    }
}

impl<'a, M, D, C> Iterator for SpeculativeGenerate<'a, M, D, C>
where
    M: SpeculativeModel,
    D: SpeculativeModel,
    C: KeyValueCache + Default,
{
    type Item = Result<SpeculativeToken, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        // First return any pending tokens from previous verification
        if let Some(token) = self.pending_tokens.pop() {
            return Some(Ok(token));
        }

        match &self.state {
            SpeculativeState::Prefill { prompt } => {
                // Process prompt through both models
                let prompt = *prompt;

                // Forward through target model
                let target_logits = match self.target_model.forward_speculative(prompt, self.target_cache) {
                    Ok(l) => l,
                    Err(e) => return Some(Err(e)),
                };

                // Forward through draft model
                if let Err(e) = self.draft_model.forward_speculative(prompt, self.draft_cache) {
                    return Some(Err(e));
                }

                // Sample first token from target model
                let first_logits = target_logits.index((.., -1, ..));
                let token = match self.sample(&first_logits) {
                    Ok(t) => t,
                    Err(e) => return Some(Err(e)),
                };

                let log_sum_exp = match logsumexp_axis(&first_logits, -1, true) {
                    Ok(l) => l,
                    Err(e) => return Some(Err(e)),
                };
                let logprobs = match first_logits.subtract(&log_sum_exp) {
                    Ok(l) => l,
                    Err(e) => return Some(Err(e)),
                };

                let _ = eval([&token]);

                self.state = SpeculativeState::Generate { last_token: token.clone() };
                self.token_count = 1;

                Some(Ok(SpeculativeToken {
                    token,
                    logprobs,
                    from_draft: false,
                }))
            }
            SpeculativeState::Generate { last_token } => {
                let last_token = last_token.clone();

                // Generate draft tokens
                let draft_tokens = match self.generate_draft_tokens(&last_token) {
                    Ok(t) => t,
                    Err(e) => return Some(Err(e)),
                };

                // Build input sequence: last_token + draft_tokens
                let mut input_tokens = vec![last_token.index((.., NewAxis))];
                for dt in &draft_tokens {
                    input_tokens.push(dt.index((.., NewAxis)));
                }

                let input_seq = match mlx_rs::ops::concatenate_axis(
                    &input_tokens.iter().collect::<Vec<_>>(),
                    1,
                ) {
                    Ok(s) => s,
                    Err(e) => return Some(Err(e)),
                };

                // Verify with target model
                let (_, target_tokens, target_logprobs) =
                    match self.verify_draft_tokens(&input_seq) {
                        Ok(r) => r,
                        Err(e) => return Some(Err(e)),
                    };

                // Compare draft tokens with target tokens to find acceptance
                let mut accepted = 0;
                for i in 0..draft_tokens.len() {
                    let draft_id = draft_tokens[i].item::<u32>();
                    let target_id = target_tokens[i].item::<u32>();

                    if draft_id == target_id {
                        accepted += 1;
                        // Queue accepted token
                        self.pending_tokens.push(SpeculativeToken {
                            token: draft_tokens[i].clone(),
                            logprobs: target_logprobs[i].clone(),
                            from_draft: true,
                        });
                    } else {
                        break;
                    }
                }

                // The token at position `accepted` is from target model (either correction or next)
                let final_token = target_tokens[accepted].clone();
                let final_logprobs = target_logprobs[accepted].clone();

                // Update state with the final token
                self.state = SpeculativeState::Generate { last_token: final_token.clone() };

                // Reverse pending tokens so they come out in order
                self.pending_tokens.reverse();

                // Return the first accepted token or the corrected token
                if let Some(token) = self.pending_tokens.pop() {
                    self.token_count += 1;
                    Some(Ok(token))
                } else {
                    self.token_count += 1;
                    Some(Ok(SpeculativeToken {
                        token: final_token,
                        logprobs: final_logprobs,
                        from_draft: false,
                    }))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Tests would go here
}
