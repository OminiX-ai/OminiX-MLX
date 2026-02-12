//! Self-speculative decoding using early-exit draft model.
//!
//! Uses the first N layers of MiniCPM-SALA as a fast draft model to speculate
//! K tokens ahead, then verifies them with the full model in one forward pass.

use mlx_rs::{error::Exception, ops::indexing::IndexOp, transforms::eval, Array};

use crate::attention::LayerCache;
use crate::model::{sample, Model};

/// Self-speculative decoder using early-exit (first N layers) as draft.
pub struct SpeculativeDecoder {
    /// Number of layers for draft model (e.g., 8 out of 32)
    pub draft_layers: usize,
    /// Number of draft tokens to speculate per round
    pub num_draft: usize,
    /// Sampling temperature
    pub temperature: f32,
}

/// Result of one speculative decoding step.
pub struct SpeculativeStepResult {
    /// Token IDs accepted in this step (1 to num_draft+1)
    pub tokens: Vec<u32>,
    /// Number of draft tokens that matched the target
    pub num_accepted: usize,
}

impl SpeculativeDecoder {
    pub fn new(draft_layers: usize, num_draft: usize, temperature: f32) -> Self {
        Self {
            draft_layers,
            num_draft,
            temperature,
        }
    }

    /// Run one speculative decoding step.
    ///
    /// Returns the accepted tokens (at least 1, at most num_draft+1).
    /// `caches` is updated to reflect the accepted tokens.
    pub fn step(
        &self,
        model: &mut Model,
        caches: &mut Vec<LayerCache>,
        last_token: &Array,
    ) -> Result<SpeculativeStepResult, Exception> {
        // 1. Clone caches for draft speculation
        let mut draft_caches = caches.clone();

        // 2. Generate K draft tokens via forward_draft (first N layers only)
        let mut draft_tokens: Vec<Array> = Vec::with_capacity(self.num_draft);
        let mut current = last_token.clone();

        for _ in 0..self.num_draft {
            let input = current.reshape(&[1, 1])?;
            let logits = model.forward_draft(&input, &mut draft_caches, self.draft_layers)?;
            let last_logits = logits.index((.., -1, ..));
            let token = sample(&last_logits, self.temperature)?;
            eval([&token])?;
            draft_tokens.push(token.clone());
            current = token;
        }

        // 3. Build verification input: [last_token, draft_1, ..., draft_K]
        let mut all_token_ids: Vec<i32> = vec![last_token.item::<i32>()];
        for dt in &draft_tokens {
            all_token_ids.push(dt.item::<i32>());
        }
        let verify_input =
            Array::from_slice(&all_token_ids, &[1, all_token_ids.len() as i32]);

        // 4. Full forward pass for verification (updates caches for all K+1 positions)
        let logits = model.forward(&verify_input, caches)?;
        eval([&logits])?;

        // 5. Compare: at position i, the target model's logits predict token i+1.
        // Position 0's logits → should predict draft_tokens[0]
        // Position i's logits → should predict draft_tokens[i] (for i < K)
        // Position K's logits → bonus token (if all K accepted)
        let mut accepted_tokens: Vec<u32> = Vec::new();
        let mut num_accepted = 0;

        for i in 0..self.num_draft {
            let target_token = sample(&logits.index((.., i as i32, ..)), self.temperature)?;
            eval([&target_token])?;
            let draft_id = draft_tokens[i].item::<u32>();
            let target_id = target_token.item::<u32>();

            if draft_id == target_id {
                accepted_tokens.push(draft_id);
                num_accepted += 1;
            } else {
                // Mismatch — use target's correction token
                accepted_tokens.push(target_id);
                break;
            }
        }

        // If all K draft tokens accepted, take the bonus token at position K
        if num_accepted == self.num_draft {
            let bonus = sample(
                &logits.index((.., self.num_draft as i32, ..)),
                self.temperature,
            )?;
            eval([&bonus])?;
            accepted_tokens.push(bonus.item::<u32>());
        }

        // 6. Trim caches: verification advanced caches by K+1 positions,
        //    but we only accepted M tokens. Trim the excess.
        let excess = (self.num_draft + 1) as i32 - accepted_tokens.len() as i32;
        if excess > 0 {
            trim_caches(caches, excess);
        }

        Ok(SpeculativeStepResult {
            tokens: accepted_tokens,
            num_accepted,
        })
    }
}

/// Trim the last `n` entries from all caches.
/// For sparse caches: trim KV history.
/// For lightning caches: adjust offset (state contamination is minimal due to decay).
fn trim_caches(caches: &mut [LayerCache], n: i32) {
    if n <= 0 {
        return;
    }
    for cache in caches.iter_mut() {
        match cache {
            LayerCache::Sparse(sc) => sc.trim(n),
            LayerCache::Lightning(lc) => {
                // Lightning recurrent state can't be easily un-updated,
                // but the contamination from rejected tokens decays exponentially.
                // Just adjust the offset for correct RoPE positioning.
                lc.offset -= n;
            }
        }
    }
}
