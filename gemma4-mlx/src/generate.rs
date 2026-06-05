//! Greedy token generation: prefill prompt, then autoregressive decode with KV cache.
use mlx_rs::{ops::indexing::argmax, Array};
use crate::model::Gemma4TextModel;
use crate::error::Result;

/// Greedy-decode up to `max_new` tokens after `prompt` (i32 token ids).
/// Returns generated ids (NOT including the prompt). Stops early if an `eos` id is produced.
pub fn generate_greedy(
    model: &mut Gemma4TextModel,
    prompt: &[i32],
    max_new: usize,
    eos: &[i32],
) -> Result<Vec<i32>> {
    let mut caches = model.new_caches();            // ONCE — reused across prefill + all decode steps
    let ptoks = Array::from_slice(prompt, &[1, prompt.len() as i32]);
    let mut logits = model.forward_cached(&ptoks, &mut caches, true)?;   // prefill → [1,1,vocab]
    let mut out = Vec::with_capacity(max_new);
    for _ in 0..max_new {
        let next = argmax_last(&logits)?;
        if eos.contains(&next) { break; }
        out.push(next);
        let t = Array::from_slice(&[next], &[1, 1]);
        logits = model.forward_cached(&t, &mut caches, true)?;           // decode 1 token
    }
    Ok(out)
}

/// argmax over the last dim of a [1,1,vocab] logits array.
fn argmax_last(logits: &Array) -> Result<i32> {
    Ok(argmax(logits, None)?.item::<i32>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_last_matches_cpu_scan() {
        let logits = Array::from_slice(&[-1.0f32, 0.25, 3.5, 3.0], &[1, 1, 4]);

        assert_eq!(argmax_last(&logits).unwrap(), 2);
    }
}
