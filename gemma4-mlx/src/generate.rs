//! Greedy token generation: prefill prompt, then autoregressive decode with KV cache.
use mlx_rs::Array;
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
    // Cast to f32 (model may output bfloat16), eval to materialise, then linear argmax.
    // Pattern copied directly from examples/model_parity.rs lines 106-108.
    let logits_f32 = logits.as_type::<f32>()?;
    logits_f32.eval()?;
    let values: Vec<f32> = logits_f32.as_slice::<f32>().to_vec();
    let idx = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    Ok(idx as i32)
}
