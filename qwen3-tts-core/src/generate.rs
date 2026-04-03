//! Backend-agnostic autoregressive generation loop for Qwen3-TTS.
//!
//! Each backend does its own prefill (building input embeddings),
//! then delegates to `run_generation_loop()` for the shared decode logic.

use rand::SeedableRng;
use tracing::info;

use crate::backend::TalkerBackend;
use crate::config::GenerationConfig;
use crate::error::Result;
use crate::sampling;

/// Timing information for each phase of generation.
#[derive(Debug, Clone, Default)]
pub struct GenerationTiming {
    pub prefill_ms: f64,
    pub generation_ms: f64,
    pub generation_frames: usize,
}

/// Average ratio of generated codec frames to text tokens.
pub const AVG_FRAMES_PER_TEXT_TOKEN: f32 = 4.0;

/// Parameters for the autoregressive generation loop.
pub struct GenerationLoopParams {
    pub trailing_len: usize,
    pub eos_token: u32,
    pub pad_id: u32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub max_new_tokens: usize,
    pub min_new_tokens: usize,
    /// Optional EOS steering: (target_frames, speed_factor)
    pub eos_steering: Option<(usize, f32)>,
}

impl GenerationLoopParams {
    pub fn from_config(
        gen_config: &GenerationConfig,
        eos_token: u32,
        pad_id: u32,
        trailing_len: usize,
    ) -> Self {
        let speed = gen_config.speed_factor;
        let eos_steering = if (speed - 1.0).abs() > 0.01 {
            let t = (trailing_len as f32 * AVG_FRAMES_PER_TEXT_TOKEN / speed) as usize;
            info!(
                "EOS steering: target_frames={}, speed={:.2}x",
                t, speed
            );
            Some((t, speed))
        } else {
            None
        };

        Self {
            trailing_len,
            eos_token,
            pad_id,
            temperature: gen_config.temperature,
            top_k: gen_config.top_k,
            top_p: gen_config.top_p,
            repetition_penalty: gen_config.repetition_penalty,
            max_new_tokens: gen_config.max_new_tokens as usize,
            min_new_tokens: 20,  // suppress EOS for first 20 steps to let model stabilize
            eos_steering,
        }
    }
}

/// Run the backend-agnostic autoregressive generation loop.
///
/// Takes initial logits/hidden from prefill (backend-specific) and generates
/// codec frames until EOS or max_new_tokens.
///
/// `logits`: flat f32 of length `vocab_size` (from last prefill position).
/// `hidden`: flat f32 of length `hidden_size` (from last prefill position).
/// `trailing_text_embeds`: pre-computed text embeddings for remaining text tokens.
///   Flat f32 of length `trailing_len * hidden_size`.
/// `tts_pad_embed`: pre-computed tts_pad embedding, length `hidden_size`.
pub fn run_generation_loop(
    talker: &mut impl TalkerBackend,
    initial_logits: &[f32],
    initial_hidden: &[f32],
    trailing_text_embeds: &[f32],
    tts_pad_embed: &[f32],
    params: &GenerationLoopParams,
    seed: Option<u64>,
) -> Result<Vec<[u32; 16]>> {
    let vocab_size = talker.vocab_size();
    let hidden_size = talker.hidden_size();

    let mut rng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    };

    // Working copies of logits/hidden (updated each step)
    let mut logits = initial_logits.to_vec();
    let mut hidden = initial_hidden.to_vec();
    let mut prev_tokens: Vec<u32> = Vec::new();
    let mut all_codes: Vec<[u32; 16]> = Vec::new();

    for step in 0..params.max_new_tokens {
        // 1. Apply suppression mask (special tokens >= 2048 except EOS)
        sampling::suppress_special_tokens(&mut logits, params.eos_token);

        // 2. Apply repetition penalty
        sampling::apply_repetition_penalty(
            &mut logits,
            &prev_tokens,
            params.repetition_penalty,
        );

        // 3. Suppress EOS in early steps
        if step < params.min_new_tokens {
            let eos_idx = params.eos_token as usize;
            if eos_idx < logits.len() {
                logits[eos_idx] = f32::NEG_INFINITY;
            }
        }

        // 4. EOS steering bias (speed control)
        if let Some((target, speed)) = params.eos_steering {
            if step >= params.min_new_tokens {
                let bias = sampling::compute_eos_steering_bias(step, target, speed);
                if bias.abs() > 0.01 {
                    let eos_idx = params.eos_token as usize;
                    if eos_idx < logits.len() {
                        logits[eos_idx] += bias;
                    }
                }
            }
        }

        // 5. Sample token 0
        let token0 = sampling::sample_token(
            &logits,
            params.temperature,
            params.top_k,
            params.top_p,
            params.temperature > 0.0,
            &mut rng,
        );

        if token0 == params.eos_token {
            let eos_logit = logits[params.eos_token as usize];
            info!("EOS at step {}, logits[eos]={:.3}, min_new_tokens={}", step, eos_logit, params.min_new_tokens);
            if step < params.min_new_tokens {
                // This should be impossible — EOS was suppressed
                panic!("BUG: EOS sampled at step {} < min_new_tokens {} with logit={:.3}", step, params.min_new_tokens, eos_logit);
            }
            break;
        }

        if step < 5 {
            info!("Step {}: token0={}, logits[0..3]={:.3} {:.3} {:.3}, logits[eos]={:.3}",
                step, token0, logits[0], logits[1], logits[2], logits[params.eos_token as usize]);
        }

        prev_tokens.push(token0);

        // 6. Generate codebooks 1-15 via code predictor
        let sub_codes = talker.predict_sub_codes(&hidden, token0)?;

        let mut frame = [params.pad_id; 16];
        frame[0] = token0;
        for (g, &code) in sub_codes.iter().enumerate() {
            if g + 1 < 16 {
                frame[g + 1] = code;
            }
        }
        all_codes.push(frame);

        // 7. Build next input embedding: codec_embed(prev_codes) + trailing_text
        let text_embed = if step < params.trailing_len {
            let offset = step * hidden_size;
            &trailing_text_embeds[offset..offset + hidden_size]
        } else {
            tts_pad_embed
        };

        let next_embed = talker.generation_embed(text_embed, &frame)?;

        // 8. Forward one step through transformer
        let (new_logits, new_hidden) = talker.forward_step(&next_embed, 1)?;
        logits = new_logits;
        hidden = new_hidden;
    }

    Ok(all_codes)
}
