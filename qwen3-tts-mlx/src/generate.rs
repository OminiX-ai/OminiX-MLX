//! Dual-track autoregressive generation for Qwen3-TTS.
//!
//! Streaming batched prefill (CustomVoice mode):
//!   Pos 0-2:   role tokens [im_start, assistant, \n] — text projection only, NO codec
//!   Pos 3-7:   tts_pad + codec_embedding([think, think_bos, lang, think_eos, spk])
//!   Pos 8:     tts_bos + codec_pad
//!   Pos 9:     first_text + codec_bos
//!
//! All 10 prefill positions processed in ONE forward pass with causal mask.
//!
//! Autoregressive (streaming text):
//!   Frame i:   codec_embed(prev_codes) + trailing_text[i]
//!   Where trailing_text = [text_token_1, ..., text_token_N-1, tts_eos, tts_pad, tts_pad, ...]

use std::time::Instant;

use mlx_rs::{module::Module, ops::indexing::IndexOp, transforms::eval, Array};
use tracing::info;

use crate::config::{GenerationConfig, Qwen3TtsConfig, TalkerConfig};
use crate::error::Result;
use crate::sampling::{build_eos_suppression_mask, build_eos_unit_mask, build_suppression_mask, compute_eos_steering_bias, sample_logits_with_mask, RepetitionPenaltyMask, SamplingKey};
use crate::talker::Talker;

/// Average ratio of generated codec frames to text tokens.
/// Empirically determined from Chinese text: ~4.0 frames per text token.
/// (12Hz codec, typical Chinese text generates ~3.3 frames per character,
/// but BPE tokens are coarser so ratio per token is higher)
const AVG_FRAMES_PER_TEXT_TOKEN: f32 = 4.0;

/// Timing information for each phase of generation.
#[derive(Debug, Clone)]
pub struct GenerationTiming {
    pub prefill_ms: f64,
    pub generation_ms: f64,
    pub generation_frames: usize,
}

/// Build the codec prefix for CustomVoice mode with specified language.
/// Returns [think, think_bos, lang_id, think_eos, spk_id]
pub fn build_codec_prefix(
    talker_config: &TalkerConfig,
    language: &str,
    speaker: &str,
) -> Result<Vec<u32>> {
    let lang_id = resolve_language_id(talker_config, language)?;

    let spk_id = talker_config
        .spk_id
        .get(speaker)
        .copied()
        .ok_or_else(|| {
            crate::error::Error::Config(format!(
                "Unknown speaker '{}'. Available: {:?}",
                speaker,
                talker_config.spk_id.keys().collect::<Vec<_>>()
            ))
        })?;

    Ok(vec![
        talker_config.codec_think_id,
        talker_config.codec_think_bos_id,
        lang_id,
        talker_config.codec_think_eos_id,
        spk_id,
    ])
}

/// Build the codec prefix for VoiceDesign mode (no speaker token).
/// Returns [think, think_bos, lang_id, think_eos]
pub fn build_codec_prefix_voice_design(
    talker_config: &TalkerConfig,
    language: &str,
) -> Result<Vec<u32>> {
    let lang_id = resolve_language_id(talker_config, language)?;
    Ok(vec![
        talker_config.codec_think_id,
        talker_config.codec_think_bos_id,
        lang_id,
        talker_config.codec_think_eos_id,
    ])
}

/// Build codec prefix for ICL voice cloning (auto-language / nothink mode).
/// Returns [nothink, think_bos, think_eos] — 3 tokens, NO language token.
/// This matches the Python's `language="auto"` path used for voice cloning.
pub fn build_codec_prefix_nothink(talker_config: &TalkerConfig) -> Vec<u32> {
    vec![
        talker_config.codec_nothink_id,    // 2155
        talker_config.codec_think_bos_id,  // 2156
        talker_config.codec_think_eos_id,  // 2157
    ]
}

fn resolve_language_id(talker_config: &TalkerConfig, language: &str) -> Result<u32> {
    talker_config
        .codec_language_id
        .get(language)
        .copied()
        .ok_or_else(|| {
            crate::error::Error::Config(format!(
                "Unknown language '{}'. Available: {:?}",
                language,
                talker_config.codec_language_id.keys().collect::<Vec<_>>()
            ))
        })
}

// ============================================================================
// Shared generation loop — used by all generate_* functions
// ============================================================================

/// Configuration for the shared autoregressive generation loop.
/// Each `generate_*` function does its own unique prefill, then delegates
/// to `run_generation_loop()` for the autoregressive decoding.
struct GenerationLoopParams {
    trailing_len: usize,
    eos_token: u32,
    pad_id: u32,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    repetition_penalty: f32,
    max_new_tokens: usize,
    /// Optional EOS steering: (target_frames, speed_factor)
    eos_steering: Option<(usize, f32)>,
}

/// Run the shared autoregressive generation loop.
///
/// Takes initial logits/hidden from prefill and generates codec frames
/// until EOS or max_new_tokens is reached.
fn run_generation_loop(
    talker: &mut Talker,
    mut logits: Array,
    mut hidden: Array,
    trailing_text_embeds: &Array,
    tts_pad_embed: &Array,
    params: &GenerationLoopParams,
    rng_key: &mut Option<SamplingKey>,
    vocab_size: usize,
) -> Result<Vec<[u32; 16]>> {
    let min_new_tokens: usize = 2;

    // Pre-build suppression masks (GPU arrays, built once)
    let suppress_mask = build_suppression_mask(vocab_size, params.eos_token);
    let eos_suppress_mask = build_eos_suppression_mask(vocab_size, params.eos_token);
    let combined_mask = suppress_mask.add(&eos_suppress_mask)?;
    let mut penalty_mask = RepetitionPenaltyMask::new(vocab_size, params.repetition_penalty)?;

    // EOS logit steering (optional)
    let eos_unit_mask = if params.eos_steering.is_some() {
        Some(build_eos_unit_mask(vocab_size, params.eos_token))
    } else {
        None
    };

    let mut all_codes: Vec<[u32; 16]> = Vec::new();

    for step in 0..params.max_new_tokens {
        let base_mask = if step < min_new_tokens {
            &combined_mask
        } else {
            &suppress_mask
        };

        // Apply EOS logit steering bias (avoid cloning: use reference when no bias needed)
        let steered_mask;
        let effective_mask: &Array = if let Some((target, speed)) = params.eos_steering {
            if step >= min_new_tokens {
                let bias = compute_eos_steering_bias(step, target, speed);
                if bias.abs() > 0.01 {
                    let unit_mask = eos_unit_mask.as_ref().unwrap();
                    let bias_mask = unit_mask.multiply(mlx_rs::array!(bias))?;
                    steered_mask = base_mask.add(&bias_mask)?;
                    &steered_mask
                } else {
                    base_mask
                }
            } else {
                base_mask
            }
        } else {
            base_mask
        };

        let token0 = sample_logits_with_mask(
            &logits,
            params.temperature,
            params.top_k,
            params.top_p,
            params.repetition_penalty,
            &[],
            rng_key.as_mut(),
            Some(effective_mask),
            Some(&penalty_mask),
        )?;

        if token0 == params.eos_token {
            info!("EOS at step {} (target={:?})", step, params.eos_steering.map(|(t, _)| t));
            break;
        }

        penalty_mask.record_token(token0)?;

        // Generate codebooks 1-15 via code predictor
        let hidden_slice = hidden.index((.., -1.., ..));
        let code0_arr = Array::from_slice(&[token0 as i32], &[1, 1]);
        let code0_embed = talker.codec_embedding.forward(&code0_arr)?;
        let sub_codes = talker
            .code_predictor
            .generate_codes(&hidden_slice, &code0_embed)?;

        let mut frame = [params.pad_id; 16];
        frame[0] = token0;
        for (g, &code) in sub_codes.iter().enumerate() {
            frame[g + 1] = code;
        }
        all_codes.push(frame);

        // Build next input: codec_embed(prev_codes) + trailing_text or tts_pad
        let text_embed_indexed;
        let text_embed: &Array = if step < params.trailing_len {
            let s = step as i32;
            text_embed_indexed = trailing_text_embeds.index((.., s..s + 1, ..));
            &text_embed_indexed
        } else {
            tts_pad_embed
        };

        let input_embed = talker.build_generation_embedding_with_text(&frame, text_embed)?;
        let result = talker.forward_step(input_embed)?;
        logits = result.0;
        hidden = result.1;
        eval([&logits])?;

        if step > 0 && step % 256 == 0 {
            unsafe { mlx_sys::mlx_clear_cache() };
        }
    }

    Ok(all_codes)
}

/// Run the full generation loop (streaming text, batched prefill).
///
/// Prefill (10 positions in one forward pass):
///   Pos 0-2: role [im_start, assistant, \n] — text only
///   Pos 3-7: tts_pad + codec [think, think_bos, lang, think_eos, spk]
///   Pos 8:   tts_bos + codec_pad
///   Pos 9:   first_text + codec_bos
///
/// Generation (streaming):
///   Each frame: codec_embed(prev_codes) + trailing_text[frame_idx]
///   Where trailing_text = [text_1, text_2, ..., text_N-1, tts_eos]
///   After trailing text exhausted: tts_pad
///
/// Returns Vec of [u32; 16] code frames.
pub fn generate(
    talker: &mut Talker,
    text_token_ids: &[u32],
    codec_prefix: &[u32],
    gen_config: &GenerationConfig,
    tts_config: &Qwen3TtsConfig,
    seed: Option<u64>,
) -> Result<(Vec<[u32; 16]>, GenerationTiming)> {
    let eos_token = tts_config.talker_config.codec_eos_token_id;
    let pad_id = tts_config.talker_config.codec_pad_id;
    let bos_id = tts_config.talker_config.codec_bos_id;

    // Initialize seeded PRNG if seed is provided
    let mut rng_key = seed
        .map(|s| SamplingKey::new(s))
        .transpose()?;

    // ====================================================================
    // Build prefill: 10 positions
    // ====================================================================
    let mut text_tokens: Vec<u32> = Vec::new();
    let mut codec_tokens: Vec<u32> = Vec::new();

    // Pos 0-2: role tokens (text only, no codec)
    text_tokens.extend_from_slice(&[
        tts_config.im_start_token_id,
        tts_config.assistant_token_id,
        198u32, // \n
    ]);
    codec_tokens.extend_from_slice(&[pad_id, pad_id, pad_id]); // will be zeroed

    // Pos 3-7: tts_pad + codec_prefix[think, think_bos, lang, think_eos, spk]
    for &codec_tok in codec_prefix {
        text_tokens.push(tts_config.tts_pad_token_id);
        codec_tokens.push(codec_tok);
    }

    // Pos 8: tts_bos + codec_pad
    text_tokens.push(tts_config.tts_bos_token_id);
    codec_tokens.push(pad_id);

    // Pos 9: first_text + codec_bos
    let first_text = if text_token_ids.is_empty() {
        tts_config.tts_pad_token_id
    } else {
        text_token_ids[0]
    };
    text_tokens.push(first_text);
    codec_tokens.push(bos_id);

    let prefill_len = text_tokens.len(); // should be 10

    // ====================================================================
    // Build trailing text: remaining text tokens + tts_eos (precomputed)
    // ====================================================================
    let mut trailing_text_ids: Vec<u32> = Vec::new();
    if text_token_ids.len() > 1 {
        trailing_text_ids.extend_from_slice(&text_token_ids[1..]);
    }
    trailing_text_ids.push(tts_config.tts_eos_token_id);
    let trailing_len = trailing_text_ids.len();

    // Precompute projected trailing text embeddings [1, trailing_len, hidden]
    let trailing_text_embeds = talker.build_projected_text_embeddings(&trailing_text_ids)?;
    // Precompute tts_pad embedding [1, 1, hidden]
    let tts_pad_embed = talker.build_text_only_embedding(tts_config.tts_pad_token_id)?;

    info!(
        "Streaming prefill: {} text tokens, {} prefill positions, {} trailing, max {} new tokens",
        text_token_ids.len(),
        prefill_len,
        trailing_len,
        gen_config.max_new_tokens
    );

    // Reset caches
    talker.reset_caches();
    talker.set_rope_speed_factor(1.0);

    let prefill_start = Instant::now();

    // ====================================================================
    // Phase 1: Batched Prefill (all 10 positions in one forward pass)
    // ====================================================================
    let input_embed = talker.build_batched_prefill_embedding(
        &text_tokens,
        &codec_tokens,
        3, // first 3 positions have no codec
    )?;
    let (prefill_logits, prefill_hidden) = talker.forward_step(input_embed)?;
    // Extract only the last position from batched output
    let logits = prefill_logits.index((.., -1.., ..)); // [1, 1, 3072]
    let hidden = prefill_hidden.index((.., -1.., ..)); // [1, 1, hidden_size]
    eval([&logits, &hidden])?;
    let prefill_time = prefill_start.elapsed();

    // ====================================================================
    // Phase 2: Autoregressive generation with streaming text
    // ====================================================================
    let gen_start = Instant::now();
    let vocab_size = tts_config.talker_config.vocab_size as usize;
    let speed = gen_config.speed_factor;
    let eos_steering = if (speed - 1.0).abs() > 0.01 {
        let t = (trailing_len as f32 * AVG_FRAMES_PER_TEXT_TOKEN / speed) as usize;
        info!("EOS steering: target_frames={}, speed={:.2}x, trailing_len={}", t, speed, trailing_len);
        Some((t, speed))
    } else {
        None
    };

    let params = GenerationLoopParams {
        trailing_len,
        eos_token,
        pad_id,
        temperature: gen_config.temperature,
        top_k: gen_config.top_k,
        top_p: gen_config.top_p,
        repetition_penalty: gen_config.repetition_penalty,
        max_new_tokens: gen_config.max_new_tokens as usize,
        eos_steering,
    };

    let all_codes = run_generation_loop(
        talker, logits, hidden, &trailing_text_embeds, &tts_pad_embed,
        &params, &mut rng_key, vocab_size,
    )?;

    let gen_time = gen_start.elapsed();

    let timing = GenerationTiming {
        prefill_ms: prefill_time.as_secs_f64() * 1000.0,
        generation_ms: gen_time.as_secs_f64() * 1000.0,
        generation_frames: all_codes.len(),
    };

    Ok((all_codes, timing))
}

/// Generate speech using VoiceDesign mode.
///
/// Prefill layout (N + 9 positions):
///   Pos 0..N-1:     instruct text (ChatML-wrapped) — text projection only
///   Pos N..N+2:     role [im_start, assistant, \n] — text projection only
///   Pos N+3..N+7:   tts_pad×4 + tts_bos overlaid with codec [think, think_bos, lang, think_eos, pad]
///   Pos N+8:        first_text + codec_bos
///
/// Generation loop is identical to CustomVoice.
pub fn generate_voice_design(
    talker: &mut Talker,
    text_token_ids: &[u32],
    instruct_token_ids: &[u32],
    codec_prefix: &[u32],
    gen_config: &GenerationConfig,
    tts_config: &Qwen3TtsConfig,
    seed: Option<u64>,
) -> Result<(Vec<[u32; 16]>, GenerationTiming)> {
    let eos_token = tts_config.talker_config.codec_eos_token_id;
    let pad_id = tts_config.talker_config.codec_pad_id;

    let mut rng_key = seed.map(|s| SamplingKey::new(s)).transpose()?;

    // Build trailing text (same as CustomVoice)
    let mut trailing_text_ids: Vec<u32> = Vec::new();
    if text_token_ids.len() > 1 {
        trailing_text_ids.extend_from_slice(&text_token_ids[1..]);
    }
    trailing_text_ids.push(tts_config.tts_eos_token_id);
    let trailing_len = trailing_text_ids.len();

    let trailing_text_embeds = talker.build_projected_text_embeddings(&trailing_text_ids)?;
    let tts_pad_embed = talker.build_text_only_embedding(tts_config.tts_pad_token_id)?;

    let prefill_positions = instruct_token_ids.len() + 9;
    info!(
        "VoiceDesign prefill: {} instruct tokens, {} text tokens, {} prefill positions, {} trailing",
        instruct_token_ids.len(),
        text_token_ids.len(),
        prefill_positions,
        trailing_len,
    );

    talker.reset_caches();
    talker.set_rope_speed_factor(1.0);

    let prefill_start = Instant::now();

    // VoiceDesign batched prefill
    let input_embed = talker.build_voice_design_prefill_embedding(
        instruct_token_ids,
        text_token_ids,
        codec_prefix,
        tts_config,
    )?;
    let (prefill_logits, prefill_hidden) = talker.forward_step(input_embed)?;
    let logits = prefill_logits.index((.., -1.., ..));
    let hidden = prefill_hidden.index((.., -1.., ..));
    eval([&logits, &hidden])?;
    let prefill_time = prefill_start.elapsed();

    // Generation loop (delegates to shared loop)
    let gen_start = Instant::now();
    let vocab_size = tts_config.talker_config.vocab_size as usize;

    let params = GenerationLoopParams {
        trailing_len,
        eos_token,
        pad_id,
        temperature: gen_config.temperature,
        top_k: gen_config.top_k,
        top_p: gen_config.top_p,
        repetition_penalty: gen_config.repetition_penalty,
        max_new_tokens: gen_config.max_new_tokens as usize,
        eos_steering: None,
    };

    let all_codes = run_generation_loop(
        talker, logits, hidden, &trailing_text_embeds, &tts_pad_embed,
        &params, &mut rng_key, vocab_size,
    )?;

    let gen_time = gen_start.elapsed();
    info!("VoiceDesign generation complete: {} frames", all_codes.len());

    let timing = GenerationTiming {
        prefill_ms: prefill_time.as_secs_f64() * 1000.0,
        generation_ms: gen_time.as_secs_f64() * 1000.0,
        generation_frames: all_codes.len(),
    };

    Ok((all_codes, timing))
}

/// Generate speech using voice cloning + instruct (emotion/style control).
pub fn generate_voice_clone_instruct(
    talker: &mut Talker,
    text_token_ids: &[u32],
    instruct_token_ids: &[u32],
    codec_prefix: &[u32],
    speaker_embedding: &Array,
    gen_config: &GenerationConfig,
    tts_config: &Qwen3TtsConfig,
    seed: Option<u64>,
) -> Result<(Vec<[u32; 16]>, GenerationTiming)> {
    let eos_token = tts_config.talker_config.codec_eos_token_id;
    let pad_id = tts_config.talker_config.codec_pad_id;

    let mut rng_key = seed.map(|s| SamplingKey::new(s)).transpose()?;

    let mut trailing_text_ids: Vec<u32> = Vec::new();
    if text_token_ids.len() > 1 {
        trailing_text_ids.extend_from_slice(&text_token_ids[1..]);
    }
    trailing_text_ids.push(tts_config.tts_eos_token_id);
    let trailing_len = trailing_text_ids.len();

    let trailing_text_embeds = talker.build_projected_text_embeddings(&trailing_text_ids)?;
    let tts_pad_embed = talker.build_text_only_embedding(tts_config.tts_pad_token_id)?;

    info!(
        "VoiceClone+Instruct prefill: {} instruct tokens, {} text tokens, {} trailing",
        instruct_token_ids.len(), text_token_ids.len(), trailing_len,
    );

    talker.reset_caches();
    talker.set_rope_speed_factor(1.0);

    let prefill_start = Instant::now();
    let input_embed = talker.build_voice_clone_instruct_prefill_embedding(
        instruct_token_ids, text_token_ids, codec_prefix, speaker_embedding, tts_config,
    )?;
    let (prefill_logits, prefill_hidden) = talker.forward_step(input_embed)?;
    let logits = prefill_logits.index((.., -1.., ..));
    let hidden = prefill_hidden.index((.., -1.., ..));
    eval([&logits, &hidden])?;
    let prefill_time = prefill_start.elapsed();

    let gen_start = Instant::now();
    let vocab_size = tts_config.talker_config.vocab_size as usize;
    let speed = gen_config.speed_factor;
    let eos_steering = if (speed - 1.0).abs() > 0.01 {
        let t = (trailing_len as f32 * AVG_FRAMES_PER_TEXT_TOKEN / speed) as usize;
        Some((t, speed))
    } else {
        None
    };

    let params = GenerationLoopParams {
        trailing_len, eos_token, pad_id,
        temperature: gen_config.temperature,
        top_k: gen_config.top_k,
        top_p: gen_config.top_p,
        repetition_penalty: gen_config.repetition_penalty,
        max_new_tokens: gen_config.max_new_tokens as usize,
        eos_steering,
    };

    let all_codes = run_generation_loop(
        talker, logits, hidden, &trailing_text_embeds, &tts_pad_embed,
        &params, &mut rng_key, vocab_size,
    )?;

    let gen_time = gen_start.elapsed();
    let n_frames = all_codes.len();
    info!("VoiceClone+Instruct generation complete: {} frames", n_frames);

    Ok((all_codes, GenerationTiming {
        prefill_ms: prefill_time.as_secs_f64() * 1000.0,
        generation_ms: gen_time.as_secs_f64() * 1000.0,
        generation_frames: n_frames,
    }))
}

/// Generate speech using voice cloning (x_vector_only mode).
///
/// Uses a continuous speaker embedding from ECAPA-TDNN instead of discrete speaker token.
/// Prefill layout is same as CustomVoice but position 7 has continuous embedding.
/// Generation loop is identical.
pub fn generate_voice_clone(
    talker: &mut Talker,
    text_token_ids: &[u32],
    codec_prefix: &[u32],       // [think, think_bos, lang, think_eos] — 4 tokens
    speaker_embedding: &Array,   // [1, enc_dim] from ECAPA-TDNN
    gen_config: &GenerationConfig,
    tts_config: &Qwen3TtsConfig,
    seed: Option<u64>,
) -> Result<(Vec<[u32; 16]>, GenerationTiming)> {
    let eos_token = tts_config.talker_config.codec_eos_token_id;
    let pad_id = tts_config.talker_config.codec_pad_id;

    let mut rng_key = seed.map(|s| SamplingKey::new(s)).transpose()?;

    // Build trailing text (same as other modes)
    let mut trailing_text_ids: Vec<u32> = Vec::new();
    if text_token_ids.len() > 1 {
        trailing_text_ids.extend_from_slice(&text_token_ids[1..]);
    }
    trailing_text_ids.push(tts_config.tts_eos_token_id);
    let trailing_len = trailing_text_ids.len();

    let trailing_text_embeds = talker.build_projected_text_embeddings(&trailing_text_ids)?;
    let tts_pad_embed = talker.build_text_only_embedding(tts_config.tts_pad_token_id)?;

    info!(
        "VoiceClone prefill: {} text tokens, 10 prefill positions, {} trailing",
        text_token_ids.len(),
        trailing_len,
    );

    talker.reset_caches();
    talker.set_rope_speed_factor(1.0);

    let prefill_start = Instant::now();

    // Voice clone batched prefill (uses continuous speaker embedding at position 7)
    let input_embed = talker.build_voice_clone_prefill_embedding(
        text_token_ids,
        codec_prefix,
        speaker_embedding,
        tts_config,
    )?;

    let (prefill_logits, prefill_hidden) = talker.forward_step(input_embed)?;
    let logits = prefill_logits.index((.., -1.., ..));
    let hidden = prefill_hidden.index((.., -1.., ..));
    eval([&logits, &hidden])?;
    let prefill_time = prefill_start.elapsed();

    // Generation loop with EOS logit steering for speed control
    let gen_start = Instant::now();
    let vocab_size = tts_config.talker_config.vocab_size as usize;
    let speed = gen_config.speed_factor;
    let eos_steering = if (speed - 1.0).abs() > 0.01 {
        let t = (trailing_len as f32 * AVG_FRAMES_PER_TEXT_TOKEN / speed) as usize;
        info!("EOS steering: target_frames={}, speed={:.2}x, trailing_len={}", t, speed, trailing_len);
        Some((t, speed))
    } else {
        None
    };

    let params = GenerationLoopParams {
        trailing_len,
        eos_token,
        pad_id,
        temperature: gen_config.temperature,
        top_k: gen_config.top_k,
        top_p: gen_config.top_p,
        repetition_penalty: gen_config.repetition_penalty,
        max_new_tokens: gen_config.max_new_tokens as usize,
        eos_steering,
    };

    let all_codes = run_generation_loop(
        talker, logits, hidden, &trailing_text_embeds, &tts_pad_embed,
        &params, &mut rng_key, vocab_size,
    )?;

    let gen_time = gen_start.elapsed();
    info!("VoiceClone generation complete: {} frames (trailing_len={})", all_codes.len(), trailing_len);

    let timing = GenerationTiming {
        prefill_ms: prefill_time.as_secs_f64() * 1000.0,
        generation_ms: gen_time.as_secs_f64() * 1000.0,
        generation_frames: all_codes.len(),
    };

    Ok((all_codes, timing))
}

/// Generate speech using ICL voice cloning.
///
/// Uses both speaker encoder embedding AND reference audio codes for full quality.
/// Prefill: 9 positions (no first_text+bos), then ICL extension block, then generation.
/// The reference codes are prepended to output codes for decoding, then proportionally trimmed.
///
/// Returns (all_codes, ref_codes, ref_text_token_count, gen_timing) where:
/// - all_codes: generated codec frames only (without reference prefix)
/// - ref_codes: reference codec frames (to prepend for decoding)
/// - ref_text_token_count: number of reference text tokens (for proportional trimming)
/// - gen_timing: timing breakdown
pub fn generate_voice_clone_icl(
    talker: &mut Talker,
    text_token_ids: &[u32],
    ref_text_ids: &[u32],         // tokenized reference text
    ref_codes: &[[u32; 16]],      // reference audio codec frames from Mimi encoder
    codec_prefix: &[u32],          // [think, think_bos, lang, think_eos]
    speaker_embedding: &Array,     // [1, enc_dim] from ECAPA-TDNN
    gen_config: &GenerationConfig,
    tts_config: &Qwen3TtsConfig,
    seed: Option<u64>,
) -> Result<(Vec<[u32; 16]>, Vec<[u32; 16]>, usize, GenerationTiming)> {
    let eos_token = tts_config.talker_config.codec_eos_token_id;
    let pad_id = tts_config.talker_config.codec_pad_id;

    let mut rng_key = seed.map(|s| SamplingKey::new(s)).transpose()?;

    let repetition_penalty = gen_config.repetition_penalty;
    let max_new_tokens = gen_config.max_new_tokens as usize;

    info!(
        "VoiceClone ICL: {} ref frames, {} ref text tokens, {} target text tokens, max {} new tokens, rep_penalty={:.2}",
        ref_codes.len(),
        ref_text_ids.len(),
        text_token_ids.len(),
        max_new_tokens,
        repetition_penalty,
    );

    talker.reset_caches();
    talker.set_rope_speed_factor(1.0);

    let prefill_start = Instant::now();

    // Phase 1: prefill (no first_text+bos in ICL mode)
    let prefill_embed = talker.build_icl_prefill_embedding(
        codec_prefix,
        speaker_embedding,
        tts_config,
    )?;
    let prefill_len = prefill_embed.dim(1);
    let (_, _) = talker.forward_step(prefill_embed)?;
    eval(std::iter::empty::<&Array>())?;

    // Phase 2: ICL extension — sum reference codec embeddings and build ICL prompt
    let ref_codec_embed = talker.sum_ref_codec_embeddings(ref_codes)?; // [1, T_ref, hidden]
    // Non-streaming mode: text block then codec block (sequential).
    // NOTE: Streaming mode (non_streaming_mode=false) matches Python reference (text+codec overlaid)
    // but produces distorted/non-voice audio on Apple Silicon. Non-streaming produces coherent
    // audio but with premature EOS. ICL voice cloning is fundamentally unreliable on Apple Silicon;
    // use x_vector_only mode instead for reliable voice cloning.
    let (icl_embed, trailing_text_embed, trailing_len) = talker.build_icl_prompt(
        ref_text_ids,
        text_token_ids,
        &ref_codec_embed,
        tts_config,
        true,
    )?;

    // Feed ICL prompt through transformer (extends KV cache)
    let icl_len = icl_embed.dim(1);
    let (icl_logits, icl_hidden) = talker.forward_step(icl_embed)?;

    // Extract last position logits/hidden
    let logits = icl_logits.index((.., -1.., ..));
    let hidden = icl_hidden.index((.., -1.., ..));
    eval([&logits, &hidden])?;
    let prefill_time = prefill_start.elapsed();

    info!(
        "ICL prefill complete: {} + {} ICL positions, trailing_len={}",
        prefill_len,
        icl_len,
        trailing_len,
    );

    // Precompute tts_pad embedding for after trailing text exhausted
    let tts_pad_embed = talker.build_text_only_embedding(tts_config.tts_pad_token_id)?;

    // Phase 3: Autoregressive generation (delegates to shared loop)
    let gen_start = Instant::now();
    let vocab_size = tts_config.talker_config.vocab_size as usize;

    let params = GenerationLoopParams {
        trailing_len,
        eos_token,
        pad_id,
        temperature: gen_config.temperature,
        top_k: gen_config.top_k,
        top_p: gen_config.top_p,
        repetition_penalty,
        max_new_tokens,
        eos_steering: None,
    };

    let all_codes = run_generation_loop(
        talker, logits, hidden, &trailing_text_embed, &tts_pad_embed,
        &params, &mut rng_key, vocab_size,
    )?;

    let gen_time = gen_start.elapsed();
    info!("VoiceClone ICL generation complete: {} frames", all_codes.len());

    let timing = GenerationTiming {
        prefill_ms: prefill_time.as_secs_f64() * 1000.0,
        generation_ms: gen_time.as_secs_f64() * 1000.0,
        generation_frames: all_codes.len(),
    };

    Ok((all_codes, ref_codes.to_vec(), ref_text_ids.len(), timing))
}

/// Holds all state needed for step-by-step autoregressive generation.
/// Used by `StreamingSession` to generate audio in chunks.
pub struct GenerationState {
    logits: Array,
    hidden: Array,
    penalty_mask: RepetitionPenaltyMask,
    step: usize,
    max_steps: usize,
    min_new_tokens: usize,
    trailing_len: usize,
    trailing_text_embeds: Array,
    tts_pad_embed: Array,
    suppress_mask: Array,
    combined_mask: Array,
    eos_token: u32,
    pad_id: u32,
    rng_key: Option<SamplingKey>,
    gen_config: GenerationConfig,
    finished: bool,
}

impl GenerationState {
    /// Perform prefill and create the generation state, ready for step-by-step generation.
    pub fn new(
        talker: &mut Talker,
        text_token_ids: &[u32],
        codec_prefix: &[u32],
        gen_config: &GenerationConfig,
        tts_config: &Qwen3TtsConfig,
        seed: Option<u64>,
    ) -> Result<Self> {
        let eos_token = tts_config.talker_config.codec_eos_token_id;
        let pad_id = tts_config.talker_config.codec_pad_id;
        let bos_id = tts_config.talker_config.codec_bos_id;

        let rng_key = seed.map(|s| SamplingKey::new(s)).transpose()?;

        // Build prefill tokens
        let mut text_tokens: Vec<u32> = Vec::new();
        let mut codec_tokens: Vec<u32> = Vec::new();

        text_tokens.extend_from_slice(&[
            tts_config.im_start_token_id,
            tts_config.assistant_token_id,
            198u32,
        ]);
        codec_tokens.extend_from_slice(&[pad_id, pad_id, pad_id]);

        for &codec_tok in codec_prefix {
            text_tokens.push(tts_config.tts_pad_token_id);
            codec_tokens.push(codec_tok);
        }

        text_tokens.push(tts_config.tts_bos_token_id);
        codec_tokens.push(pad_id);

        let first_text = if text_token_ids.is_empty() {
            tts_config.tts_pad_token_id
        } else {
            text_token_ids[0]
        };
        text_tokens.push(first_text);
        codec_tokens.push(bos_id);

        // Trailing text
        let mut trailing_text_ids: Vec<u32> = Vec::new();
        if text_token_ids.len() > 1 {
            trailing_text_ids.extend_from_slice(&text_token_ids[1..]);
        }
        trailing_text_ids.push(tts_config.tts_eos_token_id);
        let trailing_len = trailing_text_ids.len();

        let trailing_text_embeds = talker.build_projected_text_embeddings(&trailing_text_ids)?;
        let tts_pad_embed = talker.build_text_only_embedding(tts_config.tts_pad_token_id)?;

        talker.reset_caches();
        talker.set_rope_speed_factor(1.0);

        // Prefill
        let input_embed = talker.build_batched_prefill_embedding(&text_tokens, &codec_tokens, 3)?;
        let (prefill_logits, prefill_hidden) = talker.forward_step(input_embed)?;
        let logits = prefill_logits.index((.., -1.., ..));
        let hidden = prefill_hidden.index((.., -1.., ..));
        eval([&logits, &hidden])?;

        // Pre-build masks
        let vocab_size = tts_config.talker_config.vocab_size as usize;
        let suppress_mask = build_suppression_mask(vocab_size, eos_token);
        let eos_suppress_mask = build_eos_suppression_mask(vocab_size, eos_token);
        let combined_mask = suppress_mask.add(&eos_suppress_mask)?;
        let penalty_mask = RepetitionPenaltyMask::new(vocab_size, gen_config.repetition_penalty)?;

        Ok(Self {
            logits,
            hidden,
            penalty_mask,
            step: 0,
            max_steps: gen_config.max_new_tokens as usize,
            min_new_tokens: 2,
            trailing_len,
            trailing_text_embeds,
            tts_pad_embed,
            suppress_mask,
            combined_mask,
            eos_token,
            pad_id,
            rng_key,
            gen_config: gen_config.clone(),
            finished: false,
        })
    }

    /// Generate the next chunk of frames. Returns None if generation is done.
    pub fn next_chunk(
        &mut self,
        talker: &mut Talker,
        chunk_frames: usize,
    ) -> Result<Option<Vec<[u32; 16]>>> {
        if self.finished {
            return Ok(None);
        }

        let mut frames = Vec::with_capacity(chunk_frames);

        for _ in 0..chunk_frames {
            if self.step >= self.max_steps {
                self.finished = true;
                break;
            }

            let mask = if self.step < self.min_new_tokens {
                &self.combined_mask
            } else {
                &self.suppress_mask
            };

            let token0 = sample_logits_with_mask(
                &self.logits,
                self.gen_config.temperature,
                self.gen_config.top_k,
                self.gen_config.top_p,
                self.gen_config.repetition_penalty,
                &[],
                self.rng_key.as_mut(),
                Some(mask),
                Some(&self.penalty_mask),
            )?;

            if token0 == self.eos_token {
                info!("EOS at step {}", self.step);
                self.finished = true;
                break;
            }

            self.penalty_mask.record_token(token0)?;

            // Code predictor: codebooks 1-15
            let hidden_slice = self.hidden.index((.., -1.., ..));
            let code0_arr = Array::from_slice(&[token0 as i32], &[1, 1]);
            let code0_embed = talker.codec_embedding.forward(&code0_arr)?;
            let sub_codes = talker
                .code_predictor
                .generate_codes(&hidden_slice, &code0_embed)?;

            let mut frame = [self.pad_id; 16];
            frame[0] = token0;
            for (g, &code) in sub_codes.iter().enumerate() {
                frame[g + 1] = code;
            }
            frames.push(frame);

            // Build next input
            let text_embed_indexed;
            let text_embed: &Array = if self.step < self.trailing_len {
                let s = self.step as i32;
                text_embed_indexed = self.trailing_text_embeds.index((.., s..s + 1, ..));
                &text_embed_indexed
            } else {
                &self.tts_pad_embed
            };

            let input_embed = talker.build_generation_embedding_with_text(&frame, text_embed)?;
            let result = talker.forward_step(input_embed)?;
            self.logits = result.0;
            self.hidden = result.1;
            eval([&self.logits])?;

            self.step += 1;

            if self.step > 0 && self.step % 256 == 0 {
                unsafe { mlx_sys::mlx_clear_cache() };
            }
        }

        if frames.is_empty() {
            Ok(None)
        } else {
            Ok(Some(frames))
        }
    }

    /// Returns true if generation has finished (EOS or max tokens).
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Total frames generated so far.
    pub fn total_frames(&self) -> usize {
        self.step
    }
}
