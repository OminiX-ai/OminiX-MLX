//! Codec prefix builders — pure config lookups, no compute.
//!
//! These build the token ID sequences that precede generation,
//! controlling language, speaker, and generation mode.

use crate::config::TalkerConfig;
use crate::error::{Error, Result};

/// Build codec prefix for CustomVoice mode (discrete speaker token).
/// Returns: [think, think_bos, lang_id, think_eos, spk_id]
pub fn build_codec_prefix(
    config: &TalkerConfig,
    language: &str,
    speaker: &str,
) -> Result<Vec<u32>> {
    let lang_id = resolve_language_id(config, language)?;
    let spk_id = config
        .spk_id
        .get(speaker)
        .copied()
        .ok_or_else(|| {
            Error::Config(format!(
                "Unknown speaker '{}'. Available: {:?}",
                speaker,
                config.spk_id.keys().collect::<Vec<_>>()
            ))
        })?;

    Ok(vec![
        config.codec_think_id,
        config.codec_think_bos_id,
        lang_id,
        config.codec_think_eos_id,
        spk_id,
    ])
}

/// Build codec prefix for VoiceDesign / x-vector clone mode (no speaker token).
/// Returns: [think, think_bos, lang_id, think_eos]
pub fn build_codec_prefix_voice_design(
    config: &TalkerConfig,
    language: &str,
) -> Result<Vec<u32>> {
    let lang_id = resolve_language_id(config, language)?;
    Ok(vec![
        config.codec_think_id,
        config.codec_think_bos_id,
        lang_id,
        config.codec_think_eos_id,
    ])
}

/// Build codec prefix for ICL voice cloning (auto-language / nothink mode).
/// Returns: [nothink, think_bos, think_eos]
pub fn build_codec_prefix_nothink(config: &TalkerConfig) -> Vec<u32> {
    vec![
        config.codec_nothink_id,
        config.codec_think_bos_id,
        config.codec_think_eos_id,
    ]
}

fn resolve_language_id(config: &TalkerConfig, language: &str) -> Result<u32> {
    config
        .codec_language_id
        .get(language)
        .copied()
        .ok_or_else(|| {
            Error::Config(format!(
                "Unknown language '{}'. Available: {:?}",
                language,
                config.codec_language_id.keys().collect::<Vec<_>>()
            ))
        })
}
