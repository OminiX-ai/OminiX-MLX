//! Audio processing utilities for TTS
//!
//! Re-exports audio functions from mlx-rs-core.
//! All TTS-specific audio processing (mel spectrogram, HuBERT preprocessing)
//! is available from mlx-rs-core::audio.

// Re-export everything from mlx-rs-core::audio
pub use mlx_rs_core::audio::{
    // Core audio I/O
    load_wav,
    save_wav,
    resample,

    // Configuration
    AudioConfig,

    // TTS-specific functions
    compute_mel_spectrogram,
    load_audio_for_hubert,
    load_reference_mel,
};
