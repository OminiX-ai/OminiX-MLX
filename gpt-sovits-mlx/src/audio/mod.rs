//! Audio processing utilities for TTS
//!
//! Re-exports audio functions from mlx-rs-core and provides
//! MLX-native audio processing for training.

mod mel;

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

// MLX-native mel computation for training
pub use mel::{
    MelConfig,
    mel_spectrogram_mlx,
    stft_mlx,
    create_mel_filterbank,
};
