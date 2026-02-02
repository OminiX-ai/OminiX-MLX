//! Audio processing utilities for TTS
//!
//! Re-exports audio functions from mlx-rs-core and provides
//! MLX-native audio processing for training.

mod mel;
mod stft_gpu;

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
    SpectrogramConfig,
    mel_spectrogram_mlx,
    spectrogram_mlx,
    stft_mlx,
    create_mel_filterbank,
    spec_to_mel,
    slice_mel_segments,
};

// GPU-accelerated STFT using MLX rfft (O(N log N) instead of O(N²))
pub use stft_gpu::{
    stft_rfft,
    stft_rfft_for_reference,
    load_reference_mel_gpu,
};
