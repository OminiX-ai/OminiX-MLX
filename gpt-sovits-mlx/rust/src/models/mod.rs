//! GPT-SoVITS Model Components
//!
//! This module contains the neural network models used in the TTS pipeline:
//!
//! - **BERT**: Chinese BERT for text feature extraction
//! - **HuBERT**: Self-supervised speech representation for voice cloning
//! - **T2S**: Text-to-Semantic transformer for generating audio tokens
//! - **VITS**: Variational Inference TTS vocoder for audio synthesis

pub mod bert;
pub mod hubert;
pub mod t2s;
pub mod vits;

pub use hubert::{HuBertEncoder, HuBertConfig};
pub use t2s::{T2SModel, T2SConfig};
pub use vits::{SynthesizerTrn, VITSConfig};
