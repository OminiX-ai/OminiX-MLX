//! GPT-SoVITS Model Components
//!
//! This module contains the neural network models used in the TTS pipeline:
//!
//! - **BERT**: Chinese BERT for text feature extraction
//! - **HuBERT**: Self-supervised speech representation for voice cloning
//! - **T2S**: Text-to-Semantic transformer for generating audio tokens
//! - **VITS**: Variational Inference TTS vocoder for audio synthesis
//! - **Discriminator**: Multi-Period Discriminator for GAN training

pub mod bert;
pub mod discriminator;
pub mod hubert;
pub mod t2s;
pub mod vits;
pub mod vits_onnx;

pub use discriminator::{MultiPeriodDiscriminator, MPDConfig};
pub use hubert::{HuBertEncoder, HuBertConfig};
pub use t2s::{T2SModel, T2SConfig};
pub use vits::{SynthesizerTrn, VITSConfig};
pub use vits_onnx::VitsOnnx;
