//! Neural Network Layers
//!
//! Custom neural network layers for GPT-SoVITS, complementing mlx_rs::nn.

mod weight_norm;

pub use weight_norm::{WeightNormConv1d, WeightNormConvTranspose1d};
