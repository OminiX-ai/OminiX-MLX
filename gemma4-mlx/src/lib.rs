//! Gemma 4 12B (text) inference on Apple Silicon with MLX.
//! See docs/superpowers/specs/2026-06-04-gemma4-12b-text-mlx-design.md
pub mod config;
pub mod error;
pub mod mask;
pub mod mlp;
pub mod norm;
pub mod rope;
pub mod weights;

pub use error::{Error, Result};
