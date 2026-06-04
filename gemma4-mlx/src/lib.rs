//! Gemma 4 12B (text) inference on Apple Silicon with MLX.
//! See docs/superpowers/specs/2026-06-04-gemma4-12b-text-mlx-design.md
pub mod attention;
pub mod block;
pub mod config;
pub mod error;
pub mod generate;
pub mod mask;
pub mod mlp;
pub mod model;
pub mod norm;
pub mod rope;
pub mod weights;

pub use error::{Error, Result};
pub use generate::generate_greedy;
pub use model::{load_model, Gemma4TextModel};
