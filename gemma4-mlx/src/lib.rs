//! Gemma 4 12B (text) inference on Apple Silicon with MLX.
//! See `gemma4-mlx/README.md` for usage, validation, and implementation notes.
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
pub mod tokenizer;
pub mod weights;

pub use error::{Error, Result};
pub use generate::generate_greedy;
pub use model::{load_model, Gemma4TextModel};
pub use tokenizer::{encode_chat, eos_ids, load_tokenizer};
