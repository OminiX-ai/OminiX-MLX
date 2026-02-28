//! Qwen3.5-27B hybrid inference on Apple Silicon with MLX.
//!
//! This crate implements the Qwen3.5-27B model which uses a hybrid architecture:
//! - **48 DeltaNet layers** (linear attention with fixed-size recurrent state)
//! - **16 Gated Attention layers** (full attention with partial RoPE and output gate)
//!
//! Supports 4-bit and 8-bit quantized models from the `mlx-community` HuggingFace hub.

pub mod attention;
pub mod cache;
pub mod config;
pub mod deltanet;
pub mod mlp;
pub mod model;

pub use cache::HybridCache;
pub use config::ModelArgs;
pub use model::{load_model, Model};
pub use mlx_rs_core::{error::Error, load_tokenizer};

use mlx_rs::{
    argmax_axis, array, categorical,
    error::Exception,
    ops::indexing::{IndexOp, NewAxis},
    Array,
};

// ============================================================================
// Sampling
// ============================================================================

pub fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    match temp {
        t if t == 0.0 => argmax_axis!(logits, -1).map_err(Into::into),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical!(logits).map_err(Into::into)
        }
    }
}

// ============================================================================
// Generation Iterator
// ============================================================================

pub struct Generate<'a> {
    model: &'a mut Model,
    cache: Vec<HybridCache>,
    temp: f32,
    state: GenerateState<'a>,
    prefetched: Option<Array>,
    token_count: usize,
}

enum GenerateState<'a> {
    Prefill { prompt: &'a Array },
    Decode,
}

impl<'a> Generate<'a> {
    pub fn new(model: &'a mut Model, temp: f32, prompt: &'a Array) -> Self {
        Self {
            model,
            cache: Vec::new(),
            temp,
            state: GenerateState::Prefill { prompt },
            prefetched: None,
            token_count: 0,
        }
    }

    fn compute_next(&mut self, y: &Array) -> Result<Array, Exception> {
        let inputs = y.index((.., NewAxis)); // [B, 1]
        let logits = self.model.forward(&inputs, &mut self.cache)?;
        sample(&logits.index((.., -1, ..)), self.temp)
    }
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

impl Iterator for Generate<'_> {
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        use mlx_rs::transforms::{async_eval, eval};

        match &self.state {
            GenerateState::Prefill { prompt } => {
                let prompt = *prompt;
                let logits = tri!(self.model.forward(prompt, &mut self.cache));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));

                let _ = async_eval([&y]);
                let next_y = tri!(self.compute_next(&y));
                let _ = async_eval([&next_y]);
                let _ = eval([&y]);

                self.prefetched = Some(next_y);
                self.state = GenerateState::Decode;
                self.token_count = 1;

                Some(Ok(y))
            }
            GenerateState::Decode => {
                let current = self.prefetched.take()?;
                let next_y = tri!(self.compute_next(&current));
                let _ = mlx_rs::transforms::async_eval([&next_y]);

                self.prefetched = Some(next_y);
                self.token_count += 1;

                if self.token_count % 256 == 0 {
                    // SAFETY: mlx_clear_cache releases GPU memory for completed graphs.
                    // Safe to call between generation steps when no evaluation is in progress.
                    unsafe {
                        mlx_sys::mlx_clear_cache();
                    }
                }

                Some(Ok(current))
            }
        }
    }
}
