//! KV Cache implementations for LLM inference

use mlx_rs::{error::Exception, ops::concatenate_axis, ops::zeros_dtype, Array};
use mlx_rs::ops::indexing::{IndexMutOp, IndexOp, Ellipsis};

/// Trait for key-value caches used in attention
pub trait KeyValueCache {
    /// Returns the current offset (number of tokens in cache)
    fn offset(&self) -> i32;

    /// Returns the maximum cache size (for sliding window), if any
    fn max_size(&self) -> Option<i32>;

    /// Update cache with new keys/values and return full cache contents
    fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<(Array, Array), Exception>;
}

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache,
{
    fn offset(&self) -> i32 {
        T::offset(self)
    }

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<(Array, Array), Exception> {
        T::update_and_fetch(self, keys, values)
    }
}

/// Simple concatenation-based KV cache
#[derive(Debug, Clone, Default)]
pub struct ConcatKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
}

impl ConcatKeyValueCache {
    pub fn new() -> Self {
        Self::default()
    }
}

impl KeyValueCache for ConcatKeyValueCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<(Array, Array), Exception> {
        match (self.keys.take(), self.values.take()) {
            (Some(k), Some(v)) => {
                self.keys = Some(concatenate_axis(&[k, keys], -2)?);
                self.values = Some(concatenate_axis(&[v, values], -2)?);
            }
            _ => {
                self.keys = Some(keys);
                self.values = Some(values);
            }
        }
        let shape = self.keys.as_ref().expect("Keys cannot be None").shape();
        self.offset = shape[shape.len() - 2];

        Ok((
            self.keys.clone().expect("Keys cannot be None"),
            self.values.clone().expect("Values cannot be None"),
        ))
    }
}

/// Step-based KV Cache with pre-allocation (matches Python mlx-lm KVCache)
///
/// This cache pre-allocates buffers in steps of 256 tokens and uses in-place
/// slice updates, avoiding expensive concatenation on every token.
#[derive(Debug, Clone)]
pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
    step: i32,
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KVCache {
    pub fn new() -> Self {
        Self::with_step(256)
    }

    pub fn with_step(step: i32) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            step,
        }
    }
}

impl KeyValueCache for KVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<(Array, Array), Exception> {
        let prev = self.offset;
        let keys_shape = keys.shape();
        let values_shape = values.shape();
        let num_new = keys_shape[2];

        // Check if we need to grow the buffer
        let needs_grow = match &self.keys {
            None => true,
            Some(k) => (prev + num_new) > k.shape()[2],
        };

        if needs_grow {
            let b = keys_shape[0];
            let n_kv_heads = keys_shape[1];
            let k_head_dim = keys_shape[3];
            let v_head_dim = values_shape[3];

            let n_steps = (self.step + num_new - 1) / self.step;
            let new_size = n_steps * self.step;

            let k_shape = &[b, n_kv_heads, new_size, k_head_dim];
            let v_shape = &[b, n_kv_heads, new_size, v_head_dim];

            let k_dtype = keys.dtype();
            let v_dtype = values.dtype();
            let new_k = zeros_dtype(k_shape, k_dtype)?;
            let new_v = zeros_dtype(v_shape, v_dtype)?;

            match (self.keys.take(), self.values.take()) {
                (Some(old_k), Some(old_v)) => {
                    let (old_k, old_v) = if prev % self.step != 0 {
                        (
                            old_k.index((Ellipsis, ..prev, ..)),
                            old_v.index((Ellipsis, ..prev, ..)),
                        )
                    } else {
                        (old_k, old_v)
                    };
                    self.keys = Some(concatenate_axis(&[old_k, new_k], 2)?);
                    self.values = Some(concatenate_axis(&[old_v, new_v], 2)?);
                }
                _ => {
                    self.keys = Some(new_k);
                    self.values = Some(new_v);
                }
            }
        }

        self.offset += num_new;

        let k = self.keys.as_mut().unwrap();
        let v = self.values.as_mut().unwrap();
        k.index_mut((Ellipsis, prev..self.offset, ..), &keys);
        v.index_mut((Ellipsis, prev..self.offset, ..), &values);

        Ok((
            k.index((Ellipsis, ..self.offset, ..)),
            v.index((Ellipsis, ..self.offset, ..)),
        ))
    }
}
