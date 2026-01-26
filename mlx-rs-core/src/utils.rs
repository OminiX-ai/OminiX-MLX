//! Utility functions for attention and RoPE

use std::collections::HashMap;

/// Helper macro for early returns from Option<Result<T, E>> iterators
#[macro_export]
macro_rules! try_unwrap {
    ($expr:expr) => {
        match $expr {
            core::result::Result::Ok(val) => val,
            core::result::Result::Err(e) => return Some(Err(e.into())),
        }
    };
}
use mlx_rs::{
    arange, builder::Builder, error::Exception,
    fast::ScaledDotProductAttentionMask,
    nn,
    ops::indexing::{IndexOp, NewAxis},
    Array,
};
use serde::Deserialize;

use crate::cache::KeyValueCache;

// ============================================================================
// RoPE (Rotary Position Embedding)
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum FloatOrStr<'a> {
    Float(f32),
    Str(&'a str),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum FloatOrString {
    Float(f32),
    String(String),
}

impl FloatOrString {
    pub fn borrowed(&self) -> FloatOrStr<'_> {
        match self {
            FloatOrString::Float(f) => FloatOrStr::Float(*f),
            FloatOrString::String(s) => FloatOrStr::Str(s),
        }
    }
}

pub fn initialize_rope(
    dims: i32,
    base: f32,
    traditional: bool,
    scaling_config: &Option<HashMap<String, FloatOrString>>,
    _max_position_embeddings: i32,
) -> Result<nn::Rope, Exception> {
    let rope_type = scaling_config
        .as_ref()
        .and_then(|config| {
            config
                .get("type")
                .or_else(|| config.get("rope_type"))
                .map(FloatOrString::borrowed)
        })
        .unwrap_or(FloatOrStr::Str("default"));

    if rope_type == FloatOrStr::Str("default") || rope_type == FloatOrStr::Str("linear") {
        let scale = if rope_type == FloatOrStr::Str("linear") {
            let den = match scaling_config
                .as_ref()
                .and_then(|config| config.get("factor"))
                .map(FloatOrString::borrowed)
                .ok_or_else(|| Exception::custom(r#"key "factor" is not found in scaling config"#))?
            {
                FloatOrStr::Float(f) => f,
                FloatOrStr::Str(s) => s
                    .parse::<f32>()
                    .map_err(|_| Exception::custom(r#"key "factor" is not a valid float"#))?,
            };
            1.0 / den
        } else {
            1.0
        };

        let rope = nn::RopeBuilder::new(dims)
            .traditional(traditional)
            .base(base)
            .scale(scale)
            .build()
            .expect("Infallible");
        return Ok(rope);
    }

    Err(Exception::custom(format!("Unsupported RoPE type {rope_type:?}")))
}

// ============================================================================
// Attention Masks
// ============================================================================

/// Attention mask for scaled_dot_product_attention
#[derive(Debug, Clone)]
pub enum SdpaMask<'a> {
    /// Hardware-optimized causal mask (for prefill)
    Causal,
    /// Explicit array mask
    Array(&'a Array),
}

impl<'a> From<&'a Array> for SdpaMask<'a> {
    fn from(mask: &'a Array) -> Self {
        SdpaMask::Array(mask)
    }
}

#[derive(Debug, Clone)]
pub enum AttentionMask {
    Array(Array),
    Causal,
}

impl<'a> From<&'a AttentionMask> for ScaledDotProductAttentionMask<'a> {
    fn from(mask: &'a AttentionMask) -> Self {
        match mask {
            AttentionMask::Array(array) => ScaledDotProductAttentionMask::Array(array),
            AttentionMask::Causal => ScaledDotProductAttentionMask::Causal,
        }
    }
}

#[allow(non_snake_case)]
pub fn create_causal_mask(
    N: i32,
    offset: Option<i32>,
    window_size: Option<i32>,
    _lengths: Option<Array>,
) -> Result<Array, Exception> {
    let offset = offset.unwrap_or(0);

    let rinds = arange!(stop = offset + N)?;
    let linds = arange!(start = offset, stop = offset + N)?;
    let linds = linds.index((.., NewAxis));
    let rinds = rinds.index(NewAxis);

    let mut mask = linds.ge(&rinds)?;
    if let Some(window_size) = window_size {
        mask = mask.logical_and(&linds.le(&(rinds + window_size))?)?;
    }

    Ok(mask)
}

#[allow(non_snake_case)]
pub fn create_attention_mask<C>(
    h: &Array,
    cache: &[Option<C>],
    return_array: Option<bool>,
) -> Result<Option<AttentionMask>, Exception>
where
    C: KeyValueCache,
{
    let mut return_array = return_array.unwrap_or(false);
    let T = h.shape()[1];
    if T > 1 {
        let mut offset = 0;
        let mut window_size = None;
        if let Some(c) = cache.first().and_then(|c| c.as_ref()) {
            offset = c.offset();
            if let Some(window_size_) = c.max_size() {
                window_size = Some(window_size_);
                offset = offset.min(window_size_);
                return_array = return_array || (offset + T) > window_size_;
            }
        }

        if return_array {
            create_causal_mask(T, Some(offset), window_size, None)
                .map(AttentionMask::Array)
                .map(Some)
        } else {
            Ok(Some(AttentionMask::Causal))
        }
    } else {
        Ok(None)
    }
}

/// Scaled dot-product attention
pub fn scaled_dot_product_attention<'a, C>(
    queries: Array,
    keys: Array,
    values: Array,
    _cache: Option<C>,
    scale: f32,
    mask: Option<SdpaMask<'a>>,
) -> Result<Array, Exception>
where
    C: KeyValueCache,
{
    let sdpa_mask = match mask {
        Some(SdpaMask::Causal) => Some(ScaledDotProductAttentionMask::Causal),
        Some(SdpaMask::Array(m)) => Some(ScaledDotProductAttentionMask::Array(m)),
        None => None,
    };

    mlx_rs::fast::scaled_dot_product_attention(queries, keys, values, scale, sdpa_mask)
}
