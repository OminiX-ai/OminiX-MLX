use mlx_rs::Array;
use mlx_rs_core::cache::{KVCache, KeyValueCache};

/// Recurrent state for DeltaNet layers.
///
/// Stores a fixed-size state matrix and a conv1d sliding window buffer.
/// Unlike KV cache which grows with sequence length, this has constant size.
#[derive(Debug, Clone)]
pub struct RecurrentState {
    /// Delta rule state: [B, num_v_heads, k_dim, v_dim]
    pub state: Option<Array>,
    /// Conv1d sliding window: [B, conv_dim, kernel_size - 1]
    pub conv_state: Option<Array>,
    /// Number of tokens processed (for position tracking)
    pub step: i32,
}

impl RecurrentState {
    pub fn new() -> Self {
        Self {
            state: None,
            conv_state: None,
            step: 0,
        }
    }
}

impl Default for RecurrentState {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified cache for hybrid model layers.
///
/// Full attention layers use KV cache; DeltaNet layers use recurrent state.
pub enum HybridCache {
    KV(KVCache),
    Recurrent(RecurrentState),
}

impl HybridCache {
    /// Get the number of tokens processed so far.
    pub fn offset(&self) -> i32 {
        match self {
            HybridCache::KV(kv) => kv.offset(),
            HybridCache::Recurrent(rec) => rec.step,
        }
    }

    pub fn as_kv_mut(&mut self) -> Option<&mut KVCache> {
        match self {
            HybridCache::KV(kv) => Some(kv),
            _ => None,
        }
    }

    pub fn as_recurrent_mut(&mut self) -> Option<&mut RecurrentState> {
        match self {
            HybridCache::Recurrent(rec) => Some(rec),
            _ => None,
        }
    }
}
