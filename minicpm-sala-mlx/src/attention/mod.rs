pub mod lightning;
pub mod sparse;

pub use lightning::{LightningAttention, LightningCache};
pub use sparse::{SparseAttention, SparseKVCache};

use mlx_rs::{
    error::Exception,
    module::{
        ModuleParamMut, ModuleParamRef, ModuleParameters as ModuleParametersTrait,
    },
    Array,
};

use crate::config::ModelArgs;

/// Per-layer cache: either a SparseKVCache (sparse layers) or a recurrent
/// state (lightning layers).
#[derive(Debug, Clone)]
pub enum LayerCache {
    Sparse(SparseKVCache),
    Lightning(LightningCache),
}

impl LayerCache {
    pub fn offset(&self) -> i32 {
        match self {
            LayerCache::Sparse(c) => c.offset(),
            LayerCache::Lightning(c) => c.offset,
        }
    }
}

/// Hybrid attention dispatch: sparse (SDPA) or lightning (GLA).
#[derive(Debug)]
pub enum HybridAttention {
    Sparse(SparseAttention),
    Lightning(LightningAttention),
}

impl HybridAttention {
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut LayerCache,
    ) -> Result<Array, Exception> {
        match (self, cache) {
            (HybridAttention::Sparse(attn), LayerCache::Sparse(kv)) => {
                attn.forward(x, mask, kv)
            }
            (HybridAttention::Lightning(attn), LayerCache::Lightning(lc)) => {
                attn.forward(x, lc)
            }
            _ => Err(Exception::custom("Attention/cache type mismatch")),
        }
    }

    pub fn training_mode(&mut self, mode: bool) {
        match self {
            HybridAttention::Sparse(a) => a.training_mode(mode),
            HybridAttention::Lightning(a) => a.training_mode(mode),
        }
    }
}

// Manual ModuleParameters impl for enum dispatch
impl ModuleParametersTrait for HybridAttention {
    fn num_parameters(&self) -> usize {
        match self {
            HybridAttention::Sparse(a) => a.num_parameters(),
            HybridAttention::Lightning(a) => a.num_parameters(),
        }
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            HybridAttention::Sparse(a) => a.parameters(),
            HybridAttention::Lightning(a) => a.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            HybridAttention::Sparse(a) => a.parameters_mut(),
            HybridAttention::Lightning(a) => a.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        match self {
            HybridAttention::Sparse(a) => a.trainable_parameters(),
            HybridAttention::Lightning(a) => a.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            HybridAttention::Sparse(a) => a.freeze_parameters(recursive),
            HybridAttention::Lightning(a) => a.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            HybridAttention::Sparse(a) => a.unfreeze_parameters(recursive),
            HybridAttention::Lightning(a) => a.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            HybridAttention::Sparse(a) => a.all_frozen(),
            HybridAttention::Lightning(a) => a.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            HybridAttention::Sparse(a) => a.any_frozen(),
            HybridAttention::Lightning(a) => a.any_frozen(),
        }
    }
}

/// Create the per-layer cache vector based on mixer_types.
pub fn create_layer_caches(args: &ModelArgs) -> Vec<LayerCache> {
    let default_sparse_config = crate::config::SparseConfig::default();
    let sparse_config = args.sparse_config.as_ref().unwrap_or(&default_sparse_config);

    (0..args.num_hidden_layers as usize)
        .map(|i| {
            if args.is_sparse_layer(i) {
                LayerCache::Sparse(SparseKVCache::new(sparse_config.clone()))
            } else {
                LayerCache::Lightning(LightningCache::new(
                    args.lightning_num_heads(),
                    args.lightning_head_dim(),
                ))
            }
        })
        .collect()
}
