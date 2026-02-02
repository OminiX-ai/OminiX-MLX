# mlx-rs-core Refactor Plan

## Goal

Reorganize the codebase so that:
1. **mlx-rs-core** contains only shared infrastructure (model-agnostic)
2. **xxx-mlx** crates contain model-specific implementations

## Current State

### mlx-rs-core (good - keep as is)
```
mlx-rs-core/
├── cache.rs          # KVCache, ConcatKeyValueCache, KeyValueCache trait
├── utils.rs          # RoPE, attention masks, SDPA
├── metal_kernels.rs  # fused_swiglu
├── error.rs          # Common error types
└── lib.rs            # load_tokenizer, re-exports
```

### mlx-rs-lm (needs refactoring)
```
mlx-rs-lm/
├── models/           # ❌ Should move to xxx-mlx crates
│   ├── bert.rs       → gpt-sovits-mlx
│   ├── hubert.rs     → gpt-sovits-mlx
│   ├── vits.rs       → gpt-sovits-mlx
│   ├── sovits.rs     → gpt-sovits-mlx
│   ├── t2s.rs        → gpt-sovits-mlx
│   ├── paraformer.rs → funasr-nano-mlx
│   ├── qwen2.rs      → qwen3-mlx (or qwen-mlx)
│   ├── qwen3.rs      → qwen3-mlx
│   ├── qwen3_moe.rs  → qwen3-mlx
│   ├── glm4.rs       → glm4-mlx
│   ├── glm4_moe.rs   → glm4-moe-mlx
│   └── mixtral.rs    → mixtral-mlx
│
├── text/             # ⚠️ Split: shared vs TTS-specific
│   ├── preprocessor.rs  → gpt-sovits-mlx (TTS-specific)
│   ├── bert_features.rs → gpt-sovits-mlx (TTS-specific)
│   ├── g2pw.rs          → gpt-sovits-mlx (Chinese G2P)
│   ├── cmudict.rs       → gpt-sovits-mlx (English G2P)
│   ├── symbols.rs       → gpt-sovits-mlx (phoneme symbols)
│   └── mod.rs
│
├── audio.rs          # ✅ Move to mlx-rs-core (shared audio utils)
├── cache.rs          # ⚠️ Duplicate of mlx-rs-core, remove
├── utils/            # ⚠️ Merge with mlx-rs-core
│   ├── rope.rs       # Already in mlx-rs-core
│   └── tokenizer.rs  # Move to mlx-rs-core
│
├── voice_clone.rs    # ❌ Move to gpt-sovits-mlx
├── inference.rs      # ✅ Move to mlx-rs-core (generic inference)
├── sampler.rs        # ✅ Move to mlx-rs-core
├── compiled_ops.rs   # ✅ Move to mlx-rs-core
├── speculative.rs    # ✅ Move to mlx-rs-core
├── metal_kernels.rs  # ⚠️ Duplicate, merge with mlx-rs-core
├── error.rs          # ⚠️ Duplicate, merge with mlx-rs-core
└── generate/         # ✅ Move to mlx-rs-core
```

---

## Target Architecture

```
mlx-rs-core/                    # Shared infrastructure (model-agnostic)
├── cache.rs                    # KV cache implementations
├── utils.rs                    # RoPE, attention, SDPA
├── audio.rs                    # STFT, mel, resampling (NEW)
├── sampler.rs                  # Token sampling (NEW)
├── inference.rs                # Generation loop (NEW)
├── compiled_ops.rs             # Compiled MLX ops (NEW)
├── speculative.rs              # Speculative decoding (NEW)
├── metal_kernels.rs            # Custom Metal kernels
├── error.rs                    # Common error types
├── tokenizer.rs                # Tokenizer utils (NEW)
└── lib.rs

qwen3-mlx/                      # Qwen model family
├── qwen2.rs                    # Qwen2 (MOVED)
├── qwen3.rs                    # Qwen3 dense
├── qwen3_moe.rs                # Qwen3 MoE (MOVED)
├── qwen3_vl.rs                 # (TODO: port from Python)
└── lib.rs

glm4-mlx/                       # GLM4 model family
├── model.rs                    # GLM4 dense
└── lib.rs

glm4-moe-mlx/                   # GLM4 MoE (separate due to size)
├── model.rs                    # GLM4 MoE
└── lib.rs

mixtral-mlx/                    # Mixtral MoE
├── model.rs                    # (MOVED from mlx-rs-lm)
└── lib.rs

gpt-sovits-mlx/rust/            # Voice synthesis (GPT-SoVITS) - ALREADY EXISTS
├── models/
│   ├── bert.rs                 # BERT encoder ✅
│   ├── hubert.rs               # HuBERT encoder ✅
│   ├── vits.rs                 # VITS vocoder ✅
│   ├── sovits.rs               # SoVITS ✅
│   └── t2s.rs                  # Text-to-Semantic ✅
├── text/
│   ├── preprocessor.rs         # Text normalization ✅
│   ├── bert_features.rs        # BERT features ✅
│   ├── g2pw.rs                 # Chinese G2P ✅
│   ├── g2p_en.rs               # English G2P ✅
│   ├── cmudict.rs              # CMU dictionary ✅
│   └── symbols.rs              # Phoneme symbols ✅
├── voice_clone.rs              # Full pipeline ✅
├── inference.rs                # ✅
├── audio.rs                    # ⚠️ Remove, use mlx-rs-core
├── cache.rs                    # ⚠️ Remove, use mlx-rs-core
├── error.rs                    # ⚠️ Remove, use mlx-rs-core
└── lib.rs

funasr-nano-mlx/                # ASR models
├── paraformer.rs               # (MOVED from mlx-rs-lm)
├── model.rs                    # Existing
└── lib.rs

zimage-mlx/                     # Z-Image (existing)
flux-klein-mlx/                 # FLUX (existing)
```

---

## Migration Steps

### Phase 1: Consolidate mlx-rs-core (shared infra)

1. **Move audio processing to mlx-rs-core**
   - Copy `mlx-rs-lm/src/audio.rs` → `mlx-rs-core/src/audio.rs`
   - Update exports in `mlx-rs-core/src/lib.rs`

2. **Move sampling/generation to mlx-rs-core**
   - Copy `mlx-rs-lm/src/sampler.rs` → `mlx-rs-core/src/sampler.rs`
   - Copy `mlx-rs-lm/src/inference.rs` → `mlx-rs-core/src/inference.rs`
   - Copy `mlx-rs-lm/src/generate/` → `mlx-rs-core/src/generate/`

3. **Move compiled ops to mlx-rs-core**
   - Copy `mlx-rs-lm/src/compiled_ops.rs` → `mlx-rs-core/src/compiled_ops.rs`
   - Copy `mlx-rs-lm/src/speculative.rs` → `mlx-rs-core/src/speculative.rs`

4. **Merge duplicates**
   - Merge `mlx-rs-lm/src/metal_kernels.rs` into `mlx-rs-core/src/metal_kernels.rs`
   - Remove duplicate cache/error/utils

### Phase 2: Move Qwen models to qwen3-mlx

1. Move `mlx-rs-lm/src/models/qwen2.rs` → `qwen3-mlx/src/qwen2.rs`
2. Move `mlx-rs-lm/src/models/qwen3_moe.rs` → `qwen3-mlx/src/qwen3_moe.rs`
3. Update `qwen3-mlx/src/lib.rs` to export all models
4. Update dependencies to use `mlx-rs-core`

### Phase 3: Verify GLM4/Mixtral crates

1. Verify `glm4-mlx` has complete implementation (compare with mlx-rs-lm)
2. Verify `glm4-moe-mlx` has complete implementation
3. Move `mlx-rs-lm/src/models/mixtral.rs` → `mixtral-mlx/src/model.rs`
4. Remove duplicates from mlx-rs-lm

### Phase 4: Update gpt-sovits-mlx to use mlx-rs-core

**gpt-sovits-mlx already exists with complete implementation:**
```
gpt-sovits-mlx/rust/src/
├── models/           # bert, hubert, vits, sovits, t2s ✅
├── text/             # preprocessor, g2pw, cmudict, symbols, g2p_en ✅
├── voice_clone.rs    ✅
├── audio.rs          ⚠️ Duplicate - should use mlx-rs-core
├── cache.rs          ⚠️ Duplicate - should use mlx-rs-core
├── error.rs          ⚠️ Duplicate - should use mlx-rs-core
└── inference.rs      ✅
```

1. Update gpt-sovits-mlx to depend on mlx-rs-core
2. Remove duplicate files (audio.rs, cache.rs, error.rs)
3. Import shared utilities from mlx-rs-core instead
4. Remove duplicate TTS code from mlx-rs-lm

### Phase 5: Update funasr-nano-mlx

1. Move `mlx-rs-lm/src/models/paraformer.rs` → `funasr-nano-mlx/src/paraformer.rs`
2. Remove duplicate from mlx-rs-lm
3. Verify all ASR functionality works

### Phase 6: Deprecate/Remove mlx-rs-lm

1. Update all dependent crates to use xxx-mlx crates instead
2. Remove `mlx-rs-lm/src/models/` (now empty)
3. Remove `mlx-rs-lm/src/text/` (moved to gpt-sovits-mlx)
4. Either:
   - Delete mlx-rs-lm entirely, OR
   - Keep as meta-crate that re-exports from xxx-mlx crates

---

## Dependency Graph (Target)

```
                    mlx-rs (core bindings)
                           │
                    mlx-rs-core (shared infra)
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    qwen3-mlx         glm4-mlx          mixtral-mlx
         │                 │                 │
         │           glm4-moe-mlx            │
         │                                   │
         └────────────┬────────────┬─────────┘
                      │            │
              gpt-sovits-mlx   funasr-nano-mlx
                      │            │
                      └─────┬──────┘
                            │
                      zimage-mlx
                      flux-klein-mlx
                            │
                       OminiX-API
```

---

## Estimated Effort

| Phase | Description | Files | Effort |
|-------|-------------|-------|--------|
| 1 | Consolidate mlx-rs-core | ~8 files | 2-3 hours |
| 2 | Move Qwen models | ~3 files | 1 hour |
| 3 | Verify GLM4/Mixtral | ~3 files | 1 hour |
| 4 | Update gpt-sovits-mlx (use mlx-rs-core) | ~5 files | 1-2 hours |
| 5 | Update funasr-nano-mlx | ~2 files | 30 min |
| 6 | Deprecate mlx-rs-lm | cleanup | 1 hour |

**Total: ~7-9 hours**

---

## Success Criteria

1. All crates compile with `cargo check --workspace`
2. `mlx-rs-core` contains only model-agnostic code
3. Each xxx-mlx crate is self-contained for its model family
4. No duplicate code across crates
5. OminiX-API works with the new structure
6. All existing examples still work
