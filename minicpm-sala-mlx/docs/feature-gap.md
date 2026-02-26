# MiniCPM-SALA MLX Port: Feature Gap Analysis

Source model: [openbmb/MiniCPM-SALA](https://huggingface.co/openbmb/MiniCPM-SALA)

---

## 1. Model Summary

| Property | Value |
|---|---|
| Parameters | 9B (18.95 GB BF16) |
| Layers | 32 transformer layers |
| Architecture | Hybrid: 8 sparse (InfLLMv2) + 24 linear (Lightning/GLA) |
| GQA | Sparse: 32 heads / 2 KV heads; Linear: 32/32 |
| Hidden / Intermediate | 4096 / 16384 |
| Vocab | 73,448 tokens |
| Max Context | 524,288 (trained), 1M+ (claimed) |
| Activation | SiLU |
| Norm | RMSNorm (eps 1e-6) |
| Positional Encoding | RoPE (theta 10000), per-layer configurable |

### Per-Layer Mixer Layout

8 sparse (`minicpm4`) + 24 linear (`lightning-attn`), irregularly distributed:

```
 0: minicpm4          16: minicpm4
 1: lightning-attn    17: minicpm4
 2: lightning-attn    18: lightning-attn
 3: lightning-attn    19: lightning-attn
 4: lightning-attn    20: lightning-attn
 5: lightning-attn    21: lightning-attn
 6: lightning-attn    22: minicpm4
 7: lightning-attn    23: lightning-attn
 8: lightning-attn    24: lightning-attn
 9: minicpm4          25: lightning-attn
10: lightning-attn    26: lightning-attn
11: lightning-attn    27: lightning-attn
12: lightning-attn    28: lightning-attn
13: lightning-attn    29: minicpm4
14: lightning-attn    30: minicpm4
15: lightning-attn    31: minicpm4
```

Sparse layers: 0, 9, 16, 17, 22, 29, 30, 31 — cluster at boundaries and periodically in the middle.

### Special Config Values

- `scale_emb: 12` — muP embedding scaling
- `scale_depth: 1.4` — muP residual scaling
- `dim_model_base: 256` — muP reference width
- `attn_use_rope: false` — sparse layers do NOT apply RoPE to main Q/K
- `lightning_use_rope: true` — linear layers DO apply RoPE
- `qk_norm: true` — RMSNorm on Q and K per-head
- `use_output_gate: true` — gating on lightning attention output
- `attn_use_output_gate: true` — gating on sparse attention output

### Sparse Attention Config

```json
{
  "kernel_size": 32,
  "kernel_stride": 16,
  "init_blocks": 1,
  "block_size": 64,
  "window_size": 2048,
  "topk": 64,
  "use_nope": false,
  "dense_len": 8192
}
```

---

## 2. Architecture Detail

### CUDA/Triton → Metal Porting Stack

```
┌─────────────────────────────────────────────────────┐
│               MiniCPM-SALA (CUDA)                   │
├─────────────────────────────────────────────────────┤
│  SGLang Server (minicpm_flashinfer backend)         │
│       ↓                                             │
│  ┌──────────────────┐  ┌──────────────────┐         │
│  │ infllmv2_cuda    │  │ flash-linear-    │         │
│  │ (sparse attn,    │  │ attention        │         │
│  │  CompressK,      │  │ (chunk_simple_   │         │
│  │  top-k select)   │  │  gla, fused_     │         │
│  │                  │  │  recurrent_gla)  │         │
│  └──────────────────┘  └──────────────────┘         │
│       ↓                      ↓                      │
│  ┌──────────────────────────────────────────┐       │
│  │          CUDA / Triton Kernels            │       │
│  └──────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘

                 ↓ PORT TO ↓

┌─────────────────────────────────────────────────────┐
│               MLX-RS (Metal)                        │
├─────────────────────────────────────────────────────┤
│  mlx-rs + mlx-rs-core Rust bindings                │
│       ↓                                             │
│  ┌──────────────────┐  ┌──────────────────┐         │
│  │ Standard SDPA    │  │ Naive GLA        │         │
│  │ (fallback for    │  │ (matmul-based    │  Phase 1│
│  │  sparse layers)  │  │  recurrent)      │         │
│  └──────────────────┘  └──────────────────┘         │
│  ┌──────────────────┐  ┌──────────────────┐         │
│  │ Custom Metal     │  │ Custom Metal     │         │
│  │ Sparse Kernel    │  │ GLA Kernel       │  Phase 2│
│  │ (MSL shader)     │  │ (MSL shader)     │    + 3  │
│  └──────────────────┘  └──────────────────┘         │
│       ↓                      ↓                      │
│  ┌──────────────────────────────────────────┐       │
│  │          Metal Shading Language            │       │
│  └──────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

### Sparse Attention (InfLLMv2) — 8 layers

Block-sparse attention with hierarchical key compression and top-k selection.

```
Query token
    │
    ├─── Window attention ──── last 2048 tokens (full attention)
    │
    ├─── CompressK (1x) ───── kernel_size=32, stride=16
    │         │                    mean-pool chunks → compressed keys
    │         ↓
    ├─── CompressK (4x) ───── kernel_size=128, stride=64
    │         │                    coarser compression for distant context
    │         ↓
    ├─── Stage 1 scores ───── attention on compressed keys
    │         ↓
    ├─── Top-K selection ──── pick 64 best blocks (block_size=64)
    │         ↓
    └─── Stage 2 attn ─────── full attention on selected blocks only
```

When `dense_len` < 8192: falls back to standard dense SDPA (no compression needed).

### Linear Attention (Lightning/GLA) — 24 layers

Gated Linear Attention with learned per-head decay. NOT elu+1 feature-map based.

```
Prefill (chunked):
  Sequence: [──Chunk 1──][──Chunk 2──][──Chunk 3──] ...
                    ↓
  State:        S₀ → S₁ → S₂ → S₃ ...

  Per chunk:
    1. Intra-chunk: quadratic attention within chunk
    2. Inter-chunk: query against accumulated state S
    3. Update: S_{i+1} = decay * S_i + K_chunk^T @ V_chunk

Decode (recurrent, per token):
    state_{t+1} = decay * state_t + k_t^T ⊗ v_t
    output_t    = q_t @ state_t

  State shape: [n_heads, head_dim, head_dim] — fixed size, context-independent
```

### Weight Tensor Names

Per-layer tensors (prefix `model.layers.{i}.`):

| Tensor | Shape | Present In |
|---|---|---|
| `self_attn.q_proj.weight` | [n_h * d, D] | All layers |
| `self_attn.k_proj.weight` | [n_kv * d, D] | All layers |
| `self_attn.v_proj.weight` | [n_kv * d, D] | All layers |
| `self_attn.o_proj.weight` | [D, n_h * d] | All layers |
| `self_attn.q_norm.weight` | [d] | Lightning layers |
| `self_attn.k_norm.weight` | [d] | Lightning layers |
| `self_attn.o_norm.weight` | [D] | Lightning layers |
| `self_attn.o_gate.weight` | [n_h * d, D] | Sparse layers |
| `self_attn.z_proj.weight` | [n_h * d, D] | Lightning layers |
| `mlp.gate_proj.weight` | [I, D] | All layers |
| `mlp.up_proj.weight` | [I, D] | All layers |
| `mlp.down_proj.weight` | [D, I] | All layers |
| `input_layernorm.weight` | [D] | All layers |
| `post_attention_layernorm.weight` | [D] | All layers |

Global tensors:

| Tensor | Shape |
|---|---|
| `model.embed_tokens.weight` | [73448, 4096] |
| `model.norm.weight` | [4096] |
| `lm_head.weight` | [73448, 4096] |

Where D=4096, I=16384, d=128, n_h=32, n_kv varies (2 for sparse, 32 for lightning).

---

## 3. Gap Analysis

### GREEN — Ready in mlx-rs

| Component | mlx-rs API | Notes |
|---|---|---|
| RMSNorm | `nn::RmsNorm` | Fast Metal kernel via `fast::rms_norm` |
| RoPE | `nn::Rope` | Fast Metal kernel via `fast::rope`; supports theta config |
| SwiGLU MLP | `nn::Linear` + `nn::silu` | Fused `fused_swiglu` Metal kernel available in mlx-rs-core |
| QK Norm | `nn::RmsNorm` | Apply per-head to Q/K before RoPE; same primitive |
| Embedding | `nn::Embedding` | Standard |
| LM Head | `nn::Linear` | Standard; `tie_word_embeddings: false` so separate weight |
| Safetensors loading | `Array::load_safetensors` | Multi-shard with index.json; proven pattern in qwen3-mlx |
| GQA (32h/2kv) | `fast::scaled_dot_product_attention` | Automatic KV head broadcasting |
| MHA (32h/32kv) | `fast::scaled_dot_product_attention` | Standard path |
| Causal masking | `ScaledDotProductAttentionMask::Causal` | Hardware-optimized |
| KV cache | `KVCache` (pre-allocated) | 256-token step allocation |
| Quantization | `MaybeQuantized<Linear>` | 4-bit and 8-bit supported |
| muP scaling | Scalar arithmetic | `scale_emb * embed`, `residual / scale_depth` |

### YELLOW — Implementable with standard ops

| Component | What's Missing | Implementation Path | Effort |
|---|---|---|---|
| Output gating (`o_gate`) | Extra Linear + sigmoid on attn output | `sigmoid(o_gate(x)) * o_proj(attn_out)` — two extra `nn::Linear` + element-wise | ~2 hours |
| Output norm (`o_norm`) | Extra RMSNorm on lightning attn output | One more `nn::RmsNorm` instance | ~30 min |
| Gating projection (`z_proj`) | Extra Linear for lightning gating | Standard `nn::Linear`, feeds into gate computation | ~1 hour |
| Lightning Attention decode | No GLA recurrent kernel | Three matmuls per step: `state = decay * state + k^T @ v; out = q @ state`. Standard `Array::matmul` + element-wise multiply. | ~2 days |
| Lightning Attention prefill | No `chunk_simple_gla` kernel | Option A: Naive loop over tokens (slow but correct). Option B: Chunked quadratic attention within fixed-size windows. | ~4 days |
| Lightning recurrent state cache | No `LightningCacheLayer` | New struct holding `[n_heads, head_dim, head_dim]` state matrix + decay accumulator. Simple to implement. | ~1 day |
| Decay factor computation | Lightning attention uses learned per-head decay | Stored as parameter, applied as element-wise scaling on recurrent state. | ~1 hour |
| Weight key mapping | Non-standard keys (`z_proj`, `o_gate`, `o_norm`) | Custom `load_weights` function with name remapping. | ~3 hours |

### RED — Major gaps

| Component | What's Missing | Why It's Hard | Fallback | Effort |
|---|---|---|---|---|
| **InfLLMv2 sparse attention** | CUDA kernels `infllmv2_attn_varlen_func`, `infllmv2_attn_stage1` — no Metal equivalent | Block-sparse attention with variable-length masking requires a custom Metal kernel; involves top-k index gathering, block-level attention scoring, and masked scattered reads. | **Use standard SDPA.** The model code explicitly falls back to dense attention when sparse kernels are unavailable. Quality is identical for context < `dense_len` (8192). | 4-8 weeks for Metal kernel |
| **CompressK (hierarchical key compression)** | Strided chunk extraction + mean pooling at two compression scales (1x and 4x) | Variable-length batched chunk indexing with stride, reshape, and mean — complex indexing logic that needs to handle sequence boundaries correctly. | Not needed if using dense SDPA fallback. Can be implemented with standard ops (`reshape`/`mean`/`index_select`) at reduced speed if sparse path is pursued. | 1-2 weeks |
| **Top-K block selection** | Variable-length batched `topk` with causal masking | MLX has `Array::topk()` but the batched variable-length version with per-query causal bounds requires custom logic. | Not needed if using dense SDPA fallback. | 1 week |
| **Fused GLA Metal kernel** | `chunk_simple_gla` and `fused_recurrent_simple_gla` as Metal compute shaders | Linear attention's chunked parallel scan and fused recurrent update are performance-critical for 24/32 layers. Without them, prefill is O(n) with high constant factor. | Use naive implementation (YELLOW above). Correct but slower. Decode speed is acceptable; prefill will be the bottleneck. | 4-6 weeks for Metal kernel |

---

## 4. Implementation Roadmap

### Project Structure

```
minicpm-sala-mlx/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── config.rs              # Config parsing (MiniCPMSALAConfig)
│   ├── model.rs               # Model architecture + weight loading
│   ├── attention/
│   │   ├── mod.rs             # HybridAttention enum dispatch
│   │   ├── sparse.rs          # InfLLMv2 / dense SDPA fallback
│   │   └── lightning.rs       # GLA linear attention
│   ├── cache.rs               # KVCache + LightningCache
│   └── generate.rs            # Text generation loop
├── examples/
│   └── generate.rs            # CLI inference example
└── docs/
    └── feature-gap.md         # This file
```

### Phase 1: Working Inference with Fallbacks (2-3 weeks)

**Target: correct output, competitive decode speed at short/medium context.**

- [x] **1.1 Config parsing** — deserialize `config.json` with `mixer_types`, `sparse_config`, all special flags
- [x] **1.2 Weight loading** — multi-shard safetensors, map `model.` prefix, handle per-layer-type tensors (`z_proj`/`o_gate`/`o_norm` only on applicable layers)
- [x] **1.3 Standard components** — port from existing qwen3-mlx patterns: RMSNorm, RoPE, SwiGLU MLP, embeddings, LM head
- [x] **1.4 Sparse layers (`minicpm4`)** — standard SDPA with GQA (32h/2kv), standard `KVCache`, output gating (no QK norm on sparse layers)
- [x] **1.5 Lightning layers** — naive GLA: decode via recurrent state, prefill via token loop, `LightningCache` for recurrent state, QK norm, output norm/gate
- [x] **1.6 HybridAttention dispatch** — enum routing per `mixer_types[layer_idx]`
- [x] **1.7 Text generation** — greedy + temperature sampling, EOS handling (tokens 2 and 73440), incremental tokenizer decode
- [x] **1.8 Validation** — generates coherent text with `<think>` reasoning, correct arithmetic

```rust
// Core dispatch pattern
pub enum HybridAttention {
    Sparse(SparseAttention),    // → standard SDPA (Phase 1) or InfLLMv2 (Phase 3)
    Lightning(LightningAttention), // → naive GLA (Phase 1) or fused Metal (Phase 2)
}

impl HybridAttention {
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        sparse_cache: Option<&mut KVCache>,
        lightning_cache: Option<&mut LightningCache>,
    ) -> Result<Array> {
        match self {
            Self::Sparse(attn) => attn.forward(x, mask, sparse_cache.unwrap()),
            Self::Lightning(attn) => attn.forward(x, lightning_cache.unwrap()),
        }
    }
}
```

### Phase 2: Optimized Lightning Attention (2-4 weeks)

**Target: fast prefill for the 24 linear attention layers.**

- [ ] **2.1 Metal kernel for GLA recurrent** — fuse `decay * state + outer(k, v)` + query
- [ ] **2.2 Metal kernel for GLA chunked prefill** — parallel chunk processing with state passing
- [ ] **2.3 Benchmark** — measure prefill tok/s at 1K, 4K, 16K, 64K contexts
- [ ] **2.4 Tune chunk sizes** — find optimal chunk size for M-series GPUs

### Phase 3: Full Sparse Attention Path — Optional (4-8 weeks)

**Target: true long-context (>8K) efficiency for sparse layers.**

- [ ] **3.1 CompressK Metal kernel** — strided chunk extraction + mean pooling
- [ ] **3.2 Sparse block SDPA Metal kernel** — top-k gather + block-level attention
- [ ] **3.3 InfLLMv2 cache** — compressed/uncompressed dual KV storage
- [ ] **3.4 Integration** — consider `universal-metal-flash-attention` if applicable
- [ ] **3.5 Long-context validation** — test at 32K, 128K, 256K, 1M tokens

### Phase 4: Polish (1-2 weeks)

- [ ] **4.1 API cleanup** — public interface, error types
- [ ] **4.2 Quantization** — 8-bit and 4-bit model loading
- [ ] **4.3 Example** — CLI generate binary with streaming output
- [ ] **4.4 Add to workspace** — integrate into OminiX-MLX Cargo workspace

---

## 5. Memory Requirements

| Precision | Model Size | Min Mac RAM |
|---|---|---|
| BF16 | ~19 GB | 32 GB |
| 8-bit | ~9.5 GB | 16 GB |
| 4-bit | ~5 GB | 8 GB |

### KV Cache Overhead

- **Sparse layers (standard KV cache)**: 2 KV heads * 128 dim * 8 layers * 2 bytes = **4 KB/token**
- **Lightning layers (recurrent state)**: 32 heads * 128 * 128 * 24 layers * 2 bytes = **25 MB fixed** (context-independent)

The linear attention layers have **constant memory** regardless of context length — this advantage is preserved even with the naive implementation.

### Performance Targets (estimated, M3 Max)

| Context | Dense Fallback | With GLA Metal | With Full Sparse |
|---|---|---|---|
| 4K | ~30 tok/s | ~30 tok/s | ~30 tok/s |
| 32K | ~15 tok/s | ~20 tok/s | ~25 tok/s |
| 128K | ~3 tok/s | ~10 tok/s | ~15 tok/s |
| 256K | OOM risk | ~5 tok/s | ~10 tok/s |
| 1M | OOM | OOM | ~3 tok/s |

Long context primarily benefits from Phase 3 (sparse attention reduces KV memory for the 8 dense layers).

---

## 6. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Lightning attention numerical accuracy without fused kernels | Low | Math is identical; only speed differs. Validate against PyTorch reference output. |
| Sparse-to-dense fallback quality at long context | Medium | Model was trained with sparse attention; dense fallback works for short context but may degrade >8K. Document limitation. |
| 9B model memory on Apple Silicon | Medium | 8-bit quantization fits 16 GB Macs. 4-bit fits 8 GB. Quantization path proven in qwen3-mlx. |
| Lightning prefill speed without Metal kernel | Medium | 24/32 layers use lightning attention; naive prefill will dominate latency. Prioritize Phase 2. |
| BF16 precision | Low | MLX on Apple Silicon supports BF16 natively. No issue. |
| Upstream model changes | Low | Pin to specific HF commit hash for weight downloads. |

---

## 7. Dependencies

```toml
[dependencies]
mlx-rs = { path = "../mlx-rs" }
mlx-rs-core = { path = "../mlx-rs-core" }
tokenizers = "0.21"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
hf-hub = "0.4"
```

### System Requirements

- macOS 14.0+ (Metal 3)
- Apple Silicon (M1/M2/M3/M4)
- 16 GB+ RAM (8-bit), 32 GB+ (BF16)

---

## 8. References

- Model: https://huggingface.co/openbmb/MiniCPM-SALA
- Technical report: https://github.com/OpenBMB/MiniCPM/blob/main/docs/MiniCPM_SALA.pdf
- InfLLM-V2 paper: https://arxiv.org/abs/2509.24663
- HyPE paper: https://arxiv.org/abs/2601.22156
- Flash Linear Attention (GLA): https://github.com/sustcsonglin/flash-linear-attention
- MiniCPM GitHub: https://github.com/OpenBMB/MiniCPM
