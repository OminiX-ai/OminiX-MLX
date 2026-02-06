# glm-4.7-flash-mlx

GLM-4.7-Flash (30B-A3B MoE) inference on Apple Silicon with MLX.

## Model Overview

GLM-4.7-Flash is a 30B-A3B Mixture of Experts model by Zhipu AI (zai-org). It has 31B total parameters but only ~3B active per token, making it highly efficient for its capability level. Licensed under MIT.

- **Architecture**: DeepSeek2-style MoE
- **Total Parameters**: 31B
- **Active Parameters**: ~3B per token
- **Context Length**: 32,768 tokens
- **License**: MIT
- **Original**: [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)

### Benchmarks (Official)

| Benchmark | GLM-4.7-Flash | Qwen3-30B-A3B | GPT-OSS-20B |
|-----------|---------------|---------------|-------------|
| AIME 25 | **91.6** | 85.0 | 91.7 |
| GPQA | **75.2** | 73.4 | 71.5 |
| LCB v6 | 64.0 | **66.0** | 61.0 |
| HLE | **14.4** | 9.8 | 10.9 |
| SWE-bench Verified | **59.2** | 22.0 | 34.0 |
| tau2-Bench | **79.5** | 49.0 | 47.7 |
| BrowseComp | **42.8** | 2.29 | 28.3 |

## Quantization Analysis

### GGUF Variants

#### Perplexity Comparison (ubergarm, wiki.test.raw, n_ctx=512)

| Quant | Provider | Size | PPL (lower=better) | Notes |
|-------|----------|------|---------------------|-------|
| MXFP4 | ubergarm | 15.9 GiB | **8.4759** | Best PPL of any variant; anomalously better than FP |
| IQ5_K | ubergarm | 21.2 GiB | 9.7951 | Custom mix (q8_0 attn + iq5_k/iq6_k experts); beats BF16 |
| Q8_0 | ubergarm | 29.6 GiB | 9.8206 | Near-baseline |
| BF16 | ubergarm | 55.8 GiB | 9.8537 | Full precision baseline |
| smol-IQ4_KSS | ubergarm | 14.9 GiB | 10.2529 | Noticeable degradation |

**Key finding**: MXFP4 achieves paradoxically lower perplexity than BF16. This is likely because the MoE gating/expert architecture responds favorably to MXFP4's uniform quantization scheme. The IQ5_K custom quant also beats full precision while being 2.6x smaller.

#### Quality Tiers (bartowski, llama.cpp b7779)

| Tier | Quants | Size Range | bartowski Rating |
|------|--------|------------|------------------|
| Maximum | BF16, Q8_0 | 31.8-59.9 GB | Baseline |
| Near-perfect | Q6_K, Q6_K_L | 24.8-25.0 GB | Recommended |
| High quality | Q5_K_S, Q5_K_M, Q5_K_L | 20.8-21.8 GB | Recommended |
| Good default | Q4_K_M, Q4_K_L, Q4_K_S | 17.8-18.7 GB | Recommended |
| Decent | IQ4_XS, IQ4_NL, Q4_0, Q4_1 | 16.3-19.0 GB | Good |
| Low RAM | Q3_K_XL, Q3_K_L, Q3_K_M | 14.1-14.8 GB | Lower quality |
| Very low | IQ3_M, IQ3_XS, IQ3_XXS | 12.3-14.0 GB | Not recommended |
| Extreme | Q2_K, Q2_K_L, IQ2_M, IQ2_S | 8.7-11.4 GB | Surprisingly usable |
| Minimum | IQ2_XS, IQ2_XXS | 7.6-8.7 GB | Usable (SOTA techniques) |

#### GGUF Provider Comparison

| Provider | Quant Count | Highlights |
|----------|-------------|------------|
| [bartowski](https://huggingface.co/bartowski/zai-org_GLM-4.7-Flash-GGUF) | 26 | Most comprehensive; quality ratings; imatrix calibrated; llama.cpp b7779 |
| [unsloth](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF) | ~20 | Dynamic 2.0 quants; claims superior accuracy; 289K downloads/month |
| [ubergarm](https://huggingface.co/ubergarm/GLM-4.7-Flash-GGUF) | 5 | Only provider with perplexity data; gating function fixes; MXFP4 discovery |
| [ngxson](https://huggingface.co/ngxson/GLM-4.7-Flash-GGUF) | 3 | Q4_K_M, Q8_0, F16 only; 11.6K downloads/month |

### MLX Variants

#### Quality Comparison (inferencerlabs, measured on evaluation set)

| Quant | Size | PPL (lower=better) | Token Accuracy | Eff. Divergence | Missed Divergence |
|-------|------|---------------------|----------------|-----------------|-------------------|
| Base (FP) | ~60 GB | 1.219 | 100.0% | 0.00% | 0.00% |
| q8.5 | 31.8 GB | 1.227 | 97.80% | 0.22% | 9.81% |
| **q6.5** | **24.3 GB** | **1.227** | **96.65%** | **0.38%** | **11.23%** |
| q5.5 | ~21 GB | 1.250 | 94.45% | 0.96% | 17.36% |
| q4.5 | 16.9 GB | 1.367 | 90.30% | 2.82% | 29.04% |

**Key finding**: q6.5 matches q8.5 in perplexity (1.227 vs 1.227) while being 7.5 GB smaller. The jump from q6.5 to q5.5 is significant (96.65% -> 94.45% accuracy), making q6.5 the clear sweet spot.

#### Inference Performance (inferencerlabs q6.5, M3 Ultra 512GB)

| Metric | Value |
|--------|-------|
| Single inference | ~61 tokens/s @ 1000 tokens |
| Batched (3x) | ~120 total tokens/s |
| Memory usage | ~26 GB |

#### MLX Provider Comparison

| Provider | Bits | Size | Downloads/month | Notes |
|----------|------|------|-----------------|-------|
| [lmstudio-community/MLX-8bit](https://huggingface.co/lmstudio-community/GLM-4.7-Flash-MLX-8bit) | 8 | 31.8 GB | 579K | Most downloaded overall |
| [lmstudio-community/MLX-6bit](https://huggingface.co/lmstudio-community/GLM-4.7-Flash-MLX-6bit) | 6 | 24.3 GB | 577K | Second most downloaded |
| [inferencerlabs/MLX-6.5bit](https://huggingface.co/inferencerlabs/GLM-4.7-Flash-MLX-6.5bit) | 6.5 | 24.3 GB | — | Only provider with quality metrics |
| [mlx-community/4bit](https://huggingface.co/mlx-community/GLM-4.7-Flash-4bit) | 4 | 16.9 GB | 14K | Converted with mlx-lm 0.30.5 |
| [mlx-community/8bit](https://huggingface.co/mlx-community/GLM-4.7-Flash-8bit) | 8 | 31.8 GB | 13K | Converted with mlx-lm 0.30.5 |
| [lmstudio-community/MLX-4bit](https://huggingface.co/lmstudio-community/GLM-4.7-Flash-MLX-4bit) | 4 | 16.9 GB | 13K | By LM Studio team |

## Recommendations

### For This Crate (Rust MLX on Apple Silicon)

**Target the 6-bit MLX format** as the primary supported quantization:

1. **Quality**: q6.5 achieves the same perplexity as q8.5 (1.227) at 24.3 GB vs 31.8 GB
2. **Speed**: ~61 tok/s single inference on M3 Ultra — sufficient for interactive use
3. **Memory**: 26 GB fits comfortably in 32 GB unified memory Macs with headroom
4. **Accuracy**: 96.65% token accuracy vs base model — minimal quality loss

For users with 64+ GB Macs, the 8-bit variant provides 97.80% accuracy at the cost of 31.8 GB memory.

For 16 GB Macs, the 4-bit variant (16.9 GB) is the only option but comes with meaningful quality degradation (90.30% accuracy, 2.82% effective divergence).

### Memory Budget Guide

| Mac Config | Recommended Quant | Size | Expected Quality |
|------------|-------------------|------|------------------|
| 16 GB | 4-bit MLX | 16.9 GB | 90.3% accuracy (tight fit, may swap) |
| 32 GB | 6-bit MLX | 24.3 GB | 96.7% accuracy (recommended) |
| 64 GB | 8-bit MLX | 31.8 GB | 97.8% accuracy |
| 96+ GB | BF16 | 59.9 GB | 100% baseline |

### Download

```bash
# Recommended: 6-bit (best quality-to-size ratio)
huggingface-cli download lmstudio-community/GLM-4.7-Flash-MLX-6bit \
    --local-dir ./models/GLM-4.7-Flash-MLX-6bit

# Alternative: 6.5-bit from inferencerlabs (with quality metrics)
huggingface-cli download inferencerlabs/GLM-4.7-Flash-MLX-6.5bit \
    --local-dir ./models/GLM-4.7-Flash-MLX-6.5bit

# For 16 GB Macs
huggingface-cli download mlx-community/GLM-4.7-Flash-4bit \
    --local-dir ./models/GLM-4.7-Flash-4bit

# For 64+ GB Macs
huggingface-cli download lmstudio-community/GLM-4.7-Flash-MLX-8bit \
    --local-dir ./models/GLM-4.7-Flash-MLX-8bit
```

## Architecture

GLM-4.7-Flash uses a DeepSeek2-style MoE architecture:

```
GLM-4.7-Flash (30B-A3B MoE)
├── Embedding (vocab_size=151552)
├── N x MoEDecoderLayer
│   ├── input_layernorm
│   ├── Attention
│   │   ├── q_a_proj -> q_a_layernorm -> q_b_proj
│   │   ├── kv_a_proj_with_mqa -> kv_a_layernorm -> kv_b_proj
│   │   └── o_proj
│   ├── post_attention_layernorm
│   └── MoEBlock
│       ├── gate (router)
│       ├── shared_experts (always active)
│       └── routed_experts (top-k selected, ~3B active)
└── final_layernorm
```

### Compared to GLM-4.5-MoE (glm4-moe-mlx)

| Property | GLM-4.5-MoE | GLM-4.7-Flash |
|----------|-------------|---------------|
| Total params | ~400B | 31B |
| Active params | ~40B | ~3B |
| Experts | 45 (2 shared + 43 routed) | TBD |
| Architecture | DeepSeek2 | DeepSeek2 |
| Quantization target | 3-bit | 6-bit |
| Memory (quantized) | ~20 GB | ~24 GB |

## Framework Support

| Framework | Status | Notes |
|-----------|--------|-------|
| Transformers | Supported | Requires latest from git |
| vLLM | Supported | Nightly pre-release; MTP speculative decoding |
| SGLang | Supported | Specific dev version; EAGLE speculative decoding |
| llama.cpp (GGUF) | Supported | Via community quants |
| mlx-lm (Python) | Supported | Via mlx-community / lmstudio-community |
| **mlx-rs (this crate)** | **Planned** | Rust native MLX inference |

## GGUF Support in MLX Rust — Technical Analysis

### Current State

The codebase currently loads all models from **SafeTensors** format only. The loading pipeline is:

```
config.json → model.safetensors.index.json → *.safetensors files
                                                    ↓
                                          mlx_load_safetensors (C FFI)
                                                    ↓
                                          HashMap<String, Array>
                                                    ↓
                                     make_quantized_linear() / make_quantized_embedding()
                                          (weight + scales + biases → QuantizedLinear)
```

Key files in the existing pipeline:
- `mlx-rs/src/ops/io.rs` — SafeTensors load/save API
- `mlx-rs/src/utils/io.rs` — FFI wrapper for `mlx_load_safetensors`
- `mlx-rs/src/nn/quantized.rs` — `QuantizedLinear` and `QuantizedEmbedding`
- `mlx-rs/src/ops/quantization.rs` — `quantize`, `dequantize`, `quantized_matmul`, `gather_qmm`
- `glm4-mlx/src/model.rs:577-602` — `make_quantized_linear()` pattern used by all model crates

### The Gap

**mlx-c v0.4.1** (bundled, wrapping MLX C++ v0.30.1) does **not** expose `load_gguf` in its C API. The C++ MLX library *does* have `mlx::core::load_gguf()` internally, but the C wrapper `io.h` only exposes:
- `mlx_load()` — single array (numpy `.npy` format)
- `mlx_load_safetensors()` — returns `map<string, array>` + metadata

The C++ `load_gguf()` returns a `pair<map<string, array>, map<string, GGUFMetaData>>` which has no C binding.

### Three Approaches

#### Approach A: Pure Rust GGUF Parser (Recommended)

Parse GGUF entirely in Rust, then create MLX arrays from the extracted data.

```
GGUF file → gguf-rs-lib (Rust parser) → raw tensor bytes + metadata
                                              ↓
                                    Dequantize GGUF blocks in Rust
                                    (Q4_0/Q4_1/Q8_0 → weights + scales + biases)
                                              ↓
                                    Create mlx_rs::Array from buffers
                                              ↓
                                    Construct QuantizedLinear as usual
```

**Pros:**
- No C/C++ FFI changes needed
- Works with current mlx-rs and mlx-sys as-is
- Pure safe Rust (gguf-rs-lib is safe by default)
- Memory-mapped file access via mmap
- Full control over quantization format conversion

**Cons:**
- Must implement GGUF block dequantization ourselves
- Only Q4_0, Q4_1, Q8_0 map directly to MLX's affine quantization
- K-quants (Q4_K, Q5_K, Q6_K) and I-quants must be dequantized to float16

**Rust crate:** `gguf-rs-lib` v0.2.5 (MIT, ~8.2K SLoC, mmap support, async support)

#### Approach B: Add C FFI Bindings for `load_gguf`

Extend mlx-c with a new `mlx_load_gguf()` function, then expose through mlx-sys.

```
GGUF file → mlx_load_gguf() (new C FFI) → mlx::core::load_gguf() (C++)
                                                ↓
                                    map<string, array> + map<string, GGUFMetaData>
                                                ↓
                                    mlx-sys → mlx-rs → HashMap<String, Array>
```

**Pros:**
- Leverages MLX's existing GGUF parser (uses antirez/gguflib)
- MLX already handles Q4_0, Q4_1, Q8_0 decomposition into (weights, scales, biases)
- Automatically casts unsupported quant types to float16

**Cons:**
- Requires modifying mlx-c (fork or upstream PR)
- Need new C types for `GGUFMetaData` (variant of array/string/vector)
- Harder to maintain — tied to mlx-c release cycle
- Still only supports Q4_0, Q4_1, Q8_0 natively; others → float16

#### Approach C: Hybrid — Rust Parser + MLX Compute

Parse GGUF header/metadata in Rust, but use MLX operations for the heavy dequantization.

```
GGUF file → Rust parser (header + tensor offsets)
                    ↓
         mmap tensor data → mlx_rs::Array::from_raw_bytes()
                    ↓
         For Q4_0/Q4_1/Q8_0: decompose blocks into (weight, scale, bias) in Rust
         For K-quants: load raw → dequantize on GPU via custom Metal kernel
                    ↓
         QuantizedLinear / float16 layers
```

**Pros:**
- Best performance for K-quant dequantization (GPU-accelerated)
- Rust handles parsing, MLX handles compute
- Could support more quant types than Approach B

**Cons:**
- Most complex implementation
- Custom Metal kernels for K-quant unpacking
- Premature optimization unless K-quant support is critical

### GGUF Quantization Format Mapping

| GGUF Type | Block Size | Bytes/Block | MLX Treatment | Config |
|-----------|-----------|-------------|---------------|--------|
| **Q4_0** | 32 | 18 | Native → (weights, scales, biases) | `{group_size: 32, bits: 4}` |
| **Q4_1** | 32 | 20 | Native → (weights, scales, biases) | `{group_size: 32, bits: 4}` |
| **Q8_0** | 32 | 34 | Native → (weights, scales, biases) | `{group_size: 32, bits: 8}` |
| Q4_K | 256 | 144 | Cast to float16 | N/A |
| Q5_K | 256 | 176 | Cast to float16 | N/A |
| Q6_K | 256 | 210 | Cast to float16 | N/A |
| Q2_K | 256 | 84 | Cast to float16 | N/A |
| Q3_K | 256 | 110 | Cast to float16 | N/A |
| F16 | — | — | Direct pass-through | No quantization |
| F32 | — | — | Direct pass-through | No quantization |
| BF16 | — | — | Direct pass-through | No quantization |

### Block Dequantization (What Rust Code Must Do)

**Q4_0** (18 bytes → 32 weights, symmetric):
```
struct block_q4_0 {
    d: f16,           // 2 bytes: scale
    qs: [u8; 16],     // 16 bytes: 32 x 4-bit packed pairs
}
// Dequant: weight_i = d * (nibble_i - 8)
// MLX decomposition: scale = d, bias = -8 * d, weights = nibble values
```

**Q4_1** (20 bytes → 32 weights, asymmetric):
```
struct block_q4_1 {
    d: f16,           // 2 bytes: scale
    m: f16,           // 2 bytes: minimum (bias)
    qs: [u8; 16],     // 16 bytes: 32 x 4-bit packed pairs
}
// Dequant: weight_i = d * nibble_i + m
// MLX decomposition: scale = d, bias = m, weights = nibble values
```

**Q8_0** (34 bytes → 32 weights, symmetric):
```
struct block_q8_0 {
    d: f16,           // 2 bytes: scale
    qs: [i8; 32],     // 32 bytes: 32 x 8-bit signed values
}
// Dequant: weight_i = d * q_i
// MLX decomposition: scale = d, bias = -128 * d, weights = q_i ^ 0x80
```

### GGUF Weight Name Translation

GGUF uses different naming conventions than HuggingFace/MLX SafeTensors:

```
GGUF Name                          → MLX/SafeTensors Name
─────────────────────────────────────────────────────────
token_embd.weight                  → model.embed_tokens.weight
blk.{i}.attn_q.weight             → model.layers.{i}.self_attn.q_proj.weight
blk.{i}.attn_k.weight             → model.layers.{i}.self_attn.k_proj.weight
blk.{i}.attn_v.weight             → model.layers.{i}.self_attn.v_proj.weight
blk.{i}.attn_output.weight        → model.layers.{i}.self_attn.o_proj.weight
blk.{i}.ffn_gate.weight           → model.layers.{i}.mlp.gate_proj.weight
blk.{i}.ffn_up.weight             → model.layers.{i}.mlp.up_proj.weight
blk.{i}.ffn_down.weight           → model.layers.{i}.mlp.down_proj.weight
blk.{i}.attn_norm.weight          → model.layers.{i}.input_layernorm.weight
blk.{i}.ffn_norm.weight           → model.layers.{i}.post_attention_layernorm.weight
output_norm.weight                 → model.norm.weight
output.weight                     → lm_head.weight
```

Note: GLM-4.7-Flash (DeepSeek2 architecture) will have additional MoE-specific weight names for the router gate, shared experts, and routed experts that need mapping.

### Implementation Plan (Approach A)

1. **Add `gguf-rs-lib` dependency** to the crate's `Cargo.toml`
2. **Implement GGUF loader module** (`src/gguf.rs`):
   - Open GGUF file with mmap via `GGUFFileReader`
   - Extract metadata (architecture, vocab size, layer count, etc.)
   - Extract tokenizer from GGUF metadata (GGUF embeds tokenizer data)
3. **Implement block dequantization** (`src/gguf_quants.rs`):
   - `fn dequant_q4_0(block: &[u8]) -> (Array, Array, Array)` — weights, scales, biases
   - `fn dequant_q4_1(block: &[u8]) -> (Array, Array, Array)`
   - `fn dequant_q8_0(block: &[u8]) -> (Array, Array, Array)`
   - For K-quants: `fn dequant_to_f16(block: &[u8], qtype: GGUFType) -> Array`
4. **Implement weight name translation** (`src/gguf.rs`):
   - Map GGUF tensor names → model struct field paths
   - Handle DeepSeek2/MoE-specific naming for GLM-4.7-Flash
5. **Wire into model loading**:
   - `load_model_gguf(path)` → same `HashMap<String, Array>` as SafeTensors path
   - Reuse existing `make_quantized_linear()` / `make_quantized_embedding()`
   - Detect quantization config from GGUF `general.file_type` metadata

### Key Constraint

MLX's `quantized_matmul` only supports **group_size=32** with **bits=4 or bits=8** for GGUF-sourced weights (vs group_size=64 for MLX-native SafeTensors quants). The GGUF block formats use groups of 32 elements. This means:
- Q4_0/Q4_1 GGUF → `QuantizedLinear { group_size: 32, bits: 4 }`
- Q8_0 GGUF → `QuantizedLinear { group_size: 32, bits: 8 }`
- K-quants → must be fully dequantized to float16, no `QuantizedLinear` benefit

### Practical Impact

For GLM-4.7-Flash specifically:
- **Q4_K_M** (most popular GGUF quant, 18.5 GB) — K-quant, would need float16 dequantization, losing the memory benefit. Not worthwhile for MLX Rust.
- **Q4_0** (17.4 GB) — Native MLX support, keeps quantized inference. Good option.
- **Q8_0** (31.8 GB) — Native MLX support, keeps quantized inference. Same size as MLX 8-bit.
- **Q4_1** (19.0 GB) — Native MLX support, keeps quantized inference.

**Conclusion**: For this crate, GGUF loading is most valuable for **Q4_0 and Q8_0** files (which maintain quantized inference speed). For K-quants, users are better served by the MLX SafeTensors format which was purpose-built for MLX's quantization scheme. The primary use case is interoperability — letting users who already have GGUF files from llama.cpp use them directly without conversion.

## Inference Parameters

| Use Case | Temperature | Top-p | Notes |
|----------|------------|-------|-------|
| General | 1.0 | 0.95 | Default recommended |
| Tool calling | 0.7 | 1.0 | More deterministic |
| llama.cpp | — | — | Use `--min-p 0.01`, disable repeat penalty |

## License

MIT OR Apache-2.0
