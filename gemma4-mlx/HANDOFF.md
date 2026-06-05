# Gemma 4 12B (text) on MLX — Handoff / Current State

**Branch:** `feat/gemma4-12b` (in repo `OminiX-MLX`). ~35 commits ahead of `origin/main`. Working tree clean.
**Date:** 2026-06-05.
**Status:** **M0 ✅ → M1 ✅ → M2 ✅. The model generates coherent text end-to-end. Next priority: M3 = performance (currently functionally correct but ~0 tok/s).**

This crate (`gemma4-mlx`) implements Gemma 4 12B **text** inference in pure Rust on Apple MLX, loading the **4-bit** `mlx-community/gemma-4-12B-it-4bit` checkpoint, targeting a **16 GB M4 Mac mini**. Vision/audio (Gemma 4 is multimodal) are out of scope so far.

---

## 1. Read these first (authoritative, kept up to date)

- **Design spec:** `docs/superpowers/specs/2026-06-04-gemma4-12b-text-mlx-design.md`
  - §3 architecture facts; §5 component design; §6 weight loading; **§8 / §8.1 / §8.2 = the resolved M0 / M1 / M2 results and all risk resolutions (R1–R8).** Read §8.x — it captures every non-obvious decision and the numbers.
- **Plans (executed):** `docs/superpowers/plans/2026-06-04-gemma4-12b-{m0-spike,m1-model,m2-generate}.md` — task-by-task implementation plans. M3 has no plan yet (see §6 below for the starting points).

## 2. Environment (already set up on this machine)

- Weights: `/Users/alan0x/models/gemma-4-12B-it-4bit` (~10 GB, 3 shards, 48 layers). Mixed precision: attn q/k/o = 4-bit, mlp gate/up/down = 8-bit, group_size 64.
- Python ref: `mlx-vlm 0.6.1` + `mlx 0.31.1` + `transformers 5.3.0` in `~/miniconda3` (`python3`). Used ONLY to produce golden references.
- **MLX is not thread-safe** → always run Rust tests with `--test-threads=1` (matches repo CI). Always build/run examples `--release` (debug MLX is unusably slow).
- **16 GB memory gotcha:** loading the full mlx-vlm multimodal model OOMs. Use `mlx_vlm.load(model_dir, lazy=True)` so the unused vision/audio towers stay mmapped on disk. Our Rust loader already only retains text weights, so the Rust side fits.

## 3. What works + how it's validated

| Milestone | What | Proof |
|---|---|---|
| M0 | Single decoder layer (attn+mlp+block) | layer 0 & 5 forward vs mlx-vlm: **max-abs-diff ~1e-6** |
| M1 | Full 48-layer model → logits | next-token **argmax = 2818, matches mlx-vlm** |
| M2 | Generate (KV cache) + tokenizer + chat | coherent reply; **KV-cache decode == no-cache re-forward token-for-token (8 tokens)** |

Run them (each loads ~10 GB, minutes):
```bash
cargo test -p gemma4-mlx --release -- --test-threads=1                       # 8 unit tests (config/norm/rope/mask)
M=/Users/alan0x/models/gemma-4-12B-it-4bit
cargo run -p gemma4-mlx --example inspect_gemma4 --release -- $M              # weight/quant layout (R3/R6)
# Golden refs (regenerate; the /tmp/* dirs are ephemeral):
python3 scripts/dump_gemma4_layer_io.py $M /tmp/gemma4_golden
python3 scripts/dump_gemma4_logits.py  $M /tmp/gemma4_logits
python3 scripts/dump_gemma4_greedy.py  $M /tmp/gemma4_greedy 8
cargo run -p gemma4-mlx --example layer_parity       --release -- $M /tmp/gemma4_golden   # ~1e-6
cargo run -p gemma4-mlx --example model_parity       --release -- $M /tmp/gemma4_logits   # argmax 2818
cargo run -p gemma4-mlx --example decode_consistency --release -- $M 8                    # cache==no-cache PASS
cargo run -p gemma4-mlx --example chat_gemma4        --release -- $M "Explain what MLX is in one sentence."
```

## 4. Source map (`gemma4-mlx/src/`)

- `config.rs` — `ModelArgs` / `QuantConfig` (per-module bits via `quant_for`, dual `quantization`/`quantization_config` keys), `LayerKind`, `RopeSpec`.
- `weights.rs` — `load_all_weights` / `weight_keys` / `get_weight` / `make_quantized_linear` / `make_quantized_embedding`. Key prefix `language_model.model.*`.
- `norm.rs` — `GemmaRmsNorm` (standard gamma `weight*norm(x)`; `with_scale=false` for v_norm). NOT Gemma2/3's `(1+w)`.
- `rope.rs` — `proportional_inv_freq` (documents the formula; **sign convention note inside — do not feed to fast::rope directly**). Real RoPE lives in `attention.rs`.
- `attention.rs` — `Attention` (LayerKind-driven). Internal `Rope` enum: Standard (sliding, base 1e4) / Proportional (global, base 1e6, rotate-half, denominator = full head_dim=512, `rotated_dims=128`). `attend(x, mask, Option<&mut KVCache>)` is the core; `forward(x,&mask)` = prefill wrapper. `scale=1.0`. Global layers: `use_k_eq_v` → no v_proj, value = raw k_proj → v_norm, NO rope on v.
- `mask.rs` — `full_causal_mask` / `sliding_window_mask` (bool, true=visible; via `mlx_rs_core::utils::create_causal_mask`).
- `mlp.rs` — GeGLU (`down(gelu_approx(gate(x))*up(x))`, 8-bit).
- `block.rs` — `TransformerBlock`: 4 norms + 2 residuals + trailing `layer_scalar`. `forward` (prefill) / `forward_cached` (decode).
- `model.rs` — `Gemma4TextModel` + `load_model`; `forward` (no cache) / `forward_cached` (+ `new_caches`, `decode_mask`). Embed ×√3840, tied lm_head via `embed_tokens.as_linear`, final softcap `30*tanh(x/30)`, `last_only` (prefill projects only last position — vocab=262144).
- `generate.rs` — `generate_greedy(model, prompt, max_new, eos)` (prefill + KV-cache decode).
- `tokenizer.rs` — `load_tokenizer`, `encode_chat`, `eos_ids`. **Gemma 4 chat markers: `<|turn>`(105)/`<turn|>`(106)** (NOT `<start_of_turn>`); BOS(2) prepended manually; thinking-suppression suffix `<|channel>thought\n<channel|>`; eos=[1,106].

## 5. Non-obvious gotchas (will bite if forgotten)

1. **bf16 divergence vs mlx-vlm:** full-model logits differ from mlx-vlm by ~0.8 max (mean ~0.12) — benign bf16 accumulation across 48 layers (same 4-bit weights both sides, so quant error cancels). argmax/top-3 match. Greedy can diverge from mlx-vlm at *close-call* tokens — this is NOT a bug (proven by cache==no-cache internal consistency). Don't "fix" it by chasing exact-match unless you also match dtype.
2. **Proportional RoPE** denominator is the **full head_dim (512)**, not the rotary subset (128). `freqs = base^(+exp)` because `mlx::fast::rope` reciprocates internally. Rotate-half on the first 64 of each head-half. Easy to get wrong; see attention.rs.
3. **v_norm / k_eq_v ordering** (global layers): `v = v_norm(raw k_proj output)`, k = `k_norm(raw)` then rope; value does NOT get rope.
4. **Per-module quantization:** always resolve bits/group via `quant_for(prefix)`. A single global bits value will corrupt the mlp (8-bit) layers.
5. **Tests:** `--test-threads=1` or Metal SIGABRTs (false failures).

## 6. M3 — next work (priority: PERFORMANCE)

**The blocker for real use: ~0 tok/s.** First chat run: 27 tokens in 2685 s (≈99 s/token). Largely Metal shader compilation; suspected per-step recompiles because each decode step changes the KV-cache (and thus SDPA) tensor shape. Concrete starting points:
- **Profile where time goes** (shader compile vs compute). Check whether warm runs (Metal cache on disk) are dramatically faster, and whether decode shapes recompile each step.
- **RotatingKVCache** for sliding layers (cap KV at window=1024) — bounds memory AND can stabilize decode shapes. `mlx-rs-core::cache` only has growing `KVCache` today; `create_causal_mask` already supports `max_size`/window. (Spec §5.8 / M3 note.)
- **argmax via `mlx_rs::ops::argmax`** instead of copying 262144 f32 to host each step (current `generate.rs`/examples copy-to-vec).
- Possibly fixed-shape decode / pre-compiled graphs.
Other M3/back-burner: sampling (top-p/temperature), streaming, attention.rs `TODO(M1)` is really for M2-done decode offset (already wired); OminiX-API engine/handler integration; vision/audio.

## 7. How to extend safely

- Every change must keep: `cargo test ... --test-threads=1` green (8), `model_parity` argmax 2818, `decode_consistency` PASS, `layer_parity` ~1e-6. These four are the regression guardrails — run them after touching attention/block/model.
- Golden refs live in `/tmp/gemma4_*` (ephemeral) — regenerate with the `scripts/dump_gemma4_*.py` (use `lazy=True`, already in the scripts).
