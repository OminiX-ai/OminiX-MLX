# Gemma 4 12B (文本) M2 — Generate + 对话 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 让 M1 的完整模型能真正"输入一段文字 → 流式生成回答":加 KV cache + offset 解码、`Generate` 迭代器、tokenizer + chat template,产出 `chat_gemma4.rs`;并用 mlx-vlm 贪婪解码逐 token 对齐验证正确。

**Architecture:** 给 `Attention` 加可选 KV cache + RoPE offset(prefill offset=0;decode 用 `cache.offset()`),沿用 mlx-rs-core 的 `KVCache`/`KeyValueCache`(与 qwen3-mlx 同款)。`model.forward_cached(tokens, &mut [KVCache])` 串联每层 cache 与按层 decode mask(full→无需 mask;sliding→窗口 mask,M2 用普通增长 cache + mask,RotatingKVCache 留 M3)。自有 `Generate`(Prefill/Decode 状态机,镜像 `qwen3-mlx/src/model.rs:744`)。tokenizer + chat template 走 `tokenizers` crate + 仓库 `chat_template.jinja`/`tokenizer_config.json`。

**Tech Stack:** Rust + mlx-rs;参考 mlx-vlm(贪婪解码逐 token 对齐)。

**前置:** `feat/gemma4-12b`,M1 完成(全模型 argmax 对齐 2818,8 单测绿)。权重 `/Users/alan0x/models/gemma-4-12B-it-4bit`。`--release --test-threads=1`。mlx-vlm 用 `load(lazy=True)`(16GB)。

**已确立、M2 复用:**
- `mlx_rs_core::cache::{KVCache, KeyValueCache}`:`offset()`、`update_and_fetch(keys,values)->(keys,values)`、`max_size()`。
- 参考(qwen3 + mlx-vlm)decode 顺序:reshape→norm→transpose→`rope(offset=cache.offset())`→`cache.update_and_fetch`→SDPA。值同 M0:global 层 v=raw-k 过 v_norm、不过 rope。
- 我方 `Attention`/`block`/`model` 的 M1 签名:`Attention::forward(&mut,x,&mask)`、`block.forward(&mut,x,&mask)`、`model.forward(&mut,tokens,last_only)`。**M2 新增 cache 路径,保留旧无 cache 路径让 M0/M1 parity 例子不回归。**
- chat template / tokenizer_config 必用(spec §7),否则输出格式错乱。

---

## 文件结构

| 路径 | 职责 | 状态 |
|---|---|---|
| `gemma4-mlx/src/attention.rs` | 加 cache+offset 解码路径(`attend` 内部统一,`forward` 薄封装) | 修改 |
| `gemma4-mlx/src/block.rs` | 加 `forward_cached(x, mask, &mut KVCache)`;旧 `forward` 保留 | 修改 |
| `gemma4-mlx/src/model.rs` | `forward_cached(tokens, &mut [KVCache], last_only)`;`new_caches()`;按层 decode mask | 修改 |
| `gemma4-mlx/src/generate.rs` | `Generate`(prefill+greedy/temp decode 迭代器)+ `make_caches` | 新建 |
| `gemma4-mlx/src/tokenizer.rs` | 加载 tokenizer + 应用 chat template(读 tokenizer_config / chat_template.jinja) | 新建 |
| `gemma4-mlx/examples/generate_gemma4.rs` | 给定 token ids 贪婪生成(验证用) | 新建 |
| `gemma4-mlx/examples/chat_gemma4.rs` | 文字 prompt → chat template → 生成 → 解码;tok/s | 新建 |
| `scripts/dump_gemma4_greedy.py` | mlx-vlm 贪婪解码 K token 金标准 | 新建 |

---

## Task 1: Attention + block 的 KV cache / offset 解码路径

**Files:** Modify `attention.rs`, `block.rs`. Tests: 现有 8 单测须仍绿;新增不强求单测(行为由 Task 5 端到端对齐验证)。

- [ ] **Step 1: `attention.rs`** —— 把核心逻辑抽到一个接受可选 cache + offset 的方法,`forward` 变薄封装(零回归):
```rust
// 统一核心:offset 用于 rope;cache 存在则 update_and_fetch。
pub fn attend(&mut self, x: &Array, mask: Option<&Array>,
              mut cache: Option<&mut KVCache>) -> Result<Array> {
    let (B, L) = (x.shape()[0], x.shape()[1]);
    let offset = cache.as_ref().map(|c| c.offset()).unwrap_or(0);

    let q = self.q_proj.forward(x)?.reshape(&[B,L,self.n_heads,self.head_dim])?;
    let q = self.q_norm.forward(&q)?;
    let k_raw = self.k_proj.forward(x)?.reshape(&[B,L,self.n_kv_heads,self.head_dim])?;
    let v_in = if self.use_k_eq_v { k_raw.clone() }
               else { self.v_proj.as_mut().ok_or_else(|| Error::Model("v_proj missing".into()))?
                          .forward(x)?.reshape(&[B,L,self.n_kv_heads,self.head_dim])? };
    let k = self.k_norm.forward(&k_raw)?.transpose_axes(&[0,2,1,3])?;
    let k = self.rope.forward_at(&k, offset)?;          // rope at offset (see Step 2)
    let v = self.v_norm.forward(&v_in)?.transpose_axes(&[0,2,1,3])?;  // no rope
    let q = q.transpose_axes(&[0,2,1,3])?;
    let q = self.rope.forward_at(&q, offset)?;

    let (k, v) = match cache.as_mut() {
        Some(c) => c.update_and_fetch(k, v)?,
        None => (k, v),
    };
    let sdpa_mask = mask.map(SdpaMask::Array);
    let out = scaled_dot_product_attention::<KVCache>(q, k, v, None, self.scale, sdpa_mask)?
        .transpose_axes(&[0,2,1,3])?.reshape(&[B,L,-1])?;
    Ok(self.o_proj.forward(&out)?)
}

pub fn forward(&mut self, x: &Array, mask: &Array) -> Result<Array> {
    self.attend(x, Some(mask), None)        // unchanged prefill behavior (offset 0, no cache)
}
```
NOTE: the M1 `forward` currently inlines rope at offset 0 — refactor so both paths call the same rope-at-offset code. The internal `Rope` enum (Standard via `mlx_rs::fast::rope(..., offset)`, Proportional via the rotate-half + `fast::rope(..., offset=offset, freqs=..)`) ALREADY supports an offset arg in mlx fast::rope — expose a `Rope::forward_at(&self, x, offset: i32)` and have the old offset-0 call use `forward_at(x, 0)`. Verify against the M1 attention.rs proportional-rope code (it used offset; just thread it).

- [ ] **Step 2: `Rope::forward_at(&self, x: &Array, offset: i32)`** in attention.rs (or rope module): Standard layer → `mlx_rs::fast::rope(x, dims, traditional=false, base=Some(theta), scale=1.0, offset, freqs=None)`; Proportional → same rotate-half slicing as M1 but pass `offset` to the inner `fast::rope`. The M1 code already computed this at offset 0 — generalize the 0 to the param.

- [ ] **Step 3: `block.rs`** —— add `forward_cached`, keep `forward`:
```rust
pub fn forward_cached(&mut self, x: &Array, mask: Option<&Array>, cache: &mut KVCache) -> Result<Array> {
    let residual = x;
    let h = self.input_layernorm.forward(x)?;
    let h = self.attn.attend(&h, mask, Some(cache))?;
    let h = self.post_attention_layernorm.forward(&h)?;
    let x2 = residual.add(&h)?;
    let h2 = self.pre_feedforward_layernorm.forward(&x2)?;
    let h2 = self.mlp.forward(&h2)?;
    let h2 = self.post_feedforward_layernorm.forward(&h2)?;
    let x3 = x2.add(&h2)?;
    x3.multiply(&self.layer_scalar).map_err(Into::into)
}
```
Keep the existing `forward(&mut,x,&mask)` for the M0/M1 examples (it can delegate: `self.forward_cached` is NOT equivalent because no-cache prefill differs only by cache=None — actually you can make `forward` call an internal that passes cache=None; simplest is to keep the M1 `forward` body as-is and add `forward_cached` separately, sharing `attn.attend`).

- [ ] **Step 4:** `cargo build -p gemma4-mlx --release` and `cargo test -p gemma4-mlx --release -- --test-threads=1` → 8 pass (NO regression). Re-run `model_parity` (M1) → argmax still 2818 (the no-cache prefill path must be unchanged).
- [ ] **Step 5: Commit** `git commit -m "feat(gemma4): KV-cache + RoPE-offset decode path in attention/block (prefill path unchanged)"`

---

## Task 2: `model.forward_cached` + 按层 decode mask + cache 构造

**Files:** Modify `model.rs`.

- [ ] **Step 1:** add cache constructor + cached forward:
```rust
use mlx_rs_core::cache::KVCache;

impl Gemma4TextModel {
    pub fn new_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    /// tokens [B, L]. caches: one KVCache per layer (offset advances internally).
    /// Used for BOTH prefill (L>1, caches fresh) and decode (L==1, caches warm).
    pub fn forward_cached(&mut self, tokens: &Array, caches: &mut [KVCache], last_only: bool) -> Result<Array> {
        let L = tokens.shape()[1];
        let mut h = self.embed_tokens.forward(tokens)?;
        h = h.multiply(&Array::from_f32(self.embed_scale))?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let off = caches[i].offset();
            let mask = decode_mask(self.layer_types[i], L, off, self.sliding_window)?; // Option<Array>
            h = layer.forward_cached(&h, mask.as_ref(), &mut caches[i])?;
        }
        h = self.norm.forward(&h)?;
        if last_only { h = h.index((.., (L-1)..(L as i32), ..)); }
        let mut logits = self.embed_tokens.as_linear(&h)?;
        let cap = self.final_logit_softcapping;
        if cap > 0.0 { logits = mlx_rs::ops::tanh(&logits.divide(&Array::from_f32(cap))?)?.multiply(&Array::from_f32(cap))?; }
        Ok(logits)
    }
}

/// Mask for one layer given query length L at cache offset `off`.
/// Prefill (L>1, off==0): full→causal [L,L]; sliding→windowed causal.
/// Decode (L==1): full→None (query sees all cached keys); sliding→window mask over [1, off+1].
fn decode_mask(kind: LayerKind, L: i32, off: i32, window: i32) -> Result<Option<Array>> {
    if L > 1 {
        Ok(Some(match kind {
            LayerKind::Global  => full_causal_mask(L, off)?,
            LayerKind::Sliding => sliding_window_mask(L, off, window)?,
        }))
    } else {
        match kind {
            LayerKind::Global  => Ok(None),  // single query attends to all cached keys
            LayerKind::Sliding => Ok(Some(sliding_window_mask(1, off, window)?)),
        }
    }
}
```
VERIFY: `full_causal_mask`/`sliding_window_mask` shapes are [L, off+L] (they call `create_causal_mask(n, offset, window, _)`); confirm they produce the right [query, key] mask for off>0 decode (key dim = off+L). If `create_causal_mask`'s offset semantics differ, adapt (cross-check mlx-rs-core/src/utils.rs and qwen3's mask use).

- [ ] **Step 2:** build + `cargo test` 8 pass. Quick throwaway: prefill the 6-token seq via `forward_cached(tokens, &mut caches, true)` and confirm argmax==2818 (== M1, proves cached prefill path matches no-cache path). Delete throwaway. Report argmax.
- [ ] **Step 3: Commit** `git commit -m "feat(gemma4): model.forward_cached with per-layer KV cache + decode masks"`

---

## Task 3: `Generate` 迭代器(prefill + greedy/temp decode)

**Files:** Create `gemma4-mlx/src/generate.rs`; add `pub mod generate;` + re-export in `lib.rs`.

- [ ] **Step 1:** implement a simple generator (greedy first; temp optional):
```rust
//! Token generation: prefill prompt, then autoregressive decode with per-layer KV cache.
use mlx_rs::{Array, ops::indexing::IndexOp};
use mlx_rs_core::cache::KVCache;
use crate::model::Gemma4TextModel;
use crate::error::Result;

/// Greedy-decode up to `max_new` tokens after `prompt` (1-D i32 token ids).
/// Returns the generated token ids (excluding the prompt). `eos` stops early.
pub fn generate_greedy(model: &mut Gemma4TextModel, prompt: &[i32], max_new: usize, eos: &[i32]) -> Result<Vec<i32>> {
    let mut caches = model.new_caches();
    // Prefill
    let ptoks = Array::from_slice(prompt, &[1, prompt.len() as i32]);
    let mut logits = model.forward_cached(&ptoks, &mut caches, true)?; // [1,1,vocab]
    let mut out = Vec::new();
    for _ in 0..max_new {
        let next = argmax_last(&logits)?;                  // i32 token id
        if eos.contains(&next) { break; }
        out.push(next);
        let t = Array::from_slice(&[next], &[1, 1]);
        logits = model.forward_cached(&t, &mut caches, true)?;
    }
    Ok(out)
}

fn argmax_last(logits: &Array) -> Result<i32> {
    // logits [1,1,vocab] → argmax over last dim. Use mlx argmax or extract slice.
    let v: Vec<f32> = logits.as_type::<f32>()?.eval().and_then(|_| Ok(logits.as_type::<f32>()?))?.as_slice::<f32>().to_vec();
    let (mut bi, mut bv) = (0i32, f32::MIN);
    for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; bi = i as i32; } }
    Ok(bi)
}
```
(Adapt `as_type`/`eval`/`as_slice` to the exact mlx-rs API used in `model_parity.rs`/`layer_parity.rs`. Prefer `mlx_rs::ops::argmax` if available to avoid copying 262144 floats each step — check API; copying is acceptable for first cut.)

- [ ] **Step 2:** build + 8 tests pass. Commit `git commit -m "feat(gemma4): greedy Generate (prefill + KV-cache decode)"`

---

## Task 4: tokenizer + chat template + `chat_gemma4` + `generate_gemma4` examples

**Files:** Create `gemma4-mlx/src/tokenizer.rs`, `examples/generate_gemma4.rs`, `examples/chat_gemma4.rs`.

- [ ] **Step 1: `tokenizer.rs`** —— load `tokenizer.json` via `tokenizers` crate; apply Gemma chat format. Gemma turn format: `<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n`. Read special-token ids / eos from `tokenizer_config.json` (`eos_token`, plus `<end_of_turn>`). Provide:
```rust
pub fn load_tokenizer(model_dir: &Path) -> Result<tokenizers::Tokenizer>;   // mlx_rs_core::load_tokenizer may already do this — reuse if so
pub fn encode_chat(tok: &tokenizers::Tokenizer, user_msg: &str) -> Result<Vec<i32>>; // applies the turn template, returns ids (with BOS)
pub fn eos_ids(tok: &tokenizers::Tokenizer) -> Vec<i32>;                     // [<eos>=1, <end_of_turn>=106]
```
(Confirm the exact turn tokens against the repo `chat_template.jinja` and `tokenizer_config.json`; Gemma uses BOS=2, `<start_of_turn>`/`<end_of_turn>`, eos ids [1,106] per config.json `eos_token_id:[1,106]`.)

- [ ] **Step 2: `examples/generate_gemma4.rs`** (token-level, for parity): args MODEL_DIR + space-separated prompt ids + max_new; load_model, generate_greedy, print generated ids.

- [ ] **Step 3: `examples/chat_gemma4.rs`** (the payoff): args MODEL_DIR + prompt string; load tokenizer + model; `encode_chat` → `generate_greedy(max_new=~64, eos=eos_ids)` → `tok.decode(ids)`; print the reply + tok/s (time the decode loop). Use a real prompt e.g. "Explain what MLX is in one sentence."

- [ ] **Step 4: build + RUN chat:** `cargo run -p gemma4-mlx --example chat_gemma4 --release -- /Users/alan0x/models/gemma-4-12B-it-4bit "Explain what MLX is in one sentence."` → prints a COHERENT English reply + tok/s + peak behavior. Record the output.
- [ ] **Step 5: Commit** `git commit -m "feat(gemma4): tokenizer + chat template + chat/generate examples (end-to-end text generation)"`

---

## Task 5: 贪婪解码逐 token 对齐 mlx-vlm(正确性收口)

**Files:** Create `scripts/dump_gemma4_greedy.py`.

- [ ] **Step 1: `scripts/dump_gemma4_greedy.py`** —— mlx-vlm `load(lazy=True)`;固定 prompt ids `[2,1024,2048,4096,8192,16384]`;贪婪解码 K=8 个 token(每步对 growing 序列调 `lm.model` → `logits_from_hidden(last)` → argmax → append;cache 可选,K 小直接重算 prefill 即可)。打印并保存 `greedy_ids.json` = 生成的 8 个 id。
- [ ] **Step 2: 运行** `python scripts/dump_gemma4_greedy.py /Users/alan0x/models/gemma-4-12B-it-4bit /tmp/gemma4_greedy` → 记录参考的 8 个 id(第 1 个应是 2818,与 M1 一致)。
- [ ] **Step 3:** `cargo run -p gemma4-mlx --example generate_gemma4 --release -- /Users/alan0x/models/gemma-4-12B-it-4bit "2 1024 2048 4096 8192 16384" 8` → 我方生成 8 id。
- [ ] **Step 4: 对齐** —— 我方 8 id 与参考 8 id **逐 token 一致**(贪婪下应完全相同;若第 k 个起分叉,查 decode mask/offset/rope-offset)。把结果记录到 commit message。
- [ ] **Step 5: Commit** `git commit -m "feat(gemma4): greedy decode token-for-token parity vs mlx-vlm"`

---

## M2 完成判据(DoD)
- [ ] `cargo test -p gemma4-mlx --release -- --test-threads=1` 全绿;M1 `model_parity` argmax 仍 2818(无回归)。
- [ ] `chat_gemma4` 对真实文字 prompt 输出**连贯英文回答**,打印 tok/s。
- [ ] `generate_gemma4` 贪婪 8 token 与 mlx-vlm 参考**逐 token 一致**(端到端正确性)。
- [ ] 实测 prefill+decode 峰值内存适配 16GB。

## 不在 M2(M3 / 后续)
- RotatingKVCache(长上下文滑窗截断,免增长 cache 内存)、proportional RoPE 提升到 mlx-rs-core、采样策略(top-p/温度调优)、流式 SSE、性能优化(argmax 用 mlx 算子免拷贝)、OminiX-API engine/handler 接线、视觉/音频。
