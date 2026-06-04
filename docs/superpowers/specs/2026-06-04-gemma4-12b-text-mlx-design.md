# Gemma 4 12B (文本) MLX 接入设计

- 日期: 2026-06-04
- 分支: `feat/gemma4-12b`
- 范围: **纯文本先行**(视觉/音频多模态留待后续阶段)
- 目标 crate: 新建 `gemma4-mlx`(参照 `qwen3-mlx` 结构)

## 1. 目标与非目标

**目标:** 在 OminiX-MLX 内用纯 Rust + mlx-rs 实现 Gemma 4 12B 文本解码器,能加载 4-bit 量化权重、在 16GB M4 Mac mini 上跑通自回归对话,并通过 `Generate` 接口对外提供生成能力。输出需与参考实现(mlx-lm / HF transformers)逐层数值对齐。

**非目标(本期不做):**
- 视觉投影器(35M patch 线性 + 坐标嵌入)、音频直通投影、视频
- MoE 分支(`enable_moe_block=false`,本模型用不到)
- RotatingKVCache / 滑窗 KV 截断(后续内存优化)
- OminiX-API 侧接线(单独一期)
- 训练 / 微调

## 2. 硬约束

- **内存:** 16GB 统一内存 → bf16(权重 23.9GB)不可行,必须 4-bit。采用 `mlx-community/gemma-4-12B-it-4bit`(QAT int4,质量对标 bf16,~7–11GB)。
- **正确性基线:** 以 HF `Gemma4UnifiedForConditionalGeneration` 的文本路径 / mlx-lm 的 gemma4 实现为参照,逐层比对中间张量。任何 Gemma 特有细节(norm 约定、embedding 缩放、softcap、双 RoPE)必须精确匹配,否则输出乱码。

## 3. 架构事实(来自官方 `config.json`,`text_config`)

| 项 | 值 |
|---|---|
| num_hidden_layers | 48 |
| hidden_size | 3840 |
| num_attention_heads | 16 |
| num_key_value_heads | 8(滑窗层) |
| head_dim | 256(滑窗层) |
| global_head_dim | 512(全局层) |
| num_global_key_value_heads | 1(全局层,MQA) |
| attention_k_eq_v | true(全局层 K=V,统一 KV) |
| intermediate_size | 15360 |
| hidden_activation | gelu_pytorch_tanh(→ GeGLU) |
| vocab_size | 262144 |
| tie_word_embeddings | true |
| rms_norm_eps | 1e-6 |
| final_logit_softcapping | 30.0 |
| attn_logit_softcapping | 无(比 Gemma2 简化) |
| sliding_window | 1024 |
| max_position_embeddings | 131072 |

**层模式:** `layer_types` = 5×`sliding_attention` + 1×`full_attention`,重复 8 次 = 48 层。

**双 RoPE(`rope_parameters`,按层类型):**
- 滑窗层: `rope_type="default"`,θ=10000,全维旋转(256)
- 全局层: `rope_type="proportional"`,θ=1e6,`partial_rotary_factor=0.25` → 仅旋转 64 维

## 4. Crate 结构与对外接口

镜像 `qwen3-mlx`。模块划分:

```
gemma4-mlx/
  Cargo.toml          # 依赖 mlx-rs, mlx-rs-core(同 qwen3-vl-mlx)
  src/
    lib.rs            # 公开 API 重导出
    error.rs          # thiserror Error/Result(或复用 mlx-rs-core::error)
    config.rs         # ModelArgs + per-layer rope/layer_types 解析
    rope.rs           # 双 RoPE 构造(default / proportional-partial)
    attention.rs      # Attention + enum LayerKind { Sliding, Global }
    mlp.rs            # GeGLU MLP
    norm.rs           # Gemma RMSNorm((1+w) 约定)
    model.rs          # TransformerBlock, Gemma4TextModel, load_model, forward
```

**对外接口(与 qwen3-mlx 一致,保证 OminiX-API 后续接线零摩擦):**

```rust
pub use mlx_rs_core::{cache::{KVCache, KeyValueCache}, load_tokenizer};
pub fn load_model(model_dir: &str) -> Result<Gemma4TextModel>;
pub fn get_model_args(model_dir: &str) -> Result<ModelArgs>;
// 复用 mlx-rs-core::generate::Generate(builder 模式),Model 实现其 trait 约束
```

生成沿用 `Generate::builder()...build()` 迭代器(见 `mlx-rs-core/src/generate/mod.rs`),不新造生成循环。

## 5. 组件设计

### 5.1 config.rs — ModelArgs
- 解析 `text_config`。`layer_types: Vec<LayerKind>`,`rope_parameters` 拆成两套 `RopeSpec`。
- 携带量化配置(`quantization`:group_size/bits),用于加载后对 `MaybeQuantized` 模块调用 `quantize()`。

### 5.2 norm.rs — Gemma RMSNorm
- Gemma 约定:`out = x_normed * (1 + weight)`(权重存的是零中心偏移),**不同于** qwen 的 `x_normed * weight`。eps=1e-6。这是最易踩的坑之一,需对照参考确认。

### 5.3 嵌入与缩放
- `embed_tokens`:`MaybeQuantized<nn::Embedding>`,vocab 262144 × 3840。
- **embedding 缩放:** 前向时 `h = embed(x) * sqrt(hidden_size)`(=√3840)。需对照参考确认 normalizer 取值。
- `tie_word_embeddings=true` → lm_head 复用嵌入权重(`as_linear`),无独立 lm_head 张量。

### 5.4 attention.rs — 单结构 + enum(方案③A)
```rust
enum LayerKind { Sliding, Global }
struct Attention {
    kind: LayerKind,
    q_proj, k_proj, v_proj, o_proj: MaybeQuantized<nn::Linear>,
    rope: Rope,            // 按 kind 选 default / proportional
    // 维度按 kind:
    //   Sliding: head_dim=256, kv_heads=8
    //   Global : head_dim=512, kv_heads=1(MQA), k_eq_v=true
}
```
- query 缩放:默认 `head_dim^-0.5`(滑窗 256、全局 512 各自取),若参考实现用 `query_pre_attn_scalar` 另行覆盖。
- 全局层 `attention_k_eq_v=true`:K 与 V 共享同一投影/张量(统一 KV),加载与前向都按此处理。
- 注意 16 heads × 256 = 4096 ≠ hidden 3840 → q/o 投影维度独立于 hidden,严格按 head_dim×heads 设定,不可用 hidden 推算。
- SDPA 复用 `mlx_rs_core::scaled_dot_product_attention` + 传入 mask。

### 5.5 rope.rs — 双 RoPE(方案②A,本地实现)
- 滑窗层:`nn::Rope`,dims=256,θ=10000,default。
- 全局层:`proportional`,θ=1e6,`partial_rotary_factor=0.25`。
  - **首选**:用 `nn::Rope` 的 partial-dims(dims=64)表达,只旋转前 64 维、其余透传。
  - **风险**:"proportional" 是否等价于"plain partial rotary"需在 spike 验证;若语义不同(如对未旋转维另有缩放),则在 crate 内手写该变体。`mlx-rs-core` 保持不动。

### 5.6 mlp.rs — GeGLU
- `down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))`,intermediate 15360,gelu 用 tanh 近似(`gelu_pytorch_tanh`)。三个投影均 `MaybeQuantized`。

### 5.7 TransformerBlock — 四 norm 布局
- 采用 Gemma3 风格四 norm(**需对照 gemma4 权重键名确认**):
  `input_layernorm` → attn → `post_attention_layernorm` → +residual → `pre_feedforward_layernorm` → mlp → `post_feedforward_layernorm` → +residual。
- 若 gemma4 权重实际只含两 norm,则退回两 norm 布局;以权重键名为准。

### 5.8 Mask 构造(方案①A,mask-only)
- 全局层:全因果 mask(复用 `create_attention_mask`)。
- 滑窗层:因果 + 窗口=1024 的下三角带状 mask(超出窗口置 -inf)。
- prefill 阶段按序列长度生成;decode 阶段按 cache 偏移生成单步 mask。

### 5.9 KV cache(方案①A)
- 每层一个标准增长 `KVCache`(滑窗层不截断,靠 mask 实现窗口语义)。
- 全局层 1 KV head;滑窗层 8 KV head;cache 形状各层自适应。

### 5.10 输出
- 末层后 `final_norm`(RMSNorm)→ 经 tied embedding 反投影得 logits →**应用 `final_logit_softcapping=30.0`**:`logits = 30 * tanh(logits / 30)`。
- softcap 必须在采样前应用,否则分布错误。

## 6. 权重加载
- 从 `mlx-community/gemma-4-12B-it-4bit` 的 safetensors 加载(已是 MLX 4bit 格式)。
- 沿用 qwen3-mlx 模式:模块标 `#[quantizable]`,以 `MaybeQuantized::Original` 构造后,按 config 的 quantization 段对全模型 `quantize()`,再 `update`/load 权重。
- 键名映射在 `load_model` 内集中处理(`model.layers.{i}.self_attn.{q,k,v,o}_proj`、`mlp.{gate,up,down}_proj`、各 norm、`embed_tokens`)。

## 7. 验证策略(正确性优先)
1. **逐层比对**:用 transformers/mlx-lm 跑同一 prompt,dump 每个子模块输出,与 Rust 实现对齐(rtol/atol 量化容差内)。重点:RMSNorm、embedding 缩放、双 RoPE、global 层 MQA/k_eq_v、softcap。
2. **端到端**:贪婪解码(temp=0)与参考逐 token 比对,首 32 token 一致。
3. **example**:`examples/chat_gemma4.rs` 跑通交互对话,记录峰值内存与 tok/s。

## 8. 风险与未决项(进 spike 验证)
- R1 **proportional RoPE 语义**:能否用 `nn::Rope` partial-dims 表达(§5.5)。
- R2 **norm 数量与命名**:四 norm vs 两 norm(§5.7),以权重键名为准。
- R3 **global 层结构**:head_dim 512 + 1 KV head + k_eq_v 的精确加载/前向(§5.4)。
- R4 **query 缩放**:是否存在 `query_pre_attn_scalar` 覆盖默认 `head_dim^-0.5`。
- R5 **embedding normalizer** 取值(√3840 还是其它)。
- R6 **4bit 权重布局**:mlx-community 版的 quantization 参数(group_size/bits)与键名是否与 qwen3 加载路径完全兼容。

## 9. 里程碑
1. **M0 Spike**:加载 4bit 权重 + 单层前向,验证 R1–R6,产出确认结论(不求完整生成)。
2. **M1 解码器**:48 层完整前向 + mask + 双 RoPE + softcap,逐层对齐通过。
3. **M2 生成**:接 `Generate`,端到端贪婪解码与参考一致,`chat_gemma4.rs` 跑通。
4. **M3(可选)**:RotatingKVCache 滑窗优化、提升 p-RoPE 到 mlx-rs-core。

## 10. 后续阶段(本期外)
- 视觉投影器(patch 16/48、num_soft_tokens 280、mm_embed_dim 3840)
- 音频直通(audio_embed_dim 640、samples_per_token 640)
- OminiX-API catalog 条目 + engine + handler 接线
