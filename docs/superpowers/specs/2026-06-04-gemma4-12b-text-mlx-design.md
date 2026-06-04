# Gemma 4 12B (文本) MLX 接入设计

- 日期: 2026-06-04(rev2,纳入代码评审修正)
- 分支: `feat/gemma4-12b`
- 范围: **纯文本先行**(视觉/音频多模态留待后续阶段)
- 目标 crate: 新建 `gemma4-mlx`(参照 `qwen3-mlx` 结构,但**不复用其量化加载器**,见 §6)

## 1. 目标与非目标

**目标:** 用纯 Rust + mlx-rs 实现 Gemma 4 12B 文本解码器,加载 4-bit/混合精度量化权重,在 16GB M4 Mac mini 上跑通自回归对话,通过自有 `Generate` 迭代器对外提供生成能力。输出需与参考实现(mlx-lm / HF transformers)逐层数值对齐。

**非目标(本期不做):** 视觉/音频/视频投影器、MoE 分支、RotatingKVCache、OminiX-API 接线、训练。

## 2. 硬约束

- **内存:** 16GB → bf16(23.9GB)不可行。采用 `mlx-community/gemma-4-12B-it-4bit`(混合精度:默认 4bit + MLP 8bit,QAT,~11GB)。
- **上下文上限(首版):** 不实现 RotatingKVCache。11GB 权重 + 每层增长 KV cache 在 16GB 上**只承诺短上下文**(首版目标上限 ~4K tokens,M0 实测后定稿)。长上下文留待 M3 的滑窗 KV 截断。
- **正确性基线:** 以 HF `Gemma4UnifiedForConditionalGeneration` 文本路径 / mlx-lm gemma4 为参照,逐层比对。

## 3. 架构事实(官方 `config.json` 的 `text_config`)

| 项 | 值 |
|---|---|
| num_hidden_layers | 48 |
| hidden_size | 3840 |
| num_attention_heads | 16 |
| num_key_value_heads | 8(滑窗层) |
| head_dim | 256(滑窗层) |
| global_head_dim | 512(全局层) |
| num_global_key_value_heads | 1(全局层 MQA) |
| attention_k_eq_v | true(全局层 K=V,需 M0 定论 v_proj 是否独立存在) |
| intermediate_size | 15360 |
| hidden_activation | gelu_pytorch_tanh(→ GeGLU) |
| vocab_size | 262144 |
| tie_word_embeddings | true(无独立 lm_head) |
| rms_norm_eps | 1e-6 |
| final_logit_softcapping | 30.0(无 attn softcap) |
| sliding_window | 1024 |
| max_position_embeddings | 131072 |

**层模式:** 5×`sliding_attention` + 1×`full_attention`,重复 8 次 = 48 层。

**每层附加权重(经官方 index.json / 评审确认):**
- `self_attn.q_norm.weight`、`self_attn.k_norm.weight`(QK-norm,每层都有)
- `layers.{i}.layer_scalar`(**Gemma 4 特有**的每层标量,作用于残差/层输出,不可省)

**双 RoPE(`rope_parameters`):** 滑窗层 `default` θ=10000 全维(256);全局层 `proportional` θ=1e6 `partial_rotary_factor=0.25`(旋转 64 维)。

**量化布局(官方 4bit 仓库 `config.json` 的 `quantization`):** 默认 `bits=4, group_size=64, mode=affine`;**per-module override**:全部 48 层的 `mlp.{gate,down,up}_proj` 为 `bits=8, group_size=64`。

## 4. Crate 结构与对外接口

镜像 `qwen3-mlx` 模块划分;新增 `norm.rs`/`rope.rs`,并自写量化加载器与 `Generate`。

```
gemma4-mlx/src/
  lib.rs        # 公开 API
  error.rs
  config.rs     # ModelArgs + layer_types + 双 rope + 量化(含 per-module override)
  rope.rs       # 双 RoPE(default / proportional-partial,本地实现)
  norm.rs       # Gemma RMSNorm((1+w),自写,nn::RmsNorm 无 offset)
  attention.rs  # Attention + enum LayerKind { Sliding, Global };含 q/k_norm;v_proj 可选
  mlp.rs        # GeGLU
  model.rs      # TransformerBlock(含 layer_scalar)、Gemma4TextModel、load_model、自有 Generate
```

**对外接口:**
```rust
pub fn load_model(model_dir: &str) -> Result<Gemma4TextModel>;
pub fn get_model_args(model_dir: &str) -> Result<ModelArgs>;
pub use mlx_rs_core::{cache::{KVCache, KeyValueCache}, load_tokenizer};
// Generate:镜像 qwen3-mlx 的自有 Generate<'a, C>(model.rs:744 模式),
// 而非 core builder(后者需实现 mlx-rs-core::ModelInput/ModelOutput,本期不做)。
pub struct Generate<'a, C> { /* Prefill/Decode 状态机,迭代产出 token */ }
```

## 5. 组件设计

### 5.1 config.rs
- 解析 `text_config`:`layer_types: Vec<LayerKind>`、两套 `RopeSpec`。
- **量化配置**:解析默认 `{bits,group_size,mode}` + per-module override map。提供 `quant_for(weight_prefix) -> (bits, group_size)`,加载每个模块时按其前缀取精度(MLP→8bit,其余→4bit)。**不能用单一全局 bits**。

### 5.2 norm.rs — Gemma RMSNorm(自写)
- `out = (1 + weight) * x * rsqrt(mean(x²) + eps)`,eps=1e-6。本地 `nn::RmsNorm` 为 `weight * x * rsqrt(...)` 无 offset,故自写。`(1+w)` 约定需用 checkpoint/官方实现确认(R2)。

### 5.3 嵌入与缩放
- `embed_tokens`:`MaybeQuantized<nn::Embedding>`,键 `language_model.model.embed_tokens.*`。
- 前向 `h = embed(x) * normalizer`,normalizer 默认 √hidden=√3840,M0 对照确认(R5)。
- `tie_word_embeddings=true` → 输出复用嵌入权重(`as_linear`),无 `lm_head` 张量。

### 5.4 attention.rs — 单结构 + enum(方案③A)
```rust
enum LayerKind { Sliding, Global }
struct Attention {
    kind: LayerKind,
    q_proj, k_proj, o_proj: MaybeQuantized<nn::Linear>,
    v_proj: Option<MaybeQuantized<nn::Linear>>, // 缺失则 v = k(k_eq_v),M0 定论
    q_norm, k_norm: GemmaRmsNorm,               // 每层 QK-norm
    rope: Rope,                                 // 按 kind 选
}
```
- 维度按 kind:Sliding(head_dim 256, kv_heads 8);Global(head_dim 512, kv_heads 1 MQA,`attention_k_eq_v`)。
- **v_proj 可选**:加载时按权重键是否存在决定;global 层若无独立 v_proj 则前向用 `v = k`。此处对应评审 #3,M0 用真权重定论(R3)。
- query 缩放默认 `head_dim^-0.5`(滑窗 256 / 全局 512 各取),若参考用 `query_pre_attn_scalar` 则覆盖(R4)。
- 16×256=4096 ≠ hidden 3840 → q/o 维度独立,按 head_dim×heads 设定。
- QK-norm 作用于每个 head 的 head_dim 维(参照 qwen3 q_norm/k_norm 模式)。

### 5.5 rope.rs — 双 RoPE(方案②A,本地)
- 滑窗:`nn::Rope` dims=256 θ=10000 default。
- 全局:`proportional` θ=1e6 partial_rotary_factor=0.25。首选用 `nn::Rope` partial-dims(dims=64)表达;**`mlx-rs-core::initialize_rope` 仅支持 default/linear,会报 unsupported,不可调用**。若 proportional 语义≠plain partial,则在 crate 内手写(R1)。

### 5.6 mlp.rs — GeGLU
- `down(gelu_tanh(gate(x)) * up(x))`,intermediate 15360。三投影 8bit 量化。

### 5.7 model.rs — TransformerBlock(含 layer_scalar)
- 四 norm(Gemma3 风格,以权重键名为准,R2):`input_layernorm → attn → post_attention_layernorm → (+res) → pre_feedforward_layernorm → mlp → post_feedforward_layernorm → (+res)`。
- **`layer_scalar`**:Gemma 4 特有的每层标量,作用于残差/子层输出。精确作用点(乘在哪个分支)需对照参考确认(R7),不可省略。

### 5.8 Mask 构造(方案①A,但按层类型 + 修正 decode)
- 全局层:全因果 mask。
- 滑窗层:因果 + 窗口 1024 带状 mask。
- **关键修正(评审 #5)**:`create_attention_mask` 在 `T==1`(decode)返回 `None`;在不截断 KV 的方案下,滑窗层 decode 步必须**显式构造单步滑窗 mask**(对超出窗口的历史置 -inf),否则会错误看到全历史。prefill 与 decode 各自按层类型生成 mask;不复用单一全局 mask。
- 备选:给滑窗层用带 `max_size=1024` 的 cache,让 `create_attention_mask` 自动加窗(归入 M3)。

### 5.9 KV cache
- 每层一个标准增长 `KVCache`(全局 1 KV head;滑窗 8 KV head)。窗口语义靠 §5.8 的 mask 实现。

### 5.10 输出
- `final_norm` → tied embedding 反投影 → **`final_logit_softcapping`**:`logits = 30 * tanh(logits/30)`,在采样前应用。

## 6. 权重加载(自写,不复用 qwen3 加载器)
- 键前缀 **`language_model.model.`**(评审 #1;qwen3 硬编码 `model.layers` 不可用)。
- 按 §5.1 的 `quant_for(prefix)` 对每个模块取 bits/group_size(评审 #2),分别构造 `MaybeQuantized::Quantized`。
- v_proj 按键存在与否可选加载(§5.4)。
- 集中映射:`...layers.{i}.self_attn.{q,k,[v],o}_proj`、`.q_norm/.k_norm`、`.layer_scalar`、`mlp.{gate,up,down}_proj`、各 layernorm、`embed_tokens`、`model.norm`。

## 7. 验证策略
1. 逐层比对参考实现(RMSNorm(1+w)、embedding 缩放、双 RoPE、QK-norm、global MQA、layer_scalar、softcap),量化容差内。
2. 贪婪解码(temp=0)首 32 token 与参考一致。
3. `examples/chat_gemma4.rs` 跑通对话,记录峰值内存、tok/s、实测可用上下文上限。

## 8. 风险与未决项(M0 spike 定论)
- R1 proportional RoPE 是否等价 nn::Rope partial-dims(§5.5)
- R2 norm 数量/命名(四 vs 二)及 RMSNorm `(1+w)`(§5.2/5.7)
- R3 **global 层 v_proj 是否独立存在**(k_eq_v)——远程 index 检查不可信(层数报错),用真权重定论(§5.4)
- R4 query 缩放是否有 `query_pre_attn_scalar` 覆盖(§5.4)
- R5 embedding normalizer 取值(§5.3)
- R6 4bit/混合精度键名、scales/biases 布局与自写加载器兼容(§6)
- R7 `layer_scalar` 精确作用点(§5.7)

## 9. 里程碑
1. **M0 权重/配置 spike**(评审建议):解析 `config.json` + `model.safetensors.index.json`;分别实例化 layer 0(sliding)与 layer 5(full),逐键确认存在、每模块 bits/group_size 正确(MLP 8bit/其余 4bit)、global 层 v_proj 有无路径可跑;实测加载峰值内存与可用上下文。产出 R1–R7 结论。**不求完整生成**。
2. **M1 解码器**:48 层前向 + 按层 mask + 双 RoPE + QK-norm + layer_scalar + softcap,逐层对齐。
3. **M2 生成**:自有 `Generate`,端到端贪婪解码与参考一致,`chat_gemma4.rs` 跑通。
4. **M3(可选)**:滑窗 RotatingKVCache(长上下文)、proportional RoPE 提升到 mlx-rs-core。

## 10. 后续阶段(本期外)
视觉投影器、音频直通、OminiX-API catalog + engine + handler 接线。
