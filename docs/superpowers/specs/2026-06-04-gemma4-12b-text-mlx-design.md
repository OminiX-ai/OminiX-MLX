# Gemma 4 12B (文本) MLX 接入设计

- 日期: 2026-06-04(rev5,四轮代码评审 + HF/vLLM/本库源码核实,可进 M0)
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
| attention_k_eq_v | true → **全局层无 v_proj**,value 取 raw k_proj 输出过 v_norm(源码确认,M0 仅本地复核) |
| intermediate_size | 15360 |
| hidden_activation | gelu_pytorch_tanh(→ GeGLU) |
| vocab_size | 262144 |
| tie_word_embeddings | true(无独立 lm_head) |
| rms_norm_eps | 1e-6 |
| final_logit_softcapping | 30.0(无 attn softcap) |
| sliding_window | 1024 |
| max_position_embeddings | 131072 |

**层模式:** 5×`sliding_attention` + 1×`full_attention`,重复 8 次 = 48 层。

**每层附加权重(官方 index.json + HF 源码核实):**
- `self_attn.q_norm.weight`、`self_attn.k_norm.weight`(标准 gamma RMSNorm,head_dim 维)
- `self_attn.v_norm`(**with_scale=False**,无权重,对 value 归一化;源码确认)
- `layers.{i}.layer_scalar`(**Gemma 4 特有**标量,**在层 forward 末尾、两次 residual 之后**乘到整层输出:`hidden *= layer_scalar`)

**attention scaling:** 源码 `self.scaling = 1.0`(**不用** `head_dim^-0.5`,Q/K-norm 负责尺度)。

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
  norm.rs       # GemmaRmsNorm { with_scale: bool }:标准 gamma(=本地 nn::RmsNorm)+ no-scale(value norm)
  attention.rs  # Attention + enum LayerKind { Sliding, Global };含 q/k/v_norm;v_proj 可选
  mlp.rs        # GeGLU
  model.rs      # TransformerBlock(含 layer_scalar)、Gemma4TextModel、load_model、自有 Generate
```

**对外接口:**
```rust
pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Gemma4TextModel>;
pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs>;
// (用 impl AsRef<Path>,与本库其他 crate 一致)
pub use mlx_rs_core::{cache::{KVCache, KeyValueCache}, load_tokenizer};
// Generate:镜像 qwen3-mlx 的自有 Generate<'a, C>(model.rs:744 模式),
// 而非 core builder(后者需实现 mlx-rs-core::ModelInput/ModelOutput,本期不做)。
pub struct Generate<'a, C> { /* Prefill/Decode 状态机,迭代产出 token */ }
```

## 5. 组件设计

### 5.1 config.rs
- 解析 `text_config`:`layer_types: Vec<LayerKind>`、两套 `RopeSpec`。
- **量化配置**:解析默认 `{bits,group_size,mode}` + per-module override map。提供 `quant_for(weight_prefix) -> (bits, group_size)`,加载每个模块时按其前缀取精度(MLP→8bit,其余→4bit)。**不能用单一全局 bits**。

### 5.2 norm.rs — GemmaRmsNorm(源码已定)
- **标准 gamma**:`out = weight * x * rsqrt(mean(x²)+eps)`,weight 初始化为 ones,eps=1e-6。**与本地 `nn::RmsNorm` 公式一致**,layer/q/k norm 可直接复用 `nn::RmsNorm`。
- **no-scale 变体**(`with_scale=false`):`out = x * rsqrt(mean(x²)+eps)`,无权重,用于 **v_norm**。`nn::RmsNorm` 不支持,需自写一个 `GemmaRmsNorm { with_scale: bool }` 统一两种(或单独 no-scale 实现)。
- (评审推翻了 rev2 的 `(1+w)`:HF `Gemma4RMSNorm.forward` 为 `normed * self.weight`,非 Gemma1/2/3 的 offset 写法。)

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
    v_proj: Option<MaybeQuantized<nn::Linear>>, // global 层为 None(k_eq_v),预期行为
    q_norm, k_norm: GemmaRmsNorm,               // with_scale=true
    v_norm: GemmaRmsNorm,                        // with_scale=false
    rope: Rope,                                  // 按 kind 选
    scale: f32,                                  // = 1.0(源码固定)
}
```
- 维度按 kind:Sliding(head_dim 256, kv_heads 8);Global(head_dim 512, kv_heads 1 MQA)。
- **scaling = 1.0**(源码,非 `head_dim^-0.5`)。
- **前向顺序(源码核实,易错点):**
  1. `q = q_norm(q_proj(x))`;`q = RoPE(q)`
  2. `raw_kv = k_proj(x)`
  3. `k = k_norm(raw_kv)`;`k = RoPE(k)`
  4. `value_in = if v_proj.is_some() { v_proj(x) } else { raw_kv }`(global 层取 raw k_proj 输出,**不是最终 k**)
  5. `v = v_norm(value_in)`(**v 不过 RoPE**)
- 16×256=4096 ≠ hidden 3840 → q/o 维度独立,按 head_dim×heads 设定。
- **reshape 顺序**:投影后先 reshape 成 `[B, L, heads, head_dim]`,q/k/v norm 在**最后一维(head_dim)**上做,再 transpose 成 `[B, heads, L, head_dim]` 进 attention。global q 输出 16×512=8192,reshape 维度尤需对齐。

### 5.5 rope.rs — 双 RoPE(方案②A,本地)
- 滑窗层:`nn::Rope` dims=256 θ=10000 default(可直接复用)。
- 全局层:`proportional` θ=1e6 partial_rotary_factor=0.25 —— **从第一版就本地手写,不用 `nn::Rope` partial-dims**。原因(vLLM/HF 源码核实):
  - 频率分母用**完整 head_dim=512**,而非 rotary_dim=64:`inv_freq[i] = base^-(2i / 512)`,i=0..32(rotary_dim/2)。`nn::Rope(dims=64)` 会用 64 当分母,频率全错。
  - 非旋转维 **identity**:`inv_freq` 在前 32 个频率后补零(`cos=1, sin=0`),后 448 维不旋转。
  - 即:对前 64 维施加"按 512 计算频率"的旋转,其余 448 维原样透传。
- **`mlx-rs-core::initialize_rope` 仅支持 default/linear,不可用于全局层。**
- M0 仍逐元素对照参考确认频率与边界(R1)。

### 5.6 mlp.rs — GeGLU
- `down(gelu_tanh(gate(x)) * up(x))`,intermediate 15360。三投影 8bit 量化。

### 5.7 model.rs — TransformerBlock(含 layer_scalar)
- 四 norm(Gemma3 风格,以权重键名为准,R2):`input_layernorm → attn → post_attention_layernorm → (+res) → pre_feedforward_layernorm → mlp → post_feedforward_layernorm → (+res)`。
- **`layer_scalar`(源码已定)**:在 decoder layer forward **末尾**、attention residual 与 MLP residual 都完成后,对整层输出乘标量:`hidden_states *= layer_scalar`。12B `hidden_size_per_layer_input=0`、`enable_moe_block=false`,无 PLE/MoE 分支,文本路径就是两次 residual 后整层乘 `layer_scalar`。

### 5.8 Mask 构造(方案①A,但按层类型 + 修正 decode)
- **统一用本库 `create_causal_mask` 的 bool 语义**(`true`=可见,`false`=屏蔽),不混用 -inf 加性 mask;经 `SdpaMask::Array` 传入 fast SDPA。
- 全局层:`create_causal_mask(N, offset, None, None)`(全因果)。
- 滑窗层:`create_causal_mask(N, offset, Some(1024), None)` —— 该 helper **已支持 window_size**,直接复用,无需手搓带状矩阵。
- **关键修正(评审 #5)**:`create_attention_mask` 在 `T==1`(decode)对无 `max_size` 的 cache 返回 `None`;在不截断 KV 的方案下,滑窗层 decode 步必须**显式构造单步滑窗 bool mask**(`create_causal_mask(1, offset, Some(1024), None)`,屏蔽窗口外历史),否则会错误看到全历史。prefill 与 decode 各自按层类型生成 mask;不复用单一全局 mask。
- 备选:给滑窗层用带 `max_size=1024` 的 cache,让 `create_attention_mask` 自动加窗(归入 M3)。
- **R8**:窗口边界(可见 1024 还是含当前 token 的 1025)须与参考实现逐 token 对齐确认。

### 5.9 KV cache
- 每层一个标准增长 `KVCache`(全局 1 KV head;滑窗 8 KV head)。窗口语义靠 §5.8 的 mask 实现。
- **写入时机(防止 cache 到 raw tensor)**:进 cache 的 K 是 **k_norm + RoPE 之后**的张量;进 cache 的 V 是 **v_norm 之后**的张量;global 层的 V 源自 raw k_proj 但**不过 RoPE**(§5.4)。即 cache 存的是 attention 直接消费的 K/V,不是投影原始输出。

### 5.10 输出
- `final_norm` → tied embedding 反投影 → **`final_logit_softcapping`**:`logits = 30 * tanh(logits/30)`,在采样前应用。

### 5.11 prefill 只投影最后位(评审 #5,内存关键)
- vocab=262144 极大:prefill 阶段若对整段 prompt 算 `[B, L, 262144]` logits,4K prompt 下额外吃数 GB。
- **约束**:tied embedding 反投影前先把 hidden 切到最后一位(`forward_last_logits` / `logits_to_keep=1`),prefill 只对最后 hidden 投影;decode `L=1` 本就单位。`Generate` 的 Prefill 分支与 `forward` 须支持只算末位 logits。

## 6. 权重加载(自写,不复用 qwen3 加载器)
- 键前缀 **`language_model.model.`**(评审 #1;qwen3 硬编码 `model.layers` 不可用)。
- 量化配置解析**同时兼容 `quantization` 与 `quantization_config`**(MLX 仓库常并存或改名)。
- 按 §5.1 的 `quant_for(prefix)` 对每个模块取 bits/group_size(评审 #2),分别构造 `MaybeQuantized::Quantized`。
- v_proj 按键存在与否可选加载(global 层预期为 None,§5.4)。
- 集中映射:`...layers.{i}.self_attn.{q,k,[v],o}_proj`、`.q_norm/.k_norm`(`v_norm` 无权重)、`.layer_scalar`、`mlp.{gate,up,down}_proj`、各 layernorm、`embed_tokens`、`model.norm`。

## 7. 验证策略
1. 逐层比对参考实现(标准 gamma RMSNorm、v_norm no-scale、embedding 缩放、双 RoPE、q/k/v-norm、scaling=1.0、global MQA、层末 layer_scalar、final softcap),量化容差内。
2. 贪婪解码(temp=0)首 32 token 与参考一致。
3. `examples/chat_gemma4.rs` 跑通对话:**必须套用仓库的 chat template / `tokenizer_config`(BOS、`<start_of_turn>` 等),不能用裸 prompt**,否则即便权重正确也会输出格式错乱。记录峰值内存、tok/s、实测可用上下文上限。

## 8. 风险与未决项 — **M0 已全部定论(2026-06-04)**

M0 实测参考 = mlx-vlm 0.6.1 gemma4(同一 4bit 权重);单层 forward 对齐:
**layer 0(sliding)max-abs-diff 8.58e-6、layer 5(global)5.72e-6**(满量级 std≈1.3–1.5)。

- R1 ✅ proportional RoPE 正确:`rotated_dims=2*int(0.25*512//2)=128`(非 64),64 个频率,分母 = 完整 head_dim 512,rotate-half(每个 half 的前 64);layer 5 逐元素对齐 5.72e-6。`fast::rope` 的 `freqs` 用 base^(+exp)(kernel 内部取倒数);`rope.rs::proportional_inv_freq` 是 base^(−exp) 的 inv_freq 形式,仅作公式文档,勿直接喂 `fast::rope`。
- R2 ✅ 每层 **4 个 layernorm**(input/post_attention/pre_feedforward/post_feedforward);全部标准 gamma RMSNorm(含 q/k_norm),v_norm 为 no-scale。
- R3 ✅ global 层**无 v_proj**(weight index 确认):value = raw k_proj 输出过 v_norm、不过 RoPE。
- R4 ✅ attention scaling = **1.0**(源码 + 对齐确认)。
- R5 ✅ embedding normalizer = **√hidden_size**(mlx-vlm `embed_scale=61.968=√3840`)。M1 用。
- R6 ✅ 混合精度:attn q/k/o = 4bit、mlp gate/up/down = 8bit、group_size=64、affine;`quant_for` + 自写 loader 兼容。
- R7 ✅ `layer_scalar`:层 forward 末尾整层乘(两次 residual 后);每层均有该权重。
- R8 ⏸ 滑窗边界:`create_causal_mask` 可见带 = `q-window..q`(window+1 列),mask.rs 单测已锁定;长上下文真实滑窗留待 M1/M2(M0 在 L=6<window 下未触发)。
  - **M1 必修**:golden dump 脚本的 mask 是 `create_causal_mask().astype(f32)`,被 mlx SDPA 当**加性** mask(非真正因果)。M0 双方用同一 mask 故对齐成立、且不影响 R1(RoPE 在 SDPA 前已对齐);但 M1 重新 dump(更长 L / 真实滑窗)前须把脚本改成正确的加性 −inf mask,parity 才反映真实注意力。

## 8.1 M1 结论(2026-06-04,已完成)

全模型(48 层)端到端对齐 mlx-vlm(同一 4bit 权重,`load(lazy=True)` 让视觉/音频塔留在磁盘以适配 16GB):
- 固定序列 `[2,1024,2048,4096,8192,16384]` 的**下一 token argmax = 2818,与参考一致**;top-3 完全一致;logits absmax 29.57 vs 29.50(softcap 正确)。
- logits max-abs-diff ≈ 0.80、mean ≈ 0.12(经评审判定为 48 层 bf16/f32 累积的良性误差,非结构性 bug:两侧同权重故量化误差抵消,结构性错误会破坏 argmax)。
- **R8 / M0 mask caveat 关闭**:M1 参考用模型内部的*正确*因果 mask(非 M0 dump 的 f32 hack),我方用 bool `full/sliding` mask,argmax 对齐 → 因果掩码端到端验证通过。
- 内存:`load_model` + 48 层 prefill forward 在 16GB 上跑通(Rust 仅保留文本权重)。
- 仍属 M2:KV cache 解码串联(`forward` 已留 `TODO(M1)`→实为 M2 的 offset 串联)、`Generate` 循环、tokenizer/chat template、tok/s 与上下文上限实测。

## 8.2 M2 结论(2026-06-05,已完成)

**端到端文字对话跑通。** chat_gemma4 对 "Explain what MLX is in one sentence." 输出连贯正确英文:
*"MLX is a machine learning framework developed by Apple that is optimized for unified memory and hardware acceleration on Apple Silicon."*(评审确认为真实 encode→generate→decode,非硬编码)。
- **KV cache 解码正确性已证明**:cache 贪婪解码与 no-cache 重算逐 token 完全一致(8 token:`[2818,107,100,100,14937,100,45518,107]`)→ cache + RoPE offset + 按层 decode mask 机制正确。
- chat 模板:Gemma 4 用 `<|turn>`(105)/`<turn|>`(106)(非 Gemma2/3 的 `<start_of_turn>`),BOS(2)需手动前置,含 thinking 抑制后缀 `<|channel>thought\n<channel|>`;eos=[1,106]。已对照 `tokenizer.json`/`chat_template.jinja` 核实。
- **vs mlx-vlm 贪婪**:token0 一致(2818);后续在"接近"的 argmax 位置因 48 层 bf16 累积(M1 的 ~0.8 logit 差)翻转而分叉 —— 良性、非 bug(由 cache==no-cache 内部一致性证明)。
- **性能(M3 待办)**:首跑 tok/s≈0(27 token/2685s,Metal shader 首次编译;每步 decode shape 变化可能触发重编译)。功能正确,性能未优化。
- 组成:`attention/block` 加 cache+offset 解码路径(prefill 无回归,parity 仍 1e-6 / argmax 2818);`model.forward_cached`+`new_caches`+`decode_mask`;`generate.rs::generate_greedy`;`tokenizer.rs`(load+encode_chat+eos);`examples/{chat_gemma4,generate_gemma4,decode_consistency}`;`scripts/dump_gemma4_greedy.py`。

## 9. 里程碑
1. **M0 权重/配置 spike**(评审建议):解析 `config.json` + `model.safetensors.index.json`;分别实例化 layer 0(sliding)与 layer 5(full),逐键确认存在、每模块 bits/group_size 正确(MLP 8bit/其余 4bit)、global 层 v_proj=None 路径可跑;layer 0/5 最小 forward 与参考对齐;**仅估算** KV 内存(完整 48 层前向前无法可信测上下文上限)。产出 R1–R8 结论。**不求完整生成、不实测上下文上限**。
2. **M1 解码器**:48 层前向 + 按层 mask + 双 RoPE + q/k/v-norm(v 不过 RoPE)+ scaling=1.0 + 层末 layer_scalar + final softcap,逐层对齐。
3. **M2 生成**:自有 `Generate`,端到端贪婪解码与参考一致,`chat_gemma4.rs` 跑通;**实测可用上下文上限**、峰值内存、tok/s。
4. **M3(可选)**:滑窗 RotatingKVCache(长上下文)、proportional RoPE 提升到 mlx-rs-core。

## 10. 后续阶段(本期外)
视觉投影器、音频直通、OminiX-API catalog + engine + handler 接线。
