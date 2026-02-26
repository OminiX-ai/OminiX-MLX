# 用 3000 行 Rust 将 MiniCPM-SALA 移植到 Apple Silicon

MiniCPM-SALA 是一个 90 亿参数的语言模型，能在单张消费级 GPU 上处理超过一百万 token。它的秘诀是将两种注意力机制——稀疏注意力和线性注意力——按比例混合，兼顾全注意力的精确检索能力和循环模型的恒定内存效率。

我们使用 MLX 框架将其移植到了 Apple Silicon 平台，全部用 Rust 实现——从线性注意力的自定义 Metal 内核到 OpenAI 兼容的 API 服务器。本文介绍这个架构的设计思路、移植过程中的关键挑战，以及我们的实测结果。

## 为什么需要混合注意力？

标准 Transformer 在长上下文场景下有一个根本性问题：每一层都要为所有已处理的 token 存储 key-value 对，注意力计算需要在所有 token 之间进行两两配对打分。以 Qwen3-8B 为例，在 256K token 时，仅 KV 缓存就占约 16 GB，且 O(n²) 的计算复杂度让每个 token 的生成速度随上下文增长而变慢。在 1M token 时，内存直接不够用。

学术界已经提出了多种 O(n) 的替代方案：

- **Mamba**——选择性状态空间模型
- **RWKV**——通道级线性注意力
- **RetNet**——带指数衰减的保持机制
- **GLA**——门控线性注意力

这些方法用固定大小的循环状态取代不断增长的 KV 缓存，每个 token 的解码开销恒定，与上下文长度无关。

但问题是：纯线性模型会掉质量。固定大小的状态矩阵无法保留长上下文中所有细粒度信息。它们会"遗忘"——体现为在需要精确检索远距离信息的任务上性能下降。

MiniCPM-SALA 的答案是：**两种都用**。

- **32 层中的 24 层**（75%）使用 Lightning Attention（GLA）——O(n) 复杂度，每层仅需 `[32, 128, 128]` 的固定状态（无论上下文多长都只占约 2 MB）
- **32 层中的 8 层**（25%）使用稀疏注意力（InfLLMv2）——保留完整 KV 缓存，长上下文时通过智能块选择降低开销

稀疏层充当"记忆锚点"，负责精确的远程检索；线性层承担大部分计算，内存恒定。最终效果：在 256K 上下文下比 Qwen3-8B 快 3.5 倍，在 RTX 5090 或 A6000D 上支持 1M+ token。

## 架构详解

### 层分布

32 层的类型分配不是随机的，稀疏层被放置在关键位置：

```
Layer:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
Type:   S  L  L  L  L  L  L  L  L  S  L  L  L  L  L  L

Layer: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
Type:   S  S  L  L  L  L  S  L  L  L  L  L  L  S  S  S
```

第 0 层（开头）和第 29-31 层（末尾三层）全部是稀疏注意力——模型在语义接地和最终预测最关键的首尾位置使用全注意力。其余稀疏层均匀分散在中间。

### Lightning Attention：分块 GLA

每个 Lightning 层维护一个循环状态 `S`，形状为 `[num_heads, head_dim, head_dim]` = `[32, 128, 128]`。单步解码时：

```
decay = exp(alibi_slope)
S = decay * S + k^T @ v      // 用新 token 更新状态
output = q @ S                // 查询状态
```

ALiBi 风格的衰减系数意味着近期 token 的影响力呈指数级大于远处 token——状态以可控的方式自然"遗忘"旧信息。

对于 prefill（一次处理多个 token），逐 token 循环太慢。我们使用**分块 GLA**：将序列切分为 64 token 一块，每块分三步处理：

1. **块内（Intra-chunk）**：块内做二次注意力计算（64×64 足够小，速度很快）
2. **块间（Inter-chunk）**：查询累积的循环状态
3. **状态更新**：将该块的 key/value 以衰减方式融入状态

这就是我们自定义 Metal 内核的用武之地。

### 自定义 Metal 内核

分块 GLA 中有两个性能热点，我们各融合为一个 Metal 内核：

**内核 1：融合块内注意力** —— 计算 `(Q @ K^T) * decay_mask @ V`，全程不在显存中创建完整的 64×64 分数矩阵。每个 threadgroup 处理一个 (batch, head, query_position) 三元组，256 个线程。共享内存存储 Q 行向量和计算出的分数（每组约 1.75 KB）。

**内核 2：融合状态更新** —— 计算 `state_out = chunk_decay * state_in + (K * reverse_decay)^T @ V`。每个 threadgroup 处理一个 (batch, head, d_out) 三元组。

我们还写了第三个内核用于融合 GLA 解码（单步循环），但实测速度提升约为 0%——解码瓶颈在于权重读取（每个 token 需要读取约 9.6 GB 量化权重），而非注意力计算。我们在代码中保留了该内核作为参考，解码阶段仍使用标准 MLX 操作。

### 稀疏注意力：InfLLMv2

8 个稀疏层在短上下文（≤8192 token）时使用标准缩放点积注意力（SDPA）。超过这个长度后，InfLLMv2 的两阶段块选择算法启动：

1. **压缩**：对"中间区域"（初始块和滑动窗口之间）的 key 做均值池化，生成块级代表
2. **打分**：用 query 对压缩后的 key 打分，选出相关性最高的 top-64 个块
3. **聚集**：从初始块 + 选中块 + 滑动窗口收集 K, V
4. **注意力**：在聚集的子集上运行 SDPA

这意味着即使完整上下文是 1M+ token，稀疏层实际只关注约 4K-8K 个 token——每层开销可控，同时保留了从任何位置检索的能力。

### HyPE：混合位置编码

一个微妙但关键的设计：两种注意力类型使用不同的位置编码。

- **Lightning 层**：使用 RoPE（`lightning_use_rope=true`）——循环状态受益于显式的位置信息
- **稀疏层**：不使用 RoPE（`attn_use_rope=false`）——仅依赖因果掩码

这被称为 HyPE（Hybrid Positional Embedding，混合位置嵌入）。[HypeNet 论文](https://arxiv.org/abs/2601.22156)的核心发现是：RNN/线性层的"感受野"有限，主要建模短距离依赖。由于这些层对超出感受野的绝对位置不敏感，模型的长度泛化能力仅取决于注意力层。在注意力层使用 NoPE（无位置编码），模型能更好地泛化到远超训练长度的上下文。

### muP 缩放

模型使用最大更新参数化（muP）保证训练迁移的稳定性：

- **嵌入缩放**：`scale_emb = 12` —— 嵌入向量查找后乘以 12
- **残差缩放**：`scale_depth / sqrt(num_layers)` = `1.4 / sqrt(32)` ≈ 0.247 —— 每个残差连接按此比例缩小
- **Logits 缩放**：`hidden_size / dim_model_base` = `4096 / 256` = 16 —— logits 除以 16

## 自投机解码

我们实现了自投机解码（Self-Speculative Decoding），无需额外的草稿模型即可加速生成。思路是：用 MiniCPM-SALA 自身的前 N 层作为"草稿模型"，预测 K 个 token，然后在一次完整前向传播中验证所有 K 个 token。

MiniCPM-SALA 的前 8 层是 1 个稀疏层 + 7 个 Lightning 层，草稿推理成本极低。如果草稿 token 和全模型的预测一致，就相当于以约 1.25 次全前向传播的代价生成了 K 个 token。

棘手的部分是不匹配时的缓存回滚。稀疏层缓存可以通过删除末尾 N 条记录来截断。Lightning 层缓存无法精确"反更新"——但指数衰减意味着被拒绝 token 的污染会快速消退，所以我们只调整 RoPE 偏移量，接受轻微的状态近似误差。

## Rust + MLX 移植

整个实现约 3000 行 Rust 库代码加 1500 行示例代码：

| 模块 | 行数 | 功能 |
|------|------|------|
| `metal_kernels.rs` | 690 | 通过 `mlx_sys` FFI 调用的自定义 Metal 内核 |
| `model.rs` | 639 | 模型结构、MLP、权重加载（fp32 + 8-bit 量化） |
| `attention/lightning.rs` | 599 | 分块 prefill 和循环解码的 GLA 实现 |
| `attention/sparse.rs` | 486 | SDPA + InfLLMv2 稀疏注意力 |
| `config.rs` | 169 | 基于 Serde 的配置反序列化 |
| `speculative.rs` | 143 | 自投机解码 |
| `attention/mod.rs` | 143 | HybridAttention 枚举分发 |
| `lib.rs` | 90 | 公共 API、ChatML 格式化、思考块过滤 |

示例程序包括 CLI 生成器、交互式多轮对话、批量推理、投机解码器，以及 OpenAI 兼容的 HTTP 服务器。

### 权重加载

8-bit 量化模型在磁盘上占 9.6 GB。对于量化模型，我们手动加载权重——从 safetensors 文件中读取打包的 weight、scales 和 biases，构建 `QuantizedLinear` 层。非量化模型则使用 mlx-rs 内置的 `load_safetensors`，自动处理分片权重文件。

### RoPE 批量推理 Bug 修复

我们发现了 MLX `fast::rope` 实现中的一个 bug：当 `B > 1` 且 `L = 1`（批量解码）时，不同 batch 元素会得到不同的旋转角度。我们的修复方案是将 batch 维度合并到 head 维度（`[B, H, 1, D]` → `[1, B*H, 1, D]`），以 `B=1` 应用 RoPE，然后再 reshape 回来。这个修复记录在 `lightning.rs` 中，对批量推理的正确性至关重要。

## Apple Silicon 上的性能

使用 8-bit 量化模型（9.6 GB）在 Apple Silicon 上的测试结果：

| 指标 | 数值 |
|------|------|
| 模型加载时间 | 0.09 秒 |
| Prefill（10 token） | 27.5 tok/s |
| Prefill（82 token） | 106 tok/s |
| Prefill（1006 token） | 453 tok/s |
| 解码吞吐量 | 27.6 tok/s |

Prefill 速度随 prompt 长度良好扩展——Lightning 层的分块 GLA 在长 prompt 上很高效。解码吞吐量稳定在约 28 tok/s，符合内存带宽瓶颈的特征（每个 token 需要读取 9.6 GB 权重）。

## 推理质量

我们在 temperature=0 下用 11 个高难度推理问题测试了 8-bit 模型：

| 类别 | 题数 | 正确 | 说明 |
|------|------|------|------|
| 数学（算术、模运算） | 3 | 2/3 | 费马小定理正确；27×43 陷入思考循环 |
| 逻辑谜题 | 3 | 2/3 | 蝙蝠与球、脑筋急转弯正确；错误标签箱子陷入循环 |
| 多步推理 | 3 | 3/3 | 火车距离、水壶问题、百分比问题全部正确 |
| 代码 + 执行追踪 | 1 | 1/1 | 括号匹配及追踪过程正确 |
| 约束搜索 | 1 | 0/1 | 传教士与野人问题有状态跟踪错误 |

**总计：8/11 题给出了答案，其中 7/8 正确。**

主要失败模式是**思考循环**——量化模型有时会在 `<think>` 块中陷入反复重复同一段推理，无法收敛。这在 3 个问题上消耗了全部 token 额度。加大 token 预算（8192+）后部分问题能解决，表明这是量化带来的副作用，削弱了模型终止推理链的能力。

## 混合注意力的技术版图

MiniCPM-SALA 位于一个快速演进的设计空间中：

| 模型 | 年份 | 架构 | 线性组件 | 注意力比例 |
|------|------|------|---------|-----------|
| Jamba | 2024 | 交错排列 + MoE | Mamba | 1:7（注意力:SSM） |
| Falcon-H1 | 2025 | 并行混合 | Mamba2 | 每头可变 |
| MiniCPM-SALA | 2026 | 交错排列 | GLA（Lightning） | 1:3（稀疏:线性） |

SALA 的独特之处在于三个方面的组合：线性组件选择 **GLA**（而非 Mamba），稀疏组件使用 **InfLLMv2**（而非简单的滑动窗口），以及 **HALO 蒸馏**方法——从预训练的 Transformer 权重转换而非从头训练。

HALO 方案从 Qwen3 权重出发，替换 75% 的注意力层，仅需约 25% 的从头训练计算量。这使得混合架构方案对没有大规模训练预算的团队也变得可行。

## 相关论文

SALA 综合了多条研究线的成果：

**线性注意力 / 循环替代方案**
- [RetNet](https://arxiv.org/abs/2307.08621)（2023）—— 保持机制，支持并行、循环、分块三种计算模式
- [Mamba](https://arxiv.org/abs/2312.00752)（2023）—— 选择性状态空间模型
- [GLA](https://arxiv.org/abs/2312.06635)（2024）—— 门控线性注意力，SALA 线性层的直接前身
- [Lightning Attention-2](https://arxiv.org/abs/2401.04658)（2024）—— 首个在因果设定下实现理论 O(n) 的线性注意力
- [RWKV](https://arxiv.org/abs/2305.13048)（2023）—— RNN 重构为线性注意力

**稀疏注意力**
- [InfLLM-V2](https://arxiv.org/abs/2509.24663)（2025）—— 稠密-稀疏可切换注意力，SALA 稀疏层的直接组件

**混合架构与蒸馏**
- [HypeNet / HALO / HyPE](https://arxiv.org/abs/2601.22156)（2026）—— Transformer 到混合模型的蒸馏流程，混合位置编码
- [Jamba](https://arxiv.org/abs/2403.19887)（2024）—— 首个 Transformer-Mamba 混合架构
- [Falcon-H1](https://arxiv.org/abs/2507.22448)（2025）—— 并行混合（注意力和 Mamba2 头在层内并行）

**线性注意力的遗忘问题**
- [Alleviating Forgetfulness of Linear Attention](https://arxiv.org/abs/2510.20787)（2025）—— 解释了为什么 SALA 需要 25% 的稀疏层

## 后续计划

移植在功能上已经完整——生成、对话、批量推理、投机解码、量化和服务全部可用。未来的改进方向：

- API 服务器的 **SSE 流式输出**
- **Top-p / Top-k 采样**
- **思考循环检测**——在量化模型陷入重复推理时主动跳出
- **Prompt 缓存**——对共享前缀的请求复用 prefill 计算
- **长上下文基准测试**——在 Apple Silicon 上测量 32K-256K token 长度下的质量

代码基于 Apache-2.0 许可证，是 OminiX-MLX 生态的一部分——一组面向 Apple Silicon 的 Rust 推理引擎。

---

*基于 [mlx-rs](https://github.com/oxideai/mlx-rs) 构建，在 Apple Silicon 上测试。8-bit 量化模型可轻松放入 16 GB 统一内存。*
