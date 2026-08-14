# Transformer 架构详解文档

涵盖 Encoder-Decoder / Decoder-Only 架构变体、训练与推理场景、各模块作用及计算通信分析

目录

  1. Transformer 架构概述
  2. 三大架构变体对比
  3. Encoder-Decoder 架构详解
  4. Decoder-Only 架构详解
  5. 训练场景中的 Transformer
  6. 推理场景中的 Transformer
  7. 各模块作用详解
  8. 各模块计算与通信重点分析
  9. 总结与对比

## 一、Transformer 架构概述

Transformer 由 Google 团队在 2017 年论文 **"Attention Is All You Need"** 中提出，彻底改变了自然语言处理领域。其核心创新是**完全基于注意力机制（Self-Attention）** ，摒弃了 RNN/LSTM 的循环结构，实现了序列的**全并行计算** 。

### 1.1 核心设计思想

**自注意力机制（Self-Attention）** 是 Transformer 的灵魂。对于输入序列中的每个位置，模型可以同时"关注"序列中所有其他位置，通过 Query-Key-Value 三元组计算加权和，从而捕获任意距离的依赖关系。

注意力计算的核心公式：


    Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V

其中：

  * **Q（Query）** ：当前位置的查询向量，表示"我需要什么信息"
  * **K（Key）** ：所有位置的键向量，表示"我能提供什么信息"
  * **V（Value）** ：所有位置的值向量，表示"实际的信息内容"
  * **√d_k** ：缩放因子，防止点积值过大导致 softmax 梯度消失

### 1.2 与 RNN/LSTM 的对比

特性 | RNN / LSTM | Transformer
---|---|---
序列处理方式 | 逐 token 串行处理 | 全序列并行处理
长距离依赖 | 梯度消失/爆炸，依赖受限 | 任意距离直接交互（O(1) 路径）
训练并行度 | 低（必须按时间步串行） | 高（整个序列矩阵运算）
计算复杂度（序列长度 n） | O(n)（但不可并行） | O(n²)（但高度可并行）
位置信息 | 隐式（通过时间步序） | 需显式编码（位置编码）

### 1.3 整体架构图

Encoder（编码器） Input Embedding Positional Encoding Multi-Head Self-Attention Add & Norm Feed-Forward Network Add & Norm ×N Encoder Output (K, V 传给 Cross-Attention) Decoder（解码器） Output Embedding Positional Encoding Masked Multi-Head Self-Attention Add & Norm Multi-Head Cross-Attention Add & Norm Feed-Forward Network Add & Norm ×N

图 1：原始 Transformer（Encoder-Decoder）架构总览

## 二、三大架构变体对比

Transformer 架构衍生出三种主要变体，各自适用于不同的任务场景：

Encoder-Only Input Embedding + PE Multi-Head Self-Attention Add & Norm → FFN → Add & Norm 代表: BERT, RoBERTa, DeBERTa Decoder-Only Input Embedding + PE Masked Multi-Head Self-Attention Add & Norm → FFN → Add & Norm 代表: GPT, LLaMA, Qwen, DeepSeek Encoder-Decoder Encoder Decoder Self-Attn \+ Cross-Attention + Masked Self-Attn Add & Norm → FFN → Add & Norm 代表: T5, BART, Whisper

图 2：三大 Transformer 架构变体对比

维度 | Encoder-Only | Decoder-Only | Encoder-Decoder
---|---|---|---
注意力类型 | 双向 Self-Attention | 单向（因果）Masked Self-Attention | 双向编码 + 单向解码 + Cross-Attention
核心能力 | 深层语义理解 | 自回归文本生成 | 序列到序列映射
典型任务 | 分类、NER、句向量 | 对话、续写、代码生成 | 翻译、摘要、语音识别
参数效率 | 中等 | 高（结构简洁，易扩展） | 较低（双塔结构参数多）
Scaling 表现 | 一般 | 极佳（Scaling Law 验证） | 较好但扩展性不如 Decoder-Only

**关键结论：** 当前主流大语言模型（GPT-4、Claude、LLaMA、Qwen、DeepSeek 等）几乎全部采用 **Decoder-Only** 架构。核心原因是其结构简洁、Scaling Law 表现优异，且通过统一的"下一个 token 预测"目标即可完成几乎所有语言任务。

## 三、Encoder-Decoder 架构详解

### 3.1 架构结构

Encoder-Decoder 是原始 Transformer 论文中的标准架构，由两个独立的堆叠模块组成：

  * Encoder **编码器** ：由 N 层（原论文 6 层）相同的子层堆叠。每层包含一个**双向 Self-Attention** 和一个**FFN** 。编码器可以看到输入序列的全部内容，负责提取源序列的深层语义表示。
  * Decoder **解码器** ：同样由 N 层堆叠，但每层包含三个子层：
    1. **Masked Self-Attention** ：对已生成的输出序列做因果注意力（只看过去，不看未来）
    2. **Cross-Attention** ：Query 来自解码器，Key 和 Value 来自编码器输出，实现源序列到目标序列的信息对齐
    3. **FFN** ：对融合后的信息做非线性变换

### 3.2 数据流向

**编码阶段：** 源序列 → Input Embedding + Positional Encoding → Encoder（×N 层）→ 编码器输出（一组 Key-Value 向量）

**解码阶段：** 已生成的目标序列 → Output Embedding + Positional Encoding → Masked Self-Attention → Cross-Attention（接收编码器 K, V）→ FFN → Linear + Softmax → 输出概率分布 → 采样下一个 token

### 3.3 使用场景

场景 | 说明 | 代表模型
---|---|---
机器翻译 | 源语言 → 目标语言，需要完整的源端理解 + 目标端生成 | 原始 Transformer, Marian, NLLB
文本摘要 | 长文本 → 短摘要，编码器理解全文，解码器生成精炼摘要 | BART, PEGASUS, T5
语音识别（ASR） | 音频特征序列 → 文本，编码器处理声学特征 | Whisper, Conformer
文本到图像描述 | 图像特征 → 自然语言描述 | Show and Tell 变体
代码翻译 | 一种编程语言 → 另一种编程语言 | CodeT5

**为什么这些场景需要 Encoder-Decoder？**

  * 源序列和目标序列是**不同模态或不同语言** ，需要独立的编码空间
  * 编码器的**双向注意力** 可以完整理解源端上下文（不受因果约束）
  * Cross-Attention 提供了**显式的源-目标对齐机制** ，对翻译等任务天然适配
  * 源序列长度与目标序列长度通常**不对称** ，双塔结构更灵活

## 四、Decoder-Only 架构详解

### 4.1 架构结构

Decoder-Only 架构去掉了编码器，仅保留解码器主体，但**移除了 Cross-Attention 层** 。每个 Transformer 层仅包含：

  1. **Masked Multi-Head Self-Attention** ：因果注意力，每个位置只能关注自身及之前的位置
  2. **Feed-Forward Network（FFN）** ：两层 MLP，提供非线性变换能力

Decoder-Only Block（×N） Token Embedding + Position Masked Multi-Head Self-Attention Add & LayerNorm Feed-Forward Network Add & LayerNorm Linear + Softmax Next Token 概率分布 Causal Mask 仅看过去

图 3：Decoder-Only 架构与因果掩码示意

### 4.2 训练目标：下一个 Token 预测

Decoder-Only 模型采用**自回归语言建模** （Autoregressive Language Modeling）作为训练目标：


    给定序列 [t₁, t₂, ..., tₙ]，模型学习：
      P(t₂ | t₁)
      P(t₃ | t₁, t₂)
      ...
      P(tₙ | t₁, t₂, ..., tₙ₋₁)

    总损失 = -Σ log P(tᵢ | t₁, ..., tᵢ₋₁)

### 4.3 使用场景

场景 | 说明
---|---
对话 / 聊天 | ChatGPT、Claude、通义千问等对话系统，通过指令微调 + RLHF 实现
文本续写 | 给定前文，自回归生成后续文本，包括创意写作、学术写作
代码生成 | GitHub Copilot、Code Llama 等，根据注释或上下文生成代码
推理 / 数学 | 通过 Chain-of-Thought 实现逐步推理，如 GPT-4、o1、DeepSeek-R1
多模态扩展 | 将图像/音频编码为 token 序列，复用 Decoder-Only 架构（如 GPT-4V）
工具调用 / Agent | 生成结构化的函数调用 JSON，驱动外部工具执行

### 4.4 为什么现代 LLM 选择 Decoder-Only？

  1. **Scaling Law 优势** ：OpenAI 研究表明，Decoder-Only 架构在增加参数量、数据量、计算量时，loss 下降更加可预测且持续
  2. **统一任务范式** ：所有任务都可以转化为"给定前文，预测下一个 token"，无需针对任务设计特殊结构
  3. **结构简洁** ：每层只有 2 个子模块（Self-Attention + FFN），比 Encoder-Decoder 少一个 Cross-Attention，参数效率更高
  4. **In-context Learning** ：因果注意力天然支持 Few-shot Learning，模型可以从 prompt 上下文中学习新任务
  5. **工程友好** ：统一的 KV Cache 机制使推理优化更加直接
  6. **Zero-shot 能力** ：大规模预训练后，无需微调即可完成多种任务

## 五、训练场景中的 Transformer

### 5.1 训练时的核心特征：全序列并行

训练时，整个输入序列一次性送入模型，所有 token 的前向传播**同时进行** 。这是 Transformer 相比 RNN 的最大优势。

**训练流程：**

  1. 输入完整序列 [t₁, t₂, ..., tₙ]，添加位置编码
  2. 通过因果掩码（Causal Mask），位置 i 只能关注位置 ≤ i 的 token
  3. 整个序列矩阵运算，一次性得到所有位置的 logits
  4. 将 logits 右移一位，与真实标签计算交叉熵损失
  5. 反向传播更新参数

### 5.2 训练时的注意力掩码

训练中通过掩码矩阵实现因果性，同时处理所有位置：


    Attention Mask（Causal）:
         t₁   t₂   t₃   t₄
    t₁ [  0   -∞   -∞   -∞ ]     ← t₁ 只看自己
    t₂ [  0    0   -∞   -∞ ]     ← t₂ 看 t₁, t₂
    t₃ [  0    0    0   -∞ ]     ← t₃ 看 t₁, t₂, t₃
    t₄ [  0    0    0    0 ]     ← t₄ 看所有

    softmax 后 -∞ 位置变为 0，实现因果约束

**关键优势** ：训练时虽然是因果的，但所有位置的 loss 可以**同时计算** 。一个长度为 n 的序列，相当于一次前向传播产生了 n 个训练样本（每个位置都是一个预测任务），极大地提高了训练效率。

### 5.3 训练中的计算瓶颈

阶段 | 计算特征 | 瓶颈
---|---|---
前向传播 | 矩阵乘法为主：QKV 投影、Attention 矩阵、FFN 两层 MLP | 计算密集型（Compute-bound）
反向传播 | 需要保存所有中间激活值（Attention 权重、FFN 中间状态） | 显存密集型（Memory-bound）
梯度同步 | 多卡训练时 AllReduce 同步梯度 | 通信密集型（Communication-bound）
激活重计算 | 为节省显存丢弃部分激活，反向时重新计算 | 增加约 33% 计算量，换取显存节省

### 5.4 分布式训练策略

数据并行 (DP/DP) GPU 0: 完整模型 GPU 1: 完整模型 GPU 2: 完整模型 不同数据批次 梯度 AllReduce 通信: 梯度同步 显存: 完整模型副本 张量并行 (TP) GPU 0: Attention 半部分 GPU 1: Attention 半部分 GPU 2: FFN 半部分 同一模型切片 层内 AllReduce / AllGather 通信: 高频层内同步 显存: 模型分片 流水线并行 (PP) GPU 0: Layer 1-8 GPU 1: Layer 9-16 GPU 2: Layer 17-24 模型按层切分 微批次流水线 通信: 层间 P2P 显存: 层级分片 ZeRO Stage 1: 优化器分片 Stage 2: + 梯度分片 Stage 3: + 参数分片 渐进式分片 按需 AllGather 通信: 按需 Gather 显存: 极致节省

图 4：四种主流分布式训练策略对比

### 5.5 训练中的 Teacher Forcing

训练时使用**Teacher Forcing** 策略：每个位置的输入使用**真实标签** （ground truth）而非模型自己的预测。这意味着即使某个位置预测错误，后续位置的输入仍然是正确的，保证训练稳定收敛。


    训练时:
      输入:  [BOS]  我    喜欢  编程
      标签:   我    喜欢  编程  [EOS]

      位置 1: 输入 [BOS]    → 预测 "我"
      位置 2: 输入 "我"     → 预测 "喜欢"（用的是真实标签"我"，不是模型预测的）
      位置 3: 输入 "喜欢"   → 预测 "编程"
      位置 4: 输入 "编程"   → 预测 [EOS]

## 六、推理场景中的 Transformer

### 6.1 自回归生成

推理时，模型必须**逐 token 生成** ：每一步根据已生成的全部 token 预测下一个 token，然后将新 token 追加到序列中，再预测下一个。这与训练时的全序列并行形成鲜明对比。

Step 1: Prefill（预填充） 输入: "今天天气" 全序列并行计算 Attention 计算所有 KV Cache 输出: "真" ✓ 计算密集型 一次性处理所有 prompt 填充 KV Cache 瓶颈: GPU 计算力 Step 2: Decode（逐 token） 输入: "真"（仅 1 token） Q=新token, K,V=全部缓存 追加 1 条 KV Cache 输出: "好" ✗ 显存密集型 每步仅算 1 个 token 但需读取全部 KV Cache 瓶颈: 显存带宽 Step 3: 继续生成 输入: "好"（仅 1 token） 重复 Step 2 流程 KV Cache 持续增长 输出: "，" 直到 EOS 或长度上限 KV Cache 线性增长 显存压力持续增大 需 PagedAttention 优化

图 5：推理过程——Prefill 与 Decode 两阶段

### 6.2 KV Cache 机制

KV Cache 是推理加速的**核心技术** 。在自回归生成中，已生成 token 的 K 和 V 向量不会改变，因此缓存后避免重复计算：

**无 KV Cache：** 生成第 n 个 token 时，需要重新计算前 n-1 个 token 的 QKV 投影和 Attention，计算量为 O(n²)。

**有 KV Cache：** 已生成 token 的 K、V 缓存在显存中，新 token 只需计算自身的 Q，然后与缓存的 K、V 做 Attention。计算量降为 O(n)。


    KV Cache 结构（以 2 head 为例）:

    生成 token t₁ 后:
      K Cache: [K₁^head1, K₁^head2]
      V Cache: [V₁^head1, V₁^head2]

    生成 token t₂ 后:
      K Cache: [K₁^head1, K₂^head1, K₁^head2, K₂^head2]
      V Cache: [V₁^head1, V₂^head1, V₁^head2, V₂^head2]

    生成 token t₃:
      仅计算 Q₃, K₃, V₃
      Q₃ 与整个 K Cache 做 Attention
      K₃, V₃ 追加到 Cache

### 6.3 推理的两个阶段

特征 | Prefill 阶段 | Decode 阶段
---|---|---
处理内容 | 整个 prompt（可能数千 token） | 每次 1 个 token
计算特征 | 计算密集型（Compute-bound） | 显存密集型（Memory-bound）
GPU 利用率 | 高（大矩阵运算充分利用 SM） | 低（小矩阵运算，SM 利用率不足）
瓶颈 | GPU 计算算力 | 显存带宽（读取 KV Cache）
KV Cache | 初始化并填充 | 持续追加增长
优化方向 | 算子融合、Flash Attention | Continuous Batching、PagedAttention

### 6.4 推理优化技术

**1\. Flash Attention** ：通过分块计算和重排内存访问，减少 HBM 读写次数，将 Attention 从 O(n²) 显存读写降低到 O(n²/√M)（M 为 SRAM 大小），显著加速 Prefill。

**2\. PagedAttention（vLLM）** ：借鉴操作系统的虚拟内存分页机制，将 KV Cache 分成固定大小的块（Block），按需分配，避免显存碎片化，支持 Continuous Batching。

**3\. Continuous Batching** ：不同于静态批处理（等所有序列同时完成），动态地在每个 token 生成步插入/移除请求，极大提高 GPU 利用率和吞吐量。

**4\. Speculative Decoding（投机解码）** ：用小模型快速生成候选 token，大模型并行验证，将多个 token 的串行验证变为并行，加速 2-3 倍。

**5\. GQA / MQA（分组/多查询注意力）** ：多个 Query Head 共享同一组 K/V Head，大幅减少 KV Cache 体积，降低 Decode 阶段的显存带宽压力。LLaMA-2/3、Qwen 等均已采用。

## 七、各模块作用详解

### 7.1 Input Embedding（输入嵌入层）

**作用** ：将离散的 token ID 映射为连续的稠密向量表示。词表大小 V × 嵌入维度 d_model 的查找表。

  * **数学形式** ：`E = Embedding[token_id]`，输出形状 `(batch, seq_len, d_model)`
  * **权重共享** ：输入嵌入和输出投影层（LM Head）常共享权重，减少参数量
  * **缩放** ：原论文中嵌入向量乘以 √d_model，使其与位置编码量级匹配
  * **现代改进** ：SentencePiece / BPE / Tiktoken 分词 + 嵌入；RoPE 位置编码替代绝对位置编码

### 7.2 Positional Encoding（位置编码）

**作用** ：Self-Attention 本身是排列不变的（Permutation Invariant），无法区分 token 的顺序。位置编码为每个位置注入位置信息，使模型感知序列顺序。

类型 | 方法 | 特点 | 代表模型
---|---|---|---
绝对位置编码 | 正弦/余弦函数 | 固定不可学习，可外推到更长序列 | 原始 Transformer
可学习位置编码 | 随机初始化，训练学习 | 灵活但不可外推，长度固定 | BERT, GPT-2
ALiBi | 注意力偏置 | 不用位置编码，在 Attention 分数上加距离偏置 | BLOOM, MPT
RoPE（旋转位置编码） | 旋转矩阵作用于 Q, K | 相对位置编码，理论可外推，现代主流 | LLaMA, Qwen, DeepSeek

### 7.3 Multi-Head Self-Attention（多头自注意力）

**作用** ：Transformer 的核心模块。通过多头机制，让模型从不同子空间、不同视角关注序列中的依赖关系。每个 Head 可以学习不同的注意力模式（如语法依赖、语义关联、局部模式等）。

**计算流程：**


    输入: X ∈ R^(batch × seq_len × d_model)

    1. 线性投影:  Q = X·W_Q,  K = X·W_K,  V = X·W_V
       W_Q, W_K ∈ R^(d_model × d_k)
       W_V ∈ R^(d_model × d_v)

    2. 分头: 将 d_model 拆分为 h 个头
       每个头维度: d_k = d_v = d_model / h
       Q_i, K_i, V_i ∈ R^(batch × seq_len × d_k)

    3. 每个头独立计算 Attention:
       head_i = softmax(Q_i · K_i^T / √d_k) · V_i

    4. 拼接所有头:
       MultiHead = Concat(head_1, ..., head_h) · W_O

    输出: R^(batch × seq_len × d_model)

**三种注意力变体：**

  * **MHA（Multi-Head Attention）** ：标准多头，每个 Head 有独立的 Q/K/V，KV Cache 最大
  * **MQA（Multi-Query Attention）** ：所有 Head 共享一组 K/V，KV Cache 最小，但质量略降
  * **GQA（Grouped-Query Attention）** ：折中方案，每 g 个 Head 共享一组 K/V，兼顾质量和效率

### 7.4 Feed-Forward Network（前馈网络）

**作用** ：对 Attention 输出进行非线性变换。FFN 是两层 MLP，中间维度通常为 4×d_model。它承担了模型中**大部分参数量** （约 2/3），是知识存储的主要载体。


    标准 FFN:
      FFN(x) = ReLU(x · W₁ + b₁) · W₂ + b₂
      W₁ ∈ R^(d_model × 4·d_model)
      W₂ ∈ R^(4·d_model × d_model)

    现代变体（LLaMA 等）:
      FFN(x) = (SwiGLU(x · W_gate) ⊙ SiLU(x · W_up)) · W_down
      SwiGLU = Swish(x·W_gate) ⊙ (x·W_up)
      使用门控机制 + Swish 激活函数
      中间维度约为 8/3 × d_model（为保持参数量与标准 FFN 一致）

**关键理解：** Attention 负责"信息路由"（决定哪些 token 的信息需要聚合），FFN 负责"信息变换"（对聚合后的信息做非线性加工和知识检索）。

### 7.5 Layer Normalization（层归一化）

**作用** ：稳定每一层的输入分布，加速训练收敛，缓解深层网络的梯度问题。

变体 | 归一化位置 | 公式 | 特点
---|---|---|---
Post-LN（原始） | 子层之后 | LayerNorm(x + Sublayer(x)) | 深层训练不稳定，需 warmup
Pre-LN | 子层之前 | x + Sublayer(LayerNorm(x)) | 深层训练稳定，主流选择
RMSNorm | 子层之前 | x + Sublayer(RMSNorm(x)) | 去掉均值中心化，计算更快，效果相当

**RMSNorm** （Root Mean Square Normalization）是当前主流（LLaMA、Qwen 等），相比 LayerNorm 去掉了减均值的操作，仅用 RMS 缩放：


    RMSNorm(x) = x / RMS(x) · γ
    RMS(x) = √(1/d · Σ xᵢ²)

### 7.6 Residual Connection（残差连接）

**作用** ：将子层的输入直接加到输出上，形成"快捷路径"。使梯度可以绕过非线性变换直接传播，解决深层网络的梯度消失/爆炸问题，支持训练上百层的 Transformer。


    output = x + Sublayer(x)

    完整结构（Pre-LN + 残差）:
      output = x + Dropout(Sublayer(LayerNorm(x)))

残差连接使信息可以在多层之间"跳跃"传播，每层只需学习"增量"而非完整变换，降低了优化难度。

### 7.7 Cross-Attention（交叉注意力，仅 Encoder-Decoder）

**作用** ：连接编码器和解码器的桥梁。Decoder 的 Query 来自自身已生成的序列，Key 和 Value 来自 Encoder 的输出，实现源序列到目标序列的信息对齐和传递。


    Cross-Attention:
      Q = Decoder_output · W_Q    （来自解码器）
      K = Encoder_output · W_K    （来自编码器）
      V = Encoder_output · W_V    （来自编码器）

      Attention = softmax(Q · K^T / √d_k) · V

在翻译任务中，Cross-Attention 的权重矩阵直观地反映了源语言和目标语言之间的词对齐关系。

### 7.8 Output Projection（输出投影层）

**作用** ：将 Transformer 最后一层的隐状态投影到词表空间，输出每个 token 的概率分布。


    logits = Hidden_state · W_output    （W_output ∈ R^(d_model × vocab_size)）
    probs = softmax(logits / temperature)

    # Top-k / Top-p 采样
    next_token = sample_from_top_k(probs, k=50)

  * **权重共享** ：W_output 常与 Input Embedding 共享，减少大量参数（vocab_size × d_model）
  * **计算特点** ：这个投影是一个巨大的矩阵乘法，vocab_size 通常 3 万~15 万，是推理中不可忽视的开销

## 八、各模块计算与通信重点分析

本节从**计算量（FLOPs）** 、**显存占用** 、**显存带宽** 、**通信开销** 四个维度，深入分析每个模块在训练和推理中的特征。

### 8.1 全局视角：计算 vs 通信

训练场景 Attention 计算 计算密集型 | O(n²·d) FLOPs | Flash Attention 优化 FFN 计算 计算密集型 | O(n·d²) FLOPs | 占总 FLOPs 约 2/3 激活值显存 显存密集型 | 需保存所有中间值用于反向传播 梯度同步 通信密集型 | AllReduce 同步所有参数梯度 瓶颈排序: 通信 > 显存 > 计算（大模型） 推理场景 Prefill: Attention + FFN 计算密集型 | 并行处理 prompt | GPU 高利用率 Decode: KV Cache 读取 显存密集型 | 每步读全部 Cache | 带宽瓶颈 Decode: Attention 计算 计算量极小（1 token × 全 Cache）| GPU 利用率低 Output Projection 大矩阵乘法 | d × vocab_size | 不可忽略 瓶颈排序: 显存带宽 > 计算 > 通信（单卡）

图 6：训练 vs 推理场景的计算与通信瓶颈对比

### 8.2 各模块详细分析

#### ① Input Embedding

维度 | 训练 | 推理
---|---|---
计算量 | 极低（查表操作 O(batch × seq_len × d_model)） | 极低（查表 O(d_model)）
显存 | vocab_size × d_model 参数（70B 模型约 500MB） | 同左（常驻显存）
通信 | DP: 无；TP: 无（不切分）；ZeRO-3: AllGather 获取分片 | 无
重点 | 几乎不是瓶颈。唯一注意点是词表大小影响参数量和 Output Projection 的计算量。

#### ② Positional Encoding / RoPE

维度 | 训练 | 推理
---|---|---
计算量 | 低（正弦/余弦或旋转矩阵，元素级操作） | 低（仅 1 个位置的旋转）
显存 | 几乎为零（可预计算缓存） | 几乎为零
通信 | 无 | 无
重点 | RoPE 在 QK 投影后施加旋转，是元素级操作，开销可忽略。但其外推特性（NTK-aware / YaRN 等）对长上下文推理至关重要。

#### ③ Multi-Head Self-Attention（核心模块）

**这是 Transformer 中计算与通信分析最复杂的模块。**

维度 | 训练 | 推理 — Prefill | 推理 — Decode
---|---|---|---
计算量 | O(batch × n² × d_model) — QKV 投影 + Attention 矩阵 + 输出投影 | 同训练（处理整个 prompt） | O(batch × n × d_model) — QKV 投影仅 1 token，Attention 是 1×n
Attention 矩阵 | n×n 矩阵，显存 O(n²)，是显存瓶颈 | 同训练，Flash Attention 优化 | 1×n 向量，计算量极小
KV Cache | 不需要（每层重新计算） | 初始化并填充 | **核心瓶颈** ：每层每头缓存 n × d_k，总量 = 2 × n_layers × n_heads × seq_len × d_k × batch × dtype_size
显存带宽 | 中等（QKV 投影读写激活） | 中等 | **极高** ：每步需读取全部 KV Cache，带宽消耗 = KV_Cache_size / step_time
通信（TP） | 2 次 AllReduce / 层（QKV 投影后 + 输出投影后） | 同训练 | 2 次 AllReduce / 层，但计算量极小 → **通信占比极高**
通信（DP） | 反向传播时梯度 AllReduce | — | —

**训练时 Attention 的通信特点：**

  * **张量并行（TP）** ：QKV 权重按 Head 切分到不同 GPU，每个 GPU 计算部分 Head 的 Attention，然后 AllReduce 合并结果。**每层 2 次通信** （Attention 输出 + FFN 输出），通信频率高
  * **序列并行（SP）** ：将序列维度切分到不同 GPU，减少 Attention 矩阵的显存占用，但需要额外的 AllGather / Reduce-Scatter 通信
  * **Ring Attention / Ulysses** ：长序列场景下，将 Attention 计算分散到多 GPU，通过 P2P 通信传递 QKV 块

**推理 Decode 阶段 Attention 的通信特点：**

  * 单 token 计算量极小，但需读取全部 KV Cache → **显存带宽是绝对瓶颈**
  * TP 并行时每层 2 次 AllReduce，通信延迟可能占总延迟的 50% 以上
  * GQA/MQA 通过减少 KV Head 数量，直接减少 KV Cache 读取量，是推理优化的关键
  * 多卡推理时，KV Cache 按层分布（Pipeline）或按 Head 分布（TP），跨卡读取代价高

#### ④ Feed-Forward Network（FFN）

维度 | 训练 | 推理 — Prefill | 推理 — Decode
---|---|---|---
计算量 | O(batch × n × d_model × d_ff)，d_ff = 4×d_model，**占总计算量约 2/3** | 同训练 | O(d_model × d_ff)，单 token 的两层 MLP
参数量 | 2 × d_model × d_ff（标准 FFN）或 3 × d_model × (8/3 × d_model)（SwiGLU），约占总参数 2/3
显存 | 中间激活 O(batch × n × d_ff)，是显存大户 | 同训练 | O(d_ff)，极小
通信（TP） | FFN 权重按列切分，第一层后 AllGather，第二层后 Reduce-Scatter | 同训练 | 同训练，但计算量小 → 通信占比高

**FFN 的计算通信特点：**

  * FFN 是**纯矩阵乘法** ，无序列间交互，天然适合并行
  * 训练时 FFN 的计算量最大（约 2/3），是 GPU 计算力的主要消费者
  * 推理 Decode 时，FFN 需要加载 W_gate, W_up, W_down 三个权重矩阵（约模型参数的 2/3），**权重加载的显存带宽消耗** 是 Decode 慢的主要原因之一
  * TP 切分 FFN 时，通信开销与 Attention 类似，但计算/通信比更优（FFN 计算量更大）
  * **MoE（混合专家）** ：将 FFN 替换为多个专家 FFN，每个 token 只激活 2 个专家，大幅增加参数量而不增加计算量。引入额外的 All-to-All 通信进行 token 路由

#### ⑤ Layer Normalization / RMSNorm

维度 | 训练 | 推理
---|---|---
计算量 | 极低（沿 d_model 维度求均值/方差 + 缩放） | 极低
显存 | 每层 2 × d_model 参数（γ, β），可忽略 | 同左
通信 | TP 时无需通信（沿 d_model 归一化，每个 GPU 有完整维度） | 无
重点 | 计算和通信开销均可忽略。但 LN 的位置（Pre/Post）对训练稳定性影响巨大。RMSNorm 比 LN 快约 10-30%。

#### ⑥ Residual Connection（残差连接）

维度 | 训练 | 推理
---|---|---
计算量 | 极低（元素级加法） | 极低
显存 | 需保存残差路径的激活值（用于反向传播），累积效应导致深层显存线性增长 | 每层需保存残差结果，但无额外开销
通信 | TP 时残差加法在 AllReduce 之后进行，不额外通信 | 无
重点 | 计算通信开销可忽略。但在 TP 中，残差路径的数据需在每个 GPU 上保持一致，AllReduce 操作隐含了这一同步。

#### ⑦ Cross-Attention（仅 Encoder-Decoder）

维度 | 训练 | 推理
---|---|---
计算量 | O(batch × n_tgt × n_src × d_model)，target-source 交叉注意力 | O(n_src × d_model) per token
KV Cache | — | 需要缓存 Encoder 输出的 K, V（固定大小，不增长）+ Decoder 的 KV Cache（增长）
通信 | Encoder 在前向结束后需将 K, V 传给所有 Decoder 层。多卡时 Encoder 和 Decoder 可能分布在不同 GPU | Encoder 的 KV Cache 一次性计算后复用，无需重复编码
重点 | Cross-Attention 的 KV 来自 Encoder，在推理时**Encoder 只需前向一次** ，其 KV 可永久缓存复用，这是 Encoder-Decoder 架构在翻译等任务中的效率优势。

#### ⑧ Output Projection（LM Head）

维度 | 训练 | 推理 — Prefill | 推理 — Decode
---|---|---|---
计算量 | O(batch × n × d_model × vocab_size)，vocab 通常 3-15 万 → **巨大** | 同训练，但仅需最后一个位置的 logits | O(d_model × vocab_size) per token
显存 | logits 本身 O(batch × n × vocab_size) 非常大 | — | —
通信 | TP 时 vocab 维度切分，各 GPU 计算部分 logits，AllGather 合并 | — | 同训练，每步通信一次
重点 | 训练时需计算所有位置的 loss，Output Projection 的计算量与 FFN 相当甚至更大。推理 Decode 时每步都要做一次 d × vocab 矩阵乘，不可忽略。常用优化：logit tying（共享 Embedding 权重）、vocabulary parallelism（vocab 切分到多 GPU）。

### 8.3 模块级计算/通信总结矩阵

模块 | 训练计算量 | 训练显存 | 训练通信 | 推理计算量 | 推理显存/带宽 | 推理通信
---|---|---|---|---|---|---
Embedding | 极低 | 低 | 无 | 极低 | 低 | 无
Positional Encoding | 极低 | 极低 | 无 | 极低 | 极低 | 无
Self-Attention | 高 O(n²d) | 高（n² 矩阵） | 高（TP AllReduce） | 中（Decode 低） | 极高（KV Cache 带宽） | 高（TP AllReduce）
Cross-Attention | 中 O(n₁n₂d) | 中 | 中 | 中 | 中（固定 KV Cache） | 中
FFN | 最高 O(nd²) | 高（中间激活） | 中（TP AllReduce） | 中（Decode 低） | 高（权重加载带宽） | 中
LayerNorm/RMSNorm | 极低 | 极低 | 无 | 极低 | 极低 | 无
Residual | 极低 | 低 | 无 | 极低 | 极低 | 无
Output Projection | 高 O(nd·V) | 高（logits） | 中（vocab TP） | 中 O(d·V) | 中 | 中

### 8.4 不同并行策略下各模块的通信模式

张量并行（TP）下单层 Transformer 的通信流程 GPU 0 QKV 投影（Head 0-h/2） Attention 计算（Head 0-h/2） → AllReduce（通信点 ①） LayerNorm + 残差 FFN 第一层（列切分，输出 4d/2） FFN 第二层（行切分，输出 d） → AllReduce（通信点 ②） LayerNorm + 残差 GPU 1 QKV 投影（Head h/2-h） Attention 计算（Head h/2-h） → AllReduce（通信点 ①） LayerNorm + 残差 FFN 第一层（列切分，输出 4d/2） FFN 第二层（行切分，输出 d） → AllReduce（通信点 ②） LayerNorm + 残差 同步 同步

图 7：张量并行下每层 Transformer 的 2 次 AllReduce 通信点

**通信模式总结：**

  * **数据并行（DP）** ：每层反向传播后 AllReduce 梯度，通信量 = 模型参数量 × 2（梯度），通信频率 = 每步 1 次
  * **张量并行（TP）** ：每层 2 次 AllReduce（Attention 后 + FFN 后），通信量 = batch × seq_len × d_model × 2，通信频率极高（每层每步 2 次）。**必须在 NVLink/InfiniBand 高速互联的 GPU 间使用**
  * **流水线并行（PP）** ：层间 P2P 传递激活值，通信量 = batch × seq_len × d_model，通信频率 = 每层 1 次。带宽需求较低但存在 Bubble
  * **ZeRO-3** ：前向/反向时 AllGather 获取参数分片，Reduce-Scatter 同步梯度。通信量约 = DP 的 1.5 倍，但显存节省巨大
  * **MoE 专家并行** ：引入 All-to-All 通信将 token 路由到目标专家所在 GPU，通信模式更复杂

### 8.5 训练 vs 推理的计算通信对比总结

对比维度 | 训练 | 推理
---|---|---
计算特征 | 全序列并行，计算密集型 | Prefill 计算密集 + Decode 显存密集
主要瓶颈 | 通信（多卡 AllReduce）+ 显存（激活值） | Decode: 显存带宽（KV Cache 读取）
Attention 瓶颈 | n×n 矩阵的显存和计算 | KV Cache 的显存占用和带宽读取
FFN 瓶颈 | 计算量（占总 FLOPs 2/3） | 权重加载的显存带宽
通信模式 | TP AllReduce / DP AllReduce / PP P2P | TP AllReduce（Decode 时通信占比极高）
Batch 效率 | 大 batch 充分利用 GPU | Decode batch=1 时 GPU 利用率极低，需 Continuous Batching
核心优化技术 | Flash Attention, 激活重计算, 混合精度, ZeRO | KV Cache, PagedAttention, GQA, Speculative Decoding, Continuous Batching

## 九、总结与对比

### 9.1 架构选择建议

**选择 Encoder-Decoder 的场景：**

  * 源序列和目标序列属于不同模态/语言（翻译、ASR、图文）
  * 需要双向理解源端信息（不限于因果顺序）
  * 源序列固定不变，可以一次性编码后复用（如翻译系统的编码器缓存）
  * 序列长度差异较大，需要独立处理

**选择 Decoder-Only 的场景：**

  * 通用语言模型（对话、续写、推理、代码生成）
  * 需要强大的 Scaling 能力和 Zero-shot 泛化
  * 统一任务范式，简化工程实现
  * 当前所有主流 LLM 的事实标准

### 9.2 关键洞察

  1. **Attention 是信息路由器** ：决定"哪些 token 的信息需要聚合到当前位置"，计算量 O(n²d) 在训练时是主要开销
  2. **FFN 是知识存储库** ：承担约 2/3 的参数和计算量，是模型知识的主要载体
  3. **训练瓶颈在通信，推理瓶颈在带宽** ：训练时多卡 AllReduce 是主要瓶颈；推理 Decode 时 KV Cache 读取是显存带宽瓶颈
  4. **KV Cache 是推理的核心数据结构** ：它使自回归生成从 O(n²) 降为 O(n)，但其显存占用和带宽读取成为新瓶颈
  5. **GQA/MQA 是推理优化的关键** ：通过减少 KV Head 数量，在不显著损失质量的前提下大幅减少 KV Cache 体积
  6. **TP 的通信开销在 Decode 时被放大** ：因为单 token 计算量极小，AllReduce 的延迟占比显著上升
  7. **MoE 改变了 FFN 的计算通信格局** ：增加参数量而不增加计算量，但引入 All-to-All 路由通信

* * *

**本文档涵盖了 Transformer 架构的核心内容，从架构变体到模块细节，从训练到推理，从计算到通信。希望对理解 Transformer 的全貌有所帮助。**
