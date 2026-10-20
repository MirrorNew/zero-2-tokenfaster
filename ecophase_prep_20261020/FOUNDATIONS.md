# 训练与推理优化：用一条链路串起基础

如果 X/Q/K/V、缓存、服务框架或性能测试仍不清楚，先看 [逐项白话解释与手算例子](FOUNDATIONS_EXPLAINED.md)，再回来读本页。

## 1. 先把模型运行过程画出来

```text
文本 → tokenizer → token IDs → embedding
    → 多层 [attention + MLP，配合 norm/residual]
    → 下一个 token 的 logits → 采样或选取 → 接回上下文
```

Token 不等于一个汉字或单词；分词方式由 tokenizer 决定。Logits 是尚未归一化的分数，softmax 后成为概率分布。

训练时通常提供完整的已知文本，利用因果 mask 同时计算多个位置的下一 token 预测损失，再反向传播更新参数。生成时未来 token 未知，要逐步使用前一步得到的内容；不能直接把整段未知答案像输入一样并行计算。

Attention 用当前位置的 Q 与可见位置的 K 计算相关性，再加权汇总 V：

`Q = XWq; K = XWk; V = XWv`

`Attention = softmax(QKᵀ / sqrt(head_dim) + mask)V`

可以把 Q 理解为当前想查询的特征，K 为匹配特征，V 为被聚合的信息；这只是帮助理解的比喻。多头让模型使用不同投影空间。MLP 是每个位置上的非线性变换；residual 与 norm 帮助信息传递和数值稳定。

对 decoder 的因果 attention，一个位置只能看到自己和过去。PyTorch 的具体 mask 语义应看 [SDPA 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)，尤其不要猜非方形 Q/K 的 mask 对齐。

## 2. 训练、微调、强化学习、推理

| 阶段 | 在做什么 | 主要代价 |
|---|---|---|
| 预训练 | 从大量数据学习预测能力 | 前向、反向、优化器更新、通信 |
| SFT | 从已有模型继续学输入—回答示例 | 同属训练；可全参数或使用 LoRA 等 |
| RL/post-training | 生成行为、获得奖励、更新策略 | rollout 推理、环境/工具交互、训练、权重同步 |
| 推理/serving | 用已有权重响应请求 | 输入处理、逐步生成、调度、缓存与数据搬运 |

**LoRA 是参数高效更新方式，SFT 是训练目标/数据方式，二者不在同一分类层级。** RL 的环境工程负责提供任务、状态转移、奖励和可核验结果；RL systems 还需解决样本生成、训练吞吐、资源调度等。你对 Agent 的兴趣可以连接到推理和训练基础设施，不必今晚同时学会 PPO/GRPO。

读 Muon 前，先懂这几个词：

- **损失 loss**：当前预测与训练目标之间的误差度量；训练损失下降不保证所有下游能力都提高。
- **梯度 gradient**：损失对参数的局部变化率；简单梯度下降沿负梯度更新。
- **学习率 learning rate**：控制每次更新的步幅。玩具例子 `loss=(w-3)^2`，当 `w=1` 时梯度为 `-4`，学习率 `0.1` 给出 `w_new=1-0.1×(-4)=1.4`。
- **动量 momentum**：结合近期梯度形成更平滑的方向，不只用当前一批的梯度。
- **权重衰减 weight decay**：在更新中给参数一个收缩项；与减小学习率不同。
- **AdamW 与 Muon**：都是决定怎样利用梯度更新参数的优化器；两者的归一化、尺度处理与适用参数不同。不要把优化器理解成模型结构。

## 3. 三种加速，三种不同的证据

下面数字是教学算例：

| 方法 | 之前 | 之后 | 解释 |
|---|---:|---:|---|
| 每步更快 | 1,000 步 × 1 秒 | 1,000 × 0.8 秒 | 达标时间少 20% |
| 更少步达标 | 1,000 步 × 1 秒 | 700 × 1.1 秒 | 单步反而慢 10%，总时间仍少 23% |
| 停掉无收益尾部 | 计划 2,000 步 | 900 步停止 | 相对计划少 55%；若正常早停在 950 步，新增节省仅 50/950≈5.3% |

`time-to-quality ≈ 达标步数 × 平均每步时间 + 验证/保存/监控等开销`

这是你理解公司业务和 Muon 论文的主线。[MLPerf Training](https://mlcommons.org/benchmarks/training/)同样以达到质量目标所需的时间为关键测量。

GPU-hours = 各 GPU 占用时长之和。8 张卡训练 2 小时为 16 卡时，不等于 2 卡时。卡时不直接等于能耗或费用，后两者还需功率、硬件、计费方式等。

## 4. Prefill、decode、KV cache

输入 1,000 token，输出 100 token：prefill 处理输入并产生首个输出所需的 logits；随后逐次 decode，依赖此前输出。此例通常对应 prefill 后再做 99 次 decode 才得到共 100 个输出 token。

KV cache 保存各层历史 token 的 K/V。因果结构下，旧位置不会因新 token 到来而改变所能看到的过去，所以其状态可以复用。下一步只需新位置的 Q 去查询历史 K/V，不需要保留旧 Q。

缓存仍要读取过去的 K/V 做 attention；它不是把后续推理变成“完全不用计算”，也没有直接缓存最终回答。

例：完整前缀重算会重复计算旧 token 的 embedding 后各层变换；缓存路径只计算新增位置，再读历史状态。两条路径输出应在合理浮点误差内一致。机制见 [Hugging Face 缓存解释](https://huggingface.co/docs/transformers/cache_explanation)。

对于使用标准 K/V 缓存、保留全部历史 token 的 MHA（普通多头注意力）或 GQA（分组查询注意力）模型，缓存占用可以这样估算。“保留哪些位置”和“多少个 Q 头共享 K/V”是两个不同问题，GQA 也可以看全部历史：

`bytes = 2 × layers × batch × cached_tokens × KV_heads × head_dim × bytes_per_element`

假设 32 层、8 个 KV 头、128 维、BF16（2 字节）：

`2 × 32 × 8 × 128 × 2 = 131072 bytes/token = 128 KiB/token`

一条 8,192-token 序列约 1 GiB；8 条约 8 GiB，尚未计模型权重、激活、allocator 等。公式不能不加修改地套到 MLA、滑窗、混合架构或有共享前缀的物理分配。Query 头数不一定等于 KV 头数。

**注意实验计数**：本项目用预先固定的输入 token，在 prefill 后额外执行 32 个增量前向，因此核对 33 组 logits。这是机制实验，不是模拟一段真实采样出的 32-token 回答。

## 5. 不能混为一谈的三个名字

| 名字 | 主要解决什么 | 不应误解成什么 |
|---|---|---|
| KV cache | 一次自回归过程复用历史 K/V | 答案缓存 |
| Prefix caching | 多次请求的相同 token 前缀复用已有 KV | 语义相似即可命中，或减少新答案的 decode |
| FlashAttention | 将 attention 计算分块并组织在快存储中，减少慢存储访问 | 将全 attention 的理论复杂度直接变为线性，或替代 KV cache |

例如，多人询问同一篇 8,000-token 文档，把共享文档放在各自问题之前，可能获得可复用前缀；缓存被驱逐、前缀不同或长答案主导时，收益降低。[vLLM APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

FlashAttention 实现相同的 attention 运算语义，但浮点实现不保证逐位相同。原始论文解释了 IO 与分块动机：[FlashAttention](https://arxiv.org/abs/2205.14135)。FA4 是面向新硬件继续优化的后续工作。

## 6. 延迟、吞吐、显存不是同一件事

假设收到首 token 用 0.6 秒，后续 99 个 token 又用 3.96 秒：

- TTFT = 0.6 秒，可能包含排队、网络、prefill 等，取决于客户端或服务端测量位置。
- 请求级 TPOT = 3.96/99 = 40 毫秒。
- 总时长约 4.56 秒。
- ITL 指相邻输出的间隔；流式事件可能含多个 token，不要未经处理就把事件间隔当逐 token 延迟。

吞吐是固定时间窗口内的处理总量。10 秒完成 100 个各含 100 个输出 token 的请求，输出吞吐为 1,000 token/s；不意味着每个用户能看到每秒 1,000 token。

p95 为第 95 百分位。均值很低仍可能有明显尾部延迟。Goodput 通常指满足预设质量或延迟目标的有效处理量，必须先写清达标规则。

[vLLM metrics](https://docs.vllm.ai/en/latest/design/metrics/)给出相关指标；不同工具的具体统计口径仍需核对。

## 7. 框架解决了哪些问题

PyTorch 负责张量计算和自动求导等基础能力；vLLM/SGLang 在 serving 中组织模型执行、请求调度、缓存管理、并行和优化内核等。知道名字不等于具有部署和调优经验。

Continuous batching：A 要生成 20 token、B 要 200 token；A 完成后可以补入 C，减少等整批结束的浪费。

Chunked prefill：把长输入分块调度，避免它长时间阻塞正在 decode 的请求；配置影响 TTFT、输出间隔与吞吐，需针对负载测量。

PagedAttention 的重要动机是灵活管理 KV 存储、减少浪费；不能仅凭名称解释成“所有矩阵乘法都更快”。

延伸只看官方页面：[vLLM 调优](https://docs.vllm.ai/en/latest/configuration/optimization/)、[SGLang](https://docs.sglang.ai/)。今晚无需同时安装两个框架。

## 8. 访存、计算、通信，以及量化

GPU 同时要计算、从 HBM 搬数据，并可能跨卡通信。较长 prefill 常偏计算密集，低并发 decode 常受访存影响；模型、并发、长度改变以后，瓶颈也会改变。GPU 利用率高不足以单独定位问题。

8B 参数以 BF16 存裸权重约 16 GB，以 4 bit 存约 4 GB；真实占用还包含元数据、KV、激活和临时工作区。不能据裸权重估算保证 16 GB 显卡能顺利运行。

权重量化和 KV 量化是不同对象。量化是否加速取决于支持的内核、硬件、工作负载和转换开销，还要比较模型质量。[HF 量化概览](https://huggingface.co/docs/transformers/quantization/overview)

多卡并不自动更快：tensor parallel 能分担模型，但引入通信；模型能放进单卡时，多副本服务不同请求有时更合适。先认清权衡，再学 NCCL 实现。

## 9. 明天能说出的公平实验原则

1. 先固定问题：模型、硬件、dtype、输入长度、输出长度、batch/并发，以及是否冷缓存。
2. 先验正确性，再量速度；加速但输出/质量不符不能算成功。
3. CUDA 调用是异步的，需要正确同步或使用 CUDA events；先 warmup，再多次测量。
4. 分开 prefill、decode、端到端；展示原始重复结果，避免只挑最快一次。
5. 对照应该是合理的实现。教学“全前缀重算”不代表真实 serving 基线。
6. 训练要固定质量目标和验证频率；选择 checkpoint/早停用验证集，冻结决策后用独立测试集确认。

Patience 往往指连续多少次验证检查没有足够改善；改变验证频率会改变行为，不必然是 epoch 数。[Lightning early stopping](https://lightning.ai/docs/pytorch/stable/common/early_stopping.html)

## 10. Linux/Git 今晚只学够用的部分

Linux：`pwd` 看目录，`ls` 看文件，`cd` 切目录，`nvidia-smi` 看 GPU 状态，`python script.py --help` 看参数。日志先看首个报错和上下文，不先乱改环境。

Git：`status` 看状态，`diff` 看变化，`add <明确文件>` 选择提交内容，`commit` 保存快照，`push` 发送到指定远端。会执行这些动作不等于已熟悉复杂协作。

本机是 Windows，命令示例用 PowerShell 的 `& '解释器路径' '脚本路径'`，不要生搬 Linux 的 `export` 或反斜杠续行。实验使用已有 PyTorch 环境，无需今晚重装 CUDA。
