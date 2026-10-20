# 论文检索附录：原始候选与两篇精选

检索日期 2026-10-20；采用 paper-search 技能，调用本地多源脚本。查询 `LLM training inference acceleration Muon FlashAttention`，年份 2024–2026，截止 2026-10-20，每源最多 3 条。原始输出在 [paper_search_raw.txt](paper_search_raw.txt)。

**这是一次辅助候选检索，不是领域规模调查。** 查询词混合且每源只取 3 条，会遗漏重要工作，甚至召回物理学的 muon。主选论文另经题名检索、论文全文和作者来源核对。按本次用户“1–2 篇吃透”的要求，主学习路径只保留 Muon 和 FA4。

## 各源完整返回

以下引用数只是 API 当次返回值，没有用于排序推荐；0 可能是源未提供。`未知` 表示原始元数据没有可用年份，不能认定符合时间窗。

| 来源 | 标题 | 年份 | API 引用数 | 链接 |
|---|---|---:|---:|---|
| Semantic Scholar | EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test | 2025 | 382 | [记录](https://www.semanticscholar.org/paper/781bde4935c54635efc52c12a449fb4bfe2f2685) |
| Semantic Scholar | Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding | 2025 | 458 | [记录](https://www.semanticscholar.org/paper/5e9ffdd179df49be1129e32ae75fc89a2b68e676) |
| Semantic Scholar | SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration | 2024 | 84 | [记录](https://www.semanticscholar.org/paper/bacdf9671fb872287201b53d768df89b4d6630a3) |
| OpenAlex | Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads | 2024 | 27 | [记录](https://doi.org/10.48550/arxiv.2401.10774) |
| OpenAlex | AiF: Accelerating On-Device LLM Inference Using In-Flash Processing | 2025 | 15 | [记录](https://doi.org/10.1145/3695053.3731073) |
| OpenAlex | Lincoln: Real-Time 50~100B LLM Inference on Consumer Devices with LPDDR-Interfaced, Compute-Enabled Flash Memory | 2025 | 11 | [记录](https://doi.org/10.1109/hpca61900.2025.00128) |
| arXiv | YouZhi: Towards High-Concurrency Financial LLMs via Adaptive GQA-to-MLA Transition | 2026 | 0 | [记录](https://arxiv.org/abs/2606.05868v1) |
| arXiv | FlashAR: Efficient Post-Training Acceleration for Autoregressive Image Generation | 2026 | 0 | [记录](https://arxiv.org/abs/2605.09430v3) |
| arXiv | LLMs as High-Dimensional Nonlinear Autoregressive Models with Attention: Training, Alignment and Inference | 2026 | 0 | [记录](https://arxiv.org/abs/2602.00426v1) |
| Crossref | HybridCrypt-LLM: Lightweight Privacy for LLM Training and Inference | 未知 | 0 | [记录](https://doi.org/10.2139/ssrn.6017945) |
| Crossref | Towards a Universal Scaling Law of LLM Training and Inference | 未知 | 0 | [记录](https://doi.org/10.14293/pr2199.001074.v1) |
| Crossref | Rapid-Cycling HTS-Based Magnets for Muon Acceleration | 2024 | 0 | [记录](https://doi.org/10.2172/2283684) |

OpenReview：0 条。DBLP：报错后 0 条。Model knowledge：未额外添加未经在线核实的记忆条目。

脚本原样错误信息：`[dblp] Error: Expecting value: line 1 column 1 (char 0)`。另出现多次 `Rate limited. Waiting 3 seconds...`，脚本自行重试后结束。0 条不应解释成对应库没有相关论文。

## 选择与解释

共返回 12 条原始记录，其中 1 条明显是物理学主题，2 条缺年份。其余只是候选，未在本附录逐篇审读。题名涉及 speculative decoding、diffusion、存储/硬件、KV/架构变化等，但这个截断样本不能说明领域真实热度、年度趋势或职业需求频率，因此不制作伪精确的趋势与引用榜。

最终精读来自额外的 primary-source 核验：

1. [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)，2025-02-24。学习训练效率与收敛，并注意预训练结果到微调的适用边界。
2. [FlashAttention-4](https://arxiv.org/abs/2603.05451v1)，2026-03-05。学习硬件变化、Attention 内核瓶颈与算子测量边界。

选择标准是公司业务相关性、基础学习收益和 8 小时可讲清程度，而非引用数或“最新”二字。
