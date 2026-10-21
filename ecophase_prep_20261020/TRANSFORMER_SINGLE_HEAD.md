# 完整的五维单头 Transformer

代码：[transformer_single_head.py](transformer_single_head.py)。仅依赖 PyTorch，默认在 CPU 上运行。

这是用于逐词预测的 decoder-only Transformer。使用单头 Attention、两层 Post-LN block、5 维隐藏表示、20 维 FFN、可学习位置嵌入和 6 词词表。没有调用 `nn.Transformer`、`nn.MultiheadAttention` 或封装的 Attention 来实现模型。

```text
token_ids [B,T]
  -> 词嵌入 + 位置嵌入 [B,T,5]
  -> 重复两层：
       Q/K/V -> 因果 Attention -> 输出投影
       残差相加 -> LayerNorm
       Linear(5,20) -> GELU -> Linear(20,5)
       残差相加 -> LayerNorm
  -> Linear(5,6)
  -> logits [B,T,6]
```

`nn.Linear` 内部使用 `x @ weight.T + bias`。模型中的投影均可训练；前两课用于手算的固定 Q/K/V 矩阵不再沿用。词嵌入中的“我、喜欢、猫”仍以旧课的三组五维数字初始化，但位置嵌入及后续训练会改变实际计算结果。

从仓库根目录运行：

```powershell
python ecophase_prep_20261020/transformer_single_head.py
python -m unittest discover -s ecophase_prep_20261020 -p test_transformer_single_head.py -v
```

脚本固定随机种子，在两个短句上训练 600 次并打印初始/最终损失，以及从“我”和“你”开始的贪心生成。训练输入是“我 喜欢 猫”，标签是“喜欢 猫 <eos>”；另一句对应“你 喜欢 狗”。输出 logits 直接送入交叉熵，不预先做 softmax。

生成使用最后一个位置的 logits，每次追加一个词；遇到 `<eos>`、生成数量上限或 16 个位置的上下文上限即停止。此教学版本每步重算前缀，尚未添加 KV 缓存。

测试包括 Attention 与 PyTorch 参考实现的一致性、未来词替换/前缀截断后前缀 logits 不变、全部参数的有限非零梯度，以及 EOS 和长度边界。

本机 CPU 实跑（PyTorch 2.8.0.dev20250505+cu128，种子 7）：836 个参数，600 次更新后损失从 1.616686 降至 0.000321，6 个训练位置的词预测准确率为 100%，生成如下；四项测试全部通过。不同 PyTorch 版本或硬件的浮点末位可能不同。

```text
我 喜欢 猫 <eos>
你 喜欢 狗 <eos>
```

两个短句用于验证训练与生成链路。即使训练词预测全部正确，也只说明拟合了这两个训练样本，不是泛化能力或推理性能评测。
