"""跨块 softmax 第 1 步：用共同分母合并两个局部输出。

沿用最初的五维示例，只算最后一个位置“猫”。
本节用小分数直接取 exp 来说明等价性；尚未实现数值稳定的在线最大值更新，
也不是 GPU 性能实现。完整权重仅在最后重建以核对普通 Attention。
"""

import torch


def main():
    torch.set_printoptions(precision=6, sci_mode=False)
    q = torch.tensor([1., 1., 0., 1., 0.], dtype=torch.float64)
    K = torch.tensor([
        [1., 0., 1., 0., 1.],
        [0., 2., 1., 0., 1.],
        [1., 2., 0., 1., 0.],
    ], dtype=torch.float64)
    V = torch.tensor([
        [1., 0., 1., 0., 1.],
        [0., 2., 1., 0., 1.],
        [1., 2., 0., 2., 0.],
    ], dtype=torch.float64)

    # A 块：“我、喜欢”。Z 是原始分数取指数后的总和，不是 softmax 后的和。
    s_A = (q @ K[:2].T) / (5 ** 0.5)
    Z_A = torch.exp(s_A).sum()
    P_A = torch.softmax(s_A, dim=-1)
    O_A = P_A @ V[:2]

    # B 块：“猫”。只有一个词，所以局部权重为 1。
    s_B = (q @ K[2:].T) / (5 ** 0.5)
    Z_B = torch.exp(s_B).sum()
    P_B = torch.softmax(s_B, dim=-1)
    O_B = P_B @ V[2:]

    # 两块使用相同的指数基准，因此可直接合并 Z，再修正局部输出的占比。
    Z = Z_A + Z_B
    ratio_A, ratio_B = Z_A / Z, Z_B / Z
    O = ratio_A * O_A + ratio_B * O_B
    # 若还有下一块，保留合并后的 O 和 Z，继续执行同样的合并。

    # 以下重建完整权重只用于教学核对；分块计算输出不需要这一步。
    merged_weights = torch.cat([ratio_A * P_A, ratio_B * P_B])
    full_weights = torch.softmax((q @ K.T) / (5 ** 0.5), dim=-1)
    torch.testing.assert_close(merged_weights, full_weights)
    torch.testing.assert_close(O, full_weights @ V)

    for name, value in [
        ('s_A', s_A), ('P_A (local)', P_A), ('Z_A', Z_A), ('O_A', O_A),
        ('s_B', s_B), ('P_B (local)', P_B), ('Z_B', Z_B), ('O_B', O_B),
        ('ratio_A', ratio_A), ('ratio_B', ratio_B),
        ('merged weights', merged_weights), ('merged output', O),
    ]:
        print(f'{name}: {value}')
    print('PASS: merged weights and output match full softmax attention.')


if __name__ == '__main__':
    main()
