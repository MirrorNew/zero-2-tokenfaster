"""第 2 步：沿用五维 Attention 示例，添加残差连接。

教学参数未经训练；本步输出投影为单位矩阵，尚未添加归一化等模块。
"""

import torch


def main():
    torch.set_printoptions(precision=4, sci_mode=False)

    # 与第 1 步完全相同的输入和 Q/K/V 参数。
    X = torch.tensor([
        [1., 0., 1., 0., 1.],
        [0., 1., 1., 0., 1.],
        [1., 1., 0., 1., 0.],
    ])
    W_Q = torch.eye(5)
    W_K = torch.diag(torch.tensor([1., 2., 1., 1., 1.]))
    W_V = torch.diag(torch.tensor([1., 2., 1., 2., 1.]))
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V

    scores = (Q @ K.T) / (K.shape[-1] ** 0.5)
    future = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    weights = torch.softmax(scores.masked_fill(future, float('-inf')), dim=-1)
    output = weights @ V

    # 完整 Attention 还有输出投影；本步用单位矩阵保持数字不变。
    W_O = torch.eye(5)
    attention_update = output @ W_O

    # 本节新增：逐元素相加，同一个位置的同一个维度相加。
    H = X + attention_update

    print(f'X:\n{X}\n')
    print(f'Attention update:\n{attention_update}\n')
    print(f'H = X + Attention update:\n{H}')


if __name__ == '__main__':
    main()
