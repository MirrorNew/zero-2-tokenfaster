"""第 1 步：三个位置、每个位置五维的单头因果 Attention。

只实现 Transformer 中的 Attention 核心计算，还不是完整 Transformer。
输入与参数均为方便手算而指定的教学数字，没有经过训练。
"""

import torch


def main():
    torch.set_printoptions(precision=4, sci_mode=False)

    # 三行：我、喜欢、猫；五列：每个位置的五个特征。
    X = torch.tensor([
        [1., 0., 1., 0., 1.],
        [0., 1., 1., 0., 1.],
        [1., 1., 0., 1., 0.],
    ])

    # 这里选简单矩阵方便手算；实际模型中的矩阵由训练学习。
    W_Q = torch.eye(5)
    W_K = torch.diag(torch.tensor([1., 2., 1., 1., 1.]))
    W_V = torch.diag(torch.tensor([1., 2., 1., 2., 1.]))
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V

    scores = (Q @ K.T) / (K.shape[-1] ** 0.5)

    # True 表示未来位置：在 softmax 之前用负无穷取代分数。
    future = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    masked_scores = scores.masked_fill(future, float('-inf'))
    weights = torch.softmax(masked_scores, dim=-1)
    output = weights @ V

    for name, value in [
        ('X', X), ('Q', Q), ('K', K), ('V', V),
        ('Q @ K.T', Q @ K.T), ('masked scores', masked_scores),
        ('attention weights', weights), ('output', output),
    ]:
        print(f'{name}:\n{value}\n')


if __name__ == '__main__':
    main()
