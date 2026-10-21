"""完整的五维单头自回归 Transformer：前向、训练与生成。

仅依赖 PyTorch，在 CPU 上运行。两个短句用于验证训练链路，不代表语言泛化能力。
"""

import torch
from torch import nn
from torch.nn import functional as F


VOCAB = ('我', '喜欢', '猫', '你', '狗', '<eos>')
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model=5):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x、q、k、v: [batch, sequence, 5]，不拆分多头。
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        length = x.shape[1]
        future = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(future, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return self.out_proj(weights @ v)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=5):
        super().__init__()
        self.attention = SingleHeadAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # 5 -> 20
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),  # 20 -> 5
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Post-LN：先残差相加，再归一化；FFN 对每个位置分别作用。
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ffn(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=len(VOCAB), d_model=5, n_layers=2, max_len=16):
        super().__init__()
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 可学习的位置嵌入，也适用于这里的奇数维度 5。
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        if token_ids.ndim != 2:
            raise ValueError('token_ids must have shape [batch, sequence]')
        length = token_ids.shape[1]
        if not 1 <= length <= self.max_len:
            raise ValueError(f'sequence length must be between 1 and {self.max_len}')

        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        # 返回 logits，不做 softmax；交叉熵内部会处理归一化。
        return self.lm_head(x)  # [batch, sequence, vocab_size]

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens=8, eos_id=TOKEN_TO_ID['<eos>']):
        """单条非空提示的贪心生成；每步重新计算当前前缀，不使用 KV 缓存。"""
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError('generate expects one prompt with shape [1, sequence]')
        if not 1 <= token_ids.shape[1] <= self.max_len:
            raise ValueError('prompt length is outside the context window')
        if max_new_tokens < 0:
            raise ValueError('max_new_tokens must be nonnegative')
        for _ in range(max_new_tokens):
            if token_ids.shape[1] >= self.max_len or token_ids[0, -1].item() == eos_id:
                break
            logits = self(token_ids)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_ids = torch.cat([token_ids, next_id], dim=1)
        return token_ids


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = TinyTransformer()

    # 沿用前两课的三组数字初始化词嵌入，之后它们也参与训练。
    # 注意：位置嵌入和 Q/K/V 参数已变化，不会复现前两课的 Attention 数值。
    with torch.no_grad():
        model.token_embedding.weight[:3].copy_(torch.tensor([
            [1., 0., 1., 0., 1.],  # 我
            [0., 1., 1., 0., 1.],  # 喜欢
            [1., 1., 0., 1., 0.],  # 猫
        ]))

    sentences = [('我', '喜欢', '猫', '<eos>'), ('你', '喜欢', '狗', '<eos>')]
    token_ids = torch.tensor([[TOKEN_TO_ID[word] for word in row] for row in sentences])
    inputs = token_ids[:, :-1]   # 我 喜欢 猫 / 你 喜欢 狗
    targets = token_ids[:, 1:]   # 喜欢 猫 eos / 喜欢 狗 eos
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)

    print(f'Parameters: {sum(parameter.numel() for parameter in model.parameters())}')
    model.train()
    with torch.no_grad():
        initial_loss = F.cross_entropy(model(inputs).reshape(-1, len(VOCAB)), targets.reshape(-1))
    print(f'Initial loss: {initial_loss.item():.6f}')

    for step in range(1, 601):
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, len(VOCAB)), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0:
            print(f'Step {step}, loss before update: {loss.detach().item():.6f}')

    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        final_loss = F.cross_entropy(logits.reshape(-1, len(VOCAB)), targets.reshape(-1))
        accuracy = (logits.argmax(dim=-1) == targets).float().mean()
    print(f'Final loss: {final_loss.item():.6f}; training token accuracy: {accuracy.item():.1%}')

    for word in ('我', '你'):
        prompt = torch.tensor([[TOKEN_TO_ID[word]]])
        generated = model.generate(prompt)
        print(' '.join(VOCAB[index] for index in generated[0].tolist()))


if __name__ == '__main__':
    main()
