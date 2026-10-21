"""验证单头 Attention、完整网络因果性、反传和生成边界。"""

import unittest

import torch
from torch.nn import functional as F

from transformer_single_head import SingleHeadAttention, TinyTransformer, TOKEN_TO_ID, VOCAB


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)

    def test_attention_matches_pytorch_reference(self):
        attention = SingleHeadAttention()
        x = torch.randn(2, 4, 5)
        reference = F.scaled_dot_product_attention(
            attention.q_proj(x), attention.k_proj(x), attention.v_proj(x),
            is_causal=True,
        )
        torch.testing.assert_close(attention(x), attention.out_proj(reference))

    def test_future_tokens_do_not_change_prefix_logits(self):
        model = TinyTransformer().eval()
        tokens = torch.tensor([[0, 1, 2, 5], [3, 1, 4, 5]])
        changed = tokens.clone()
        changed[:, 2:] = torch.tensor([[4, 3], [2, 0]])
        with torch.no_grad():
            original = model(tokens)[:, :2]
            torch.testing.assert_close(original, model(changed)[:, :2])
            torch.testing.assert_close(original, model(tokens[:, :2]))

    def test_next_token_loss_reaches_all_parameters(self):
        model = TinyTransformer()
        tokens = torch.tensor([[0, 1, 2, 5], [3, 1, 4, 5]])
        logits = model(tokens[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, len(VOCAB)), tokens[:, 1:].reshape(-1))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0, name)

    def test_generation_stops_at_eos_and_context_limit(self):
        model = TinyTransformer(max_len=4).eval()
        prompt = torch.tensor([[0, 1]])
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.lm_head.bias.zero_()
            model.lm_head.bias[TOKEN_TO_ID['<eos>']] = 10.0
        torch.testing.assert_close(model.generate(prompt), torch.tensor([[0, 1, 5]]))
        torch.testing.assert_close(model.generate(torch.tensor([[0, 5]])), torch.tensor([[0, 5]]))
        torch.testing.assert_close(model.generate(prompt, max_new_tokens=0), prompt)

        with torch.no_grad():
            model.lm_head.bias.zero_()
            model.lm_head.bias[2] = 10.0
        torch.testing.assert_close(model.generate(prompt, max_new_tokens=1), torch.tensor([[0, 1, 2]]))
        torch.testing.assert_close(model.generate(prompt, max_new_tokens=20), torch.tensor([[0, 1, 2, 2]]))
        with self.assertRaises(ValueError):
            model(torch.tensor([[0, 1, 2, 3, 4]]))
        with self.assertRaises(ValueError):
            model.generate(torch.empty((1, 0), dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
