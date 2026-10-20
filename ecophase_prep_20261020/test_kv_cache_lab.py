"""Small CPU checks for the two easy-to-miss incremental attention bugs."""

import unittest

import torch

from kv_cache_lab import TinyDecoder, check_equivalence


class KVCacheCorrectnessTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = TinyDecoder(
            vocab=31, width=32, layers=2, heads=4, max_length=32
        ).eval()

    def test_matches_full_prefix_at_every_position(self):
        # Nonzero positions and multiple requests catch mask/offset mistakes.
        tokens = torch.randint(31, (2, 13))
        check = check_equivalence(self.model, tokens, prompt_length=5, decode_steps=8)
        self.assertTrue(check["allclose"], check)
        self.assertEqual(check["positions_checked"], 9)

    @torch.inference_mode()
    def test_cache_resets_on_prefill_and_never_reads_unwritten_slots(self):
        cache = self.model.make_cache(2, 24)
        tokens = torch.randint(31, (2, 10))
        for k, v in cache:
            k.fill_(float("nan"))
            v.fill_(float("nan"))
        self.model(tokens[:, :7], cache)
        self.model(tokens[:, :3], cache)  # Start a shorter new request.
        actual = self.model(tokens[:, 3:4], cache, start_pos=3)
        expected = self.model(tokens[:, :4])
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    @torch.inference_mode()
    def test_unsupported_chunk_decode_is_rejected(self):
        cache = self.model.make_cache(1, 16)
        tokens = torch.randint(31, (1, 8))
        self.model(tokens[:, :5], cache)
        with self.assertRaises(ValueError):
            self.model(tokens[:, 5:8], cache, start_pos=5)


if __name__ == "__main__":
    unittest.main()
