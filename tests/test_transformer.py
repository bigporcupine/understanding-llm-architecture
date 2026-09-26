"""Small correctness tests for the educational Transformer components."""

import unittest

import torch

from transformer_tutorial import (
    CausalSelfAttention,
    RMSNorm,
    TinyLM,
    apply_rope,
    causal_attention,
)


class TransformerTests(unittest.TestCase):
    def test_causal_attention_hides_future_positions(self):
        q = k = torch.randn(2, 5, 4)
        v = torch.randn(2, 5, 3)
        output, weights = causal_attention(q, k, v)
        self.assertEqual(output.shape, (2, 5, 3))
        self.assertTrue(torch.all(weights.triu(diagonal=1) == 0))

    def test_rmsnorm_produces_unit_rms(self):
        x = torch.randn(2, 3, 8)
        y = RMSNorm(8)(x)
        rms = y.pow(2).mean(dim=-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-5))

    def test_rope_preserves_norm(self):
        x = torch.randn(2, 3, 5, 8)
        y = apply_rope(x)
        self.assertTrue(torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5))

    def test_multihead_shape(self):
        attention = CausalSelfAttention(d_model=16, num_heads=4)
        x = torch.randn(2, 7, 16)
        self.assertEqual(attention(x).shape, x.shape)

    def test_language_model_shape_and_gradient(self):
        model = TinyLM(vocab_size=11, d_model=16, num_layers=1, num_heads=4, d_ff=32)
        tokens = torch.randint(0, 11, (2, 5))
        logits, loss = model(tokens, tokens)
        self.assertEqual(logits.shape, (2, 5, 11))
        loss.backward()
        self.assertIsNotNone(model.embedding.weight.grad)


if __name__ == "__main__":
    unittest.main()
