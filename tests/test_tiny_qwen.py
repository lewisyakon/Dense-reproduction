from __future__ import annotations

import unittest

import torch

from dense_kv.adapter import DenseKVAdapter
from dense_kv.cache import cache_shape, tensors_to_dynamic_cache
from dense_kv.config import ModelSpec
from dense_kv.runtime import extract_cache, manual_generate

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers is not installed")
class TinyQwenIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        config = Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            rope_theta=10000.0,
            attention_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            use_cache=True,
        )
        self.sender = Qwen3ForCausalLM(config).eval()
        self.receiver = Qwen3ForCausalLM(config).eval()
        self.spec = ModelSpec(
            model_id="tiny-qwen3",
            path="unused",
            num_layers=2,
            num_kv_heads=2,
            head_dim=8,
            rope_theta=10000.0,
            max_position_embeddings=128,
            parameter_count=sum(p.numel() for p in self.sender.parameters()),
        )
        self.input_ids = torch.tensor([[1, 5, 7, 9, 2]])
        self.attention_mask = torch.ones_like(self.input_ids)
        self.position_ids = torch.arange(5).unsqueeze(0)

    def test_extract_transform_and_inject(self) -> None:
        source = extract_cache(
            self.sender, self.input_ids, self.attention_mask, self.position_ids
        )
        self.assertEqual(cache_shape(source), (2, 1, 2, 5, 8))
        adapter = DenseKVAdapter(self.spec, self.spec, expansion=2)
        transformed = adapter(source, self.position_ids)
        generated, usage = manual_generate(
            self.receiver,
            [1],
            past_layers=transformed,
            max_new_tokens=2,
            eos_token_ids=set(),
        )
        self.assertEqual(len(generated), 2)
        self.assertEqual(usage, {"prefill_tokens": 1, "decode_tokens": 2})

    def test_rope_matches_qwen3_implementation(self) -> None:
        from dense_kv.rope import apply_rope

        values = torch.randn(2, 3, 5, 8)
        positions = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 2, 3]])
        inverse_frequency = 1.0 / (
            10000.0 ** (torch.arange(0, 8, 2).float() / 8)
        )
        frequencies = positions.float().unsqueeze(-1) * inverse_frequency
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        expected = apply_rotary_pos_emb(
            values, values, embedding.cos(), embedding.sin()
        )[1]
        actual = apply_rope(values, positions, 10000.0)
        torch.testing.assert_close(actual, expected)

    def test_generation_loss_backpropagates_to_adapter(self) -> None:
        for parameter in self.receiver.parameters():
            parameter.requires_grad_(False)
        source = extract_cache(
            self.sender, self.input_ids, self.attention_mask, self.position_ids
        )
        adapter = DenseKVAdapter(self.spec, self.spec, expansion=2)
        transformed = adapter(source, self.position_ids)
        past = tensors_to_dynamic_cache(transformed)
        new_ids = torch.tensor([[1, 13, 17]])
        labels = torch.tensor([[-100, 13, 17]])
        past_length = past.get_seq_length()
        positions = torch.arange(past_length, past_length + new_ids.shape[1])
        outputs = self.receiver(
            input_ids=new_ids,
            labels=labels,
            attention_mask=torch.ones(1, past_length + new_ids.shape[1], dtype=torch.long),
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            past_key_values=past,
            use_cache=False,
            return_dict=True,
        )
        outputs.loss.backward()
        self.assertTrue(torch.isfinite(outputs.loss))
        self.assertIsNotNone(adapter.transforms[0].key_mlp.weight1.grad)
        self.assertGreater(adapter.transforms[0].key_mlp.weight1.grad.abs().sum(), 0)
        self.assertIsNone(next(self.receiver.parameters()).grad)


if __name__ == "__main__":
    unittest.main()
