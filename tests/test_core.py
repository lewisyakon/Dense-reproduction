from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from dense_kv.adapter import DenseKVAdapter, HeadwiseMLP
from dense_kv.answers import (
    extract_arc_answer,
    extract_gsm8k_answer,
    extract_last_boxed,
)
from dense_kv.cache import cache_shape, cache_to_tensors
from dense_kv.checkpoint import load_adapter_checkpoint, save_adapter_checkpoint
from dense_kv.config import (
    AdapterSpec,
    ExperimentConfig,
    ModelSpec,
    PathSpec,
    TrainingSpec,
    load_config,
    monotonic_layer_mapping,
)
from dense_kv.losses import reconstruction_loss
from dense_kv.rope import apply_rope
from dense_kv.runtime import adapter_parameter_count


def model_spec(name: str, layers: int = 2, heads: int = 2, dim: int = 8) -> ModelSpec:
    return ModelSpec(
        model_id=name,
        path=name,
        num_layers=layers,
        num_kv_heads=heads,
        head_dim=dim,
        rope_theta=10000.0,
        max_position_embeddings=128,
        parameter_count=1000,
    )


def experiment() -> ExperimentConfig:
    return ExperimentConfig(
        direction="small_to_small",
        sender=model_spec("sender"),
        receiver=model_spec("receiver"),
        adapter=AdapterSpec(expansion=2, gate_init=0.5, dtype="float32"),
        training=TrainingSpec(
            sender_device="cpu", receiver_device="cpu", adapter_device="cpu"
        ),
        paths=PathSpec("train", "traces", "eval", "out", "p1.pt", "p2.pt"),
    )


class ConfigTests(unittest.TestCase):
    def test_monotonic_mapping(self) -> None:
        self.assertEqual(monotonic_layer_mapping(10, 4), [0, 3, 6, 9])
        self.assertEqual(monotonic_layer_mapping(3, 5), [0, 1, 1, 2, 2])
        self.assertEqual(monotonic_layer_mapping(1, 3), [0, 0, 0])

    def test_real_adapter_parameter_formula(self) -> None:
        cfg = load_config("configs/4b_to_8b.json")
        self.assertEqual(adapter_parameter_count(cfg), 303_243_552)
        self.assertEqual(cfg.sender.parameter_count, 4_022_468_096)
        self.assertEqual(cfg.receiver.parameter_count, 8_190_735_360)
        self.assertEqual(
            cfg.receiver.revision, "b968826d9c46dd6066d109eabc6255188de91218"
        )


class RopeTests(unittest.TestCase):
    def test_round_trip_with_batched_positions(self) -> None:
        torch.manual_seed(2)
        values = torch.randn(2, 3, 7, 8)
        positions = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6], [0, 0, 1, 2, 3, 4, 5]]
        )
        rotated = apply_rope(values, positions, 1_000_000.0)
        restored = apply_rope(rotated, positions, 1_000_000.0, inverse=True)
        torch.testing.assert_close(restored, values, rtol=1e-5, atol=1e-5)


class AdapterTests(unittest.TestCase):
    def test_shapes_gate_and_gradients(self) -> None:
        torch.manual_seed(3)
        adapter = DenseKVAdapter(
            model_spec("sender", layers=2),
            model_spec("receiver", layers=3),
            expansion=2,
            gate_init=0.25,
        )
        cache = [
            (torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8))
            for _ in range(2)
        ]
        output = adapter(cache, torch.arange(5).unsqueeze(0))
        self.assertEqual(cache_shape(output), (3, 1, 2, 5, 8))
        torch.testing.assert_close(
            adapter.gate_values(), torch.full((3, 2), 0.25)
        )
        loss = sum(key.square().mean() + value.square().mean() for key, value in output)
        loss.backward()
        self.assertIsNotNone(adapter.transforms[0].key_mlp.weight1.grad)
        self.assertIsNotNone(adapter.transforms[0].value_mlp.weight1.grad)
        self.assertIsNotNone(adapter.transforms[0].gate_logits.grad)

    def test_headwise_mlp_does_not_mix_heads(self) -> None:
        layer = HeadwiseMLP(2, 4, 6, 4)
        with torch.no_grad():
            layer.bias1.zero_()
            layer.bias2.zero_()
        values = torch.zeros(1, 2, 1, 4)
        values[:, 0] = 1.0
        output = layer(values)
        torch.testing.assert_close(output[:, 1], torch.zeros_like(output[:, 1]))

    def test_masked_reconstruction_loss(self) -> None:
        target = [(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4))]
        projected = [(torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3, 4) * 2)]
        loss, metrics = reconstruction_loss(
            projected, target, torch.tensor([[1, 1, 0]])
        )
        self.assertAlmostEqual(loss.item(), 5.0)
        self.assertAlmostEqual(metrics["key_mse"], 1.0)
        self.assertAlmostEqual(metrics["value_mse"], 4.0)

    def test_checkpoint_round_trip(self) -> None:
        cfg = experiment()
        source = DenseKVAdapter(cfg.sender, cfg.receiver, expansion=2)
        destination = DenseKVAdapter(cfg.sender, cfg.receiver, expansion=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase1_small_to_small.pt"
            save_adapter_checkpoint(path, source, cfg, phase=1, step=7)
            state = load_adapter_checkpoint(
                path, destination, expected_direction=cfg.direction
            )
        self.assertEqual(state["step"], 7)
        for source_value, destination_value in zip(
            source.state_dict().values(), destination.state_dict().values()
        ):
            torch.testing.assert_close(source_value, destination_value)


class CacheAndAnswerTests(unittest.TestCase):
    def test_legacy_cache_conversion(self) -> None:
        layers = tuple(
            (torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))
            for _ in range(2)
        )
        converted = cache_to_tensors(layers)
        self.assertEqual(cache_shape(converted), (2, 1, 2, 3, 4))

    def test_answer_extractors(self) -> None:
        self.assertEqual(extract_gsm8k_answer("work... #### 1,024"), "1024")
        self.assertEqual(extract_gsm8k_answer("answer is -2.50"), "-2.5")
        self.assertEqual(extract_last_boxed(r"x then \boxed{\frac{3}{4}}"), r"\frac{3}{4}")
        self.assertEqual(extract_arc_answer("Therefore, the answer is (C)."), "C")
        self.assertEqual(extract_arc_answer("The final answer is 3."), "3")


if __name__ == "__main__":
    unittest.main()
