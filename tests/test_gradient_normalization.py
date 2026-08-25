from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from adversarial_harness.attacks import TargetedPGD
from adversarial_harness.config import AttackConfig


class _FakeSurrogate:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.prompts = {
            "object": SimpleNamespace(
                normal_embeddings=torch.tensor([[1.0, 0.0]]),
                abnormal_embeddings=torch.tensor([[0.0, 1.0]]),
            )
        }


def _attacker(normalization: str) -> TargetedPGD:
    return TargetedPGD(
        _FakeSurrogate(),
        AttackConfig(
            image_size=8,
            epsilon=4 / 255,
            step_size=1 / 255,
            steps=1,
            universal_steps=1,
            global_weight=0.2,
            local_weight=0.8,
            gradient_normalization=normalization,
        ),
    )


class GradientNormalizationConfigTests(unittest.TestCase):
    def test_default_is_disabled(self) -> None:
        self.assertEqual(AttackConfig().gradient_normalization, "none")

    def test_unknown_value_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AttackConfig(gradient_normalization="minmax")


class CombineGradientsTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        # Deliberately mismatched magnitudes: this is the imbalance the feature exists to remove.
        self.global_gradient = torch.randn(1, 3, 8, 8) * 1e-3
        self.local_gradient = torch.randn(1, 3, 8, 8) * 1e3

    def test_disabled_reproduces_the_plain_weighted_sum(self) -> None:
        attacker = _attacker("none")
        expected = 0.2 * self.global_gradient + 0.8 * self.local_gradient
        combined = attacker._combine_gradients(self.global_gradient, self.local_gradient)
        self.assertTrue(torch.allclose(combined, expected))

    def test_l2_makes_the_realized_ratio_equal_the_configured_ratio(self) -> None:
        attacker = _attacker("l2")
        scaled_global = (
            0.2 * self.global_gradient / self.global_gradient.norm()
        )
        scaled_local = 0.8 * self.local_gradient / self.local_gradient.norm()

        self.assertAlmostEqual(float(scaled_global.norm()), 0.2, places=6)
        self.assertAlmostEqual(float(scaled_local.norm()), 0.8, places=6)
        self.assertAlmostEqual(
            float(scaled_local.norm() / scaled_global.norm()), 4.0, places=5
        )

        combined = attacker._combine_gradients(self.global_gradient, self.local_gradient)
        self.assertTrue(torch.allclose(combined, scaled_global + scaled_local, atol=1e-6))

    def test_l2_is_invariant_to_component_rescaling(self) -> None:
        attacker = _attacker("l2")
        baseline = attacker._combine_gradients(self.global_gradient, self.local_gradient)
        rescaled = attacker._combine_gradients(
            self.global_gradient * 1e6, self.local_gradient * 1e-6
        )
        self.assertTrue(torch.allclose(baseline, rescaled, atol=1e-6))

    def test_zero_gradient_does_not_produce_nan(self) -> None:
        attacker = _attacker("l2")
        combined = attacker._combine_gradients(
            torch.zeros_like(self.global_gradient), self.local_gradient
        )
        self.assertTrue(torch.isfinite(combined).all())


if __name__ == "__main__":
    unittest.main()
