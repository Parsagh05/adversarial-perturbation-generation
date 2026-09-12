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


class _DifferentiableFakeSurrogate(_FakeSurrogate):
    def encode_visual(self, images_01, include_patches=True):
        signal = images_01.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
        token = torch.stack((1.0 - signal, signal), dim=-1)
        cls = token[:, None, :]
        patches = token[:, None, :].expand(-1, 4, -1)
        return token, [torch.cat((cls, patches), dim=1)] if include_patches else []


class MaskAwareLocalLossTests(unittest.TestCase):
    def test_defect_mask_and_fixed_normal_region_focus_local_loss(self) -> None:
        attacker = TargetedPGD(
            _FakeSurrogate(),
            AttackConfig(
                temperature=1.0,
                mask_local_loss=True,
                local_background_weight=0.0,
            ),
        )
        global_features = torch.zeros((1, 2))
        # CLS, one anomalous defect token, and three normal background tokens.
        patch_features = [
            torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
        ]
        defect_mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        zero_mask = torch.zeros_like(defect_mask)

        unmasked = attacker._group_losses(
            global_features,
            patch_features,
            ["object"],
            target_label=0,
            mode="local",
        )["local"]
        masked = attacker._group_losses(
            global_features,
            patch_features,
            ["object"],
            target_label=0,
            mode="local",
            spatial_masks=defect_mask,
        )["local"]
        normal_fallback = attacker._group_losses(
            global_features,
            patch_features,
            ["object"],
            target_label=1,
            mode="local",
            spatial_masks=zero_mask,
        )["local"]
        full_image_attacker = TargetedPGD(
            _FakeSurrogate(),
            AttackConfig(
                temperature=1.0,
                mask_local_loss=True,
                local_background_weight=0.0,
                normal_local_target="full_image",
            ),
        )
        normal_full_image = full_image_attacker._group_losses(
            global_features,
            patch_features,
            ["object"],
            target_label=1,
            mode="local",
        )["local"]

        self.assertGreater(float(masked), float(unmasked))
        self.assertNotAlmostEqual(
            float(normal_fallback), float(normal_full_image), places=6
        )

    def test_one_targeted_local_step_reduces_same_batch_loss(self) -> None:
        attacker = TargetedPGD(
            _DifferentiableFakeSurrogate(),
            AttackConfig(
                image_size=2,
                epsilon=0.5,
                step_size=0.05,
                steps=1,
                random_start=False,
                temperature=1.0,
                local_focal_weight=0.5,
                local_dice_weight=0.5,
            ),
        )
        # Stay inside the fake surrogate's clamp interval so this tests the
        # attack gradient rather than PyTorch's boundary derivative for clamp.
        clean = torch.full((1, 3, 2, 2), 0.25)
        before = attacker.objective(clean, ["object"], 1, "local")
        adversarial, _ = attacker.perturb_batch(clean, ["object"], 1, "local")
        after = attacker.objective(adversarial, ["object"], 1, "local")

        self.assertTrue(torch.isfinite(before))
        self.assertTrue(torch.isfinite(after))
        self.assertLess(float(after), float(before))


class MarginTopKLossTests(unittest.TestCase):
    def _attacker(self, **overrides) -> TargetedPGD:
        settings = {"temperature": 1.0, "loss_formulation": "margin_topk"}
        settings.update(overrides)
        return TargetedPGD(_FakeSurrogate(), AttackConfig(**settings))

    def test_direction_sign_maximizes_for_normal_and_minimizes_for_anomalous(
        self,
    ) -> None:
        attacker = self._attacker()
        global_features = torch.tensor([[0.0, 1.0]])
        patch_features = [
            torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
        ]
        to_abnormal = attacker._group_losses(
            global_features, patch_features, ["object"], 1, "combined"
        )
        to_normal = attacker._group_losses(
            global_features, patch_features, ["object"], 0, "combined"
        )
        self.assertAlmostEqual(
            float(to_abnormal["global"]), -float(to_abnormal["global_margin"])
        )
        self.assertAlmostEqual(
            float(to_normal["global"]), float(to_normal["global_margin"])
        )
        self.assertAlmostEqual(
            float(to_abnormal["local"]), -float(to_abnormal["local_topk"])
        )
        self.assertAlmostEqual(
            float(to_normal["local"]), float(to_normal["local_topk"])
        )

    def test_topk_ignores_tokens_below_the_cut(self) -> None:
        # After the CLS token, only the first patch has a positive anomaly
        # margin. With K=1, changing the other patches must not change TopK(H).
        attacker = self._attacker(margin_topk_fraction=0.25)
        global_features = torch.zeros((1, 2))
        first = [
            torch.tensor(
                [[[1.0, 0.0], [0.0, 2.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]
            )
        ]
        second = [
            torch.tensor(
                [[[1.0, 0.0], [0.0, 2.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]]]
            )
        ]
        first_topk = attacker._group_losses(
            global_features, first, ["object"], 1, "local"
        )["local_topk"]
        second_topk = attacker._group_losses(
            global_features, second, ["object"], 1, "local"
        )["local_topk"]
        self.assertAlmostEqual(float(first_topk), float(second_topk))

    def test_ground_truth_masks_do_not_change_relaxed_loss(self) -> None:
        attacker = self._attacker(mask_local_loss=True)
        global_features = torch.zeros((1, 2))
        patches = [
            torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
        ]
        zero_mask = torch.zeros((1, 2, 2))
        one_mask = torch.ones((1, 2, 2))
        without_mask = attacker._group_losses(
            global_features, patches, ["object"], 1, "local"
        )["total"]
        with_zero = attacker._group_losses(
            global_features, patches, ["object"], 1, "local", zero_mask
        )["total"]
        with_one = attacker._group_losses(
            global_features, patches, ["object"], 1, "local", one_mask
        )["total"]
        self.assertAlmostEqual(float(without_mask), float(with_zero))
        self.assertAlmostEqual(float(without_mask), float(with_one))

    def test_one_targeted_step_reduces_relaxed_loss_in_every_mode(self) -> None:
        for mode in ("global", "local", "combined"):
            with self.subTest(mode=mode):
                attacker = TargetedPGD(
                    _DifferentiableFakeSurrogate(),
                    AttackConfig(
                        image_size=2,
                        epsilon=0.5,
                        step_size=0.05,
                        steps=1,
                        random_start=False,
                        temperature=1.0,
                        loss_formulation="margin_topk",
                        margin_topk_fraction=0.5,
                    ),
                )
                # Stay inside the fake surrogate's clamp interval so this tests
                # the attack gradient, not clamp's derivative at zero.
                clean = torch.full((1, 3, 2, 2), 0.25)
                before = attacker.objective(clean, ["object"], 1, mode)
                adversarial, _ = attacker.perturb_batch(
                    clean, ["object"], 1, mode
                )
                after = attacker.objective(adversarial, ["object"], 1, mode)
                self.assertLess(float(after), float(before))

    def test_configuration_rejects_unknown_formulation_and_invalid_topk(self) -> None:
        with self.assertRaisesRegex(ValueError, "loss_formulation"):
            AttackConfig(loss_formulation="topk_only")
        with self.assertRaisesRegex(ValueError, "margin_topk_fraction"):
            AttackConfig(margin_topk_fraction=0.0)


if __name__ == "__main__":
    unittest.main()


class StepSizeScheduleTests(unittest.TestCase):
    """Both decaying schedules span the same range; only the shape differs."""

    def _attacker(self, schedule: str) -> TargetedPGD:
        return TargetedPGD(_FakeSurrogate(), AttackConfig(
            steps=500, universal_steps=500, step_size=0.25 / 255,
            step_size_schedule=schedule, step_size_min_ratio=0.1,
        ))

    def test_linear_and_cosine_share_endpoints(self) -> None:
        for schedule in ("cosine", "linear"):
            attacker = self._attacker(schedule)
            with self.subTest(schedule=schedule):
                self.assertAlmostEqual(attacker.step_size_at(0, 500), 0.25 / 255)
                self.assertAlmostEqual(
                    attacker.step_size_at(499, 500), 0.1 * 0.25 / 255, places=9
                )

    def test_linear_decays_monotonically(self) -> None:
        attacker = self._attacker("linear")
        sizes = [attacker.step_size_at(step, 500) for step in range(500)]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_cosine_holds_the_large_step_longer(self) -> None:
        # The schedules cross: cosine is above linear in the first half.
        cosine, linear = self._attacker("cosine"), self._attacker("linear")
        self.assertGreater(cosine.step_size_at(125, 500), linear.step_size_at(125, 500))
        self.assertLess(cosine.step_size_at(375, 500), linear.step_size_at(375, 500))

    def test_unknown_schedule_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AttackConfig(step_size_schedule="exponential")
