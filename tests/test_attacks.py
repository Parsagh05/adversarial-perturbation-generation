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
        clean = torch.zeros((1, 3, 2, 2))
        before = attacker.objective(clean, ["object"], 1, "local")
        adversarial, _ = attacker.perturb_batch(clean, ["object"], 1, "local")
        after = attacker.objective(adversarial, ["object"], 1, "local")

        self.assertTrue(torch.isfinite(before))
        self.assertTrue(torch.isfinite(after))
        self.assertLess(float(after), float(before))


if __name__ == "__main__":
    unittest.main()
