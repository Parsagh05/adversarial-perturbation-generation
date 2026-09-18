from __future__ import annotations

import math
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


class LossModeSelectionTests(unittest.TestCase):
    """LOSS_MODES selects which objectives run; it is not a fixed set.

    run_per_dataset.py used to require all three, so LOSS_MODES=global,local
    ran under per_category and per_image and then aborted at the dataset
    scope. Validation belongs to AttackConfig, which every runner builds.
    """

    def test_a_subset_of_loss_modes_is_accepted(self) -> None:
        for modes in (
            ("global",),
            ("local",),
            ("combined",),
            ("global", "local"),
            ("global", "local", "combined"),
        ):
            with self.subTest(modes=modes):
                self.assertEqual(AttackConfig(loss_modes=modes).loss_modes, modes)

    def test_unknown_or_empty_loss_modes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown loss_modes"):
            AttackConfig(loss_modes=("global", "segmentation"))
        with self.assertRaisesRegex(ValueError, "loss_modes cannot be empty"):
            AttackConfig(loss_modes=())

    def test_no_runner_requires_the_complete_set(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for name in ("run_per_dataset.py", "run_per_category.py", "run_per_image.py"):
            with self.subTest(runner=name):
                source = (root / name).read_text(encoding="utf-8")
                self.assertNotIn("set(LOSS_MODES) !=", source)


class DirectionSelectionTests(unittest.TestCase):
    """DIRECTIONS selects which attack directions run, like LOSS_MODES.

    run_per_dataset.py used to require both. The audit is what made that look
    necessary: it demanded both directions in every manifest group. It now
    reads DIRECTIONS itself, so a one-direction run audits cleanly and a run
    that silently dropped a requested direction still fails.
    """

    def test_a_single_direction_is_accepted(self) -> None:
        for directions in (
            ("normal_to_abnormal",),
            ("abnormal_to_normal",),
            ("normal_to_abnormal", "abnormal_to_normal"),
        ):
            with self.subTest(directions=directions):
                self.assertEqual(
                    AttackConfig(directions=directions).directions, directions
                )

    def test_unknown_or_empty_directions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown directions"):
            AttackConfig(directions=("normal_to_abnormal", "sideways"))
        with self.assertRaisesRegex(ValueError, "directions cannot be empty"):
            AttackConfig(directions=())

    def test_no_runner_requires_both_directions(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for name in ("run_per_dataset.py", "run_per_category.py", "run_per_image.py"):
            with self.subTest(runner=name):
                source = (root / name).read_text(encoding="utf-8")
                self.assertNotIn("set(DIRECTIONS) !=", source)


class AuditDirectionExpectationTests(unittest.TestCase):
    """The audit checks the directions the run was asked to produce."""

    def _audit(self, **environment):
        import importlib
        import os
        import sys
        import tempfile
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            base = {"OUTPUT_BASE": tmp}
            base.update({k: v for k, v in environment.items() if v is not None})
            with mock.patch.dict(os.environ, base, clear=False):
                for key, value in environment.items():
                    if value is None:
                        os.environ.pop(key, None)
                sys.modules.pop("audit_generation", None)
                try:
                    return importlib.import_module("audit_generation")
                finally:
                    sys.modules.pop("audit_generation", None)

    def test_defaults_to_both_directions(self) -> None:
        module = self._audit(DIRECTIONS=None)
        self.assertEqual(
            set(module.EXPECTED_DIRECTIONS),
            {"normal_to_abnormal", "abnormal_to_normal"},
        )

    def test_an_explicitly_empty_value_is_rejected(self) -> None:
        # Matches the runners: csv_tuple("") is empty and AttackConfig refuses
        # an empty selection, so the audit must not silently fall back.
        with self.assertRaises(ValueError):
            self._audit(DIRECTIONS="")

    def test_follows_a_single_requested_direction(self) -> None:
        module = self._audit(DIRECTIONS="normal_to_abnormal")
        self.assertEqual(module.EXPECTED_DIRECTIONS, ("normal_to_abnormal",))

    def test_rejects_an_unknown_direction(self) -> None:
        with self.assertRaises(ValueError):
            self._audit(DIRECTIONS="normal_to_abnormal,sideways")


class _MultiCategorySurrogate:
    """Two prompt banks, so a batch splits into unequal category groups."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.prompts = {
            "bottle": SimpleNamespace(
                normal_embeddings=torch.tensor([[1.0, 0.0]]),
                abnormal_embeddings=torch.tensor([[0.0, 1.0]]),
            ),
            "cable": SimpleNamespace(
                normal_embeddings=torch.tensor([[0.6, 0.8]]),
                abnormal_embeddings=torch.tensor([[-0.8, 0.6]]),
            ),
        }

    def encode_visual(self, images_01, include_patches=True):
        signal = images_01.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
        token = torch.stack((1.0 - signal, signal), dim=-1)
        cls = token[:, None, :]
        patches = token[:, None, :].expand(-1, 4, -1)
        return token, [torch.cat((cls, patches), dim=1)] if include_patches else []


class ImageMeanReductionTests(unittest.TestCase):
    """The batch loss is the plain mean over images, not a mean of group means.

    Averaging the per-category means made an image's weight depend on how many
    of its category were drawn into the batch, so the gradient was not the
    unbiased minibatch estimate a stochastic universal attack assumes.
    """

    def _attacker(self, **settings) -> TargetedPGD:
        return TargetedPGD(
            _MultiCategorySurrogate(),
            AttackConfig(temperature=1.0, **settings),
        )

    def _images(self, count: int) -> torch.Tensor:
        steps = torch.linspace(0.1, 0.9, count)
        return torch.stack([torch.full((3, 2, 2), float(value)) for value in steps])

    def test_batch_loss_equals_the_mean_of_per_image_losses(self) -> None:
        # Three bottles and one cable: the old form gave the cable 3x weight.
        categories = ["bottle", "bottle", "bottle", "cable"]
        images = self._images(len(categories))
        for formulation in ("margin_topk", "ce_focal_dice"):
            for mode in ("global", "local", "combined"):
                for target_label in (0, 1):
                    with self.subTest(
                        formulation=formulation, mode=mode, target=target_label
                    ):
                        attacker = self._attacker(loss_formulation=formulation)
                        batched = float(
                            attacker.objective(images, categories, target_label, mode)
                        )
                        singles = [
                            float(
                                attacker.objective(
                                    images[index : index + 1],
                                    [category],
                                    target_label,
                                    mode,
                                )
                            )
                            for index, category in enumerate(categories)
                        ]
                        self.assertAlmostEqual(
                            batched, sum(singles) / len(singles), places=5
                        )

    def test_group_order_and_composition_do_not_move_the_loss(self) -> None:
        # Same four images, different arrangement: a plain mean is invariant.
        images = self._images(4)
        attacker = self._attacker()
        balanced = ["bottle", "cable", "bottle", "cable"]
        skewed = ["bottle", "bottle", "bottle", "cable"]
        by_image = {
            category: [
                float(
                    attacker.objective(
                        images[index : index + 1], [category], 1, "combined"
                    )
                )
                for index in range(4)
            ]
            for category in ("bottle", "cable")
        }
        for categories in (balanced, skewed):
            with self.subTest(categories=categories):
                expected = sum(
                    by_image[category][index]
                    for index, category in enumerate(categories)
                ) / len(categories)
                actual = float(attacker.objective(images, categories, 1, "combined"))
                self.assertAlmostEqual(actual, expected, places=5)

    def test_a_single_category_batch_is_unchanged(self) -> None:
        # per_category and per_image draw one category, where the two forms
        # already coincide; their existing results must not move.
        images = self._images(3)
        categories = ["bottle"] * 3
        for formulation in ("margin_topk", "ce_focal_dice"):
            with self.subTest(formulation=formulation):
                attacker = self._attacker(loss_formulation=formulation)
                actual = float(attacker.objective(images, categories, 1, "combined"))
                singles = [
                    float(
                        attacker.objective(
                            images[index : index + 1], ["bottle"], 1, "combined"
                        )
                    )
                    for index in range(3)
                ]
                self.assertAlmostEqual(actual, sum(singles) / 3, places=5)


class _Sample:
    def __init__(self, category: str, value: float) -> None:
        self.category = category
        self.value = value


class MarginHingeTests(unittest.TestCase):
    """An image stops contributing once it has moved far enough toward target.

    The floor is each image's own clean margin minus the displacement, so the
    hinge means "moved this far from where it started" rather than "past an
    absolute value". That is what lets one setting work across conditions
    whose margins differ by an order of magnitude.
    """

    def _attacker(self, displacement=None) -> TargetedPGD:
        return TargetedPGD(
            _MultiCategorySurrogate(),
            AttackConfig(
                temperature=1.0,
                loss_formulation="margin_topk",
                margin_hinge_displacement=displacement,
                universal_batch_size=4,
            ),
        )

    def _images(self, values) -> torch.Tensor:
        return torch.stack([torch.full((3, 2, 2), float(v)) for v in values])

    def _floors(self, attacker, samples, target_label, mode="global"):
        return attacker.hinge_floors(
            samples,
            lambda s: torch.full((3, 2, 2), float(s.value)),
            target_label,
            mode,
        )

    def test_disabled_by_default_and_costs_nothing(self) -> None:
        attacker = self._attacker()
        samples = [_Sample("bottle", 0.3), _Sample("cable", 0.3)]
        self.assertIsNone(self._floors(attacker, samples, 1))

    def test_a_saturated_image_stops_changing_the_loss(self) -> None:
        attacker = self._attacker(displacement=0.0)
        samples = [_Sample("bottle", 0.3), _Sample("bottle", 0.3)]
        floors = self._floors(attacker, samples, target_label=1)
        categories = ["bottle", "bottle"]
        # Both images have moved toward abnormal, so both are saturated and
        # moving them further must not change the objective.
        moved = float(
            attacker.objective_components(
                self._images([0.6, 0.6]), categories, 1, "global",
                hinge_floors=floors,
            )["global"]
        )
        further = float(
            attacker.objective_components(
                self._images([0.9, 0.9]), categories, 1, "global",
                hinge_floors=floors,
            )["global"]
        )
        self.assertAlmostEqual(moved, further, places=6)

    def test_an_unmoved_image_still_contributes(self) -> None:
        attacker = self._attacker(displacement=0.5)
        samples = [_Sample("bottle", 0.3), _Sample("bottle", 0.3)]
        floors = self._floors(attacker, samples, target_label=1)
        categories = ["bottle", "bottle"]
        clean = float(
            attacker.objective_components(
                self._images([0.3, 0.3]), categories, 1, "global",
                hinge_floors=floors,
            )["global"]
        )
        nudged = float(
            attacker.objective_components(
                self._images([0.35, 0.35]), categories, 1, "global",
                hinge_floors=floors,
            )["global"]
        )
        self.assertLess(nudged, clean)

    def test_a_large_displacement_reproduces_the_unhinged_loss(self) -> None:
        attacker = self._attacker(displacement=1000.0)
        samples = [_Sample("bottle", 0.3), _Sample("cable", 0.7)]
        floors = self._floors(attacker, samples, target_label=1)
        categories = ["bottle", "cable"]
        images = self._images([0.6, 0.2])
        hinged = float(
            attacker.objective_components(
                images, categories, 1, "global", hinge_floors=floors
            )["global"]
        )
        plain = float(
            attacker.objective_components(images, categories, 1, "global")["global"]
        )
        self.assertAlmostEqual(hinged, plain, places=6)

    def test_the_floor_follows_each_image_not_an_absolute_margin(self) -> None:
        # Two categories whose clean margins differ; the same displacement must
        # saturate both, which an absolute floor would not do.
        attacker = self._attacker(displacement=0.0)
        samples = [_Sample("bottle", 0.3), _Sample("cable", 0.3)]
        floors = self._floors(attacker, samples, target_label=1)
        clean = attacker.objective_components(
            self._images([0.3, 0.3]), ["bottle", "cable"], 1, "global",
            hinge_floors=floors,
        )
        moved = attacker.objective_components(
            self._images([0.8, 0.8]), ["bottle", "cable"], 1, "global",
            hinge_floors=floors,
        )
        self.assertAlmostEqual(
            float(clean["global_saturated_fraction"]), 0.0, places=6
        )
        self.assertAlmostEqual(
            float(moved["global_saturated_fraction"]), 1.0, places=6
        )

    def test_full_saturation_zeroes_the_gradient(self) -> None:
        # sign(0) = 0 freezes delta, which is why the fraction is recorded.
        attacker = self._attacker(displacement=0.0)
        samples = [_Sample("bottle", 0.3), _Sample("bottle", 0.3)]
        floors = self._floors(attacker, samples, target_label=1)
        images = self._images([0.8, 0.8]).requires_grad_(True)
        loss = attacker.objective_components(
            images, ["bottle", "bottle"], 1, "global", hinge_floors=floors
        )["global"]
        gradient = torch.autograd.grad(loss, images, allow_unused=True)[0]
        self.assertTrue(gradient is None or float(gradient.abs().max()) == 0.0)

    def test_the_hinge_is_ignored_for_the_bounded_loss(self) -> None:
        # focal and Dice are already bounded, so ce_focal_dice needs no hinge.
        attacker = TargetedPGD(
            _MultiCategorySurrogate(),
            AttackConfig(
                temperature=1.0,
                loss_formulation="ce_focal_dice",
                margin_hinge_displacement=0.0,
            ),
        )
        self.assertIsNone(
            self._floors(attacker, [_Sample("bottle", 0.3)], target_label=1)
        )


class MomentumTests(unittest.TestCase):
    """m = decay * m + g, stepping along sign(m). Decay 0 is plain sign-PGD."""

    def _attacker(self, decay: float) -> TargetedPGD:
        return TargetedPGD(
            _FakeSurrogate(), AttackConfig(temperature=1.0, momentum_decay=decay)
        )

    def test_zero_decay_returns_the_gradient_untouched(self) -> None:
        attacker = self._attacker(0.0)
        gradient = torch.tensor([1.0, -2.0, 3.0])
        stale = torch.tensor([9.0, 9.0, 9.0])
        direction, cosine = attacker.update_direction(gradient, stale)
        self.assertTrue(torch.equal(direction, gradient))
        self.assertTrue(math.isnan(cosine))

    def test_the_accumulator_carries_the_previous_gradient(self) -> None:
        attacker = self._attacker(0.9)
        first = torch.tensor([1.0, 0.0])
        second = torch.tensor([0.0, 1.0])
        momentum = torch.zeros(2)
        momentum, _ = attacker.update_direction(first, momentum)
        self.assertTrue(torch.allclose(momentum, first))
        momentum, _ = attacker.update_direction(second, momentum)
        self.assertTrue(torch.allclose(momentum, torch.tensor([0.9, 1.0])))

    def test_momentum_can_flip_the_step_the_gradient_alone_would_take(self) -> None:
        # What the sign actually follows is the accumulation, not the gradient.
        attacker = self._attacker(0.9)
        momentum = torch.zeros(2)
        for _ in range(3):
            momentum, _ = attacker.update_direction(torch.tensor([1.0, 0.0]), momentum)
        reversal = torch.tensor([-1.2, 0.0])
        direction, _ = attacker.update_direction(reversal, momentum)
        self.assertEqual(float(reversal.sign()[0]), -1.0)
        self.assertEqual(float(direction.sign()[0]), 1.0)

    def test_the_cosine_reports_whether_momentum_turned_the_direction(self) -> None:
        attacker = self._attacker(0.9)
        gradient = torch.tensor([1.0, 0.0])
        aligned, cosine_aligned = attacker.update_direction(gradient, torch.zeros(2))
        self.assertAlmostEqual(cosine_aligned, 1.0, places=6)
        _, cosine_turned = attacker.update_direction(
            torch.tensor([0.0, 1.0]), aligned
        )
        self.assertLess(cosine_turned, 1.0)

    def test_memory_length_follows_the_decay(self) -> None:
        # Why 1.0 cancels the hinge: the first gradient never fades.
        first = torch.tensor([1.0])
        for decay, expected in ((0.9, 0.9 ** 20), (1.0, 1.0)):
            with self.subTest(decay=decay):
                attacker = self._attacker(decay)
                momentum, _ = attacker.update_direction(first, torch.zeros(1))
                for _ in range(20):
                    momentum, _ = attacker.update_direction(torch.zeros(1), momentum)
                self.assertAlmostEqual(float(momentum), expected, places=6)

    def test_the_decay_is_validated(self) -> None:
        for bad in (-0.1, 1.5):
            with self.subTest(decay=bad):
                with self.assertRaisesRegex(ValueError, "momentum_decay"):
                    AttackConfig(momentum_decay=bad)


class CheckpointSelectionTests(unittest.TestCase):
    """best keeps the lowest-scoring iterate; final keeps the last step.

    The universal-attack papers return the final iterate and do no checkpoint
    selection; best-iterate selection is the per-image convention. The two
    modes share an identical trajectory, so only the retained point differs.
    """

    def _samples(self):
        return [_Sample("object", 0.2), _Sample("object", 0.5)]

    def _run(self, selection: str):
        attacker = TargetedPGD(
            _DifferentiableFakeSurrogate(),
            AttackConfig(
                temperature=1.0,
                image_size=2,
                epsilon=0.2,
                step_size=0.1,
                universal_steps=6,
                universal_batch_size=2,
                diagnostic_interval=1,
                seed=7,
                random_start=False,
                checkpoint_selection=selection,
            ),
        )
        return attacker.optimize_universal(
            self._samples(),
            lambda s: torch.full((3, 2, 2), float(s.value)),
            target_label=1,
            mode="global",
        )

    def test_the_trajectory_is_identical_in_both_modes(self) -> None:
        best = self._run("best")
        final = self._run("final")
        self.assertEqual(len(best.history), len(final.history))
        for left, right in zip(best.history, final.history):
            self.assertAlmostEqual(
                left["total_loss"], right["total_loss"], places=6
            )
            self.assertAlmostEqual(
                left["pre_update_total_loss"],
                right["pre_update_total_loss"],
                places=6,
            )

    def test_final_returns_the_last_step(self) -> None:
        final = self._run("final")
        self.assertEqual(final.selected_step, 6)
        self.assertAlmostEqual(
            final.selected_diagnostic_loss, final.final_losses["total"], places=6
        )

    def test_best_never_returns_something_worse_than_clean(self) -> None:
        best = self._run("best")
        self.assertLessEqual(
            best.selected_diagnostic_loss, best.initial_losses["total"] + 1e-9
        )

    def test_the_default_is_best(self) -> None:
        from adversarial_harness.config import AttackConfig as Config

        self.assertEqual(Config().checkpoint_selection, "best")

    def test_the_per_image_path_takes_the_same_switch(self) -> None:
        images = torch.full((2, 3, 2, 2), 0.4)
        results = {}
        for selection in ("best", "final"):
            attacker = TargetedPGD(
                _DifferentiableFakeSurrogate(),
                AttackConfig(
                    temperature=1.0, epsilon=0.2, step_size=0.15, steps=5,
                    random_start=False, checkpoint_selection=selection,
                ),
            )
            _, delta = attacker.perturb_batch(
                images, ["object", "object"], 1, "global"
            )
            results[selection] = delta
        self.assertEqual(results["best"].shape, results["final"].shape)

    def test_an_invalid_selection_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint_selection"):
            AttackConfig(checkpoint_selection="last")
