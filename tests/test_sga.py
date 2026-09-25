"""SGA (Liu et al., ICCV 2023) as a switchable alternative to sign-PGD."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from adversarial_harness.attacks import TargetedPGD, sga_inner_batches
from adversarial_harness.config import AttackConfig
from adversarial_harness.prompts import CategoryPromptBank

ROOT = Path(__file__).resolve().parents[1]


class _Surrogate:
    """A small differentiable stand-in for CLIP."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        torch.manual_seed(0)
        self.projection = torch.nn.Linear(3, 16)
        self.prompts = {
            name: CategoryPromptBank(
                category=name, normal_prompts=("n",), abnormal_prompts=("a",),
                normal_embeddings=torch.randn(1, 16), abnormal_embeddings=torch.randn(1, 16),
            )
            for name in ("widget", "gasket")
        }

    def encode_visual(self, images_01, include_patches=True):
        pooled = torch.nn.functional.adaptive_avg_pool2d(images_01, 6)
        tokens = self.projection(pooled.flatten(2).transpose(1, 2))
        glob = tokens.mean(dim=1)
        patches = torch.cat((glob[:, None, :], tokens), dim=1)
        return glob, [patches] if include_patches else []


def _run(samples_count: int, mode: str = "local", **config):
    torch.manual_seed(7)
    samples = [
        SimpleNamespace(category=("widget", "gasket")[i % 2], protocol_id=f"p{i}")
        for i in range(samples_count)
    ]
    images = {s.protocol_id: torch.rand(3, 24, 24) for s in samples}
    settings = dict(
        loss_formulation="margin_topk", image_size=24, epsilon=8 / 255,
        step_size=1 / 255, universal_steps=30, universal_batch_size=4,
        diagnostic_interval=10, seed=5,
    )
    settings.update(config)
    torch.manual_seed(123)
    attacker = TargetedPGD(_Surrogate(), AttackConfig(**settings))
    return attacker.optimize_universal(
        samples, lambda s: images[s.protocol_id], 1, mode, diagnostic_samples=samples
    )


class InnerBatchTests(unittest.TestCase):
    def test_every_image_is_used_exactly_k_times(self) -> None:
        rng = np.random.default_rng(0)
        for outer, inner, passes in ((8, 2, 1), (8, 3, 4), (5, 2, 2), (4, 8, 1)):
            with self.subTest(outer=outer, inner=inner, passes=passes):
                batches = sga_inner_batches(outer, inner, passes, rng)
                used = sorted(i for batch in batches for i in batch)
                self.assertEqual(used, sorted(list(range(outer)) * passes))
                self.assertEqual(len(batches), passes * -(-outer // inner))
                self.assertTrue(all(0 < len(b) <= inner for b in batches))

    def test_the_defaults_are_the_papers_k_with_inner_batches_of_two(self) -> None:
        config = AttackConfig()
        self.assertEqual((config.sga_inner_batch_size, config.sga_inner_passes), (2, 4))
        batches = sga_inner_batches(
            8, config.sga_inner_batch_size, config.sga_inner_passes,
            np.random.default_rng(0),
        )
        # An outer batch of 8, four passes of four inner batches of 2.
        self.assertEqual([len(b) for b in batches], [2] * 16)


class OptimizeUniversalTests(unittest.TestCase):
    def test_pgd_is_the_default(self) -> None:
        self.assertEqual(AttackConfig().optimizer, "pgd")

    def test_one_inner_batch_covering_the_outer_batch_is_pgd(self) -> None:
        """M = 1 inner step at delta itself: the sum is PGD's gradient."""

        for mode in ("global", "local", "combined"):
            with self.subTest(mode=mode):
                # Every step sees all four samples, so batch order cannot matter.
                pgd = _run(4, mode, optimizer="pgd")
                sga = _run(4, mode, optimizer="sga", sga_inner_batch_size=4, sga_inner_passes=1)
                agree = (pgd.delta.sign() == sga.delta.sign()).float().mean().item()
                self.assertGreater(agree, 0.999)

    def test_sga_runs_every_mode_within_the_budget(self) -> None:
        for mode in ("global", "local", "combined"):
            with self.subTest(mode=mode):
                result = _run(8, mode, optimizer="sga", sga_inner_passes=2, momentum_decay=0.9)
                self.assertEqual(len(result.history), 30)
                self.assertLessEqual(float(result.delta.abs().max()), 8 / 255 + 1e-6)
                self.assertTrue(all(
                    np.isfinite(row["pre_update_total_loss"]) for row in result.history
                ))

    def test_small_inner_batches_change_the_trajectory(self) -> None:
        pgd = _run(8, optimizer="pgd")
        sga = _run(8, optimizer="sga", sga_inner_batch_size=1)
        self.assertFalse(torch.equal(pgd.delta, sga.delta))

    def test_bad_settings_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AttackConfig(optimizer="adam")
        with self.assertRaises(ValueError):
            AttackConfig(sga_inner_batch_size=0)
        with self.assertRaises(ValueError):
            AttackConfig(sga_inner_passes=0)


class NamingTests(unittest.TestCase):
    def test_pgd_adds_nothing_and_sga_names_itself(self) -> None:
        from setup_catalog import settings_tag

        base = dict(epsilon_label="4/255", loss_formulation="margin_topk",
                    split_protocol="full", full_data_cross=False)
        self.assertEqual(settings_tag(**base), "eps4_full_halfcross")
        self.assertEqual(settings_tag(**base, optimizer=("pgd", 2, 4)), "eps4_full_halfcross")
        self.assertEqual(settings_tag(**base, optimizer=("sga", 2, 4)), "eps4_sga_full_halfcross")
        self.assertEqual(settings_tag(**base, optimizer=("sga", 2, 1)), "eps4_sgak1_full_halfcross")
        self.assertEqual(settings_tag(**base, optimizer=("sga", 1, 4)), "eps4_sgaib1_full_halfcross")
        self.assertEqual(settings_tag(**base, optimizer=("sga", 1, 1)), "eps4_sgak1ib1_full_halfcross")
        self.assertEqual(
            settings_tag(**base, momentum_decay=0.9, optimizer=("sga", 2, 4)),
            "eps4_mom0p9_sga_full_halfcross",
        )

    def test_the_setting_reads_the_environment(self) -> None:
        from setup_catalog import optimizer_setting

        with mock.patch.dict(os.environ, {}, clear=False):
            for name in ("OPTIMIZER", "SGA_INNER_BATCH_SIZE", "SGA_INNER_PASSES"):
                os.environ.pop(name, None)
            self.assertEqual(optimizer_setting(), ("pgd", 2, 4))
        with mock.patch.dict(os.environ, {
            "OPTIMIZER": "SGA", "SGA_INNER_BATCH_SIZE": "1", "SGA_INNER_PASSES": "4",
        }):
            self.assertEqual(optimizer_setting(), ("sga", 1, 4))
        with mock.patch.dict(os.environ, {"OPTIMIZER": "adam"}):
            with self.assertRaises(ValueError):
                optimizer_setting()


class RunnerWiringTests(unittest.TestCase):
    def test_universal_runners_pass_it_on_and_key_reuse_on_it(self) -> None:
        for runner in ("run_per_dataset.py", "run_per_category.py"):
            with self.subTest(runner=runner):
                source = (ROOT / runner).read_text(encoding="utf-8")
                self.assertIn("OPTIMIZER = optimizer_setting()", source)
                self.assertIn("optimizer=OPTIMIZER[0],", source)
                self.assertIn("sga_inner_batch_size=OPTIMIZER[1],", source)
                self.assertIn("sga_inner_passes=OPTIMIZER[2],", source)
                start = source.index("expected = {")
                end = source.index("reusable(pt_path, expected)")
                self.assertIn('"optimizer": list(OPTIMIZER),', source[start:end])

    def test_per_category_has_its_own_sga_loop(self) -> None:
        source = (ROOT / "run_per_category.py").read_text(encoding="utf-8")
        self.assertIn("sga_inner_batches(", source)
        self.assertIn("inner_delta = (inner_delta - step_size * inner.sign())", source)

    def test_per_image_falls_back_to_pgd_and_says_so(self) -> None:
        source = (ROOT / "run_per_image.py").read_text(encoding="utf-8")
        self.assertNotIn("SystemExit(\"OPTIMIZER", source)
        self.assertIn(
            'OPTIMIZER_USED = {"name": "pgd", "requested": optimizer_setting()[0]}',
            source,
        )

    def test_every_bundle_records_the_optimizer_it_used(self) -> None:
        import os as _os

        _os.environ.setdefault("MVTEC_ROOT", ".")
        _os.environ.setdefault("VISA_ROOT", ".")
        _os.environ.setdefault("OUTPUT_BASE", ".")
        import common

        for runner in ("run_per_dataset.py", "run_per_category.py", "run_per_image.py"):
            with self.subTest(runner=runner):
                self.assertIn("OPTIMIZER_USED", common._script_constants(ROOT / runner))

    def test_the_random_baseline_ignores_the_optimizer(self) -> None:
        source = (ROOT / "run_per_dataset.py").read_text(encoding="utf-8")
        self.assertIn('{"name": "none", "note": "random baseline: no optimisation"', source)
        launcher = (ROOT / "train.sh").read_text(encoding="utf-8")
        self.assertNotIn("has no optimiser", launcher)

    def test_the_launcher_exports_and_passes_it(self) -> None:
        source = (ROOT / "train.sh").read_text(encoding="utf-8")
        self.assertIn('export OPTIMIZER="${OPTIMIZER:-pgd}"', source)
        self.assertIn("optimizer = optimizer_setting()", source)
        self.assertEqual(source.count("random_baseline, optimizer,"), 3)


if __name__ == "__main__":
    unittest.main()
