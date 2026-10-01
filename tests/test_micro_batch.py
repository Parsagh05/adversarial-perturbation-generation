"""Per-dataset PGD gradient accumulation gives the whole batch's update."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from adversarial_harness.attacks import TargetedPGD
from adversarial_harness.config import AttackConfig
from tests.test_sga import _Surrogate


def _run(mode: str, micro_batch_size: int, optimizer: str = "pgd"):
    torch.manual_seed(7)
    samples = [
        SimpleNamespace(category=("widget", "gasket")[i % 2], protocol_id=f"p{i}")
        for i in range(10)
    ]
    images = {s.protocol_id: torch.rand(3, 24, 24) for s in samples}
    torch.manual_seed(123)
    attacker = TargetedPGD(_Surrogate(), AttackConfig(
        loss_formulation="margin_topk", image_size=24, epsilon=8 / 255,
        step_size=1 / 255, universal_steps=12, universal_batch_size=8,
        diagnostic_interval=4, seed=5, optimizer=optimizer,
        sga_inner_batch_size=4, sga_inner_passes=2,
    ))
    return attacker.optimize_universal(
        samples, lambda s: images[s.protocol_id], 1, mode,
        diagnostic_samples=samples, micro_batch_size=micro_batch_size,
    )


class MicroBatchTests(unittest.TestCase):
    def test_accumulated_pgd_matches_the_whole_batch(self) -> None:
        # 3 does not divide 8, and the last batch of each pass is 2 images.
        for mode in ("global", "local", "combined"):
            whole = _run(mode, 0)
            for micro in (1, 3, 4):
                with self.subTest(mode=mode, micro=micro):
                    split = _run(mode, micro)
                    self.assertTrue(torch.equal(whole.delta, split.delta))
                    for a, b in zip(whole.history, split.history):
                        self.assertAlmostEqual(
                            a["pre_update_total_loss"], b["pre_update_total_loss"], places=5
                        )

    def test_accumulated_sga_matches_whole_inner_batches(self) -> None:
        # Inner batches of 4: chunks of 1 and 3 (3 leaves a chunk of 1).
        for mode in ("global", "local", "combined"):
            whole = _run(mode, 0, "sga")
            for micro in (1, 3):
                with self.subTest(mode=mode, micro=micro):
                    split = _run(mode, micro, "sga")
                    self.assertTrue(torch.equal(whole.delta, split.delta))
                    for a, b in zip(whole.history, split.history):
                        self.assertAlmostEqual(
                            a["pre_update_total_loss"], b["pre_update_total_loss"], places=5
                        )

    def test_diagnostics_run_in_chunks_no_larger_than_a_training_forward(self) -> None:
        seen = []
        surrogate = _Surrogate()
        original = surrogate.encode_visual

        def counting(images, *args, **kwargs):
            seen.append(images.shape[0])
            return original(images, *args, **kwargs)

        surrogate.encode_visual = counting
        torch.manual_seed(123)
        attacker = TargetedPGD(surrogate, AttackConfig(
            loss_formulation="margin_topk", image_size=24, epsilon=8 / 255,
            step_size=1 / 255, universal_steps=2, universal_batch_size=8,
            diagnostic_interval=1, seed=5, optimizer="sga",
            sga_inner_batch_size=4, sga_inner_passes=2,
        ))
        samples = [SimpleNamespace(category="widget", protocol_id=f"p{i}") for i in range(10)]
        attacker.optimize_universal(
            samples, lambda s: torch.rand(3, 24, 24), 1, "global",
            diagnostic_samples=samples, micro_batch_size=3,
        )
        self.assertLessEqual(max(seen), 3)

    def test_a_chunk_as_large_as_the_batch_is_the_plain_path(self) -> None:
        self.assertTrue(torch.equal(_run("local", 0).delta, _run("local", 8).delta))


if __name__ == "__main__":
    unittest.main()
