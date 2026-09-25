"""The objective must survive mixed precision.

run_per_category.py and run_per_image.py wrap the objective in
torch.autocast(bfloat16) when USE_AMP=true (off by default); run_per_dataset.py
never does.
Under autocast the logits come back in the autocast dtype while the
accumulation buffers are allocated from the float32 visual features, and
index_copy_ rejects a dtype mismatch instead of promoting it the way the
surrounding additions do.

Only the margin_topk branch performs those copies - ce_focal_dice goes through
F.cross_entropy - so margin_topk plus autocast plus per_category/per_image was
the first combination to reach it, which is why it appeared when margin_topk
became the default formulation.

CPU autocast produces bfloat16 exactly as CUDA does, so these run everywhere
rather than only on a GPU box.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from adversarial_harness.attacks import TargetedPGD
from adversarial_harness.config import AttackConfig
from adversarial_harness.prompts import CategoryPromptBank

CATEGORIES = ("widget", "widget", "gasket")
EMBEDDING = 16
PATCHES = 17          # a CLS token plus a 4x4 grid, once CLS is stripped


def _attacker(formulation: str) -> TargetedPGD:
    torch.manual_seed(0)
    prompts = {
        category: CategoryPromptBank(
            category=category,
            normal_prompts=("a photo of a normal object",),
            abnormal_prompts=("a photo of a damaged object",),
            normal_embeddings=torch.randn(1, EMBEDDING),
            abnormal_embeddings=torch.randn(1, EMBEDDING),
        )
        for category in set(CATEGORIES)
    }
    return TargetedPGD(
        SimpleNamespace(prompts=prompts, device="cpu"),
        AttackConfig(loss_formulation=formulation),
    )


def _features() -> tuple[torch.Tensor, list[torch.Tensor]]:
    torch.manual_seed(1)
    batch = len(CATEGORIES)
    return (
        torch.randn(batch, EMBEDDING, requires_grad=True),
        [torch.randn(batch, PATCHES, EMBEDDING) for _ in range(2)],
    )


class AutocastDtypeTests(unittest.TestCase):
    def _run(self, formulation: str, mode: str, autocast: bool):
        attacker = _attacker(formulation)
        global_features, patch_features = _features()
        # [B, H, W]; the objective adds the channel dimension itself.
        masks = torch.zeros(len(CATEGORIES), 24, 24)
        masks[:, 4:12, 4:12] = 1.0
        context = (
            torch.autocast(device_type="cpu", dtype=torch.bfloat16)
            if autocast else torch.autocast(device_type="cpu", enabled=False)
        )
        with context:
            return attacker._group_losses(
                global_features, patch_features, CATEGORIES,
                target_label=1, mode=mode,
                spatial_masks=masks if mode in {"local", "combined"} else None,
                return_per_sample=True,
            )

    def test_margin_topk_under_autocast(self) -> None:
        """The reported crash: both modes that do the strict copies."""

        for mode in ("global", "local", "combined"):
            with self.subTest(mode=mode):
                result = self._run("margin_topk", mode, autocast=True)
                self.assertTrue(torch.isfinite(result["total"]).all())

    def test_the_accumulation_stays_float32_under_autocast(self) -> None:
        """Casting the source, not the buffer, is what keeps the reduction f32."""

        result = self._run("margin_topk", "global", autocast=True)
        self.assertEqual(result["per_sample_global"].dtype, torch.float32)

    def test_autocast_matches_full_precision_closely(self) -> None:
        """A dtype fix must not quietly change the objective."""

        for mode in ("global", "local"):
            with self.subTest(mode=mode):
                mixed = self._run("margin_topk", mode, autocast=True)
                exact = self._run("margin_topk", mode, autocast=False)
                self.assertTrue(
                    torch.allclose(
                        mixed["total"].float(), exact["total"].float(),
                        rtol=0.05, atol=0.05,
                    ),
                    f"{mode}: {mixed['total']} vs {exact['total']}",
                )

    def test_ce_focal_dice_under_autocast(self) -> None:
        """The other formulation shares the category-logit copies."""

        for mode in ("global", "local", "combined"):
            with self.subTest(mode=mode):
                result = self._run("ce_focal_dice", mode, autocast=True)
                self.assertTrue(torch.isfinite(result["total"]).all())

    def test_surrogate_scores_under_autocast(self) -> None:
        """The other two strict copies live here, not in _group_losses.

        surrogate_scores fills its own float32 logit buffers from the same
        autocast logits, so it carries the identical exposure and is reached
        by a different call path.
        """

        for mode in ("global", "local", "combined"):
            with self.subTest(mode=mode):
                attacker = _attacker("margin_topk")
                global_features, patch_features = _features()
                attacker.surrogate.encode_visual = (
                    lambda *args, _g=global_features.detach(),
                    _p=patch_features, **kwargs: (_g, _p)
                )
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    scores = attacker.surrogate_scores(
                        torch.rand(len(CATEGORIES), 3, 24, 24), CATEGORIES, mode
                    )
                for key, value in scores.items():
                    # local_score is deliberately NaN when the mode does not
                    # compute local logits; that is the documented contract,
                    # not a dtype failure.
                    if key == "local_score" and mode == "global":
                        continue
                    with self.subTest(score=key):
                        self.assertTrue(np.isfinite(value).all())

    def test_gradients_still_flow_under_autocast(self) -> None:
        """A stray .to() can detach; the attack would silently stop learning."""

        attacker = _attacker("margin_topk")
        global_features, patch_features = _features()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            losses = attacker._group_losses(
                global_features, patch_features, CATEGORIES,
                target_label=1, mode="global",
            )
        gradient, = torch.autograd.grad(losses["total"], global_features)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)


class PrecisionDefaultTests(unittest.TestCase):
    """fp32 unless asked, and a delta's precision decides whether it is reused.

    The runners need CUDA, so these read their source.
    """

    ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

    def test_every_scope_defaults_to_fp32(self) -> None:
        for runner in ("run_per_category.py", "run_per_image.py"):
            with self.subTest(runner=runner):
                source = (self.ROOT / runner).read_text(encoding="utf-8")
                self.assertIn('USE_AMP = bool_env("USE_AMP", False)', source)
        launcher = (self.ROOT / "train.sh").read_text(encoding="utf-8")
        self.assertIn('export USE_AMP="${USE_AMP:-false}"', launcher)
        dataset = (self.ROOT / "run_per_dataset.py").read_text(encoding="utf-8")
        self.assertNotIn("autocast", dataset)

    def test_every_scope_disables_tf32(self) -> None:
        """PyTorch enables TF32 for convolutions by default, so it must be set."""

        for runner in ("run_per_dataset.py", "run_per_category.py", "run_per_image.py"):
            with self.subTest(runner=runner):
                source = (self.ROOT / runner).read_text(encoding="utf-8")
                self.assertIn("ALLOW_TF32 = False", source)
                self.assertIn("torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32", source)
                self.assertIn("torch.backends.cudnn.allow_tf32 = ALLOW_TF32", source)
                self.assertNotIn("allow_tf32 = True", source)
                start = source.index("expected = {")
                end = source.index("reusable(pt_path, expected)")
                self.assertIn('"allow_tf32": ALLOW_TF32,', source[start:end])

    def test_precision_is_part_of_the_reuse_key(self) -> None:
        for runner in ("run_per_category.py", "run_per_image.py"):
            with self.subTest(runner=runner):
                source = (self.ROOT / runner).read_text(encoding="utf-8")
                self.assertIn(
                    '"autocast_dtype": AMP_DTYPE_NAME if AMP_ENABLED else "float32",',
                    source,
                )
                # The reuse key is only checked if it sits inside `expected`.
                start = source.index("expected = {")
                end = source.index("reusable(pt_path, expected)")
                self.assertIn('"autocast_dtype"', source[start:end])


if __name__ == "__main__":
    unittest.main()
