"""The surrogate is OpenAI's own CLIP, reading the final block only.

Loading the real ViT-L weights needs a ~1 GB download, so the positional
stretch is checked on a stand-in; the runners are checked through their source.
"""
from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from adversarial_harness.models import (
    CLIPSurrogate,
    SURROGATE_CLIP,
    SURROGATE_FEATURE_LAYERS,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = ("run_per_dataset.py", "run_per_category.py", "run_per_image.py")


def _anomalyclip_stretch(positional: torch.Tensor, new_side: int) -> torch.Tensor:
    """AnomalyCLIP's VisionTransformer.forward, which the targets run at 518 px."""

    side = int((positional.shape[0] - 1) ** 0.5)
    width = positional.shape[-1]
    new_pos = positional[1:, :].reshape(-1, side, side, width).permute(0, 3, 1, 2)
    new_pos = F.interpolate(new_pos, (new_side, new_side), mode="bilinear")
    new_pos = new_pos.reshape(-1, width, new_side * new_side).transpose(1, 2)
    return torch.cat([positional[:1, :], new_pos[0]], 0)


class PositionalStretchTests(unittest.TestCase):
    def _surrogate(self, native: torch.Tensor) -> CLIPSurrogate:
        surrogate = object.__new__(CLIPSurrogate)
        visual = SimpleNamespace(positional_embedding=torch.nn.Parameter(native.clone()))
        surrogate.model = SimpleNamespace(visual=visual)
        surrogate._native_positional_embedding = native.clone()
        surrogate._position_grid = 24
        return surrogate

    def test_matches_the_targets_stretch_to_518(self) -> None:
        torch.manual_seed(0)
        native = torch.randn(24 * 24 + 1, 8)
        surrogate = self._surrogate(native)
        surrogate._fit_positional_embedding(37)
        stretched = surrogate.model.visual.positional_embedding.data
        self.assertEqual(tuple(stretched.shape), (37 * 37 + 1, 8))
        self.assertTrue(torch.equal(stretched, _anomalyclip_stretch(native, 37)))

    def test_always_stretches_from_the_native_grid(self) -> None:
        """Resizing twice must not compound the interpolation."""

        torch.manual_seed(1)
        native = torch.randn(24 * 24 + 1, 8)
        surrogate = self._surrogate(native)
        surrogate._fit_positional_embedding(30)
        surrogate._fit_positional_embedding(37)
        self.assertTrue(torch.equal(
            surrogate.model.visual.positional_embedding.data,
            _anomalyclip_stretch(native, 37),
        ))
        surrogate._fit_positional_embedding(24)
        self.assertTrue(torch.equal(surrogate.model.visual.positional_embedding.data, native))


class SurrogateWiringTests(unittest.TestCase):
    def test_final_block_only(self) -> None:
        from adversarial_harness.config import AttackConfig

        self.assertEqual(SURROGATE_FEATURE_LAYERS, (24,))
        self.assertEqual(AttackConfig().feature_layers, (24,))

    def test_every_runner_uses_it_and_keys_reuse_on_it(self) -> None:
        for runner in RUNNERS:
            with self.subTest(runner=runner):
                source = (ROOT / runner).read_text(encoding="utf-8")
                self.assertIn("feature_layers=SURROGATE_FEATURE_LAYERS,", source)
                self.assertNotIn("feature_layers=(6, 12, 18, 24)", source)
                start = source.index("expected = {")
                end = source.index("reusable(pt_path, expected)")
                self.assertIn('"surrogate_clip": SURROGATE_CLIP,', source[start:end])
                self.assertIn(
                    '"feature_layers": list(attack_config.feature_layers),',
                    source[start:end],
                )

    def test_the_pinned_package_is_the_one_named(self) -> None:
        commit = SURROGATE_CLIP.split("@", 1)[1]
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        pinned = re.search(r"openai/CLIP\.git@([0-9a-f]{40})", requirements)
        self.assertIsNotNone(pinned)
        self.assertEqual(pinned.group(1), commit)

    def test_the_surrogate_loads_official_clip(self) -> None:
        source = (ROOT / "adversarial_harness" / "models.py").read_text(encoding="utf-8")
        body = source[source.index("class CLIPSurrogate"):source.index("class TargetAdapter")]
        self.assertIn("import clip as openai_clip", body)
        self.assertIn("openai_clip.load(", body)
        self.assertIn("openai_clip.tokenize", body)
        self.assertNotIn("library.load(", body)
        self.assertNotIn("DAPM_replace(", body.replace("do not call DAPM_replace", ""))


if __name__ == "__main__":
    unittest.main()
