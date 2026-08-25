from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import unittest

from setup_catalog import SETUPS


class SetupCatalogTests(unittest.TestCase):
    def test_legacy_and_relaxed_setups_are_separate(self) -> None:
        legacy_ids = [
            setup_id
            for setup_id, setup in SETUPS.items()
            if setup.prompt_mode == "frozen_winclip"
            and "margin_topk" not in setup_id
            and "_gradnorm" not in setup_id
        ]
        self.assertEqual(len(legacy_ids), 4)
        for legacy_id in legacy_ids:
            relaxed_id = f"{legacy_id}_margin_topk"
            self.assertIn(relaxed_id, SETUPS)
            self.assertEqual(SETUPS[legacy_id].loss_formulation, "ce_focal_dice")
            self.assertEqual(SETUPS[relaxed_id].loss_formulation, "margin_topk")
            self.assertEqual(SETUPS[legacy_id].steps, SETUPS[relaxed_id].steps)
            self.assertEqual(SETUPS[legacy_id].epsilon, SETUPS[relaxed_id].epsilon)

    def test_each_frozen_setup_has_a_learnable_counterpart(self) -> None:
        frozen = {
            setup_id: setup
            for setup_id, setup in SETUPS.items()
            if setup.prompt_mode == "frozen_winclip"
        }
        self.assertEqual(len(frozen), 16)
        for setup_id, setup in frozen.items():
            counterpart_id = f"{setup_id}_learnable_prompt"
            self.assertIn(counterpart_id, SETUPS)
            counterpart = SETUPS[counterpart_id]
            self.assertEqual(counterpart.prompt_mode, "learnable_object_agnostic")
            self.assertEqual(counterpart.steps, setup.steps)
            self.assertEqual(counterpart.epsilon, setup.epsilon)
            self.assertEqual(counterpart.loss_formulation, setup.loss_formulation)
            self.assertEqual(
                counterpart.gradient_normalization, setup.gradient_normalization
            )
            self.assertEqual(counterpart.loss_modes, setup.loss_modes)

    def test_shell_launcher_derives_the_table_from_the_catalog(self) -> None:
        """train.sh embeds a snippet that reads setup_catalog.py; run it and compare.

        The launcher used to duplicate the matrix in parallel bash arrays, which
        could silently drift from the catalog that audit_generation.py reads.
        """
        root = Path(__file__).resolve().parents[1]
        script = (root / "train.sh").read_text(encoding="utf-8")
        self.assertNotIn(
            "SETUP_STEPS=(", script, "train.sh must not re-declare the matrix"
        )
        snippet = re.search(r"<<'PYEOF'\n(.*?)\nPYEOF", script, re.DOTALL)
        self.assertIsNotNone(snippet)

        emitted = subprocess.run(
            [sys.executable, "-c", snippet.group(1)],
            capture_output=True, text=True, cwd=root, check=True,
        ).stdout.strip().splitlines()

        expected = [
            "\t".join((
                setup_id, str(setup.steps), setup.epsilon_label,
                setup.loss_formulation, setup.prompt_mode,
                setup.gradient_normalization, ",".join(setup.loss_modes),
            ))
            for setup_id, setup in SETUPS.items()
        ]
        self.assertEqual(emitted, expected)


class GradientNormalizedSetupTests(unittest.TestCase):
    def test_every_base_setup_has_a_gradnorm_counterpart(self) -> None:
        base = [sid for sid in SETUPS if "_gradnorm" not in sid]
        self.assertEqual(len(base), 16)
        for base_id in base:
            if base_id.endswith("_learnable_prompt"):
                gradnorm_id = base_id.replace("_learnable_prompt", "_gradnorm_learnable_prompt")
            else:
                gradnorm_id = f"{base_id}_gradnorm"
            self.assertIn(gradnorm_id, SETUPS)
            original, variant = SETUPS[base_id], SETUPS[gradnorm_id]
            self.assertEqual(original.steps, variant.steps)
            self.assertEqual(original.epsilon, variant.epsilon)
            self.assertEqual(original.loss_formulation, variant.loss_formulation)
            self.assertEqual(original.prompt_mode, variant.prompt_mode)
            self.assertEqual(original.gradient_normalization, "none")
            self.assertEqual(variant.gradient_normalization, "l2")

    def test_gradnorm_setups_run_combined_only(self) -> None:
        # A single-component objective is unchanged by positive rescaling once
        # sign() is taken, so global/local would duplicate the base setup.
        for setup_id, setup in SETUPS.items():
            with self.subTest(setup_id=setup_id):
                if setup.gradient_normalization == "l2":
                    self.assertEqual(setup.loss_modes, ("combined",))
                else:
                    self.assertEqual(setup.loss_modes, ("global", "local", "combined"))

    def test_catalog_size(self) -> None:
        self.assertEqual(len(SETUPS), 32)
        self.assertEqual(
            sum(1 for s in SETUPS.values() if s.gradient_normalization == "l2"), 16
        )
