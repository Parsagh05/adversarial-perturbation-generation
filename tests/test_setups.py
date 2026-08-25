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
            if setup.prompt_mode == "frozen_winclip" and "margin_topk" not in setup_id
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
        self.assertEqual(len(frozen), 8)
        for setup_id, setup in frozen.items():
            counterpart_id = f"{setup_id}_learnable_prompt"
            self.assertIn(counterpart_id, SETUPS)
            counterpart = SETUPS[counterpart_id]
            self.assertEqual(counterpart.prompt_mode, "learnable_object_agnostic")
            self.assertEqual(counterpart.steps, setup.steps)
            self.assertEqual(counterpart.epsilon, setup.epsilon)
            self.assertEqual(counterpart.loss_formulation, setup.loss_formulation)

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
            ))
            for setup_id, setup in SETUPS.items()
        ]
        self.assertEqual(emitted, expected)

    def test_prompt_family_selector_supports_all_three_choices(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = (root / "config.sh").read_text(encoding="utf-8")
        launcher = (root / "train.sh").read_text(encoding="utf-8")
        self.assertIn('PROMPT_SETUP="${PROMPT_SETUP:-both}"', config)
        self.assertRegex(launcher, r"frozen\|learnable\|both")
        self.assertIn('base_id="${id%_learnable_prompt}"', launcher)
        self.assertIn('prompt_folder="frozen_prompt"', launcher)
        self.assertIn('prompt_folder="learnable_prompt"', launcher)
        self.assertIn('setups/$prompt_folder/$id', launcher)
