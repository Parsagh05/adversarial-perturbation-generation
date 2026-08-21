from __future__ import annotations

from pathlib import Path
import re
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

    def test_shell_launcher_matches_catalog(self) -> None:
        script = (Path(__file__).resolve().parents[1] / "train.sh").read_text(
            encoding="utf-8"
        )

        def array(name: str) -> list[str]:
            match = re.search(rf"{name}=\((.*?)\)", script, re.DOTALL)
            self.assertIsNotNone(match)
            return match.group(1).split()

        self.assertEqual(array("SETUP_IDS"), list(SETUPS))
        self.assertEqual(
            [int(value) for value in array("SETUP_STEPS")],
            [setup.steps for setup in SETUPS.values()],
        )
        self.assertEqual(
            array("SETUP_EPS"), [setup.epsilon_label for setup in SETUPS.values()]
        )
        self.assertEqual(
            array("SETUP_LOSSES"),
            [setup.loss_formulation for setup in SETUPS.values()],
        )
        self.assertEqual(
            array("SETUP_PROMPTS"),
            [setup.prompt_mode for setup in SETUPS.values()],
        )

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
