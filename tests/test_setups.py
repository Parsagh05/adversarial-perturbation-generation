from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

from setup_catalog import SETUPS, compose_setup_id, effective_setup_id


def _launcher_table(**overrides: str) -> list[list[str]]:
    """Run the snippet train.sh embeds, under the given environment."""

    root = Path(__file__).resolve().parents[1]
    script = (root / "train.sh").read_text(encoding="utf-8")
    snippet = re.search(r"<<'PYEOF'\n(.*?)\nPYEOF", script, re.DOTALL)
    assert snippet is not None, "train.sh no longer embeds a catalog snippet"
    environment = {
        **os.environ,
        "SMOKE_TEST": "false",
        "SMOKE_STEPS": "2",
        "ATTACK_TRAIN_FRACTION": "1.0",
        **overrides,
    }
    completed = subprocess.run(
        [sys.executable, "-c", snippet.group(1)],
        capture_output=True, text=True, cwd=root, check=True, env=environment,
    )
    return [line.split("\t") for line in completed.stdout.strip().splitlines()]


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

    def test_prompt_family_selector_supports_all_three_choices(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = (root / "config.sh").read_text(encoding="utf-8")
        launcher = (root / "train.sh").read_text(encoding="utf-8")
        self.assertIn('PROMPT_SETUP="${PROMPT_SETUP:-both}"', config)
        self.assertRegex(launcher, r"frozen\|learnable\|both")
        self.assertIn('base_id="${id%_learnable_prompt}"', launcher)
        self.assertIn('prompt_folder="frozen_prompt"', launcher)
        self.assertIn('prompt_folder="learnable_prompt"', launcher)


class DerivedSetupIdTests(unittest.TestCase):
    """The ID must be a function of the settings that change the work.

    A stored name can drift from the parameters it claims to describe as soon
    as anything overrides those parameters.
    """

    def test_every_catalog_key_equals_its_own_derivation(self) -> None:
        for setup_id, setup in SETUPS.items():
            with self.subTest(setup_id=setup_id):
                self.assertEqual(effective_setup_id(setup), setup_id)

    def test_changing_steps_renames_the_setup(self) -> None:
        setup = SETUPS["steps500_eps4_margin_topk"]
        self.assertEqual(
            effective_setup_id(setup, 1200), "steps1200_eps4_margin_topk"
        )
        self.assertEqual(effective_setup_id(setup, 2), "steps2_eps4_margin_topk")

    def test_partial_train_fraction_is_folded_into_the_id(self) -> None:
        setup = SETUPS["steps500_eps4_margin_topk"]
        self.assertEqual(
            effective_setup_id(setup, attack_train_fraction=0.2),
            "steps500_eps4_margin_topk_train20",
        )
        self.assertEqual(
            effective_setup_id(setup, attack_train_fraction=0.05),
            "steps500_eps4_margin_topk_train5",
        )
        # A full run keeps the historical name so existing outputs stay valid.
        self.assertEqual(
            effective_setup_id(setup, attack_train_fraction=1.0),
            "steps500_eps4_margin_topk",
        )

    def test_learnable_suffix_stays_last_so_base_stripping_works(self) -> None:
        setup = SETUPS["steps500_eps2_margin_topk_learnable_prompt"]
        derived = effective_setup_id(setup, 1200, 0.25)
        self.assertTrue(derived.endswith("_learnable_prompt"))
        self.assertEqual(
            derived, "steps1200_eps2_margin_topk_train25_learnable_prompt"
        )

    def test_distinct_configurations_never_share_a_name(self) -> None:
        names = {
            compose_setup_id(steps, eps, loss, prompt, fraction)
            for steps in (500, 1200)
            for eps in ("2/255", "4/255")
            for loss in ("ce_focal_dice", "margin_topk")
            for prompt in ("frozen_winclip", "learnable_object_agnostic")
            for fraction in (1.0, 0.2)
        }
        self.assertEqual(len(names), 2 * 2 * 2 * 2 * 2)


class ShellLauncherTests(unittest.TestCase):
    def test_launcher_derives_the_table_from_the_catalog(self) -> None:
        script = (Path(__file__).resolve().parents[1] / "train.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(
            "SETUP_STEPS=(", script, "train.sh must not re-declare the matrix"
        )
        rows = _launcher_table()
        expected = [
            [
                setup_id, str(setup.steps), setup.epsilon_label,
                setup.loss_formulation, setup.prompt_mode, setup_id,
            ]
            for setup_id, setup in SETUPS.items()
        ]
        self.assertEqual(rows, expected)

    def test_launcher_output_name_follows_a_step_override(self) -> None:
        """The bug this guards: smoke steps changed the work but not the name."""

        rows = _launcher_table(SMOKE_TEST="true", SMOKE_STEPS="1200")
        for row in rows:
            requested, steps, effective = row[0], row[1], row[5]
            with self.subTest(requested=requested):
                self.assertEqual(steps, "1200")
                self.assertTrue(effective.startswith("steps1200_"), effective)
                self.assertNotEqual(effective, requested)

    def test_launcher_output_name_follows_the_train_fraction(self) -> None:
        rows = _launcher_table(ATTACK_TRAIN_FRACTION="0.2")
        for row in rows:
            effective = row[5]
            with self.subTest(effective=effective):
                self.assertIn("_train20", effective)

    def test_launcher_uses_the_effective_name_for_output_and_label(self) -> None:
        launcher = (Path(__file__).resolve().parents[1] / "train.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('setups/$prompt_folder/$effective_id', launcher)
        self.assertIn('export SETUP_ID="$effective_id"', launcher)
        self.assertNotIn('steps="$SMOKE_STEPS"', launcher)


if __name__ == "__main__":
    unittest.main()
