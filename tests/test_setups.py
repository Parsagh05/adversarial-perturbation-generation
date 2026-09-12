from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import unittest
from unittest import mock

from setup_catalog import (
    SETUPS,
    build_setups,
    compose_setup_id,
    effective_setup_id,
    epsilon_grid,
    split_protocol_setting,
    step_grid,
)


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
            compose_setup_id(steps, steps, steps, eps, loss, prompt, fraction)
            for steps in (500, 1200)
            for eps in ("2/255", "4/255")
            for loss in ("ce_focal_dice", "margin_topk")
            for prompt in ("frozen_winclip", "learnable_object_agnostic")
            for fraction in (1.0, 0.2)
        }
        self.assertEqual(len(names), 2 * 2 * 2 * 2 * 2)


class SetupGridTests(unittest.TestCase):
    """The matrix is generated from parameter lists, not written out by hand."""

    def test_default_grid_reproduces_the_historical_matrix(self) -> None:
        self.assertEqual(step_grid(), ((500, 500, 500), (800, 800, 800)))
        self.assertEqual(epsilon_grid(), ("2/255", "4/255"))
        self.assertEqual(len(SETUPS), 16)

    def test_adding_a_step_count_widens_every_family_at_once(self) -> None:
        widened = build_setups(
            steps_grid=((500, 500, 500), (800, 800, 800), (1200, 1200, 1200)),
            epsilons=("2/255", "4/255"),
        )
        self.assertEqual(len(widened), 24)
        for loss_suffix in ("", "_margin_topk"):
            for prompt_suffix in ("", "_learnable_prompt"):
                for epsilon in ("eps2", "eps4"):
                    name = f"steps1200_{epsilon}{loss_suffix}{prompt_suffix}"
                    with self.subTest(name=name):
                        self.assertIn(name, widened)
                        self.assertEqual(widened[name].steps, 1200)

    def test_a_single_step_count_halves_the_matrix(self) -> None:
        narrowed = build_setups(
            steps_grid=((1200, 1200, 1200),), epsilons=("2/255", "4/255")
        )
        self.assertEqual(len(narrowed), 8)
        self.assertTrue(all(setup.steps == 1200 for setup in narrowed.values()))
        self.assertTrue(all(name.startswith("steps1200_") for name in narrowed))

    def test_epsilon_grid_widens_the_same_way(self) -> None:
        widened = build_setups(
            steps_grid=((500, 500, 500),), epsilons=("2/255", "4/255", "8/255")
        )
        self.assertEqual(len(widened), 12)
        self.assertIn("steps500_eps8_margin_topk", widened)
        self.assertAlmostEqual(widened["steps500_eps8_margin_topk"].epsilon, 8 / 255)

    def test_generated_entries_are_self_naming(self) -> None:
        for grid in (
            ((300, 300, 300),),
            ((500, 500, 500), (800, 800, 800)),
            ((100, 40, 20), (1200, 300, 150)),
        ):
            generated = build_setups(steps_grid=grid, epsilons=("1/255", "16/255"))
            for setup_id, setup in generated.items():
                with self.subTest(setup_id=setup_id):
                    self.assertEqual(effective_setup_id(setup), setup_id)

    def test_grid_rejects_duplicates_and_non_positive_steps(self) -> None:
        for value in ("500,500", "0", "-100"):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"SETUP_STEPS": value}):
                    with self.assertRaises(ValueError):
                        step_grid()


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
                setup_id, str(setup.steps), str(setup.category_steps),
                str(setup.image_steps), setup.epsilon_label,
                setup.loss_formulation, setup.prompt_mode, setup_id,
            ]
            for setup_id, setup in SETUPS.items()
        ]
        self.assertEqual(rows, expected)

    def test_launcher_output_name_follows_a_step_override(self) -> None:
        """The bug this guards: smoke steps changed the work but not the name."""

        rows = _launcher_table(SMOKE_TEST="true", SMOKE_STEPS="1200")
        # 500 and 800 both become 1200, so the matrix must collapse rather than
        # emit two rows that would overwrite each other's output directory.
        self.assertEqual(len(rows), len(SETUPS) // len(step_grid()))
        self.assertEqual(len({row[7] for row in rows}), len(rows))
        for row in rows:
            requested, steps, effective = row[0], row[1], row[7]
            with self.subTest(requested=requested):
                self.assertEqual(steps, "1200")
                self.assertTrue(effective.startswith("steps1200_"), effective)
                self.assertNotEqual(effective, requested)

    def test_launcher_output_name_follows_the_train_fraction(self) -> None:
        rows = _launcher_table(ATTACK_TRAIN_FRACTION="0.2")
        for row in rows:
            effective = row[7]
            with self.subTest(effective=effective):
                self.assertIn("_train20", effective)

    def test_launcher_exports_a_step_count_per_scope(self) -> None:
        rows = _launcher_table(SETUP_STEPS="800:200:100")
        for row in rows:
            requested, dataset, category, image, effective = (
                row[0], row[1], row[2], row[3], row[7]
            )
            with self.subTest(requested=requested):
                self.assertEqual((dataset, category, image), ("800", "200", "100"))
                self.assertTrue(effective.startswith("steps800_cat200_img100_"),
                                effective)
        launcher = (Path(__file__).resolve().parents[1] / "train.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('export PER_DATASET_STEPS="$steps"', launcher)
        self.assertIn('export PER_CATEGORY_STEPS="$category_steps"', launcher)
        self.assertIn('export PER_IMAGE_STEPS="$image_steps"', launcher)

    def test_smoke_override_collapses_every_scope(self) -> None:
        rows = _launcher_table(
            SETUP_STEPS="800:200:100", SMOKE_TEST="true", SMOKE_STEPS="3"
        )
        for row in rows:
            with self.subTest(requested=row[0]):
                self.assertEqual((row[1], row[2], row[3]), ("3", "3", "3"))
                # uniform again, so the name returns to its compact form
                self.assertTrue(row[7].startswith("steps3_"), row[7])
                self.assertNotIn("_cat", row[7])

    def test_launcher_uses_the_effective_name_for_output_and_label(self) -> None:
        launcher = (Path(__file__).resolve().parents[1] / "train.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('setups/$prompt_folder/$effective_id', launcher)
        self.assertIn('export SETUP_ID="$effective_id"', launcher)
        self.assertNotIn('steps="$SMOKE_STEPS"', launcher)


if __name__ == "__main__":
    unittest.main()


class SplitProtocolTests(unittest.TestCase):
    """balanced downsamples each category to min(); full keeps every image."""

    def test_balanced_is_the_default_and_adds_no_name_component(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPLIT_PROTOCOL", None)
            self.assertEqual(split_protocol_setting(), "balanced")
        setup = SETUPS["steps500_eps4_margin_topk"]
        self.assertEqual(effective_setup_id(setup), "steps500_eps4_margin_topk")

    def test_full_is_named_so_the_protocols_cannot_collide(self) -> None:
        setup = SETUPS["steps500_eps4_margin_topk"]
        self.assertEqual(
            effective_setup_id(setup, None, 1.0, "full"),
            "steps500_eps4_margin_topk_full",
        )
        self.assertEqual(
            effective_setup_id(setup, None, 0.2, "full"),
            "steps500_eps4_margin_topk_full_train20",
        )

    def test_learnable_suffix_stays_last(self) -> None:
        setup = SETUPS["steps500_eps2_margin_topk_learnable_prompt"]
        derived = effective_setup_id(setup, None, 1.0, "full")
        self.assertTrue(derived.endswith("_learnable_prompt"))
        self.assertIn("_full_", derived)

    def test_unknown_protocol_is_rejected(self) -> None:
        with mock.patch.dict(os.environ, {"SPLIT_PROTOCOL": "kfold"}):
            with self.assertRaises(ValueError):
                split_protocol_setting()

    def test_launcher_names_carry_the_protocol(self) -> None:
        rows = _launcher_table(SPLIT_PROTOCOL="full")
        self.assertTrue(all(row[7].count("_full") == 1 for row in rows))
        balanced = _launcher_table(SPLIT_PROTOCOL="balanced")
        self.assertTrue(all("_full" not in row[7] for row in balanced))

    def test_config_exposes_the_switch(self) -> None:
        config = (Path(__file__).resolve().parents[1] / "config.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('SPLIT_PROTOCOL="${SPLIT_PROTOCOL:-balanced}"', config)
