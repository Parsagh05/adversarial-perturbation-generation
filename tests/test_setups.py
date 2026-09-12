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
    derive_steps,
    effective_setup_id,
    epoch_grid,
    epsilon_grid,
    split_protocol_setting,
)

BASE = "ep7p14_cat100_img100"


def _launcher_table(**overrides: str) -> list[list[str]]:
    """Run the snippet train.sh embeds, under the given environment."""

    root = Path(__file__).resolve().parents[1]
    script = (root / "train.sh").read_text(encoding="utf-8")
    snippet = re.search(r"<<'PYEOF'\n(.*?)\nPYEOF", script, re.DOTALL)
    assert snippet is not None, "train.sh no longer embeds a catalog snippet"
    environment = {
        **os.environ,
        "SMOKE_TEST": "false",
        "SMOKE_EPOCHS": "0.02",
        "ATTACK_TRAIN_FRACTION": "1.0",
        **overrides,
    }
    completed = subprocess.run(
        [sys.executable, "-c", snippet.group(1)],
        capture_output=True, text=True, cwd=root, check=True, env=environment,
    )
    return [line.split("\t") for line in completed.stdout.strip().splitlines()]


class SetupCatalogTests(unittest.TestCase):
    def test_default_and_alternate_losses_are_separate(self) -> None:
        default_ids = [
            setup_id
            for setup_id, setup in SETUPS.items()
            if setup.prompt_mode == "frozen_winclip"
            and "ce_focal_dice" not in setup_id
        ]
        self.assertEqual(len(default_ids), 2)
        for default_id in default_ids:
            alternate_id = f"{default_id}_ce_focal_dice"
            self.assertIn(alternate_id, SETUPS)
            self.assertEqual(SETUPS[default_id].loss_formulation, "margin_topk")
            self.assertEqual(
                SETUPS[alternate_id].loss_formulation, "ce_focal_dice"
            )
            self.assertEqual(SETUPS[default_id].epochs, SETUPS[alternate_id].epochs)
            self.assertEqual(
                SETUPS[default_id].epsilon, SETUPS[alternate_id].epsilon
            )

    def test_each_frozen_setup_has_a_learnable_counterpart(self) -> None:
        frozen = {
            setup_id: setup
            for setup_id, setup in SETUPS.items()
            if setup.prompt_mode == "frozen_winclip"
        }
        self.assertEqual(len(frozen), 4)
        for setup_id, setup in frozen.items():
            counterpart_id = f"{setup_id}_learnable_prompt"
            self.assertIn(counterpart_id, SETUPS)
            counterpart = SETUPS[counterpart_id]
            self.assertEqual(counterpart.prompt_mode, "learnable_object_agnostic")
            self.assertEqual(counterpart.epochs, setup.epochs)
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


class DeriveStepsTests(unittest.TestCase):
    """An epoch is one pass over whatever the delta trains on."""

    def test_reproduces_the_historical_step_counts(self) -> None:
        # MVTec, balanced protocol: the budget that matches 800 / 200 / 100.
        self.assertEqual(derive_steps(7.14, 224, 2), 800)   # per-dataset
        self.assertEqual(derive_steps(100, 14, 8), 200)     # per-category
        self.assertEqual(derive_steps(100, 1, 1), 100)      # per-image

    def test_one_image_makes_epochs_and_steps_the_same_number(self) -> None:
        for epochs in (1, 8, 100):
            with self.subTest(epochs=epochs):
                self.assertEqual(derive_steps(epochs, 1, 1), epochs)

    def test_budget_is_invariant_to_training_set_size(self) -> None:
        # Twice the images at the same budget means twice the updates, so each
        # image is still seen the same number of times.
        small = derive_steps(7.14, 224, 2)
        large = derive_steps(7.14, 448, 2)
        self.assertAlmostEqual(large / small, 2.0, places=1)

    def test_rejects_degenerate_inputs(self) -> None:
        for images, batch in ((0, 2), (10, 0), (-1, 1)):
            with self.subTest(images=images, batch=batch):
                with self.assertRaises(ValueError):
                    derive_steps(10, images, batch)

    def test_never_returns_zero(self) -> None:
        self.assertEqual(derive_steps(0.001, 4, 8), 1)


class DerivedSetupIdTests(unittest.TestCase):
    """The ID must be a function of the settings that change the work."""

    def test_every_catalog_key_equals_its_own_derivation(self) -> None:
        for setup_id, setup in SETUPS.items():
            with self.subTest(setup_id=setup_id):
                self.assertEqual(effective_setup_id(setup), setup_id)

    def test_changing_the_budget_renames_the_setup(self) -> None:
        setup = SETUPS[f"{BASE}_eps4"]
        self.assertEqual(effective_setup_id(setup, 12), "ep12_eps4")
        self.assertEqual(effective_setup_id(setup, 0.5), "ep0p5_eps4")

    def test_partial_train_fraction_is_folded_into_the_id(self) -> None:
        setup = SETUPS[f"{BASE}_eps4"]
        self.assertEqual(
            effective_setup_id(setup, attack_train_fraction=0.2),
            f"{BASE}_eps4_train20",
        )
        # A full run keeps the plain name so existing outputs stay valid.
        self.assertEqual(
            effective_setup_id(setup, attack_train_fraction=1.0),
            f"{BASE}_eps4",
        )

    def test_learnable_suffix_stays_last_so_base_stripping_works(self) -> None:
        setup = SETUPS[f"{BASE}_eps2_learnable_prompt"]
        derived = effective_setup_id(setup, 12, 0.25)
        self.assertTrue(derived.endswith("_learnable_prompt"))
        self.assertEqual(
            derived, "ep12_eps2_train25_learnable_prompt"
        )

    def test_distinct_configurations_never_share_a_name(self) -> None:
        names = {
            compose_setup_id(ep, ep, ep, eps, loss, prompt, fraction)
            for ep in (7.14, 100)
            for eps in ("2/255", "4/255")
            for loss in ("ce_focal_dice", "margin_topk")
            for prompt in ("frozen_winclip", "learnable_object_agnostic")
            for fraction in (1.0, 0.2)
        }
        self.assertEqual(len(names), 2 * 2 * 2 * 2 * 2)


class SetupGridTests(unittest.TestCase):
    """The matrix is generated from parameter lists, not written out by hand."""

    def test_default_grid_reproduces_the_historical_budgets(self) -> None:
        self.assertEqual(epoch_grid(), ((7.14, 100.0, 100.0),))
        self.assertEqual(epsilon_grid(), ("2/255", "4/255"))
        self.assertEqual(len(SETUPS), 8)

    def test_adding_a_budget_widens_every_family_at_once(self) -> None:
        widened = build_setups(
            epochs_grid=((7.14, 100, 100), (12, 12, 12)),
            epsilons=("2/255", "4/255"),
        )
        self.assertEqual(len(widened), 16)
        for loss_suffix in ("", "_ce_focal_dice"):
            for prompt_suffix in ("", "_learnable_prompt"):
                for epsilon in ("eps2", "eps4"):
                    name = f"ep12_{epsilon}{loss_suffix}{prompt_suffix}"
                    with self.subTest(name=name):
                        self.assertIn(name, widened)
                        self.assertEqual(widened[name].epochs, 12)

    def test_a_uniform_budget_uses_the_compact_name(self) -> None:
        narrowed = build_setups(
            epochs_grid=((12, 12, 12),), epsilons=("2/255", "4/255")
        )
        self.assertEqual(len(narrowed), 8)
        self.assertTrue(all(setup.epochs == 12 for setup in narrowed.values()))
        self.assertTrue(all(name.startswith("ep12_") for name in narrowed))
        self.assertTrue(all("_cat" not in name for name in narrowed))

    def test_epsilon_grid_widens_the_same_way(self) -> None:
        widened = build_setups(
            epochs_grid=((12, 12, 12),), epsilons=("2/255", "4/255", "8/255")
        )
        self.assertEqual(len(widened), 12)
        self.assertIn("ep12_eps8", widened)
        self.assertAlmostEqual(widened["ep12_eps8"].epsilon, 8 / 255)

    def test_generated_entries_are_self_naming(self) -> None:
        for grid in (
            ((3, 3, 3),),
            ((7.14, 100, 100), (12, 12, 12)),
            ((1, 40, 20), (12, 300, 150)),
        ):
            generated = build_setups(epochs_grid=grid, epsilons=("1/255", "16/255"))
            for setup_id, setup in generated.items():
                with self.subTest(setup_id=setup_id):
                    self.assertEqual(effective_setup_id(setup), setup_id)

    def test_grid_rejects_duplicates_and_non_positive_budgets(self) -> None:
        for value in ("12,12", "0", "-5"):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"SETUP_EPOCHS": value}):
                    with self.assertRaises(ValueError):
                        epoch_grid()


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
                setup_id, str(setup.epochs), str(setup.category_epochs),
                str(setup.image_epochs), setup.epsilon_label,
                setup.loss_formulation, setup.prompt_mode, setup_id,
            ]
            for setup_id, setup in SETUPS.items()
        ]
        self.assertEqual(rows, expected)

    def test_launcher_exports_a_budget_per_scope(self) -> None:
        rows = _launcher_table(SETUP_EPOCHS="7.14:100:100")
        for row in rows:
            with self.subTest(requested=row[0]):
                self.assertEqual((row[1], row[2], row[3]), ("7.14", "100.0", "100.0"))
                self.assertTrue(row[7].startswith("ep7p14_cat100_img100_"), row[7])
        launcher = (Path(__file__).resolve().parents[1] / "train.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('export PER_DATASET_EPOCHS="$epochs"', launcher)
        self.assertIn('export PER_CATEGORY_EPOCHS="$category_epochs"', launcher)
        self.assertIn('export PER_IMAGE_EPOCHS="$image_epochs"', launcher)

    def test_smoke_override_collapses_every_scope(self) -> None:
        rows = _launcher_table(SMOKE_TEST="true", SMOKE_EPOCHS="3")
        for row in rows:
            with self.subTest(requested=row[0]):
                self.assertEqual((row[1], row[2], row[3]), ("3.0", "3.0", "3.0"))
                # uniform again, so the name returns to its compact form
                self.assertTrue(row[7].startswith("ep3_"), row[7])
                self.assertNotIn("_cat", row[7])

    def test_launcher_output_name_follows_the_train_fraction(self) -> None:
        rows = _launcher_table(ATTACK_TRAIN_FRACTION="0.2")
        for row in rows:
            with self.subTest(effective=row[7]):
                self.assertIn("_train20", row[7])

    def test_launcher_uses_the_effective_name_for_output_and_label(self) -> None:
        launcher = (Path(__file__).resolve().parents[1] / "train.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('setups/$prompt_folder/$effective_id', launcher)
        self.assertIn('export SETUP_ID="$effective_id"', launcher)


class SplitProtocolTests(unittest.TestCase):
    """balanced downsamples each category to min(); full keeps every image."""

    def test_balanced_is_the_default_and_adds_no_name_component(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPLIT_PROTOCOL", None)
            self.assertEqual(split_protocol_setting(), "balanced")
        setup = SETUPS[f"{BASE}_eps4"]
        self.assertEqual(effective_setup_id(setup), f"{BASE}_eps4")

    def test_full_is_named_so_the_protocols_cannot_collide(self) -> None:
        setup = SETUPS[f"{BASE}_eps4"]
        self.assertEqual(
            effective_setup_id(setup, None, 1.0, "full"),
            f"{BASE}_eps4_full",
        )
        self.assertEqual(
            effective_setup_id(setup, None, 0.2, "full"),
            f"{BASE}_eps4_full_train20",
        )

    def test_learnable_suffix_stays_last(self) -> None:
        setup = SETUPS[f"{BASE}_eps2_learnable_prompt"]
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


if __name__ == "__main__":
    unittest.main()
