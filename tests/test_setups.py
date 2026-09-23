from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import shutil
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
    full_data_cross_setting,
    SCOPE_DIRECTORIES,
    scope_output_path,
    settings_tag,
    split_protocol_setting,
)

BASE = "ep7p14_cat100_img100"


NL_CONST = chr(10)
BS_CONST = chr(92)

def _launcher_snippet() -> str:
    """The catalog snippet train.sh embeds."""

    script = (Path(__file__).resolve().parents[1] / 'train.sh').read_text(
        encoding='utf-8'
    )
    found = re.search(r"<<'PYEOF'" + chr(10) + '(.*?)' + chr(10) + "PYEOF", script, re.DOTALL)
    assert found is not None, 'train.sh no longer embeds a catalog snippet'
    return found.group(1)

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
    # Split on the unit separator, exactly as train.sh does. Tab would be
    # wrong twice over: bash collapses runs of it, and a bare strip()
    # would eat a trailing empty snapshot field off the final row.
    return [
        line.split(chr(31))
        for line in completed.stdout.strip("\n").splitlines()
    ]


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
            compose_setup_id(ep, ep, ep, ep, eps, loss, prompt, fraction)
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
        self.assertEqual(epoch_grid(), ((7.14, 7.14, 100.0, 100.0),))
        self.assertEqual(epsilon_grid(), ("2/255", "4/255"))
        self.assertEqual(len(SETUPS), 8)

    def test_adding_a_budget_widens_every_family_at_once(self) -> None:
        widened = build_setups(
            epochs_grid=((7.14, 7.14, 100, 100), (12, 12, 12, 12)),
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
            epochs_grid=((12, 12, 12, 12),), epsilons=("2/255", "4/255")
        )
        self.assertEqual(len(narrowed), 8)
        self.assertTrue(all(setup.epochs == 12 for setup in narrowed.values()))
        self.assertTrue(all(name.startswith("ep12_") for name in narrowed))
        self.assertTrue(all("_cat" not in name for name in narrowed))

    def test_epsilon_grid_widens_the_same_way(self) -> None:
        widened = build_setups(
            epochs_grid=((12, 12, 12, 12),), epsilons=("2/255", "4/255", "8/255")
        )
        self.assertEqual(len(widened), 12)
        self.assertIn("ep12_eps8", widened)
        self.assertAlmostEqual(widened["ep12_eps8"].epsilon, 8 / 255)

    def test_generated_entries_are_self_naming(self) -> None:
        for grid in (
            ((3, 3, 3, 3),),
            ((7.14, 7.14, 100, 100), (12, 12, 12, 12)),
            ((1, 40, 20, 20), (12, 5, 300, 150)),
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
                setup_id, str(setup.epochs), str(setup.cross_epochs),
                str(setup.category_epochs),
                str(setup.image_epochs), setup.epsilon_label,
                setup.loss_formulation, setup.prompt_mode,
                effective_setup_id(setup, full_data_cross=True),
                "",
                settings_tag(
                    setup.epsilon_label, setup.loss_formulation, 1.0,
                    "balanced", True,
                ),
                *(
                    scope_output_path(
                        scope,
                        (setup.epochs, setup.cross_epochs,
                         setup.category_epochs, setup.image_epochs),
                        setup.prompt_mode,
                        settings_tag(
                            setup.epsilon_label, setup.loss_formulation, 1.0,
                            "balanced", True,
                        ),
                    )
                    for scope in SCOPE_DIRECTORIES
                ),
            ]
            for setup_id, setup in SETUPS.items()
        ]
        self.assertEqual(rows, expected)

    def test_launcher_exports_a_budget_per_scope(self) -> None:
        rows = _launcher_table(SETUP_EPOCHS="7.14:100:100")
        for row in rows:
            with self.subTest(requested=row[0]):
                self.assertEqual(
                    (row[1], row[2], row[3], row[4]),
                    ("7.14", "7.14", "100.0", "100.0"),
                )
                self.assertTrue(row[8].startswith("ep7p14_cat100_img100_"), row[8])
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
                self.assertEqual(
                    (row[1], row[2], row[3], row[4]),
                    ("3.0", "3.0", "3.0", "3.0"),
                )
                # uniform again, so the name returns to its compact form
                self.assertTrue(row[8].startswith("ep3_"), row[8])
                self.assertNotIn("_cat", row[8])

    def test_launcher_output_name_follows_the_train_fraction(self) -> None:
        rows = _launcher_table(ATTACK_TRAIN_FRACTION="0.2")
        for row in rows:
            with self.subTest(effective=row[8]):
                self.assertIn("_train20", row[8])

    def test_launcher_builds_the_tree_from_the_emitted_paths(self) -> None:
        launcher = (Path(__file__).resolve().parents[1] / "train.sh").read_text(
            encoding="utf-8"
        )
        # The layout lives in scope_output_path, not in shell string building.
        self.assertIn('settings_root="$PIPELINE_OUTPUT/setups/$settings_tag"', launcher)
        self.assertIn(
            'export BUNDLE_PER_DATASET="$PIPELINE_OUTPUT/setups/$bundle_per_dataset"',
            launcher,
        )
        self.assertIn('export PROTOCOL_DIR="$settings_root/protocol"', launcher)
        self.assertIn('export SETUP_ID="$effective_id"', launcher)
        self.assertNotIn("setups/$prompt_folder/$effective_id", launcher)


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

    def test_both_cross_modes_get_distinct_full_protocol_names(self) -> None:
        setup = SETUPS[f"{BASE}_eps4"]
        self.assertEqual(
            effective_setup_id(setup, None, 1.0, "full", False),
            f"{BASE}_eps4_full_halfcross",
        )
        self.assertEqual(
            effective_setup_id(setup, None, 1.0, "full", True),
            f"{BASE}_eps4_full_fullcross",
        )

    def test_balanced_protocol_supports_both_cross_modes(self) -> None:
        setup = SETUPS[f"{BASE}_eps4"]
        self.assertEqual(
            effective_setup_id(setup, None, 1.0, "balanced", False),
            f"{BASE}_eps4_halfcross",
        )
        self.assertEqual(
            effective_setup_id(setup, None, 1.0, "balanced", True),
            f"{BASE}_eps4_fullcross",
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
        self.assertTrue(all("_full_fullcross" in row[8] for row in rows))
        balanced = _launcher_table(SPLIT_PROTOCOL="balanced")
        self.assertTrue(all("_full_fullcross" not in row[8] for row in balanced))
        self.assertTrue(all("_fullcross" in row[8] for row in balanced))

    def test_launcher_names_carry_the_cross_half_mode(self) -> None:
        rows = _launcher_table(SPLIT_PROTOCOL="full", FULL_DATA_CROSS="false")
        self.assertTrue(all("_full_halfcross" in row[8] for row in rows))
        self.assertTrue(all(row[8].endswith("_learnable_prompt") for row in rows[4:]))

    def test_full_data_cross_defaults_true_and_validates(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FULL_DATA_CROSS", None)
            self.assertTrue(full_data_cross_setting())
        with mock.patch.dict(os.environ, {"FULL_DATA_CROSS": "false"}):
            self.assertFalse(full_data_cross_setting())
        with mock.patch.dict(os.environ, {"FULL_DATA_CROSS": "yes"}):
            with self.assertRaises(ValueError):
                full_data_cross_setting()

    def test_build_setups_threads_the_cross_mode_into_ids(self) -> None:
        generated = build_setups(
            epochs_grid=((12, 12, 12, 12),),
            epsilons=("4/255",),
            split_protocol="full",
            full_data_cross=False,
        )
        self.assertEqual(len(generated), 4)
        self.assertTrue(all("_full_halfcross" in name for name in generated))

    def test_config_exposes_the_switch(self) -> None:
        config = (Path(__file__).resolve().parents[1] / "config.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('SPLIT_PROTOCOL="${SPLIT_PROTOCOL:-balanced}"', config)
        self.assertIn('FULL_DATA_CROSS="${FULL_DATA_CROSS:-true}"', config)


if __name__ == "__main__":
    unittest.main()


class StepSizeScheduleIdTests(unittest.TestCase):
    """The step-size schedule changes the perturbation, so it names the output.

    A decaying step depends on the total step count, so the first N steps of a
    long run are not an N-step run. constant removes that coupling, which is
    what makes one long run sliceable into shorter budgets.
    """

    def _setup(self):
        return SETUPS[f"{BASE}_eps4"]

    def test_constant_is_the_default_and_adds_nothing(self) -> None:
        setup = self._setup()
        self.assertEqual(effective_setup_id(setup), f"{BASE}_eps4")
        self.assertEqual(
            effective_setup_id(setup, None, 1.0, "balanced", None, "constant"),
            f"{BASE}_eps4",
        )

    def test_decaying_schedules_name_themselves(self) -> None:
        setup = self._setup()
        self.assertEqual(
            effective_setup_id(setup, None, 1.0, "balanced", None, "linear"),
            f"{BASE}_eps4_linear_step",
        )
        self.assertEqual(
            effective_setup_id(setup, None, 1.0, "balanced", None, "cosine"),
            f"{BASE}_eps4_cosine_step",
        )

    def test_schedules_never_share_a_name(self) -> None:
        setup = self._setup()
        names = {
            effective_setup_id(setup, None, 0.2, "full", True, schedule)
            for schedule in ("constant", "linear", "cosine")
        }
        self.assertEqual(len(names), 3)

    def test_learnable_prompt_stays_last(self) -> None:
        setup = SETUPS[f"{BASE}_eps4_learnable_prompt"]
        derived = effective_setup_id(setup, None, 1.0, "full", True, "linear")
        self.assertTrue(derived.endswith("_learnable_prompt"))
        self.assertIn("_linear_step_", derived)

    def test_unknown_schedules_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "step_size_schedule"):
            effective_setup_id(self._setup(), None, 1.0, "balanced", None, "sqrt")

    def test_the_environment_setting_defaults_to_constant(self) -> None:
        from setup_catalog import step_size_schedule_setting

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STEP_SIZE_SCHEDULE", None)
            self.assertEqual(step_size_schedule_setting(), "constant")
        with mock.patch.dict(os.environ, {"STEP_SIZE_SCHEDULE": "LINEAR"}):
            self.assertEqual(step_size_schedule_setting(), "linear")
        with mock.patch.dict(os.environ, {"STEP_SIZE_SCHEDULE": "sqrt"}):
            with self.assertRaises(ValueError):
                step_size_schedule_setting()


class MarginHingeIdTests(unittest.TestCase):
    """The hinge changes the perturbation, so it names the output."""

    def _setup(self, loss: str = ""):
        return SETUPS[f"{BASE}_eps4{loss}"]

    def test_no_hinge_adds_nothing(self) -> None:
        self.assertEqual(effective_setup_id(self._setup()), f"{BASE}_eps4")

    def test_each_displacement_gets_its_own_name(self) -> None:
        names = {
            effective_setup_id(
                self._setup(), None, 1.0, "balanced", None, "constant", value
            )
            for value in (None, 0.0, 0.1, 0.25)
        }
        self.assertEqual(len(names), 4)
        self.assertIn(f"{BASE}_eps4_hinge0p25", names)
        self.assertIn(f"{BASE}_eps4_hinge0", names)

    def test_the_bounded_loss_never_carries_the_tag(self) -> None:
        # focal and Dice are bounded, so ce_focal_dice has nothing to hinge.
        derived = effective_setup_id(
            self._setup("_ce_focal_dice"), None, 1.0, "balanced", None,
            "constant", 0.25,
        )
        self.assertNotIn("hinge", derived)

    def test_learnable_prompt_stays_last(self) -> None:
        derived = effective_setup_id(
            SETUPS[f"{BASE}_eps4_learnable_prompt"], None, 1.0, "full", True,
            "constant", 0.25,
        )
        self.assertTrue(derived.endswith("_learnable_prompt"))
        self.assertIn("_hinge0p25_", derived)

    def test_the_environment_setting_is_off_unless_asked(self) -> None:
        from setup_catalog import margin_hinge_setting

        for value in ("", "none", "off"):
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ, {"MARGIN_HINGE_DISPLACEMENT": value}
                ):
                    self.assertIsNone(margin_hinge_setting())
        with mock.patch.dict(os.environ, {"MARGIN_HINGE_DISPLACEMENT": "0.25"}):
            self.assertEqual(margin_hinge_setting(), 0.25)
        with mock.patch.dict(os.environ, {"MARGIN_HINGE_DISPLACEMENT": "-1"}):
            with self.assertRaises(ValueError):
                margin_hinge_setting()


class MomentumIdTests(unittest.TestCase):
    """Momentum changes the perturbation, so it names the output."""

    def _setup(self):
        return SETUPS[f"{BASE}_eps4"]

    def test_plain_sign_pgd_adds_nothing(self) -> None:
        self.assertEqual(effective_setup_id(self._setup()), f"{BASE}_eps4")
        self.assertEqual(
            effective_setup_id(
                self._setup(), None, 1.0, "balanced", None, "constant", None, 0.0
            ),
            f"{BASE}_eps4",
        )

    def test_each_decay_gets_its_own_name(self) -> None:
        names = {
            effective_setup_id(
                self._setup(), None, 1.0, "balanced", None, "constant", None, value
            )
            for value in (0.0, 0.9, 1.0)
        }
        self.assertEqual(len(names), 3)
        self.assertIn(f"{BASE}_eps4_mom0p9", names)

    def test_the_hinge_and_momentum_compose(self) -> None:
        derived = effective_setup_id(
            self._setup(), None, 1.0, "balanced", None, "constant", 0.25, 0.9
        )
        self.assertEqual(derived, f"{BASE}_eps4_hinge0p25_mom0p9")

    def test_the_environment_setting_is_off_unless_asked(self) -> None:
        from setup_catalog import momentum_decay_setting

        for value in ("", "none", "off", "0"):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"MOMENTUM_DECAY": value}):
                    self.assertEqual(momentum_decay_setting(), 0.0)
        with mock.patch.dict(os.environ, {"MOMENTUM_DECAY": "0.9"}):
            self.assertEqual(momentum_decay_setting(), 0.9)
        with mock.patch.dict(os.environ, {"MOMENTUM_DECAY": "1.5"}):
            with self.assertRaises(ValueError):
                momentum_decay_setting()


class CheckpointSelectionIdTests(unittest.TestCase):
    """The retained iterate changes the artifact, so it names the output."""

    def _setup(self):
        return SETUPS[f"{BASE}_eps4"]

    def test_final_is_the_default_and_adds_nothing(self) -> None:
        self.assertEqual(effective_setup_id(self._setup()), f"{BASE}_eps4")
        self.assertEqual(
            effective_setup_id(
                self._setup(), None, 1.0, "balanced", None, "constant", None,
                0.0, "final",
            ),
            f"{BASE}_eps4",
        )

    def test_best_names_itself(self) -> None:
        self.assertEqual(
            effective_setup_id(
                self._setup(), None, 1.0, "balanced", None, "constant", None,
                0.0, "best",
            ),
            f"{BASE}_eps4_best",
        )

    def test_it_composes_with_the_other_switches(self) -> None:
        derived = effective_setup_id(
            self._setup(), None, 1.0, "balanced", None, "constant", 0.25, 0.9,
            "best",
        )
        self.assertEqual(derived, f"{BASE}_eps4_hinge0p25_mom0p9_best")

    def test_unknown_selections_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint_selection"):
            effective_setup_id(
                self._setup(), None, 1.0, "balanced", None, "constant", None,
                0.0, "last",
            )

    def test_the_environment_setting_defaults_to_final(self) -> None:
        from setup_catalog import checkpoint_selection_setting

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CHECKPOINT_SELECTION", None)
            self.assertEqual(checkpoint_selection_setting(), "final")
        with mock.patch.dict(os.environ, {"CHECKPOINT_SELECTION": "BEST"}):
            self.assertEqual(checkpoint_selection_setting(), "best")
        with mock.patch.dict(os.environ, {"CHECKPOINT_SELECTION": "last"}):
            with self.assertRaises(ValueError):
                checkpoint_selection_setting()


class SnapshotEpochTests(unittest.TestCase):
    """Snapshot budgets land on the same step index a standalone run would."""

    def _setting(self, **environment):
        from setup_catalog import snapshot_epochs_setting

        base = {"STEP_SIZE_SCHEDULE": "constant", **environment}
        with mock.patch.dict(os.environ, base, clear=False):
            return snapshot_epochs_setting()

    def test_off_by_default(self) -> None:
        for value in ("", "none", "off"):
            with self.subTest(value=value):
                self.assertEqual(self._setting(SNAPSHOT_EPOCHS=value), ())

    def test_entries_are_setup_budgets_like_the_run(self) -> None:
        # Same format as SETUP_EPOCHS, so each snapshot names a whole setup.
        self.assertEqual(
            self._setting(SNAPSHOT_EPOCHS="10:200:200,5:100:100"),
            ((5.0, 5.0, 100.0, 100.0), (10.0, 10.0, 200.0, 200.0)),
        )

    def test_the_cross_budget_can_differ(self) -> None:
        self.assertEqual(
            self._setting(SNAPSHOT_EPOCHS="10:3:200:200"),
            ((10.0, 3.0, 200.0, 200.0),),
        )

    def test_a_bare_number_applies_to_every_scope(self) -> None:
        self.assertEqual(
            self._setting(SNAPSHOT_EPOCHS="5"), ((5.0, 5.0, 5.0, 5.0),)
        )

    def test_duplicates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._setting(SNAPSHOT_EPOCHS="5:100:100,5:100:100")

    def test_non_positive_budgets_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._setting(SNAPSHOT_EPOCHS="0,5")

    def test_a_decaying_step_size_is_refused(self) -> None:
        # The equivalence only holds for a budget-independent step size.
        with self.assertRaisesRegex(ValueError, "STEP_SIZE_SCHEDULE=constant"):
            self._setting(SNAPSHOT_EPOCHS="5", STEP_SIZE_SCHEDULE="linear")

    def test_steps_match_the_budget_derivation(self) -> None:
        from setup_catalog import derive_steps, snapshot_steps

        # A snapshot at epoch 5 must be the step a 5-epoch run would end on.
        self.assertEqual(
            snapshot_steps((5.0, 6.0), 224, 2),
            (derive_steps(5.0, 224, 2), derive_steps(6.0, 224, 2)),
        )

    def test_each_snapshot_names_a_complete_setup(self) -> None:
        from setup_catalog import compose_setup_id

        setup = SETUPS[f"{BASE}_eps4"]
        names = [
            compose_setup_id(
                dataset, cross, category, image, setup.epsilon_label,
                setup.loss_formulation, setup.prompt_mode,
            )
            for dataset, cross, category, image in self._setting(
                SNAPSHOT_EPOCHS="5:100:100,10:200:200"
            )
        ]
        self.assertEqual(
            names, ["ep5_cat100_img100_eps4", "ep10_cat200_img200_eps4"]
        )

    def test_a_snapshot_must_be_a_prefix_of_the_run(self) -> None:
        from setup_catalog import assert_snapshots_fit

        budget = (20.0, 20.0, 400.0, 400.0)
        assert_snapshots_fit(
            ((5.0, 5.0, 100.0, 100.0), (10.0, 3.0, 200.0, 200.0)), budget
        )
        # A scope can be stopped early, never extended.
        with self.assertRaisesRegex(ValueError, "exceeds the run budget"):
            assert_snapshots_fit(((5.0, 5.0, 500.0, 100.0),), budget)
        # And a snapshot equal to the run would claim the run's own name.
        with self.assertRaisesRegex(ValueError, "the run's own budget"):
            assert_snapshots_fit((budget,), budget)


class CrossDatasetBudgetTests(unittest.TestCase):
    """cross_dataset has its own budget, because it trains on its own cohort.

    Under fullcross it optimizes a delta on the complete source rather than
    the attack_train half, so the same epoch number buys about twice the
    steps. Before this it borrowed the per-dataset budget and there was no way
    to run it shorter without shortening per_dataset too.
    """

    def _grid(self, value: str):
        with mock.patch.dict(os.environ, {"SETUP_EPOCHS": value}):
            from setup_catalog import epoch_grid

            return epoch_grid()

    def test_the_historical_form_keeps_cross_on_the_dataset_budget(self) -> None:
        self.assertEqual(self._grid("7.14:100:100"), ((7.14, 7.14, 100.0, 100.0),))

    def test_a_bare_number_still_means_every_scope(self) -> None:
        self.assertEqual(self._grid("100"), ((100.0, 100.0, 100.0, 100.0),))

    def test_the_four_part_form_separates_cross(self) -> None:
        self.assertEqual(self._grid("7.14:5:100:100"), ((7.14, 5.0, 100.0, 100.0),))

    def test_two_part_entries_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "dataset:cross:category:image"):
            self._grid("7:100")

    def test_the_name_is_unchanged_while_cross_matches_the_dataset(self) -> None:
        # Every existing output name predates the cross budget and must survive.
        setup = SETUPS[f"{BASE}_eps4"]
        self.assertEqual(setup.cross_epochs, setup.epochs)
        self.assertEqual(effective_setup_id(setup), f"{BASE}_eps4")
        self.assertNotIn("cross", effective_setup_id(setup))

    def test_a_different_cross_budget_names_itself(self) -> None:
        generated = build_setups(
            epochs_grid=((7.14, 5, 100, 100),), epsilons=("4/255",)
        )
        self.assertIn("ep7p14_cross5_cat100_img100_eps4", generated)
        for setup_id, setup in generated.items():
            with self.subTest(setup_id=setup_id):
                self.assertEqual(effective_setup_id(setup), setup_id)

    def test_budgets_that_differ_only_in_cross_never_collide(self) -> None:
        names = {
            effective_setup_id(setup)
            for cross in (7.14, 5, 3)
            for setup in build_setups(
                epochs_grid=((7.14, cross, 100, 100),), epsilons=("4/255",)
            ).values()
        }
        # 3 cross budgets x 2 losses x 2 prompt families, all distinct.
        self.assertEqual(len(names), 3 * 2 * 2)

    def test_the_launcher_exports_the_cross_budget(self) -> None:
        script = (Path(__file__).resolve().parents[1] / "train.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('export PER_CROSS_EPOCHS="$cross_epochs"', script)

    def test_the_audit_keys_cross_dataset_to_its_own_budget(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "audit_generation.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"cross_dataset": SMOKE_EPOCHS if SMOKE else setup.cross_epochs,', source
        )

    def test_the_runner_uses_it_only_for_the_complete_source_delta(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "run_per_dataset.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "PER_CROSS_EPOCHS if use_full_source else PER_DATASET_EPOCHS", source
        )


class UnusedCrossBudgetTests(unittest.TestCase):
    """halfcross spends no cross budget, so the value must not split the name.

    Under halfcross the cross-dataset scope delivers the per-dataset delta and
    optimizes nothing, so two runs differing only in the cross number produce
    byte-identical artifacts and must share one output directory.
    """

    def _name(self, entry: str, full_data_cross: bool) -> str:
        with mock.patch.dict(os.environ, {"SETUP_EPOCHS": entry}):
            from setup_catalog import build_setups

            generated = build_setups(
                epsilons=("4/255",), full_data_cross=full_data_cross
            )
        return sorted(generated)[0]

    def test_halfcross_ignores_the_cross_budget_in_the_name(self) -> None:
        self.assertEqual(
            self._name("7.14:5:100:100", False),
            self._name("7.14:100:100", False),
        )
        self.assertNotIn("cross5", self._name("7.14:5:100:100", False))

    def test_fullcross_keeps_it(self) -> None:
        self.assertNotEqual(
            self._name("7.14:5:100:100", True),
            self._name("7.14:100:100", True),
        )
        self.assertIn("cross5", self._name("7.14:5:100:100", True))

    def test_an_unknown_cross_mode_keeps_it(self) -> None:
        # The bare catalog does not know the mode yet, so it must not drop a
        # component that fullcross would need.
        self.assertIn("cross5", self._name("7.14:5:100:100", None))


class SnapshotLauncherTests(unittest.TestCase):
    """A snapshot budget is emitted as an ordinary row, so nothing special-cases it."""

    def _rows(self, **overrides):
        return _launcher_table(
            SETUP_EPOCHS="20:400:400",
            STEP_SIZE_SCHEDULE="constant",
            PROMPT_SETUP="frozen",
            **overrides,
        )

    def test_no_snapshot_rows_by_default(self) -> None:
        rows = self._rows(SNAPSHOT_EPOCHS="")
        self.assertTrue(all(row[8].startswith("ep20_") for row in rows))
        self.assertTrue(all(row[9] == "" for row in rows))

    def test_each_budget_gets_its_own_row_and_directory(self) -> None:
        rows = self._rows(SNAPSHOT_EPOCHS="5:100:100,10:200:200")
        names = [row[8] for row in rows]
        self.assertTrue(any(name.startswith("ep5_cat100_img100_") for name in names))
        self.assertTrue(any(name.startswith("ep10_cat200_img200_") for name in names))
        self.assertTrue(any(name.startswith("ep20_cat400_img400_") for name in names))

    def test_the_longest_budget_runs_before_the_budgets_it_produces(self) -> None:
        # The shorter budgets reuse deltas the longer run writes, so their rows
        # must come after it.
        rows = self._rows(SNAPSHOT_EPOCHS="5:100:100")
        first = rows[0]
        self.assertTrue(first[8].startswith("ep20_"))
        self.assertTrue(rows[1][8].startswith("ep5_"))

    def test_only_the_producing_row_carries_the_snapshot_spec(self) -> None:
        rows = self._rows(SNAPSHOT_EPOCHS="5:100:100")
        self.assertIn("ep5_cat100_img100", rows[0][9])
        self.assertEqual(rows[1][9], "")

    def test_a_snapshot_row_keeps_the_parent_catalog_id(self) -> None:
        # So RUN_SETUPS selection reaches the snapshots of a selected setup.
        rows = self._rows(SNAPSHOT_EPOCHS="5:100:100")
        self.assertEqual(rows[0][0], rows[1][0])

    def test_a_budget_longer_than_the_run_is_refused(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            self._rows(SNAPSHOT_EPOCHS="25:100:100")


class SnapshotTargetParsingTests(unittest.TestCase):
    """The launcher owns the naming; runners only read absolute roots."""

    def _targets(self, value: str):
        from setup_catalog import snapshot_targets

        with mock.patch.dict(os.environ, {"SNAPSHOT_SETUP_ROOTS": value}):
            return snapshot_targets()

    def test_empty_means_no_snapshots(self) -> None:
        for value in ("", ";", "  "):
            with self.subTest(value=value):
                self.assertEqual(self._targets(value), ())

    def test_entries_carry_a_budget_and_a_root(self) -> None:
        self.assertEqual(
            self._targets("5:5:100:100=/out/ep5;10:10:200:200=/out/ep10"),
            (
                ((5.0, 5.0, 100.0, 100.0), "/out/ep5"),
                ((10.0, 10.0, 200.0, 200.0), "/out/ep10"),
            ),
        )

    def test_a_malformed_entry_is_refused(self) -> None:
        for value in ("5:100=/out/ep5", "5:5:100:100", "5:5:100:100="):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self._targets(value)

    def test_every_runner_reads_the_shared_parser(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in ("run_per_dataset.py", "run_per_category.py", "run_per_image.py"):
            with self.subTest(runner=name):
                source = (root / name).read_text(encoding="utf-8")
                self.assertIn("snapshot_targets", source)
                self.assertIn("write_snapshot_artifact", source)


class OutputLayoutTests(unittest.TestCase):
    """The directory layout and the flat setup ID share one source of truth.

    The tree groups by settings, then scope, then that scope's own budget,
    then prompt family, so a budget sweep is one directory listing. The flat
    ID is still what manifests and the reuse guard key on.
    """

    def _settings(self, **overrides):
        from setup_catalog import settings_tag

        arguments = {
            "epsilon_label": "4/255",
            "loss_formulation": "margin_topk",
            "attack_train_fraction": 1.0,
            "split_protocol": "balanced",
            "full_data_cross": None,
            "step_size_schedule": "constant",
            "margin_hinge_displacement": None,
            "momentum_decay": 0.0,
            "checkpoint_selection": "final",
        }
        arguments.update(overrides)
        return settings_tag(**arguments)

    def test_the_settings_tag_carries_no_epoch_numbers(self) -> None:
        tag = self._settings()
        self.assertEqual(tag, "eps4")
        self.assertNotIn("ep", tag.replace("eps", ""))

    def test_it_carries_everything_else_that_shapes_the_attack(self) -> None:
        tag = self._settings(
            loss_formulation="ce_focal_dice", attack_train_fraction=0.2,
            split_protocol="full", full_data_cross=True,
            step_size_schedule="linear", checkpoint_selection="best",
        )
        for component in ("eps4", "ce_focal_dice", "linear_step", "best",
                          "full", "fullcross", "train20"):
            with self.subTest(component=component):
                self.assertIn(component, tag)

    def test_the_flat_id_is_the_epochs_tag_plus_the_settings_tag(self) -> None:
        # One source of truth: the two can never describe different runs.
        from setup_catalog import compose_setup_id

        derived = compose_setup_id(
            7.14, 7.14, 100, 100, "4/255", "margin_topk", "frozen_winclip",
            0.2, "full", True,
        )
        self.assertTrue(
            derived.endswith(
                self._settings(
                    attack_train_fraction=0.2, split_protocol="full",
                    full_data_cross=True,
                )
            )
        )

    def test_each_scope_directory_shows_its_own_budget(self) -> None:
        from setup_catalog import scope_epochs_tag

        budget = (7.14, 3.0, 100.0, 60.0)
        self.assertEqual(scope_epochs_tag("per_dataset", budget), "ep7p14")
        self.assertEqual(scope_epochs_tag("cross_dataset", budget), "ep3")
        self.assertEqual(scope_epochs_tag("per_category", budget), "ep100")
        self.assertEqual(scope_epochs_tag("per_image", budget), "ep60")

    def test_an_unknown_scope_is_refused(self) -> None:
        from setup_catalog import scope_epochs_tag

        with self.assertRaisesRegex(ValueError, "scope must be one of"):
            scope_epochs_tag("per_pixel", (1, 1, 1, 1))

    def test_the_path_orders_settings_scope_epochs_family(self) -> None:
        from setup_catalog import scope_output_path

        self.assertEqual(
            scope_output_path(
                "per_category", (7.14, 7.14, 100.0, 100.0),
                "learnable_object_agnostic", "eps4_full_fullcross",
            ),
            "eps4_full_fullcross/per_category/ep100/learnable_prompt",
        )

    def test_an_unknown_prompt_mode_is_refused(self) -> None:
        from setup_catalog import scope_output_path

        with self.assertRaisesRegex(ValueError, "Unknown prompt mode"):
            scope_output_path("per_dataset", (1, 1, 1, 1), "invented", "eps4")

    def test_budgets_sharing_settings_land_side_by_side(self) -> None:
        from setup_catalog import scope_output_path

        paths = [
            scope_output_path("per_dataset", budget, "frozen_winclip", "eps4")
            for budget in ((5, 5, 60, 60), (7.14, 7.14, 100, 100))
        ]
        parents = {path.rsplit("/", 2)[0] for path in paths}
        self.assertEqual(len(parents), 1)


class LauncherAuditAgreementTests(unittest.TestCase):
    """The launcher writes where the audit looks, for every scope.

    These are two separate derivations of the same tree, so a divergence
    would surface as a run that completes and then fails its own audit.
    """

    def _launcher_paths(self, **overrides):
        rows = _launcher_table(
            SETUP_EPOCHS="7.14:100:100", SETUP_EPSILONS="4/255",
            STEP_SIZE_SCHEDULE="constant", SPLIT_PROTOCOL="full",
            FULL_DATA_CROSS="true", **overrides,
        )
        return {row[8]: (row[10], row[11:15]) for row in rows}

    def _audit_paths(self, setup):
        from setup_catalog import scope_output_path, settings_tag

        settings = settings_tag(
            setup.epsilon_label, setup.loss_formulation, 1.0, "full", True,
            "constant", None, 0.0, "final",
        )
        budget = (
            setup.epochs, setup.cross_epochs, setup.category_epochs,
            setup.image_epochs,
        )
        return settings, [
            scope_output_path(scope, budget, setup.prompt_mode, settings)
            for scope in SCOPE_DIRECTORIES
        ]

    def test_every_scope_path_matches(self) -> None:
        from setup_catalog import build_setups

        emitted = self._launcher_paths()
        catalog = build_setups(
            epsilons=("4/255",), split_protocol="full", full_data_cross=True,
        )
        self.assertTrue(emitted)
        for name, setup in catalog.items():
            with self.subTest(setup=name):
                self.assertIn(name, emitted)
                settings, scopes = self._audit_paths(setup)
                self.assertEqual(emitted[name][0], settings)
                self.assertEqual(list(emitted[name][1]), scopes)

    def test_the_audit_no_longer_walks_the_old_tree(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "audit_generation.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("canonical_clip_per_dataset", source)
        self.assertNotIn("prompt_folder", source)
        self.assertIn("scope_output_path", source)

    def test_the_runners_take_their_bundle_from_the_launcher(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name, variable in (
            ("run_per_dataset.py", "BUNDLE_PER_DATASET"),
            ("run_per_category.py", "BUNDLE_PER_CATEGORY"),
            ("run_per_image.py", "BUNDLE_PER_IMAGE"),
        ):
            with self.subTest(runner=name):
                source = (root / name).read_text(encoding="utf-8")
                self.assertIn(f'os.environ["{variable}"]', source)
                self.assertNotIn('OUTPUT_BASE / "canonical_clip', source)


class SetupIdReachesTheManifestTests(unittest.TestCase):
    """The setup ID must survive into the manifest.

    It used to reach the evaluator through the directory name and the archive
    filename. The settings/scope/budget tree spells neither, so the manifest is
    now the only carrier, and losing it is silent: generation completes and
    every evaluation stage then fails to match any condition.
    """

    RUNNERS = ("run_per_dataset.py", "run_per_category.py", "run_per_image.py")

    def _source(self, name):
        return (Path(__file__).resolve().parents[1] / name).read_text(
            encoding="utf-8"
        )

    def test_each_runner_writes_the_exported_id(self) -> None:
        for name in self.RUNNERS:
            with self.subTest(runner=name):
                source = self._source(name)
                self.assertIn('SETUP_ID = os.environ["SETUP_ID"]', source)
                self.assertIn('"setup_id": SETUP_ID,', source)

    def test_the_exported_id_is_not_a_dead_read(self) -> None:
        """A read with no use is how this broke the first time."""

        for name in self.RUNNERS:
            with self.subTest(runner=name):
                tree = ast.parse(self._source(name))
                uses = [
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.Name)
                    and node.id == "SETUP_ID"
                    and isinstance(node.ctx, ast.Load)
                ]
                self.assertTrue(
                    uses, f"{name} reads SETUP_ID but never uses it"
                )

    def test_the_manifest_row_carries_it_beside_the_scope(self) -> None:
        """Pin it to the row dict, not merely to the file."""

        for name, scope_key in (
            ("run_per_dataset.py", '"scope": BUNDLE_SCOPES[setting],'),
            ("run_per_category.py", '"scope": "per_category",'),
            ("run_per_image.py", '"scope": "per_image",'),
        ):
            with self.subTest(runner=name):
                source = self._source(name)
                marker = '"setup_id": SETUP_ID,'
                self.assertIn(marker, source, f"{name}: setup_id is not written")
                index = source.index(marker)
                self.assertIn(
                    scope_key, source[index:index + 200],
                    f"{name}: setup_id is not in the manifest row dict",
                )

    def test_the_audit_rejects_a_manifest_without_it(self) -> None:
        source = self._source("audit_generation.py")
        self.assertIn('if "setup_id" not in manifest.columns:', source)
        self.assertIn("expected_setup_id", source)


class LauncherFieldBindingTests(unittest.TestCase):
    """Replay train.sh's own read loop, in bash, against the real table.

    Splitting the table in Python cannot see this class of bug: Python's
    split keeps empty fields, while bash's read with a whitespace IFS
    collapses a run of delimiters and silently drops one. The table must
    therefore be parsed the way the launcher parses it.
    """

    def _rows(self, **overrides):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not available")
        root = Path(__file__).resolve().parents[1]
        script = (root / "train.sh").read_text(encoding="utf-8")
        # Take the launcher's actual loop header so this tracks train.sh
        # instead of a copy that can drift away from it.
        header = re.search(
            r"^while IFS=.*read -r id epochs .*do$", script, re.MULTILINE
        )
        self.assertIsNotNone(header, "train.sh no longer has the setup loop")
        table = subprocess.run(
            [sys.executable, "-c", _launcher_snippet()],
            capture_output=True, text=True, cwd=root, check=True,
            env={
                **os.environ, "SMOKE_TEST": "false", "SMOKE_EPOCHS": "0.02",
                "ATTACK_TRAIN_FRACTION": "1.0", **overrides,
            },
        ).stdout
        program = NL_CONST.join((
            header.group(0),
            '  echo "$settings_tag|$bundle_per_image|$snapshot_spec|$id"',
            'done <<< "$SETUP_TABLE"',
        ))
        completed = subprocess.run(
            [bash, "-c", program], capture_output=True, text=True, check=True,
            # $(...) strips trailing newlines in train.sh, so the launcher
            # never reads a blank final line; match that here.
            env={**os.environ, "SETUP_TABLE": table.strip(NL_CONST)},
        )
        return [
            line.split("|")
            for line in completed.stdout.strip(NL_CONST).splitlines()
        ]

    def test_every_column_binds_when_snapshot_spec_is_empty(self) -> None:
        """Without SNAPSHOT_EPOCHS every row's spec is empty."""

        rows = self._rows(SETUP_EPOCHS="10", SETUP_EPSILONS="4/255")
        self.assertTrue(rows)
        for settings, bundle_per_image, spec, setup_id in rows:
            with self.subTest(setup=setup_id):
                # A collapsed field shifts a path into settings_tag and
                # empties the last column.
                self.assertNotIn("/", settings)
                self.assertTrue(bundle_per_image)
                self.assertIn("/", bundle_per_image)
                self.assertEqual(spec, "")

    def test_every_column_binds_on_the_non_first_snapshot_rows(self) -> None:
        """With SNAPSHOT_EPOCHS only the first row carries a spec."""

        rows = self._rows(
            SETUP_EPOCHS="20:200:40", SETUP_EPSILONS="4/255",
            SNAPSHOT_EPOCHS="5:50:10,10:100:20",
        )
        self.assertTrue(rows)
        for settings, bundle_per_image, _, setup_id in rows:
            with self.subTest(setup=setup_id):
                self.assertNotIn("/", settings)
                self.assertTrue(bundle_per_image)

    def test_the_table_is_not_separated_by_whitespace(self) -> None:
        """The delimiter itself is the fix; pin it."""

        script = (
            Path(__file__).resolve().parents[1] / "train.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("IFS=$'" + BS_CONST + "t'", script)
        self.assertIn("IFS=$'" + BS_CONST + "x1f'", script)


class RandomBaselineNamingTests(unittest.TestCase):
    """A random control must never share a directory with the run it controls."""

    def test_the_default_adds_nothing(self) -> None:
        from setup_catalog import settings_tag

        base = dict(epsilon_label="4/255", loss_formulation="margin_topk")
        self.assertEqual(
            settings_tag(**base), settings_tag(**base, random_baseline=False)
        )

    def test_random_names_itself_last(self) -> None:
        from setup_catalog import SETUPS, effective_setup_id

        setup = next(iter(SETUPS.values()))
        args = (setup, 10.0, 1.0, "full", False, "constant", 0.5, 0.0, "final",
                "all")
        optimised = effective_setup_id(*args)
        control = effective_setup_id(*args, True)
        self.assertEqual(control, optimised + "_random")
        self.assertTrue(optimised.endswith("_full_halfcross_alltargets"))

    def test_only_per_dataset_supports_it(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for runner in ("run_per_category.py", "run_per_image.py"):
            with self.subTest(runner=runner):
                source = (root / runner).read_text(encoding="utf-8")
                self.assertIn("RANDOM_BASELINE is implemented for", source)


class PerImageCohortTests(unittest.TestCase):
    """Which images per-image attacks is an explicit, named setting.

    The generator and the evaluator disagreed about the per-image cohort:
    under the full split the generator attacked every retained image while
    the evaluator scored the evaluation partition, so the evaluator found
    4 images where the manifest claimed 8 and refused the bundle. The
    cohort is now its own setting, defaulting to the comparable one.
    """

    def test_the_default_is_the_comparable_cohort(self) -> None:
        from setup_catalog import per_image_cohort_setting

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PER_IMAGE_ATTACK_COHORT", None)
            self.assertEqual(per_image_cohort_setting(), "evaluation")

    def test_the_full_split_no_longer_widens_the_cohort(self) -> None:
        """The regression: the two decisions must stay independent."""

        from setup_catalog import per_image_cohort_setting

        with mock.patch.dict(os.environ, {"SPLIT_PROTOCOL": "full"}):
            os.environ.pop("PER_IMAGE_ATTACK_COHORT", None)
            self.assertEqual(per_image_cohort_setting(), "evaluation")

    def test_an_unknown_cohort_is_rejected(self) -> None:
        from setup_catalog import per_image_cohort_setting

        with mock.patch.dict(os.environ, {"PER_IMAGE_ATTACK_COHORT": "half"}):
            with self.assertRaises(ValueError):
                per_image_cohort_setting()

    def test_only_the_wider_cohort_names_itself(self) -> None:
        from setup_catalog import settings_tag

        base = dict(
            epsilon_label="4/255", loss_formulation="margin_topk",
            attack_train_fraction=1.0, split_protocol="full",
            full_data_cross=False,
        )
        self.assertNotIn("alltargets", settings_tag(**base))
        self.assertIn(
            "alltargets", settings_tag(**base, per_image_cohort="all")
        )

    def test_the_two_cohorts_cannot_share_a_directory(self) -> None:
        """They produce different per-image bundles, so they must not collide."""

        from setup_catalog import SETUPS, effective_setup_id

        setup = next(iter(SETUPS.values()))
        comparable = effective_setup_id(
            setup, None, 1.0, "full", False, "constant", None, 0.0, "final",
            "evaluation",
        )
        everything = effective_setup_id(
            setup, None, 1.0, "full", False, "constant", None, 0.0, "final",
            "all",
        )
        self.assertNotEqual(comparable, everything)

    def test_the_runner_records_the_cohort_in_the_manifest(self) -> None:
        """A bundle must say which cohort it covers, not leave it inferred."""

        source = (
            Path(__file__).resolve().parents[1] / "run_per_image.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"per_image_attack_cohort": PER_IMAGE_ATTACK_COHORT,', source
        )

    def test_the_selection_and_the_leakage_guard_still_agree(self) -> None:
        """Both sides read the same flag, or full aborts after doing the work."""

        source = (
            Path(__file__).resolve().parents[1] / "run_per_image.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "if not ATTACK_EVERY_IMAGE and assignments[pid] != " '"evaluation"',
            source,
        )
        self.assertIn(
            "if not ATTACK_EVERY_IMAGE and set(sample_ids) & attack_train_ids:",
            source,
        )
