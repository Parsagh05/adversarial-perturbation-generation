"""generation_config.json: the per-bundle record of what a run was asked to do.

The runners need CUDA and a checkout of AnomalyCLIP, so they are checked here
through their source; the helpers they call are exercised directly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

os.environ.setdefault("MVTEC_ROOT", ".")
os.environ.setdefault("VISA_ROOT", ".")
os.environ.setdefault("OUTPUT_BASE", ".")

import common
from common import (
    check_generation_config,
    environment_record,
    finish_generation_config,
    read_generation_config,
    script_settings,
    start_generation_configs,
    write_generation_config,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = ("run_per_dataset.py", "run_per_category.py", "run_per_image.py")


def _payload(**hyperparameters) -> dict:
    return {
        "setup": {"setup_id": "ep1_eps4", "scope": "dataset", "epochs": 1.0},
        "hyperparameters": {
            "epsilon": 4 / 255,
            "margin_topk_fractions": {"normal_to_abnormal": 0.2},
            "directions": ("normal_to_abnormal",),
            **hyperparameters,
        },
        "data": {"protocol_split_sha256": "abc"},
    }


class WriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.bundle = Path(self._dir.name) / "bundle"
        # Each test starts with no folders watched by the failure hook.
        self.addCleanup(common._WATCHED.clear)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_writes_sorted_indented_json_with_unix_newlines(self) -> None:
        path = write_generation_config(self.bundle, {"b": 1, "a": {"d": 2, "c": 3}})
        raw = path.read_bytes()
        self.assertNotIn(b"\r\n", raw)
        self.assertTrue(raw.endswith(b"\n"))
        text = raw.decode("utf-8")
        self.assertEqual(json.loads(text), {"a": {"c": 3, "d": 2}, "b": 1})
        self.assertLess(text.index('"a"'), text.index('"b"'))
        self.assertIn('\n  "a": {', text)

    def test_leaves_no_temporary_file(self) -> None:
        write_generation_config(self.bundle, {"a": 1})
        write_generation_config(self.bundle, {"a": 2})
        self.assertEqual(
            sorted(p.name for p in self.bundle.iterdir()), ["generation_config.json"]
        )
        self.assertEqual(read_generation_config(self.bundle), {"a": 2})

    def test_replaces_through_a_rename(self) -> None:
        """The atomic step is os.replace from the .tmp file beside the target."""

        with mock.patch.object(common.os, "replace", wraps=os.replace) as replace:
            path = write_generation_config(self.bundle, {"a": 1})
        (source, destination), _ = replace.call_args
        self.assertEqual(Path(source), path.with_name(path.name + ".tmp"))
        self.assertEqual(Path(destination), path)

    def test_running_then_completed(self) -> None:
        start_generation_configs({self.bundle: _payload()}, overwrite=False)
        started = read_generation_config(self.bundle)
        self.assertEqual(started["status"], "running")
        self.assertEqual(started["schema_version"], 1)
        self.assertIsNone(started["finished_at_utc"])
        self.assertTrue(started["created_at_utc"])

        finish_generation_config(self.bundle)
        finished = read_generation_config(self.bundle)
        self.assertEqual(finished["status"], "completed")
        self.assertTrue(finished["finished_at_utc"])
        self.assertIsNone(finished["error"])
        self.assertEqual(finished["hyperparameters"], started["hyperparameters"])

    def test_an_uncaught_exception_marks_the_run_failed_and_still_propagates(self) -> None:
        script = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(ROOT)!r})
            from common import start_generation_configs
            start_generation_configs({{{str(self.bundle)!r}: {{"setup": {{}}}}}}, overwrite=False)
            raise ValueError("boom")
        """)
        environment = {**os.environ, "MVTEC_ROOT": ".", "VISA_ROOT": ".", "OUTPUT_BASE": "."}
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=environment
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("ValueError: boom", completed.stderr)
        record = read_generation_config(self.bundle)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], "ValueError: boom")
        self.assertTrue(record["finished_at_utc"])

    def test_a_failure_leaves_completed_folders_alone(self) -> None:
        done, running = self.bundle / "done", self.bundle / "running"
        start_generation_configs({done: _payload(), running: _payload()}, overwrite=False)
        finish_generation_config(done)
        common.mark_failed(RuntimeError("late"))
        self.assertEqual(read_generation_config(done)["status"], "completed")
        self.assertEqual(read_generation_config(running)["status"], "failed")


class ResumeCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.bundle = Path(self._dir.name)
        self.addCleanup(common._WATCHED.clear)
        start_generation_configs({self.bundle: _payload()}, overwrite=False)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_identical_configuration_passes(self) -> None:
        check_generation_config(self.bundle, _payload())

    def test_a_different_margin_topk_fraction_stops_the_run(self) -> None:
        """K is not in the setup ID, so only this check keeps two K runs apart."""

        changed = _payload(margin_topk_fractions={"normal_to_abnormal": 0.4})
        with self.assertRaises(RuntimeError) as caught:
            check_generation_config(self.bundle, changed)
        self.assertIn(
            "hyperparameters.margin_topk_fractions.normal_to_abnormal",
            str(caught.exception),
        )

    def test_an_added_or_removed_setting_is_named(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            check_generation_config(self.bundle, _payload(momentum_decay=0.9))
        self.assertIn("hyperparameters.momentum_decay", str(caught.exception))

    def test_a_different_setup_stops_the_run(self) -> None:
        changed = _payload()
        changed["setup"] = {**changed["setup"], "epochs": 2.0}
        with self.assertRaises(RuntimeError) as caught:
            check_generation_config(self.bundle, changed)
        self.assertIn("setup.epochs", str(caught.exception))

    def test_status_timestamps_and_provenance_are_not_compared(self) -> None:
        finish_generation_config(self.bundle, "failed", RuntimeError("crash"))
        changed = _payload()
        changed["data"] = {"protocol_split_sha256": "different"}
        changed["execution"] = {"micro_batch_size": 1}
        check_generation_config(self.bundle, changed)

    def test_a_refused_run_writes_nothing(self) -> None:
        before = (self.bundle / "generation_config.json").read_bytes()
        other = Path(self._dir.name) / "other"
        with self.assertRaises(RuntimeError):
            start_generation_configs(
                {other: _payload(), self.bundle: _payload(epsilon=8 / 255)},
                overwrite=False,
            )
        self.assertFalse(other.exists())
        self.assertEqual((self.bundle / "generation_config.json").read_bytes(), before)

    def test_overwrite_skips_the_check(self) -> None:
        start_generation_configs({self.bundle: _payload(epsilon=8 / 255)}, overwrite=True)
        self.assertEqual(
            read_generation_config(self.bundle)["hyperparameters"]["epsilon"], 8 / 255
        )


class SettingsDerivationTests(unittest.TestCase):
    """The hyperparameters come from the script's own constants, not a list."""

    def _settings(self, source: str, namespace: dict, overrides=None):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "runner.py"
            script.write_text(textwrap.dedent(source), encoding="utf-8")
            return script_settings(script, namespace, overrides)

    def test_takes_every_plain_constant_the_script_assigns(self) -> None:
        hyperparameters, execution = self._settings(
            """
            from somewhere import IMPORTED
            EPSILON = 1
            NEW_SETTING = 2
            OUTPUT_ROOT = 3
            SETUP_ID = 4
            MICRO_BATCH_SIZE = 5
            lowercase = 6
            """,
            {
                "IMPORTED": "x", "EPSILON": 0.1, "NEW_SETTING": ("a", "b"),
                "OUTPUT_ROOT": Path("/tmp"), "SETUP_ID": "ep1",
                "MICRO_BATCH_SIZE": 2, "lowercase": 1,
            },
        )
        # A setting nobody listed is still recorded, and so still compared.
        self.assertEqual(hyperparameters, {"epsilon": 0.1, "new_setting": ["a", "b"]})
        self.assertEqual(execution, {"micro_batch_size": 2})

    def test_overrides_replace_values_for_a_snapshot(self) -> None:
        hyperparameters, _ = self._settings(
            "PER_CATEGORY_EPOCHS = 1\n", {"PER_CATEGORY_EPOCHS": 100.0},
            {"PER_CATEGORY_EPOCHS": 50.0},
        )
        self.assertEqual(hyperparameters, {"per_category_epochs": 50.0})

    def test_the_runners_expose_the_settings_the_manifest_lacks(self) -> None:
        for runner, expected in {
            "run_per_dataset.py": {
                "MARGIN_HINGE_DISPLACEMENT", "MOMENTUM_DECAY", "CHECKPOINT_SELECTION",
                "UNIVERSAL_BATCH_SIZE", "MARGIN_TOPK_FRACTIONS", "RANDOM_BASELINE",
                "TRANSFER_SETTINGS", "STEP_SIZE_MIN_RATIO", "DIAGNOSTIC_INTERVAL",
            },
            "run_per_category.py": {
                "MARGIN_HINGE_DISPLACEMENT", "MOMENTUM_DECAY", "CHECKPOINT_SELECTION",
                "EFFECTIVE_BATCH_SIZE", "MARGIN_TOPK_FRACTIONS", "USE_AMP",
            },
            "run_per_image.py": {
                "PER_IMAGE_STEPS", "EFFECTIVE_BATCH_SIZE", "EVALUATION_FRACTION",
                "PER_IMAGE_ATTACK_COHORT", "MARGIN_TOPK_FRACTIONS",
            },
        }.items():
            with self.subTest(runner=runner):
                names = set(common._script_constants(ROOT / runner))
                self.assertLessEqual(expected, names)


class EnvironmentRecordTests(unittest.TestCase):
    def test_records_only_variables_the_code_reads(self) -> None:
        with mock.patch.dict(os.environ, {
            "MARGIN_HINGE_DISPLACEMENT": "0.5", "SOME_API_TOKEN": "secret",
        }):
            record = environment_record(ROOT / "run_per_dataset.py")
        self.assertEqual(record["MARGIN_HINGE_DISPLACEMENT"], "0.5")
        self.assertNotIn("SOME_API_TOKEN", record)
        self.assertNotIn("PATH", record)
        for name in (
            "EPSILON", "PER_DATASET_EPOCHS", "RANDOM_BASELINE", "SNAPSHOT_EPOCHS",
            "LEARNABLE_PROMPT_MVTEC_CHECKPOINT",
        ):
            self.assertIn(name, record)

    def test_unset_variables_are_null(self) -> None:
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("MOMENTUM_DECAY", None)
            record = environment_record(ROOT / "run_per_dataset.py")
        self.assertIsNone(record["MOMENTUM_DECAY"])


class ProtocolDataTests(unittest.TestCase):
    def test_counts_images_and_matches_the_split_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, rows in (
                ("attack_train_indices.csv", "mvtec,0\nmvtec,1\nvisa,1\n"),
                ("evaluation_test_indices.csv", "mvtec,0\n"),
                ("complete_retained_indices.csv", "mvtec,0\nmvtec,0\n"),
            ):
                path = Path(directory) / name
                path.write_text("dataset,label\n" + rows, encoding="utf-8")
                paths.append(path)
            record = common.protocol_data_record(paths)
            with mock.patch.multiple(
                common, ATTACK_TRAIN_CSV=paths[0], EVALUATION_CSV=paths[1],
                COMPLETE_RETAINED_CSV=paths[2],
            ):
                self.assertEqual(record["protocol_split_sha256"], common.split_sha256())
        counts = record["files"]["attack_train_indices.csv"]["image_counts_by_dataset_and_label"]
        self.assertEqual(counts, {"mvtec": {"0": 1, "1": 1}, "visa": {"1": 1}})


class SnapshotSetupIdTests(unittest.TestCase):
    def test_matches_the_name_the_launcher_gives_the_snapshot(self) -> None:
        from setup_catalog import compose_setup_id, snapshot_setup_id

        for full_data_cross in (True, False):
            args = ("4/255", "margin_topk", "learnable_object_agnostic", 1.0,
                    "full", full_data_cross, "constant", 0.5, 0.0, "final")
            run, snapshot = (20.0, 20.0, 400.0, 400.0), (5.0, 5.0, 100.0, 100.0)
            environment = {
                "SETUP_ID": compose_setup_id(*run, *args),
                "FULL_DATA_CROSS": str(full_data_cross).lower(),
                "PER_DATASET_EPOCHS": "20", "PER_CROSS_EPOCHS": "20",
                "PER_CATEGORY_EPOCHS": "400", "PER_IMAGE_EPOCHS": "400",
            }
            with self.subTest(full_data_cross=full_data_cross), \
                    mock.patch.dict(os.environ, environment):
                self.assertEqual(
                    snapshot_setup_id(snapshot), compose_setup_id(*snapshot, *args)
                )


class PayloadTests(unittest.TestCase):
    def test_builds_every_section_and_serialises(self) -> None:
        from adversarial_harness.config import AttackConfig
        from adversarial_harness.prompts import frozen_ensemble_sha256, prompt_setup_record

        with tempfile.TemporaryDirectory() as directory:
            csvs = []
            for name in ("attack_train_indices.csv", "evaluation_test_indices.csv",
                         "complete_retained_indices.csv"):
                path = Path(directory) / name
                path.write_text("dataset,label\nmvtec,0\n", encoding="utf-8")
                csvs.append(path)
            prompts = {"mvtec": prompt_setup_record("mvtec", "frozen_winclip")}
            payload = common.build_generation_payload(
                ROOT / "run_per_dataset.py",
                {
                    "EPSILON": 4 / 255, "MARGIN_HINGE_DISPLACEMENT": None,
                    "MARGIN_TOPK_FRACTIONS": {"normal_to_abnormal": 0.2},
                    "OVERWRITE_EXISTING": False, "OUTPUT_ROOT": Path(directory),
                },
                setup={"setup_id": "ep1_eps4", "prompts": prompts},
                attack_config=AttackConfig(),
                csv_paths=csvs,
            )
            start_generation_configs({Path(directory) / "b": payload}, overwrite=False)
            common._WATCHED.clear()
            record = read_generation_config(Path(directory) / "b")

        self.assertEqual(
            set(record),
            {"schema_version", "status", "created_at_utc", "finished_at_utc", "error",
             "hostname", "gpu_name", "torch_version", "cuda_version", "python_version",
             "code", "setup", "hyperparameters", "execution", "data", "environment"},
        )
        self.assertEqual(
            record["setup"]["prompts"]["mvtec"]["prompt_ensemble_sha256"],
            frozen_ensemble_sha256(),
        )
        hyperparameters = record["hyperparameters"]
        self.assertIsNone(hyperparameters["margin_hinge_displacement"])
        self.assertEqual(hyperparameters["margin_topk_fractions"], {"normal_to_abnormal": 0.2})
        self.assertIn("temperature", hyperparameters["attack_config"])
        self.assertNotIn("output_root", hyperparameters)
        self.assertEqual(record["execution"]["overwrite_existing"], False)
        self.assertIn("snapshot_epochs", record["execution"])
        self.assertEqual(
            set(record["code"]),
            {"generator_commit", "generator_working_tree_dirty", "generator_script",
             "generator_script_sha256", "attack_code_sha256",
             "anomalyclip_loader_commit", "prompt_training_commit"},
        )


class RunnerWiringTests(unittest.TestCase):
    """Source checks: the record is started before the first optimisation."""

    def test_every_runner_starts_the_record_before_loading_the_surrogate(self) -> None:
        for runner in RUNNERS:
            with self.subTest(runner=runner):
                source = (ROOT / runner).read_text(encoding="utf-8")
                start = source.index("\nstart_generation_configs(")
                self.assertLess(start, source.index("    surrogate = CLIPSurrogate("))
                self.assertIn("overwrite=OVERWRITE_EXISTING", source[start:start + 800])
                self.assertIn("finish_generation_config(", source[start:])

    def test_snapshot_folders_get_their_own_record(self) -> None:
        for runner in ("run_per_dataset.py", "run_per_category.py"):
            with self.subTest(runner=runner):
                source = (ROOT / runner).read_text(encoding="utf-8")
                self.assertIn("start_snapshot_generation_config(", source)
                self.assertIn("snapshot_setup_id(budget)", source)
                self.assertIn("finish_running_generation_configs()", source)


if __name__ == "__main__":
    unittest.main()
