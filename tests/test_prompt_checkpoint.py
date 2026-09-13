from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import ensure_prompt_checkpoint as ensure_module
from ensure_prompt_checkpoint import (
    checkpoint_mismatch,
    checkpoint_path,
    cohort_directory,
    ensure,
)


def _payload(
    *,
    dataset: str = "mvtec",
    split_protocol: str | None = "balanced",
    attack_train_fraction: float | None = 1.0,
    seed: int = 111,
    epoch: int = 15,
) -> dict:
    prompt_config = {
        "n_ctx": 12,
        "normal_suffix": "object.",
        "abnormal_suffix": "damaged object.",
        "context_length": 77,
        "category_specific": False,
        "deep_text_prompt_tuning": False,
    }
    # A legacy checkpoint predates these keys entirely.
    if split_protocol is not None:
        prompt_config["split_protocol"] = split_protocol
    if attack_train_fraction is not None:
        prompt_config["attack_train_fraction"] = attack_train_fraction
    return {
        "schema_version": 1,
        "dataset": dataset,
        "epoch": epoch,
        "seed": seed,
        "prompt_config": prompt_config,
        "training_config": {"epochs": epoch, "batch_size": 2, "seed": seed},
        "sample_manifest_sha256": "a" * 64,
        "prompt_state": {
            "normal_context": torch.zeros(12, 4),
            "abnormal_context": torch.zeros(12, 4),
        },
    }


REQUIRED = {
    "dataset": "mvtec",
    "split_protocol": "balanced",
    "attack_train_fraction": 1.0,
    "seed": 111,
    "epochs": 15,
}


class CohortDirectoryTests(unittest.TestCase):
    def test_full_fraction_keeps_the_protocol_bare(self):
        self.assertEqual(cohort_directory("balanced", 1.0), "balanced")
        self.assertEqual(cohort_directory("full", 1.0), "full")

    def test_partial_fraction_adds_a_suffix(self):
        self.assertEqual(cohort_directory("full", 0.25), "full_train25")
        self.assertEqual(cohort_directory("balanced", 0.2), "balanced_train20")

    def test_protocols_never_share_a_directory(self):
        for fraction in (1.0, 0.5, 0.05):
            self.assertNotEqual(
                cohort_directory("balanced", fraction),
                cohort_directory("full", fraction),
            )

    def test_checkpoint_path_layout(self):
        path = checkpoint_path(Path("/out"), "full", 0.25, "mvtec", 15)
        self.assertEqual(path.parts[-3:], ("full_train25", "mvtec", "prompts_epoch15.pt"))


class CheckpointMismatchTests(unittest.TestCase):
    def test_matching_checkpoint_is_accepted(self):
        self.assertEqual(checkpoint_mismatch(_payload(), **REQUIRED), "")

    def test_protocol_mismatch_is_reported(self):
        required = {**REQUIRED, "split_protocol": "full"}
        reason = checkpoint_mismatch(_payload(split_protocol="balanced"), **required)
        self.assertIn("split protocol", reason)

    def test_fraction_mismatch_is_reported(self):
        required = {**REQUIRED, "attack_train_fraction": 0.25}
        reason = checkpoint_mismatch(_payload(attack_train_fraction=1.0), **required)
        self.assertIn("fraction", reason)

    def test_seed_mismatch_is_reported(self):
        reason = checkpoint_mismatch(_payload(seed=7), **REQUIRED)
        self.assertIn("seed", reason)

    def test_epoch_mismatch_is_reported(self):
        reason = checkpoint_mismatch(_payload(epoch=10), **REQUIRED)
        self.assertIn("epochs", reason)

    def test_dataset_mismatch_is_reported(self):
        reason = checkpoint_mismatch(_payload(dataset="visa"), **REQUIRED)
        self.assertIn("visa", reason)

    def test_legacy_checkpoint_counts_as_balanced_at_full_fraction(self):
        legacy = _payload(split_protocol=None, attack_train_fraction=None)
        self.assertEqual(checkpoint_mismatch(legacy, **REQUIRED), "")
        required = {**REQUIRED, "split_protocol": "full"}
        self.assertIn("split protocol", checkpoint_mismatch(legacy, **required))


class _EnsureCase(unittest.TestCase):
    """Drive ensure() against a temporary checkpoint tree."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        manifest = self.root / "attack_train_indices.csv"
        manifest.write_text("protocol_id\n", encoding="utf-8")
        self.environment = {
            "SPLIT_PROTOCOL": "balanced",
            "ATTACK_TRAIN_FRACTION": "1.0",
            "SPLIT_SEED": "111",
            "EVALUATION_FRACTION": "0.50",
            "PROMPT_TRAINING_EPOCHS": "15",
            "PROMPT_TRAINING_BATCH_SIZE": "2",
            "PROMPT_TRAINING_OUTPUT_ROOT": str(self.root / "prompts"),
            "PROMPT_TRAINING_SEARCH_ROOTS": "",
            "WORK_DIR": str(self.root / "runtime"),
            "ATTACK_TRAIN_CSV": str(manifest),
            "LEARNABLE_PROMPT_MVTEC_CHECKPOINT": "",
        }
        self.published = self.root / "published"

    def write(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path

    def derived(self, protocol: str = "balanced", fraction: float = 1.0) -> Path:
        return checkpoint_path(
            Path(self.environment["PROMPT_TRAINING_OUTPUT_ROOT"]),
            protocol,
            fraction,
            "mvtec",
            15,
        )

    def run_ensure(self, **overrides):
        environment = {**self.environment, **overrides}
        with mock.patch.dict(os.environ, environment, clear=False):
            return ensure("mvtec")


class EnsureTests(_EnsureCase):
    def test_matching_checkpoint_is_reused_without_training(self):
        expected = self.write(self.derived(), _payload())
        with mock.patch.object(ensure_module, "train_prompts") as trainer:
            self.assertEqual(self.run_ensure(), expected)
        trainer.assert_not_called()

    def test_missing_checkpoint_triggers_training(self):
        target = self.derived()

        def fake_train(repo_root, config_path, dataset):
            self.write(target, _payload())

        with mock.patch.object(ensure_module, "resolve_training_repo") as repo:
            repo.return_value = self.root / "repo"
            with mock.patch.object(ensure_module, "train_prompts", fake_train):
                self.assertEqual(self.run_ensure(), target)

    def test_protocol_change_retrains_into_its_own_directory(self):
        # A balanced checkpoint must not satisfy a full run.
        self.write(self.derived("balanced"), _payload(split_protocol="balanced"))
        target = self.derived("full")

        def fake_train(repo_root, config_path, dataset):
            self.write(target, _payload(split_protocol="full"))

        with mock.patch.object(ensure_module, "resolve_training_repo") as repo:
            repo.return_value = self.root / "repo"
            with mock.patch.object(ensure_module, "train_prompts", fake_train):
                resolved = self.run_ensure(SPLIT_PROTOCOL="full")
        self.assertEqual(resolved, target)
        self.assertTrue(self.derived("balanced").is_file())

    def test_explicit_path_is_used_when_it_matches(self):
        explicit = self.write(self.root / "elsewhere" / "prompts.pt", _payload())
        with mock.patch.object(ensure_module, "train_prompts") as trainer:
            resolved = self.run_ensure(
                LEARNABLE_PROMPT_MVTEC_CHECKPOINT=str(explicit)
            )
        self.assertEqual(resolved, explicit)
        trainer.assert_not_called()

    def test_explicit_path_describing_another_split_is_rejected(self):
        explicit = self.write(
            self.root / "elsewhere" / "prompts.pt",
            _payload(split_protocol="full"),
        )
        target = self.derived("balanced")

        def fake_train(repo_root, config_path, dataset):
            self.write(target, _payload(split_protocol="balanced"))

        with mock.patch.object(ensure_module, "resolve_training_repo") as repo:
            repo.return_value = self.root / "repo"
            with mock.patch.object(ensure_module, "train_prompts", fake_train):
                resolved = self.run_ensure(
                    LEARNABLE_PROMPT_MVTEC_CHECKPOINT=str(explicit)
                )
        self.assertEqual(resolved, target)

    def test_placeholder_path_is_treated_as_unset(self):
        expected = self.write(self.derived(), _payload())
        with mock.patch.object(ensure_module, "train_prompts") as trainer:
            resolved = self.run_ensure(
                LEARNABLE_PROMPT_MVTEC_CHECKPOINT="/ABSOLUTE/PATH/TO/prompts.pt"
            )
        self.assertEqual(resolved, expected)
        trainer.assert_not_called()

    def test_training_that_writes_a_wrong_checkpoint_is_refused(self):
        def fake_train(repo_root, config_path, dataset):
            self.write(self.derived(), _payload(seed=999))

        with mock.patch.object(ensure_module, "resolve_training_repo") as repo:
            repo.return_value = self.root / "repo"
            with mock.patch.object(ensure_module, "train_prompts", fake_train):
                with self.assertRaises(RuntimeError):
                    self.run_ensure()


class SearchRootTests(_EnsureCase):
    """Published prompts are read from search roots and never written to."""

    def published_path(self, protocol: str = "balanced", fraction: float = 1.0) -> Path:
        return checkpoint_path(self.published, protocol, fraction, "mvtec", 15)

    def test_published_checkpoint_is_used_without_training(self):
        expected = self.write(self.published_path(), _payload())
        with mock.patch.object(ensure_module, "train_prompts") as trainer:
            resolved = self.run_ensure(
                PROMPT_TRAINING_SEARCH_ROOTS=str(self.published)
            )
        self.assertEqual(resolved, expected)
        trainer.assert_not_called()

    def test_published_checkpoint_wins_over_a_later_root(self):
        expected = self.write(self.published_path(), _payload())
        other = self.root / "other"
        self.write(checkpoint_path(other, "balanced", 1.0, "mvtec", 15), _payload())
        with mock.patch.object(ensure_module, "train_prompts"):
            resolved = self.run_ensure(
                PROMPT_TRAINING_SEARCH_ROOTS=f"{self.published},{other}"
            )
        self.assertEqual(resolved, expected)

    def test_mismatched_published_checkpoint_retrains_into_the_output_root(self):
        # The published tree is read-only, so training must not target it.
        stale = self.write(self.published_path(), _payload(seed=999))
        target = self.derived()

        def fake_train(config_path, dataset):
            document = json.loads(Path(config_path).read_text(encoding="utf-8"))
            self.assertEqual(
                document["artifacts"]["output_root"],
                self.environment["PROMPT_TRAINING_OUTPUT_ROOT"],
            )
            self.write(target, _payload())

        with mock.patch.object(ensure_module, "resolve_training_repo") as repo:
            repo.return_value = self.root / "repo"
            with mock.patch.object(
                ensure_module, "train_prompts", lambda r, c, d: fake_train(c, d)
            ):
                resolved = self.run_ensure(
                    PROMPT_TRAINING_SEARCH_ROOTS=str(self.published)
                )
        self.assertEqual(resolved, target)
        # The published checkpoint is left exactly as it was.
        self.assertTrue(stale.is_file())
        untouched = torch.load(stale, map_location="cpu", weights_only=True)
        self.assertEqual(untouched["seed"], 999)

    def test_a_missing_search_root_is_skipped(self):
        expected = self.write(self.derived(), _payload())
        with mock.patch.object(ensure_module, "train_prompts") as trainer:
            resolved = self.run_ensure(
                PROMPT_TRAINING_SEARCH_ROOTS=str(self.root / "absent")
            )
        self.assertEqual(resolved, expected)
        trainer.assert_not_called()


class TrainingConfigTests(_EnsureCase):
    def test_config_pins_the_split_and_hands_over_the_manifest(self):
        captured = {}

        def fake_train(repo_root, config_path, dataset):
            captured["document"] = json.loads(
                Path(config_path).read_text(encoding="utf-8")
            )
            self.write(self.derived("full", 0.25), _payload(
                split_protocol="full", attack_train_fraction=0.25
            ))

        with mock.patch.object(ensure_module, "resolve_training_repo") as repo:
            repo.return_value = self.root / "repo"
            with mock.patch.object(ensure_module, "train_prompts", fake_train):
                self.run_ensure(SPLIT_PROTOCOL="full", ATTACK_TRAIN_FRACTION="0.25")

        data = captured["document"]["data"]
        self.assertEqual(data["split_protocol"], "full")
        self.assertEqual(data["attack_train_fraction"], 0.25)
        self.assertEqual(data["datasets"], ["mvtec"])
        self.assertEqual(data["automatic_evaluation_fraction"], 0.5)
        self.assertEqual(
            data["mvtec_training_manifest"], self.environment["ATTACK_TRAIN_CSV"]
        )
        self.assertIsNone(data["visa_training_manifest"])
        training = captured["document"]["training"]
        self.assertEqual(training["seed"], 111)
        self.assertEqual(training["epochs"], 15)
        self.assertEqual(training["selected_epoch"], 15)
        self.assertEqual(training["batch_size"], 2)


class TrainingRepositoryContractTests(unittest.TestCase):
    """Check the cohort layout against the training pipeline that writes it.

    Skipped unless PROMPT_TRAINING_ROOT points at a clone, so the suite still
    runs standalone. When it is set, this is what catches the two repositories
    drifting apart: we would train into one directory and look in another.
    """

    def setUp(self):
        root = os.environ.get("PROMPT_TRAINING_ROOT", "").strip()
        if not root:
            self.skipTest("PROMPT_TRAINING_ROOT is not set")
        source = Path(root).expanduser().resolve() / "src"
        if not (source / "object_agnostic_prompt_attack").is_dir():
            self.skipTest(f"Not a prompt-training clone: {root}")
        import sys

        if str(source) not in sys.path:
            sys.path.insert(0, str(source))

    def test_cohort_directory_matches_the_training_pipeline(self):
        from object_agnostic_prompt_attack.config import DataConfig

        for protocol in ("balanced", "full"):
            for fraction in (1.0, 0.5, 0.25, 0.2, 0.05):
                with self.subTest(protocol=protocol, fraction=fraction):
                    theirs = DataConfig(
                        datasets=("mvtec",),
                        split_protocol=protocol,
                        attack_train_fraction=fraction,
                    ).cohort_directory
                    self.assertEqual(cohort_directory(protocol, fraction), theirs)


if __name__ == "__main__":
    unittest.main()
