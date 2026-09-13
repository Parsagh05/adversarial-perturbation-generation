from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DatasetRoutingContractTests(unittest.TestCase):
    def test_per_dataset_separates_optimization_and_delivery_loops(self) -> None:
        script = (ROOT / "run_per_dataset.py").read_text(encoding="utf-8")
        self.assertIn("for source_dataset in SOURCE_DATASETS:", script)
        self.assertIn("for target_dataset in EVALUATION_DATASETS:", script)
        self.assertIn(
            'if any(sample.dataset != source_dataset for sample in source_train):',
            script,
        )

    def test_cross_dataset_rows_reuse_one_artifact_and_checksum(self) -> None:
        script = (ROOT / "run_per_dataset.py").read_text(encoding="utf-8")
        self.assertIn('relative_noise = artifact.relative_to(OUTPUT_ROOT)', script)
        self.assertIn('"noise_file": str(relative_noise)', script)
        self.assertIn('"artifact_sha256": row["artifact_file_sha256"]', script)
        self.assertNotIn("artifact_path(target_dataset", script)

    def test_category_and_image_remain_same_dataset(self) -> None:
        for filename in ("run_per_category.py", "run_per_image.py"):
            script = (ROOT / filename).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.assertIn('"source_dataset": dataset_name', script)
                self.assertIn('"target_dataset": dataset_name', script)
                self.assertNotIn("for target_dataset in EVALUATION_DATASETS", script)

    def test_notebook_exposes_both_dataset_selections(self) -> None:
        notebook = (ROOT / "kaggle_generate_corrected_perturbations.ipynb").read_text(
            encoding="utf-8"
        )
        self.assertIn("SOURCE_DATASETS = ('mvtec',)", notebook)
        self.assertIn("EVALUATION_DATASETS = ('mvtec', 'visa')", notebook)
        self.assertIn("'SOURCE_DATASETS': ','.join(SOURCE_DATASETS)", notebook)
        self.assertIn("'EVALUATION_DATASETS': ','.join(EVALUATION_DATASETS)", notebook)

    def test_protocol_writes_only_source_train_and_selected_evaluation_rows(self) -> None:
        common = (ROOT / "common.py").read_text(encoding="utf-8")
        self.assertIn("if dataset in source_datasets():", common)
        self.assertIn(
            'selected_partitions.append(("attack_train", train_samples))', common
        )
        self.assertIn("if dataset in evaluation_datasets():", common)
        self.assertIn(
            'selected_partitions.append(("evaluation", evaluation_samples))', common
        )


if __name__ == "__main__":
    unittest.main()


class CompleteSourceLeakageCheckTests(unittest.TestCase):
    """The leakage check compares a delta with what it is actually attacked on.

    Under SPLIT_PROTOCOL=full the generator deliberately fits one delta on the
    complete source dataset, including that source's own evaluation half, and
    delivers it only to the other dataset. Comparing it against every
    evaluation id flagged that as leakage and aborted the run, so `full` could
    not be generated with cross-dataset delivery at all.
    """

    def test_check_uses_the_delivered_ids(self) -> None:
        script = (ROOT / "run_per_dataset.py").read_text(encoding="utf-8")
        self.assertIn("overlap = train_ids & set(attacked_eval_ids)", script)
        # The old artifact-level form compared against every evaluation id.
        self.assertNotIn("if train_ids & evaluation_ids:", script)

    def test_check_runs_after_the_delivered_set_is_known(self) -> None:
        script = (ROOT / "run_per_dataset.py").read_text(encoding="utf-8")
        computed = script.index("attacked_eval_ids = sorted(")
        checked = script.index("overlap = train_ids & set(attacked_eval_ids)")
        appended = script.index("delivery_rows[setting].append({")
        self.assertLess(computed, checked)
        self.assertLess(checked, appended)

    def test_the_pre_generation_protocol_check_is_kept(self) -> None:
        script = (ROOT / "run_per_dataset.py").read_text(encoding="utf-8")
        self.assertIn("if attack_train_ids & evaluation_ids:", script)

    def test_protocol_ids_never_collide_across_datasets(self) -> None:
        """Why delivering a complete-source delta to the other dataset is safe.

        VisA ids are namespaced and MVTec ids are not, so a delta fitted on
        every MVTec image cannot contain a VisA id. Without this the complete
        source delta would be unsafe rather than merely unusual.
        """

        from adversarial_harness.dataset import MVTecSample

        def ids(dataset: str) -> set:
            return {
                MVTecSample(
                    index=0, category=category, defect_type=defect,
                    image_path=Path(f"/{dataset}/{category}/{defect}/000.png"),
                    mask_path=None, label=1, split=split, dataset=dataset,
                ).protocol_id
                for category in ("bottle", "candle", "visa")
                for defect in ("good", "broken_large")
                for split in ("train", "test")
            }

        self.assertEqual(ids("mvtec") & ids("visa"), set())
