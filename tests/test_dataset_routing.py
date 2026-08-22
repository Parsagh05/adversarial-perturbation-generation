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
