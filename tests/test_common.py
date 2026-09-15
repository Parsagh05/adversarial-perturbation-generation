import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import pandas as pd


os.environ.setdefault("MVTEC_ROOT", ".")
os.environ.setdefault("VISA_ROOT", ".")
os.environ.setdefault("OUTPUT_BASE", ".")

import common
from common import (
    _balanced_category_groups,
    evaluation_datasets,
    generation_datasets,
    parse_numeric,
    protocol_datasets,
    prepare_protocol_split,
    retained_protocol_samples,
    source_datasets,
)


class NumericConfigurationTests(unittest.TestCase):
    def test_parses_integer_and_decimal_fraction_expressions(self) -> None:
        self.assertAlmostEqual(parse_numeric("8/255"), 8.0 / 255.0)
        self.assertAlmostEqual(parse_numeric("0.25/255"), 0.25 / 255.0)
        self.assertAlmostEqual(parse_numeric("0.5"), 0.5)

    def test_rejects_invalid_or_zero_denominator(self) -> None:
        with self.assertRaises(ValueError):
            parse_numeric("1/2/3")
        with self.assertRaises(ValueError):
            parse_numeric("1/0")


class DatasetSelectionTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("GENERATION_DATASETS", None)
        os.environ.pop("SOURCE_DATASETS", None)
        os.environ.pop("EVALUATION_DATASETS", None)

    def test_separates_source_evaluation_and_protocol_datasets(self) -> None:
        os.environ["SOURCE_DATASETS"] = "mvtec"
        os.environ["EVALUATION_DATASETS"] = "mvtec,visa"
        self.assertEqual(source_datasets(), ("mvtec",))
        self.assertEqual(generation_datasets(), ("mvtec",))
        self.assertEqual(evaluation_datasets(), ("mvtec", "visa"))
        self.assertEqual(protocol_datasets(), ("mvtec", "visa"))

    def test_protocol_union_preserves_order_and_removes_overlap(self) -> None:
        os.environ["SOURCE_DATASETS"] = "visa"
        os.environ["EVALUATION_DATASETS"] = "visa,mvtec"
        self.assertEqual(protocol_datasets(), ("visa", "mvtec"))

    def test_legacy_generation_selection_maps_to_sources(self) -> None:
        os.environ["GENERATION_DATASETS"] = "mvtec,visa"
        self.assertEqual(source_datasets(), ("mvtec", "visa"))

    def test_rejects_unknown_or_duplicate_datasets(self) -> None:
        for value in ("mvtec,mvtec", "unknown", ""):
            os.environ["SOURCE_DATASETS"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                source_datasets()


class BalancedProtocolTests(unittest.TestCase):
    @staticmethod
    def sample(dataset: str, category: str, label: int, index: int):
        return SimpleNamespace(
            dataset=dataset,
            category=category,
            label=label,
            protocol_id=f"{dataset}:{category}:{label}:{index}",
        )

    def test_balances_each_category_for_both_datasets_deterministically(self) -> None:
        samples = []
        specifications = {
            ("mvtec", "bottle"): (3, 7),
            ("mvtec", "transistor"): (6, 4),
            ("visa", "candle"): (5, 8),
            ("visa", "pipe_fryum"): (9, 2),
        }
        for (dataset, category), (normal_count, anomalous_count) in specifications.items():
            samples.extend(
                self.sample(dataset, category, 0, index)
                for index in range(normal_count)
            )
            samples.extend(
                self.sample(dataset, category, 1, index)
                for index in range(anomalous_count)
            )

        first, original = _balanced_category_groups(samples, split_seed=111)
        second, _ = _balanced_category_groups(samples, split_seed=111)

        for (dataset, category), counts in specifications.items():
            expected = min(counts)
            self.assertEqual(len(first[(dataset, category, 0)]), expected)
            self.assertEqual(len(first[(dataset, category, 1)]), expected)
            self.assertEqual(original[(dataset, category, 0)], counts[0])
            self.assertEqual(original[(dataset, category, 1)], counts[1])
        self.assertEqual(
            {key: [sample.protocol_id for sample in value] for key, value in first.items()},
            {key: [sample.protocol_id for sample in value] for key, value in second.items()},
        )

    def test_requires_both_labels_in_every_category(self) -> None:
        samples = [self.sample("mvtec", "bottle", 0, index) for index in range(3)]
        with self.assertRaisesRegex(RuntimeError, "Need both labels"):
            _balanced_category_groups(samples, split_seed=111)

    def test_protocol_keeps_evaluation_only_dataset_out_of_attack_train(self) -> None:
        samples = []
        for dataset, category in (("mvtec", "bottle"), ("visa", "candle")):
            for label in (0, 1):
                for index in range(4):
                    samples.append(
                        SimpleNamespace(
                            dataset=dataset,
                            category=category,
                            label=label,
                            defect_type="good" if label == 0 else "defect",
                            protocol_id=f"{dataset}:{category}:{label}:{index}",
                            image_path=Path(f"/{dataset}/{category}/{label}/{index}.png"),
                            mask_path=None,
                        )
                    )

        import common
        import adversarial_harness.dataset as dataset_module

        with tempfile.TemporaryDirectory() as temporary:
            protocol = Path(temporary) / "protocol"
            train_csv = protocol / "attack_train_indices.csv"
            evaluation_csv = protocol / "evaluation_test_indices.csv"
            complete_csv = protocol / "complete_retained_indices.csv"
            environment = {
                "SOURCE_DATASETS": "mvtec",
                "EVALUATION_DATASETS": "mvtec,visa",
                "SPLIT_SEED": "111",
                "EVALUATION_FRACTION": "0.5",
            }
            with (
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(common, "PROTOCOL_DIR", protocol),
                mock.patch.object(common, "ATTACK_TRAIN_CSV", train_csv),
                mock.patch.object(common, "EVALUATION_CSV", evaluation_csv),
                mock.patch.object(common, "COMPLETE_RETAINED_CSV", complete_csv),
                mock.patch.object(
                    dataset_module, "discover_anomaly_datasets", return_value=samples
                ),
            ):
                prepare_protocol_split()

            train = pd.read_csv(train_csv)
            evaluation = pd.read_csv(evaluation_csv)
            complete = pd.read_csv(complete_csv)
            self.assertEqual(set(train.dataset), {"mvtec"})
            self.assertEqual(set(evaluation.dataset), {"mvtec", "visa"})
            self.assertEqual(set(complete.dataset), {"mvtec", "visa"})
            self.assertEqual(
                set(complete.partition), {"attack_train", "evaluation"}
            )
            visa_complete = complete[complete.dataset.eq("visa")]
            self.assertGreater(len(visa_complete), len(evaluation[evaluation.dataset.eq("visa")]))
            self.assertFalse(set(train.protocol_id) & set(evaluation.protocol_id))


if __name__ == "__main__":
    unittest.main()


class SplitProtocolGroupingTests(unittest.TestCase):
    """The two protocols differ only in whether the surplus label is discarded."""

    def _samples(self):
        from types import SimpleNamespace
        out = []
        for category, (normal, abnormal) in (("bottle", (20, 63)), ("cable", (58, 92))):
            for label, count in ((0, normal), (1, abnormal)):
                for index in range(count):
                    out.append(SimpleNamespace(
                        dataset="mvtec", category=category, label=label,
                        protocol_id=f"test/{category}/{label}/{index:04d}"))
        return out

    def test_balanced_truncates_each_category_to_the_smaller_label(self) -> None:
        groups, sizes = _balanced_category_groups(
            self._samples(), 111, "balanced"
        )
        self.assertEqual(len(groups[("mvtec", "bottle", 0)]), 20)
        self.assertEqual(len(groups[("mvtec", "bottle", 1)]), 20)   # 63 -> 20
        self.assertEqual(len(groups[("mvtec", "cable", 0)]), 58)
        self.assertEqual(len(groups[("mvtec", "cable", 1)]), 58)    # 92 -> 58
        self.assertEqual(sizes[("mvtec", "bottle", 1)], 63)         # original recorded

    def test_full_keeps_every_image_and_its_class_ratio(self) -> None:
        groups, _ = _balanced_category_groups(self._samples(), 111, "full")
        self.assertEqual(len(groups[("mvtec", "bottle", 0)]), 20)
        self.assertEqual(len(groups[("mvtec", "bottle", 1)]), 63)
        self.assertEqual(len(groups[("mvtec", "cable", 1)]), 92)
        self.assertEqual(sum(len(v) for v in groups.values()), 20 + 63 + 58 + 92)

    def test_complete_cross_cohort_respects_the_selected_protocol(self) -> None:
        samples = self._samples()
        with mock.patch.dict(
            os.environ, {"SPLIT_PROTOCOL": "balanced", "SPLIT_SEED": "111"}
        ):
            balanced = retained_protocol_samples(samples)
        with mock.patch.dict(
            os.environ, {"SPLIT_PROTOCOL": "full", "SPLIT_SEED": "111"}
        ):
            full = retained_protocol_samples(samples)
        self.assertEqual(len(balanced), 2 * (20 + 58))
        self.assertEqual(len(full), len(samples))

    def test_protocols_are_named_and_validated(self) -> None:
        self.assertEqual(common.label_policy_for("balanced"), common.LABEL_BALANCE_POLICY)
        self.assertEqual(common.label_policy_for("full"), common.FULL_LABEL_POLICY)
        with mock.patch.dict(os.environ, {"SPLIT_PROTOCOL": "nope"}):
            with self.assertRaises(ValueError):
                common.split_protocol()


class AuditProtocolAssumptionTests(unittest.TestCase):
    """The audit must not assume equal label counts outside ``balanced``.

    Only balanced downsamples a category to equal label counts. The two attack
    directions fit opposite labels, so under full they legitimately train on
    different numbers of images and any equal-count assertion rejects a sound
    run. This has now happened twice, so the gate is pinned here.
    """

    def _audit_source(self) -> str:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        return (root / "audit_generation.py").read_text(encoding="utf-8")

    def test_direction_count_checks_are_gated_on_balanced(self) -> None:
        source = self._audit_source()
        guard = source.index('if SPLIT_PROTOCOL == "balanced":\n            if scope')
        train_check = source.index("attack_train_image_count.nunique() != 1")
        evaluation_check = source.index(
            "evaluation_attacked_image_count.nunique() != 1"
        )
        self.assertLess(guard, train_check)
        self.assertLess(guard, evaluation_check)

    def test_no_ungated_equal_count_assertion_is_added(self) -> None:
        # Every nunique() comparison in the audit is an equal-count claim, so
        # each one needs a protocol gate. Adding another should fail here
        # rather than after a full generation pass.
        source = self._audit_source()
        self.assertEqual(source.count(".nunique()"), 2)

    def test_the_protocol_audit_only_demands_balance_for_balanced(self) -> None:
        source = self._audit_source()
        self.assertIn(
            'if SPLIT_PROTOCOL == "balanced" and not counts[0].eq(counts[1]).all():',
            source,
        )
        # Both labels must be present under either protocol.
        self.assertIn("A protocol stratum is missing a label", source)
