import os
from types import SimpleNamespace
import unittest


os.environ.setdefault("MVTEC_ROOT", ".")
os.environ.setdefault("VISA_ROOT", ".")
os.environ.setdefault("OUTPUT_BASE", ".")

from common import _balanced_category_groups, generation_datasets, parse_numeric


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

    def test_accepts_single_or_both_datasets(self) -> None:
        os.environ["GENERATION_DATASETS"] = "mvtec"
        self.assertEqual(generation_datasets(), ("mvtec",))
        os.environ["GENERATION_DATASETS"] = "mvtec,visa"
        self.assertEqual(generation_datasets(), ("mvtec", "visa"))

    def test_rejects_unknown_or_duplicate_datasets(self) -> None:
        for value in ("mvtec,mvtec", "unknown", ""):
            os.environ["GENERATION_DATASETS"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                generation_datasets()


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


if __name__ == "__main__":
    unittest.main()
