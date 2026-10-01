"""CONTINUE_FROM: which earlier delta a run_per_dataset.py delta continues from.

The runner needs a GPU at import, so the matcher is taken from its source.
"""

import ast
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
NAME = "dataset__normal_to_abnormal__global.pt"


def _matcher(continue_from: Path):
    tree = ast.parse((ROOT / "run_per_dataset.py").read_text(encoding="utf-8"))
    wanted = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "continuation_source")
        or (
            isinstance(node, ast.Assign)
            and any(getattr(target, "id", "") == "CONTINUATION_IGNORED" for target in node.targets)
        )
    ]
    namespace = {"torch": torch, "Path": Path, "Dict": dict, "CONTINUE_FROM": str(continue_from)}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "run_per_dataset.py", "exec"), namespace)
    return namespace["continuation_source"]


def _expected(steps: int, **overrides) -> dict:
    return {
        "source_dataset": "mvtec", "direction": "normal_to_abnormal", "loss_mode": "global",
        "seed": 111, "universal_batch_size": 64, "optimization_epochs": steps // 10,
        "universal_steps": steps, "attack_code_sha256": "new", "benchmark_commit": "new",
        **overrides,
    }


class ContinuationSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())

    def write(self, folder: str, steps: int, name: str = NAME, **overrides) -> Path:
        path = self.root / folder / "noises" / "mvtec" / "perturbations" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = _expected(steps, attack_code_sha256="old", benchmark_commit="old", **overrides)
        torch.save({"delta": torch.zeros(1, 3, 2, 2).half(), "metadata": metadata}, path)
        return path

    def test_the_longest_shorter_matching_run_is_chosen(self) -> None:
        self.write("ep5", 50)
        longest = self.write("ep20", 200)
        self.write("ep40", 400)  # the run's own budget: nothing to continue
        self.write("ep60", 600)  # longer than the run
        steps, path, _ = _matcher(self.root)(NAME, _expected(400))
        self.assertEqual((steps, path), (200, longest.resolve()))

    def test_code_provenance_may_differ_but_settings_may_not(self) -> None:
        self.write("seed", 200, seed=222)
        self.write("batch", 200, universal_batch_size=8)
        self.write("random", 200, delta_source="random_rademacher")
        self.assertIsNone(_matcher(self.root)(NAME, _expected(400)))
        match = self.write("ok", 100)
        self.assertEqual(_matcher(self.root)(NAME, _expected(400))[1], match.resolve())

    def test_another_direction_or_loss_is_never_used(self) -> None:
        self.write("other", 200, name="dataset__normal_to_abnormal__local.pt", loss_mode="local")
        self.assertIsNone(_matcher(self.root)(NAME, _expected(400)))


if __name__ == "__main__":
    unittest.main()
