"""GENERATION_DEADLINE helpers of run_per_dataset.py.

The runner needs a GPU at import, so the helpers are taken from its source.
"""

import ast
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]


def _helpers(deadline: float) -> dict:
    tree = ast.parse((ROOT / "run_per_dataset.py").read_text(encoding="utf-8"))
    wanted = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"deadline_check", "write_checkpoint"}
    ]
    import os
    import time
    namespace = {"torch": torch, "Path": Path, "os": os, "time": time,
                 "GENERATION_DEADLINE": deadline}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "run_per_dataset.py", "exec"), namespace)
    return namespace


class DeadlineCheckTests(unittest.TestCase):
    def test_stops_before_a_step_that_would_end_past_the_deadline(self) -> None:
        clock = iter([0.0, 100.0, 200.0, 300.0, 400.0])
        should_stop = _helpers(deadline=520.0)["deadline_check"]()
        with mock.patch("time.time", lambda: next(clock)):
            # Steps take 100 s: at 400 the next would end at 400 + 1.2 * 100 > 520.
            self.assertEqual([should_stop() for _ in range(5)], [False, False, False, False, True])

    def test_the_first_step_only_needs_the_deadline_not_passed(self) -> None:
        with mock.patch("time.time", lambda: 10.0):
            self.assertFalse(_helpers(deadline=11.0)["deadline_check"]()())
            self.assertTrue(_helpers(deadline=10.0)["deadline_check"]()())


class WriteCheckpointTests(unittest.TestCase):
    def test_round_trip_in_float32_with_snapshots(self) -> None:
        path = Path(tempfile.mkdtemp()) / "noises" / "x.pt"
        delta = torch.rand(1, 3, 4, 4, dtype=torch.float64)
        _helpers(deadline=0.0)["write_checkpoint"](
            path, delta, {30: torch.ones(1, 3, 4, 4)}, {"universal_steps": 7}
        )
        payload = torch.load(path, weights_only=False)
        self.assertEqual(payload["delta"].dtype, torch.float32)
        self.assertTrue(torch.equal(payload["delta"], delta.float()))
        self.assertEqual(list(payload["snapshots"]), [30])
        self.assertEqual(payload["metadata"], {"universal_steps": 7})
        self.assertFalse(path.with_name("x.pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
