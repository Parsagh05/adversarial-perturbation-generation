"""WRITE_BUNDLE_ARCHIVES gates every per-bundle archive.

bundle.zip is a shipping convenience: every file in it already sits in the
bundle directory beside it, and the evaluator reads the directory. A pipeline
that evaluates in place pays to build it and uses none of it.

It also currently triggers a live evaluator defect. The evaluator reads
setup_id from the manifest but prompt_mode from the bundle path, and an
extracted archive has no prompt-family path component, so with both families
a learnable bundle is read as frozen and collides with the real frozen one
("Conflicting duplicate condition"). This switch stops our runs producing that
input; it does not fix the evaluator, which still needs its own fix because it
advertises ZIP bundles as a supported shape.

The runners cannot be imported here (CUDA, an AnomalyCLIP checkout and both
datasets are required first), so the guard is checked structurally: every
archive write must sit inside the flag. That is the property worth pinning -
a fourth write added outside the guard would make the flag silently partial.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNERS = ("run_per_dataset.py", "run_per_category.py", "run_per_image.py")
FLAG = "WRITE_BUNDLE_ARCHIVES"


def _guarded_spans(tree: ast.AST) -> list[tuple[int, int]]:
    """Line ranges of every `if WRITE_BUNDLE_ARCHIVES:` body."""

    spans = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
            if node.test.id == FLAG:
                for statement in node.body:
                    spans.append((statement.lineno, statement.end_lineno))
    return spans


def _inside(spans: list[tuple[int, int]], line: int) -> bool:
    return any(start <= line <= end for start, end in spans)


class BundleArchiveFlagTests(unittest.TestCase):
    def _tree(self, name: str) -> ast.AST:
        return ast.parse((REPO / name).read_text(encoding="utf-8"))

    def test_every_runner_reads_the_flag_defaulting_to_true(self) -> None:
        """Default true, so existing behaviour is unchanged."""

        for name in RUNNERS:
            with self.subTest(runner=name):
                source = (REPO / name).read_text(encoding="utf-8")
                self.assertIn(
                    f'{FLAG} = bool_env("{FLAG}", True)', source,
                    f"{name} must read {FLAG}, defaulting to True",
                )

    def test_no_archive_is_written_outside_the_guard(self) -> None:
        """Catches a new write site that forgets the flag."""

        for name in RUNNERS:
            with self.subTest(runner=name):
                tree = self._tree(name)
                spans = _guarded_spans(tree)
                self.assertTrue(spans, f"{name} has no {FLAG} guard")
                for node in ast.walk(tree):
                    mentions_bundle_zip = (
                        isinstance(node, ast.Constant)
                        and node.value == "bundle.zip"
                    )
                    opens_for_writing = (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "ZipFile"
                        and any(
                            isinstance(argument, ast.Constant)
                            and argument.value == "w"
                            for argument in node.args
                        )
                    )
                    if mentions_bundle_zip or opens_for_writing:
                        self.assertTrue(
                            _inside(spans, node.lineno),
                            f"{name} line {node.lineno}: archive work outside "
                            f"the {FLAG} guard",
                        )

    def test_the_reopen_verification_is_skipped_with_the_write(self) -> None:
        """It reads the archive's namelist; with no archive it has nothing."""

        for name in RUNNERS:
            with self.subTest(runner=name):
                tree = self._tree(name)
                spans = _guarded_spans(tree)
                found = False
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Name)
                        and node.id == "expected_archive_names"
                    ):
                        found = True
                        self.assertTrue(
                            _inside(spans, node.lineno),
                            f"{name} line {node.lineno}: archive verification "
                            "runs even when no archive was written",
                        )
                self.assertTrue(found, f"{name}: no archive verification found")

    def test_the_shell_layer_exposes_the_flag(self) -> None:
        launcher = (REPO / "train.sh").read_text(encoding="utf-8")
        config = (REPO / "config.sh").read_text(encoding="utf-8")
        self.assertIn(f'export {FLAG}="${{{FLAG}:-true}}"', launcher)
        self.assertIn(f'{FLAG}="${{{FLAG}:-true}}"', config)

    def test_the_combined_archive_is_untouched(self) -> None:
        """full_outputs.zip lives outside setups/ and already excludes *.zip.

        Dropping bundle.zip does not change its contents, so it stays
        ungated: a no-archive run still gets one shippable artifact.
        """

        source = (REPO / "package_full_outputs.py").read_text(encoding="utf-8")
        self.assertNotIn(FLAG, source)
        self.assertIn('suffix.lower() != ".zip"', source)


if __name__ == "__main__":
    unittest.main()
