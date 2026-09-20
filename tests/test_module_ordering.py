"""Module-level execution order in the runner entry points.

The runners cannot be imported in a test: they require a CUDA device, an
AnomalyCLIP checkout and both datasets before the first statement of interest
runs. So their module-level order is checked statically instead.

That is not a weaker check for this class of bug. SNAPSHOT_TARGETS read
PROMPT_MODE thirty lines before it was defined, and every run passed anyway,
because a list comprehension evaluates its element expression only once the
iterable yields something and snapshot_targets() returns () unless
SNAPSHOT_EPOCHS is set. The defect was always present in the source and
invisible at runtime, which is exactly what a static check sees and a
smoke run does not.
"""

from __future__ import annotations

import ast
import builtins
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNERS = ("run_per_dataset.py", "run_per_category.py", "run_per_image.py")
ENTRY_POINTS = RUNNERS + (
    "audit_generation.py", "setup_catalog.py", "common.py",
    "package_full_outputs.py", "ensure_prompt_checkpoint.py",
)
DEFERRED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _stores(node: ast.AST) -> set[str]:
    """Every name this top-level statement binds.

    Includes comprehension and loop targets, imported aliases, and functions
    defined inside a module-level loop, all of which are bound by the time
    the statement finishes.
    """

    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            names.add(sub.id)
        elif isinstance(sub, ast.alias):
            names.add((sub.asname or sub.name).split(".")[0])
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(sub.name)
    return names - {""}


def _eager_loads(statement: ast.stmt) -> list[ast.Name]:
    """Names read the moment this top-level statement executes.

    Bodies of functions, classes and lambdas are skipped: they run later, so
    reading a name defined further down the module is legitimate there.
    """

    found: list[ast.Name] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, DEFERRED):
            for decorator in getattr(node, "decorator_list", []):
                visit(decorator)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            found.append(node)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    if isinstance(statement, DEFERRED):
        for decorator in getattr(statement, "decorator_list", []):
            visit(decorator)
        return found
    visit(statement)
    return found


def forward_references(source: str) -> list[tuple[int, str]]:
    """Module-level names read before any statement binds them."""

    tree = ast.parse(source)
    bound = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__spec__"}
    problems = []
    for statement in tree.body:
        local = _stores(statement)
        for name in _eager_loads(statement):
            if name.id not in bound and name.id not in local:
                problems.append((name.lineno, name.id))
        bound |= local
    return problems


class ModuleLevelOrderingTests(unittest.TestCase):
    def test_no_entry_point_reads_a_name_before_it_is_defined(self) -> None:
        for name in ENTRY_POINTS:
            with self.subTest(module=name):
                found = forward_references(
                    (REPO / name).read_text(encoding="utf-8")
                )
                self.assertEqual(
                    found, [],
                    f"{name} reads these at module level before binding them: "
                    + ", ".join(f"{who} (line {line})" for line, who in found),
                )

    def test_snapshot_targets_is_built_after_prompt_mode(self) -> None:
        """The specific ordering SNAPSHOT_EPOCHS depends on."""

        for name in RUNNERS:
            with self.subTest(runner=name):
                source = (REPO / name).read_text(encoding="utf-8")
                prompt_mode = source.index(
                    'PROMPT_MODE = os.environ.get("PROMPT_MODE"'
                )
                snapshot = source.index("SNAPSHOT_TARGETS = [")
                self.assertLess(
                    prompt_mode, snapshot,
                    f"{name}: SNAPSHOT_TARGETS uses PROMPT_MODE before it is "
                    "defined; an empty snapshot_targets() hides this until "
                    "SNAPSHOT_EPOCHS is set",
                )

    def test_the_detector_sees_a_comprehension_forward_reference(self) -> None:
        """Guard the guard: the empty-iterable shape must be caught."""

        source = "\n".join((
            "import os",
            "VALUES = [str(LATER) for item in os.environ]",
            "LATER = 1",
        ))
        self.assertEqual(forward_references(source), [(2, "LATER")])

    def test_the_detector_allows_ordinary_deferred_reads(self) -> None:
        """A function body may legitimately read a name defined below it."""

        source = "\n".join((
            "def read():",
            "    return LATER",
            "LATER = 1",
        ))
        self.assertEqual(forward_references(source), [])


if __name__ == "__main__":
    unittest.main()
