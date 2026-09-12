"""Canonical standalone generation setups.

The matrix is generated from parameter lists rather than written out entry by
entry, so widening a sweep means editing one list. ``SETUP_STEPS`` holds one
``dataset:category:image`` triple per configuration, because the scopes solve
different problems: a per-dataset delta must satisfy hundreds of images at
once, a per-category delta about a dozen, and a per-image delta exactly one.
``SETUP_STEPS=800:200:100,500:150:50`` sweeps two such settings. A bare
``500`` still means all three scopes use 500. Cross-dataset takes no value of
its own: it delivers the per-dataset delta.

A setup ID is a pure function of the settings that change the work, so a run
can never be filed under a name that describes different parameters. Overriding
steps to 1200 produces ``steps1200_...`` on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import os


@dataclass(frozen=True)
class Setup:
    # Per-dataset steps; cross_dataset delivers this same delta.
    steps: int
    category_steps: int
    image_steps: int
    epsilon: float
    epsilon_label: str
    loss_formulation: str
    prompt_mode: str


def _epsilon_tag(epsilon_label: str) -> str:
    """``"2/255"`` -> ``"eps2"``; keeps fractional budgets filesystem-safe."""

    numerator = str(epsilon_label).split("/")[0].strip()
    return "eps" + numerator.replace(".", "p")


def _fraction_tag(attack_train_fraction: float) -> str:
    """``0.2`` -> ``"train20"``. A full run adds nothing, keeping IDs stable."""

    if abs(float(attack_train_fraction) - 1.0) < 1e-12:
        return ""
    percent = f"{float(attack_train_fraction) * 100:g}"
    return "train" + percent.replace(".", "p")


SPLIT_PROTOCOLS = ("balanced", "full")


def _protocol_tag(split_protocol: str) -> str:
    """``""`` for the historical balanced protocol, ``full`` for the new one.

    balanced adds no component so existing output names are unchanged, the same
    way a full train fraction does not.
    """

    if split_protocol not in SPLIT_PROTOCOLS:
        raise ValueError(
            f"split_protocol must be one of {SPLIT_PROTOCOLS}, got {split_protocol!r}"
        )
    return "" if split_protocol == "balanced" else "full"


def split_protocol_setting() -> str:
    """``balanced`` (downsample each category to min) or ``full`` (keep all)."""

    protocol = os.environ.get("SPLIT_PROTOCOL", "balanced").strip().lower()
    if protocol not in SPLIT_PROTOCOLS:
        raise ValueError(
            f"SPLIT_PROTOCOL must be one of {SPLIT_PROTOCOLS}, got {protocol!r}"
        )
    return protocol


def _steps_tag(steps: int, category_steps: int, image_steps: int) -> str:
    """``steps800`` when the scopes agree, ``steps800_cat200_img100`` when not.

    Keeping the compact form for a uniform setting leaves historical output
    names untouched, the same way a full train fraction adds no component.
    """

    tag = f"steps{int(steps)}"
    if int(category_steps) == int(steps) and int(image_steps) == int(steps):
        return tag
    return f"{tag}_cat{int(category_steps)}_img{int(image_steps)}"


def compose_setup_id(
    steps: int,
    category_steps: int,
    image_steps: int,
    epsilon_label: str,
    loss_formulation: str,
    prompt_mode: str,
    attack_train_fraction: float = 1.0,
    split_protocol: str = "balanced",
) -> str:
    """Build the canonical ID for one effective configuration.

    Every component that changes the produced perturbations appears in the
    name. ``_learnable_prompt`` stays last so ``${id%_learnable_prompt}`` keeps
    recovering the frozen base ID in the launcher.
    """

    parts = [
        _steps_tag(steps, category_steps, image_steps),
        _epsilon_tag(epsilon_label),
    ]
    if loss_formulation == "margin_topk":
        parts.append("margin_topk")
    protocol = _protocol_tag(split_protocol)
    if protocol:
        parts.append(protocol)
    fraction = _fraction_tag(attack_train_fraction)
    if fraction:
        parts.append(fraction)
    if prompt_mode == "learnable_object_agnostic":
        parts.append("learnable_prompt")
    return "_".join(parts)


def effective_setup_id(
    setup: Setup,
    steps: int | None = None,
    attack_train_fraction: float = 1.0,
    split_protocol: str = "balanced",
) -> str:
    """Canonical ID for a catalog entry after any step/fraction override.

    ``steps`` overrides every scope at once, which is what SMOKE_TEST does.
    """

    return compose_setup_id(
        setup.steps if steps is None else steps,
        setup.category_steps if steps is None else steps,
        setup.image_steps if steps is None else steps,
        setup.epsilon_label,
        setup.loss_formulation,
        setup.prompt_mode,
        attack_train_fraction,
        split_protocol,
    )


def parse_epsilon(epsilon_label: str) -> float:
    """Parse ``"4/255"`` or ``"0.02"`` without importing the runtime modules."""

    text = str(epsilon_label).strip()
    parts = [part.strip() for part in text.split("/")]
    if len(parts) == 1:
        value = float(Fraction(parts[0]))
    elif len(parts) == 2 and all(parts):
        value = float(Fraction(parts[0]) / Fraction(parts[1]))
    else:
        raise ValueError(f"Invalid epsilon: {epsilon_label!r}")
    if not 0.0 < value <= 1.0:
        raise ValueError(f"Epsilon must be in (0, 1]: {epsilon_label!r}")
    return value


def _unique_list(name: str, default: str) -> tuple[str, ...]:
    values = tuple(
        part.strip() for part in os.environ.get(name, default).split(",") if part.strip()
    )
    if not values:
        raise ValueError(f"{name} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates: {values}")
    return values


def step_grid() -> tuple[tuple[int, int, int], ...]:
    """Per-scope step counts to sweep, as ``(dataset, category, image)`` triples.

    ``SETUP_STEPS=800:200:100,500`` sweeps a scope-specific setting and a
    uniform one. A bare number expands to the same count for every scope.
    """

    grid = []
    for entry in _unique_list("SETUP_STEPS", "500,800"):
        parts = [part.strip() for part in entry.split(":")]
        if len(parts) == 1:
            parts = parts * 3
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                f"SETUP_STEPS entry must be N or dataset:category:image, got {entry!r}"
            )
        try:
            values = tuple(int(part) for part in parts)
        except ValueError as error:
            raise ValueError(f"SETUP_STEPS entry is not numeric: {entry!r}") from error
        if any(value <= 0 for value in values):
            raise ValueError(f"SETUP_STEPS must be positive: {entry!r}")
        grid.append(values)
    if len(set(grid)) != len(grid):
        raise ValueError(f"SETUP_STEPS contains duplicate settings: {tuple(grid)}")
    return tuple(grid)


def epsilon_grid() -> tuple[str, ...]:
    """Linf budgets to sweep. Override with ``SETUP_EPSILONS=2/255,4/255,8/255``."""

    labels = _unique_list("SETUP_EPSILONS", "2/255,4/255")
    for label in labels:
        parse_epsilon(label)
    return labels


LOSS_FORMULATIONS = ("ce_focal_dice", "margin_topk")
PROMPT_MODES = ("frozen_winclip", "learnable_object_agnostic")


def build_setups(
    steps_grid: tuple[tuple[int, int, int], ...] | None = None,
    epsilons: tuple[str, ...] | None = None,
) -> dict[str, Setup]:
    """Cartesian product over prompt family, loss, steps and epsilon.

    The iteration order groups by prompt family first, then loss formulation,
    matching how the setups are run and reported.
    """

    steps_grid = step_grid() if steps_grid is None else steps_grid
    epsilons = epsilon_grid() if epsilons is None else epsilons
    setups: dict[str, Setup] = {}
    for prompt_mode in PROMPT_MODES:
        for loss_formulation in LOSS_FORMULATIONS:
            for steps, category_steps, image_steps in steps_grid:
                for label in epsilons:
                    setup_id = compose_setup_id(
                        steps, category_steps, image_steps,
                        label, loss_formulation, prompt_mode,
                    )
                    setups[setup_id] = Setup(
                        steps=steps,
                        category_steps=category_steps,
                        image_steps=image_steps,
                        epsilon=parse_epsilon(label),
                        epsilon_label=label,
                        loss_formulation=loss_formulation,
                        prompt_mode=prompt_mode,
                    )
    return setups


SETUPS = build_setups()
