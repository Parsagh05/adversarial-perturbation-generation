"""Canonical standalone generation setups.

The matrix is generated from parameter lists rather than written out entry by
entry, so widening a sweep means editing one list. ``SETUP_EPOCHS`` holds one
``dataset:category:image`` triple per configuration, because the scopes solve
different problems: a per-dataset delta must satisfy hundreds of images at
once, a per-category delta about a dozen, and a per-image delta exactly one.
``SETUP_EPOCHS=7.14:100:100,5:60:50`` sweeps two such settings. A bare ``100``
means all three scopes use 100. Cross-dataset takes no value of its own: it
either delivers the per-dataset delta (``halfcross``) or uses the same epoch
budget for its complete-cohort delta (``fullcross``).

An epoch is one pass over whatever that delta trains on, so the PGD step count
is derived at run time as ``ceil(epochs * ceil(n_images / batch))``. That keeps
the budget constant when the training set changes size, which it does between
SPLIT_PROTOCOL=balanced and full. A per-image delta trains on one image, so
there epochs and steps are the same number.

A setup ID is a pure function of the settings that change the work, so a run
can never be filed under a name that describes different parameters. Overriding
the epoch budget to 12 produces ``ep12_...`` on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import os


@dataclass(frozen=True)
class Setup:
    # Per-dataset epochs; cross_dataset delivers this same delta.
    epochs: float
    category_epochs: float
    image_epochs: float
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


def full_data_cross_setting() -> bool:
    """Whether cross-dataset uses the complete source and target cohorts."""

    raw = os.environ.get("FULL_DATA_CROSS", "true").strip().lower()
    if raw not in {"true", "false"}:
        raise ValueError(f"FULL_DATA_CROSS must be true or false, got {raw!r}")
    return raw == "true"


# Mirrors VALID_STEP_SIZE_SCHEDULES in adversarial_harness.config. Declared
# here too so the launcher can derive names without importing the attack code,
# the same way SPLIT_PROTOCOLS is declared in both places.
STEP_SIZE_SCHEDULES = ("constant", "linear", "cosine")


def _schedule_tag(step_size_schedule: str) -> str:
    """``""`` for the default flat step, ``linear_step`` / ``cosine_step`` else.

    A decaying step size depends on the total step count, so the first N steps
    of a long run differ from an N-step run. constant keeps them identical,
    which is what lets one long run be sliced into shorter budgets.
    """

    if step_size_schedule not in STEP_SIZE_SCHEDULES:
        raise ValueError(
            f"step_size_schedule must be one of {STEP_SIZE_SCHEDULES}, got "
            f"{step_size_schedule!r}"
        )
    return "" if step_size_schedule == "constant" else f"{step_size_schedule}_step"


def step_size_schedule_setting() -> str:
    """Flat step by default; the decaying schedules name themselves."""

    schedule = os.environ.get("STEP_SIZE_SCHEDULE", "constant").strip().lower()
    if schedule not in STEP_SIZE_SCHEDULES:
        raise ValueError(
            f"STEP_SIZE_SCHEDULE must be one of {STEP_SIZE_SCHEDULES}, got "
            f"{schedule!r}"
        )
    return schedule


def _hinge_tag(margin_hinge_displacement: float | None) -> str:
    """``0.25`` -> ``"hinge0p25"``; the unhinged margin adds nothing."""

    if margin_hinge_displacement is None:
        return ""
    value = float(margin_hinge_displacement)
    if value < 0.0:
        raise ValueError("margin_hinge_displacement must be non-negative or None")
    return "hinge" + f"{value:g}".replace(".", "p")


def margin_hinge_setting() -> float | None:
    """Displacement past each image's clean margin, or None for no hinge."""

    raw = os.environ.get("MARGIN_HINGE_DISPLACEMENT", "").strip()
    if not raw or raw.lower() in {"none", "off", "false"}:
        return None
    value = float(raw)
    if value < 0.0:
        raise ValueError(
            f"MARGIN_HINGE_DISPLACEMENT must be non-negative or empty, got {raw!r}"
        )
    return value


def _epoch_number(value: float) -> str:
    """``7.14`` -> ``7p14``; keeps fractional budgets filesystem-safe."""

    return f"{float(value):g}".replace(".", "p")


def _epochs_tag(epochs: float, category_epochs: float, image_epochs: float) -> str:
    """``ep100`` when the scopes agree, ``ep7p14_cat100_img100`` when not."""

    tag = f"ep{_epoch_number(epochs)}"
    if float(category_epochs) == float(epochs) and float(image_epochs) == float(epochs):
        return tag
    return (
        f"{tag}_cat{_epoch_number(category_epochs)}"
        f"_img{_epoch_number(image_epochs)}"
    )


def compose_setup_id(
    epochs: float,
    category_epochs: float,
    image_epochs: float,
    epsilon_label: str,
    loss_formulation: str,
    prompt_mode: str,
    attack_train_fraction: float = 1.0,
    split_protocol: str = "balanced",
    full_data_cross: bool | None = None,
    step_size_schedule: str = "constant",
    margin_hinge_displacement: float | None = None,
) -> str:
    """Build the canonical ID for one effective configuration.

    Every component that changes the produced perturbations appears in the
    name. ``_learnable_prompt`` stays last so ``${id%_learnable_prompt}`` keeps
    recovering the frozen base ID in the launcher.
    """

    parts = [
        _epochs_tag(epochs, category_epochs, image_epochs),
        _epsilon_tag(epsilon_label),
    ]
    # margin_topk is the default loss and adds nothing; ce_focal_dice names
    # itself, so switching back to it cannot overwrite a default-loss run.
    if loss_formulation == "ce_focal_dice":
        parts.append("ce_focal_dice")
    else:
        # The hinge only applies to the margin terms, so it can only appear on
        # a margin_topk setup; ce_focal_dice is already bounded.
        hinge = _hinge_tag(margin_hinge_displacement)
        if hinge:
            parts.append(hinge)
    schedule = _schedule_tag(step_size_schedule)
    if schedule:
        parts.append(schedule)
    protocol = _protocol_tag(split_protocol)
    if protocol:
        parts.append(protocol)
    if full_data_cross is not None:
        parts.append("fullcross" if full_data_cross else "halfcross")
    fraction = _fraction_tag(attack_train_fraction)
    if fraction:
        parts.append(fraction)
    if prompt_mode == "learnable_object_agnostic":
        parts.append("learnable_prompt")
    return "_".join(parts)


def effective_setup_id(
    setup: Setup,
    epochs: float | None = None,
    attack_train_fraction: float = 1.0,
    split_protocol: str = "balanced",
    full_data_cross: bool | None = None,
    step_size_schedule: str = "constant",
    margin_hinge_displacement: float | None = None,
) -> str:
    """Canonical ID for a catalog entry after any epoch/fraction override.

    ``epochs`` overrides every scope at once, which is what SMOKE_TEST does.
    """

    return compose_setup_id(
        setup.epochs if epochs is None else epochs,
        setup.category_epochs if epochs is None else epochs,
        setup.image_epochs if epochs is None else epochs,
        setup.epsilon_label,
        setup.loss_formulation,
        setup.prompt_mode,
        attack_train_fraction,
        split_protocol,
        full_data_cross,
        step_size_schedule,
        margin_hinge_displacement,
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


def epoch_grid() -> tuple[tuple[float, float, float], ...]:
    """Per-scope epoch budgets, as ``(dataset, category, image)`` triples.

    ``SETUP_EPOCHS=7.14:100:100,5`` sweeps a scope-specific setting and a
    uniform one. A bare number expands to the same budget for every scope.
    """

    grid = []
    for entry in _unique_list("SETUP_EPOCHS", "7.14:100:100"):
        parts = [part.strip() for part in entry.split(":")]
        if len(parts) == 1:
            parts = parts * 3
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                f"SETUP_EPOCHS entry must be N or dataset:category:image, got {entry!r}"
            )
        try:
            values = tuple(float(part) for part in parts)
        except ValueError as error:
            raise ValueError(f"SETUP_EPOCHS entry is not numeric: {entry!r}") from error
        if any(value <= 0 for value in values):
            raise ValueError(f"SETUP_EPOCHS must be positive: {entry!r}")
        grid.append(values)
    if len(set(grid)) != len(grid):
        raise ValueError(f"SETUP_EPOCHS contains duplicate settings: {tuple(grid)}")
    return tuple(grid)


def derive_steps(epochs: float, n_images: int, batch_size: int) -> int:
    """PGD updates for one delta: ``ceil(epochs * ceil(n_images / batch))``.

    A per-image delta trains on a single image, so this returns the epoch
    count unchanged and the two units coincide.
    """

    if n_images < 1 or batch_size < 1:
        raise ValueError("n_images and batch_size must be positive")
    updates_per_epoch = math.ceil(n_images / batch_size)
    return max(1, math.ceil(float(epochs) * updates_per_epoch))


def epsilon_grid() -> tuple[str, ...]:
    """Linf budgets to sweep. Override with ``SETUP_EPSILONS=2/255,4/255,8/255``."""

    labels = _unique_list("SETUP_EPSILONS", "2/255,4/255")
    for label in labels:
        parse_epsilon(label)
    return labels


LOSS_FORMULATIONS = ("ce_focal_dice", "margin_topk")
PROMPT_MODES = ("frozen_winclip", "learnable_object_agnostic")


def build_setups(
    epochs_grid: tuple[tuple[float, float, float], ...] | None = None,
    epsilons: tuple[str, ...] | None = None,
    split_protocol: str = "balanced",
    full_data_cross: bool | None = None,
) -> dict[str, Setup]:
    """Cartesian product over prompt family, loss, epochs and epsilon.

    The iteration order groups by prompt family first, then loss formulation,
    matching how the setups are run and reported.
    """

    epochs_grid = epoch_grid() if epochs_grid is None else epochs_grid
    epsilons = epsilon_grid() if epsilons is None else epsilons
    setups: dict[str, Setup] = {}
    for prompt_mode in PROMPT_MODES:
        for loss_formulation in LOSS_FORMULATIONS:
            for epochs, category_epochs, image_epochs in epochs_grid:
                for label in epsilons:
                    setup_id = compose_setup_id(
                        epochs, category_epochs, image_epochs,
                        label, loss_formulation, prompt_mode,
                        split_protocol=split_protocol,
                        full_data_cross=full_data_cross,
                    )
                    setups[setup_id] = Setup(
                        epochs=epochs,
                        category_epochs=category_epochs,
                        image_epochs=image_epochs,
                        epsilon=parse_epsilon(label),
                        epsilon_label=label,
                        loss_formulation=loss_formulation,
                        prompt_mode=prompt_mode,
                    )
    return setups


SETUPS = build_setups()
