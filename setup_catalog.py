"""Canonical standalone generation setups.

The matrix is generated from parameter lists rather than written out entry by
entry, so widening a sweep means editing one list. ``SETUP_EPOCHS`` holds one
``dataset:category:image`` triple per configuration, because the scopes solve
different problems: a per-dataset delta must satisfy hundreds of images at
once, a per-category delta about a dozen, and a per-image delta exactly one.
``SETUP_EPOCHS=7.14:100:100,5:60:50`` sweeps two such settings. A bare ``100``
means every scope uses 100. A fourth component gives cross-dataset its own
budget (``dataset:cross:category:image``), which is used only by ``fullcross``,
where it optimizes a separate delta on the complete source; ``halfcross``
delivers the per-dataset delta and has nothing to budget. The three-part form
still works, with cross inheriting the dataset budget.

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
    epochs: float
    # Only used when FULL_DATA_CROSS optimizes a separate complete-cohort
    # delta; under halfcross cross_dataset delivers the per-dataset one and
    # has nothing to budget.
    cross_epochs: float
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


def _momentum_tag(momentum_decay: float) -> str:
    """``0.9`` -> ``"mom0p9"``; plain sign-PGD adds nothing."""

    value = float(momentum_decay)
    if not 0.0 <= value <= 1.0:
        raise ValueError("momentum_decay must be in [0, 1]")
    if value == 0.0:
        return ""
    return "mom" + f"{value:g}".replace(".", "p")


def momentum_decay_setting() -> float:
    """Gradient accumulation on the shared update; 0 is plain sign-PGD."""

    raw = os.environ.get("MOMENTUM_DECAY", "").strip()
    if not raw or raw.lower() in {"none", "off", "false"}:
        return 0.0
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"MOMENTUM_DECAY must be in [0, 1], got {raw!r}")
    return value


CHECKPOINT_SELECTIONS = ("best", "final")


def _selection_tag(checkpoint_selection: str) -> str:
    """``""`` for the last iterate, ``best`` for checkpoint selection.

    final is the default because it is what the universal-attack papers
    return, so it adds nothing; best names itself.
    """

    if checkpoint_selection not in CHECKPOINT_SELECTIONS:
        raise ValueError(
            f"checkpoint_selection must be one of {CHECKPOINT_SELECTIONS}, got "
            f"{checkpoint_selection!r}"
        )
    return "" if checkpoint_selection == "final" else "best"


def checkpoint_selection_setting() -> str:
    """``final`` keeps the last iterate; ``best`` selects on attack-train."""

    selection = os.environ.get("CHECKPOINT_SELECTION", "final").strip().lower()
    if selection not in CHECKPOINT_SELECTIONS:
        raise ValueError(
            f"CHECKPOINT_SELECTION must be one of {CHECKPOINT_SELECTIONS}, got "
            f"{selection!r}"
        )
    return selection


def snapshot_epochs_setting() -> tuple[tuple[float, float, float], ...]:
    """Shorter budgets to capture inside one run, as the same kind of triple.

    ``SNAPSHOT_EPOCHS="5:100:100,10:200:200"`` reads exactly like
    ``SETUP_EPOCHS``, so every snapshot names a complete setup of its own and
    each scope snapshots at its own boundary. Empty by default.

    Equivalence to a standalone run of that budget holds only while the step
    size is independent of the total budget, so a decaying
    STEP_SIZE_SCHEDULE is refused rather than silently producing deltas that
    are valid perturbations but not the runs they claim to stand in for.
    """

    raw = os.environ.get("SNAPSHOT_EPOCHS", "").strip()
    if not raw or raw.lower() in {"none", "off", "false"}:
        return ()
    triples = tuple(
        _epoch_budget(entry, "SNAPSHOT_EPOCHS")
        for entry in raw.split(",") if entry.strip()
    )
    if len(set(triples)) != len(triples):
        raise ValueError(f"SNAPSHOT_EPOCHS contains duplicate settings: {triples}")
    if step_size_schedule_setting() != "constant":
        raise ValueError(
            "SNAPSHOT_EPOCHS requires STEP_SIZE_SCHEDULE=constant: a decaying "
            "step size depends on the total budget, so a snapshot would not "
            "equal a standalone run of that budget"
        )
    return tuple(sorted(triples))


def assert_snapshots_fit(
    snapshot_triples: tuple[tuple[float, float, float], ...],
    budget: tuple[float, float, float],
) -> None:
    """Every snapshot must be a prefix of the run it is taken from.

    A scope can only be stopped early, never extended, and a snapshot equal to
    the run in all three scopes would claim the run's own setup ID.
    """

    for triple in snapshot_triples:
        if any(value > limit for value, limit in zip(triple, budget)):
            raise ValueError(
                f"SNAPSHOT_EPOCHS entry {triple} exceeds the run budget {budget} "
                "in at least one scope"
            )
        if triple == tuple(budget):
            raise ValueError(
                f"SNAPSHOT_EPOCHS entry {triple} is the run's own budget"
            )


def snapshot_steps(
    snapshot_epochs, n_images: int, batch_size: int
) -> tuple[int, ...]:
    """Absolute step indices for one scope's listed boundaries.

    Uses the same derivation as the run's own budget, so an integer epoch
    lands on the same step index in a short run and a long one.
    """

    return tuple(
        derive_steps(epochs, n_images, batch_size) for epochs in snapshot_epochs
    )


def snapshot_targets() -> tuple[tuple[tuple[float, float, float, float], str], ...]:
    """``((dataset, cross, category, image), setup root)`` from the launcher.

    Each entry is a shorter budget the current run also produces. The launcher
    owns the naming and hands over absolute roots, so a runner never composes a
    setup ID of its own.
    """

    raw = os.environ.get("SNAPSHOT_SETUP_ROOTS", "").strip().strip(";")
    targets = []
    for entry in raw.split(";"):
        if not entry.strip():
            continue
        budget, _, root = entry.partition("=")
        values = tuple(float(part) for part in budget.split(":"))
        if len(values) != 4 or not root:
            raise ValueError(f"Malformed SNAPSHOT_SETUP_ROOTS entry: {entry!r}")
        targets.append((values, root))
    return tuple(targets)


def _epoch_number(value: float) -> str:
    """``7.14`` -> ``7p14``; keeps fractional budgets filesystem-safe."""

    return f"{float(value):g}".replace(".", "p")


def _epochs_tag(
    epochs: float,
    cross_epochs: float,
    category_epochs: float,
    image_epochs: float,
    full_data_cross: bool | None = None,
) -> str:
    """``ep100`` when every scope agrees, ``ep7p14_cat100_img100`` when not.

    ``_cross`` appears only when cross_dataset both has a budget of its own to
    spend and differs from the per-dataset one. Under ``halfcross`` it spends
    none, because it delivers the per-dataset delta, so the value changes
    nothing and must not split the output directory. Every name predating the
    cross budget is likewise unchanged.
    """

    tag = f"ep{_epoch_number(epochs)}"
    cross_is_optimized = full_data_cross is not False
    if cross_is_optimized and float(cross_epochs) != float(epochs):
        tag = f"{tag}_cross{_epoch_number(cross_epochs)}"
    if float(category_epochs) == float(epochs) and float(image_epochs) == float(epochs):
        return tag
    return (
        f"{tag}_cat{_epoch_number(category_epochs)}"
        f"_img{_epoch_number(image_epochs)}"
    )


def compose_setup_id(
    epochs: float,
    cross_epochs: float,
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
    momentum_decay: float = 0.0,
    checkpoint_selection: str = "final",
) -> str:
    """Build the canonical ID for one effective configuration.

    Every component that changes the produced perturbations appears in the
    name. ``_learnable_prompt`` stays last so ``${id%_learnable_prompt}`` keeps
    recovering the frozen base ID in the launcher.
    """

    parts = [
        _epochs_tag(
            epochs, cross_epochs, category_epochs, image_epochs, full_data_cross
        ),
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
    momentum = _momentum_tag(momentum_decay)
    if momentum:
        parts.append(momentum)
    schedule = _schedule_tag(step_size_schedule)
    if schedule:
        parts.append(schedule)
    selection = _selection_tag(checkpoint_selection)
    if selection:
        parts.append(selection)
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
    momentum_decay: float = 0.0,
    checkpoint_selection: str = "final",
) -> str:
    """Canonical ID for a catalog entry after any epoch/fraction override.

    ``epochs`` overrides every scope at once, which is what SMOKE_TEST does.
    """

    return compose_setup_id(
        setup.epochs if epochs is None else epochs,
        setup.cross_epochs if epochs is None else epochs,
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
        momentum_decay,
        checkpoint_selection,
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


def _epoch_budget(entry: str, name: str) -> tuple[float, float, float, float]:
    """``(dataset, cross, category, image)`` from 1, 3 or 4 components.

    A bare ``"5"`` is every scope. Three components are the historical
    ``dataset:category:image``, with cross_dataset inheriting the dataset
    budget, which is what it used before it had one of its own. Four are
    ``dataset:cross:category:image``.
    """

    parts = [part.strip() for part in entry.split(":")]
    if len(parts) == 1:
        parts = parts * 4
    elif len(parts) == 3:
        parts = [parts[0], parts[0], parts[1], parts[2]]
    if len(parts) != 4 or not all(parts):
        raise ValueError(
            f"{name} entry must be N, dataset:category:image, or "
            f"dataset:cross:category:image, got {entry!r}"
        )
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"{name} entry is not numeric: {entry!r}") from error
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} must be positive: {entry!r}")
    return values


def epoch_grid() -> tuple[tuple[float, float, float, float], ...]:
    """Per-scope epoch budgets, as ``(dataset, category, image)`` triples.

    ``SETUP_EPOCHS=7.14:100:100,5`` sweeps a scope-specific setting and a
    uniform one. A bare number expands to the same budget for every scope.
    """

    grid = [
        _epoch_budget(entry, "SETUP_EPOCHS")
        for entry in _unique_list("SETUP_EPOCHS", "7.14:100:100")
    ]
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
    epochs_grid: tuple[tuple[float, float, float, float], ...] | None = None,
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
            for epochs, cross_epochs, category_epochs, image_epochs in epochs_grid:
                for label in epsilons:
                    setup_id = compose_setup_id(
                        epochs, cross_epochs, category_epochs, image_epochs,
                        label, loss_formulation, prompt_mode,
                        split_protocol=split_protocol,
                        full_data_cross=full_data_cross,
                    )
                    setups[setup_id] = Setup(
                        epochs=epochs,
                        cross_epochs=cross_epochs,
                        category_epochs=category_epochs,
                        image_epochs=image_epochs,
                        epsilon=parse_epsilon(label),
                        epsilon_label=label,
                        loss_formulation=loss_formulation,
                        prompt_mode=prompt_mode,
                    )
    return setups


SETUPS = build_setups()
