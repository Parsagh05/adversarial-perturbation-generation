"""Canonical standalone generation setups.

A setup ID is a pure function of the settings that change the work, so a
run can never be filed under a name that describes different parameters.
Overriding steps to 1200 produces ``steps1200_...`` on its own.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Setup:
    steps: int
    epsilon: float
    epsilon_label: str
    loss_formulation: str
    prompt_mode: str


SETUPS = {
    "steps500_eps2": Setup(500, 2 / 255, "2/255", "ce_focal_dice", "frozen_winclip"),
    "steps500_eps4": Setup(500, 4 / 255, "4/255", "ce_focal_dice", "frozen_winclip"),
    "steps800_eps2": Setup(800, 2 / 255, "2/255", "ce_focal_dice", "frozen_winclip"),
    "steps800_eps4": Setup(800, 4 / 255, "4/255", "ce_focal_dice", "frozen_winclip"),
    "steps500_eps2_margin_topk": Setup(500, 2 / 255, "2/255", "margin_topk", "frozen_winclip"),
    "steps500_eps4_margin_topk": Setup(500, 4 / 255, "4/255", "margin_topk", "frozen_winclip"),
    "steps800_eps2_margin_topk": Setup(800, 2 / 255, "2/255", "margin_topk", "frozen_winclip"),
    "steps800_eps4_margin_topk": Setup(800, 4 / 255, "4/255", "margin_topk", "frozen_winclip"),
    "steps500_eps2_learnable_prompt": Setup(500, 2 / 255, "2/255", "ce_focal_dice", "learnable_object_agnostic"),
    "steps500_eps4_learnable_prompt": Setup(500, 4 / 255, "4/255", "ce_focal_dice", "learnable_object_agnostic"),
    "steps800_eps2_learnable_prompt": Setup(800, 2 / 255, "2/255", "ce_focal_dice", "learnable_object_agnostic"),
    "steps800_eps4_learnable_prompt": Setup(800, 4 / 255, "4/255", "ce_focal_dice", "learnable_object_agnostic"),
    "steps500_eps2_margin_topk_learnable_prompt": Setup(500, 2 / 255, "2/255", "margin_topk", "learnable_object_agnostic"),
    "steps500_eps4_margin_topk_learnable_prompt": Setup(500, 4 / 255, "4/255", "margin_topk", "learnable_object_agnostic"),
    "steps800_eps2_margin_topk_learnable_prompt": Setup(800, 2 / 255, "2/255", "margin_topk", "learnable_object_agnostic"),
    "steps800_eps4_margin_topk_learnable_prompt": Setup(800, 4 / 255, "4/255", "margin_topk", "learnable_object_agnostic"),
}


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


def compose_setup_id(
    steps: int,
    epsilon_label: str,
    loss_formulation: str,
    prompt_mode: str,
    attack_train_fraction: float = 1.0,
) -> str:
    """Build the canonical ID for one effective configuration.

    Every component that changes the produced perturbations appears in the
    name. ``_learnable_prompt`` stays last so ``${id%_learnable_prompt}`` keeps
    recovering the frozen base ID in the launcher.
    """

    parts = [f"steps{int(steps)}", _epsilon_tag(epsilon_label)]
    if loss_formulation == "margin_topk":
        parts.append("margin_topk")
    fraction = _fraction_tag(attack_train_fraction)
    if fraction:
        parts.append(fraction)
    if prompt_mode == "learnable_object_agnostic":
        parts.append("learnable_prompt")
    return "_".join(parts)


def effective_setup_id(
    setup: Setup, steps: int | None = None, attack_train_fraction: float = 1.0
) -> str:
    """Canonical ID for a catalog entry after any step/fraction override."""

    return compose_setup_id(
        setup.steps if steps is None else steps,
        setup.epsilon_label,
        setup.loss_formulation,
        setup.prompt_mode,
        attack_train_fraction,
    )
