"""Canonical standalone generation setups."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Setup:
    steps: int
    epsilon: float
    epsilon_label: str
    loss_formulation: str
    prompt_mode: str
    gradient_normalization: str = "none"
    # Gradient normalization only changes the combined objective: a single
    # component is unaffected by any positive rescaling once sign() is taken.
    loss_modes: tuple[str, ...] = ("global", "local", "combined")


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

    # Gradient-normalized counterparts: each component gradient is rescaled
    # to unit L2 norm before the 0.2/0.8 weights, so the configured balance
    # describes influence on the update instead of raw loss magnitudes.
    "steps500_eps2_gradnorm": Setup(500, 2 / 255, "2/255", "ce_focal_dice", "frozen_winclip", "l2", ("combined",)),
    "steps500_eps4_gradnorm": Setup(500, 4 / 255, "4/255", "ce_focal_dice", "frozen_winclip", "l2", ("combined",)),
    "steps800_eps2_gradnorm": Setup(800, 2 / 255, "2/255", "ce_focal_dice", "frozen_winclip", "l2", ("combined",)),
    "steps800_eps4_gradnorm": Setup(800, 4 / 255, "4/255", "ce_focal_dice", "frozen_winclip", "l2", ("combined",)),
    "steps500_eps2_margin_topk_gradnorm": Setup(500, 2 / 255, "2/255", "margin_topk", "frozen_winclip", "l2", ("combined",)),
    "steps500_eps4_margin_topk_gradnorm": Setup(500, 4 / 255, "4/255", "margin_topk", "frozen_winclip", "l2", ("combined",)),
    "steps800_eps2_margin_topk_gradnorm": Setup(800, 2 / 255, "2/255", "margin_topk", "frozen_winclip", "l2", ("combined",)),
    "steps800_eps4_margin_topk_gradnorm": Setup(800, 4 / 255, "4/255", "margin_topk", "frozen_winclip", "l2", ("combined",)),
    "steps500_eps2_gradnorm_learnable_prompt": Setup(500, 2 / 255, "2/255", "ce_focal_dice", "learnable_object_agnostic", "l2", ("combined",)),
    "steps500_eps4_gradnorm_learnable_prompt": Setup(500, 4 / 255, "4/255", "ce_focal_dice", "learnable_object_agnostic", "l2", ("combined",)),
    "steps800_eps2_gradnorm_learnable_prompt": Setup(800, 2 / 255, "2/255", "ce_focal_dice", "learnable_object_agnostic", "l2", ("combined",)),
    "steps800_eps4_gradnorm_learnable_prompt": Setup(800, 4 / 255, "4/255", "ce_focal_dice", "learnable_object_agnostic", "l2", ("combined",)),
    "steps500_eps2_margin_topk_gradnorm_learnable_prompt": Setup(500, 2 / 255, "2/255", "margin_topk", "learnable_object_agnostic", "l2", ("combined",)),
    "steps500_eps4_margin_topk_gradnorm_learnable_prompt": Setup(500, 4 / 255, "4/255", "margin_topk", "learnable_object_agnostic", "l2", ("combined",)),
    "steps800_eps2_margin_topk_gradnorm_learnable_prompt": Setup(800, 2 / 255, "2/255", "margin_topk", "learnable_object_agnostic", "l2", ("combined",)),
    "steps800_eps4_margin_topk_gradnorm_learnable_prompt": Setup(800, 4 / 255, "4/255", "margin_topk", "learnable_object_agnostic", "l2", ("combined",)),
}
