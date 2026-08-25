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
