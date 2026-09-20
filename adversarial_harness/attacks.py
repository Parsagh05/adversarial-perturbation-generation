"""Targeted PGD attacks for instance, category, and dataset scopes."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import AttackConfig, VALID_LOSS_MODES
from .prompts import ensemble_class_logits


@dataclass
class UniversalAttackResult:
    """Universal perturbation plus crash-persistable optimization diagnostics."""

    delta: torch.Tensor
    history: List[Dict[str, float]]
    initial_losses: Dict[str, float]
    final_losses: Dict[str, float]
    diagnostic_sample_ids: List[str]
    selected_step: int
    selected_diagnostic_loss: float
    # step -> the delta this run would have returned had it stopped there.
    snapshots: Dict[int, torch.Tensor] = field(default_factory=dict)


def scatter_rows(
    destination: torch.Tensor, index: torch.Tensor, source: torch.Tensor
) -> None:
    """Scatter ``source`` into ``destination`` at ``index``, matching dtype.

    The accumulation buffers are allocated from the float32 visual features,
    while under ``torch.autocast`` the logits come back in the autocast dtype.
    ``index_copy_`` rejects a dtype mismatch instead of promoting the way the
    surrounding additions do, so the source is cast to the destination.

    Casting the source rather than allocating the buffer in the autocast dtype
    keeps the reduction in float32, which is the usual AMP convention and
    leaves the numerics unchanged when autocast is off, where the cast is a
    no-op.
    """

    destination.index_copy_(0, index, source.to(destination.dtype))


def direction_labels(direction: str) -> Tuple[int, int]:
    """Return ``(source_label, target_label)`` for a threat direction."""

    if direction == "normal_to_abnormal":
        return 0, 1
    if direction == "abnormal_to_normal":
        return 1, 0
    raise ValueError(f"Unknown direction: {direction}")


class TargetedPGD:
    """PGD optimizer that never queries or differentiates through the target."""

    def __init__(self, surrogate, config: AttackConfig):
        self.surrogate = surrogate
        self.config = config
        self.device = surrogate.device
        self.rng = np.random.default_rng(config.seed)

    def _group_losses(
        self,
        global_features: torch.Tensor,
        patch_features: Sequence[torch.Tensor],
        categories: Sequence[str],
        target_label: int,
        mode: str,
        spatial_masks: Optional[torch.Tensor] = None,
        hinge_floors: Optional[Dict[str, torch.Tensor]] = None,
        return_per_sample: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if mode not in VALID_LOSS_MODES:
            raise ValueError(f"Unknown loss mode: {mode}")
        # Images are grouped by category because each category has its own text
        # prompt bank and cannot share one logit computation. The grouping is a
        # mechanical necessity, not a weighting: every group accumulates the
        # SUM of its per-image losses and the total is divided by the batch
        # size, so each image counts exactly once no matter how many of its
        # category were drawn. Averaging the per-category means instead made an
        # image's weight depend on its category's share of the batch, which is
        # not the unbiased minibatch estimate a stochastic universal attack
        # assumes (Shafahi et al. 1811.11304 Alg. 2; CD-UAP 2010.03300 Alg. 1).
        batch_size = len(categories)
        total_global = global_features.new_zeros(())
        total_local = global_features.new_zeros(())
        total_local_focal = global_features.new_zeros(())
        total_local_dice = global_features.new_zeros(())
        total_global_margin = global_features.new_zeros(())
        total_local_topk = global_features.new_zeros(())
        saturated_global = global_features.new_zeros(())
        saturated_local = global_features.new_zeros(())
        per_sample_global = global_features.new_zeros(batch_size)
        per_sample_local = global_features.new_zeros(batch_size)
        direction_sign = -1.0 if target_label == 1 else 1.0

        for category in sorted(set(categories)):
            indices = [
                index for index, value in enumerate(categories) if value == category
            ]
            index_tensor = torch.as_tensor(
                indices, device=self.device, dtype=torch.long
            )
            bank = self.surrogate.prompts[category]
            target = torch.full(
                (len(indices),), target_label, device=self.device, dtype=torch.long
            )

            if mode in {"global", "combined"}:
                global_logits = ensemble_class_logits(
                    global_features.index_select(0, index_tensor),
                    bank,
                    self.config.temperature,
                )
                if self.config.loss_formulation == "margin_topk":
                    global_margin = global_logits[:, 1] - global_logits[:, 0]
                    scatter_rows(per_sample_global, index_tensor, global_margin)
                    signed = direction_sign * global_margin
                    if hinge_floors is not None:
                        floor = hinge_floors["global"].index_select(0, index_tensor)
                        saturated_global = saturated_global + (signed < floor).sum()
                        signed = torch.maximum(signed, floor)
                    total_global = total_global + signed.sum()
                    total_global_margin = total_global_margin + global_margin.sum()
                else:
                    total_global = total_global + F.cross_entropy(
                        global_logits, target, reduction="sum"
                    )

            if mode in {"local", "combined"}:
                layer_losses = []
                layer_focal_losses = []
                layer_dice_losses = []
                layer_anomaly_maps = []
                for patch in patch_features:
                    selected = patch.index_select(0, index_tensor)
                    # The first token is CLS; dense loss is defined only on patches.
                    if selected.shape[1] > 1:
                        selected = selected[:, 1:, :]
                    local_logits = ensemble_class_logits(
                        selected, bank, self.config.temperature
                    )
                    token_count = selected.shape[1]
                    if self.config.loss_formulation == "margin_topk":
                        layer_anomaly_maps.append(
                            local_logits[..., 1] - local_logits[..., 0]
                        )
                        continue
                    local_target = target[:, None].expand(-1, token_count)
                    token_ce = F.cross_entropy(
                        local_logits.reshape(-1, 2),
                        local_target.reshape(-1),
                        reduction="none",
                    ).reshape(len(indices), token_count)
                    target_probability = local_logits.softmax(dim=-1).gather(
                        -1, local_target[..., None]
                    )[..., 0]
                    token_focal = (
                        (1.0 - target_probability).clamp_min(0.0)
                        ** self.config.local_focal_gamma
                    ) * token_ce
                    token_weights = torch.ones_like(token_focal)
                    if self.config.mask_local_loss and spatial_masks is not None:
                        category_masks = spatial_masks.index_select(0, index_tensor)
                        side = int(token_count**0.5)
                        if side * side != token_count:
                            raise ValueError(
                                "Mask-aware local loss requires a square patch grid, "
                                f"got {token_count} tokens"
                            )
                        pooled_masks = F.adaptive_max_pool2d(
                            category_masks[:, None].float(), (side, side)
                        )[:, 0].reshape(len(indices), token_count)
                        has_defect = pooled_masks.sum(dim=1, keepdim=True) > 0
                        if (
                            target_label == 1
                            and self.config.normal_local_target == "fixed_region"
                        ):
                            region_side = max(
                                1,
                                int(round(side * self.config.normal_target_region_fraction)),
                            )
                            center_x = int(
                                round(self.config.normal_target_center_x * (side - 1))
                            )
                            center_y = int(
                                round(self.config.normal_target_center_y * (side - 1))
                            )
                            left = min(
                                max(center_x - region_side // 2, 0),
                                side - region_side,
                            )
                            top = min(
                                max(center_y - region_side // 2, 0),
                                side - region_side,
                            )
                            normal_region = torch.zeros(
                                (side, side),
                                device=pooled_masks.device,
                                dtype=pooled_masks.dtype,
                            )
                            normal_region[
                                top : top + region_side, left : left + region_side
                            ] = 1.0
                            normal_region = normal_region.reshape(1, token_count).expand(
                                len(indices), -1
                            )
                            effective_masks = torch.where(
                                has_defect, pooled_masks, normal_region
                            )
                        else:
                            effective_masks = torch.where(
                                has_defect, pooled_masks, torch.ones_like(pooled_masks)
                            )
                        masked_weights = (
                            self.config.local_background_weight
                            + (1.0 - self.config.local_background_weight)
                            * effective_masks
                        )
                        token_weights = masked_weights
                    weight_sum = token_weights.sum(dim=1).clamp_min(1e-12)
                    per_image_focal = (
                        token_focal * token_weights
                    ).sum(dim=1) / weight_sum

                    # Soft Dice is computed for the requested target class on
                    # the same spatial support as the focal term.  This works
                    # for both directions: anomaly probability is encouraged
                    # on normal-source images, while normal probability is
                    # encouraged inside the defect region on anomalous images.
                    weighted_probability = target_probability * token_weights
                    dice_numerator = (
                        2.0 * weighted_probability.sum(dim=1)
                        + self.config.local_dice_smooth
                    )
                    dice_denominator = (
                        weighted_probability.sum(dim=1)
                        + weight_sum
                        + self.config.local_dice_smooth
                    )
                    per_image_dice = 1.0 - dice_numerator / dice_denominator
                    focal_loss = per_image_focal.sum()
                    dice_loss = per_image_dice.sum()
                    layer_focal_losses.append(focal_loss)
                    layer_dice_losses.append(dice_loss)
                    layer_losses.append(
                        self.config.local_focal_weight * focal_loss
                        + self.config.local_dice_weight * dice_loss
                    )
                if self.config.loss_formulation == "margin_topk":
                    if not layer_anomaly_maps:
                        raise RuntimeError("The surrogate returned no patch features")
                    anomaly_map = torch.stack(layer_anomaly_maps).mean(dim=0)
                    token_count = anomaly_map.shape[1]
                    topk_count = max(
                        1,
                        min(
                            token_count,
                            int(math.ceil(
                                token_count * self.config.margin_topk_fraction
                            )),
                        ),
                    )
                    per_image_topk = anomaly_map.topk(
                        topk_count, dim=1, largest=True, sorted=False
                    ).values.mean(dim=1)
                    scatter_rows(per_sample_local, index_tensor, per_image_topk)
                    signed = direction_sign * per_image_topk
                    if hinge_floors is not None:
                        floor = hinge_floors["local"].index_select(0, index_tensor)
                        saturated_local = saturated_local + (signed < floor).sum()
                        signed = torch.maximum(signed, floor)
                    total_local = total_local + signed.sum()
                    total_local_topk = total_local_topk + per_image_topk.sum()
                else:
                    if not layer_losses:
                        raise RuntimeError("The surrogate returned no patch features")
                    total_local = total_local + torch.stack(layer_losses).mean()
                    total_local_focal = (
                        total_local_focal + torch.stack(layer_focal_losses).mean()
                    )
                    total_local_dice = (
                        total_local_dice + torch.stack(layer_dice_losses).mean()
                    )

        result: Dict[str, torch.Tensor] = {}
        if mode in {"global", "combined"}:
            result["global"] = total_global / batch_size
            if self.config.loss_formulation == "margin_topk":
                result["global_margin"] = total_global_margin / batch_size
        if mode in {"local", "combined"}:
            result["local"] = total_local / batch_size
            if self.config.loss_formulation == "margin_topk":
                result["local_topk"] = total_local_topk / batch_size
            else:
                # These components are diagnostics; ``local`` is the weighted
                # segmentation-aware objective used for optimization.
                result["local_focal"] = total_local_focal / batch_size
                result["local_dice"] = total_local_dice / batch_size
        if mode == "global":
            result["total"] = result["global"]
        elif mode == "local":
            result["total"] = result["local"]
        else:
            result["total"] = (
                self.config.global_weight * result["global"]
                + self.config.local_weight * result["local"]
            )
        if hinge_floors is not None:
            # A saturated image contributes a constant, so it adds no gradient.
            # If every image saturates the gradient is exactly zero and
            # sign(0) = 0 freezes delta, so the fractions are recorded.
            if mode in {"global", "combined"}:
                result["global_saturated_fraction"] = saturated_global / batch_size
            if mode in {"local", "combined"}:
                result["local_saturated_fraction"] = saturated_local / batch_size
        if return_per_sample:
            result["per_sample_global"] = per_sample_global
            result["per_sample_local"] = per_sample_local
        return result

    def step_size_at(self, step: int, total_steps: int) -> float:
        """Return the configured PGD step size for a zero-based iteration."""

        if self.config.step_size_schedule == "constant" or total_steps <= 1:
            return self.config.step_size
        progress = step / max(total_steps - 1, 1)
        if self.config.step_size_schedule == "linear":
            # Straight decay to step_size_min_ratio; spends less of the budget
            # at the large initial step than cosine does.
            decay = 1.0 - progress
        else:
            decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        ratio = self.config.step_size_min_ratio + (
            1.0 - self.config.step_size_min_ratio
        ) * decay
        return float(self.config.step_size * ratio)

    def hinge_floors(
        self,
        samples: Sequence[object],
        image_loader: Callable[[object], torch.Tensor],
        target_label: int,
        mode: str,
        mask_loader: Optional[Callable[[object], torch.Tensor]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Per-image floors that saturate the margin a fixed distance past clean.

        The floor for image i is ``sign * clean_margin_i - displacement``, so an
        image stops contributing gradient once it has moved ``displacement``
        toward the attacked class *from its own unperturbed margin*. Measuring
        the displacement rather than the absolute margin is what makes one
        setting usable across conditions whose margins differ by an order of
        magnitude, and across terms that already start on the attacked side.

        Returns ``None`` when the hinge is disabled, costing nothing.
        """

        displacement = self.config.margin_hinge_displacement
        if displacement is None or self.config.loss_formulation != "margin_topk":
            return None
        direction_sign = -1.0 if target_label == 1 else 1.0
        batch_size = max(1, min(self.config.universal_batch_size, len(samples)))
        clean_global = []
        clean_local = []
        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                images = torch.stack([image_loader(s) for s in batch]).to(self.device)
                masks = (
                    torch.stack([mask_loader(s) for s in batch]).to(self.device)
                    if mask_loader is not None and mode in {"local", "combined"}
                    else None
                )
                components = self.objective_components(
                    images,
                    [str(getattr(s, "category")) for s in batch],
                    target_label,
                    mode,
                    spatial_masks=masks,
                    return_per_sample=True,
                )
                clean_global.append(components["per_sample_global"].detach())
                clean_local.append(components["per_sample_local"].detach())
        return {
            "global": direction_sign * torch.cat(clean_global) - displacement,
            "local": direction_sign * torch.cat(clean_local) - displacement,
        }

    def update_direction(
        self, gradient: torch.Tensor, momentum: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Accumulate the gradient and report how far momentum turned it.

        ``m = decay * m + g``; the step then follows ``sign(m)``, so the
        epsilon projection and the budget are untouched. A decay of 0 returns
        the gradient unchanged and is exactly plain sign-PGD.

        The cosine between the accumulated direction and the current gradient
        is what says whether momentum is doing anything; it is 1.0 when off.
        """

        if self.config.momentum_decay <= 0.0:
            return gradient, float("nan")
        accumulated = self.config.momentum_decay * momentum + gradient
        scale = accumulated.norm() * gradient.norm()
        cosine = (
            float((accumulated * gradient).sum() / scale)
            if float(scale) > 0.0
            else float("nan")
        )
        return accumulated, cosine

    def objective_components(
        self,
        images_01: torch.Tensor,
        categories: Sequence[str],
        target_label: int,
        mode: str,
        spatial_masks: Optional[torch.Tensor] = None,
        hinge_floors: Optional[Dict[str, torch.Tensor]] = None,
        return_per_sample: bool = False,
    ) -> Dict[str, torch.Tensor]:
        global_features, patch_features = self.surrogate.encode_visual(
            images_01, include_patches=mode in {"local", "combined"}
        )
        return self._group_losses(
            global_features,
            patch_features,
            categories,
            target_label,
            mode,
            spatial_masks=spatial_masks,
            hinge_floors=hinge_floors,
            return_per_sample=return_per_sample,
        )

    def objective(
        self,
        images_01: torch.Tensor,
        categories: Sequence[str],
        target_label: int,
        mode: str,
        spatial_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.objective_components(
            images_01,
            categories,
            target_label,
            mode,
            spatial_masks=spatial_masks,
        )["total"]

    def surrogate_scores(
        self,
        images_01: torch.Tensor,
        categories: Sequence[str],
        mode: str,
    ) -> Dict[str, np.ndarray]:
        """Return global/local/mode anomaly scores from the public surrogate."""

        include_local = mode in {"local", "combined"}
        with torch.no_grad():
            global_features, patch_features = self.surrogate.encode_visual(
                images_01.to(self.device), include_patches=include_local
            )
            batch_size = global_features.shape[0]
            global_logits = global_features.new_zeros((batch_size, 2))
            local_logits: Optional[torch.Tensor] = (
                global_features.new_zeros((batch_size, 2)) if include_local else None
            )
            for category in sorted(set(categories)):
                indices = [
                    index for index, value in enumerate(categories) if value == category
                ]
                index_tensor = torch.as_tensor(
                    indices, device=self.device, dtype=torch.long
                )
                bank = self.surrogate.prompts[category]
                category_global = ensemble_class_logits(
                    global_features.index_select(0, index_tensor),
                    bank,
                    self.config.temperature,
                )
                scatter_rows(global_logits, index_tensor, category_global)
                if include_local and local_logits is not None:
                    layer_logits = []
                    for patch in patch_features:
                        selected = patch.index_select(0, index_tensor)
                        if selected.shape[1] > 1:
                            selected = selected[:, 1:, :]
                        patch_logits = ensemble_class_logits(
                            selected, bank, self.config.temperature
                        )
                        layer_logits.append(patch_logits.mean(dim=1))
                    category_local = torch.stack(layer_logits).mean(dim=0)
                    scatter_rows(local_logits, index_tensor, category_local)

            global_scores = global_logits.softmax(dim=-1)[:, 1]
            if mode == "global":
                mode_logits = global_logits
            elif mode == "local":
                if local_logits is None:
                    raise RuntimeError("Local surrogate logits were not computed")
                mode_logits = local_logits
            else:
                if local_logits is None:
                    raise RuntimeError("Combined surrogate logits require local logits")
                mode_logits = (
                    self.config.global_weight * global_logits
                    + self.config.local_weight * local_logits
                )
            mode_scores = mode_logits.softmax(dim=-1)[:, 1]
            if local_logits is None:
                local_scores = torch.full_like(global_scores, float("nan"))
            else:
                local_scores = local_logits.softmax(dim=-1)[:, 1]
        return {
            "global_score": global_scores.detach().cpu().numpy().astype(np.float32),
            "local_score": local_scores.detach().cpu().numpy().astype(np.float32),
            "mode_score": mode_scores.detach().cpu().numpy().astype(np.float32),
        }

    def _initial_delta(self, shape: Sequence[int], clean: torch.Tensor) -> torch.Tensor:
        if self.config.random_start:
            delta = torch.empty(shape, device=self.device, dtype=clean.dtype).uniform_(
                -self.config.epsilon, self.config.epsilon
            )
            delta = (clean + delta).clamp(0.0, 1.0) - clean
        else:
            delta = torch.zeros(shape, device=self.device, dtype=clean.dtype)
        return delta.detach()

    def perturb_batch(
        self,
        clean_images: torch.Tensor,
        categories: Sequence[str],
        target_label: int,
        mode: str,
        spatial_masks: Optional[torch.Tensor] = None,
        snapshot_steps: Sequence[int] = (),
        snapshot_sink: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimize independent per-image perturbations for a batch.

        ``snapshot_sink`` receives, per listed step, the deltas this call would
        have returned had ``steps`` been set to that value. The return shape is
        unchanged so existing callers are untouched.
        """

        wanted_snapshots = {
            int(step) for step in snapshot_steps if 0 < int(step) < self.config.steps
        }

        clean = clean_images.to(self.device)
        masks = spatial_masks.to(self.device) if spatial_masks is not None else None
        delta = self._initial_delta(clean.shape, clean)
        # The clean image is the baseline checkpoint. A failed random start can
        # therefore never be published as a perturbation that worsens its own
        # targeted surrogate objective.
        with torch.no_grad():
            best_loss = float(
                self.objective(
                    clean,
                    categories,
                    target_label,
                    mode,
                    spatial_masks=masks,
                )
            )
        best_delta = torch.zeros_like(delta)
        for step in range(self.config.steps):
            delta.requires_grad_(True)
            adversarial = (clean + delta).clamp(0.0, 1.0)
            loss = self.objective(
                adversarial,
                categories,
                target_label,
                mode,
                spatial_masks=masks,
            )
            current_loss = float(loss.detach())
            if current_loss < best_loss:
                best_loss = current_loss
                best_delta = delta.detach().clone()
            if step in wanted_snapshots and snapshot_sink is not None:
                snapshot_sink[step] = (
                    delta.detach().clone()
                    if self.config.checkpoint_selection == "final"
                    else best_delta.detach().clone()
                )
            gradient = torch.autograd.grad(loss, delta, only_inputs=True)[0]
            # Targeted PGD minimizes the requested global/local objective.
            step_size = self.step_size_at(step, self.config.steps)
            delta = delta.detach() - step_size * gradient.sign()
            delta = delta.clamp(-self.config.epsilon, self.config.epsilon)
            delta = ((clean + delta).clamp(0.0, 1.0) - clean).detach()
        with torch.no_grad():
            final_loss = float(
                self.objective(
                    (clean + delta).clamp(0.0, 1.0),
                    categories,
                    target_label,
                    mode,
                    spatial_masks=masks,
                )
            )
        if final_loss < best_loss:
            best_loss = final_loss
            best_delta = delta.detach().clone()
        selected = (
            delta.detach()
            if self.config.checkpoint_selection == "final"
            else best_delta
        )
        return (
            (clean + selected).clamp(0.0, 1.0).detach(),
            selected.detach(),
        )

    def optimize_universal(
        self,
        samples: Sequence[object],
        image_loader: Callable[[object], torch.Tensor],
        target_label: int,
        mode: str,
        mask_loader: Optional[Callable[[object], torch.Tensor]] = None,
        diagnostic_samples: Optional[Sequence[object]] = None,
        progress: Callable[[int, int, Dict[str, float]], None] | None = None,
        snapshot_steps: Sequence[int] = (),
    ) -> UniversalAttackResult:
        """Optimize one shared perturbation across the supplied samples.

        ``snapshot_steps`` captures, at each listed step, the delta this run
        would have returned had it been configured to stop there. That is only
        equivalent to a standalone shorter run while the step size is
        independent of the total budget, which ``constant`` guarantees and the
        decaying schedules do not.
        """

        wanted_snapshots = {
            int(step) for step in snapshot_steps
            if 0 < int(step) < self.config.universal_steps
        }
        snapshots: Dict[int, torch.Tensor] = {}

        if not samples:
            raise ValueError("Universal optimization requires at least one sample")
        size = self.config.image_size
        reference = image_loader(samples[0]).unsqueeze(0).to(self.device)
        delta = self._initial_delta((1, 3, size, size), reference)
        # Starts at zero and evolves deterministically from the step index, so
        # the first N steps of a long run stay identical to an N-step run.
        momentum = torch.zeros_like(delta)
        # One pass at delta = 0; None when the hinge is off, so it costs nothing.
        hinge_floors = self.hinge_floors(
            samples, image_loader, target_label, mode, mask_loader=mask_loader
        )
        order = np.arange(len(samples))
        cursor = len(order)
        diagnostic_samples = list(diagnostic_samples or samples[:1])
        diagnostic_ids = [
            str(
                getattr(
                    sample,
                    "protocol_id",
                    getattr(sample, "sample_id", index),
                )
            )
            for index, sample in enumerate(diagnostic_samples)
        ]
        # The unperturbed image is the baseline checkpoint, matching
        # ``perturb_batch``. Measuring the initial losses at delta=0 rather than
        # at the random start keeps loss reductions comparable with the
        # per-image scope and lets a run that never beats "no attack" say so.
        best_delta = torch.zeros_like(delta)
        initial_losses = self._diagnostic_losses(
            diagnostic_samples,
            image_loader,
            best_delta,
            target_label,
            mode,
            mask_loader=mask_loader,
        )
        history: List[Dict[str, float]] = []
        best_diagnostic_loss = initial_losses["total"]
        selected_step = 0

        for step in range(self.config.universal_steps):
            batch_size = min(self.config.universal_batch_size, len(samples))
            if cursor >= len(order):
                self.rng.shuffle(order)
                cursor = 0
            # Consume the short final batch instead of discarding it when the
            # dataset size is not divisible by universal_batch_size. The next
            # update starts a newly shuffled pass over all samples.
            stop = min(cursor + batch_size, len(order))
            indices = order[cursor:stop]
            cursor = stop
            batch_samples = [samples[int(index)] for index in indices]
            clean = torch.stack([image_loader(sample) for sample in batch_samples]).to(
                self.device
            )
            categories = [str(getattr(sample, "category")) for sample in batch_samples]
            spatial_masks = (
                torch.stack([mask_loader(sample) for sample in batch_samples]).to(
                    self.device
                )
                if mask_loader is not None and mode in {"local", "combined"}
                else None
            )

            batch_floors = None
            if hinge_floors is not None:
                selector = torch.as_tensor(
                    [int(index) for index in indices],
                    device=self.device,
                    dtype=torch.long,
                )
                batch_floors = {
                    key: value.index_select(0, selector)
                    for key, value in hinge_floors.items()
                }

            delta.requires_grad_(True)
            adversarial = (clean + delta).clamp(0.0, 1.0)
            components = self.objective_components(
                adversarial,
                categories,
                target_label,
                mode,
                spatial_masks=spatial_masks,
                hinge_floors=batch_floors,
            )
            global_gradient_norm = float("nan")
            local_gradient_norm = float("nan")
            if mode == "combined":
                global_gradient = torch.autograd.grad(
                    components["global"], delta, retain_graph=True, only_inputs=True
                )[0]
                local_gradient = torch.autograd.grad(
                    components["local"], delta, only_inputs=True
                )[0]
                global_gradient_norm = float(global_gradient.norm().detach())
                local_gradient_norm = float(local_gradient.norm().detach())
                gradient = (
                    self.config.global_weight * global_gradient
                    + self.config.local_weight * local_gradient
                )
            else:
                gradient = torch.autograd.grad(
                    components["total"], delta, only_inputs=True
                )[0]
                if mode == "global":
                    global_gradient_norm = float(gradient.norm().detach())
                else:
                    local_gradient_norm = float(gradient.norm().detach())
            pre_update_loss = float(components["total"].detach())
            step_size = self.step_size_at(step, self.config.universal_steps)
            direction, momentum_cosine = self.update_direction(gradient, momentum)
            momentum = direction if self.config.momentum_decay > 0.0 else momentum
            delta = delta.detach() - step_size * direction.sign()
            delta = delta.clamp(-self.config.epsilon, self.config.epsilon)
            delta = delta.detach()
            with torch.no_grad():
                updated_components = self.objective_components(
                    (clean + delta).clamp(0.0, 1.0),
                    categories,
                    target_label,
                    mode,
                    spatial_masks=spatial_masks,
                    hinge_floors=batch_floors,
                )
            diagnostic_losses: Dict[str, float] = {}
            if (
                step == 0
                or (step + 1) % self.config.diagnostic_interval == 0
                or step + 1 == self.config.universal_steps
                or (step + 1) in wanted_snapshots
            ):
                diagnostic_losses = self._diagnostic_losses(
                    diagnostic_samples,
                    image_loader,
                    delta,
                    target_label,
                    mode,
                    mask_loader=mask_loader,
                )
                if diagnostic_losses["total"] < best_diagnostic_loss:
                    best_diagnostic_loss = diagnostic_losses["total"]
                    best_delta = delta.detach().clone()
                    selected_step = step + 1
            step_record = {
                    "step": float(step + 1),
                    "pre_update_total_loss": pre_update_loss,
                    "total_loss": float(updated_components["total"].detach()),
                    "global_loss": (
                        float(updated_components["global"].detach())
                        if "global" in updated_components
                        else float("nan")
                    ),
                    "local_loss": (
                        float(updated_components["local"].detach())
                        if "local" in updated_components
                        else float("nan")
                    ),
                    "global_gradient_l2": global_gradient_norm,
                    "local_gradient_l2": local_gradient_norm,
                    "combined_gradient_l2": float(gradient.norm().detach()),
                    "combined_gradient_linf": float(gradient.abs().max().detach()),
                    "step_size": step_size,
                    "momentum_gradient_cosine": momentum_cosine,
                    "global_margin_saturated_fraction": float(
                        updated_components.get(
                            "global_saturated_fraction", torch.tensor(float("nan"))
                        )
                    ),
                    "local_margin_saturated_fraction": float(
                        updated_components.get(
                            "local_saturated_fraction", torch.tensor(float("nan"))
                        )
                    ),
                    "delta_saturation_fraction": float(
                        (delta.abs() >= self.config.epsilon - 1e-7)
                        .float()
                        .mean()
                        .detach()
                    ),
                    "diagnostic_total_loss": diagnostic_losses.get(
                        "total", float("nan")
                    ),
                    "diagnostic_local_focal": diagnostic_losses.get(
                        "local_focal", float("nan")
                    ),
                    "diagnostic_local_dice": diagnostic_losses.get(
                        "local_dice", float("nan")
                    ),
                    "diagnostic_global_margin": diagnostic_losses.get(
                        "global_margin", float("nan")
                    ),
                    "diagnostic_local_topk": diagnostic_losses.get(
                        "local_topk", float("nan")
                    ),
                }
            if (step + 1) in wanted_snapshots:
                # Exactly what the return below would pick at this step.
                snapshots[step + 1] = (
                    delta.detach().clone()
                    if self.config.checkpoint_selection == "final"
                    else best_delta.detach().clone()
                )
            history.append(step_record)
            if progress is not None:
                progress(step + 1, self.config.universal_steps, step_record)
        # The trajectory above is identical either way; only the retained
        # point differs. "final" returns the last step, as the universal-attack
        # papers do, and reports the diagnostic loss of that step rather than
        # the argmin over the checkpoints.
        if self.config.checkpoint_selection == "final":
            selected = delta.detach()
            selected_step = self.config.universal_steps
        else:
            selected = best_delta.detach()
        final_losses = self._diagnostic_losses(
            diagnostic_samples,
            image_loader,
            selected,
            target_label,
            mode,
            mask_loader=mask_loader,
        )
        return UniversalAttackResult(
            delta=selected,
            history=history,
            initial_losses=initial_losses,
            final_losses=final_losses,
            diagnostic_sample_ids=diagnostic_ids,
            selected_step=selected_step,
            selected_diagnostic_loss=(
                final_losses["total"]
                if self.config.checkpoint_selection == "final"
                else best_diagnostic_loss
            ),
            snapshots=snapshots,
        )

    def _diagnostic_losses(
        self,
        samples: Sequence[object],
        image_loader: Callable[[object], torch.Tensor],
        delta: torch.Tensor,
        target_label: int,
        mode: str,
        mask_loader: Optional[Callable[[object], torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Average losses over the supplied fixed checkpoint-selection set."""

        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        batch_size = min(self.config.universal_batch_size, len(samples))
        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                batch_samples = samples[start : start + batch_size]
                clean = torch.stack(
                    [image_loader(sample) for sample in batch_samples]
                ).to(self.device)
                attacked = self.apply_universal(clean, delta)
                categories = [
                    str(getattr(sample, "category")) for sample in batch_samples
                ]
                spatial_masks = (
                    torch.stack([mask_loader(sample) for sample in batch_samples]).to(
                        self.device
                    )
                    if mask_loader is not None and mode in {"local", "combined"}
                    else None
                )
                components = self.objective_components(
                    attacked,
                    categories,
                    target_label,
                    mode,
                    spatial_masks=spatial_masks,
                )
                for key, value in components.items():
                    totals[key] = totals.get(key, 0.0) + (
                        float(value.detach()) * len(batch_samples)
                    )
                    counts[key] = counts.get(key, 0) + len(batch_samples)
        return {key: totals[key] / counts[key] for key in totals if counts[key] > 0}

    @staticmethod
    def apply_universal(
        clean_images: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        return (clean_images.to(delta.device) + delta).clamp(0.0, 1.0)
