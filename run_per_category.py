#!/usr/bin/env python3
"""Generate category-level universal perturbations from attack_train only.

Every artifact is fitted on one dataset/category/source-label stratum and is
applied only to held-out evaluation images from the same dataset/category.
Nested attack-training fractions support data-efficiency experiments.
"""
from __future__ import annotations

import gc
import hashlib
import math
import os
import random
import shutil
import subprocess
import zipfile
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if not torch.cuda.is_available():
    raise RuntimeError("A CUDA-capable GPU is required")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

PROJECT_ROOT = Path(__file__).resolve().parent
WORKING = Path(os.environ["WORK_DIR"]).expanduser().resolve()
OUTPUT_BASE = Path(os.environ["OUTPUT_BASE"]).expanduser().resolve()
ANOMALYCLIP_ROOT = WORKING / "AnomalyCLIP"
MVTEC_ROOT = Path(os.environ["MVTEC_ROOT"]).expanduser().resolve()
VISA_ROOT = Path(os.environ["VISA_ROOT"]).expanduser().resolve()
ATTACK_TRAIN_CSV = Path(os.environ["ATTACK_TRAIN_CSV"]).expanduser().resolve()
EVALUATION_CSV = Path(os.environ["EVALUATION_CSV"]).expanduser().resolve()
SETUP_ID = os.environ["SETUP_ID"]

if not ANOMALYCLIP_ROOT.exists():
    raise FileNotFoundError(ANOMALYCLIP_ROOT)

from setup_catalog import (
    derive_steps,
    margin_hinge_setting,
    momentum_decay_setting,
    checkpoint_selection_setting,
)
from adversarial_harness.attacks import TargetedPGD, direction_labels
from adversarial_harness.config import AttackConfig, VALID_LOSS_FORMULATIONS
from adversarial_harness.dataset import MVTecSample, discover_anomaly_datasets, load_image_tensor, load_mask
from adversarial_harness.models import CLIPSurrogate
from adversarial_harness.prompts import (
    PROMPT_PROVENANCE_FIELDS,
    VALID_PROMPT_MODES,
    learnable_prompt_checkpoint,
)
from common import (
    COMPLETE_RETAINED_CSV,
    LABEL_BALANCE_POLICY,
    split_protocol,
    assert_partition_disjoint,
    bind_discovered_samples_from_partition_csvs,
    fraction_tag,
    generation_datasets,
    parse_fraction_list,
    parse_numeric,
    protocol_datasets,
    select_attack_train_fraction,
    sha256_file,
    split_sha256,
)


def bool_env(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def csv_tuple(name: str, default: str):
    return tuple(x.strip() for x in os.environ.get(name, default).split(",") if x.strip())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def condition_seed(base: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, (base, *parts))).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable-standalone-copy"


def sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def chunked(sequence, size):
    for start in range(0, len(sequence), size):
        yield sequence[start:start + size]


IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "518"))
EPSILON = parse_numeric(os.environ["EPSILON"])
UNIVERSAL_STEP_SIZE = parse_numeric(os.environ["PER_CATEGORY_STEP_SIZE"])
PER_CATEGORY_EPOCHS = float(os.environ["PER_CATEGORY_EPOCHS"])
EFFECTIVE_BATCH_SIZE = int(os.environ.get("PER_CATEGORY_EFFECTIVE_BATCH_SIZE", "8"))
MICRO_BATCH_SIZE = int(os.environ.get("PER_CATEGORY_MICRO_BATCH_SIZE", "2"))
LOCAL_FOCAL_WEIGHT = float(os.environ.get("LOCAL_FOCAL_WEIGHT", "0.5"))
LOCAL_DICE_WEIGHT = float(os.environ.get("LOCAL_DICE_WEIGHT", "0.5"))
LOCAL_FOCAL_GAMMA = float(os.environ.get("LOCAL_FOCAL_GAMMA", "2.0"))
LOCAL_DICE_SMOOTH = float(os.environ.get("LOCAL_DICE_SMOOTH", "1.0"))
LOCAL_BACKGROUND_WEIGHT = float(os.environ.get("LOCAL_BACKGROUND_WEIGHT", "0.1"))
NORMAL_LOCAL_TARGET = os.environ.get("NORMAL_LOCAL_TARGET", "fixed_region")
NORMAL_TARGET_REGION_FRACTION = float(os.environ.get("NORMAL_TARGET_REGION_FRACTION", "0.25"))
NORMAL_TARGET_CENTER_X = float(os.environ.get("NORMAL_TARGET_CENTER_X", "0.5"))
NORMAL_TARGET_CENTER_Y = float(os.environ.get("NORMAL_TARGET_CENTER_Y", "0.5"))
STEP_SIZE_SCHEDULE = os.environ.get("STEP_SIZE_SCHEDULE", "constant")
MARGIN_HINGE_DISPLACEMENT = margin_hinge_setting()
MOMENTUM_DECAY = momentum_decay_setting()
CHECKPOINT_SELECTION = checkpoint_selection_setting()
STEP_SIZE_MIN_RATIO = float(os.environ.get("STEP_SIZE_MIN_RATIO", "0.1"))
DIAGNOSTIC_INTERVAL = int(os.environ.get("DIAGNOSTIC_INTERVAL", "8"))
SEED = int(os.environ.get("ATTACK_SEED", "111"))
OVERWRITE_EXISTING = bool_env("OVERWRITE_EXISTING", False)
USE_AMP = bool_env("USE_AMP", True)
CACHE_INPUTS_IN_RAM = bool_env("CACHE_INPUTS_IN_RAM", True)
AUTO_REDUCE_MICRO_BATCH_ON_OOM = True
TRAIN_FRACTIONS = parse_fraction_list(
    os.environ.get("PER_CATEGORY_ATTACK_TRAIN_FRACTIONS", "1.0"),
    name="PER_CATEGORY_ATTACK_TRAIN_FRACTIONS",
)
DIRECTIONS = csv_tuple("DIRECTIONS", "normal_to_abnormal,abnormal_to_normal")
LOSS_MODES = csv_tuple("LOSS_MODES", "global,local,combined")
LOSS_FORMULATION = os.environ.get("LOSS_FORMULATION", "ce_focal_dice")
if LOSS_FORMULATION not in VALID_LOSS_FORMULATIONS:
    raise ValueError(f"Unknown LOSS_FORMULATION: {LOSS_FORMULATION}")
PROMPT_MODE = os.environ.get("PROMPT_MODE", "frozen_winclip")
if PROMPT_MODE not in VALID_PROMPT_MODES:
    raise ValueError(f"Unknown PROMPT_MODE: {PROMPT_MODE}")
MARGIN_TOPK_FRACTIONS = {
    "normal_to_abnormal": float(os.environ.get("MARGIN_TOPK_FRACTION_NORMAL_TO_ABNORMAL", "0.20")),
    "abnormal_to_normal": float(os.environ.get("MARGIN_TOPK_FRACTION_ABNORMAL_TO_NORMAL", "0.40")),
}
if any(not 0.0 < value <= 1.0 for value in MARGIN_TOPK_FRACTIONS.values()):
    raise ValueError("MARGIN_TOPK_FRACTION values must be in (0, 1]")
DATASETS = generation_datasets()
PROTOCOL_DATASETS = protocol_datasets()
DISCOVERY_MODE = PROTOCOL_DATASETS[0] if len(PROTOCOL_DATASETS) == 1 else "both"
for dataset_name, dataset_root in (("mvtec", MVTEC_ROOT), ("visa", VISA_ROOT)):
    if dataset_name in PROTOCOL_DATASETS and not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

if EFFECTIVE_BATCH_SIZE < 1 or MICRO_BATCH_SIZE < 1:
    raise ValueError("Batch sizes must be positive")
if MICRO_BATCH_SIZE > EFFECTIVE_BATCH_SIZE:
    raise ValueError("PER_CATEGORY_MICRO_BATCH_SIZE cannot exceed effective batch size")

# Sign-PGD consumes only gradient.sign(), so an fp16 gradient that underflows
# silently zeroes part of the update instead of shrinking it. Autocast is
# therefore restricted to bf16 hardware; elsewhere the attack stays fp32.
AMP_DTYPE_NAME = "bfloat16" if torch.cuda.is_bf16_supported() else "disabled_no_bf16"
AMP_ENABLED = USE_AMP and AMP_DTYPE_NAME == "bfloat16"


def autocast_context():
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
        if AMP_ENABLED else nullcontext()
    )


OUTPUT_ROOT = OUTPUT_BASE / "canonical_clip_per_category"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CLIP_CACHE = WORKING / "clip_cache"
CLIP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["ANOMALYCLIP_CLIP_CACHE"] = str(CLIP_CACHE)

print("GPU:", torch.cuda.get_device_name(0))
print("Protocol SHA256:", split_sha256())
print("Attack-train fractions:", TRAIN_FRACTIONS)
print("Epoch budget / step size:", PER_CATEGORY_EPOCHS, UNIVERSAL_STEP_SIZE)
print("Effective batch / micro-batch:", EFFECTIVE_BATCH_SIZE, MICRO_BATCH_SIZE)
print("Autocast:", AMP_DTYPE_NAME if AMP_ENABLED else "disabled; fp32 sign-PGD")

all_discovered = discover_anomaly_datasets(
    dataset=DISCOVERY_MODE,
    mvtec_root=str(MVTEC_ROOT) if "mvtec" in PROTOCOL_DATASETS else None,
    visa_root=str(VISA_ROOT) if "visa" in PROTOCOL_DATASETS else None,
    categories=None,
    max_samples_per_category=None,
    train_normal=False,
)
samples, assignments, rank_info, protocol_frame = bind_discovered_samples_from_partition_csvs(
    all_discovered, ATTACK_TRAIN_CSV, EVALUATION_CSV
)
assert_partition_disjoint(assignments)
SPLIT_PROTOCOL = split_protocol()
evaluation_ids = {pid for pid, part in assignments.items() if part == "evaluation"}

IMAGE_CACHE = {}
MASK_CACHE = {}


def image_loader(sample: MVTecSample) -> torch.Tensor:
    if not CACHE_INPUTS_IN_RAM:
        return load_image_tensor(sample, IMAGE_SIZE)
    if sample.protocol_id not in IMAGE_CACHE:
        IMAGE_CACHE[sample.protocol_id] = load_image_tensor(sample, IMAGE_SIZE).half().contiguous()
    return IMAGE_CACHE[sample.protocol_id]


def mask_loader(sample: MVTecSample) -> torch.Tensor:
    if not CACHE_INPUTS_IN_RAM:
        return torch.from_numpy(load_mask(sample, IMAGE_SIZE)).float()
    if sample.protocol_id not in MASK_CACHE:
        MASK_CACHE[sample.protocol_id] = torch.from_numpy(load_mask(sample, IMAGE_SIZE)).to(torch.uint8).contiguous()
    return MASK_CACHE[sample.protocol_id].float()


def draw_logical_batch(samples, order, cursor, rng, logical_batch_size):
    selected = []
    while len(selected) < logical_batch_size:
        if cursor >= len(order):
            rng.shuffle(order)
            cursor = 0
        take = min(logical_batch_size - len(selected), len(order) - cursor)
        selected.extend(samples[int(i)] for i in order[cursor:cursor + take])
        cursor += take
    return selected, cursor


@dataclass
class AccumulatedResult:
    delta: torch.Tensor
    actual_micro_batch_size: int
    gradient_accumulation_steps: int
    history: list[dict[str, float]]
    initial_losses: dict[str, float]
    final_losses: dict[str, float]
    diagnostic_sample_ids: list[str]
    selected_step: int
    selected_diagnostic_loss: float


def optimize_accumulated(
    attacker: TargetedPGD,
    source_samples: Sequence[MVTecSample],
    target_label: int,
    loss_mode: str,
    run_seed: int,
    total_steps: int,
    mask_fn=None,
    progress=None,
):
    """Universal PGD with gradient accumulation and automatic OOM fallback."""
    if not source_samples:
        raise ValueError("Universal optimization requires attack_train samples")

    reference = image_loader(source_samples[0]).float().unsqueeze(0).to(attacker.device)
    delta = attacker._initial_delta((1, 3, IMAGE_SIZE, IMAGE_SIZE), reference)
    del reference

    # The category delta is shared across its images, so the hinge applies
    # here too. optimize_universal does this internally; this loop accumulates
    # over micro-batches, so the floors are sliced per micro-batch instead.
    hinge_floors = attacker.hinge_floors(
        source_samples, image_loader, target_label, loss_mode, mask_loader=mask_fn
    )
    sample_position = {
        sample.protocol_id: index for index, sample in enumerate(source_samples)
    }
    # A category delta is shared too, so it accumulates the same way
    # optimize_universal does. Zero decay leaves this as plain sign-PGD.
    momentum = torch.zeros_like(delta)

    order = np.arange(len(source_samples))
    cursor = len(order)
    rng = np.random.default_rng(run_seed)
    micro_batch_size = min(MICRO_BATCH_SIZE, EFFECTIVE_BATCH_SIZE)
    diagnostic_samples = list(source_samples)
    diagnostic_sample_ids = [sample.protocol_id for sample in diagnostic_samples]
    # Baseline checkpoint is the unperturbed image, as in optimize_universal.
    best_delta = torch.zeros_like(delta)
    initial_losses = attacker._diagnostic_losses(
        diagnostic_samples,
        image_loader,
        best_delta,
        target_label,
        loss_mode,
        mask_loader=mask_fn,
    )
    history = []
    best_diagnostic_loss = initial_losses["total"]
    selected_step = 0

    for step in range(total_steps):
        logical_samples, cursor = draw_logical_batch(
            source_samples, order, cursor, rng, EFFECTIVE_BATCH_SIZE
        )
        while True:
            try:
                delta_leaf = delta.detach().requires_grad_(True)
                accumulated_global = torch.zeros_like(delta_leaf)
                accumulated_local = torch.zeros_like(delta_leaf)
                accumulated_total = torch.zeros_like(delta_leaf)
                pre_loss = 0.0

                for micro in chunked(logical_samples, micro_batch_size):
                    clean = torch.stack([image_loader(s) for s in micro]).float().to(attacker.device)
                    categories = [s.category for s in micro]
                    masks = (
                        torch.stack([mask_fn(s) for s in micro]).to(attacker.device)
                        if mask_fn is not None and loss_mode in {"local", "combined"} else None
                    )
                    micro_floors = None
                    if hinge_floors is not None:
                        selector = torch.as_tensor(
                            [sample_position[s.protocol_id] for s in micro],
                            device=attacker.device,
                            dtype=torch.long,
                        )
                        micro_floors = {
                            key: value.index_select(0, selector)
                            for key, value in hinge_floors.items()
                        }
                    with autocast_context():
                        components = attacker.objective_components(
                            (clean + delta_leaf).clamp(0, 1), categories, target_label,
                            loss_mode, spatial_masks=masks,
                            hinge_floors=micro_floors,
                        )
                    weight = len(micro) / EFFECTIVE_BATCH_SIZE
                    pre_loss += float(components["total"].detach()) * weight
                    if loss_mode == "combined":
                        global_grad = torch.autograd.grad(
                            components["global"], delta_leaf, retain_graph=True, only_inputs=True
                        )[0]
                        local_grad = torch.autograd.grad(
                            components["local"], delta_leaf, only_inputs=True
                        )[0]
                        accumulated_global.add_(global_grad.detach(), alpha=weight)
                        accumulated_local.add_(local_grad.detach(), alpha=weight)
                        del global_grad, local_grad
                    else:
                        total_grad = torch.autograd.grad(
                            components["total"], delta_leaf, only_inputs=True
                        )[0]
                        accumulated_total.add_(total_grad.detach(), alpha=weight)
                        del total_grad
                    del clean, masks, components

                gradient = (
                    attacker.config.global_weight * accumulated_global
                    + attacker.config.local_weight * accumulated_local
                    if loss_mode == "combined" else accumulated_total
                )
                step_size = attacker.step_size_at(step, total_steps)
                direction, momentum_cosine = attacker.update_direction(
                    gradient, momentum
                )
                if attacker.config.momentum_decay > 0.0:
                    momentum = direction
                delta = (delta.detach() - step_size * direction.sign()).clamp(
                    -EPSILON, EPSILON
                ).detach()
                fixed_losses = {}
                if (
                    step == 0
                    or (step + 1) % DIAGNOSTIC_INTERVAL == 0
                    or step + 1 == total_steps
                ):
                    fixed_losses = attacker._diagnostic_losses(
                        diagnostic_samples,
                        image_loader,
                        delta,
                        target_label,
                        loss_mode,
                        mask_loader=mask_fn,
                    )
                    if fixed_losses["total"] < best_diagnostic_loss:
                        best_diagnostic_loss = fixed_losses["total"]
                        best_delta = delta.detach().clone()
                        selected_step = step + 1
                record = {
                    "step": float(step + 1),
                    "batch_pre_total_loss": pre_loss,
                    "fixed_diagnostic_total_loss": fixed_losses.get(
                        "total", float("nan")
                    ),
                    "fixed_diagnostic_local_focal": fixed_losses.get(
                        "local_focal", float("nan")
                    ),
                    "fixed_diagnostic_local_dice": fixed_losses.get(
                        "local_dice", float("nan")
                    ),
                    "gradient_l2": float(gradient.norm().detach()),
                    "gradient_linf": float(gradient.abs().max().detach()),
                    "step_size": step_size,
                    "momentum_gradient_cosine": momentum_cosine,
                    "delta_saturation_fraction": float(
                        (delta.abs() >= EPSILON - 1e-7).float().mean().detach()
                    ),
                }
                history.append(record)
                if progress is not None:
                    progress(step + 1, total_steps, record)
                break
            except Exception as error:
                if not (AUTO_REDUCE_MICRO_BATCH_ON_OOM and is_cuda_oom(error) and micro_batch_size > 1):
                    raise
                micro_batch_size = max(1, micro_batch_size // 2)
                print("CUDA OOM: micro-batch reduced to", micro_batch_size)
                release_cuda()

    # Same switch as optimize_universal: the trajectory above is unchanged and
    # only the retained point differs.
    if attacker.config.checkpoint_selection == "final":
        selected = delta.detach()
        selected_step = total_steps
    else:
        selected = best_delta.detach()
    final_losses = attacker._diagnostic_losses(
        diagnostic_samples,
        image_loader,
        selected,
        target_label,
        loss_mode,
        mask_loader=mask_fn,
    )
    return AccumulatedResult(
        delta=selected,
        actual_micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=math.ceil(EFFECTIVE_BATCH_SIZE / micro_batch_size),
        history=history,
        initial_losses=initial_losses,
        final_losses=final_losses,
        diagnostic_sample_ids=diagnostic_sample_ids,
        selected_step=selected_step,
        selected_diagnostic_loss=(
            final_losses["total"]
            if attacker.config.checkpoint_selection == "final"
            else best_diagnostic_loss
        ),
    )

def artifact_path(dataset: str, category: str, fraction: float, direction: str, loss_mode: str):
    root = OUTPUT_ROOT / "noises" / dataset / fraction_tag(fraction) / "perturbations"
    name = f"per_category__{direction}__{loss_mode}__{category}.pt"
    return root / name


def reusable(pt_path: Path, expected: Dict) -> bool:
    if OVERWRITE_EXISTING or not pt_path.is_file():
        return False
    metadata = torch.load(pt_path, map_location="cpu", weights_only=False)["metadata"]
    return all(metadata.get(k) == v for k, v in expected.items())


attack_config = AttackConfig(
    image_size=IMAGE_SIZE,
    epsilon=EPSILON,
    step_size=UNIVERSAL_STEP_SIZE,
    steps=10,
    # Replaced per condition; the category's own image count sets it.
    universal_steps=1,
    random_start=True,
    temperature=0.07,
    global_weight=0.2,
    local_weight=0.8,
    mask_local_loss=True,
    local_background_weight=LOCAL_BACKGROUND_WEIGHT,
    normal_local_target=NORMAL_LOCAL_TARGET,
    normal_target_region_fraction=NORMAL_TARGET_REGION_FRACTION,
    normal_target_center_x=NORMAL_TARGET_CENTER_X,
    normal_target_center_y=NORMAL_TARGET_CENTER_Y,
    local_focal_weight=LOCAL_FOCAL_WEIGHT,
    local_dice_weight=LOCAL_DICE_WEIGHT,
    local_focal_gamma=LOCAL_FOCAL_GAMMA,
    local_dice_smooth=LOCAL_DICE_SMOOTH,
    loss_formulation=LOSS_FORMULATION,
    margin_topk_fraction=MARGIN_TOPK_FRACTIONS["normal_to_abnormal"],
    step_size_schedule=STEP_SIZE_SCHEDULE,
    checkpoint_selection=CHECKPOINT_SELECTION,
    margin_hinge_displacement=MARGIN_HINGE_DISPLACEMENT,
    momentum_decay=MOMENTUM_DECAY,
    step_size_min_ratio=STEP_SIZE_MIN_RATIO,
    diagnostic_interval=DIAGNOSTIC_INTERVAL,
    feature_layers=(6, 12, 18, 24),
    scopes=("per_category",),
    directions=DIRECTIONS,
    loss_modes=LOSS_MODES,
    per_image_batch_size=1,
    universal_batch_size=MICRO_BATCH_SIZE,
    seed=SEED,
)

REPO_COMMIT = git_commit(PROJECT_ROOT)
ANOMALYCLIP_COMMIT = git_commit(ANOMALYCLIP_ROOT)
GENERATOR_SCRIPT_SHA256 = sha256_file(Path(__file__))
ATTACK_CODE_SHA256 = sha256_file(
    PROJECT_ROOT / "adversarial_harness" / "attacks.py"
)
protocol_sha = split_sha256()
artifact_rows = []

for dataset_name in DATASETS:
    categories = sorted({s.category for s in samples if s.dataset == dataset_name})
    prompt_checkpoint = learnable_prompt_checkpoint(dataset_name, PROMPT_MODE)
    print(f"\n===== {dataset_name}: {PROMPT_MODE} =====")
    surrogate = CLIPSurrogate(
        anomalyclip_root=str(ANOMALYCLIP_ROOT),
        categories=categories,
        device="cuda",
        feature_layers=attack_config.feature_layers,
        clip_download_root=str(CLIP_CACHE),
        prompt_mode=PROMPT_MODE,
        learnable_prompt_checkpoint=prompt_checkpoint,
        prompt_dataset=dataset_name,
    )
    try:
        for fraction in TRAIN_FRACTIONS:
            fraction_pool = select_attack_train_fraction(samples, assignments, rank_info, fraction)
            if any(assignments[s.protocol_id] != "attack_train" for s in fraction_pool):
                raise RuntimeError("Evaluation image entered category optimization")
            for category in categories:
                eval_all = sorted(
                    [
                        s for s in samples
                        if s.dataset == dataset_name
                        and s.category == category
                        and assignments[s.protocol_id] == "evaluation"
                    ],
                    key=lambda s: s.protocol_id,
                )
                for direction in DIRECTIONS:
                    source_label, target_label = direction_labels(direction)
                    condition_config = replace(
                        attack_config,
                        margin_topk_fraction=MARGIN_TOPK_FRACTIONS[direction],
                    )
                    train_samples = sorted(
                        [
                            s for s in fraction_pool
                            if s.dataset == dataset_name
                            and s.category == category
                            and s.label == source_label
                        ],
                        key=lambda s: s.protocol_id,
                    )
                    attacked_eval = [s for s in eval_all if s.label == source_label]
                    condition_steps = derive_steps(
                        PER_CATEGORY_EPOCHS,
                        max(len(train_samples), 1),
                        EFFECTIVE_BATCH_SIZE,
                    )
                    condition_config = replace(
                        condition_config, universal_steps=condition_steps
                    )
                    if not train_samples or not attacked_eval:
                        raise RuntimeError(
                            f"Missing train/eval stratum for {dataset_name}/{category}/{direction}"
                        )
                    for loss_mode in LOSS_MODES:
                        pt_path = artifact_path(
                            dataset_name, category, fraction, direction, loss_mode
                        )
                        pt_path.parent.mkdir(parents=True, exist_ok=True)
                        expected = {
                            "format_version": "canonical_clip_per_category_segmentation_loss_v2",
                            "source_dataset": dataset_name,
                            "target_dataset": dataset_name,
                            "scope": "per_category",
                            "category": category,
                            "direction": direction,
                            "loss_mode": loss_mode,
                            "loss_formulation": LOSS_FORMULATION,
                            "attack_train_fraction": fraction,
                            "epsilon": EPSILON,
                            "step_size": UNIVERSAL_STEP_SIZE,
                            "optimization_epochs": PER_CATEGORY_EPOCHS,
                            "universal_steps": condition_steps,
                            "image_size": IMAGE_SIZE,
                            "seed": SEED,
                            "protocol_split_sha256": protocol_sha,
                            "label_balance_policy": protocol_frame.label_balance_policy.iloc[0],
                            "split_protocol": SPLIT_PROTOCOL,
                            "benchmark_commit": REPO_COMMIT,
                            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                            "configured_micro_batch_size": MICRO_BATCH_SIZE,
                            "anomalyclip_loader_commit": ANOMALYCLIP_COMMIT,
                            "generator_script_sha256": GENERATOR_SCRIPT_SHA256,
                            "attack_code_sha256": ATTACK_CODE_SHA256,
                            "global_objective": (
                                "signed_abnormal_minus_normal_margin"
                                if LOSS_FORMULATION == "margin_topk"
                                else "target_class_cross_entropy"
                            ),
                            "local_objective": (
                                "signed_topk_anomaly_margin"
                                if LOSS_FORMULATION == "margin_topk"
                                else "target_class_focal_plus_soft_dice"
                            ),
                            "margin_topk_fraction": MARGIN_TOPK_FRACTIONS[direction],
                            "local_focal_weight": LOCAL_FOCAL_WEIGHT,
                            "local_dice_weight": LOCAL_DICE_WEIGHT,
                            "local_focal_gamma": LOCAL_FOCAL_GAMMA,
                            "local_dice_smooth": LOCAL_DICE_SMOOTH,
                            "local_background_weight": LOCAL_BACKGROUND_WEIGHT,
                            "normal_local_target": NORMAL_LOCAL_TARGET,
                            "normal_target_region_fraction": NORMAL_TARGET_REGION_FRACTION,
                            "normal_target_center_x": NORMAL_TARGET_CENTER_X,
                            "normal_target_center_y": NORMAL_TARGET_CENTER_Y,
                            "step_size_schedule": STEP_SIZE_SCHEDULE,
                        "checkpoint_selection": CHECKPOINT_SELECTION,
                        "momentum_decay": MOMENTUM_DECAY,
                        "margin_hinge_displacement": (
                            MARGIN_HINGE_DISPLACEMENT
                            if MARGIN_HINGE_DISPLACEMENT is not None
                            else ""
                        ),
                            "step_size_min_ratio": STEP_SIZE_MIN_RATIO,
                            "diagnostic_interval": DIAGNOSTIC_INTERVAL,
                            "checkpoint_selection_partition": "full_attack_train",
                        }
                        expected.update(surrogate.prompt_provenance)
                        if reusable(pt_path, expected):
                            print(f"[reuse] {dataset_name}/{category}/{fraction_tag(fraction)}/{direction}/{loss_mode}")
                            metadata = torch.load(pt_path, map_location="cpu", weights_only=False)["metadata"]
                        else:
                            run_seed = condition_seed(
                                SEED, dataset_name, category, fraction, direction, loss_mode
                            )
                            seed_everything(run_seed)
                            print(
                                f"[generate] {dataset_name}/{category} fraction={fraction:.2f} "
                                f"direction={direction} loss={loss_mode} train={len(train_samples)}"
                            )
                            attacker = TargetedPGD(surrogate, condition_config)
                            bar = tqdm(total=condition_steps, desc="PGD", unit="step")

                            def progress(step, total, metrics):
                                bar.update(step - bar.n)
                                postfix = {
                                    "batch_pre": f"{metrics['batch_pre_total_loss']:.6f}",
                                    "sat": f"{metrics['delta_saturation_fraction']:.1%}",
                                }
                                fixed = metrics.get(
                                    "fixed_diagnostic_total_loss", float("nan")
                                )
                                if math.isfinite(fixed):
                                    postfix["full_train"] = f"{fixed:.6f}"
                                bar.set_postfix(postfix)

                            result = optimize_accumulated(
                                attacker,
                                train_samples,
                                target_label,
                                loss_mode,
                                run_seed,
                                condition_steps,
                                mask_fn=(
                                    mask_loader
                                    if LOSS_FORMULATION == "ce_focal_dice"
                                    and loss_mode in {"local", "combined"}
                                    else None
                                ),
                                progress=progress,
                            )
                            bar.close()
                            delta = result.delta.detach().cpu().float()
                            actual_linf = float(delta.abs().max())
                            if actual_linf > EPSILON + 1e-6:
                                raise RuntimeError(f"Linf budget violation: {actual_linf} > {EPSILON}")
                            metadata = {
                                **expected,
                                "run_seed": run_seed,
                                "source_label": source_label,
                                "target_label": target_label,
                                "attack_generator": "frozen_public_CLIP_surrogate",
                                "target_model_access_during_optimization": False,
                                "target_model_training_or_finetuning": False,
                                "optimization_partition": "attack_train",
                                "evaluation_partition_seen_during_optimization": False,
                                "attack_train_sample_count": len(train_samples),
                                "attack_train_sample_ids": [s.protocol_id for s in train_samples],
                                "evaluation_attacked_sample_count": len(attacked_eval),
                                "evaluation_attacked_sample_ids": [s.protocol_id for s in attacked_eval],
                                "actual_micro_batch_size": result.actual_micro_batch_size,
                                "gradient_accumulation_steps": result.gradient_accumulation_steps,
                                "diagnostic_sample_ids": result.diagnostic_sample_ids,
                                "initial_losses": result.initial_losses,
                                "final_losses": result.final_losses,
                                "loss_reduction": {
                                    key: result.initial_losses[key] - result.final_losses[key]
                                    for key in result.initial_losses
                                    if key in result.final_losses
                                },
                                "optimization_history": result.history,
                                "selected_step": result.selected_step,
                                "selected_diagnostic_loss": result.selected_diagnostic_loss,
                                "actual_linf": actual_linf,
                                "delta_sha256_float32": sha256_tensor(delta),
                                "protocol_attack_train_csv": str(ATTACK_TRAIN_CSV),
                                "protocol_evaluation_csv": str(EVALUATION_CSV),
                                "notes": (
                                    "Category-level universal delta fitted only on the selected nested "
                                    "attack_train subset. It applies only to held-out evaluation images "
                                    "of the same dataset/category/source label."
                                ),
                            }
                            torch.save({"delta": delta.half(), "metadata": metadata}, pt_path)
                            del result, attacker, delta
                            release_cuda()

                        row = dict(metadata)
                        row["artifact_path"] = str(pt_path)
                        row["artifact_file_sha256"] = sha256_file(pt_path)
                        artifact_rows.append(row)
    finally:
        surrogate.release()
        del surrogate
        release_cuda()

manifest_rows = []
noise_paths = []
for row in artifact_rows:
    artifact = Path(row["artifact_path"])
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    delta = payload["delta"].float()
    if tuple(delta.shape) != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"Unexpected delta shape in {artifact}: {tuple(delta.shape)}")
    if set(row["attack_train_sample_ids"]) & evaluation_ids:
        raise RuntimeError(f"Leakage in {artifact}")
    relative_noise = artifact.relative_to(OUTPUT_ROOT)
    noise_paths.append(artifact)
    manifest_rows.append({
        "scope": "per_category",
        "source_dataset": row["source_dataset"],
        "target_dataset": row["target_dataset"],
        "category": row["category"],
        "direction": row["direction"],
        "source_label": row["source_label"],
        "target_label": row["target_label"],
        "loss_mode": row["loss_mode"],
        "loss_formulation": row["loss_formulation"],
        "split_protocol": row["split_protocol"],
        "seed": row["seed"],
        "run_seed": row["run_seed"],
        **{field: row[field] for field in PROMPT_PROVENANCE_FIELDS},
        "attack_train_fraction": row["attack_train_fraction"],
        "attack_train_image_count": row["attack_train_sample_count"],
        "evaluation_attacked_image_count": row["evaluation_attacked_sample_count"],
        "noise_file": str(relative_noise),
        "noise_tensor_key": "delta",
        "artifact_sha256": row["artifact_file_sha256"],
        "protocol_split_sha256": protocol_sha,
        "label_balance_policy": row["label_balance_policy"],
        "apply_only_to_clean_label": row["source_label"],
        "keep_opposite_label_clean": True,
        "image_size": IMAGE_SIZE,
        "epsilon": EPSILON,
        "step_size": UNIVERSAL_STEP_SIZE,
        "optimization_epochs": row["optimization_epochs"],
        "optimization_steps": row["universal_steps"],
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "configured_micro_batch_size": MICRO_BATCH_SIZE,
        "local_objective": row["local_objective"],
        "global_objective": row["global_objective"],
        "margin_topk_fraction": row["margin_topk_fraction"],
        "local_focal_weight": row["local_focal_weight"],
        "local_dice_weight": row["local_dice_weight"],
        "local_focal_gamma": row["local_focal_gamma"],
        "local_dice_smooth": row["local_dice_smooth"],
        "local_background_weight": row["local_background_weight"],
        "normal_local_target": row["normal_local_target"],
        "normal_target_region_fraction": row["normal_target_region_fraction"],
        "normal_target_center_x": row["normal_target_center_x"],
        "normal_target_center_y": row["normal_target_center_y"],
        "step_size_schedule": row["step_size_schedule"],
        "application_order": (
            "load RGB [0,1] -> resize 518x518 -> clamp(clean + delta,0,1) "
            "-> target model default preprocessing"
        ),
    })

attack_manifest_path = OUTPUT_ROOT / "attack_manifest.csv"
pd.DataFrame(manifest_rows).sort_values(
    ["attack_train_fraction", "source_dataset", "category", "direction", "loss_mode"]
).to_csv(attack_manifest_path, index=False)
diagnostics_path = OUTPUT_ROOT / "optimization_diagnostics.csv"
pd.DataFrame([
    {
        "scope": row["scope"],
        "source_dataset": row["source_dataset"],
        "category": row["category"],
        "direction": row["direction"],
        "loss_mode": row["loss_mode"],
        "loss_formulation": row["loss_formulation"],
        "split_protocol": row["split_protocol"],
        "prompt_mode": row["prompt_mode"],
        "initial_total_loss": row["initial_losses"]["total"],
        "final_total_loss": row["final_losses"]["total"],
        "total_loss_reduction": row["loss_reduction"]["total"],
        "initial_local_focal": row["initial_losses"].get("local_focal", ""),
        "final_local_focal": row["final_losses"].get("local_focal", ""),
        "initial_local_dice": row["initial_losses"].get("local_dice", ""),
        "final_local_dice": row["final_losses"].get("local_dice", ""),
        "initial_global_margin": row["initial_losses"].get("global_margin", ""),
        "final_global_margin": row["final_losses"].get("global_margin", ""),
        "initial_local_topk": row["initial_losses"].get("local_topk", ""),
        "final_local_topk": row["final_losses"].get("local_topk", ""),
        "selected_step": row["selected_step"],
        "checkpoint_selection_partition": row["checkpoint_selection_partition"],
        "checkpoint_selection_image_count": len(row["diagnostic_sample_ids"]),
        "convergence_check_passed": (
            row["initial_losses"]["total"] - row["final_losses"]["total"] > 1e-8
        ),
    }
    for row in artifact_rows
]).to_csv(diagnostics_path, index=False)

for protocol_path in (ATTACK_TRAIN_CSV, EVALUATION_CSV, COMPLETE_RETAINED_CSV):
    shutil.copy2(protocol_path, OUTPUT_ROOT / protocol_path.name)
for required_path in (
    attack_manifest_path,
    diagnostics_path,
    OUTPUT_ROOT / "attack_train_indices.csv",
    OUTPUT_ROOT / "evaluation_test_indices.csv",
    OUTPUT_ROOT / "complete_retained_indices.csv",
):
    if not required_path.is_file():
        raise FileNotFoundError(f"Incomplete directory bundle: {required_path}")
for row in manifest_rows:
    recorded_noise = OUTPUT_ROOT / Path(str(row["noise_file"]))
    if not recorded_noise.is_file():
        raise FileNotFoundError(f"Manifest noise path is absent: {recorded_noise}")
    if sha256_file(recorded_noise) != row["artifact_sha256"]:
        raise RuntimeError(f"Manifest checksum mismatch: {recorded_noise}")

dataset_tag = "_".join(DATASETS)
archive_path = OUTPUT_BASE / (
    f"canonical_clip_per_category_{dataset_tag}_{SETUP_ID}.zip"
)
if archive_path.exists():
    archive_path.unlink()
with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
    for path in (ATTACK_TRAIN_CSV, EVALUATION_CSV, COMPLETE_RETAINED_CSV):
        archive.write(path, path.name, compress_type=zipfile.ZIP_DEFLATED)
    archive.write(attack_manifest_path, "attack_manifest.csv", compress_type=zipfile.ZIP_DEFLATED)
    archive.write(diagnostics_path, "optimization_diagnostics.csv", compress_type=zipfile.ZIP_DEFLATED)
    for artifact in sorted(set(noise_paths)):
        archive.write(
            artifact,
            artifact.relative_to(OUTPUT_ROOT),
            compress_type=zipfile.ZIP_STORED,
        )

with zipfile.ZipFile(archive_path, "r") as archive:
    archived_names = set(archive.namelist())
expected_archive_names = {
    "attack_train_indices.csv",
    "evaluation_test_indices.csv",
    "complete_retained_indices.csv",
    "attack_manifest.csv",
    "optimization_diagnostics.csv",
    *(str(row["noise_file"]).replace("\\", "/") for row in manifest_rows),
}
missing_archive_names = expected_archive_names - archived_names
if missing_archive_names:
    raise RuntimeError(
        f"ZIP bundle is missing entries: {sorted(missing_archive_names)[:5]}"
    )

print("\nPer-category artifacts:", len(artifact_rows))
print("ZIP:", archive_path)
