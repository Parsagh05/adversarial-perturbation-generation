#!/usr/bin/env python3
"""Generate source-dataset universal CLIP perturbations without split leakage.

For each attack-training fraction, one set of deltas is optimized for every
selected source dataset (2 directions x 3 losses). A delta is optimized once
from its source dataset and can then be evaluated on every selected evaluation
dataset. Target anomaly models are never loaded here.
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
from dataclasses import replace
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if not torch.cuda.is_available():
    raise RuntimeError("A CUDA-capable GPU is required")

PROJECT_ROOT = Path(__file__).resolve().parent
WORKING = Path(os.environ["WORK_DIR"]).expanduser().resolve()
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
    full_data_cross_setting,
    margin_hinge_setting,
    momentum_decay_setting,
    checkpoint_selection_setting,
    scope_output_path,
    snapshot_targets,
)
from adversarial_harness.attacks import TargetedPGD, direction_labels
from adversarial_harness.config import AttackConfig, VALID_LOSS_FORMULATIONS
from adversarial_harness.dataset import (
    MVTecSample,
    discover_anomaly_datasets,
    load_image_tensor,
    load_mask,
)
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
    bind_complete_retained_samples,
    fraction_tag,
    evaluation_datasets,
    parse_fraction_list,
    parse_numeric,
    protocol_datasets,
    select_attack_train_fraction,
    sha256_file,
    source_datasets,
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


IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "518"))
EPSILON = parse_numeric(os.environ["EPSILON"])
STEP_SIZE = parse_numeric(os.environ["PER_DATASET_STEP_SIZE"])
PER_DATASET_EPOCHS = float(os.environ["PER_DATASET_EPOCHS"])
# The fullsource delta trains on the complete source rather than the
# attack_train half, so it gets its own budget. Defaults to the per-dataset
# one, which is what it used before it had one.
PER_CROSS_EPOCHS = float(
    os.environ.get("PER_CROSS_EPOCHS", "") or PER_DATASET_EPOCHS
)


UNIVERSAL_BATCH_SIZE = int(os.environ.get("PER_DATASET_BATCH_SIZE", "1"))
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
DIAGNOSTIC_INTERVAL = int(os.environ.get("DIAGNOSTIC_INTERVAL", "10"))
SEED = int(os.environ.get("ATTACK_SEED", "111"))
OVERWRITE_EXISTING = bool_env("OVERWRITE_EXISTING", False)
TRAIN_FRACTIONS = parse_fraction_list(
    os.environ.get("PER_DATASET_ATTACK_TRAIN_FRACTIONS", "1.0"),
    name="PER_DATASET_ATTACK_TRAIN_FRACTIONS",
)
DIRECTIONS = csv_tuple("DIRECTIONS", "normal_to_abnormal,abnormal_to_normal")
LOSS_MODES = csv_tuple("LOSS_MODES", "global,local,combined")
LOSS_FORMULATION = os.environ.get("LOSS_FORMULATION", "ce_focal_dice")
if LOSS_FORMULATION not in VALID_LOSS_FORMULATIONS:
    raise ValueError(f"Unknown LOSS_FORMULATION: {LOSS_FORMULATION}")
PROMPT_MODE = os.environ.get("PROMPT_MODE", "frozen_winclip")
if PROMPT_MODE not in VALID_PROMPT_MODES:
    raise ValueError(f"Unknown PROMPT_MODE: {PROMPT_MODE}")

# Kept below PROMPT_MODE: this reads it, and a list comprehension only
# evaluates its element expression once the iterable yields, so an empty
# snapshot_targets() would hide the forward reference until the day
# SNAPSHOT_EPOCHS is finally set.
SETTINGS_TAG = os.environ["SETTINGS_TAG"]
SNAPSHOT_TARGETS = [
    (
        budget,
        Path(setups).expanduser().resolve()
        # The delta store is the per-dataset bundle; the snapshot's own row
        # builds its cross bundle by copying from there, as the main row does.
        / scope_output_path("per_dataset", budget, PROMPT_MODE, SETTINGS_TAG),
    )
    for budget, setups in snapshot_targets()
]
MARGIN_TOPK_FRACTIONS = {
    "normal_to_abnormal": float(os.environ.get("MARGIN_TOPK_FRACTION_NORMAL_TO_ABNORMAL", "0.20")),
    "abnormal_to_normal": float(os.environ.get("MARGIN_TOPK_FRACTION_ABNORMAL_TO_NORMAL", "0.40")),
}
if any(not 0.0 < value <= 1.0 for value in MARGIN_TOPK_FRACTIONS.values()):
    raise ValueError("MARGIN_TOPK_FRACTION values must be in (0, 1]")
TRANSFER_SETTINGS = csv_tuple("DATASET_TRANSFER_SETTINGS", "same_dataset,cross_dataset")
if not TRANSFER_SETTINGS or not set(TRANSFER_SETTINGS) <= {"same_dataset", "cross_dataset"}:
    raise ValueError(f"Unexpected DATASET_TRANSFER_SETTINGS: {TRANSFER_SETTINGS}")
SOURCE_DATASETS = source_datasets()
EVALUATION_DATASETS = evaluation_datasets()
PROTOCOL_DATASETS = protocol_datasets()
DISCOVERY_MODE = PROTOCOL_DATASETS[0] if len(PROTOCOL_DATASETS) == 1 else "both"
for dataset_name, dataset_root in (("mvtec", MVTEC_ROOT), ("visa", VISA_ROOT)):
    if dataset_name in PROTOCOL_DATASETS and not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

# DIRECTIONS and LOSS_MODES are selections, not fixed sets: every use below
# iterates over them, and AttackConfig rejects an empty list or an unknown name.
# Requiring the complete set here made this the only scope that refused a
# subset, so LOSS_MODES=global,local ran under per_category and per_image and
# aborted at the dataset scope. audit_generation.py reads the same two
# variables, so it checks for what the run was asked to produce.

# Deltas are optimized once into this store no matter which delivery bundles
# are requested; the path is unchanged so existing artifacts still reuse.
# settings / scope / epochs / prompt family. same_dataset and cross_dataset
# share one optimisation pass but sit at their own budgets, so the deltas are
# stored under the per-dataset bundle and copied into the cross one.
OUTPUT_ROOT = Path(os.environ["BUNDLE_PER_DATASET"]).expanduser().resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
BUNDLE_DIRECTORIES = {
    "same_dataset": OUTPUT_ROOT,
    "cross_dataset": Path(
        os.environ["BUNDLE_CROSS_DATASET"]
    ).expanduser().resolve(),
}
BUNDLE_SCOPES = {"same_dataset": "dataset", "cross_dataset": "cross_dataset"}
CLIP_CACHE = WORKING / "clip_cache"
CLIP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["ANOMALYCLIP_CLIP_CACHE"] = str(CLIP_CACHE)

print("GPU:", torch.cuda.get_device_name(0))
print("Protocol SHA256:", split_sha256())
print("Attack-train fractions:", TRAIN_FRACTIONS)
print("Loss formulation:", LOSS_FORMULATION)
print("Prompt mode:", PROMPT_MODE)
print("Source datasets:", SOURCE_DATASETS)
print("Transfer settings:", TRANSFER_SETTINGS)
print("Evaluation datasets:", EVALUATION_DATASETS)
print("Expected optimization runs:", len(SOURCE_DATASETS) * len(TRAIN_FRACTIONS) * len(DIRECTIONS) * len(LOSS_MODES))
print("Important: each source delta is optimized once and referenced by every evaluation dataset.")

all_discovered = discover_anomaly_datasets(
    dataset=DISCOVERY_MODE,
    mvtec_root=str(MVTEC_ROOT) if "mvtec" in PROTOCOL_DATASETS else None,
    visa_root=str(VISA_ROOT) if "visa" in PROTOCOL_DATASETS else None,
    categories=None,
    max_samples_per_category=None,
    train_normal=False,
)
complete_protocol_samples, complete_protocol_frame = bind_complete_retained_samples(
    all_discovered
)
samples, assignments, rank_info, protocol_frame = bind_discovered_samples_from_partition_csvs(
    all_discovered, ATTACK_TRAIN_CSV, EVALUATION_CSV
)
assert_partition_disjoint(assignments)

# Hard leakage guards.
SPLIT_PROTOCOL = split_protocol()
FULL_DATA_CROSS = full_data_cross_setting()
# The split protocol controls which natural dataset rows are retained. This
# independent switch controls whether cross-dataset consumes both retained
# halves or reuses the held-out per-dataset perturbation.
CROSS_DATASET_FULL_SOURCE = FULL_DATA_CROSS


def dataset_partitions():
    """Yield ``(key, use_full_source, settings)`` for every delta to optimize.

    The ordinary branch keeps one attack_train delta serving both transfer
    settings. FULL_DATA_CROSS=true adds a complete-source delta for
    complete-target delivery under either balanced or full protocol.
    """

    if not CROSS_DATASET_FULL_SOURCE:
        yield "", False, tuple(TRANSFER_SETTINGS)
        return
    if "same_dataset" in TRANSFER_SETTINGS:
        yield "", False, ("same_dataset",)
    if "cross_dataset" in TRANSFER_SETTINGS:
        yield "fullsource", True, ("cross_dataset",)

attack_train_ids = {pid for pid, part in assignments.items() if part == "attack_train"}
evaluation_ids = {pid for pid, part in assignments.items() if part == "evaluation"}
if attack_train_ids & evaluation_ids:
    raise RuntimeError("Protocol leakage detected before attack generation")


def image_loader(sample: MVTecSample) -> torch.Tensor:
    return load_image_tensor(sample, IMAGE_SIZE)


def mask_loader(sample: MVTecSample) -> torch.Tensor:
    return torch.from_numpy(load_mask(sample, IMAGE_SIZE)).float()


def source_training_samples(source_dataset: str, source_label: int, pool, full: bool):
    """Images this delta may train on.

    ``full`` uses every image of the source dataset for the cross-dataset
    delta; otherwise the nested attack_train subset is used, as before.
    """

    if not full:
        return sorted(
            [s for s in pool if s.dataset == source_dataset and s.label == source_label],
            key=lambda s: s.protocol_id,
        )
    return sorted(
        [
            s for s in complete_protocol_samples
            if s.dataset == source_dataset and s.label == source_label
        ],
        key=lambda s: s.protocol_id,
    )


def artifact_path(
    source_dataset: str, fraction: float, direction: str, loss_mode: str,
    partition_key: str = "", root: Path | None = None,
):
    root = (
        (OUTPUT_ROOT if root is None else root)
        / "noises"
        / source_dataset
        / fraction_tag(fraction)
        / "perturbations"
    )
    if partition_key:
        root = root / partition_key
    return root / f"dataset__{direction}__{loss_mode}.pt"


def write_snapshot_artifact(
    captured, metadata, attacker, source_train, target_label, loss_mode,
    snapshot_epochs, snapshot_steps, root, source_dataset, fraction, direction,
    partition_key,
):
    """Write one shorter budget's delta into the setup directory it owns.

    The launcher replays that budget as an ordinary setup afterwards. Its
    reuse guard compares configuration rather than results, so the epoch count
    and derived step count are what must describe the snapshot rather than the
    run that produced it. The diagnostics are recomputed on the captured delta
    so the recorded provenance describes it too, and the history is truncated
    to the steps the shorter run would have taken.
    """

    path = artifact_path(
        source_dataset, fraction, direction, loss_mode, partition_key,
        root=root,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    final_losses = attacker._diagnostic_losses(
        source_train, image_loader, captured, target_label, loss_mode,
        mask_loader=(
            mask_loader
            if LOSS_FORMULATION == "ce_focal_dice"
            and loss_mode in {"local", "combined"}
            else None
        ),
    )
    snapshot_metadata = {
        **metadata,
        "optimization_epochs": snapshot_epochs,
        "universal_steps": snapshot_steps,
        "final_losses": final_losses,
        "loss_reduction": {
            key: metadata["initial_losses"][key] - final_losses[key]
            for key in metadata["initial_losses"]
            if key in final_losses
        },
        "optimization_history": [
            row for row in metadata["optimization_history"]
            if float(row["step"]) <= snapshot_steps
        ],
        "selected_step": (
            snapshot_steps
            if attacker.config.checkpoint_selection == "final"
            else min(int(metadata["selected_step"]), snapshot_steps)
        ),
        "selected_diagnostic_loss": final_losses["total"],
        "actual_linf": float(captured.abs().max()),
        "delta_sha256_float32": sha256_tensor(captured),
        "snapshot_of_optimization_epochs": metadata["optimization_epochs"],
    }
    torch.save(
        {"delta": captured.half(), "metadata": snapshot_metadata}, path
    )
    print(f"[snapshot] {snapshot_epochs} epochs -> {path}")


def reusable(pt_path: Path, expected: Dict) -> bool:
    if OVERWRITE_EXISTING or not pt_path.is_file():
        return False
    metadata = torch.load(pt_path, map_location="cpu", weights_only=False)["metadata"]
    return all(metadata.get(key) == value for key, value in expected.items())


attack_config = AttackConfig(
    image_size=IMAGE_SIZE,
    epsilon=EPSILON,
    step_size=STEP_SIZE,
    steps=20,
    # Replaced per condition: the step count follows the epoch budget and
    # that condition's own training-set size.
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
    scopes=("dataset",),
    directions=DIRECTIONS,
    loss_modes=LOSS_MODES,
    per_image_batch_size=1,
    universal_batch_size=UNIVERSAL_BATCH_SIZE,
    seed=SEED,
)

REPO_COMMIT = git_commit(PROJECT_ROOT)
ANOMALYCLIP_COMMIT = git_commit(ANOMALYCLIP_ROOT)
GENERATOR_SCRIPT_SHA256 = sha256_file(Path(__file__))
ATTACK_CODE_SHA256 = sha256_file(PROJECT_ROOT / "adversarial_harness" / "attacks.py")
protocol_sha = split_sha256()
artifact_rows = []

for source_dataset in SOURCE_DATASETS:
    categories = sorted({s.category for s in samples if s.dataset == source_dataset})
    prompt_checkpoint = learnable_prompt_checkpoint(source_dataset, PROMPT_MODE)
    print(f"\n===== SOURCE {source_dataset}: {PROMPT_MODE} =====")
    surrogate = CLIPSurrogate(
        anomalyclip_root=str(ANOMALYCLIP_ROOT),
        categories=categories,
        device="cuda",
        feature_layers=attack_config.feature_layers,
        clip_download_root=str(CLIP_CACHE),
        prompt_mode=PROMPT_MODE,
        learnable_prompt_checkpoint=prompt_checkpoint,
        prompt_dataset=source_dataset,
    )
    try:
        for partition_key, use_full_source, partition_settings in dataset_partitions():
          if partition_key:
              print(f"--- {partition_key}: training on the complete source dataset ---")
          for fraction in TRAIN_FRACTIONS:
            fraction_pool = select_attack_train_fraction(samples, assignments, rank_info, fraction)
            if any(assignments[s.protocol_id] != "attack_train" for s in fraction_pool):
                raise RuntimeError("Evaluation image entered per-dataset optimization")
            for direction in DIRECTIONS:
                source_label, target_label = direction_labels(direction)
                condition_config = replace(
                    attack_config, margin_topk_fraction=MARGIN_TOPK_FRACTIONS[direction]
                )
                source_train = source_training_samples(
                    source_dataset, source_label, fraction_pool, use_full_source
                )
                condition_epochs = (
                    PER_CROSS_EPOCHS if use_full_source else PER_DATASET_EPOCHS
                )
                condition_steps = derive_steps(
                    condition_epochs, max(len(source_train), 1), UNIVERSAL_BATCH_SIZE
                )
                # The cross delta trains on its own cohort, so a snapshot
                # budget maps through the same derivation the run itself uses.
                snapshot_budgets = {
                    derive_steps(
                        budget[1] if use_full_source else budget[0],
                        max(len(source_train), 1),
                        UNIVERSAL_BATCH_SIZE,
                    ): (budget, root)
                    for budget, root in SNAPSHOT_TARGETS
                }
                condition_config = replace(
                    condition_config, universal_steps=condition_steps
                )
                if not source_train:
                    raise RuntimeError(
                        f"No attack_train images for {source_dataset}/{fraction}/{direction}"
                    )
                if any(sample.dataset != source_dataset for sample in source_train):
                    raise RuntimeError("A non-source dataset entered attack optimization")
                for loss_mode in LOSS_MODES:
                    pt_path = artifact_path(
                        source_dataset, fraction, direction, loss_mode, partition_key
                    )
                    pt_path.parent.mkdir(parents=True, exist_ok=True)
                    expected = {
                        "format_version": "canonical_clip_per_dataset_segmentation_loss_v2",
                        "source_dataset": source_dataset,
                        "scope": "dataset",
                        "direction": direction,
                        "loss_mode": loss_mode,
                        "loss_formulation": LOSS_FORMULATION,
                        "attack_train_fraction": fraction,
                        "epsilon": EPSILON,
                        "step_size": STEP_SIZE,
                        "optimization_epochs": condition_epochs,
                        "universal_steps": condition_steps,
                        "universal_batch_size": UNIVERSAL_BATCH_SIZE,
                        "image_size": IMAGE_SIZE,
                        "seed": SEED,
                        "protocol_split_sha256": protocol_sha,
                        "label_balance_policy": protocol_frame.label_balance_policy.iloc[0],
                        "split_protocol": SPLIT_PROTOCOL,
                        "training_source": (
                            "complete_source_dataset" if use_full_source
                            else "attack_train_partition"
                        ),
                        **({
                            "full_data_cross": FULL_DATA_CROSS,
                            "source_partition_policy": "all",
                        } if use_full_source else {}),
                        "benchmark_commit": REPO_COMMIT,
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
                        "checkpoint_selection_partition": (
                            "complete_source_dataset" if use_full_source
                            else "full_attack_train"
                        ),
                    }
                    expected.update(surrogate.prompt_provenance)
                    if reusable(pt_path, expected):
                        print(f"[reuse] {source_dataset}/{fraction_tag(fraction)}/{direction}/{loss_mode}")
                        metadata = torch.load(pt_path, map_location="cpu", weights_only=False)["metadata"]
                    else:
                        print(
                            f"[generate] source={source_dataset} fraction={fraction:.2f} "
                            f"direction={direction} loss={loss_mode} train={len(source_train)}"
                        )
                        run_seed = condition_seed(SEED, source_dataset, fraction, direction, loss_mode)
                        seed_everything(run_seed)
                        attacker = TargetedPGD(surrogate, condition_config)
                        bar = tqdm(total=condition_steps, desc="PGD", unit="step")

                        def progress(step, total, metrics):
                            bar.update(step - bar.n)
                            postfix = {
                                "batch_post": f"{metrics['total_loss']:.6f}",
                                "sat": f"{metrics['delta_saturation_fraction']:.1%}",
                            }
                            fixed = metrics.get("diagnostic_total_loss", float("nan"))
                            if math.isfinite(fixed):
                                postfix["full_train"] = f"{fixed:.6f}"
                            bar.set_postfix(postfix)

                        result = attacker.optimize_universal(
                            source_train,
                            image_loader,
                            target_label,
                            loss_mode,
                            mask_loader=(
                                mask_loader
                                if LOSS_FORMULATION == "ce_focal_dice"
                                and loss_mode in {"local", "combined"}
                                else None
                            ),
                            diagnostic_samples=source_train,
                            progress=progress,
                            snapshot_steps=tuple(snapshot_budgets),
                        )
                        bar.close()
                        delta = result.delta.detach().cpu().float()
                        actual_linf = float(delta.abs().max())
                        if actual_linf > EPSILON + 1e-6:
                            raise RuntimeError(f"Linf budget violation: {actual_linf} > {EPSILON}")
                        evaluation_counts = {
                            target: sum(
                                1 for s in (
                                    complete_protocol_samples if use_full_source else samples
                                )
                                if s.dataset == target
                                and s.label == source_label
                                and (
                                    use_full_source
                                    or assignments[s.protocol_id] == "evaluation"
                                )
                            )
                            for target in EVALUATION_DATASETS
                        }
                        metadata = {
                            **expected,
                            "run_seed": run_seed,
                            "source_label": source_label,
                            "target_label": target_label,
                            "attack_generator": "frozen_public_CLIP_surrogate",
                            "target_model_access_during_optimization": False,
                            "target_model_training_or_finetuning": False,
                            "optimization_partition": (
                                "complete_source_dataset" if use_full_source
                                else "attack_train"
                            ),
                            "evaluation_partition_seen_during_optimization": use_full_source,
                            "attack_train_sample_count": len(source_train),
                            "attack_train_sample_ids": [s.protocol_id for s in source_train],
                            "source_datasets": list(SOURCE_DATASETS),
                            "evaluation_datasets": list(EVALUATION_DATASETS),
                            "applicable_target_datasets": list(EVALUATION_DATASETS),
                            "evaluation_attacked_counts_by_target": evaluation_counts,
                            "source_categories": categories,
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
                            "protocol_complete_retained_csv": str(COMPLETE_RETAINED_CSV),
                            "notes": (
                                "One source-dataset universal delta, optimized exactly once on "
                                + (
                                    "the complete source test dataset for delivery only to the "
                                    "complete other dataset."
                                    if use_full_source else
                                    "the selected attack_train subset for same-dataset or "
                                    "held-out cross-dataset delivery."
                                )
                                + " No target anomaly model is used during optimization."
                            ),
                        }
                        torch.save({"delta": delta.half(), "metadata": metadata}, pt_path)
                        for steps, (budget, root) in snapshot_budgets.items():
                            captured = result.snapshots.get(steps)
                            if captured is None:
                                continue
                            write_snapshot_artifact(
                                captured,
                                metadata,
                                attacker,
                                source_train,
                                target_label,
                                loss_mode,
                                budget[1] if use_full_source else budget[0],
                                steps,
                                root,
                                source_dataset,
                                fraction,
                                direction,
                                partition_key,
                            )
                        del result, attacker, delta
                        release_cuda()

                    row = dict(metadata)
                    row["artifact_path"] = str(pt_path)
                    row["artifact_file_sha256"] = sha256_file(pt_path)
                    row["transfer_settings"] = tuple(partition_settings)
                    artifact_rows.append(row)
    finally:
        surrogate.release()
        del surrogate
        release_cuda()

# One artifact row per optimization. Delivery rows are grouped by transfer
# setting so same-dataset and cross-dataset become independently packaged
# scopes without ever optimizing the same delta twice.
delivery_rows = {setting: [] for setting in TRANSFER_SETTINGS}
unique_noise_paths = []
for row in artifact_rows:
    artifact = Path(row["artifact_path"])
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    delta = payload["delta"].float()
    if tuple(delta.shape) != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"Unexpected delta shape in {artifact}: {tuple(delta.shape)}")
    if float(delta.abs().max()) > EPSILON + 1e-6:
        raise RuntimeError(f"Budget violation in {artifact}")
    train_ids = set(row["attack_train_sample_ids"])
    relative_noise = artifact.relative_to(OUTPUT_ROOT)
    unique_noise_paths.append(artifact)
    for target_dataset in EVALUATION_DATASETS:
        setting = (
            "same_dataset"
            if row["source_dataset"] == target_dataset
            else "cross_dataset"
        )
        if setting not in delivery_rows or setting not in row["transfer_settings"]:
            continue
        # Only the explicit full-cross mode delivers both target partitions.
        # Otherwise cross-dataset reuses the per-dataset attack_train delta and
        # attacks the other dataset's evaluation partition.
        deliver_whole_target = (
            CROSS_DATASET_FULL_SOURCE and setting == "cross_dataset"
        )
        attacked_eval_ids = sorted(
            s.protocol_id for s in (
                complete_protocol_samples if deliver_whole_target else samples
            )
            if s.dataset == target_dataset
            and s.label == row["source_label"]
            and (deliver_whole_target or assignments[s.protocol_id] == "evaluation")
        )
        # The invariant is that a delta never trains on the images it is
        # attacked against, so compare it with what this bundle actually
        # delivers. Comparing against every evaluation id instead flagged the
        # complete-source delta that "full" creates on purpose: it trains on
        # all of the source including the source's own evaluation half, which
        # is why it is only ever delivered to the other dataset.
        overlap = train_ids & set(attacked_eval_ids)
        if overlap:
            raise RuntimeError(
                f"Leakage: {artifact} trains on {len(overlap)} of the images it "
                f"attacks in {target_dataset} ({setting})"
            )
        cross_provenance = ({
            "full_data_cross": FULL_DATA_CROSS,
            "cross_data_mode": (
                "fullcross"
                if CROSS_DATASET_FULL_SOURCE
                else "halfcross"
            ),
            "source_partition_policy": (
                "all" if CROSS_DATASET_FULL_SOURCE else "attack_train"
            ),
            "target_partition_policy": (
                "all" if CROSS_DATASET_FULL_SOURCE else "evaluation"
            ),
            "source_target_id_overlap_count": len(overlap),
        } if setting == "cross_dataset" else {})
        delivery_rows[setting].append({
            "setup_id": SETUP_ID,
            "scope": BUNDLE_SCOPES[setting],
            "source_dataset": row["source_dataset"],
            "target_dataset": target_dataset,
            "transfer_setting": setting,
            "split_protocol": row["split_protocol"],
            "training_source": row["training_source"],
            **cross_provenance,
            "direction": row["direction"],
            "source_label": row["source_label"],
            "target_label": row["target_label"],
            "loss_mode": row["loss_mode"],
            "loss_formulation": row["loss_formulation"],
            "seed": row["seed"],
            "run_seed": row["run_seed"],
            **{field: row[field] for field in PROMPT_PROVENANCE_FIELDS},
            "attack_train_fraction": row["attack_train_fraction"],
            "attack_train_image_count": row["attack_train_sample_count"],
            "evaluation_attacked_image_count": len(attacked_eval_ids),
            "noise_file": str(relative_noise),
            "noise_tensor_key": "delta",
            "artifact_sha256": row["artifact_file_sha256"],
            "protocol_split_sha256": protocol_sha,
            "label_balance_policy": row["label_balance_policy"],
            "evaluation_ids_source": (
                "complete_retained_indices.csv"
                if deliver_whole_target else "evaluation_test_indices.csv"
            ),
            "apply_only_to_clean_label": row["source_label"],
            "keep_opposite_label_clean": True,
            "image_size": IMAGE_SIZE,
            "epsilon": EPSILON,
            "step_size": STEP_SIZE,
            "optimization_epochs": row["optimization_epochs"],
            "optimization_steps": row["universal_steps"],
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

empty = [setting for setting, rows in delivery_rows.items() if not rows]
if empty:
    raise RuntimeError(
        f"No delivery rows for {empty}. Check SOURCE_DATASETS and EVALUATION_DATASETS: "
        "cross_dataset needs an evaluation dataset that is not a source dataset."
    )

diagnostics_frame = pd.DataFrame([
    {
        "scope": row["scope"],
        "source_dataset": row["source_dataset"],
        "category": "",
        "direction": row["direction"],
        "loss_mode": row["loss_mode"],
        "loss_formulation": row["loss_formulation"],
        "split_protocol": row["split_protocol"],
        "training_source": row["training_source"],
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
])

for setting in TRANSFER_SETTINGS:
    bundle = BUNDLE_DIRECTORIES[setting]
    rows = delivery_rows[setting]
    bundle.mkdir(parents=True, exist_ok=True)

    # Every bundle stays independently evaluable, so it carries its own deltas.
    referenced = sorted({Path(str(row["noise_file"])) for row in rows})
    for relative in referenced:
        destination = bundle / relative
        if destination.resolve() == (OUTPUT_ROOT / relative).resolve():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUTPUT_ROOT / relative, destination)

    attack_manifest_path = bundle / "attack_manifest.csv"
    pd.DataFrame(rows).sort_values(
        ["attack_train_fraction", "source_dataset", "target_dataset", "direction", "loss_mode"]
    ).to_csv(attack_manifest_path, index=False)
    diagnostics_path = bundle / "optimization_diagnostics.csv"
    diagnostics_frame.assign(scope=BUNDLE_SCOPES[setting]).to_csv(
        diagnostics_path, index=False
    )

    for protocol_path in (
        ATTACK_TRAIN_CSV, EVALUATION_CSV, COMPLETE_RETAINED_CSV
    ):
        shutil.copy2(protocol_path, bundle / protocol_path.name)
    for required_path in (
        attack_manifest_path,
        diagnostics_path,
        bundle / "attack_train_indices.csv",
        bundle / "evaluation_test_indices.csv",
        bundle / "complete_retained_indices.csv",
    ):
        if not required_path.is_file():
            raise FileNotFoundError(f"Incomplete directory bundle: {required_path}")
    for delivery in rows:
        recorded_noise = bundle / Path(str(delivery["noise_file"]))
        if not recorded_noise.is_file():
            raise FileNotFoundError(
                f"Manifest noise path is absent from directory bundle: {recorded_noise}"
            )
        if sha256_file(recorded_noise) != delivery["artifact_sha256"]:
            raise RuntimeError(f"Manifest checksum mismatch: {recorded_noise}")

    archive_path = bundle / "bundle.zip"
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
        for path in (ATTACK_TRAIN_CSV, EVALUATION_CSV, COMPLETE_RETAINED_CSV):
            archive.write(path, path.name, compress_type=zipfile.ZIP_DEFLATED)
        archive.write(
            attack_manifest_path, "attack_manifest.csv", compress_type=zipfile.ZIP_DEFLATED
        )
        archive.write(
            diagnostics_path,
            "optimization_diagnostics.csv",
            compress_type=zipfile.ZIP_DEFLATED,
        )
        for relative in referenced:
            archive.write(
                bundle / relative, relative.as_posix(), compress_type=zipfile.ZIP_STORED
            )

    with zipfile.ZipFile(archive_path, "r") as archive:
        archived_names = set(archive.namelist())
    expected_archive_names = {
        "attack_train_indices.csv",
        "evaluation_test_indices.csv",
        "complete_retained_indices.csv",
        "attack_manifest.csv",
        "optimization_diagnostics.csv",
        *(Path(str(row["noise_file"])).as_posix() for row in rows),
    }
    missing_archive_names = expected_archive_names - archived_names
    if missing_archive_names:
        raise RuntimeError(
            f"ZIP bundle is missing entries: {sorted(missing_archive_names)[:5]}"
        )
    print(f"\n[{setting}] manifest rows: {len(rows)}  deltas: {len(referenced)}")
    print(f"[{setting}] ZIP: {archive_path}")

print("\nPer-dataset optimization artifacts:", len(artifact_rows))
print(
    "Expected optimizations:",
    len(SOURCE_DATASETS) * len(TRAIN_FRACTIONS) * len(DIRECTIONS) * len(LOSS_MODES),
)
