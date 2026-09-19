#!/usr/bin/env python3
"""Generate independent per-image perturbations for evaluation rows only.

Per-image attacks are test-time instance-specific: each evaluation image gets
its own delta. No attack_train image is read, and gradients/deltas are never
mixed across images. Target anomaly models are never loaded here.
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
from dataclasses import replace
from pathlib import Path

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
ANOMALYCLIP_ROOT = WORKING / "AnomalyCLIP"
MVTEC_ROOT = Path(os.environ["MVTEC_ROOT"]).expanduser().resolve()
VISA_ROOT = Path(os.environ["VISA_ROOT"]).expanduser().resolve()
ATTACK_TRAIN_CSV = Path(os.environ["ATTACK_TRAIN_CSV"]).expanduser().resolve()
EVALUATION_CSV = Path(os.environ["EVALUATION_CSV"]).expanduser().resolve()
SETUP_ID = os.environ["SETUP_ID"]

if not ANOMALYCLIP_ROOT.exists():
    raise FileNotFoundError(ANOMALYCLIP_ROOT)

from setup_catalog import (
    checkpoint_selection_setting,
    derive_steps,
    scope_output_path,
    snapshot_targets,
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
    generation_datasets,
    parse_numeric,
    protocol_datasets,
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
STEP_SIZE = parse_numeric(os.environ["PER_IMAGE_STEP_SIZE"])
# A per-image delta trains on exactly one image, so an epoch is one PGD
# step and the two units coincide.
PER_IMAGE_EPOCHS = float(os.environ["PER_IMAGE_EPOCHS"])
PER_IMAGE_STEPS = derive_steps(PER_IMAGE_EPOCHS, 1, 1)
# Shorter budgets this run also produces. A per-image delta trains on one
# image, so an epoch is a step and the image component maps straight across.
SETTINGS_TAG = os.environ["SETTINGS_TAG"]
SNAPSHOT_TARGETS = [
    (
        budget,
        Path(setups).expanduser().resolve()
        / scope_output_path("per_image", budget, PROMPT_MODE, SETTINGS_TAG),
    )
    for budget, setups in snapshot_targets()
]
SNAPSHOT_IMAGE_STEPS = {
    derive_steps(budget[3], 1, 1): (budget, root)
    for budget, root in SNAPSHOT_TARGETS
}
EFFECTIVE_BATCH_SIZE = int(os.environ.get("PER_IMAGE_EFFECTIVE_BATCH_SIZE", "2"))
MICRO_BATCH_SIZE = int(os.environ.get("PER_IMAGE_MICRO_BATCH_SIZE", "2"))
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
CHECKPOINT_SELECTION = checkpoint_selection_setting()
STEP_SIZE_MIN_RATIO = float(os.environ.get("STEP_SIZE_MIN_RATIO", "0.1"))
DIAGNOSTIC_INTERVAL = int(os.environ.get("DIAGNOSTIC_INTERVAL", "8"))
EVALUATION_FRACTION = float(os.environ.get("PER_IMAGE_EVALUATION_FRACTION", "1.0"))
SEED = int(os.environ.get("ATTACK_SEED", "111"))
OVERWRITE_EXISTING = bool_env("OVERWRITE_EXISTING", False)
USE_AMP = bool_env("USE_AMP", True)
CACHE_INPUTS_IN_RAM = bool_env("CACHE_INPUTS_IN_RAM", True)
AUTO_REDUCE_MICRO_BATCH_ON_OOM = True
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

if not (0.0 < EVALUATION_FRACTION <= 1.0):
    raise ValueError("PER_IMAGE_EVALUATION_FRACTION must be in (0,1]")
if MICRO_BATCH_SIZE < 1 or EFFECTIVE_BATCH_SIZE < 1:
    raise ValueError("Batch sizes must be positive")
if MICRO_BATCH_SIZE > EFFECTIVE_BATCH_SIZE:
    raise ValueError("PER_IMAGE_MICRO_BATCH_SIZE cannot exceed effective batch size")

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


OUTPUT_ROOT = Path(os.environ["BUNDLE_PER_IMAGE"]).expanduser().resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CLIP_CACHE = WORKING / "clip_cache"
CLIP_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["ANOMALYCLIP_CLIP_CACHE"] = str(CLIP_CACHE)

print("GPU:", torch.cuda.get_device_name(0))
print("Protocol SHA256:", split_sha256())
print("Per-image steps / step size:", PER_IMAGE_STEPS, STEP_SIZE)
print("Evaluation fraction:", EVALUATION_FRACTION)
print("Autocast:", AMP_DTYPE_NAME if AMP_ENABLED else "disabled; fp32 sign-PGD")
print("Per-image uses zero attack_train images; each delta sees exactly its aligned evaluation image.")

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
attack_train_ids = {pid for pid, part in assignments.items() if part == "attack_train"}

SPLIT_PROTOCOL = split_protocol()
# Per-image fits the very image it attacks, so the train/evaluation split does
# not constrain it. Under "full" it therefore covers every test image; the
# historical protocol keeps it on the evaluation partition only.
ATTACK_EVERY_IMAGE = SPLIT_PROTOCOL == "full"

# Deterministic nested evaluation subset per dataset/category/label. Full=1.0 by default.
evaluation_samples = []
for sample in samples:
    if sample.dataset not in DATASETS:
        continue
    pid = sample.protocol_id
    if not ATTACK_EVERY_IMAGE and assignments[pid] != "evaluation":
        continue
    info = rank_info[pid]
    rank = int(info["evaluation_rank"])
    size = int(info["evaluation_stratum_size"])
    keep = max(1, int(math.ceil(size * EVALUATION_FRACTION)))
    if rank <= keep:
        evaluation_samples.append(sample)
if not ATTACK_EVERY_IMAGE and any(
    s.protocol_id in attack_train_ids for s in evaluation_samples
):
    raise RuntimeError("Attack-train image entered per-image generation")
print(f"Split protocol: {SPLIT_PROTOCOL}; per-image targets: {len(evaluation_samples)}")

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


def run_logical_batch(attacker, batch_samples, target_label, loss_mode):
    """Optimize one independent delta per image; reduce micro-batch on CUDA OOM."""
    micro_batch_size = min(MICRO_BATCH_SIZE, len(batch_samples))
    while True:
        try:
            output_deltas = []
            snapshot_deltas = {steps: [] for steps in SNAPSHOT_IMAGE_STEPS}
            diagnostic_batches = []
            for micro in chunked(list(batch_samples), micro_batch_size):
                clean = torch.stack([image_loader(s) for s in micro]).float()
                masks = (
                    torch.stack([mask_loader(s) for s in micro])
                    if LOSS_FORMULATION == "ce_focal_dice"
                    and loss_mode in {"local", "combined"}
                    else None
                )
                with autocast_context():
                    with torch.no_grad():
                        initial_components = attacker.objective_components(
                            clean.to(attacker.device),
                            [s.category for s in micro],
                            target_label,
                            loss_mode,
                            spatial_masks=(
                                masks.to(attacker.device) if masks is not None else None
                            ),
                        )
                    micro_snapshots = {}
                    adversarial, delta = attacker.perturb_batch(
                        clean,
                        [s.category for s in micro],
                        target_label,
                        loss_mode,
                        spatial_masks=masks,
                        snapshot_steps=tuple(SNAPSHOT_IMAGE_STEPS),
                        snapshot_sink=micro_snapshots,
                    )
                    for steps, captured in micro_snapshots.items():
                        snapshot_deltas[steps].extend(
                            captured[index].detach().cpu()
                            for index in range(captured.shape[0])
                        )
                    with torch.no_grad():
                        final_components = attacker.objective_components(
                            adversarial,
                            [s.category for s in micro],
                            target_label,
                            loss_mode,
                            spatial_masks=(
                                masks.to(attacker.device) if masks is not None else None
                            ),
                        )
                diagnostic_batches.append(
                    {
                        "count": len(micro),
                        "initial": {
                            key: float(value.detach())
                            for key, value in initial_components.items()
                        },
                        "final": {
                            key: float(value.detach())
                            for key, value in final_components.items()
                        },
                    }
                )
                output_deltas.extend(x.detach().cpu().half() for x in delta)
                del clean, masks, adversarial, delta, initial_components, final_components
            return output_deltas, micro_batch_size, diagnostic_batches
        except Exception as error:
            if not (AUTO_REDUCE_MICRO_BATCH_ON_OOM and is_cuda_oom(error) and micro_batch_size > 1):
                raise
            micro_batch_size = max(1, micro_batch_size // 2)
            print("CUDA OOM: per-image micro-batch reduced to", micro_batch_size)
            release_cuda()


def aggregate_diagnostics(batches):
    totals = {"initial": {}, "final": {}}
    counts = {"initial": {}, "final": {}}
    for batch in batches:
        count = int(batch["count"])
        for phase in ("initial", "final"):
            for key, value in batch[phase].items():
                totals[phase][key] = totals[phase].get(key, 0.0) + value * count
                counts[phase][key] = counts[phase].get(key, 0) + count
    return {
        phase: {
            key: totals[phase][key] / counts[phase][key]
            for key in totals[phase]
        }
        for phase in ("initial", "final")
    }

def artifact_path(
    dataset: str, category: str, direction: str, loss_mode: str,
    root: Path | None = None,
):
    root = (
        (OUTPUT_ROOT if root is None else root)
        / "noises"
        / dataset
        / "perturbations"
        / f"per_image__{direction}__{loss_mode}"
    )
    return root / f"{category}.pt"


def write_snapshot_artifact(
    captured, sample_ids, metadata, snapshot_epochs, snapshot_steps, root,
    dataset_name, category, direction, loss_mode,
):
    """Write one shorter budget's per-image deltas into the directory it owns.

    Per-image has no training cohort, so there are no diagnostics to recompute:
    the recorded losses already describe each image against itself. Only the
    budget and the deltas differ from the run that produced them.
    """

    path = artifact_path(
        dataset_name, category, direction, loss_mode,
        root=root,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_metadata = {
        **metadata,
        "optimization_epochs": snapshot_epochs,
        "per_image_steps": snapshot_steps,
        "actual_linf": float(captured.abs().max()),
        "snapshot_of_optimization_epochs": metadata["optimization_epochs"],
    }
    torch.save(
        {
            "deltas": captured.half(),
            "sample_ids": sample_ids,
            "metadata": snapshot_metadata,
        },
        path,
    )
    print(f"[snapshot] {snapshot_epochs} epochs {category} -> {path}")


def reusable(pt_path: Path, expected: dict) -> bool:
    if OVERWRITE_EXISTING or not pt_path.is_file():
        return False
    metadata = torch.load(pt_path, map_location="cpu", weights_only=False)["metadata"]
    return all(metadata.get(k) == v for k, v in expected.items())


attack_config = AttackConfig(
    image_size=IMAGE_SIZE,
    epsilon=EPSILON,
    step_size=STEP_SIZE,
    steps=PER_IMAGE_STEPS,
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
    step_size_min_ratio=STEP_SIZE_MIN_RATIO,
    diagnostic_interval=DIAGNOSTIC_INTERVAL,
    feature_layers=(6, 12, 18, 24),
    scopes=("per_image",),
    directions=DIRECTIONS,
    loss_modes=LOSS_MODES,
    per_image_batch_size=EFFECTIVE_BATCH_SIZE,
    universal_batch_size=1,
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
seed_everything(SEED)

for dataset_name in DATASETS:
    categories = sorted({s.category for s in evaluation_samples if s.dataset == dataset_name})
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
        for category in categories:
            category_eval = sorted(
                [s for s in evaluation_samples if s.dataset == dataset_name and s.category == category],
                key=lambda s: s.protocol_id,
            )
            for direction in DIRECTIONS:
                source_label, target_label = direction_labels(direction)
                condition_config = replace(
                    attack_config, margin_topk_fraction=MARGIN_TOPK_FRACTIONS[direction]
                )
                attacked_eval = [s for s in category_eval if s.label == source_label]
                if not attacked_eval:
                    raise RuntimeError(f"No evaluation images for {dataset_name}/{category}/{direction}")
                for loss_mode in LOSS_MODES:
                    run_seed = condition_seed(
                        SEED, dataset_name, category, direction, loss_mode
                    )
                    pt_path = artifact_path(dataset_name, category, direction, loss_mode)
                    pt_path.parent.mkdir(parents=True, exist_ok=True)
                    expected = {
                        "format_version": "canonical_clip_per_image_segmentation_loss_v2",
                        "source_dataset": dataset_name,
                        "target_dataset": dataset_name,
                        "scope": "per_image",
                        "category": category,
                        "direction": direction,
                        "loss_mode": loss_mode,
                        "loss_formulation": LOSS_FORMULATION,
                        "epsilon": EPSILON,
                        "step_size": STEP_SIZE,
                        "optimization_epochs": PER_IMAGE_EPOCHS,
                        "per_image_steps": PER_IMAGE_STEPS,
                        "image_size": IMAGE_SIZE,
                        "seed": SEED,
                        "run_seed": run_seed,
                        "protocol_split_sha256": protocol_sha,
                        "label_balance_policy": protocol_frame.label_balance_policy.iloc[0],
                        "split_protocol": SPLIT_PROTOCOL,
                        "benchmark_commit": REPO_COMMIT,
                        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                        "configured_micro_batch_size": MICRO_BATCH_SIZE,
                        "evaluation_fraction": EVALUATION_FRACTION,
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
                        "step_size_min_ratio": STEP_SIZE_MIN_RATIO,
                    }
                    expected.update(surrogate.prompt_provenance)
                    if reusable(pt_path, expected):
                        print(f"[reuse] {dataset_name}/{category}/{direction}/{loss_mode}")
                        metadata = torch.load(pt_path, map_location="cpu", weights_only=False)["metadata"]
                    else:
                        seed_everything(run_seed)
                        print(
                            f"[generate] {dataset_name}/{category}/{direction}/{loss_mode}; "
                            f"evaluation_images={len(attacked_eval)}"
                        )
                        attacker = TargetedPGD(surrogate, condition_config)
                        delta_list = []
                        actual_micro_sizes = []
                        diagnostic_batches = []
                        logical_batches = list(chunked(attacked_eval, EFFECTIVE_BATCH_SIZE))
                        for logical_batch in tqdm(logical_batches, desc="per-image PGD", unit="batch"):
                            batch_deltas, actual_micro, batch_diagnostics = run_logical_batch(
                                attacker, logical_batch, target_label, loss_mode
                            )
                            delta_list.extend(batch_deltas)
                            actual_micro_sizes.append(actual_micro)
                            diagnostic_batches.extend(batch_diagnostics)
                        losses = aggregate_diagnostics(diagnostic_batches)
                        deltas = torch.stack(delta_list).float()
                        sample_ids = [s.protocol_id for s in attacked_eval]
                        if len(sample_ids) != deltas.shape[0]:
                            raise RuntimeError("Per-image alignment mismatch")
                        actual_linf = float(deltas.abs().max())
                        if actual_linf > EPSILON + 1e-6:
                            raise RuntimeError(f"Linf budget violation: {actual_linf} > {EPSILON}")
                        metadata = {
                            **expected,
                            "source_label": source_label,
                            "target_label": target_label,
                            "artifact_layout": "deltas[i] belongs only to sample_ids[i]",
                            "attack_generator": "frozen_public_CLIP_surrogate",
                            "target_model_access_during_optimization": False,
                            "target_model_training_or_finetuning": False,
                            "optimization_partition": "evaluation_instance_itself",
                            "attack_train_sample_count": 0,
                            "attack_train_sample_ids": [],
                            "evaluation_sample_count": len(sample_ids),
                            "evaluation_sample_ids": sample_ids,
                            "independent_delta_per_image": True,
                            "cross_image_gradient_mixing": False,
                            "actual_micro_batch_size": min(actual_micro_sizes) if actual_micro_sizes else MICRO_BATCH_SIZE,
                            "actual_linf": actual_linf,
                            "initial_losses": losses["initial"],
                            "final_losses": losses["final"],
                            "loss_reduction": {
                                key: losses["initial"][key] - losses["final"][key]
                                for key in losses["initial"]
                                if key in losses["final"]
                            },
                            "deltas_sha256_float32": sha256_tensor(deltas),
                            "protocol_evaluation_csv": str(EVALUATION_CSV),
                            "notes": (
                                "Instance-specific test-time attack. Each delta is optimized only for its "
                                "aligned held-out evaluation image. This mode does not use attack_train and "
                                "must not be described as universal attack training."
                            ),
                        }
                        torch.save(
                            {"deltas": deltas.half(), "sample_ids": sample_ids, "metadata": metadata},
                            pt_path,
                        )
                        del attacker, delta_list, deltas, diagnostic_batches
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
    sample_ids = list(payload["sample_ids"])
    deltas = payload["deltas"].float()
    if deltas.shape[0] != len(sample_ids):
        raise RuntimeError(f"Alignment mismatch in {artifact}")
    if sample_ids != list(row["evaluation_sample_ids"]):
        raise RuntimeError(f"Stored sample ID order mismatch in {artifact}")
    # Under "full" per-image deliberately covers every test image, the
    # attack_train half included, because each delta fits the single image it
    # attacks and so has nothing to hold out. The selection above is gated on
    # ATTACK_EVERY_IMAGE for that reason; this check has to agree with it.
    if not ATTACK_EVERY_IMAGE and set(sample_ids) & attack_train_ids:
        raise RuntimeError(f"Attack-train leakage in per-image artifact {artifact}")
    relative_noise = artifact.relative_to(OUTPUT_ROOT)
    noise_paths.append(artifact)
    manifest_rows.append({
        "setup_id": SETUP_ID,
        "scope": "per_image",
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
        "attack_train_fraction": 0.0,
        "attack_train_image_count": 0,
        "evaluation_attacked_image_count": row["evaluation_sample_count"],
        "noise_file": str(relative_noise),
        "noise_tensor_key": "deltas",
        "sample_ids_key": "sample_ids",
        "alignment_rule": "sample_ids[i] maps exactly to deltas[i]; never reuse on another image",
        "artifact_sha256": row["artifact_file_sha256"],
        "protocol_split_sha256": protocol_sha,
        "label_balance_policy": row["label_balance_policy"],
        "image_size": IMAGE_SIZE,
        "epsilon": EPSILON,
        "step_size": STEP_SIZE,
        "optimization_epochs": row["optimization_epochs"],
        "optimization_steps": PER_IMAGE_STEPS,
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
            "load aligned RGB [0,1] -> resize 518x518 -> clamp(clean + deltas[i],0,1) "
            "-> target model default preprocessing"
        ),
    })

attack_manifest_path = OUTPUT_ROOT / "attack_manifest.csv"
pd.DataFrame(manifest_rows).sort_values(
    ["source_dataset", "category", "direction", "loss_mode"]
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

archive_path = OUTPUT_ROOT / "bundle.zip"
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

print("\nPer-image artifact shards:", len(artifact_rows))
print("ZIP:", archive_path)
