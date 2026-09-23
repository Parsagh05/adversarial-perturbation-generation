#!/usr/bin/env python3
"""Shared paths, split handling, and small utilities for all attack modes."""
from __future__ import annotations

import ast
import dataclasses
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
WORK_DIR = Path(os.environ.get("WORK_DIR", PROJECT_ROOT / "runtime")).expanduser().resolve()
ANOMALYCLIP_ROOT = WORK_DIR / "AnomalyCLIP"
MVTEC_ROOT = Path(os.environ["MVTEC_ROOT"]).expanduser().resolve()
VISA_ROOT = Path(os.environ["VISA_ROOT"]).expanduser().resolve()
OUTPUT_BASE = Path(os.environ["OUTPUT_BASE"]).expanduser().resolve()
PROTOCOL_DIR = OUTPUT_BASE / "protocol"
ATTACK_TRAIN_CSV = PROTOCOL_DIR / "attack_train_indices.csv"
EVALUATION_CSV = PROTOCOL_DIR / "evaluation_test_indices.csv"
COMPLETE_RETAINED_CSV = PROTOCOL_DIR / "complete_retained_indices.csv"
# balanced: downsample each category to min(normal, abnormal) so both labels
#   have equal counts, discarding the surplus. The historical protocol.
# full: keep every image and split each label by the same fraction, so the
#   category's natural class ratio survives into both partitions.
LABEL_BALANCE_POLICY = "per_dataset_category_equal_labels_v1"
FULL_LABEL_POLICY = "per_dataset_category_all_images_v1"
SPLIT_PROTOCOLS = ("balanced", "full")

REQUIRED_COLUMNS = {
    "protocol_id", "dataset", "category", "defect_type", "label", "partition",
    "image_relative_path", "mask_relative_path", "attack_train_rank",
    "attack_train_stratum_size", "evaluation_rank", "evaluation_stratum_size",
    "split_seed", "evaluation_fraction", "label_balance_policy",
    "original_label_stratum_size", "balanced_label_stratum_size",
}


def bool_env(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def csv_tuple(name: str, default: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in os.environ.get(name, default).split(",") if x.strip())


def _dataset_selection(name: str, default: str) -> tuple[str, ...]:
    datasets = csv_tuple(name, default)
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError(f"{name} must contain unique dataset names")
    unknown = sorted(set(datasets) - {"mvtec", "visa"})
    if unknown:
        raise ValueError(f"Unknown {name} values: {unknown}")
    return datasets


def source_datasets() -> tuple[str, ...]:
    """Datasets whose attack-training images may optimize perturbations."""

    legacy = os.environ.get("GENERATION_DATASETS")
    return _dataset_selection(
        "SOURCE_DATASETS",
        legacy if legacy is not None else "mvtec",
    )


def evaluation_datasets() -> tuple[str, ...]:
    """Datasets whose fixed held-out IDs may receive universal perturbations."""

    return _dataset_selection("EVALUATION_DATASETS", "mvtec,visa")


def protocol_datasets() -> tuple[str, ...]:
    """Stable union needed to create attack-train and evaluation CSVs."""

    return tuple(dict.fromkeys((*source_datasets(), *evaluation_datasets())))


def generation_datasets() -> tuple[str, ...]:
    """Backward-compatible alias used by same-dataset attack scopes."""

    return source_datasets()


def split_protocol() -> str:
    """``balanced`` (historical) or ``full`` (keep every image)."""

    protocol = os.environ.get("SPLIT_PROTOCOL", "balanced").strip().lower()
    if protocol not in SPLIT_PROTOCOLS:
        raise ValueError(
            f"SPLIT_PROTOCOL must be one of {SPLIT_PROTOCOLS}, got {protocol!r}"
        )
    return protocol


def label_policy_for(protocol: str) -> str:
    return LABEL_BALANCE_POLICY if protocol == "balanced" else FULL_LABEL_POLICY


def parse_numeric(raw: str) -> float:
    """Parse a decimal or one division expression without using ``eval``.

    ``fractions.Fraction`` accepts ``"8/255"`` and ``"0.25"`` separately,
    but it does not accept ``"0.25/255"``. Step sizes use the latter form, so
    parse each side independently before performing the division.
    """

    text = str(raw).strip()
    parts = [part.strip() for part in text.split("/")]
    if len(parts) == 1:
        return float(Fraction(parts[0]))
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Invalid numeric expression: {raw!r}")
    numerator = Fraction(parts[0])
    denominator = Fraction(parts[1])
    if denominator == 0:
        raise ValueError(f"Numeric expression divides by zero: {raw!r}")
    return float(numerator / denominator)


def parse_fraction_list(raw: str, *, name: str) -> tuple[float, ...]:
    values = sorted({float(x.strip()) for x in raw.split(",") if x.strip()})
    if not values or any(not (0.0 < x <= 1.0) for x in values):
        raise ValueError(f"{name} must contain values in (0,1]")
    return tuple(values)


def fraction_tag(value: float) -> str:
    return f"f{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_sha256() -> str:
    digest = hashlib.sha256()
    for path in (ATTACK_TRAIN_CSV, EVALUATION_CSV, COMPLETE_RETAINED_CSV):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def condition_seed(base: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, (base, *parts))).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _relative_path(path_value, dataset_name: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    root = MVTEC_ROOT if dataset_name == "mvtec" else VISA_ROOT
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _validate_partition_frame(path: Path, expected_partition: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"protocol_id": str})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path.name} is missing columns: {sorted(missing)}")
    if frame["protocol_id"].duplicated().any():
        raise RuntimeError(f"Duplicate protocol_id values in {path}")
    if set(frame["partition"].astype(str)) != {expected_partition}:
        raise RuntimeError(f"{path.name} must contain only {expected_partition} rows")
    return frame


def _assert_protocol_label_balance(frame: pd.DataFrame) -> None:
    """Both labels must always be present; equal counts only under ``balanced``."""

    policies = set(frame["label_balance_policy"].astype(str))
    if policies not in ({LABEL_BALANCE_POLICY}, {FULL_LABEL_POLICY}):
        raise RuntimeError(
            "Protocol CSVs use an unsupported or mixed label policy: "
            f"{sorted(policies)}"
        )
    counts = frame.groupby(
        ["dataset", "category", "partition", "label"]
    ).size().unstack(fill_value=0)
    if set(counts.columns) != {0, 1}:
        raise RuntimeError("Every protocol stratum must contain labels 0 and 1")
    if (counts[[0, 1]] == 0).any().any():
        raise RuntimeError("Every protocol stratum must contain both labels")
    if policies == {FULL_LABEL_POLICY}:
        # Keeping every image means the counts are deliberately unequal; the
        # split is stratified instead, so each label appears on both sides.
        return
    unequal = counts[counts[0] != counts[1]]
    if not unequal.empty:
        raise RuntimeError(
            "Protocol is not label-balanced within dataset/category/partition: "
            f"{list(unequal.index[:5])}"
        )


def _balanced_category_groups(
    samples: Sequence, split_seed: int, protocol: str = "balanced"
):
    """Deterministically shuffled label groups per category.

    ``balanced`` truncates both labels to min(normal, abnormal), discarding the
    surplus. ``full`` keeps every image, so the category's own class ratio is
    preserved and nothing is thrown away.
    """

    raw_groups: dict[tuple[str, str, int], list] = {}
    for sample in samples:
        key = (sample.dataset, sample.category, int(sample.label))
        raw_groups.setdefault(key, []).append(sample)

    category_keys = sorted({(dataset, category) for dataset, category, _ in raw_groups})
    balanced = {}
    original_sizes = {}
    for dataset, category in category_keys:
        normal = raw_groups.get((dataset, category, 0), [])
        anomalous = raw_groups.get((dataset, category, 1), [])
        if not normal or not anomalous:
            raise RuntimeError(
                f"Need both labels in {dataset}/{category} for a balanced protocol"
            )
        if min(len(normal), len(anomalous)) < 2:
            raise RuntimeError(
                f"Need at least two images per label in {dataset}/{category}"
            )
        balanced_size = min(len(normal), len(anomalous))
        for label, group in ((0, normal), (1, anomalous)):
            shuffled = sorted(group, key=lambda sample: sample.protocol_id)
            rng = random.Random(_stable_seed(split_seed, dataset, category, label))
            rng.shuffle(shuffled)
            key = (dataset, category, label)
            keep = balanced_size if protocol == "balanced" else len(shuffled)
            balanced[key] = shuffled[:keep]
            original_sizes[key] = len(group)
    return balanced, original_sizes


def retained_protocol_samples(samples: Sequence) -> list:
    """Return every sample retained by the selected balanced/full protocol."""

    groups, _ = _balanced_category_groups(
        samples,
        int(os.environ.get("SPLIT_SEED", "111")),
        split_protocol(),
    )
    return sorted(
        (sample for group in groups.values() for sample in group),
        key=lambda sample: sample.protocol_id,
    )


def load_protocol() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = _validate_partition_frame(ATTACK_TRAIN_CSV, "attack_train")
    evaluation = _validate_partition_frame(EVALUATION_CSV, "evaluation")
    overlap = set(train.protocol_id) & set(evaluation.protocol_id)
    if overlap:
        raise RuntimeError(f"Train/evaluation overlap detected: {sorted(overlap)[:5]}")
    _assert_protocol_label_balance(pd.concat([train, evaluation], ignore_index=True))
    return train, evaluation


def load_complete_retained_protocol() -> pd.DataFrame:
    """Load the self-contained cohort used by fullcross generation/evaluation."""

    if not COMPLETE_RETAINED_CSV.is_file():
        raise FileNotFoundError(COMPLETE_RETAINED_CSV)
    frame = pd.read_csv(COMPLETE_RETAINED_CSV, dtype={"protocol_id": str})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"{COMPLETE_RETAINED_CSV.name} is missing columns: {sorted(missing)}"
        )
    if frame.protocol_id.duplicated().any():
        raise RuntimeError(f"Duplicate protocol_id values in {COMPLETE_RETAINED_CSV}")
    if set(frame.partition.astype(str)) != {"attack_train", "evaluation"}:
        raise RuntimeError(
            f"{COMPLETE_RETAINED_CSV.name} must contain both protocol partitions"
        )
    _assert_protocol_label_balance(frame)
    return frame


def prepare_protocol_split() -> None:
    """Create an immutable category- and label-balanced 50/50-style split."""
    from adversarial_harness.dataset import discover_anomaly_datasets

    split_seed = int(os.environ.get("SPLIT_SEED", "111"))
    evaluation_fraction = float(os.environ.get("EVALUATION_FRACTION", "0.50"))
    protocol = split_protocol()
    policy = label_policy_for(protocol)
    datasets = protocol_datasets()
    discovery_mode = datasets[0] if len(datasets) == 1 else "both"
    if not (0.0 < evaluation_fraction < 1.0):
        raise ValueError("EVALUATION_FRACTION must be between 0 and 1")

    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    protocol_paths = (ATTACK_TRAIN_CSV, EVALUATION_CSV, COMPLETE_RETAINED_CSV)
    existing_protocol_paths = [path for path in protocol_paths if path.is_file()]
    if existing_protocol_paths and len(existing_protocol_paths) != len(protocol_paths):
        raise RuntimeError(
            "Incomplete protocol files. Use a new OUTPUT_BASE or remove the protocol "
            "directory before rebuilding: "
            + ", ".join(path.name for path in existing_protocol_paths)
        )
    if len(existing_protocol_paths) == len(protocol_paths):
        train, evaluation = load_protocol()
        complete = load_complete_retained_protocol()
        stored_datasets = set(pd.concat([train, evaluation]).dataset.astype(str))
        if stored_datasets != set(datasets):
            raise RuntimeError(
                "Existing protocol CSVs use datasets "
                f"{sorted(stored_datasets)}, requested {sorted(datasets)}. "
                "Use a dataset-specific OUTPUT_BASE."
            )
        if set(train.dataset.astype(str)) != set(source_datasets()):
            raise RuntimeError(
                "Existing attack_train_indices.csv does not match SOURCE_DATASETS"
            )
        if set(evaluation.dataset.astype(str)) != set(evaluation_datasets()):
            raise RuntimeError(
                "Existing evaluation_test_indices.csv does not match "
                "EVALUATION_DATASETS"
            )
        if set(complete.dataset.astype(str)) != set(datasets):
            raise RuntimeError(
                "Existing complete_retained_indices.csv does not match protocol datasets"
            )
        expected_train_ids = set(
            complete[
                complete.dataset.astype(str).isin(source_datasets())
                & complete.partition.astype(str).eq("attack_train")
            ].protocol_id
        )
        expected_evaluation_ids = set(
            complete[
                complete.dataset.astype(str).isin(evaluation_datasets())
                & complete.partition.astype(str).eq("evaluation")
            ].protocol_id
        )
        if set(train.protocol_id) != expected_train_ids:
            raise RuntimeError("attack_train_indices.csv disagrees with complete cohort")
        if set(evaluation.protocol_id) != expected_evaluation_ids:
            raise RuntimeError("evaluation_test_indices.csv disagrees with complete cohort")
        stored_policy = set(pd.concat([train, evaluation]).label_balance_policy.astype(str))
        if stored_policy != {policy}:
            raise RuntimeError(
                f"Existing protocol CSVs use {sorted(stored_policy)}, requested "
                f"{policy}. Use a protocol-specific OUTPUT_BASE."
            )
        stored_seed = set(pd.concat([train, evaluation]).split_seed.astype(int))
        stored_fraction = set(pd.concat([train, evaluation]).evaluation_fraction.astype(float))
        if stored_seed != {split_seed} or len(stored_fraction) != 1 or abs(next(iter(stored_fraction)) - evaluation_fraction) > 1e-12:
            raise RuntimeError(
                "Existing protocol CSVs use different split settings. Delete OUTPUT_BASE/protocol "
                "before intentionally rebuilding the benchmark split."
            )
        print(
            f"[reuse split] train={len(train)} evaluation={len(evaluation)} "
            f"complete={len(complete)} overlap=0"
        )
        return

    samples = discover_anomaly_datasets(
        dataset=discovery_mode,
        mvtec_root=str(MVTEC_ROOT) if "mvtec" in datasets else None,
        visa_root=str(VISA_ROOT) if "visa" in datasets else None,
        categories=None,
        max_samples_per_category=None,
        train_normal=False,
    )
    if not samples:
        raise RuntimeError("No MVTec/VisA images were discovered")

    groups, original_sizes = _balanced_category_groups(samples, split_seed, protocol)

    complete_rows = []
    for (dataset, category, label), group in sorted(groups.items()):
        n_eval = min(max(int(round(len(group) * evaluation_fraction)), 1), len(group) - 1)
        evaluation_samples = group[:n_eval]
        train_samples = group[n_eval:]

        for partition, subset in (
            ("attack_train", train_samples),
            ("evaluation", evaluation_samples),
        ):
            for rank, sample in enumerate(subset, start=1):
                complete_rows.append({
                    "protocol_id": sample.protocol_id,
                    "dataset": sample.dataset,
                    "category": sample.category,
                    "defect_type": sample.defect_type,
                    "label": int(sample.label),
                    "partition": partition,
                    "image_relative_path": _relative_path(sample.image_path, sample.dataset),
                    "mask_relative_path": _relative_path(sample.mask_path, sample.dataset) if sample.mask_path else "",
                    "attack_train_rank": rank if partition == "attack_train" else 0,
                    "attack_train_stratum_size": len(train_samples),
                    "evaluation_rank": rank if partition == "evaluation" else 0,
                    "evaluation_stratum_size": len(evaluation_samples),
                    "split_seed": split_seed,
                    "evaluation_fraction": evaluation_fraction,
                    "label_balance_policy": policy,
                    "original_label_stratum_size": original_sizes[
                        (dataset, category, label)
                    ],
                    "balanced_label_stratum_size": len(group),
                })

    complete = pd.DataFrame(complete_rows).sort_values(
        ["dataset", "partition", "category", "label", "protocol_id"]
    ).reset_index(drop=True)
    train = complete[
        complete.partition.eq("attack_train")
        & complete.dataset.isin(source_datasets())
    ].reset_index(drop=True)
    evaluation = complete[
        complete.partition.eq("evaluation")
        & complete.dataset.isin(evaluation_datasets())
    ].reset_index(drop=True)
    overlap = set(train.protocol_id) & set(evaluation.protocol_id)
    if overlap:
        raise RuntimeError("Generated train/evaluation split overlaps")
    _assert_protocol_label_balance(complete)

    train.to_csv(ATTACK_TRAIN_CSV, index=False)
    evaluation.to_csv(EVALUATION_CSV, index=False)
    complete.to_csv(COMPLETE_RETAINED_CSV, index=False)
    label_counts = {
        partition: subset.groupby("label").size().to_dict()
        for partition, subset in (("train", train), ("evaluation", evaluation))
    }
    kept = sum(len(group) for group in groups.values())
    available = sum(original_sizes.values())
    print(
        f"[created split] protocol={protocol} policy={policy} "
        f"train={len(train)} evaluation={len(evaluation)} complete={len(complete)} "
        f"labels={label_counts} "
        f"overlap=0 kept={kept}/{available} images"
    )
    print("Train CSV:", ATTACK_TRAIN_CSV)
    print("Evaluation CSV:", EVALUATION_CSV)
    print("Complete retained CSV:", COMPLETE_RETAINED_CSV)


def bind_discovered_samples(discovered_samples: Sequence):
    train, evaluation = load_protocol()
    frame = pd.concat([train, evaluation], ignore_index=True).sort_values(
        ["dataset", "partition", "category", "label", "protocol_id"]
    ).reset_index(drop=True)
    by_id = {sample.protocol_id: sample for sample in discovered_samples}
    missing = sorted(set(frame.protocol_id) - set(by_id))
    if missing:
        raise RuntimeError(f"CSV images missing under configured roots: {missing[:5]}")
    samples = [by_id[pid] for pid in frame.protocol_id]
    assignments = dict(zip(frame.protocol_id, frame.partition))
    rank_info = frame.set_index("protocol_id")[[
        "attack_train_rank", "attack_train_stratum_size",
        "evaluation_rank", "evaluation_stratum_size",
    ]].to_dict(orient="index")
    return samples, assignments, rank_info, frame


def bind_discovered_samples_from_partition_csvs(
    discovered_samples: Sequence, _attack_train_csv: Path, _evaluation_csv: Path
):
    return bind_discovered_samples(discovered_samples)


def bind_complete_retained_samples(discovered_samples: Sequence):
    """Bind exact fullcross samples from the packaged complete cohort CSV."""

    frame = load_complete_retained_protocol().sort_values(
        ["dataset", "partition", "category", "label", "protocol_id"]
    ).reset_index(drop=True)
    by_id = {sample.protocol_id: sample for sample in discovered_samples}
    missing = sorted(set(frame.protocol_id) - set(by_id))
    if missing:
        raise RuntimeError(f"Complete cohort images missing under roots: {missing[:5]}")
    return [by_id[protocol_id] for protocol_id in frame.protocol_id], frame


def assert_partition_disjoint(assignments: Mapping[str, str]) -> None:
    train_ids = {pid for pid, part in assignments.items() if part == "attack_train"}
    eval_ids = {pid for pid, part in assignments.items() if part == "evaluation"}
    overlap = train_ids & eval_ids
    if overlap:
        raise RuntimeError(f"Partition leakage: {sorted(overlap)[:5]}")


def select_attack_train_fraction(
    samples: Sequence,
    assignments: Mapping[str, str],
    rank_info: Mapping[str, Mapping[str, float]],
    fraction: float,
) -> list:
    if not (0.0 < fraction <= 1.0):
        raise ValueError("ATTACK_TRAIN_FRACTION must be in (0,1]")
    selected = []
    for sample in samples:
        pid = sample.protocol_id
        if assignments.get(pid) != "attack_train":
            continue
        info = rank_info[pid]
        keep = max(1, math.ceil(int(info["attack_train_stratum_size"]) * fraction))
        if int(info["attack_train_rank"]) <= keep:
            selected.append(sample)
    return selected


# --- generation_config.json -------------------------------------------------
# One human-readable record per bundle folder, written before any optimisation
# so that even a crashed run says what it was asked to do. "setup" and
# "hyperparameters" are what a resume must agree with; everything else is
# provenance and is never compared.

GENERATION_CONFIG_NAME = "generation_config.json"
GENERATION_CONFIG_SCHEMA_VERSION = 1
GENERATION_CONFIG_COMPARED_SECTIONS = ("setup", "hyperparameters")
# Module constants that are recorded elsewhere in the file (setup, code) or
# are not settings at all.
_NOT_SETTINGS = frozenset({
    "SETUP_ID", "SETTINGS_TAG", "PROMPT_MODE",
    "REPO_COMMIT", "ANOMALYCLIP_COMMIT",
    "GENERATOR_SCRIPT_SHA256", "ATTACK_CODE_SHA256",
    "IMAGE_CACHE", "MASK_CACHE", "BUNDLE_SCOPES",
})
# Knobs that change how a run executes but not the deltas it produces, so a
# resume may change them: micro-batching is exact by construction and is tuned
# to the GPU. Recorded under "execution", never compared.
EXECUTION_SETTINGS = frozenset({
    "OVERWRITE_EXISTING", "WRITE_BUNDLE_ARCHIVES", "CACHE_INPUTS_IN_RAM",
    "MICRO_BATCH_SIZE", "AUTO_REDUCE_MICRO_BATCH_ON_OOM",
})
# Read through an f-string in adversarial_harness/prompts.py, so the source
# scan below cannot see them.
_DYNAMIC_ENVIRONMENT_NAMES = (
    "LEARNABLE_PROMPT_MVTEC_CHECKPOINT", "LEARNABLE_PROMPT_VISA_CHECKPOINT",
)
_ENVIRONMENT_READ = re.compile(
    r"(?:environ(?:\.get|\.setdefault|\.pop)?\s*[\[(]|getenv\(|bool_env\("
    r"|csv_tuple\(|_dataset_selection\(|_unique_list\(|_csv_env\()\s*"
    r"[\"']([A-Z][A-Z0-9_]*)[\"']"
)
_NOT_PLAIN = object()


def _plain(value):
    """``value`` as JSON data, or ``_NOT_PLAIN`` when it holds anything else.

    Paths, tensors and objects are not settings; they are recorded in the
    data and environment sections instead.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (tuple, list)):
        items = [_plain(item) for item in value]
        return _NOT_PLAIN if any(item is _NOT_PLAIN for item in items) else items
    if isinstance(value, Mapping):
        items = {str(key): _plain(item) for key, item in value.items()}
        return _NOT_PLAIN if any(item is _NOT_PLAIN for item in items.values()) else items
    return _NOT_PLAIN


def _script_constants(script_path: Path) -> list[str]:
    """Every top-level ``UPPER_CASE = ...`` the script itself assigns."""

    tree = ast.parse(Path(script_path).read_text(encoding="utf-8"))
    names = []
    for node in tree.body:
        targets = (
            node.targets if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper() and target.id not in names:
                names.append(target.id)
    return names


def script_settings(
    script_path: Path, namespace: Mapping, overrides: Mapping | None = None
) -> tuple[dict, dict]:
    """``(hyperparameters, execution)`` resolved from the script's own constants.

    Derived from what the script assigns rather than listed by hand, so a new
    setting cannot be left out: it lands in hyperparameters, and therefore in
    the resume check, unless it is named in EXECUTION_SETTINGS.
    """

    values = {**namespace, **(overrides or {})}
    hyperparameters, execution = {}, {}
    for name in _script_constants(script_path):
        if name in _NOT_SETTINGS or name not in values:
            continue
        value = _plain(values[name])
        if value is _NOT_PLAIN:
            continue
        (execution if name in EXECUTION_SETTINGS else hyperparameters)[name.lower()] = value
    return hyperparameters, execution


def environment_record(script_path: Path) -> dict:
    """The raw value of every variable the script and its modules read.

    A whitelist taken from the source, never the whole environment: that can
    hold credentials. Unset variables are recorded as null.
    """

    sources = [
        Path(script_path), PROJECT_ROOT / "common.py", PROJECT_ROOT / "setup_catalog.py",
        *sorted((PROJECT_ROOT / "adversarial_harness").glob("*.py")),
    ]
    names = set(_DYNAMIC_ENVIRONMENT_NAMES)
    for source in sources:
        if source.is_file():
            names.update(_ENVIRONMENT_READ.findall(source.read_text(encoding="utf-8")))
    return {name: os.environ.get(name) for name in sorted(names)}


def git_state(path: Path) -> dict:
    """Commit and dirtiness of the checkout at ``path``; null when not a checkout."""

    def run(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def prompt_training_commit() -> str | None:
    """The object-agnostic prompt-training checkout, where one exists."""

    local = os.environ.get("PROMPT_TRAINING_ROOT", "").strip()
    root = Path(local).expanduser() if local else WORK_DIR / "object-agnostic-prompt-training"
    return git_state(root)["commit"] if root.is_dir() else None


def protocol_data_record(csv_paths: Sequence[Path]) -> dict:
    """The split fingerprint, each partition CSV, and its image counts."""

    digest = hashlib.sha256()
    files = {}
    for path in csv_paths:
        path = Path(path)
        digest.update(path.read_bytes())
        frame = pd.read_csv(path, usecols=["dataset", "label"])
        counts = {}
        for (dataset, label), count in frame.groupby(["dataset", "label"]).size().items():
            counts.setdefault(str(dataset), {})[str(label)] = int(count)
        files[path.name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "image_counts_by_dataset_and_label": counts,
        }
    return {"protocol_split_sha256": digest.hexdigest(), "files": files}


def runtime_record() -> dict:
    return {
        "hostname": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def generation_config_path(bundle_dir: Path) -> Path:
    return Path(bundle_dir) / GENERATION_CONFIG_NAME


def write_generation_config(bundle_dir: Path, payload: Mapping) -> Path:
    """Write ``bundle_dir/generation_config.json`` atomically.

    A reader never sees a half-written file: the JSON goes to a temporary file
    beside it, which then replaces the old one in a single rename.
    """

    path = generation_config_path(bundle_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def read_generation_config(bundle_dir: Path) -> dict | None:
    path = generation_config_path(bundle_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _differing_keys(old, new, prefix: str = "") -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        keys = []
        for key in sorted(set(old) | set(new)):
            keys.extend(_differing_keys(
                old.get(key, _NOT_PLAIN), new.get(key, _NOT_PLAIN), f"{prefix}{key}."
            ))
        return keys
    return [] if old == new else [prefix.rstrip(".")]


def check_generation_config(bundle_dir: Path, payload: Mapping) -> None:
    """Stop when an earlier run in this folder used a different configuration.

    Only setup and hyperparameters are compared. Some of them - the margin
    top-k fraction among them - are not part of the setup ID, so without this
    two different runs could land in the same folder and reuse each other.
    """

    existing = read_generation_config(bundle_dir)
    if existing is None:
        return
    # Round-trip so tuples and lists compare alike, as they were stored.
    current = json.loads(json.dumps(payload))
    differing = [
        f"{section}.{key}" if key else section
        for section in GENERATION_CONFIG_COMPARED_SECTIONS
        for key in _differing_keys(existing.get(section), current.get(section))
    ]
    if differing:
        raise RuntimeError(
            f"{generation_config_path(bundle_dir)} was written by a run with a "
            f"different configuration; differing keys: {', '.join(differing)}. "
            "Use a different output folder, set OVERWRITE_EXISTING=true to "
            "regenerate, or delete that file if the difference is intended."
        )


def start_generation_configs(
    records: Mapping[Path, Mapping], *, overwrite: bool
) -> list[Path]:
    """Check every folder first, then mark each one running.

    Nothing is written when any folder conflicts, so a refused run leaves the
    earlier records exactly as they were. Uncaught exceptions afterwards mark
    every still-running folder failed, then propagate as before.
    """

    if not overwrite:
        for bundle_dir, payload in records.items():
            check_generation_config(bundle_dir, payload)
    started = []
    for bundle_dir, payload in records.items():
        write_generation_config(bundle_dir, {
            **payload,
            "schema_version": GENERATION_CONFIG_SCHEMA_VERSION,
            "status": "running",
            "created_at_utc": _utc_now(),
            "finished_at_utc": None,
            "error": None,
        })
        started.append(Path(bundle_dir))
    _watch_for_failure(started)
    return started


def update_generation_config(bundle_dir: Path, **fields) -> None:
    """Replace top-level fields; a mapping merges into the mapping it replaces."""

    payload = read_generation_config(bundle_dir)
    if payload is None:
        raise FileNotFoundError(generation_config_path(bundle_dir))
    for key, value in fields.items():
        if isinstance(value, Mapping) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    write_generation_config(bundle_dir, payload)


def finish_generation_config(
    bundle_dir: Path, status: str = "completed", error: BaseException | None = None
) -> None:
    update_generation_config(
        bundle_dir,
        status=status,
        finished_at_utc=_utc_now(),
        error=None if error is None else f"{type(error).__name__}: {error}",
    )


def build_generation_payload(
    script_path: Path,
    namespace: Mapping,
    *,
    setup: Mapping,
    attack_config,
    csv_paths: Sequence[Path],
    overrides: Mapping | None = None,
) -> dict:
    """Every section but the status fields, which the writers own.

    ``overrides`` replaces script constants for a snapshot, whose own budget
    differs from the run that produces it.
    """

    from setup_catalog import snapshot_epochs_setting

    script_path = Path(script_path)
    hyperparameters, execution = script_settings(script_path, namespace, overrides)
    # The resolved AttackConfig holds values the scripts pass as literals
    # (temperature, loss weights, feature layers). universal_steps and
    # margin_topk_fraction in it are replaced per condition.
    hyperparameters["attack_config"] = {
        key: value
        for key, value in (
            (key, _plain(value))
            for key, value in dataclasses.asdict(attack_config).items()
        )
        if value is not _NOT_PLAIN
    }
    # Execution, not a hyperparameter: a snapshot budget must stay
    # indistinguishable from a standalone run at that budget.
    execution["snapshot_epochs"] = _plain(snapshot_epochs_setting())
    generator = git_state(PROJECT_ROOT)
    return {
        **runtime_record(),
        "code": {
            "generator_commit": generator["commit"],
            "generator_working_tree_dirty": generator["dirty"],
            "generator_script": script_path.name,
            "generator_script_sha256": sha256_file(script_path),
            "attack_code_sha256": sha256_file(
                PROJECT_ROOT / "adversarial_harness" / "attacks.py"
            ),
            "anomalyclip_loader_commit": git_state(ANOMALYCLIP_ROOT)["commit"],
            "prompt_training_commit": prompt_training_commit(),
        },
        "setup": dict(setup),
        "hyperparameters": hyperparameters,
        "execution": execution,
        "data": protocol_data_record(csv_paths),
        "environment": environment_record(script_path),
    }


def start_snapshot_generation_config(
    bundle_dir: Path, payload: Mapping, *, overwrite: bool
) -> None:
    """The first time this run writes into a snapshot folder, start its record.

    The launcher later replays that folder as an ordinary setup, whose own
    start compares against this record, so the payload must describe the
    snapshot's budget rather than the run that produced it.
    """

    if Path(bundle_dir) not in _WATCHED:
        start_generation_configs({Path(bundle_dir): payload}, overwrite=overwrite)


def finish_running_generation_configs() -> None:
    """Mark every folder this run started and has not finished completed."""

    for bundle_dir in _WATCHED:
        if (read_generation_config(bundle_dir) or {}).get("status") == "running":
            finish_generation_config(bundle_dir)


_WATCHED: list[Path] = []


def mark_failed(error: BaseException) -> None:
    """Mark every watched folder that is still running as failed."""

    for bundle_dir in _WATCHED:
        try:
            if (read_generation_config(bundle_dir) or {}).get("status") == "running":
                finish_generation_config(bundle_dir, "failed", error)
        except Exception as record_error:  # never mask the original failure
            print(f"could not record failure in {bundle_dir}: {record_error}", file=sys.stderr)


def _watch_for_failure(bundle_dirs: Sequence[Path]) -> None:
    """Chain onto sys.excepthook: record the failure, then report it as before.

    The runners are flat scripts, so a hook rather than a try block around the
    whole file. The exception still propagates and the exit status is unchanged.
    """

    for bundle_dir in bundle_dirs:
        if bundle_dir not in _WATCHED:
            _WATCHED.append(bundle_dir)
    if getattr(sys.excepthook, "marks_generation_failed", False):
        return
    previous = sys.excepthook

    def hook(kind, error, traceback):
        mark_failed(error)
        previous(kind, error, traceback)

    hook.marks_generation_failed = True
    sys.excepthook = hook


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "split":
        raise SystemExit("Usage: python common.py split")
    prepare_protocol_split()
