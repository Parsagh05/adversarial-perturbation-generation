#!/usr/bin/env python3
"""Shared paths, split handling, and small utilities for all attack modes."""
from __future__ import annotations

import gc
import hashlib
import math
import os
import random
import sys
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
LABEL_BALANCE_POLICY = "per_dataset_category_equal_labels_v1"

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


def generation_datasets() -> tuple[str, ...]:
    datasets = csv_tuple("GENERATION_DATASETS", "mvtec,visa")
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("GENERATION_DATASETS must contain unique dataset names")
    unknown = sorted(set(datasets) - {"mvtec", "visa"})
    if unknown:
        raise ValueError(f"Unknown GENERATION_DATASETS values: {unknown}")
    return datasets


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
    for path in (ATTACK_TRAIN_CSV, EVALUATION_CSV):
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
    policies = set(frame["label_balance_policy"].astype(str))
    if policies != {LABEL_BALANCE_POLICY}:
        raise RuntimeError(
            "Protocol CSVs use an unsupported label-balancing policy: "
            f"{sorted(policies)}"
        )
    counts = frame.groupby(
        ["dataset", "category", "partition", "label"]
    ).size().unstack(fill_value=0)
    if set(counts.columns) != {0, 1}:
        raise RuntimeError("Every protocol stratum must contain labels 0 and 1")
    unequal = counts[counts[0] != counts[1]]
    if not unequal.empty:
        raise RuntimeError(
            "Protocol is not label-balanced within dataset/category/partition: "
            f"{list(unequal.index[:5])}"
        )


def _balanced_category_groups(samples: Sequence, split_seed: int):
    """Return equally sized, deterministically shuffled label groups per category."""

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
        balanced_size = min(len(normal), len(anomalous))
        if balanced_size < 2:
            raise RuntimeError(
                f"Need at least two images per label in {dataset}/{category}"
            )
        for label, group in ((0, normal), (1, anomalous)):
            shuffled = sorted(group, key=lambda sample: sample.protocol_id)
            rng = random.Random(_stable_seed(split_seed, dataset, category, label))
            rng.shuffle(shuffled)
            key = (dataset, category, label)
            balanced[key] = shuffled[:balanced_size]
            original_sizes[key] = len(group)
    return balanced, original_sizes


def load_protocol() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = _validate_partition_frame(ATTACK_TRAIN_CSV, "attack_train")
    evaluation = _validate_partition_frame(EVALUATION_CSV, "evaluation")
    overlap = set(train.protocol_id) & set(evaluation.protocol_id)
    if overlap:
        raise RuntimeError(f"Train/evaluation overlap detected: {sorted(overlap)[:5]}")
    _assert_protocol_label_balance(pd.concat([train, evaluation], ignore_index=True))
    return train, evaluation


def prepare_protocol_split() -> None:
    """Create an immutable category- and label-balanced 50/50-style split."""
    from adversarial_harness.dataset import discover_anomaly_datasets

    split_seed = int(os.environ.get("SPLIT_SEED", "111"))
    evaluation_fraction = float(os.environ.get("EVALUATION_FRACTION", "0.50"))
    datasets = generation_datasets()
    discovery_mode = datasets[0] if len(datasets) == 1 else "both"
    if not (0.0 < evaluation_fraction < 1.0):
        raise ValueError("EVALUATION_FRACTION must be between 0 and 1")

    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    if ATTACK_TRAIN_CSV.is_file() and EVALUATION_CSV.is_file():
        train, evaluation = load_protocol()
        stored_datasets = set(pd.concat([train, evaluation]).dataset.astype(str))
        if stored_datasets != set(datasets):
            raise RuntimeError(
                "Existing protocol CSVs use datasets "
                f"{sorted(stored_datasets)}, requested {sorted(datasets)}. "
                "Use a dataset-specific OUTPUT_BASE."
            )
        stored_seed = set(pd.concat([train, evaluation]).split_seed.astype(int))
        stored_fraction = set(pd.concat([train, evaluation]).evaluation_fraction.astype(float))
        if stored_seed != {split_seed} or len(stored_fraction) != 1 or abs(next(iter(stored_fraction)) - evaluation_fraction) > 1e-12:
            raise RuntimeError(
                "Existing protocol CSVs use different split settings. Delete OUTPUT_BASE/protocol "
                "before intentionally rebuilding the benchmark split."
            )
        print(f"[reuse split] train={len(train)} evaluation={len(evaluation)} overlap=0")
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

    groups, original_sizes = _balanced_category_groups(samples, split_seed)

    rows = []
    for (dataset, category, label), group in sorted(groups.items()):
        n_eval = min(max(int(round(len(group) * evaluation_fraction)), 1), len(group) - 1)
        evaluation_samples = group[:n_eval]
        train_samples = group[n_eval:]

        for partition, subset in (("attack_train", train_samples), ("evaluation", evaluation_samples)):
            for rank, sample in enumerate(subset, start=1):
                rows.append({
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
                    "label_balance_policy": LABEL_BALANCE_POLICY,
                    "original_label_stratum_size": original_sizes[
                        (dataset, category, label)
                    ],
                    "balanced_label_stratum_size": len(group),
                })

    frame = pd.DataFrame(rows).sort_values(
        ["dataset", "partition", "category", "label", "protocol_id"]
    ).reset_index(drop=True)
    train = frame[frame.partition.eq("attack_train")].reset_index(drop=True)
    evaluation = frame[frame.partition.eq("evaluation")].reset_index(drop=True)
    overlap = set(train.protocol_id) & set(evaluation.protocol_id)
    if overlap:
        raise RuntimeError("Generated train/evaluation split overlaps")
    _assert_protocol_label_balance(frame)

    train.to_csv(ATTACK_TRAIN_CSV, index=False)
    evaluation.to_csv(EVALUATION_CSV, index=False)
    label_counts = {
        partition: subset.groupby("label").size().to_dict()
        for partition, subset in (("train", train), ("evaluation", evaluation))
    }
    print(
        f"[created balanced split] train={len(train)} evaluation={len(evaluation)} "
        f"labels={label_counts} overlap=0"
    )
    print("Train CSV:", ATTACK_TRAIN_CSV)
    print("Evaluation CSV:", EVALUATION_CSV)


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


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "split":
        raise SystemExit("Usage: python common.py split")
    prepare_protocol_split()
