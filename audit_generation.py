#!/usr/bin/env python3
"""Audit completed generation setups without loading attack models."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from setup_catalog import SETUPS, effective_setup_id


ROOT = Path(os.environ["OUTPUT_BASE"]).expanduser().resolve()
RUN_SETUPS = os.environ.get("RUN_SETUPS", "all")
PROMPT_SETUP = os.environ.get("PROMPT_SETUP", "both").strip().lower()
SMOKE = os.environ.get("SMOKE_TEST", "false").lower() in {
    "1", "true", "yes", "on"
}
SMOKE_STEPS = int(os.environ.get("SMOKE_STEPS", "2"))
ATTACK_TRAIN_FRACTION = float(os.environ.get("ATTACK_TRAIN_FRACTION", "1.0"))

SCOPES = {
    "dataset": (
        os.environ.get("RUN_PER_DATASET", "true"),
        "canonical_clip_per_dataset",
    ),
    "per_category": (
        os.environ.get("RUN_PER_CATEGORY", "true"),
        "canonical_clip_per_category",
    ),
    "cross_dataset": (
        os.environ.get("RUN_CROSS_DATASET", "true"),
        "canonical_clip_cross_dataset",
    ),
    "per_image": (
        os.environ.get("RUN_PER_IMAGE", "true"),
        "canonical_clip_per_image",
    ),
}


def enabled(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def selected_setups() -> list[str]:
    requested_ids = (
        set(SETUPS)
        if RUN_SETUPS == "all"
        else {value.strip() for value in RUN_SETUPS.split(",") if value.strip()}
    )
    if PROMPT_SETUP not in {"frozen", "learnable", "both"}:
        raise ValueError(f"Unknown PROMPT_SETUP: {PROMPT_SETUP}")
    expected_modes = {
        "frozen": {"frozen_winclip"},
        "learnable": {"learnable_object_agnostic"},
        "both": {"frozen_winclip", "learnable_object_agnostic"},
    }[PROMPT_SETUP]
    return [
        setup_id
        for setup_id, setup in SETUPS.items()
        if setup.prompt_mode in expected_modes
        and (
            setup_id in requested_ids
            or setup_id.removesuffix("_learnable_prompt") in requested_ids
        )
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_protocol(setup_root: Path) -> None:
    protocol = setup_root / "protocol"
    train = pd.read_csv(protocol / "attack_train_indices.csv")
    evaluation = pd.read_csv(protocol / "evaluation_test_indices.csv")
    if set(train.protocol_id) & set(evaluation.protocol_id):
        raise RuntimeError(f"Train/evaluation leakage in {setup_root.name}")
    frame = pd.concat([train, evaluation], ignore_index=True)
    counts = frame.groupby(
        ["dataset", "category", "partition", "label"]
    ).size().unstack(fill_value=0)
    if set(counts.columns) != {0, 1} or not counts[0].eq(counts[1]).all():
        raise RuntimeError(f"Unbalanced protocol in {setup_root.name}")


def audit_scope(
    setup_root: Path,
    scope: str,
    bundle_name: str,
    expected_steps: int,
    expected_epsilon: float,
    expected_loss_formulation: str,
    expected_prompt_mode: str,
) -> set[str]:
    bundle = setup_root / bundle_name
    manifest_path = bundle / "attack_manifest.csv"
    diagnostics_path = bundle / "optimization_diagnostics.csv"
    for required in (
        manifest_path,
        diagnostics_path,
        bundle / "attack_train_indices.csv",
        bundle / "evaluation_test_indices.csv",
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Incomplete {scope} bundle: {required}")

    manifest = pd.read_csv(manifest_path)
    diagnostics = pd.read_csv(diagnostics_path)
    if manifest.empty or diagnostics.empty or set(manifest.scope) != {scope}:
        raise RuntimeError(f"Invalid {scope} tables in {setup_root.name}")
    if set(manifest.optimization_steps.astype(int)) != {expected_steps}:
        raise RuntimeError(f"Wrong steps in {manifest_path}")
    if not manifest.epsilon.astype(float).map(
        lambda value: abs(value - expected_epsilon) <= 1e-12
    ).all():
        raise RuntimeError(f"Wrong epsilon in {manifest_path}")
    formulations = (
        manifest.loss_formulation.fillna("ce_focal_dice").astype(str)
        if "loss_formulation" in manifest.columns
        else pd.Series("ce_focal_dice", index=manifest.index)
    )
    if set(formulations) != {expected_loss_formulation}:
        raise RuntimeError(f"Wrong loss formulation in {manifest_path}")
    prompt_modes = (
        manifest.prompt_mode.fillna("frozen_winclip").astype(str)
        if "prompt_mode" in manifest.columns
        else pd.Series("frozen_winclip", index=manifest.index)
    )
    if set(prompt_modes) != {expected_prompt_mode}:
        raise RuntimeError(f"Wrong prompt mode in {manifest_path}")
    if expected_prompt_mode == "learnable_object_agnostic":
        required_prompt_columns = {
            "prompt_checkpoint_sha256",
            "prompt_checkpoint_dataset",
            "prompt_checkpoint_epoch",
            "prompt_n_ctx",
        }
        if not required_prompt_columns.issubset(manifest.columns):
            raise RuntimeError(f"Missing prompt provenance in {manifest_path}")
        if manifest.prompt_checkpoint_sha256.fillna("").str.len().ne(64).any():
            raise RuntimeError(f"Invalid prompt checkpoint hash in {manifest_path}")
        if not manifest.apply(
            lambda row: str(row.prompt_checkpoint_dataset).lower()
            == str(row.source_dataset).lower(),
            axis=1,
        ).all():
            raise RuntimeError(f"Prompt/source dataset mismatch in {manifest_path}")

    numeric_columns = (
        "initial_total_loss",
        "final_total_loss",
        "total_loss_reduction",
    )
    for column in numeric_columns:
        values = pd.to_numeric(diagnostics[column], errors="coerce")
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite {column} values in {diagnostics_path}")

    passed = diagnostics.convergence_check_passed.astype(str).str.lower().eq("true")
    if not passed.all():
        message = (
            f"{int((~passed).sum())} convergence checks did not pass in "
            f"{diagnostics_path}"
        )
        if SMOKE:
            print(f"SMOKE NOTE: {message}; expected with very few test steps.")
        else:
            raise RuntimeError(message)

    for row in manifest.itertuples(index=False):
        artifact = bundle / Path(str(row.noise_file))
        if not artifact.is_file():
            raise FileNotFoundError(f"Missing manifest artifact: {artifact}")
        if sha256(artifact) != str(row.artifact_sha256):
            raise RuntimeError(f"Checksum mismatch: {artifact}")

    group_columns = [
        column for column in (
            "scope", "source_dataset", "target_dataset", "category",
            "loss_formulation", "prompt_mode", "loss_mode", "attack_train_fraction",
        ) if column in manifest.columns
    ]
    for _, group in manifest.groupby(group_columns, dropna=False):
        by_direction = group.groupby("direction")[
            ["attack_train_image_count", "evaluation_attacked_image_count"]
        ].first()
        if set(by_direction.index) != {"normal_to_abnormal", "abnormal_to_normal"}:
            raise RuntimeError(f"Missing direction in {manifest_path}")
        if scope != "per_image" and by_direction.attack_train_image_count.nunique() != 1:
            raise RuntimeError(f"Direction train counts differ in {manifest_path}")
        if by_direction.evaluation_attacked_image_count.nunique() != 1:
            raise RuntimeError(f"Direction evaluation counts differ in {manifest_path}")
        if scope == "per_image" and set(by_direction.attack_train_image_count) != {0}:
            raise RuntimeError(f"Per-image attacks must use zero train images: {manifest_path}")
    return set(manifest.protocol_split_sha256.astype(str))


def main() -> None:
    protocol_hashes: set[str] = set()
    for setup_id in selected_setups():
        setup = SETUPS[setup_id]
        expected_steps = SMOKE_STEPS if SMOKE else setup.steps
        # Same derivation as train.sh, so the audit looks where the run wrote.
        effective_id = effective_setup_id(
            setup, expected_steps, ATTACK_TRAIN_FRACTION
        )
        prompt_folder = (
            "frozen_prompt"
            if setup.prompt_mode == "frozen_winclip"
            else "learnable_prompt"
        )
        setup_root = ROOT / "setups" / prompt_folder / effective_id
        audit_protocol(setup_root)
        for scope, (flag, bundle_name) in SCOPES.items():
            if enabled(flag):
                protocol_hashes.update(
                    audit_scope(
                        setup_root,
                        scope,
                        bundle_name,
                        expected_steps,
                        setup.epsilon,
                        setup.loss_formulation,
                        setup.prompt_mode,
                    )
                )
    if len(protocol_hashes) != 1:
        raise RuntimeError(
            f"Selected outputs do not share one protocol hash: {sorted(protocol_hashes)}"
        )
    print("GENERATION AUDIT PASSED")


if __name__ == "__main__":
    main()
