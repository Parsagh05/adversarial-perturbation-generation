#!/usr/bin/env python3
"""Audit completed generation setups without loading attack models."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from adversarial_harness.config import VALID_DIRECTIONS
from adversarial_harness.prompts import (
    FROZEN_PROMPT_AGGREGATION,
    LEARNABLE_PROMPT_AGGREGATION,
    frozen_ensemble_sha256,
)
from setup_catalog import (
    SETUPS,
    effective_setup_id,
    scope_output_path,
    settings_tag,
    full_data_cross_setting,
    checkpoint_selection_setting,
    margin_hinge_setting,
    momentum_decay_setting,
    per_image_cohort_setting,
    split_protocol_setting,
    step_size_schedule_setting,
)


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    return tuple(
        value.strip() for value in os.environ.get(name, default).split(",")
        if value.strip()
    )


ROOT = Path(os.environ["OUTPUT_BASE"]).expanduser().resolve()
RUN_SETUPS = os.environ.get("RUN_SETUPS", "all")
PROMPT_SETUP = os.environ.get("PROMPT_SETUP", "both").strip().lower()
SMOKE = os.environ.get("SMOKE_TEST", "false").lower() in {
    "1", "true", "yes", "on"
}
SMOKE_EPOCHS = float(os.environ.get("SMOKE_EPOCHS", "0.02"))
EXPECTED_ATTACK_SEED = int(os.environ.get("ATTACK_SEED", "111"))
ATTACK_TRAIN_FRACTION = float(os.environ.get("ATTACK_TRAIN_FRACTION", "1.0"))
SPLIT_PROTOCOL = split_protocol_setting()
FULL_DATA_CROSS = full_data_cross_setting()
STEP_SIZE_SCHEDULE = step_size_schedule_setting()
MARGIN_HINGE = margin_hinge_setting()
MOMENTUM_DECAY = momentum_decay_setting()
CHECKPOINT_SELECTION = checkpoint_selection_setting()
PER_IMAGE_ATTACK_COHORT = per_image_cohort_setting()
RANDOM_BASELINE = os.environ.get("RANDOM_BASELINE", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
CROSS_DATASET_FULL_SOURCE = FULL_DATA_CROSS
# Read the same way the runners read it, so auditing a run that generated one
# direction checks for that direction instead of reporting the other missing.
EXPECTED_DIRECTIONS = _csv_env("DIRECTIONS", ",".join(VALID_DIRECTIONS))
_unknown_directions = sorted(set(EXPECTED_DIRECTIONS) - set(VALID_DIRECTIONS))
if not EXPECTED_DIRECTIONS or _unknown_directions:
    raise ValueError(
        f"DIRECTIONS must be a non-empty subset of {VALID_DIRECTIONS}, "
        f"got {EXPECTED_DIRECTIONS}"
    )

# The manifest calls the per-dataset scope "dataset", but the tree names every
# directory after the runner that fills it, so the two spellings are mapped
# rather than reconciled.
SCOPES = {
    "dataset": os.environ.get("RUN_PER_DATASET", "true"),
    "per_category": os.environ.get("RUN_PER_CATEGORY", "true"),
    "cross_dataset": os.environ.get("RUN_CROSS_DATASET", "true"),
    "per_image": os.environ.get("RUN_PER_IMAGE", "true"),
}
SCOPE_DIRECTORIES_BY_SCOPE = {
    "dataset": "per_dataset",
    "cross_dataset": "cross_dataset",
    "per_category": "per_category",
    "per_image": "per_image",
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


def audit_protocol(settings_root: Path) -> None:
    protocol = settings_root / "protocol"
    train = pd.read_csv(protocol / "attack_train_indices.csv")
    evaluation = pd.read_csv(protocol / "evaluation_test_indices.csv")
    complete = pd.read_csv(protocol / "complete_retained_indices.csv")
    if set(train.protocol_id) & set(evaluation.protocol_id):
        raise RuntimeError(f"Train/evaluation leakage in {settings_root.name}")
    if complete.protocol_id.duplicated().any():
        raise RuntimeError(f"Duplicate complete-cohort IDs in {settings_root.name}")
    if set(complete.partition.astype(str)) != {"attack_train", "evaluation"}:
        raise RuntimeError(f"Incomplete retained cohort in {settings_root.name}")
    expected_train = complete[
        complete.dataset.astype(str).isin(set(train.dataset.astype(str)))
        & complete.partition.astype(str).eq("attack_train")
    ]
    expected_evaluation = complete[
        complete.dataset.astype(str).isin(set(evaluation.dataset.astype(str)))
        & complete.partition.astype(str).eq("evaluation")
    ]
    if set(train.protocol_id.astype(str)) != set(expected_train.protocol_id.astype(str)):
        raise RuntimeError(f"Attack-train CSV disagrees with complete cohort")
    if set(evaluation.protocol_id.astype(str)) != set(
        expected_evaluation.protocol_id.astype(str)
    ):
        raise RuntimeError(f"Evaluation CSV disagrees with complete cohort")
    counts = complete.groupby(
        ["dataset", "category", "partition", "label"]
    ).size().unstack(fill_value=0)
    if set(counts.columns) != {0, 1} or (counts[[0, 1]] == 0).any().any():
        raise RuntimeError(f"A protocol stratum is missing a label in {settings_root.name}")
    if SPLIT_PROTOCOL == "balanced" and not counts[0].eq(counts[1]).all():
        raise RuntimeError(f"Unbalanced protocol in {settings_root.name}")


def audit_dataset_cohort_routing(
    setup_root: Path, scope: str, manifest: pd.DataFrame, manifest_path: Path
) -> None:
    """Verify dataset-scope cohorts from explicit manifest provenance."""

    if scope == "dataset":
        if set(manifest.training_source.astype(str)) != {"attack_train_partition"}:
            raise RuntimeError(f"per_dataset routing changed in {manifest_path}")
        return
    if scope != "cross_dataset":
        return

    required = {
        "full_data_cross",
        "cross_data_mode",
        "source_partition_policy",
        "target_partition_policy",
        "training_source",
        "source_target_id_overlap_count",
    }
    if not required.issubset(manifest.columns):
        raise RuntimeError(f"Missing cross-dataset cohort provenance in {manifest_path}")

    recorded_flags = set(
        manifest.full_data_cross.astype(str).str.strip().str.lower()
    )
    if recorded_flags != {str(FULL_DATA_CROSS).lower()}:
        raise RuntimeError(f"Wrong FULL_DATA_CROSS value in {manifest_path}")

    if CROSS_DATASET_FULL_SOURCE:
        expected_mode = "fullcross"
        expected_training_source = "complete_source_dataset"
        expected_source_policy = "all"
        expected_target_policy = "all"
        expected_target_partitions = {"attack_train", "evaluation"}
    else:
        expected_mode = "halfcross"
        expected_training_source = "attack_train_partition"
        expected_source_policy = "attack_train"
        expected_target_policy = "evaluation"
        expected_target_partitions = {"evaluation"}

    expected_values = {
        "cross_data_mode": expected_mode,
        "training_source": expected_training_source,
        "source_partition_policy": expected_source_policy,
        "target_partition_policy": expected_target_policy,
    }
    for column, expected in expected_values.items():
        if set(manifest[column].astype(str)) != {expected}:
            raise RuntimeError(
                f"Wrong {column} in {manifest_path}: expected {expected}"
            )
    if set(manifest.source_target_id_overlap_count.astype(int)) != {0}:
        raise RuntimeError(f"Cross-dataset source/target leakage in {manifest_path}")

    protocol = pd.read_csv(
        setup_root / "protocol" / "complete_retained_indices.csv",
        dtype={"protocol_id": str},
    )

    for row in manifest.itertuples(index=False):
        source = protocol[
            (protocol.dataset.astype(str) == str(row.source_dataset))
            & (protocol.label.astype(int) == int(row.source_label))
        ]
        target = protocol[
            (protocol.dataset.astype(str) == str(row.target_dataset))
            & (protocol.label.astype(int) == int(row.source_label))
            & protocol.partition.astype(str).isin(expected_target_partitions)
        ]
        if expected_source_policy != "all":
            source = source[source.partition.astype(str) == "attack_train"]
            fraction = float(row.attack_train_fraction)
            source = source[
                source.attack_train_rank.astype(int)
                <= np.maximum(
                    1,
                    np.ceil(
                        source.attack_train_stratum_size.astype(int) * fraction
                    ),
                )
            ]
            expected_source_count = len(source)
        else:
            expected_source_count = len(source)
        expected_target_count = (
            len(target)
        )
        if expected_source_count != int(row.attack_train_image_count):
            raise RuntimeError(
                f"Wrong source training count in {manifest_path}: expected "
                f"{expected_source_count}, found {row.attack_train_image_count}"
            )
        if expected_target_count != int(row.evaluation_attacked_image_count):
            raise RuntimeError(
                f"Wrong attacked target count in {manifest_path}: expected "
                f"{expected_target_count}, found {row.evaluation_attacked_image_count}"
            )
        if str(row.source_dataset) == str(row.target_dataset):
            raise RuntimeError(f"Cross-dataset row has the same source and target")
        if set(source.protocol_id.astype(str)) & set(target.protocol_id.astype(str)):
            raise RuntimeError(f"Cross-dataset source/target leakage in {manifest_path}")


def audit_scope(
    settings_root: Path,
    scope: str,
    bundle: Path,
    expected_epochs: dict[str, float],
    expected_epsilon: float,
    expected_loss_formulation: str,
    expected_prompt_mode: str,
    expected_setup_id: str,
) -> set[str]:
    manifest_path = bundle / "attack_manifest.csv"
    diagnostics_path = bundle / "optimization_diagnostics.csv"
    for required in (
        manifest_path,
        diagnostics_path,
        bundle / "attack_train_indices.csv",
        bundle / "evaluation_test_indices.csv",
        bundle / "complete_retained_indices.csv",
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Incomplete {scope} bundle: {required}")

    manifest = pd.read_csv(manifest_path)
    diagnostics = pd.read_csv(diagnostics_path)
    if manifest.empty or diagnostics.empty or set(manifest.scope) != {scope}:
        raise RuntimeError(f"Invalid {scope} tables in {settings_root.name}")
    # The step count is derived from the epoch budget and each condition's own
    # training-set size, so one manifest legitimately holds several values; the
    # budget is what must match.
    # The tree no longer spells the setup ID, so the manifest is the only
    # place the evaluator can read it from. A missing column here becomes
    # "No attack conditions matched the configuration" downstream, after the
    # whole generation cost has been paid.
    if "setup_id" not in manifest.columns:
        raise RuntimeError(f"Missing setup_id column in {manifest_path}")
    recorded_ids = set(manifest.setup_id.astype(str))
    if recorded_ids != {expected_setup_id}:
        raise RuntimeError(
            f"Wrong setup_id in {manifest_path}: expected {expected_setup_id}, "
            f"found {sorted(recorded_ids)}"
        )
    if "optimization_epochs" not in manifest.columns:
        raise RuntimeError(f"Missing optimization_epochs in {manifest_path}")
    recorded = {round(float(value), 6) for value in manifest.optimization_epochs}
    if recorded != {round(float(expected_epochs[scope]), 6)}:
        raise RuntimeError(
            f"Wrong epoch budget in {manifest_path}: scope {scope} expects "
            f"{expected_epochs[scope]}, found {sorted(recorded)}"
        )
    if (manifest.optimization_steps.astype(int) < 1).any():
        raise RuntimeError(f"Non-positive derived steps in {manifest_path}")
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
    required_replicate_columns = {"seed", "run_seed", "artifact_sha256"}
    if not required_replicate_columns.issubset(manifest.columns):
        raise RuntimeError(f"Missing seed/checksum provenance in {manifest_path}")
    if set(manifest.seed.astype(int)) != {EXPECTED_ATTACK_SEED}:
        raise RuntimeError(f"Wrong attack seed in {manifest_path}")
    if manifest.run_seed.isna().any():
        raise RuntimeError(f"Missing condition seed in {manifest_path}")
    required_ensemble_columns = {"prompt_aggregation", "prompt_ensemble_sha256"}
    if not required_ensemble_columns.issubset(manifest.columns):
        raise RuntimeError(f"Missing prompt-ensemble provenance in {manifest_path}")
    if manifest.prompt_ensemble_sha256.fillna("").str.len().ne(64).any():
        raise RuntimeError(f"Invalid prompt-ensemble hash in {manifest_path}")
    expected_aggregation = (
        FROZEN_PROMPT_AGGREGATION
        if expected_prompt_mode == "frozen_winclip"
        else LEARNABLE_PROMPT_AGGREGATION
    )
    if set(manifest.prompt_aggregation.astype(str)) != {expected_aggregation}:
        raise RuntimeError(f"Wrong prompt aggregation in {manifest_path}")
    if expected_prompt_mode == "frozen_winclip" and set(
        manifest.prompt_ensemble_sha256.astype(str)
    ) != {frozen_ensemble_sha256()}:
        raise RuntimeError(f"Wrong frozen WinCLIP ensemble in {manifest_path}")
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

    if "step_size_schedule" in manifest.columns:
        recorded_schedules = set(manifest.step_size_schedule.astype(str))
        if recorded_schedules != {STEP_SIZE_SCHEDULE}:
            raise RuntimeError(
                f"Wrong step-size schedule in {manifest_path}: expected "
                f"{STEP_SIZE_SCHEDULE}, found {sorted(recorded_schedules)}"
            )
    if "split_protocol" not in manifest.columns:
        raise RuntimeError(f"Missing split-protocol provenance in {manifest_path}")
    if set(manifest.split_protocol.astype(str)) != {SPLIT_PROTOCOL}:
        raise RuntimeError(
            f"Wrong split protocol in {manifest_path}: expected {SPLIT_PROTOCOL}"
        )
    audit_dataset_cohort_routing(settings_root, scope, manifest, manifest_path)

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
        found_directions = set(by_direction.index)
        if found_directions != set(EXPECTED_DIRECTIONS):
            raise RuntimeError(
                f"Expected directions {sorted(EXPECTED_DIRECTIONS)}, found "
                f"{sorted(found_directions)} in {manifest_path}"
            )
        # The two directions attack opposite labels: normal_to_abnormal fits
        # normal images, abnormal_to_normal fits abnormal ones. Only balanced
        # downsamples a category to equal label counts, so only there do the
        # two directions see the same number of images. Under full the counts
        # differ by the category's natural class ratio, by construction.
        # audit_protocol checks the label structure itself for both protocols.
        if SPLIT_PROTOCOL == "balanced":
            if scope != "per_image" and by_direction.attack_train_image_count.nunique() != 1:
                raise RuntimeError(
                    f"Direction train counts differ under balanced in "
                    f"{manifest_path}: "
                    f"{by_direction.attack_train_image_count.to_dict()}"
                )
            if by_direction.evaluation_attacked_image_count.nunique() != 1:
                raise RuntimeError(
                    f"Direction evaluation counts differ under balanced in "
                    f"{manifest_path}: "
                    f"{by_direction.evaluation_attacked_image_count.to_dict()}"
                )
        if scope == "per_image" and set(by_direction.attack_train_image_count) != {0}:
            raise RuntimeError(f"Per-image attacks must use zero train images: {manifest_path}")
    return set(manifest.protocol_split_sha256.astype(str))


def main() -> None:
    protocol_hashes: set[str] = set()
    for setup_id in selected_setups():
        setup = SETUPS[setup_id]
        # cross_dataset delivers the per-dataset delta, so it shares its count.
        expected_epochs = {
            "dataset": SMOKE_EPOCHS if SMOKE else setup.epochs,
            "cross_dataset": SMOKE_EPOCHS if SMOKE else setup.cross_epochs,
            "per_category": SMOKE_EPOCHS if SMOKE else setup.category_epochs,
            "per_image": SMOKE_EPOCHS if SMOKE else setup.image_epochs,
        }
        # Same derivation as train.sh, so the audit looks where the run wrote.
        effective_id = effective_setup_id(
            setup, SMOKE_EPOCHS if SMOKE else None, ATTACK_TRAIN_FRACTION,
            SPLIT_PROTOCOL, FULL_DATA_CROSS, STEP_SIZE_SCHEDULE, MARGIN_HINGE,
            MOMENTUM_DECAY, CHECKPOINT_SELECTION, PER_IMAGE_ATTACK_COHORT,
            RANDOM_BASELINE,
        )
        expected_cross_tag = "_fullcross" if FULL_DATA_CROSS else "_halfcross"
        other_cross_tag = "_halfcross" if FULL_DATA_CROSS else "_fullcross"
        if expected_cross_tag not in effective_id or other_cross_tag in effective_id:
            raise RuntimeError(
                f"Setup ID does not encode cross cohort mode: {effective_id}"
            )
        # settings / scope / epochs / prompt family: the split is shared by
        # every scope and budget beneath one settings directory.
        settings = settings_tag(
            setup.epsilon_label, setup.loss_formulation, ATTACK_TRAIN_FRACTION,
            SPLIT_PROTOCOL, FULL_DATA_CROSS, STEP_SIZE_SCHEDULE, MARGIN_HINGE,
            MOMENTUM_DECAY, CHECKPOINT_SELECTION, PER_IMAGE_ATTACK_COHORT,
            RANDOM_BASELINE,
        )
        budget = (
            (SMOKE_EPOCHS,) * 4 if SMOKE else
            (setup.epochs, setup.cross_epochs, setup.category_epochs,
             setup.image_epochs)
        )
        settings_root = ROOT / "setups" / settings
        audit_protocol(settings_root)
        for scope, flag in SCOPES.items():
            if enabled(flag):
                bundle = ROOT / "setups" / scope_output_path(
                    SCOPE_DIRECTORIES_BY_SCOPE[scope], budget,
                    setup.prompt_mode, settings,
                )
                protocol_hashes.update(
                    audit_scope(
                        settings_root,
                        scope,
                        bundle,
                        expected_epochs,
                        setup.epsilon,
                        setup.loss_formulation,
                        setup.prompt_mode,
                        effective_id,
                    )
                )
    if len(protocol_hashes) != 1:
        raise RuntimeError(
            f"Selected outputs do not share one protocol hash: {sorted(protocol_hashes)}"
        )
    print("GENERATION AUDIT PASSED")


if __name__ == "__main__":
    main()
