#!/usr/bin/env python3
"""Resolve the learnable-prompt checkpoint this run needs, training it if absent.

Prompts fitted under one split protocol or attack-train fraction saw a
different set of images than a run using another. Pairing them produces no
error on either side -- the prompts have simply seen part of what this run
holds out for evaluation. The previous guard only checked that a file existed,
so a checkpoint left over from a `balanced` run was silently accepted by a
`full` one.

This resolves the checkpoint the current settings require, checks any existing
one actually describes those settings, and invokes the prompt-training
pipeline when it does not. The attack pipeline's own protocol CSV is handed
over as the training manifest, so the cohort is the one the perturbations are
optimized on by construction rather than by two implementations agreeing.

Prints `export NAME='path'` lines on stdout for the launcher to eval; all
progress goes to stderr.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

from setup_catalog import _fraction_tag

# Checkpoints written before the training pipeline recorded its split predate
# the `full` protocol and the fraction sweep, so they can only be balanced
# runs over the complete cohort.
LEGACY_PROTOCOL = "balanced"
LEGACY_FRACTION = 1.0
PLACEHOLDER_PREFIX = "/ABSOLUTE/PATH/TO/"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def cohort_directory(split_protocol: str, attack_train_fraction: float) -> str:
    """``full`` at 25% -> ``full_train25``; a full ``balanced`` stays bare.

    Mirrors ``DataConfig.cohort_directory`` in the training pipeline, which in
    turn mirrors this repository's setup IDs, so a prompt directory and the
    setups that load it carry the same suffix.
    """

    tag = _fraction_tag(attack_train_fraction)
    return f"{split_protocol}_{tag}" if tag else split_protocol


def checkpoint_path(
    output_root: Path,
    split_protocol: str,
    attack_train_fraction: float,
    dataset: str,
    epochs: int,
) -> Path:
    """Where the training pipeline writes the checkpoint for these settings."""

    cohort = cohort_directory(split_protocol, attack_train_fraction)
    return output_root / cohort / dataset / f"prompts_epoch{epochs}.pt"


def _load_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 2.0 compatibility.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Prompt checkpoint is not a mapping: {path}")
    return payload


def checkpoint_mismatch(
    payload: Mapping[str, Any],
    *,
    dataset: str,
    split_protocol: str,
    attack_train_fraction: float,
    seed: int,
    epochs: int,
) -> str:
    """Return why this checkpoint does not describe the requested run, or ""."""

    prompt_config = payload.get("prompt_config")
    if not isinstance(prompt_config, Mapping):
        return "prompt_config is missing"
    training_config = payload.get("training_config")
    if not isinstance(training_config, Mapping):
        return "training_config is missing"

    found_dataset = str(payload.get("dataset", "")).lower()
    if found_dataset != dataset.lower():
        return f"trained on {found_dataset!r}, this run needs {dataset!r}"

    found_protocol = str(prompt_config.get("split_protocol", LEGACY_PROTOCOL))
    if found_protocol != split_protocol:
        return f"split protocol {found_protocol!r}, this run uses {split_protocol!r}"

    found_fraction = float(prompt_config.get("attack_train_fraction", LEGACY_FRACTION))
    if abs(found_fraction - float(attack_train_fraction)) > 1e-12:
        return (
            f"attack-train fraction {found_fraction:g}, this run uses "
            f"{float(attack_train_fraction):g}"
        )

    found_seed = int(training_config.get("seed", payload.get("seed", -1)))
    if found_seed != int(seed):
        return f"split seed {found_seed}, this run uses {int(seed)}"

    found_epoch = int(payload.get("epoch", -1))
    if found_epoch != int(epochs):
        return f"trained {found_epoch} epochs, this run asks for {int(epochs)}"

    return ""


def resolve_training_repo(work_dir: Path) -> Path:
    """Use a local clone when given one, otherwise take the branch tip.

    The tip is resolved at run time rather than pinned, so prompts always come
    from the current training code. The commit it landed on is logged, and the
    training pipeline stamps its own git revision into the checkpoint's
    manifest.json, so the run stays traceable without a hardcoded SHA.
    """

    local = os.environ.get("PROMPT_TRAINING_ROOT", "").strip()
    if local:
        root = Path(local).expanduser().resolve()
        if not (root / "src" / "object_agnostic_prompt_attack").is_dir():
            raise FileNotFoundError(
                f"PROMPT_TRAINING_ROOT is not the prompt-training repository: {root}"
            )
        log(f"[prompt training] using local clone {root}")
        return root

    url = os.environ["PROMPT_TRAINING_GIT_URL"]
    branch = os.environ.get("PROMPT_TRAINING_BRANCH", "main")
    root = work_dir / "object-agnostic-prompt-training"
    if not (root / ".git").is_dir():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(root)],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(root), "fetch", "--depth", "1", "origin", branch], check=True
    )
    subprocess.run(
        ["git", "-C", str(root), "checkout", "--detach", "--force", "FETCH_HEAD"],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    log(f"[prompt training] {url} {branch} -> {head}")
    return root


def write_training_config(
    destination: Path,
    *,
    dataset: str,
    split_protocol: str,
    attack_train_fraction: float,
    manifest: Path,
    seed: int,
    evaluation_fraction: float,
    epochs: int,
    batch_size: int,
    output_root: Path,
    work_dir: Path,
) -> Path:
    """Emit the resolved training config for exactly this run's split.

    Written as JSON, which ``yaml.safe_load`` reads as flow-style YAML, so the
    launcher needs no YAML dependency of its own. Only the keys this run has to
    pin are set; everything else keeps the training pipeline's own defaults.
    """

    document = {
        "data": {
            "datasets": [dataset],
            "training_mode": "per_source_dataset",
            "mvtec_root": os.environ.get("MVTEC_ROOT") or None,
            "visa_root": os.environ.get("VISA_ROOT") or None,
            # Handing over the protocol CSV is what makes the cohort match:
            # the training pipeline asserts its own seed, evaluation fraction
            # and label policy against the columns this repository stamped on
            # every row, and refuses the run when they disagree.
            "mvtec_training_manifest": str(manifest) if dataset == "mvtec" else None,
            "visa_training_manifest": str(manifest) if dataset == "visa" else None,
            "automatic_evaluation_fraction": float(evaluation_fraction),
            "split_protocol": split_protocol,
            "attack_train_fraction": float(attack_train_fraction),
        },
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "selected_epoch": int(epochs),
        },
        "model": {
            # train.sh clones the pinned AnomalyCLIP here and the runners cache
            # CLIP weights alongside it, so prompt training reuses both rather
            # than downloading its own copy.
            "anomalyclip_root": str(work_dir / "AnomalyCLIP"),
            "clip_download_root": str(work_dir / "clip_cache"),
            "device": os.environ.get("PROMPT_TRAINING_DEVICE", "auto"),
        },
        # Only settings that survived the match check reach training, so
        # anything already in the target directory describes a run this one
        # has rejected. Without this the training pipeline refuses to start
        # whenever a rejected checkpoint shares its cohort directory, which
        # every seed or epoch change does.
        "artifacts": {"output_root": str(output_root), "overwrite": True},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return destination


def train_prompts(repo_root: Path, config_path: Path, dataset: str) -> None:
    """Run the prompt-training pipeline as a subprocess, inheriting our stdio."""

    environment = dict(os.environ)
    source = str(repo_root / "src")
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = f"{source}{os.pathsep}{existing}" if existing else source
    command = [
        sys.executable,
        "-m",
        "object_agnostic_prompt_attack.cli",
        "--config",
        str(config_path),
        "--datasets",
        dataset,
    ]
    log(f"[prompt training] {' '.join(shlex.quote(part) for part in command)}")
    subprocess.run(command, check=True, cwd=str(repo_root), env=environment)


def ensure(dataset: str) -> Path:
    """Return a checkpoint matching this run's split, training one if needed."""

    split_protocol = os.environ["SPLIT_PROTOCOL"]
    attack_train_fraction = float(os.environ.get("ATTACK_TRAIN_FRACTION", "1.0"))
    seed = int(os.environ.get("SPLIT_SEED", "111"))
    evaluation_fraction = float(os.environ.get("EVALUATION_FRACTION", "0.50"))
    epochs = int(os.environ.get("PROMPT_TRAINING_EPOCHS", "15"))
    batch_size = int(os.environ.get("PROMPT_TRAINING_BATCH_SIZE", "2"))
    output_root = Path(os.environ["PROMPT_TRAINING_OUTPUT_ROOT"]).expanduser().resolve()
    work_dir = Path(os.environ["WORK_DIR"]).expanduser().resolve()
    manifest = Path(os.environ["ATTACK_TRAIN_CSV"]).expanduser().resolve()

    requirements = {
        "dataset": dataset,
        "split_protocol": split_protocol,
        "attack_train_fraction": attack_train_fraction,
        "seed": seed,
        "epochs": epochs,
    }
    derived = checkpoint_path(
        output_root, split_protocol, attack_train_fraction, dataset, epochs
    )

    # An explicitly configured path wins when it describes this run; the
    # derived location is both the fallback and where training writes.
    explicit = os.environ.get(f"LEARNABLE_PROMPT_{dataset.upper()}_CHECKPOINT", "").strip()
    if explicit.startswith(PLACEHOLDER_PREFIX):
        explicit = ""
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    if derived not in candidates:
        candidates.append(derived)

    for candidate in candidates:
        if not candidate.is_file():
            log(f"[{dataset}] no checkpoint at {candidate}")
            continue
        reason = checkpoint_mismatch(_load_payload(candidate), **requirements)
        if not reason:
            log(f"[{dataset}] using {candidate}")
            return candidate
        log(f"[{dataset}] rejecting {candidate}: {reason}")

    if not manifest.is_file():
        raise FileNotFoundError(
            f"Protocol CSV is required to train prompts for {dataset}: {manifest}"
        )
    log(
        f"[{dataset}] training prompts: protocol={split_protocol} "
        f"fraction={attack_train_fraction:g} seed={seed} epochs={epochs} "
        f"batch={batch_size}"
    )
    repo_root = resolve_training_repo(work_dir)
    config_path = write_training_config(
        work_dir / "prompt_training" / f"{dataset}_resolved.yaml",
        dataset=dataset,
        split_protocol=split_protocol,
        attack_train_fraction=attack_train_fraction,
        manifest=manifest,
        seed=seed,
        evaluation_fraction=evaluation_fraction,
        epochs=epochs,
        batch_size=batch_size,
        output_root=output_root,
        work_dir=work_dir,
    )
    train_prompts(repo_root, config_path, dataset)

    if not derived.is_file():
        raise RuntimeError(
            f"Prompt training did not write the expected checkpoint: {derived}"
        )
    reason = checkpoint_mismatch(_load_payload(derived), **requirements)
    if reason:
        raise RuntimeError(f"Freshly trained checkpoint still mismatches: {reason}")
    log(f"[{dataset}] trained {derived}")
    return derived


def main() -> None:
    datasets = [
        name.strip().lower()
        for name in os.environ.get("SOURCE_DATASETS", "").split(",")
        if name.strip()
    ]
    if not datasets:
        raise SystemExit("SOURCE_DATASETS is required")
    for dataset in datasets:
        path = ensure(dataset)
        variable = f"LEARNABLE_PROMPT_{dataset.upper()}_CHECKPOINT"
        print(f"export {variable}={shlex.quote(str(path))}")


if __name__ == "__main__":
    main()
