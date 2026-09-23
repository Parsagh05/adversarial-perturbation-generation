"""Exact WinCLIP text ensemble and object-agnostic learnable prompts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import torch


# WinCLIP's published compositional ensemble, verbatim and in order. This is
# the same vocabulary the backbone_eval evaluation harness encodes under its
# "fixed" prompt mode, so the surrogate and the evaluated backbone read the
# same text. The article lives in the template ("a photo of a {}.") and the
# state carries none ("flawless {}"), which is WinCLIP's own split.
# Component 1: photographic/context prefixes.
PREFIX_TEMPLATES = (
    "a cropped photo of the {}.",
    "a cropped photo of a {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of the {}.",
    "a dark photo of a {}.",
    "a jpeg corrupted photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of the {}.",
    "a blurry photo of a {}.",
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of the {} for visual inspection.",
    "a photo of a {} for visual inspection.",
    "a photo of the {} for anomaly detection.",
    "a photo of a {} for anomaly detection.",
)

# Component 2: normal and abnormal object states. Component 3 is the category.
NORMAL_STATES = (
    "{}",
    "flawless {}",
    "perfect {}",
    "unblemished {}",
    "{} without flaw",
    "{} without defect",
    "{} without damage",
)
ABNORMAL_STATES = (
    "damaged {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
)

VALID_PROMPT_MODES = {"frozen_winclip", "learnable_object_agnostic"}
FROZEN_PROMPT_AGGREGATION = "mean_normalized_embedding_prototype"
LEARNABLE_PROMPT_AGGREGATION = "single_learned_prompt_per_class"
PROMPT_PROVENANCE_FIELDS = (
    "prompt_mode",
    "prompt_checkpoint_sha256",
    "prompt_checkpoint_dataset",
    "prompt_checkpoint_epoch",
    "prompt_checkpoint_schema_version",
    "prompt_checkpoint_sample_manifest_sha256",
    "prompt_n_ctx",
    "prompt_normal_suffix",
    "prompt_abnormal_suffix",
    "prompt_category_specific",
    "prompt_deep_text_tuning",
    "prompt_aggregation",
    "prompt_ensemble_sha256",
)


def frozen_ensemble_sha256() -> str:
    """Fingerprint the frozen vocabulary.

    ``attack_code_sha256`` covers only ``attacks.py``, so without this a change
    to the templates or state words would alter every perturbation while the
    manifests stayed identical. Including it in the artifact metadata makes the
    vocabulary auditable and forces regeneration when it changes.
    """

    payload = "\n".join(
        (*PREFIX_TEMPLATES, "--", *NORMAL_STATES, "--", *ABNORMAL_STATES)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def learnable_prompt_checkpoint(dataset: str, prompt_mode: str) -> str:
    """Resolve the dataset-specific shallow-prompt checkpoint from the environment."""

    if prompt_mode not in VALID_PROMPT_MODES:
        raise ValueError(f"Unknown PROMPT_MODE: {prompt_mode}")
    if prompt_mode == "frozen_winclip":
        return ""
    dataset_key = dataset.strip().lower()
    if dataset_key not in {"mvtec", "visa"}:
        raise ValueError(f"No learnable-prompt checkpoint mapping for {dataset!r}")
    variable = f"LEARNABLE_PROMPT_{dataset_key.upper()}_CHECKPOINT"
    value = os.environ.get(variable, "").strip()
    if not value:
        raise ValueError(f"{variable} is required for learnable-prompt setups")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{variable} does not point to a file: {path}")
    return str(path)


def category_display_name(category: str) -> str:
    return category.replace("_", " ").strip()


def cartesian_prompts(category: str, states: Sequence[str]) -> List[str]:
    """Form prefix x state x category Cartesian-product prompts."""

    object_name = category_display_name(category)
    prompted_states = [state.format(object_name) for state in states]
    return [prefix.format(state) for state in prompted_states for prefix in PREFIX_TEMPLATES]


@dataclass(frozen=True)
class CategoryPromptBank:
    category: str
    normal_prompts: Sequence[str]
    abnormal_prompts: Sequence[str]
    normal_embeddings: torch.Tensor
    abnormal_embeddings: torch.Tensor


class PromptEnsemble:
    """Cache WinCLIP's two normalized mean prototypes for every category."""

    def __init__(self, model, tokenizer, categories: Iterable[str], device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.banks: Dict[str, CategoryPromptBank] = {}
        for category in sorted(set(categories)):
            self.banks[category] = self._encode(category)

    def _embed(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(prompts)).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_text(tokens).float()
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            prototype = torch.nn.functional.normalize(
                embeddings.mean(dim=0, keepdim=True), dim=-1
            )
        return prototype.detach()

    def _encode(self, category: str) -> CategoryPromptBank:
        normal = cartesian_prompts(category, NORMAL_STATES)
        abnormal = cartesian_prompts(category, ABNORMAL_STATES)
        return CategoryPromptBank(
            category=category,
            normal_prompts=normal,
            abnormal_prompts=abnormal,
            normal_embeddings=self._embed(normal),
            abnormal_embeddings=self._embed(abnormal),
        )

    def __getitem__(self, category: str) -> CategoryPromptBank:
        return self.banks[category]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_prompt_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 2.0 compatibility.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Prompt checkpoint is not a mapping: {path}")
    required = {
        "schema_version",
        "dataset",
        "epoch",
        "seed",
        "prompt_config",
        "training_config",
        "sample_manifest_sha256",
        "prompt_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Prompt checkpoint is missing keys: {missing}")
    if payload["schema_version"] != 1:
        raise ValueError(
            f"Unsupported prompt checkpoint schema: {payload['schema_version']}"
        )
    return payload


def learnable_prompt_text(
    n_ctx: int, normal_suffix: str, abnormal_suffix: str
) -> list[str]:
    """The two prompts a shallow checkpoint's contexts are spliced into."""

    placeholder = " ".join(["X"] * n_ctx)
    return [f"{placeholder} {normal_suffix}", f"{placeholder} {abnormal_suffix}"]


def prompt_setup_record(dataset: str, prompt_mode: str) -> dict:
    """The prompt fingerprints a run commits to, read without loading CLIP.

    Same values the surrogate later reports as ``prompt_checkpoint_sha256`` and
    ``prompt_ensemble_sha256``, available before any optimisation starts.
    """

    if prompt_mode == "frozen_winclip":
        return {
            "prompt_checkpoint_sha256": "",
            "prompt_ensemble_sha256": frozen_ensemble_sha256(),
        }
    path = Path(learnable_prompt_checkpoint(dataset, prompt_mode))
    config = _load_prompt_checkpoint(path)["prompt_config"]
    prompt_text = learnable_prompt_text(
        int(config.get("n_ctx", 0)),
        str(config.get("normal_suffix", "")).strip(),
        str(config.get("abnormal_suffix", "")).strip(),
    )
    return {
        "prompt_checkpoint_sha256": _sha256_file(path),
        "prompt_ensemble_sha256": hashlib.sha256(
            "\n".join(prompt_text).encode("utf-8")
        ).hexdigest(),
    }


class ObjectAgnosticPromptEnsemble:
    """Restore two shallow CoOp-style contexts and encode them with public CLIP."""

    def __init__(
        self,
        model,
        tokenizer: Callable[[list[str]], torch.Tensor],
        categories: Iterable[str],
        device: str,
        checkpoint_path: str,
        expected_dataset: str,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Learnable-prompt checkpoint missing: {path}")
        payload = _load_prompt_checkpoint(path)
        if str(payload["dataset"]).lower() != str(expected_dataset).lower():
            raise ValueError(
                f"Prompt checkpoint dataset {payload['dataset']!r} does not match "
                f"source dataset {expected_dataset!r}"
            )

        config = payload["prompt_config"]
        if not isinstance(config, Mapping):
            raise ValueError("prompt_config must be a mapping")
        if bool(config.get("category_specific", False)):
            raise ValueError("Only object-agnostic prompt checkpoints are supported")
        if bool(config.get("deep_text_prompt_tuning", False)):
            raise ValueError("Deep text-prompt tuning is not supported")
        n_ctx = int(config.get("n_ctx", 0))
        normal_suffix = str(config.get("normal_suffix", "")).strip()
        abnormal_suffix = str(config.get("abnormal_suffix", "")).strip()
        if n_ctx <= 0 or not normal_suffix or not abnormal_suffix:
            raise ValueError("Invalid shallow prompt configuration")

        state = payload["prompt_state"]
        if not isinstance(state, Mapping) or set(state) != {
            "normal_context",
            "abnormal_context",
        }:
            raise ValueError(
                "prompt_state must contain only normal_context and abnormal_context"
            )
        normal_context = state["normal_context"]
        abnormal_context = state["abnormal_context"]
        if (
            not isinstance(normal_context, torch.Tensor)
            or not isinstance(abnormal_context, torch.Tensor)
            or normal_context.ndim != 2
            or normal_context.shape != abnormal_context.shape
            or normal_context.shape[0] != n_ctx
        ):
            raise ValueError("Checkpoint contexts must be equal-shape [n_ctx, width] tensors")
        token_width = int(model.token_embedding.weight.shape[1])
        if normal_context.shape[1] != token_width:
            raise ValueError(
                f"Prompt width {normal_context.shape[1]} does not match CLIP width "
                f"{token_width}"
            )

        prompt_text = learnable_prompt_text(n_ctx, normal_suffix, abnormal_suffix)
        token_ids = tokenizer(prompt_text).to(self.device)
        if token_ids.ndim != 2:
            raise ValueError("Tokenizer must return a two-dimensional tensor")
        context_length = int(config.get("context_length", token_ids.shape[1]))
        if token_ids.shape != (2, context_length):
            raise ValueError(
                "Tokenizer output does not match checkpoint prompt context length"
            )
        with torch.no_grad():
            embedded = model.token_embedding(token_ids).detach().clone()
            contexts = torch.stack((normal_context, abnormal_context), dim=0).to(
                device=self.device, dtype=embedded.dtype
            )
            embedded[:, 1 : 1 + n_ctx, :] = contexts
            text_features = self._encode_embedded(embedded, token_ids)
            text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)

        normal_embedding = text_features[0:1].detach()
        abnormal_embedding = text_features[1:2].detach()
        self.banks = {
            category: CategoryPromptBank(
                category=category,
                normal_prompts=(prompt_text[0],),
                abnormal_prompts=(prompt_text[1],),
                normal_embeddings=normal_embedding,
                abnormal_embeddings=abnormal_embedding,
            )
            for category in sorted(set(categories))
        }
        self.provenance = {
            "prompt_mode": "learnable_object_agnostic",
            "prompt_checkpoint_sha256": _sha256_file(path),
            "prompt_checkpoint_dataset": str(payload["dataset"]),
            "prompt_checkpoint_epoch": int(payload["epoch"]),
            "prompt_checkpoint_schema_version": int(payload["schema_version"]),
            "prompt_checkpoint_sample_manifest_sha256": str(
                payload["sample_manifest_sha256"]
            ),
            "prompt_n_ctx": n_ctx,
            "prompt_normal_suffix": normal_suffix,
            "prompt_abnormal_suffix": abnormal_suffix,
            "prompt_category_specific": False,
            "prompt_deep_text_tuning": False,
            "prompt_aggregation": LEARNABLE_PROMPT_AGGREGATION,
            "prompt_ensemble_sha256": hashlib.sha256(
                "\n".join(prompt_text).encode("utf-8")
            ).hexdigest(),
        }

    def _encode_embedded(
        self, prompt_embeddings: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        length = prompt_embeddings.shape[1]
        positional = self.model.positional_embedding[:length].to(
            prompt_embeddings.dtype
        )
        x = (prompt_embeddings + positional).permute(1, 0, 2)
        x = self.model.transformer(x).permute(1, 0, 2)
        x = self.model.ln_final(x)
        eot_positions = token_ids.argmax(dim=-1)
        rows = torch.arange(x.shape[0], device=x.device)
        return x[rows, eot_positions] @ self.model.text_projection

    def __getitem__(self, category: str) -> CategoryPromptBank:
        return self.banks[category]


def ensemble_class_logits(
    visual_features: torch.Tensor,
    bank: CategoryPromptBank,
    temperature: float,
) -> torch.Tensor:
    """Return logits against the normal and abnormal text prototypes.

    Frozen WinCLIP banks contain the normalized mean embedding of every class's
    Cartesian prompt ensemble, matching ``backbone_eval``'s ``fixed`` mode.
    Learnable banks already contain one normalized embedding per class, so the
    same two-prototype calculation applies to both prompt modes.

    Both sides are L2-normalized, so this is a cosine classifier and only
    direction counts, matching what backbone_eval and AnomalyCLIP score with.

    ``visual_features`` may be ``[B, D]`` (global) or ``[B, P, D]`` (patches).
    """

    features = torch.nn.functional.normalize(visual_features.float(), dim=-1)
    prototypes = torch.cat(
        [bank.normal_embeddings, bank.abnormal_embeddings], dim=0
    ).float()
    if prototypes.shape[0] != 2:
        raise ValueError(
            "Prompt banks must contain one normal and one abnormal prototype"
        )
    prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
    return torch.matmul(features, prototypes.t()) / temperature
