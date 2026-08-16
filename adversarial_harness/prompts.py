"""WinCLIP-style Cartesian prompt ensembles and contrastive class logits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch


# Component 1: photographic/context prefixes.
PREFIX_TEMPLATES = (
    "a photo of {}.",
    "a close-up photo of {}.",
    "this is a photo of {}.",
    "there is {} in the scene.",
    "a cropped photo of {}.",
    "a bright photo of {}.",
    "a dark photo of {}.",
    "a low resolution photo of {}.",
    "a blurry photo of {}.",
    "a black and white photo of {}.",
)

# Component 2: normal and abnormal object states. Component 3 is the category.
NORMAL_STATES = (
    "a {}",
    "a flawless {}",
    "a perfect {}",
    "an unblemished {}",
    "a normal {}",
    "an undamaged {}",
    "a {} without flaw",
    "a {} without defect",
    "a {} without damage",
)
ABNORMAL_STATES = (
    "a damaged {}",
    "a broken {}",
    "an abnormal {}",
    "a defective {}",
    "a {} with flaw",
    "a {} with defect",
    "a {} with damage",
)


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
    """Cache complete per-category prompt embeddings on the attack device."""

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
        return embeddings.detach()

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


def ensemble_class_logits(
    visual_features: torch.Tensor,
    bank: CategoryPromptBank,
    temperature: float,
) -> torch.Tensor:
    """Return normal/abnormal logits while retaining every prompt.

    For each class, similarities to all Cartesian prompts are aggregated with
    log-mean-exp. This is a smooth ensemble decision: every prompt contributes,
    prompt-count imbalance is corrected, and cross entropy can directly target
    the incorrect normal/abnormal class.

    ``visual_features`` may be ``[B, D]`` (global) or ``[B, P, D]`` (patches).
    """

    features = torch.nn.functional.normalize(visual_features.float(), dim=-1)

    def class_logit(text: torch.Tensor) -> torch.Tensor:
        similarities = torch.matmul(features, text.t()) / temperature
        count = torch.tensor(
            text.shape[0], device=similarities.device, dtype=similarities.dtype
        )
        return torch.logsumexp(similarities, dim=-1) - torch.log(count)

    return torch.stack(
        [class_logit(bank.normal_embeddings), class_logit(bank.abnormal_embeddings)],
        dim=-1,
    )
