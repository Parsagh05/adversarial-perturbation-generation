from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from adversarial_harness import prompts
from adversarial_harness.prompts import (
    ObjectAgnosticPromptEnsemble,
    PromptEnsemble,
    ensemble_class_logits,
)


class _MixTokens(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + value.mean(dim=0, keepdim=True)


class _FakeClip(torch.nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.token_embedding = torch.nn.Embedding(128, width)
        self.positional_embedding = torch.nn.Parameter(torch.zeros(77, width))
        self.transformer = _MixTokens()
        self.ln_final = torch.nn.Identity()
        self.text_projection = torch.nn.Parameter(torch.eye(width))
        with torch.no_grad():
            values = torch.arange(128 * width, dtype=torch.float32).reshape(128, width)
            self.token_embedding.weight.copy_(values / values.max())

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.token_embedding(token_ids)
        x = self.transformer(embedded.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x)
        rows = torch.arange(x.shape[0], device=x.device)
        return x[rows, token_ids.argmax(dim=-1)] @ self.text_projection


def _tokenize(prompts: list[str]) -> torch.Tensor:
    result = torch.zeros((len(prompts), 77), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        words = prompt.split()
        result[row, 0] = 1
        for column, word in enumerate(words, start=1):
            result[row, column] = 3 + sum(word.encode("utf-8")) % 80
        result[row, len(words) + 1] = 127  # CLIP-style maximum-token EOT.
    return result


def _payload(dataset: str = "mvtec", *, deep: bool = False) -> dict:
    return {
        "schema_version": 1,
        "dataset": dataset,
        "epoch": 15,
        "seed": 111,
        "prompt_config": {
            "n_ctx": 3,
            "normal_suffix": "object.",
            "abnormal_suffix": "damaged object.",
            "context_length": 77,
            "category_specific": False,
            "deep_text_prompt_tuning": deep,
        },
        "training_config": {"epochs": 15},
        "sample_manifest_sha256": "a" * 64,
        "prompt_state": {
            "normal_context": torch.full((3, 4), 0.25),
            "abnormal_context": torch.full((3, 4), 0.75),
        },
    }


class ObjectAgnosticPromptTests(unittest.TestCase):
    def _save(self, directory: str, payload: dict) -> Path:
        path = Path(directory) / "prompts_epoch15.pt"
        torch.save(payload, path)
        return path

    def test_loads_shallow_contexts_and_reuses_them_for_every_category(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, _payload())
            prompts = ObjectAgnosticPromptEnsemble(
                _FakeClip(), _tokenize, ["bottle", "cable"], "cpu", str(path), "mvtec"
            )

            self.assertTrue(
                torch.equal(
                    prompts["bottle"].normal_embeddings,
                    prompts["cable"].normal_embeddings,
                )
            )
            self.assertFalse(
                torch.equal(
                    prompts["bottle"].normal_embeddings,
                    prompts["bottle"].abnormal_embeddings,
                )
            )
            self.assertEqual(prompts.provenance["prompt_checkpoint_epoch"], 15)
            self.assertEqual(len(prompts.provenance["prompt_checkpoint_sha256"]), 64)

    def test_rejects_checkpoint_from_another_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, _payload("visa"))
            with self.assertRaisesRegex(ValueError, "does not match"):
                ObjectAgnosticPromptEnsemble(
                    _FakeClip(), _tokenize, ["bottle"], "cpu", str(path), "mvtec"
                )

    def test_rejects_deep_prompt_tuning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(directory, _payload(deep=True))
            with self.assertRaisesRegex(ValueError, "Deep text-prompt tuning"):
                ObjectAgnosticPromptEnsemble(
                    _FakeClip(), _tokenize, ["bottle"], "cpu", str(path), "mvtec"
                )


class WinCLIPVocabularyTests(unittest.TestCase):
    """Pin the frozen ensemble to WinCLIP's published set.

    The same vocabulary is encoded by the backbone_eval harness under its
    "fixed" prompt mode, so the surrogate and the evaluated backbone read
    identical text. Drift here changes every perturbation silently.
    """

    TEMPLATES = (
        "a cropped photo of the {}.", "a cropped photo of a {}.",
        "a close-up photo of a {}.", "a close-up photo of the {}.",
        "a bright photo of a {}.", "a bright photo of the {}.",
        "a dark photo of the {}.", "a dark photo of a {}.",
        "a jpeg corrupted photo of a {}.", "a jpeg corrupted photo of the {}.",
        "a blurry photo of the {}.", "a blurry photo of a {}.",
        "a photo of a {}.", "a photo of the {}.",
        "a photo of a small {}.", "a photo of the small {}.",
        "a photo of a large {}.", "a photo of the large {}.",
        "a photo of the {} for visual inspection.",
        "a photo of a {} for visual inspection.",
        "a photo of the {} for anomaly detection.",
        "a photo of a {} for anomaly detection.",
    )
    NORMAL = ("{}", "flawless {}", "perfect {}", "unblemished {}",
              "{} without flaw", "{} without defect", "{} without damage")
    ANOMALOUS = ("damaged {}", "{} with flaw", "{} with defect", "{} with damage")

    def test_vocabulary_matches_winclip_verbatim_and_in_order(self) -> None:
        self.assertEqual(tuple(prompts.PREFIX_TEMPLATES), self.TEMPLATES)
        self.assertEqual(tuple(prompts.NORMAL_STATES), self.NORMAL)
        self.assertEqual(tuple(prompts.ABNORMAL_STATES), self.ANOMALOUS)

    def test_generated_prompts_match_the_reference_composition(self) -> None:
        # backbone_eval: template.format(state.format(name)) for state, then template
        for category in ("bottle", "metal_nut", "pcb1"):
            name = category.replace("_", " ")
            for states, mine in ((self.NORMAL, prompts.NORMAL_STATES),
                                 (self.ANOMALOUS, prompts.ABNORMAL_STATES)):
                expected = [t.format(s.format(name))
                            for s in states for t in self.TEMPLATES]
                with self.subTest(category=category, count=len(expected)):
                    self.assertEqual(prompts.cartesian_prompts(category, mine),
                                     expected)

    def test_category_name_is_inserted_not_a_fixed_object_label(self) -> None:
        texts = prompts.cartesian_prompts("bottle", prompts.NORMAL_STATES)
        self.assertTrue(all("bottle" in text for text in texts))
        self.assertFalse(any("object" in text for text in texts))

    def test_frozen_bank_matches_backbone_eval_mean_prototypes(self) -> None:
        model = _FakeClip()
        ensemble = PromptEnsemble(model, _tokenize, ["metal_nut"], "cpu")
        bank = ensemble["metal_nut"]

        self.assertEqual(tuple(bank.normal_embeddings.shape), (1, 4))
        self.assertEqual(tuple(bank.abnormal_embeddings.shape), (1, 4))
        for texts, actual in (
            (bank.normal_prompts, bank.normal_embeddings),
            (bank.abnormal_prompts, bank.abnormal_embeddings),
        ):
            individual = torch.nn.functional.normalize(
                model.encode_text(_tokenize(list(texts))).float(), dim=-1
            )
            expected = torch.nn.functional.normalize(
                individual.mean(dim=0, keepdim=True), dim=-1
            )
            self.assertTrue(torch.allclose(actual, expected))

    def test_logits_compare_visual_features_with_two_prototypes(self) -> None:
        bank = PromptEnsemble(_FakeClip(), _tokenize, ["bottle"], "cpu")["bottle"]
        visual = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        temperature = 0.07
        expected = torch.nn.functional.normalize(visual, dim=-1) @ torch.cat(
            [bank.normal_embeddings, bank.abnormal_embeddings], dim=0
        ).t() / temperature
        actual = ensemble_class_logits(visual, bank, temperature)
        self.assertTrue(torch.allclose(actual, expected))

    def test_ensemble_fingerprint_tracks_the_vocabulary(self) -> None:
        baseline = prompts.frozen_ensemble_sha256()
        self.assertEqual(len(baseline), 64)
        original = prompts.NORMAL_STATES
        try:
            prompts.NORMAL_STATES = original + ("pristine {}",)
            self.assertNotEqual(prompts.frozen_ensemble_sha256(), baseline)
        finally:
            prompts.NORMAL_STATES = original
        self.assertEqual(prompts.frozen_ensemble_sha256(), baseline)


if __name__ == "__main__":
    unittest.main()
