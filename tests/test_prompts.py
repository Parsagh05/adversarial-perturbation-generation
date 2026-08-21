from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from adversarial_harness.prompts import ObjectAgnosticPromptEnsemble


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


if __name__ == "__main__":
    unittest.main()
