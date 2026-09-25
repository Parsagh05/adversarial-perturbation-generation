"""Surrogate and black-box target model adapters.

The attack code only receives :class:`CLIPSurrogate`. The target adapter is
invoked after perturbations are produced and is never part of the gradient
graph, which makes the transfer/black-box boundary explicit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Type
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .prompts import (
    FROZEN_PROMPT_AGGREGATION,
    ObjectAgnosticPromptEnsemble,
    PromptEnsemble,
    frozen_ensemble_sha256,
)


# The surrogate is OpenAI's own CLIP package at this commit (requirements.txt
# pins the same one); it is part of every delta's reuse key.
SURROGATE_CLIP = "openai/CLIP@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
# The final block only, as plain CLIP reads its patch tokens.
SURROGATE_FEATURE_LAYERS = (24,)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def normalize_clip(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def _prepare_anomalyclip_import(root_value: str):
    root = Path(root_value).expanduser().resolve()
    if not (root / "AnomalyCLIP_lib").is_dir():
        raise FileNotFoundError(
            f"AnomalyCLIP source not found at {root}. Expected AnomalyCLIP_lib/."
        )
    root_string = str(root)
    if root_string in sys.path:
        sys.path.remove(root_string)
    sys.path.insert(0, root_string)
    importlib.invalidate_caches()
    for module_name in ("utils", "prompt_ensemble"):
        module = sys.modules.get(module_name)
        module_file = str(getattr(module, "__file__", "")) if module else ""
        if module and not module_file.startswith(root_string):
            sys.modules.pop(module_name, None)
    library = importlib.import_module("AnomalyCLIP_lib")
    prompt_module = importlib.import_module("prompt_ensemble")
    return root, library, prompt_module


def _design_details(depth: int = 9, n_ctx: int = 12, t_n_ctx: int = 4) -> Dict[str, int]:
    return {
        "Prompt_length": n_ctx,
        "learnabel_text_embedding_depth": depth,
        "learnabel_text_embedding_length": t_n_ctx,
    }


class CLIPSurrogate:
    """Frozen public CLIP trunk used only for adversarial optimization."""

    def __init__(
        self,
        anomalyclip_root: str,
        categories: Sequence[str],
        device: str,
        feature_layers: Sequence[int] = SURROGATE_FEATURE_LAYERS,
        clip_model_name: str = "ViT-L/14@336px",
        clip_download_root: str = "",
        prompt_mode: str = "frozen_winclip",
        learnable_prompt_checkpoint: str = "",
        prompt_dataset: str = "",
    ) -> None:
        self.device = torch.device(device)
        self.feature_layers = tuple(feature_layers)
        # OpenAI's own package, not AnomalyCLIP's copy of it: the same weights,
        # blocks, tokenizer and forward as clip.load. anomalyclip_root is kept
        # for the callers' provenance only.
        import clip as openai_clip

        cache = clip_download_root or os.environ.get("ANOMALYCLIP_CLIP_CACHE", "")
        self.model, _ = openai_clip.load(
            clip_model_name,
            device=self.device,
            jit=False,
            download_root=str(Path(cache).expanduser()) if cache else None,
        )
        # clip.load casts the model to fp16 on a GPU. Precision is set by the
        # runners instead (fp32 weights, bf16 autocast by default), so undo it;
        # the released weights are fp16 values either way.
        self.model.float().eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        visual = self.model.visual
        self._patch_size = int(visual.conv1.kernel_size[0])
        # clip.load only accepts its native 336 px. The target detectors run
        # this CLIP at 518 px by stretching the positional grid bilinearly
        # (AnomalyCLIP's VisionTransformer.forward); encode_visual does the
        # same from the untouched native grid.
        self._native_positional_embedding = visual.positional_embedding.detach().clone()
        self._position_grid = int(round((self._native_positional_embedding.shape[0] - 1) ** 0.5))
        # No DPAM: this is the plain CLIP path. Plain CLIP exposes only the
        # final CLS token, so the patch tokens of the requested blocks are read
        # with forward hooks.
        self._captured_features: Dict[int, torch.Tensor] = {}
        self._capture_patches = False
        self._feature_handles = []
        blocks = self.model.visual.transformer.resblocks
        for layer in self.feature_layers:
            if layer < 1 or layer > len(blocks):
                raise ValueError(
                    f"CLIP feature layer {layer} is outside the valid range "
                    f"[1, {len(blocks)}]"
                )

            def capture_hook(_module, _inputs, output, *, layer_index=layer):
                if self._capture_patches:
                    self._captured_features[layer_index] = output

            self._feature_handles.append(
                blocks[layer - 1].register_forward_hook(capture_hook)
            )
        if prompt_mode == "frozen_winclip":
            self.prompts = PromptEnsemble(
                self.model, openai_clip.tokenize, categories, str(self.device)
            )
            self.prompt_provenance = {
                "prompt_mode": "frozen_winclip",
                "prompt_checkpoint_sha256": "",
                "prompt_checkpoint_dataset": "",
                "prompt_checkpoint_epoch": "",
                "prompt_checkpoint_schema_version": "",
                "prompt_checkpoint_sample_manifest_sha256": "",
                "prompt_n_ctx": "",
                "prompt_normal_suffix": "winclip_cartesian_normal_states",
                "prompt_abnormal_suffix": "winclip_cartesian_abnormal_states",
                "prompt_category_specific": True,
                "prompt_deep_text_tuning": False,
                "prompt_aggregation": FROZEN_PROMPT_AGGREGATION,
                "prompt_ensemble_sha256": frozen_ensemble_sha256(),
            }
        elif prompt_mode == "learnable_object_agnostic":
            if not learnable_prompt_checkpoint or not prompt_dataset:
                raise ValueError(
                    "Learnable prompts require checkpoint and source dataset"
                )
            self.prompts = ObjectAgnosticPromptEnsemble(
                self.model,
                openai_clip.tokenize,
                categories,
                str(self.device),
                learnable_prompt_checkpoint,
                prompt_dataset,
            )
            self.prompt_provenance = dict(self.prompts.provenance)
        else:
            raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    def _fit_positional_embedding(self, side: int) -> None:
        """Stretch the native positional grid to ``side`` x ``side`` patches.

        Bilinear, align_corners=False, from the native grid: the same numbers
        AnomalyCLIP's forward produces for the targets at 518 px.
        """

        if side == self._position_grid:
            return
        native = self._native_positional_embedding
        grid = int(round((native.shape[0] - 1) ** 0.5))
        patches = native[1:].reshape(1, grid, grid, -1).permute(0, 3, 1, 2)
        patches = F.interpolate(
            patches, size=(side, side), mode="bilinear", align_corners=False
        )
        patches = patches.permute(0, 2, 3, 1).reshape(side * side, -1)
        self.model.visual.positional_embedding.data = torch.cat(
            (native[:1], patches), dim=0
        ).to(self.model.visual.positional_embedding.dtype)
        self._position_grid = side

    def encode_visual(
        self, images_01: torch.Tensor, include_patches: bool = True
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
        images = normalize_clip(images_01.to(self.device))
        self._fit_positional_embedding(images.shape[-1] // self._patch_size)
        self._captured_features.clear()
        self._capture_patches = include_patches
        try:
            # Official CLIP returns ln_post(CLS) @ proj, shape [B, D].
            global_features = self.model.encode_image(images).float()
        finally:
            self._capture_patches = False
        if not include_patches:
            return global_features, []

        missing = [
            layer for layer in self.feature_layers
            if layer not in self._captured_features
        ]
        if missing:
            self._captured_features.clear()
            raise RuntimeError(
                f"CLIP forward hooks did not capture feature layers: {missing}"
            )
        captured_features = [
            self._captured_features[layer] for layer in self.feature_layers
        ]
        # Do not retain the final PGD graph on the surrogate between calls or
        # while target inference runs. The local tensors below keep the graph
        # alive only for the objective that consumes them.
        self._captured_features.clear()
        visual = self.model.visual
        patch_features = []
        for feature in captured_features:
            # Hook outputs are [N, B, width]. Match AnomalyCLIP's public-path
            # feature projection by applying the visual post norm and matrix.
            feature = feature.permute(1, 0, 2)
            feature = visual.ln_post(feature)
            if visual.proj is not None:
                feature = feature @ visual.proj
            patch_features.append(feature.float())
        return global_features, patch_features

    def release(self) -> None:
        for handle in self._feature_handles:
            handle.remove()
        self._feature_handles.clear()
        self._captured_features.clear()
        del self.prompts
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TargetAdapter(ABC):
    """Interface implemented by every black-box anomaly detector."""

    model_name: str

    @abstractmethod
    def predict(self, images_01: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Return anomaly probabilities and low-resolution anomaly maps."""

    @abstractmethod
    def release(self) -> None:
        pass


class AnomalyCLIPTarget(TargetAdapter):
    model_name = "AnomalyCLIP"

    def __init__(
        self,
        anomalyclip_root: str,
        checkpoint_path: str,
        device: str,
        image_size: int = 518,
        features_list: Sequence[int] = (6, 12, 18, 24),
        feature_map_indices: Sequence[int] = (0, 1, 2, 3),
        depth: int = 9,
        n_ctx: int = 12,
        t_n_ctx: int = 4,
        dpam_layer: int = 20,
        clip_model_name: str = "ViT-L/14@336px",
        clip_download_root: str = "",
        **_: object,
    ) -> None:
        self.device = torch.device(device)
        self.image_size = image_size
        self.features_list = tuple(features_list)
        self.feature_map_indices = set(feature_map_indices)
        self.dpam_layer = dpam_layer
        root, library, prompt_module = _prepare_anomalyclip_import(anomalyclip_root)
        self.library = library
        details = _design_details(depth=depth, n_ctx=n_ctx, t_n_ctx=t_n_ctx)
        cache = clip_download_root or os.environ.get("ANOMALYCLIP_CLIP_CACHE", "")
        load_kwargs = {"device": self.device, "design_details": details}
        if cache:
            load_kwargs["download_root"] = str(Path(cache).expanduser())
        self.model, _ = library.load(clip_model_name, **load_kwargs)
        self.model.eval()

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"AnomalyCLIP checkpoint not found: {checkpoint}")
        prompt_learner = prompt_module.AnomalyCLIP_PromptLearner(
            self.model.to("cpu"), details
        )
        state = torch.load(checkpoint, map_location="cpu")
        if "prompt_learner" not in state:
            raise KeyError(f"Checkpoint has no 'prompt_learner' key: {checkpoint}")
        prompt_learner.load_state_dict(state["prompt_learner"])
        prompt_learner.to(self.device).eval()
        self.model.to(self.device)
        self.model.visual.DAPM_replace(DPAM_layer=dpam_layer)
        self.model.requires_grad_(False)
        prompt_learner.requires_grad_(False)

        with torch.inference_mode():
            prompts, tokenized, compound = prompt_learner(cls_id=None)
            text = self.model.encode_text_learn(prompts, tokenized, compound).float()
            text = torch.stack(torch.chunk(text, chunks=2, dim=0), dim=1)
            self.text_features = F.normalize(text, dim=-1).detach()
        self.prompt_learner = prompt_learner

    def predict(self, images_01: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        if images_01.ndim != 4 or images_01.shape[1] != 3:
            raise ValueError("Target input must have shape [B, 3, H, W]")
        if images_01.shape[-2:] != (self.image_size, self.image_size):
            images_01 = F.interpolate(
                images_01,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        images = normalize_clip(images_01.to(self.device))
        with torch.inference_mode():
            image_features, patch_features = self.model.encode_image(
                images,
                list(self.features_list),
                DPAM_layer=self.dpam_layer,
            )
            image_features = F.normalize(image_features.float(), dim=-1)
            logits = image_features @ self.text_features[0].t()
            scores = (logits / 0.07).softmax(dim=-1)[:, 1]

            maps = []
            for index, patch in enumerate(patch_features):
                if index not in self.feature_map_indices:
                    continue
                patch = F.normalize(patch.float(), dim=-1)
                similarity, _ = self.library.compute_similarity(
                    patch, self.text_features[0]
                )
                similarity = similarity[:, 1:, :]
                side = int(similarity.shape[1] ** 0.5)
                if side * side != similarity.shape[1]:
                    raise ValueError(
                        f"Patch-token count is not square: {similarity.shape[1]}"
                    )
                similarity = similarity.reshape(similarity.shape[0], side, side, 2)
                maps.append((similarity[..., 1] + 1.0 - similarity[..., 0]) / 2.0)
            if not maps:
                raise RuntimeError("No AnomalyCLIP feature maps were selected")
            lowres_maps = torch.stack(maps, dim=0).sum(dim=0)
        return (
            scores.cpu().numpy().astype(np.float32),
            lowres_maps.cpu().numpy().astype(np.float32),
        )

    def release(self) -> None:
        del self.text_features
        del self.prompt_learner
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


TARGET_REGISTRY: Dict[str, Type[TargetAdapter]] = {
    "AnomalyCLIP": AnomalyCLIPTarget,
}


def build_target(name: str, **kwargs) -> TargetAdapter:
    try:
        adapter = TARGET_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown target model {name!r}. Available: {sorted(TARGET_REGISTRY)}"
        ) from exc
    return adapter(**kwargs)
