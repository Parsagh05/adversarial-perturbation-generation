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

from .prompts import PromptEnsemble


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
        feature_layers: Sequence[int] = (6, 12, 18, 24),
        clip_model_name: str = "ViT-L/14@336px",
        clip_download_root: str = "",
    ) -> None:
        self.device = torch.device(device)
        self.feature_layers = tuple(feature_layers)
        _, library, prompt_module = _prepare_anomalyclip_import(anomalyclip_root)
        self.library = library
        cache = clip_download_root or os.environ.get("ANOMALYCLIP_CLIP_CACHE", "")
        load_kwargs = {
            "device": self.device,
        }
        if cache:
            load_kwargs["download_root"] = str(Path(cache).expanduser())
        # The surrogate uses ordinary manual text prompts, so it must use the
        # public CLIP text transformer. Passing AnomalyCLIP's design_details
        # enables the compound-prompt transformer, whose encode_text path does
        # not accept a plain token tensor in the official implementation.
        self.model, _ = library.load(clip_model_name, **load_kwargs)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        # Deliberately do not call DAPM_replace: this is the public CLIP path.
        # The current official repository's plain CLIP encode_image method does
        # not accept AnomalyCLIP's DPAM_layer/features-list arguments. Capture
        # the requested original-CLIP transformer layers with forward hooks.
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
        self.prompts = PromptEnsemble(
            self.model, prompt_module.tokenize, categories, str(self.device)
        )

    def encode_visual(
        self, images_01: torch.Tensor, include_patches: bool = True
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
        images = normalize_clip(images_01.to(self.device))
        self._captured_features.clear()
        self._capture_patches = include_patches
        try:
            # The official plain CLIP implementation returns all final visual
            # tokens as [B, N, D]; token zero is the global CLS representation.
            visual_tokens = self.model.encode_image(images)
        finally:
            self._capture_patches = False
        if visual_tokens.ndim != 3:
            raise RuntimeError(
                "The public CLIP surrogate must return visual tokens with "
                f"shape [B, N, D], got {tuple(visual_tokens.shape)}"
            )
        global_features = visual_tokens[:, 0, :].float()
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
