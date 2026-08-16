"""MVTec AD/VisA discovery and resolution-aware tensor loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class MVTecSample:
    index: int
    category: str
    defect_type: str
    image_path: Path
    mask_path: Optional[Path]
    label: int
    split: str = "test"
    dataset: str = "mvtec"

    @property
    def sample_id(self) -> str:
        relative_id = f"{self.category}/{self.defect_type}/{self.image_path.stem}"
        # Keep historical MVTec IDs stable so existing matched-split manifests
        # remain valid, while namespacing VisA IDs for mixed-dataset runs.
        return (
            relative_id
            if self.dataset == "mvtec"
            else f"{self.dataset}/{relative_id}"
        )

    @property
    def protocol_id(self) -> str:
        """Split-qualified identifier used by held-out protocol manifests."""

        return f"{self.split}/{self.sample_id}"


def _image_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _balanced_cap(
    samples: Sequence[MVTecSample], max_samples: Optional[int]
) -> List[MVTecSample]:
    if max_samples is None:
        return list(samples)
    normal = [sample for sample in samples if sample.label == 0]
    anomaly = [sample for sample in samples if sample.label == 1]
    half = max(1, max_samples // 2)
    return normal[:half] + anomaly[: max_samples - half]


def _reindex(samples: Sequence[MVTecSample]) -> List[MVTecSample]:
    return [replace(sample, index=index) for index, sample in enumerate(samples)]


def discover_mvtec(
    root: str,
    categories: Optional[Sequence[str]] = None,
    max_samples_per_category: Optional[int] = None,
) -> List[MVTecSample]:
    """Discover the official MVTec ``test`` split.

    ``test/good`` samples receive an all-zero mask. Defective samples are
    matched to ``ground_truth/<defect>/<stem>_mask.<ext>``.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"MVTec root does not exist: {root_path}")

    available = sorted(
        path.name for path in root_path.iterdir()
        if path.is_dir() and (path / "test").is_dir()
    )
    selected = list(categories) if categories is not None else available
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"MVTec categories not found under {root_path}: {missing}")

    provisional: List[MVTecSample] = []
    for category in selected:
        category_samples: List[MVTecSample] = []
        test_root = root_path / category / "test"
        for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            is_good = defect_dir.name.lower() == "good"
            for image_path in _image_files(defect_dir):
                mask_path = None
                if not is_good:
                    gt_dir = root_path / category / "ground_truth" / defect_dir.name
                    candidates = sorted(
                        path for path in gt_dir.glob(f"{image_path.stem}_mask.*")
                        if path.suffix.lower() in IMAGE_EXTENSIONS
                    )
                    if not candidates:
                        raise FileNotFoundError(
                            f"Ground-truth mask missing for {image_path} under {gt_dir}"
                        )
                    mask_path = candidates[0]
                category_samples.append(
                    MVTecSample(
                        index=-1,
                        category=category,
                        defect_type=defect_dir.name,
                        image_path=image_path,
                        mask_path=mask_path,
                        label=0 if is_good else 1,
                        split="test",
                    )
                )
        # Deterministic, label-aware cap so smoke runs retain both classes.
        category_samples = _balanced_cap(category_samples, max_samples_per_category)
        provisional.extend(category_samples)

    samples = _reindex(provisional)
    if not samples:
        raise RuntimeError(f"No MVTec test samples found under {root_path}")
    return samples


def _visa_manifest(root: Path) -> Path:
    candidates = (
        root / "split_csv" / "1cls.csv",
        root / "split_csv" / "1cls.csv.csv",
        root / "1cls.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"VisA split manifest not found under {root}. Expected split_csv/1cls.csv"
    )


def _visa_rows(root: Path) -> List[Dict[str, str]]:
    manifest = _visa_manifest(root)
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [
            {
                str(key).strip().lower(): str(value or "").strip()
                for key, value in row.items()
            }
            for row in reader
        ]
    required = {"object", "split", "label", "image"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"VisA manifest {manifest} must contain columns {sorted(required)}"
        )
    return rows


def _visa_path(root: Path, value: str) -> Optional[Path]:
    value = value.strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return None
    # Official manifests use POSIX separators even when consumed on Windows.
    relative = Path(value.replace("\\", "/"))
    return relative if relative.is_absolute() else root / relative


def _discover_visa_split(
    root: str,
    split: str,
    categories: Optional[Sequence[str]] = None,
    max_samples_per_category: Optional[int] = None,
) -> List[MVTecSample]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"VisA root does not exist: {root_path}")
    rows = _visa_rows(root_path)
    available = sorted({row["object"] for row in rows})
    selected = list(categories) if categories is not None else available
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"VisA categories not found under {root_path}: {missing}")

    provisional: List[MVTecSample] = []
    for category in selected:
        category_samples: List[MVTecSample] = []
        for row in rows:
            if row["object"] != category or row["split"].lower() != split.lower():
                continue
            label_name = row["label"].lower()
            if label_name not in {"normal", "good", "0", "anomaly", "bad", "1"}:
                raise ValueError(
                    f"Unknown VisA label {row['label']!r} for {row['image']}"
                )
            is_normal = label_name in {"normal", "good", "0"}
            image_path = _visa_path(root_path, row["image"])
            if image_path is None or not image_path.is_file():
                raise FileNotFoundError(
                    f"VisA image listed in manifest is missing: {image_path}"
                )
            mask_path = (
                None
                if is_normal
                else _visa_path(root_path, row.get("mask", ""))
            )
            if not is_normal and (mask_path is None or not mask_path.is_file()):
                raise FileNotFoundError(
                    f"VisA anomaly mask listed in manifest is missing: {mask_path}"
                )
            category_samples.append(
                MVTecSample(
                    index=-1,
                    category=category,
                    defect_type="normal" if is_normal else "anomaly",
                    image_path=image_path,
                    mask_path=mask_path,
                    label=0 if is_normal else 1,
                    split=split.lower(),
                    dataset="visa",
                )
            )
        category_samples = sorted(
            category_samples, key=lambda sample: str(sample.image_path)
        )
        category_samples = _balanced_cap(category_samples, max_samples_per_category)
        provisional.extend(category_samples)
    if not provisional:
        raise RuntimeError(f"No VisA {split} samples found under {root_path}")
    return _reindex(provisional)


def discover_visa(
    root: str,
    categories: Optional[Sequence[str]] = None,
    max_samples_per_category: Optional[int] = None,
) -> List[MVTecSample]:
    """Discover VisA's official test split from ``split_csv/1cls.csv``."""

    return _discover_visa_split(root, "test", categories, max_samples_per_category)


def discover_visa_train_normal(
    root: str, categories: Optional[Sequence[str]] = None
) -> List[MVTecSample]:
    """Discover VisA normal training images for fitting and calibration."""

    samples = _discover_visa_split(root, "train", categories)
    normal = [sample for sample in samples if sample.label == 0]
    if not normal:
        raise RuntimeError(f"No VisA train/normal samples found under {root}")
    return _reindex(normal)


def discover_anomaly_datasets(
    dataset: str,
    mvtec_root: Optional[str] = None,
    visa_root: Optional[str] = None,
    categories: Optional[Sequence[str]] = None,
    max_samples_per_category: Optional[int] = None,
    train_normal: bool = False,
) -> List[MVTecSample]:
    """Discover MVTec, VisA, or their union through one stable interface."""

    mode = str(dataset).lower()
    if mode not in {"mvtec", "visa", "both"}:
        raise ValueError("dataset must be one of: mvtec, visa, both")
    samples: List[MVTecSample] = []
    if mode in {"mvtec", "both"}:
        if not mvtec_root:
            raise ValueError(f"mvtec_root is required when dataset={mode!r}")
        loader = discover_mvtec_train_good if train_normal else discover_mvtec
        kwargs = (
            {}
            if train_normal
            else {"max_samples_per_category": max_samples_per_category}
        )
        samples.extend(loader(mvtec_root, categories=None, **kwargs))
    if mode in {"visa", "both"}:
        if not visa_root:
            raise ValueError(f"visa_root is required when dataset={mode!r}")
        loader = discover_visa_train_normal if train_normal else discover_visa
        kwargs = (
            {}
            if train_normal
            else {"max_samples_per_category": max_samples_per_category}
        )
        samples.extend(loader(visa_root, categories=None, **kwargs))

    if categories is not None:
        available = {sample.category for sample in samples}
        missing = sorted(set(categories) - available)
        if missing:
            raise ValueError(f"Categories not found in selected dataset(s): {missing}")
        selected = set(categories)
        samples = [sample for sample in samples if sample.category in selected]
    return _reindex(samples)


def discover_mvtec_train_good(
    root: str,
    categories: Optional[Sequence[str]] = None,
) -> List[MVTecSample]:
    """Discover normal MVTec ``train/good`` images for universal fitting.

    These samples are deliberately kept separate from the indexed test set.
    They never participate in evaluation metrics or target prediction arrays.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"MVTec root does not exist: {root_path}")
    available = sorted(
        path.name
        for path in root_path.iterdir()
        if path.is_dir() and (path / "train" / "good").is_dir()
    )
    selected = list(categories) if categories is not None else available
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(
            f"MVTec train/good categories not found under {root_path}: {missing}"
        )

    samples: List[MVTecSample] = []
    for category in selected:
        for image_path in _image_files(root_path / category / "train" / "good"):
            samples.append(
                MVTecSample(
                    index=-1,
                    category=category,
                    defect_type="good",
                    image_path=image_path,
                    mask_path=None,
                    label=0,
                    split="train",
                )
            )
    if not samples:
        raise RuntimeError(f"No MVTec train/good samples found under {root_path}")
    return samples


def group_by_category(samples: Sequence[MVTecSample]) -> Dict[str, List[MVTecSample]]:
    grouped: Dict[str, List[MVTecSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.category, []).append(sample)
    return grouped


def load_image_tensor(sample: MVTecSample, image_size: int) -> torch.Tensor:
    """Load RGB image as ``[3, H, W]`` float tensor in ``[0, 1]``."""

    image = Image.open(sample.image_path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return tensor.squeeze(0).clamp(0.0, 1.0)


def load_mask(sample: MVTecSample, image_size: int) -> np.ndarray:
    """Load a binary mask at the benchmark metric resolution."""

    if sample.mask_path is None:
        return np.zeros((image_size, image_size), dtype=np.uint8)
    mask = np.asarray(Image.open(sample.mask_path).convert("L"), dtype=np.uint8)
    tensor = torch.from_numpy((mask > 0).astype(np.float32))[None, None]
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="nearest")
    return tensor[0, 0].numpy().astype(np.uint8)
