"""Typed configuration for the adversarial robustness benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


VALID_SCOPES = ("per_image", "per_category", "dataset")
VALID_DIRECTIONS = ("normal_to_abnormal", "abnormal_to_normal")
VALID_LOSS_MODES = ("global", "local", "combined")
VALID_STEP_SIZE_SCHEDULES = ("constant", "cosine")
VALID_NORMAL_LOCAL_TARGETS = ("fixed_region", "full_image")
VALID_UNIVERSAL_PROTOCOLS = ("transductive", "held_out")
VALID_THRESHOLD_MODES = ("normal_train_quantile",)
VALID_DATASETS = (
    "mvtec",
    "visa",
    "both",
    "mvtec_to_visa",
    "visa_to_mvtec",
)


def _tuple(value: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(item) for item in value)


@dataclass
class AttackConfig:
    """PGD and attack-protocol settings.

    Pixel values and epsilon/step size are expressed in the unnormalized
    ``[0, 1]`` image domain. Universal attacks use ``universal_steps`` update
    batches; per-image attacks use ``steps`` updates per image.
    """

    image_size: int = 518
    epsilon: float = 8.0 / 255.0
    step_size: float = 2.0 / 255.0
    steps: int = 20
    universal_steps: int = 200
    random_start: bool = True
    temperature: float = 0.07
    global_weight: float = 0.2
    local_weight: float = 0.8
    mask_local_loss: bool = True
    local_background_weight: float = 0.1
    normal_local_target: str = "fixed_region"
    normal_target_region_fraction: float = 0.25
    normal_target_center_x: float = 0.5
    normal_target_center_y: float = 0.5
    local_focal_weight: float = 0.5
    local_dice_weight: float = 0.5
    local_focal_gamma: float = 2.0
    local_dice_smooth: float = 1.0
    step_size_schedule: str = "constant"
    step_size_min_ratio: float = 0.1
    diagnostic_interval: int = 10
    feature_layers: Tuple[int, ...] = (6, 12, 18, 24)
    scopes: Tuple[str, ...] = VALID_SCOPES
    directions: Tuple[str, ...] = VALID_DIRECTIONS
    loss_modes: Tuple[str, ...] = VALID_LOSS_MODES
    per_image_batch_size: int = 1
    universal_batch_size: int = 2
    seed: int = 111

    def __post_init__(self) -> None:
        self.feature_layers = tuple(int(layer) for layer in self.feature_layers)
        self.scopes = _tuple(self.scopes)
        self.directions = _tuple(self.directions)
        self.loss_modes = _tuple(self.loss_modes)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 < self.epsilon <= 1.0:
            raise ValueError("epsilon must be in (0, 1]")
        if not 0.0 < self.step_size <= 1.0:
            raise ValueError("step_size must be in (0, 1]")
        if self.steps <= 0 or self.universal_steps <= 0:
            raise ValueError("steps and universal_steps must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.per_image_batch_size <= 0 or self.universal_batch_size <= 0:
            raise ValueError("attack batch sizes must be positive")
        for value, valid, name in (
            (self.scopes, VALID_SCOPES, "scopes"),
            (self.directions, VALID_DIRECTIONS, "directions"),
            (self.loss_modes, VALID_LOSS_MODES, "loss_modes"),
        ):
            if not value:
                raise ValueError(f"{name} cannot be empty")
            unknown = sorted(set(value) - set(valid))
            if unknown:
                raise ValueError(f"Unknown {name}: {unknown}. Valid values: {valid}")
        if not self.feature_layers or any(layer <= 0 for layer in self.feature_layers):
            raise ValueError("feature_layers must contain positive layer indices")
        if self.global_weight < 0 or self.local_weight < 0:
            raise ValueError("loss weights cannot be negative")
        if self.global_weight + self.local_weight <= 0:
            raise ValueError("at least one loss weight must be positive")
        if not 0.0 <= self.local_background_weight <= 1.0:
            raise ValueError("local_background_weight must be in [0, 1]")
        self.normal_local_target = str(self.normal_local_target)
        if self.normal_local_target not in VALID_NORMAL_LOCAL_TARGETS:
            raise ValueError(
                "normal_local_target must be one of "
                f"{VALID_NORMAL_LOCAL_TARGETS}, got {self.normal_local_target!r}"
            )
        if not 0.0 < self.normal_target_region_fraction <= 1.0:
            raise ValueError("normal_target_region_fraction must be in (0, 1]")
        if not 0.0 <= self.normal_target_center_x <= 1.0:
            raise ValueError("normal_target_center_x must be in [0, 1]")
        if not 0.0 <= self.normal_target_center_y <= 1.0:
            raise ValueError("normal_target_center_y must be in [0, 1]")
        if self.local_focal_weight < 0 or self.local_dice_weight < 0:
            raise ValueError("local focal/dice weights cannot be negative")
        if self.local_focal_weight + self.local_dice_weight <= 0:
            raise ValueError("at least one local focal/dice weight must be positive")
        if self.local_focal_gamma < 0:
            raise ValueError("local_focal_gamma cannot be negative")
        if self.local_dice_smooth <= 0:
            raise ValueError("local_dice_smooth must be positive")
        self.step_size_schedule = str(self.step_size_schedule)
        if self.step_size_schedule not in VALID_STEP_SIZE_SCHEDULES:
            raise ValueError(
                "step_size_schedule must be one of "
                f"{VALID_STEP_SIZE_SCHEDULES}, got {self.step_size_schedule!r}"
            )
        if not 0.0 < self.step_size_min_ratio <= 1.0:
            raise ValueError("step_size_min_ratio must be in (0, 1]")
        if self.diagnostic_interval <= 0:
            raise ValueError("diagnostic_interval must be positive")


@dataclass
class ExperimentConfig:
    """End-to-end benchmark settings."""

    mvtec_root: Optional[str]
    output_root: str
    anomalyclip_root: str
    anomalyclip_checkpoint: str
    dataset: str = "mvtec"
    visa_root: Optional[str] = None
    device: str = "cuda"
    target_model: str = "AnomalyCLIP"
    categories: Optional[Tuple[str, ...]] = None
    source_categories: Optional[Tuple[str, ...]] = None
    target_batch_size: int = 2
    metric_size: int = 518
    aupro_fpr_limit: float = 0.30
    aupro_max_thresholds: int = 200
    anomaly_map_sigma: float = 4.0
    compute_lpips: bool = True
    lpips_backbone: str = "alex"
    save_universal_perturbations: bool = True
    save_adversarial_examples: int = 0
    universal_protocol: str = "transductive"
    fit_fraction: float = 0.5
    split_seed: int = 111
    diagnostic_max_samples: int = 64
    threshold_mode: str = "normal_train_quantile"
    threshold_quantile: float = 0.95
    thresholds_path: Optional[str] = None
    # A map-direction success flag/rate is only reported when this threshold
    # has been frozen before evaluation (for example from a held-out,
    # norm-matched random-perturbation baseline). Continuous shifts are always
    # reported and remain the primary local-map results.
    map_success_min_mean_shift: Optional[float] = None
    map_success_min_pixel_fraction: float = 0.5
    map_false_positive_threshold: float = 2.0
    localization_success_min_p_ap_drop: float = 0.0
    max_samples_per_category: Optional[int] = None
    use_split_manifest: bool = False
    split_manifest_csv: Optional[str] = None
    split_manifest_json: Optional[str] = None
    attack: AttackConfig = field(default_factory=AttackConfig)
    target_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataset = str(self.dataset).lower()
        if self.dataset not in VALID_DATASETS:
            raise ValueError(
                f"dataset must be one of {VALID_DATASETS}, got {self.dataset!r}"
            )
        if (
            self.dataset in {"mvtec", "both", "mvtec_to_visa", "visa_to_mvtec"}
            and not self.mvtec_root
        ):
            raise ValueError(f"mvtec_root is required when dataset={self.dataset!r}")
        if (
            self.dataset in {"visa", "both", "mvtec_to_visa", "visa_to_mvtec"}
            and not self.visa_root
        ):
            raise ValueError(f"visa_root is required when dataset={self.dataset!r}")
        if self.categories is not None:
            self.categories = tuple(self.categories)
        if self.source_categories is not None:
            self.source_categories = tuple(self.source_categories)
        if not self.is_cross_dataset and self.source_categories is not None:
            raise ValueError(
                "source_categories is only valid for cross-dataset transfer modes"
            )
        if self.target_batch_size <= 0:
            raise ValueError("target_batch_size must be positive")
        if self.metric_size <= 0:
            raise ValueError("metric_size must be positive")
        if not 0.0 < self.aupro_fpr_limit <= 1.0:
            raise ValueError("aupro_fpr_limit must be in (0, 1]")
        if self.aupro_max_thresholds < 2:
            raise ValueError("aupro_max_thresholds must be at least 2")
        if self.anomaly_map_sigma < 0:
            raise ValueError("anomaly_map_sigma cannot be negative")
        if self.save_adversarial_examples < 0:
            raise ValueError("save_adversarial_examples cannot be negative")
        self.universal_protocol = str(self.universal_protocol)
        if self.universal_protocol not in VALID_UNIVERSAL_PROTOCOLS:
            raise ValueError(
                "universal_protocol must be one of "
                f"{VALID_UNIVERSAL_PROTOCOLS}, got {self.universal_protocol!r}"
            )
        if not 0.0 < self.fit_fraction < 1.0:
            raise ValueError("fit_fraction must be in (0, 1)")
        if self.diagnostic_max_samples <= 0:
            raise ValueError("diagnostic_max_samples must be positive")
        self.threshold_mode = str(self.threshold_mode)
        if self.threshold_mode not in VALID_THRESHOLD_MODES:
            raise ValueError(
                "threshold_mode must be one of "
                f"{VALID_THRESHOLD_MODES}, got {self.threshold_mode!r}"
            )
        if not 0.0 < self.threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")
        if self.thresholds_path is not None:
            self.thresholds_path = str(self.thresholds_path)
        if (
            self.map_success_min_mean_shift is not None
            and self.map_success_min_mean_shift < 0
        ):
            raise ValueError("map_success_min_mean_shift cannot be negative")
        if not 0.0 <= self.map_success_min_pixel_fraction <= 1.0:
            raise ValueError("map_success_min_pixel_fraction must be in [0, 1]")
        if not math.isfinite(self.map_false_positive_threshold):
            raise ValueError("map_false_positive_threshold must be finite")
        if self.localization_success_min_p_ap_drop < 0:
            raise ValueError("localization_success_min_p_ap_drop cannot be negative")
        if (
            self.max_samples_per_category is not None
            and self.max_samples_per_category < 2
        ):
            raise ValueError(
                "max_samples_per_category must be at least 2 so both labels remain present"
            )
        if self.split_manifest_csv is not None:
            self.split_manifest_csv = str(self.split_manifest_csv)
        if self.split_manifest_json is not None:
            self.split_manifest_json = str(self.split_manifest_json)
        if bool(self.split_manifest_csv) != bool(self.split_manifest_json):
            raise ValueError(
                "split_manifest_csv and split_manifest_json must be provided together"
            )
        if (
            self.use_split_manifest
            and self.split_manifest_csv is None
            and self.split_manifest_json is None
        ):
            artifact_name = (
                f"{self.manifest_dataset}_matched_test_per_category_v1_"
                f"seed{self.split_seed}"
            )
            split_root = Path(__file__).resolve().parents[1] / "splits"
            self.split_manifest_csv = str(split_root / f"{artifact_name}.csv")
            self.split_manifest_json = str(split_root / f"{artifact_name}.json")
        if self.use_split_manifest and not self.split_manifest_csv:
            raise ValueError(
                "use_split_manifest=True requires split_manifest_csv and "
                "split_manifest_json"
            )
        if (
            self.use_split_manifest
            and self.max_samples_per_category is not None
            and self.max_samples_per_category < 4
        ):
            raise ValueError(
                "Manifest-based max_samples_per_category must be at least 4 "
                "to retain normal/anomaly x fit/evaluation strata"
            )
        if self.is_cross_dataset:
            non_dataset_scopes = sorted(set(self.attack.scopes) - {"dataset"})
            if non_dataset_scopes:
                raise ValueError(
                    "Cross-dataset transfer supports only the universal 'dataset' "
                    f"scope; remove {non_dataset_scopes}"
                )
            if self.universal_protocol != "held_out":
                raise ValueError(
                    "Cross-dataset transfer requires universal_protocol='held_out'"
                )

    @property
    def is_cross_dataset(self) -> bool:
        return self.dataset in {"mvtec_to_visa", "visa_to_mvtec"}

    @property
    def source_dataset(self) -> str:
        if self.dataset == "mvtec_to_visa":
            return "mvtec"
        if self.dataset == "visa_to_mvtec":
            return "visa"
        return self.dataset

    @property
    def evaluation_dataset(self) -> str:
        if self.dataset == "mvtec_to_visa":
            return "visa"
        if self.dataset == "visa_to_mvtec":
            return "mvtec"
        return self.dataset

    @property
    def manifest_dataset(self) -> str:
        return "both" if self.is_cross_dataset else self.dataset

    @property
    def output_path(self) -> Path:
        return Path(self.output_root).expanduser()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
