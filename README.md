# Corrected adversarial perturbation generator (v2)

This directory regenerates the fixed CLIP-surrogate perturbations used by the
black-box evaluation pipeline. Version 2 replaces patch-wise mean cross entropy
with a segmentation-aware local objective and writes to new artifact names, so
it cannot silently reuse or overwrite the original Kaggle dataset.

This project is self-contained: its local `adversarial_harness/` package owns
the attack, configuration, dataset, CLIP adapter, and prompt code required by
the generator.

## Local objective

For every selected CLIP layer and spatial token, the generator forms normal and
abnormal logits from the frozen public-CLIP prompt ensemble. The requested
target class is optimized with:

```
local = LOCAL_FOCAL_WEIGHT * target_class_focal
      + LOCAL_DICE_WEIGHT  * target_class_soft_dice
```

- `normal_to_abnormal`: a configurable fixed synthetic region is targeted as
  anomalous. The default is a centred square spanning 25% of each image side.
  Set `NORMAL_LOCAL_TARGET=full_image` only to reproduce the previous behavior.
- `abnormal_to_normal`: the ground-truth defect mask receives weight `1.0` and
  background receives `LOCAL_BACKGROUND_WEIGHT`. The target-class probability
  is normal, so the objective suppresses the known defect region.
- `combined`: the segmentation-aware local objective is combined with global
  targeted cross entropy using the existing global/local weights.

The attack still uses only the frozen public CLIP surrogate. It never loads or
differentiates through an evaluated anomaly detector.

## Balanced protocol

For both MVTec and VisA, each category is deterministically downsampled to the
same number of normal and anomalous test images. Each label is then split into
attack-training and held-out evaluation partitions with the same seed and
fraction. Consequently, both attack directions use equal counts within every
category and partition. The discarded surplus images are not used by either
partition.

Per-dataset and per-category perturbations are fitted only on the balanced
attack-training half. Per-image attacks are instance-specific rather than
trained universal perturbations: they operate only on the balanced held-out
evaluation half, with one aligned delta per image. Both directions therefore
still use matched counts, while per-image mode correctly reports zero
attack-training images.

## Optimization safeguards

- Dataset-level attacks default to a T4-safe batch of 2.
- PGD uses cosine step-size decay and smaller initial steps.
- The complete selected attack-training set is evaluated periodically. Random
  batch loss is labelled separately and is never presented as a convergence
  curve.
- Gradient norms, Linf-bound saturation, initial/final focal and Dice losses,
  and the full universal-optimization history are stored in artifact metadata.
- The checkpoint with the lowest complete attack-training loss is saved, rather
  than blindly saving the last stochastic iterate. The held-out evaluation
  split is never used for checkpoint selection.
- Every bundle includes `optimization_diagnostics.csv`.
- Artifact reuse checks include all loss/schedule settings, generator hashes,
  repository commit, and the pinned AnomalyCLIP commit.

## Run locally

Edit paths in `config.sh`, or override them as environment variables:

```bash
export MVTEC_ROOT=/data/mvtec_anomaly_detection
export VISA_ROOT=/data/VisA_20220922
export OUTPUT_BASE=/data/attack_generation
bash train.sh
```

`train.sh` uses the active Python interpreter directly by default, which is
required on Kaggle images where `venv`/`ensurepip` may be unavailable. Set
`USE_VENV=true` on a local server if an isolated virtual environment is wanted;
if its `pip` bootstrap fails, the launcher safely falls back to the active
interpreter. `PYTHON_BIN` can select a specific interpreter explicitly.

The three modes can be selected independently with `RUN_PER_DATASET`,
`RUN_PER_CATEGORY`, and `RUN_PER_IMAGE`. Long Kaggle runs should normally run
one scope per session. Existing `.pt` files are safely resumed only when every
reproducibility field matches.

## Step/epsilon matrix

`RUN_SETUPS` selects one or more independent configurations:

| Setup | PGD steps | Linf epsilon |
|---|---:|---:|
| `steps500_eps2` | 500 | 2/255 |
| `steps500_eps4` | 500 | 4/255 |
| `steps800_eps2` | 800 | 2/255 |
| `steps800_eps4` | 800 | 4/255 |

The selected step count and epsilon are applied consistently to every enabled
scope: per-dataset, per-category, and per-image. On Kaggle, select one setup,
one dataset, and normally one scope per saved session. Per-image generation is
especially expensive because every held-out image receives its own 500- or
800-step perturbation.

Use `GENERATION_DATASETS=mvtec` or `GENERATION_DATASETS=visa` to split the two
collections across sessions. In the Kaggle notebook, set `DATASETS =
('mvtec',)` or `('visa',)`. Each selection uses a separate output/protocol
directory, preventing a split generated for one selection from being reused by
another.

## Outputs

Each setup is independent under `OUTPUT_BASE/setups/<setup_id>/` and contains
its own protocol CSVs, logs, uncompressed bundles, and archives:

- `canonical_clip_per_dataset/`
- `canonical_clip_per_category/`
- `canonical_clip_per_image/`
- `canonical_clip_per_dataset_<datasets>_<setup_id>.zip`
- `canonical_clip_per_category_<datasets>_<setup_id>.zip`
- `canonical_clip_per_image_<datasets>_<setup_id>.zip`

For example, a MVTec dataset-level run is packaged as
`canonical_clip_per_dataset_mvtec_steps500_eps2.zip`. Dataset, scope, steps,
and epsilon remain separate and visible without unnecessary loss-version text.

Do not merge these archives with the old `canonical_clip_*` bundles under the
same dataset version. Publish them as a new Kaggle dataset version and rerun the
black-box evaluations before drawing conclusions from local or combined losses.

The ready-to-run Kaggle notebook is
`kaggle_generate_corrected_perturbations.ipynb` in this directory.
