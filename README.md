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
      + LOCAL_DICE_WEIGHT  * target_class_region_dice
```

Both terms follow AnomalyCLIP's focal-plus-Dice segmentation objective, adapted
from *training a predictor against a ground-truth mask* to *steering a
perturbation toward a chosen class inside a chosen region*. The mask is
therefore used as a spatial weight `w = LOCAL_BACKGROUND_WEIGHT + (1 -
LOCAL_BACKGROUND_WEIGHT) * mask` rather than as a regression target:

- `target_class_focal` is standard focal loss, `(1 - p_target)^gamma * CE`,
  averaged over tokens with weights `w` in place of a scalar alpha.
- `target_class_region_dice` is `1 - (2*sum(p*w) + s) / (sum(p*w) + sum(w) + s)`,
  with `LOCAL_DICE_SMOOTH=1.0` matching the hardcoded `smooth = 1` of
  AnomalyCLIP's `BinaryDiceLoss`, which computes
  `1 - (2*sum(p*g) + 1) / (sum(p) + sum(g) + 1)`.

Two differences from AnomalyCLIP are deliberate and worth stating when citing
this objective:

1. The denominator uses `sum(p*w)` where `BinaryDiceLoss` uses the unweighted
   `sum(p)`. The term is therefore a strictly decreasing function of `sum(p*w)`
   alone: it rewards target-class probability inside the weighted region but
   does not penalize it outside.
2. AnomalyCLIP applies Dice twice per layer, on the abnormal channel against
   `gt` and on the normal channel against `1 - gt`, so its complement is scored
   by the second call. This generator scores only the requested target class,
   because the opposite class is what the attack is trying to move away from.

- `normal_to_abnormal`: a configurable fixed synthetic region is targeted as
  anomalous. The default is a centred square spanning 25% of each image side.
  Set `NORMAL_LOCAL_TARGET=full_image` only to reproduce the previous behavior.
- `abnormal_to_normal`: the ground-truth defect mask receives weight `1.0` and
  background receives `LOCAL_BACKGROUND_WEIGHT`. The target-class probability
  is normal, so the objective suppresses the known defect region.
- `combined`: the segmentation-aware local objective is combined with global
  targeted cross entropy using the existing global/local weights.

The attack still uses a frozen public CLIP image/text backbone. In learnable-
prompt setups, only pre-trained shallow context tensors are restored; the CLIP
weights remain frozen. The generator never loads or differentiates through an
evaluated anomaly detector.

### Text prompts used by the surrogate

Frozen setups use **WinCLIP's published compositional prompt ensemble,
verbatim and in order** — the same vocabulary the `backbone_eval` harness
encodes under its `fixed` prompt mode, so the surrogate and the evaluated
backbone read identical text:

- 22 photographic templates (`a photo of a {}.`, `a cropped photo of the {}.`,
  `a photo of the {} for anomaly detection.`, …)
- 7 normal states (`{}`, `flawless {}`, `{} without defect`, …)
- 4 abnormal states (`damaged {}`, `{} with flaw`, `{} with defect`,
  `{} with damage`)

That is 154 normal and 88 abnormal prompts per category. The **real category
name** fills every template (`bottle`, `metal nut`), not a fixed `object`
label, which is what separates this from the object-agnostic learnable setups.

Frozen public CLIP encodes and normalizes every prompt. Matching WinCLIP and
`backbone_eval`, the normalized embeddings are averaged separately for the
normal and abnormal classes, and each mean is normalized again to form two
prototype embeddings. Image and patch features are compared directly with
those two prototypes.

Because `attack_code_sha256` covers only `attacks.py`, the vocabulary is
fingerprinted separately as `prompt_ensemble_sha256`, and the reduction method
is recorded as `prompt_aggregation`, in every artifact and manifest row.
Changing either therefore invalidates artifact reuse instead of silently
altering results.

These frozen-prompt setups are labelled `frozen_winclip` in artifact manifests.
Artifacts generated with the previous compact vocabulary or log-mean-exp
aggregation are incompatible and must be regenerated.

### Object-agnostic learnable prompts

Every frozen setup also has a separate counterpart ending in
`_learnable_prompt`. These setups load the dataset-specific shallow checkpoints
produced by
[`object-agnostic-prompt-training`](https://github.com/Parsagh05/object-agnostic-prompt-training):

- MVTec: `artifacts/prompts/mvtec/prompts_epoch15.pt`
- VisA: `artifacts/prompts/visa/prompts_epoch15.pt`

Set `LEARNABLE_PROMPT_MVTEC_CHECKPOINT` and
`LEARNABLE_PROMPT_VISA_CHECKPOINT` to their Kaggle paths. The implementation is
[CoOp](https://github.com/KaiyangZhou/CoOp)-style: it inserts the learned normal
and abnormal context tensors into the ordinary public-CLIP text encoder. The
prompts are object-agnostic, so one
normal/abnormal pair is reused across all categories in its source dataset. A
fixed textual suffix such as `object.` or `damaged object.` is retained from
the checkpoint configuration. There is no deep token tuning inside text
transformer layers, and prompts are not retrained during attack generation.

The loader validates schema version, source dataset, context shape, CLIP text
width, `category_specific=false`, and `deep_text_prompt_tuning=false`. The
checkpoint SHA-256, epoch, suffixes, and prompt configuration are recorded in
every artifact and manifest.

## Relaxed margin/TopK objective

The four original setup IDs keep the segmentation-aware loss above. Four new
`*_margin_topk` setup IDs select a separate relaxed objective:

- Image-level score: `s(x) = z_abnormal(x) - z_normal(x)`.
- `normal_to_abnormal` minimizes `-s(x + delta)`, which maximizes the abnormal
  margin; `abnormal_to_normal` minimizes `s(x + delta)`.
- Pixel-level loss applies the same direction-aware sign to `TopK(H(x+delta))`,
  where `H` is the patch-token abnormal-minus-normal margin map averaged across
  selected CLIP layers.
- This formulation is location-free and does not read ground-truth masks.

`MARGIN_TOPK_FRACTION_NORMAL_TO_ABNORMAL` defaults to 0.20 and
`MARGIN_TOPK_FRACTION_ABNORMAL_TO_NORMAL` defaults to 0.40. The global, local,
and combined attack modes remain separate within every setup.

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

Dataset selection separates optimization from delivery. `SOURCE_DATASETS`
controls which attack-training images may optimize perturbations, while
`EVALUATION_DATASETS` controls the held-out targets for per-dataset universal
transfer. For example, `SOURCE_DATASETS=mvtec` and
`EVALUATION_DATASETS=mvtec,visa` optimizes one MVTec delta and records both
MVTec-to-MVTec and MVTec-to-VisA delivery rows referencing that same file and
checksum. The protocol CSV includes fixed evaluation IDs for both datasets,
but no VisA image enters optimization. Per-category and per-image outputs use
`SOURCE_DATASETS` only and remain same-dataset.

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
- The unperturbed image is the baseline checkpoint in every scope, so reported
  initial losses and loss reductions are measured from `delta=0` rather than
  from the random start, and a run that never beats "no attack" is reported as
  such instead of being credited with the random start's loss.
- Universal scopes use mixed precision only on bf16-capable hardware. Sign-PGD
  reads `gradient.sign()`, so an underflowed fp16 gradient would silently zero
  part of the update; the attack falls back to fp32 elsewhere.
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

The four scopes can be selected independently with `RUN_PER_DATASET`,
`RUN_CROSS_DATASET`, `RUN_PER_CATEGORY`, and `RUN_PER_IMAGE`.

`RUN_PER_DATASET` delivers a universal delta to held-out images of its own
source dataset; `RUN_CROSS_DATASET` delivers the same delta to the other
dataset. They share one optimization pass, so enabling both costs no more GPU
time than enabling either alone, and each writes a self-contained bundle with
its own copy of the deltas. Long Kaggle runs should normally run
one scope per session. Existing `.pt` files are safely resumed only when every
reproducibility field matches.

## Setup matrix

`RUN_SETUPS` selects one or more independent configurations:

`PROMPT_SETUP` independently chooses the prompt family:

- `PROMPT_SETUP=frozen` runs only the selected frozen WinCLIP setups.
- `PROMPT_SETUP=learnable` runs only their object-agnostic learnable counterparts.
- `PROMPT_SETUP=both` runs both families and is the default.

For example, `RUN_SETUPS=all PROMPT_SETUP=learnable` runs all eight learnable
configurations. A frozen base ID can also act as the loss/steps/epsilon choice:
`RUN_SETUPS=steps500_eps2_margin_topk PROMPT_SETUP=learnable` automatically
runs `steps500_eps2_margin_topk_learnable_prompt`. With `PROMPT_SETUP=both`,
the same base ID runs both prompt variants.

| Base setup | Loss | PGD steps | Linf epsilon |
|---|---|---:|---:|
| `steps500_eps2` | legacy `ce_focal_dice` | 500 | 2/255 |
| `steps500_eps4` | legacy `ce_focal_dice` | 500 | 4/255 |
| `steps800_eps2` | legacy `ce_focal_dice` | 800 | 2/255 |
| `steps800_eps4` | legacy `ce_focal_dice` | 800 | 4/255 |
| `steps500_eps2_margin_topk` | relaxed `margin_topk` | 500 | 2/255 |
| `steps500_eps4_margin_topk` | relaxed `margin_topk` | 500 | 4/255 |
| `steps800_eps2_margin_topk` | relaxed `margin_topk` | 800 | 2/255 |
| `steps800_eps4_margin_topk` | relaxed `margin_topk` | 800 | 4/255 |

Each base ID above uses frozen WinCLIP prompts. Append `_learnable_prompt` to
any base ID to run exactly the same loss, steps, epsilon, and attack scopes with
the object-agnostic learned checkpoint. Thus `RUN_SETUPS=all` runs 16 isolated
setups: eight frozen and eight learnable.

### The setup matrix is a grid, not a list

`setup_catalog.py` generates the matrix as the Cartesian product of four
lists, so widening a sweep means editing one of them rather than writing new
entries:

| List | Default | Override |
|---|---|---|
| step counts | `500, 800` | `SETUP_STEPS="800:200:100"` |
| Linf budgets | `2/255, 4/255` | `SETUP_EPSILONS="2/255,4/255,8/255"` |
| loss formulations | `ce_focal_dice, margin_topk` | fixed |
| prompt families | `frozen_winclip, learnable_object_agnostic` | fixed |

The defaults reproduce the original 16 setups exactly. `SETUP_STEPS="1200"`
gives 8 setups all named `steps1200_*`; `SETUP_STEPS="500,800,1200"` gives 24.
Because IDs are derived, generated entries name themselves and nothing else
needs editing.

### Steps are per scope

A `SETUP_STEPS` entry is a `dataset:category:image` triple. The scopes fit a
very different number of images per delta, so a single count either
undertrains the per-dataset scope or massively over-optimizes the others:
at 500 steps a per-dataset delta makes 4.5 passes over its 224 training
images, a per-category delta 286 passes over its ~14, and a per-image delta
500 passes over the single image it attacks.

`SETUP_STEPS="800:200:100"` therefore names itself
`steps800_cat200_img100_eps2`, and each scope receives its own count.
cross-dataset takes no value of its own because it delivers the per-dataset
delta. A bare number keeps the compact `steps800_eps2` name and gives every
scope the same count, so existing output names are unchanged.

Note that per-image fits the very image it attacks, so "overfitting" does
not apply there; its step count is purely a convergence choice. Checkpoint
selection uses the attack-training loss, so it cannot detect overfitting in
the universal scopes either -- the step count is the only regulariser.

### Setup IDs are derived, not stored

A setup ID is a pure function of the settings that change the work, so a run
can never be filed under a name describing different parameters:

```
steps + epsilon + [margin_topk] + [trainNN] + [learnable_prompt]
```

Overriding the step count renames the output on its own. `SMOKE_STEPS=1200`
against `steps500_eps4_margin_topk` writes to `steps1200_eps4_margin_topk`,
and `audit_generation.py` applies the same derivation, so it looks where the
run actually wrote. When an override makes two catalog rows resolve to
the same name, the launcher keeps the first and reports the collapse instead
of letting them overwrite each other.

`ATTACK_TRAIN_FRACTION` is folded in the same way: any value below 1.00 adds a
`_trainNN` component, so a 20% run lands in `..._train20` and cannot overwrite
or pool with the 100% run. A full run adds nothing, keeping existing names.

Do not set `PER_DATASET_STEPS` or the other per-scope step variables directly.
They bypass the derivation and change the work without changing the name.
Other knobs (step size, loss weights, batch sizes, TopK fractions) are not in
the ID; they are recorded in every artifact's metadata and checked by the
reuse guard, so they are safe within one output tree but not across merged
trees.

The selected step count and epsilon are applied consistently to every enabled
scope: per-dataset, per-category, and per-image. On Kaggle, select one setup,
one dataset, and normally one scope per saved session. Per-image generation is
especially expensive because every held-out image receives its own 500- or
800-step perturbation.

Use `SOURCE_DATASETS=mvtec` and `EVALUATION_DATASETS=mvtec,visa` for a MVTec
source attack transferred to both datasets. In the Kaggle notebook, set
`SOURCE_DATASETS = ('mvtec',)` and `EVALUATION_DATASETS = ('mvtec', 'visa')`.
Each source/evaluation selection uses a separate output/protocol directory.

## Outputs

Outputs are first separated by prompt family and then by setup ID:

```text
OUTPUT_BASE/setups/
├── frozen_prompt/
│   └── <frozen_setup_id>/
└── learnable_prompt/
    └── <learnable_setup_id>/
```

Each setup directory contains its own protocol CSVs, logs, uncompressed
bundles, and archives:

- `canonical_clip_per_dataset/`
- `canonical_clip_cross_dataset/`
- `canonical_clip_per_category/`
- `canonical_clip_per_image/`
- `canonical_clip_per_dataset_<datasets>_<setup_id>.zip`
- `canonical_clip_per_category_<datasets>_<setup_id>.zip`
- `canonical_clip_per_image_<datasets>_<setup_id>.zip`

After every selected setup passes the generation audit, the launcher also
creates `OUTPUT_BASE/full_outputs.zip`. This combined archive contains the
complete `setups/` directory tree, including perturbations, manifests,
diagnostics, protocols, and logs. Existing per-scope ZIP files are not nested
inside it, avoiding duplicate copies of the same perturbations.

For example, a MVTec dataset-level run is packaged as
`canonical_clip_per_dataset_mvtec_steps500_eps2.zip`. Dataset, scope, steps,
epsilon, and loss setup remain separate. A relaxed-loss run uses a distinct
name such as `canonical_clip_per_dataset_mvtec_steps500_eps2_margin_topk.zip`
and cannot overwrite the legacy setup.
Likewise, a learnable-prompt run has a distinct name such as
`canonical_clip_per_dataset_mvtec_steps500_eps2_margin_topk_learnable_prompt.zip`.

Do not merge these archives with the old `canonical_clip_*` bundles under the
same dataset version. Publish them as a new Kaggle dataset version and rerun the
black-box evaluations before drawing conclusions from local or combined losses.

The ready-to-run Kaggle notebook is
`kaggle_generate_corrected_perturbations.ipynb` in this directory.
