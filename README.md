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

The checkpoints are resolved automatically; see
[Prompt checkpoints follow the split](#prompt-checkpoints-follow-the-split).
The implementation is
[CoOp](https://github.com/KaiyangZhou/CoOp)-style: it inserts the learned normal
and abnormal context tensors into the ordinary public-CLIP text encoder. The
prompts are object-agnostic, so one
normal/abnormal pair is reused across all categories in its source dataset. A
fixed textual suffix such as `object.` or `damaged object.` is retained from
the checkpoint configuration. There is no deep token tuning inside text
transformer layers, and prompts are not retrained during attack generation.

### Prompt checkpoints follow the split

Prompts fitted under one `SPLIT_PROTOCOL` or `ATTACK_TRAIN_FRACTION` saw a
different set of images than a run using another, so pairing them means the
prompts have already seen part of what this run holds out. Neither repository
raises on that by itself.

So the launcher resolves the checkpoint rather than trusting a configured path.
Before each learnable setup it looks for one matching this run's dataset, split
protocol, attack-train fraction, split seed and epoch count, and runs
`object-agnostic-prompt-training` when there is none. Every tree is laid out
the same way:

```
<root>/<protocol>[_trainNN]/<dataset>/prompts_epoch<N>.pt
```

The directory carries the same suffix the setup ID does, so `balanced` and
`full` never overwrite each other and a fraction sweep files each cohort
separately.

Roots are searched in order: `PROMPT_TRAINING_SEARCH_ROOTS` first, then
`PROMPT_TRAINING_OUTPUT_ROOT`. The search roots hold published prompts and are
read-only; the output root is the only place training ever writes. That split
matters on Kaggle, where the published dataset is mounted read-only, so a
published checkpoint that does not describe this run is replaced by a locally
trained one rather than failing on a read-only filesystem. The default search
root is the dataset holding the `balanced` and `full` cohorts:

```
PROMPT_TRAINING_SEARCH_ROOTS=/kaggle/input/learned-prompts/prompts
```

Training receives this run's own `attack_train_indices.csv` as its manifest, so
the prompts are fitted on the images the perturbation is optimized on by
construction; the training pipeline additionally asserts the split seed,
evaluation fraction and label policy stamped on those rows against its own
configuration and refuses to run when they disagree.

`LEARNABLE_PROMPT_MVTEC_CHECKPOINT` and `LEARNABLE_PROMPT_VISA_CHECKPOINT` still
work and take precedence, but only when the checkpoint they name describes this
run's split. A checkpoint that does not is reported and retrained rather than
used. Checkpoints written before the training pipeline recorded its split are
read as `balanced` at a full fraction, which is what they are.

Training code is taken from the tip of `PROMPT_TRAINING_BRANCH` at run time
rather than a pinned commit, and the commit it resolved to is logged and
recorded in the checkpoint's `manifest.json`. Set `PROMPT_TRAINING_ROOT` to a
local clone to skip the fetch, or
`PROMPT_TRAINING_EPOCHS` / `PROMPT_TRAINING_BATCH_SIZE` to change the budget.
The defaults reproduce the published checkpoints.

The loader validates schema version, source dataset, context shape, CLIP text
width, `category_specific=false`, and `deep_text_prompt_tuning=false`. The
checkpoint SHA-256, epoch, suffixes, and prompt configuration are recorded in
every artifact and manifest.

## Margin/TopK objective (default)

The margin/TopK objective is the default, so it adds nothing to a setup ID. The
segmentation-aware loss above remains available under `*_ce_focal_dice` IDs.
The default objective is:

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

### Hinging the per-image margin

`MARGIN_HINGE_DISPLACEMENT` bounds how much any single image can contribute to
a shared perturbation. It is empty by default, which leaves the margin
unbounded and reproduces the behaviour described above.

**What it does.** For image `i`, let `m_i` be its image-level margin
`z_abnormal - z_normal`, and `m_i^clean` the same quantity at `delta = 0`.
With `s = -1` for `normal_to_abnormal` and `s = +1` for `abnormal_to_normal`,
the per-image contribution becomes

```
contribution_i = max( s * m_i ,  s * m_i^clean - D )
```

where `D` is `MARGIN_HINGE_DISPLACEMENT`. Below the floor the term is a
constant, so a saturated image produces no gradient. The same floor is applied
to the local term, using each image's mean over its TopK patch margins in place
of `m_i`. The reported `global_margin` and `local_topk` diagnostics stay the
raw, unhinged values so they remain comparable with earlier runs.

Read plainly: an image stops pulling on the shared perturbation once it has
moved `D` toward the attacked class **from where that image started**.

**Why.** The margin is otherwise unbounded, so an image already far past the
decision boundary keeps producing gradient and drags the shared delta toward
over-serving samples that are already fooled, at the expense of borderline
ones. Two universal-attack papers identify this independently:

- Shafahi et al., *Universal Adversarial Training*, AAAI 2020
  ([arXiv:1811.11304](https://arxiv.org/abs/1811.11304)), section 3, clip the
  loss as `l_hat = min{ l(w, x_i + delta), beta }`, to "prevent any single
  image from dominating the objective, and giving us a better surrogate of
  misclassification accuracy". Their figure 5 ablation is the reason `D` is a
  tunable value rather than zero: no clipping was worst, but the most
  aggressive clipping was also worse than the middle setting.
- Zhang and Benz et al., *CD-UAP*, AAAI 2020
  ([arXiv:2010.03300](https://arxiv.org/abs/2010.03300)), equation 5, use a
  bounded logit loss `L_BL = ( L_c(x+delta) - max_{i != c} L_i(x+delta) )_+`
  with `(s)_+ = max(s, 0)`. That is a hinge on a difference of logits, which
  is exactly the shape of the margin term here, so it is the form this
  implementation follows.

**Why displacement rather than an absolute floor.** CD-UAP hinges at the
decision boundary, which is correct when the attacked network is the network
being optimized. Here the margins are measured on a frozen CLIP surrogate, and
they do not sit where an absolute floor would assume. Measured from completed
runs, as progress toward the attacked class where positive means past the
boundary:

| setup | term | clean | converged |
|---|---|---:|---:|
| frozen, normal to abnormal | global | +0.01 | +0.44 |
| frozen, abnormal to normal | local TopK | +0.12 | +0.18 |
| learnable, normal to abnormal | global | +0.15 | +3.05 |
| learnable, visa, normal to abnormal | local TopK | -4.74 | -3.78 |

A floor at the boundary would saturate every image of the abnormal-to-normal
local term at step one, because those images already start past it, and would
never bind at all for the learnable visa condition, which never crosses. The
learnable margins are also about ten times the frozen ones, so no single
absolute number serves both prompt families. Anchoring the floor to each
image's own clean margin removes all three problems: `D` means the same thing
in every condition.

**Scope.** The hinge applies to `margin_topk` only. The `ce_focal_dice` terms
are already bounded, so its setups never carry the tag. It is used by the
`per_dataset`, `cross_dataset` and `per_category` scopes, where one delta is
shared across images. `per_image` does not use it: a delta fitted to the single
image it attacks has no sharing to balance, so hinging would only stop it
early. That is also why per-image attack literature such as AutoAttack
([arXiv:2003.01690](https://arxiv.org/abs/2003.01690)) does not do this.

**Choosing a value.** `D` is in margin units, so read it off a completed run of
the same setup rather than guessing. `MARGIN_HINGE_DISPLACEMENT=0` reproduces
CD-UAP's hinge relative to the clean margin, stopping each image the moment it
improves at all. Larger values push further before saturating.

Every value names its own output (`0.25` gives `..._hinge0p25`), so an A/B
cannot overwrite itself:

```bash
RUN_SETUPS=ep7p14_cat100_img100_eps4 bash train.sh
MARGIN_HINGE_DISPLACEMENT=0.25 RUN_SETUPS=ep7p14_cat100_img100_eps4 bash train.sh
```

**Watch the saturated fraction.** Every optimization-diagnostics row carries
`global_margin_saturated_fraction` and `local_margin_saturated_fraction`. If
either reaches 1.0 the gradient is exactly zero, and because the update is
`sign(gradient)` the perturbation then stops moving for the rest of the run.
Checkpoint selection keeps the best delta, so nothing is corrupted, but the
remaining steps are wasted and `D` is too small for that condition.

## Split protocol: balanced or full

`SPLIT_PROTOCOL` selects how the test images are divided. Both split each
category independently and hold out `EVALUATION_FRACTION` of it.

**`balanced`** (default, historical). Per category it computes
`n_c = min(N_normal, N_abnormal)`, keeps `n_c` of each label and discards the
surplus, then splits each label by the fraction. Equal normal/abnormal counts,
at the cost of throwing images away: on MVTec this keeps 894 of 1725.

**`full`** keeps every image. Each label is still split by the same fraction,
so the split stays stratified and both labels appear on both sides, but the
category's natural class ratio is preserved. `bottle` stays 20 normal / 63
abnormal instead of being cut to 20 / 20. On MVTec this keeps all 1725.

```bash
export SPLIT_PROTOCOL=full            # keep every image
export EVALUATION_FRACTION=0.50       # share of each category/label held out
export FULL_DATA_CROSS=true           # both retained halves -> both retained halves
# Or reuse the per-dataset delta for a held-out transfer experiment:
export FULL_DATA_CROSS=false          # source attack_train -> target evaluation
export SPLIT_PROTOCOL=balanced        # back to the historical protocol
```

The two switches produce four cross-dataset behaviors:

| `SPLIT_PROTOCOL` | `FULL_DATA_CROSS` | Cross training | Cross evaluation |
|---|---:|---|---|
| `balanced` | `true` | complete balanced source cohort | complete balanced target cohort |
| `balanced` | `false` | balanced source `attack_train` | balanced target `evaluation` |
| `full` | `true` | complete natural source cohort | complete natural target cohort |
| `full` | `false` | full source `attack_train` | full target `evaluation` |

`FULL_DATA_CROSS=false` reuses the exact per-dataset delta. The flag never
changes the per-dataset scope.

One additional scope behaves differently under `full`:

- **per-image** covers every test image rather than only the held-out half,
  because it fits the very image it attacks and the split does not constrain it.

Per-dataset and per-category are unchanged in shape: train on the attack-train
part of each category, deliver to the held-out part.

The protocol is recorded so an evaluator can detect it:

- the setup ID contains `_fullcross` or `_halfcross`; `full` retains its
  existing `_full` protocol component while `balanced` adds no protocol component
- `label_balance_policy` on every protocol CSV row
- `complete_retained_indices.csv`, containing both deterministic partitions
  for every source or evaluation dataset, including evaluation-only targets
- `split_protocol` on every manifest and diagnostics row, plus `training_source`
  on per-dataset rows (`attack_train_partition` or `complete_source_dataset`)
- `full_data_cross`, `cross_data_mode`, `source_partition_policy`, and
  `target_partition_policy` on every cross-dataset manifest row

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
- PGD uses a flat step size by default; see [The step size is flat by default](#the-step-size-is-flat-by-default).
- The complete selected attack-training set is evaluated periodically. Random
  batch loss is labelled separately and is never presented as a convergence
  curve.
- Gradient norms, Linf-bound saturation, initial/final focal and Dice losses,
  and the full universal-optimization history are stored in artifact metadata.
- `CHECKPOINT_SELECTION=final`, the default, returns the last iterate, as the
  universal-attack papers do. Set `best` to instead keep the iterate with the
  lowest complete attack-training loss; the held-out evaluation split is never
  used for that selection either way. See
  [Which iterate is returned](#which-iterate-is-returned).
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

For example, `RUN_SETUPS=all PROMPT_SETUP=learnable` runs all four learnable
configurations. A frozen base ID can also act as the loss/epochs/epsilon choice:
`RUN_SETUPS=ep7p14_cat100_img100_eps2 PROMPT_SETUP=learnable` automatically
runs `ep7p14_cat100_img100_eps2_learnable_prompt`. With `PROMPT_SETUP=both`,
the same base ID runs both prompt variants.

| Base setup | Loss | Epoch budget | Linf epsilon |
|---|---|---:|---:|
| `ep7p14_cat100_img100_eps2` | default `margin_topk` | 7.14 / 100 / 100 | 2/255 |
| `ep7p14_cat100_img100_eps4` | default `margin_topk` | 7.14 / 100 / 100 | 4/255 |
| `ep7p14_cat100_img100_eps2_ce_focal_dice` | alternate `ce_focal_dice` | 7.14 / 100 / 100 | 2/255 |
| `ep7p14_cat100_img100_eps4_ce_focal_dice` | alternate `ce_focal_dice` | 7.14 / 100 / 100 | 4/255 |

The epoch budget column is the `dataset / category / image` triple. Each base ID
above uses frozen WinCLIP prompts. Append `_learnable_prompt` to any base ID to
run exactly the same loss, epochs, epsilon, and attack scopes with the
object-agnostic learned checkpoint. Thus `RUN_SETUPS=all` runs 8 isolated
setups: four frozen and four learnable.

### The setup matrix is a grid, not a list

`setup_catalog.py` generates the matrix as the Cartesian product of four
lists, so widening a sweep means editing one of them rather than writing new
entries:

| List | Default | Override |
|---|---|---|
| epoch budgets | `7.14:100:100` | `SETUP_EPOCHS="7.14:5:100:100"` |
| Linf budgets | `2/255, 4/255` | `SETUP_EPSILONS="2/255,4/255,8/255"` |
| loss formulations | `margin_topk` (default), `ce_focal_dice` | fixed |
| prompt families | `frozen_winclip, learnable_object_agnostic` | fixed |

`SETUP_EPOCHS="50"` gives 4 setups all named `ep50_*`;
`SETUP_EPOCHS="7.14:100:100,50"` gives 8. Because IDs are derived, generated
entries name themselves and nothing else needs editing. These are selection
IDs; the output's effective ID also records `_fullcross` or `_halfcross`.

### Budgets are epochs, and each scope gets its own

An epoch is one pass over the images that delta trains on, and the PGD step
count follows from it:

```
steps = ceil(epochs * ceil(n_images / batch_size))
```

The scopes fit a very different number of images per delta, so a shared budget
is meaningless in steps but meaningful in epochs. A `SETUP_EPOCHS` entry is a
`dataset:cross:category:image` budget because the right number of passes
differs per scope: the per-dataset delta generalises across 224 images and
saturates early, whereas a per-image delta fits the one image it attacks and
simply needs enough steps to converge.

The `cross` value matters only under `fullcross`, which optimises a separate
delta on the complete source rather than on the attack-train half. That cohort
is about twice the size, so the same epoch number buys about twice the steps
and twice the compute:

| delta | cohort | images | steps at 7.14 epochs |
|---|---|---:|---:|
| same_dataset | attack-train half | 704 | 2514 |
| cross_dataset | complete source | 1408 | 5027 |

That is what an epoch means, and it may be what you want, but without a
separate value there is no way to run cross-dataset shorter without also
shortening per-dataset. The component appears in the name only when it differs
from the dataset budget, so every existing output name is unchanged, and
`7.14:5:100:100` gives `ep7p14_cross5_cat100_img100`. Under `halfcross`
cross-dataset delivers the per-dataset delta and the value is unused. The
older three-part form still works, with cross inheriting the dataset budget.

The default `7.14:100:100` reproduces the historical 800 / 200 / 100 step
counts exactly at the default batch sizes, and names itself
`ep7p14_cat100_img100_eps2`. Cross-dataset has no separate epoch value: a
`halfcross` run delivers the per-dataset delta, while `fullcross` trains its
complete-cohort delta using the per-dataset epoch budget. A bare number gives
every scope the same budget and keeps the compact epoch component.

Because steps are derived, a category holding more images automatically gets
more steps at the same budget -- which is the point. Checkpoint selection uses
the attack-training loss, so it cannot detect overfitting in the universal
scopes; the epoch budget is the only regulariser.

### Setup IDs are derived, not stored

A setup ID is a pure function of the settings that change the work, so a run
can never be filed under a name describing different parameters:

```
epochs + epsilon + [ce_focal_dice | hinge] + [momentum] + [step_schedule]
       + [best] + [protocol] + cross_mode + [trainNN] + [learnable_prompt]
```

Overriding the epoch budget renames the output on its own. `SMOKE_EPOCHS=50`
against `ep7p14_cat100_img100_eps4` writes to `ep50_eps4_fullcross` under the
default balanced/full-cross configuration,
and `audit_generation.py` applies the same derivation, so it looks where the
run actually wrote. When an override makes two catalog rows resolve to
the same name, the launcher keeps the first and reports the collapse instead
of letting them overwrite each other.

### The step size is flat by default

`STEP_SIZE_SCHEDULE` is `constant`: one step size for the whole run. The
decaying schedules remain available as `linear` and `cosine`, and each names
itself in the setup ID (`..._linear_step`), because the schedule changes the
perturbation.

Two reasons for the default. The universal-attack literature does not use a
budget-normalised schedule -- UAP, UAT and CD-UAP either decay not at all or
delegate adaptation to Adam -- and decay was inherited from the per-image
setting. More practically, a decaying step depends on the total step count, so
the first N steps of a long run are not an N-step run: six epochs into a
twenty-epoch run the steps are still far larger than a finished six-epoch run's.
With a flat step the two are identical, because the initial perturbation and the
batch order depend only on `ATTACK_SEED` and not on the budget. One long run can
then stand in for several shorter ones, given delta snapshots at epoch
boundaries, which are not implemented yet.

`INITIAL_STEP_SIZE` stays `0.25/255`, which is epsilon/16 at epsilon 4/255. UAT
uses epsilon/8, so `0.5/255` is the literature-matching alternative. The step
size is recorded in every manifest but is not part of the setup ID, so sweeping
it needs a separate output tree.

### Momentum on the shared update

`MOMENTUM_DECAY` accumulates past gradients instead of stepping along the
current one alone. It is empty by default, which is plain sign-PGD.

**What it does.** With `g` the batch gradient and `m` starting at zero,

```
m <- MOMENTUM_DECAY * m + g
delta <- clip( delta - step_size * sign(m) )
```

The sign and the epsilon projection are unchanged, so the budget is untouched.
A decay of `0` makes `m` equal `g` and is exactly the behaviour without it.
`0.9` is the recommended value when enabling it. This is the gradient-momentum
form used by MI-FGSM (Dong et al., *Boosting Adversarial Attacks with
Momentum*, CVPR 2018) and by UAT's MSGD variant, not the iterate momentum of
APGD.

`m` starts at zero and evolves only from the step index, so momentum does not
break the prefix-consistency the flat step size buys.

**Scope.** The scopes that share one delta: `per_dataset`, `cross_dataset` and
`per_category`. `per_image` does not use it, matching how the hinge is scoped.

**Why not APGD's momentum.** APGD
([arXiv:2003.01690](https://arxiv.org/abs/2003.01690)) equation 2 puts momentum
on the iterate rather than the gradient, and says why: "Since in the early
iterations of APGD the step size is particularly large, we want to keep a bias
from the previous steps." APGD starts at twice epsilon. This pipeline runs a
flat epsilon/16 at epsilon 4/255, so APGD's initial step is 32 times larger
than ours and the problem its momentum solves does not arise here. APGD is also
a per-image attack whose momentum is entangled with its adaptive step halving
and its restart-from-best; taking one clause out of that loop is a different
algorithm, not APGD.

**What the evidence does and does not say.** UAT
([arXiv:1811.11304](https://arxiv.org/abs/1811.11304)) compares optimizers for
a universal perturbation on CIFAR-10, reporting accuracy after the attack where
lower is stronger: SGD 42.56, MSGD 13.08, ADAM 13.30, PGD 13.79. Read carefully
before relying on the 0.71 point gap. Their MSGD is momentum on the **raw**
gradient with no sign; their PGD is the sign with **no** momentum. The
combination implemented here, the sign of an accumulated gradient, is in
neither column. UAT also states that unless otherwise specified they use
sign-of-gradient PGD for generating universal perturbations, so their headline
results do not use momentum at all.

The better reason to try it is that momentum in the MI-FGSM sense exists to
improve **transferability**, and these perturbations are fitted on a frozen
CLIP surrogate and delivered to other detectors. Treat it as an ablation
switch, not as a setting with a guaranteed payoff.

**Interaction with the margin hinge.** They compose, but the decay decides
whether the hinge still does anything. A saturated image contributes no fresh
gradient, yet its earlier pull survives in `m` for about `1 / (1 - decay)`
steps:

| decay | memory | old pull down to 5% after |
|---:|---:|---:|
| 0.9 | 10 steps | 28 steps |
| 0.99 | 100 steps | 298 steps |
| 1.0 | never forgets | never |

Against an 800-step per-dataset budget, `0.9` clears a saturated image within
about 30 steps and the two settings coexist. `1.0`, which is MI-FGSM's
published convention, keeps that pull at full strength to the last step and
cancels the hinge outright. Do not copy the paper constant if the hinge is on.

One side effect is useful: with momentum, full saturation no longer freezes the
run, because `m` stays non-zero when `g` reaches zero and `sign(0) = 0` no
longer applies.

**Measuring whether it did anything.** Every optimization-diagnostics row
carries `momentum_gradient_cosine`, the cosine between the accumulated
direction and the current gradient. Near 1.0 means momentum is not changing the
step. It is `NaN` when momentum is off.

**Running the ablation.** Each decay names its own output (`0.9` gives
`..._mom0p9`), so nothing collides. Change one thing at a time, with the hinge
off, so the comparison measures momentum alone:

```bash
RUN_SETUPS=ep7p14_cat100_img100_eps4 bash train.sh
MOMENTUM_DECAY=0.9 RUN_SETUPS=ep7p14_cat100_img100_eps4 bash train.sh
```

### Which iterate is returned

`CHECKPOINT_SELECTION` decides which point of the optimization becomes the
published perturbation. It does not change the optimization itself: both modes
follow an identical trajectory and differ only in which step is kept.

**`final`, the default.** Return the delta at the last step, with no
selection.

**`best`.** Every `DIAGNOSTIC_INTERVAL` steps the current delta is scored over
the whole attack-train cohort, and the lowest-scoring iterate is retained. The
baseline for that comparison is the clean delta, so a run that never beats "no
attack" returns zeros rather than a perturbation that made its own surrogate
objective worse.

**Why `final` is the default.** Every universal-attack paper returns the final
iterate and none performs checkpoint selection: UAP (Moosavi-Dezfooli et al.,
CVPR 2017) algorithm 1 exits on reaching a target fooling rate, UAT
([arXiv:1811.11304](https://arxiv.org/abs/1811.11304)) algorithm 2 is a plain
epoch loop that updates, projects and returns, and CD-UAP
([arXiv:2010.03300](https://arxiv.org/abs/2010.03300)) algorithm 1 iterates a
fixed count and returns.

Best-iterate selection is the **per-image** convention. APGD
([arXiv:2003.01690](https://arxiv.org/abs/2003.01690)) tracks the best point
and even restarts from it whenever it halves the step size, and it is cheap
there because scoring a candidate is the forward pass already performed on the
single target image. For a universal attack, scoring a candidate costs a pass
over the entire cohort, which is why the universal literature skips it.

So `best` is stricter than any universal baseline and closer to the per-image
convention. That was this pipeline's original behaviour. It is defensible, but
it is a deviation, and a reader who assumes UAT's algorithm 2 will assume the
last iterate, which is why `final` is now the default and `best` is the option
that names itself.

**What `final` changes relative to `best`.**

- **The selection bias goes away.** `best` takes an argmin over roughly
  `universal_steps / DIAGNOSTIC_INTERVAL` candidates scored on `attack_train`,
  while the reported metrics come from the held-out half. Nothing leaks, but
  the selected advantage does not transfer perfectly. `final` has no such step.
- **The silent zero-delta case goes away.** Under `best`, a run that never
  beats clean publishes a zero perturbation, visible only as a near-zero attack
  success rate. `final` always returns the actual endpoint of the optimization.
- **`selected_step` becomes the last step** and `selected_diagnostic_loss`
  becomes the diagnostic loss at that step rather than the argmin over
  checkpoints. Both are recorded either way.

**The risk of the default.** A universal delta fluctuates from step to step,
so the last iterate is an arbitrary draw from that fluctuation rather than a
chosen point, and the guarantee that a published perturbation never scores
worse than no attack is gone. Report the spread across attack seeds, not only
the mean attack strength. If `final` is noisier across seeds, that is itself a
result, and `best` remains one variable away.

**Scope.** All four scopes, including the per-image path, which applies the
same rule to its batch loss.

**Diagnostics are unchanged.** Under `best` the periodic cohort pass is what
selection needs; under `final` it is only logging. Whether to widen
`DIAGNOSTIC_INTERVAL` or drop those passes to save the compute is a separate
decision from the selection rule, and this flag does not make it.

```bash
CHECKPOINT_SELECTION=best RUN_SETUPS=ep7p14_cat100_img100_eps4 bash train.sh
```

### Epoch snapshots: several budgets from one run

`SNAPSHOT_EPOCHS` lists shorter budgets to capture inside one run. It is
written exactly like `SETUP_EPOCHS`, as `dataset:category:image` triples, so
every snapshot names a complete setup of its own and each scope stops at its
own boundary. It is empty by default.

```bash
SETUP_EPOCHS="20:400:400" SNAPSHOT_EPOCHS="5:100:100,10:200:200"
```

That run produces three setups: `ep5_cat100_img100_...`,
`ep10_cat200_img200_...` and its own `ep20_cat400_img400_...`. Each entry has
to be a prefix of the run's budget, so no scope may exceed it and the triple
may not equal it, which would claim the run's own name.

A sweep over five, six, seven and eight epochs costs 26 epochs of optimization
as four runs. As snapshots inside one eight-epoch run it costs eight, and the
saving grows with the number of points. The snapshots are not approximations:
they are the runs they replace.

**The equivalence, and what it rests on.** A snapshot at epoch E inside a
longer run is byte-identical to a standalone E-epoch run only while four
conditions hold. Three are structural and one is enforced.

| condition | status |
|---|---|
| step size independent of the total budget | enforced: a decaying `STEP_SIZE_SCHEDULE` is refused |
| batch order and initialisation independent of the budget | holds: a fresh attacker per condition, seeded from `ATTACK_SEED` |
| integer epochs land on the same step index | holds: `ceil(n_images / batch_size)` is fixed by the cohort |
| the returned-iterate rule does not look ahead | handled for both selection modes |

The last one is the subtle one. Under `CHECKPOINT_SELECTION=final` a snapshot
is just the delta at that step. Under `best` the running best has to be taken
**at** the boundary, because the run's global best would otherwise be
contaminated by progress made after it, which fails silently and produces an
artifact that looks valid. A snapshot boundary also forces a diagnostic
evaluation regardless of `DIAGNOSTIC_INTERVAL`, since the standalone run it
stands in for always evaluates one on its final step.

The step-size condition is not a preference. Under the linear decay that used
to be the default, the step size was a function of the total budget, so a
twenty-epoch run at the end of epoch six was still stepping several times
larger than a finished six-epoch run and the two trajectories diverged from
the first iteration. That is also why the budget curves in APGD's figure 3 are
six distinct trajectories rather than one curve read at six points: its
checkpoints scale with the budget. None of UAP, UAT or CD-UAP uses a
budget-normalised schedule; that convention came from the per-image
literature, where sliceability is irrelevant because each attack targets one
image.

**The gate.** The claim is verified directly rather than argued. The test
suite runs a short budget standalone, runs a longer budget with a snapshot at
the short one, and compares the two deltas by SHA-256. They must match
exactly, and it runs for both selection modes. A companion test asserts the
opposite under a decaying schedule, so the precondition cannot quietly stop
being a precondition.

**Status.** The per-dataset optimizer captures snapshots, the configuration
parses and validates, and each budget derives its own setup ID. Writing those
captured deltas out as complete setup directories is not wired up yet, and
`per_category` and `per_image` do not capture snapshots yet either; see
[what remains](#snapshot-bundles-what-remains).

### Snapshot bundles: what remains

A snapshot directory is meant to be indistinguishable from a standalone run of
that budget, holding all four scopes. Two pieces are still missing.

- **The other two scopes.** `per_category` and `per_image` each optimize a
  delta of their own, so each needs the same capture the per-dataset scope
  now has. Both satisfy the equivalence conditions already: the per-category
  loop seeds its sampler per condition, and the per-image path draws no
  batches at all.
- **Writing the bundles.** One optimization pass has to produce several setup
  directories, which the launcher currently forbids by fixing one output
  directory per setup before a runner starts. The intended route reuses the
  existing bundling code untouched: each runner writes its snapshot deltas
  into the snapshot setup's own artifact path with that budget's metadata,
  and the launcher then replays the ordinary per-setup block for each
  snapshot budget, where the reuse guard finds the deltas already on disk and
  skips optimization. That guard compares configuration rather than results,
  so only `optimization_epochs` and the derived step count have to differ.

One open decision: with `OVERWRITE_EXISTING=false`, a snapshot whose setup ID
already exists from a standalone run should fail loudly on a SHA-256 mismatch
rather than skip or overwrite, because a mismatch means one of the four
conditions above has broken.

Snapshots do not change generation cost but multiply evaluation runs: a run
with two snapshots still needs two extra full evaluation passes downstream.

`ATTACK_TRAIN_FRACTION` is folded in the same way: any value below 1.00 adds a
`_trainNN` component, so a 20% run lands in `..._train20` and cannot overwrite
or pool with the 100% run. A full run adds nothing, keeping existing names.

Do not set `PER_DATASET_EPOCHS` or the other per-scope epoch variables
directly. They bypass the derivation and change the work without changing the
name.
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

```text
protocol/
├── attack_train_indices.csv
├── evaluation_test_indices.csv
└── complete_retained_indices.csv
```

The complete file is authoritative for `fullcross`; `halfcross` continues to
use the two role-specific files.

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

For example, a balanced/full-cross MVTec dataset-level run is packaged as
`canonical_clip_per_dataset_mvtec_ep7p14_cat100_img100_eps2_fullcross.zip`.
Dataset, scope, epochs, epsilon, cross mode, and loss setup remain separate. A
relaxed-loss run uses a distinct name such as
`canonical_clip_per_dataset_mvtec_ep7p14_cat100_img100_eps2_ce_focal_dice_fullcross.zip`
and cannot overwrite the default-loss setup.
Likewise, a learnable-prompt run has a distinct name such as
`canonical_clip_per_dataset_mvtec_ep7p14_cat100_img100_eps2_ce_focal_dice_fullcross_learnable_prompt.zip`.

Do not merge these archives with the old `canonical_clip_*` bundles under the
same dataset version. Publish them as a new Kaggle dataset version and rerun the
black-box evaluations before drawing conclusions from local or combined losses.

The ready-to-run Kaggle notebook is
`kaggle_generate_corrected_perturbations.ipynb` in this directory.
