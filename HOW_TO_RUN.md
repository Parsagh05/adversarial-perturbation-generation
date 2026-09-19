# Run instructions

Run this project in a CUDA-enabled Linux, Kaggle, or WSL environment. You need
local MVTec AD and VisA dataset directories.

```bash
git clone https://github.com/Parsagh05/adversarial-perturbation-generation.git
cd adversarial-perturbation-generation
git checkout main
git pull --ff-only origin main
```

Install a CUDA-compatible PyTorch build first. `train.sh` installs the remaining
Python dependencies and downloads the pinned AnomalyCLIP dependency.

## Quick smoke test

Replace the three paths, then run:

```bash
export MVTEC_ROOT=/absolute/path/to/mvtec_anomaly_detection
export VISA_ROOT=/absolute/path/to/VisA_20220922
export OUTPUT_BASE=/absolute/path/to/smoke_test_outputs
export PYTHON_BIN="$(command -v python3)"

export SOURCE_DATASETS=mvtec
export EVALUATION_DATASETS=mvtec,visa
export RUN_SETUPS=ep7p14_cat100_img100_eps2
export PROMPT_SETUP=frozen
export RUN_PER_DATASET=true
export RUN_CROSS_DATASET=true
export RUN_PER_CATEGORY=false
export RUN_PER_IMAGE=false
export SMOKE_TEST=true
export SMOKE_EPOCHS=0.02

bash train.sh
```

The smoke test is only for checking the setup. It is not a final result.

To smoke-test the alternate segmentation-aware loss, use:

```bash
export RUN_SETUPS=ep7p14_cat100_img100_eps2_ce_focal_dice
```

To run the default loss with the learned object-agnostic MVTec prompt:

```bash
export RUN_SETUPS=ep7p14_cat100_img100_eps2
export PROMPT_SETUP=learnable
```

No checkpoint path is required. The launcher looks for prompts matching this
run's split protocol, attack-train fraction, split seed and epoch count, and
trains them with `object-agnostic-prompt-training` when there are none.
Training happens once per cohort and is reused by every later setup sharing it.

On Kaggle, attach
[`parsaorbot/learned-prompts`](https://www.kaggle.com/datasets/parsaorbot/learned-prompts)
and the published `balanced` and `full` cohorts are found automatically at the
default `PROMPT_TRAINING_SEARCH_ROOTS=/kaggle/input/learned-prompts/prompts`.
That mount is read-only, so anything needing different prompts, a fraction
below 1.00 for instance, is trained into `PROMPT_TRAINING_OUTPUT_ROOT` instead.
Point `PROMPT_TRAINING_SEARCH_ROOTS` elsewhere, or set it empty, to ignore the
published prompts entirely.

Set `LEARNABLE_PROMPT_MVTEC_CHECKPOINT` or `LEARNABLE_PROMPT_VISA_CHECKPOINT` to
prefer a specific checkpoint. It is used only if it describes this run's split,
and is reported and retrained otherwise. Only source datasets need prompts; an
evaluation-only VisA target does not. Frozen setup IDs ignore all of this.

Use `PROMPT_SETUP=both` to run frozen and learnable variants together. You can
still name one exact learnable ID, such as
`RUN_SETUPS=ep7p14_cat100_img100_eps2_learnable_prompt`; set
`PROMPT_SETUP=learnable` or `both` for that explicit selection.

## Complete run

Use a new output directory for the final run:

```bash
export MVTEC_ROOT=/absolute/path/to/mvtec_anomaly_detection
export VISA_ROOT=/absolute/path/to/VisA_20220922
export OUTPUT_BASE=/absolute/path/to/final_perturbation_outputs
export PYTHON_BIN="$(command -v python3)"

export SOURCE_DATASETS=mvtec
export EVALUATION_DATASETS=mvtec,visa
export RUN_SETUPS=all
export PROMPT_SETUP=both
export RUN_PER_DATASET=true
export RUN_CROSS_DATASET=true
export RUN_PER_CATEGORY=true
export RUN_PER_IMAGE=true
export ATTACK_TRAIN_FRACTION=1.0
export SPLIT_PROTOCOL=full
# true: complete retained source -> complete retained target (default)
# false: source attack_train -> target evaluation, reusing per_dataset delta
export FULL_DATA_CROSS=true
export SMOKE_TEST=false
export OVERWRITE_EXISTING=false

bash train.sh
```

Successful completion ends with:

```text
GENERATION AUDIT PASSED
Done. Results are in: <OUTPUT_BASE>/setups
Combined archive: <OUTPUT_BASE>/full_outputs.zip
```

Outputs are grouped by settings, then scope, then that scope's epoch budget,
then prompt family. `<settings>` is the effective setup ID without the epoch
component and without the prompt-family suffix, so budgets worth comparing sit
side by side:

```text
<OUTPUT_BASE>/setups/<settings>/protocol/
<OUTPUT_BASE>/setups/<settings>/per_dataset/ep<epochs>/frozen_prompt/
<OUTPUT_BASE>/setups/<settings>/cross_dataset/ep<cross_epochs>/frozen_prompt/
<OUTPUT_BASE>/setups/<settings>/per_category/ep<category_epochs>/frozen_prompt/
<OUTPUT_BASE>/setups/<settings>/per_image/ep<image_epochs>/frozen_prompt/
<OUTPUT_BASE>/full_outputs.zip
```

`learnable_prompt/` sits beside `frozen_prompt/` wherever that family was run.
The per-dataset and cross-dataset scopes share a single optimization pass but
keep their own budgets and their own directories. Each leaf holds its bundle
plus a `bundle.zip` of itself; `full_outputs.zip` contains the complete tree
but excludes those redundant per-bundle archives.

The cross mode is explicit in every effective setup ID: `_fullcross` uses both
retained halves and `_halfcross` uses source `attack_train` plus target
`evaluation`. The split convention remains unchanged: `full` contributes
`_full`, while `balanced` contributes no protocol component. For example,
`..._full_fullcross` and `..._full_halfcross` are distinct. The per-dataset
cohort is unchanged.

Every protocol and scope bundle includes `complete_retained_indices.csv`.
Fullcross evaluators select the target dataset and attacked label from this
file, which stays complete even when the target is absent from
`SOURCE_DATASETS`. Halfcross continues to use `evaluation_test_indices.csv`.

The default setup IDs are `ep7p14_cat100_img100_eps2` and `ep7p14_cat100_img100_eps4`, both using the
default `margin_topk` loss. The alternate `ce_focal_dice` loss adds two more,
formed by appending `_ce_focal_dice` to each. Every one of those four
frozen-prompt IDs has a learnable counterpart formed by appending
`_learnable_prompt`, for 8 setups in total. The learned contexts are shallow and object-agnostic; the pipeline
does not perform AnomalyCLIP-style deep text-token tuning.
