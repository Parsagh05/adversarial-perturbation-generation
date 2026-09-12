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

To smoke-test only the new relaxed loss, use:

```bash
export RUN_SETUPS=ep7p14_cat100_img100_eps2_margin_topk
```

To run the same relaxed loss with the learned object-agnostic MVTec prompt,
upload the prompt artifact to Kaggle (or copy it locally) and use:

```bash
export LEARNABLE_PROMPT_MVTEC_CHECKPOINT=/absolute/path/to/artifacts/prompts/mvtec/prompts_epoch15.pt
export RUN_SETUPS=ep7p14_cat100_img100_eps2_margin_topk
export PROMPT_SETUP=learnable
```

For VisA learnable setups, set `LEARNABLE_PROMPT_VISA_CHECKPOINT` to the VisA
checkpoint. Only source datasets require learned-prompt checkpoints; an
evaluation-only VisA target does not require the VisA prompt checkpoint.
Frozen setup IDs ignore these variables.

Use `PROMPT_SETUP=both` to run frozen and learnable variants together. You can
still name one exact learnable ID, such as
`RUN_SETUPS=ep7p14_cat100_img100_eps2_margin_topk_learnable_prompt`; set
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

Outputs for each setup are under:

Each setup directory holds one bundle per enabled scope:
`canonical_clip_per_dataset`, `canonical_clip_cross_dataset`,
`canonical_clip_per_category`, `canonical_clip_per_image`. The first two share a
single optimization pass.

```text
<OUTPUT_BASE>/setups/frozen_prompt/<frozen_setup_id>/
<OUTPUT_BASE>/setups/learnable_prompt/<learnable_setup_id>/
<OUTPUT_BASE>/full_outputs.zip
```

`full_outputs.zip` contains the complete setup tree but excludes the redundant
per-scope ZIP files already stored inside individual setup directories.

The default setup IDs are `ep7p14_cat100_img100_eps2` and `ep7p14_cat100_img100_eps4`. The new loss adds
two more formed by appending `_margin_topk` to each. Every one of those four
frozen-prompt IDs has a learnable counterpart formed by appending
`_learnable_prompt`, for 8 setups in total. The learned contexts are shallow and object-agnostic; the pipeline
does not perform AnomalyCLIP-style deep text-token tuning.
