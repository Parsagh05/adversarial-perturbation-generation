#!/usr/bin/env bash
# Edit the three paths, then run: bash train.sh

MVTEC_ROOT="${MVTEC_ROOT:-/ABSOLUTE/PATH/TO/mvtec_anomaly_detection}"
VISA_ROOT="${VISA_ROOT:-/ABSOLUTE/PATH/TO/VisA_20220922}"
OUTPUT_BASE="${OUTPUT_BASE:-/ABSOLUTE/PATH/TO/canonical_clip_outputs}"

# all, or a comma-separated subset of 16 isolated setups. Eight base IDs cover
# loss/steps/epsilon with frozen WinCLIP prompts; append _learnable_prompt to
# any base ID to load the object-agnostic shallow prompt checkpoint.
# The setup matrix is the Cartesian product of these two lists with the two
# loss formulations and the two prompt families. Widen a sweep by editing one
# list: SETUP_STEPS="500,800,1200" adds a third step count everywhere at once.
# Setup IDs are derived from these values, so new entries name themselves.
SETUP_STEPS="${SETUP_STEPS:-500,800}"
SETUP_EPSILONS="${SETUP_EPSILONS:-2/255,4/255}"

RUN_SETUPS="${RUN_SETUPS:-all}"

# frozen: run only WinCLIP prompt setups
# learnable: run only object-agnostic learned-prompt setups
# both: run both prompt families selected by RUN_SETUPS
PROMPT_SETUP="${PROMPT_SETUP:-both}"

# Pin the external feature-loader implementation used by every run.
ANOMALYCLIP_COMMIT="${ANOMALYCLIP_COMMIT:-3911738c0867544f545a076ad78f3f11d9ecbfdf}"

# Within each dataset/category, downsample to equal label counts, then split
# each label 50% attack_train and 50% evaluation/test.
SPLIT_SEED="${SPLIT_SEED:-111}"
EVALUATION_FRACTION="${EVALUATION_FRACTION:-0.50}"

# Fraction used from the attack_train half. Use one value per run.
# Later change it to 0.05, 0.10, 0.25, 0.50, or 1.00 for data-efficiency.
# Any value below 1.00 is folded into the setup ID (0.20 -> _train20), so
# runs at different fractions cannot overwrite or pool with each other.
ATTACK_TRAIN_FRACTION="${ATTACK_TRAIN_FRACTION:-1.00}"

# Only these datasets may contribute attack-training images. Per-category and
# per-image scopes remain same-dataset and therefore use this selection too.
SOURCE_DATASETS="${SOURCE_DATASETS:-mvtec}"

# Per-dataset universal perturbations are delivered to held-out IDs from these
# datasets. They never contribute images to attack optimization unless they
# are also explicitly listed in SOURCE_DATASETS.
EVALUATION_DATASETS="${EVALUATION_DATASETS:-mvtec,visa}"

# Same-dataset delivery (mvtec->mvtec). Optimization is shared with
# RUN_CROSS_DATASET, so enabling both costs one optimization pass, not two.
RUN_PER_DATASET="${RUN_PER_DATASET:-true}"
# Cross-dataset delivery (mvtec->visa). Needs an evaluation dataset that is
# not also a source dataset.
RUN_CROSS_DATASET="${RUN_CROSS_DATASET:-true}"
RUN_PER_CATEGORY="${RUN_PER_CATEGORY:-true}"
RUN_PER_IMAGE="${RUN_PER_IMAGE:-true}"

GPU="${GPU:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-518}"
ATTACK_SEED="${ATTACK_SEED:-111}"

# Every selected setup applies the same step count and epsilon to the
# per-dataset, per-category, and per-image scopes. The launcher supplies these
# values from RUN_SETUPS and keeps every setup in an independent folder.
INITIAL_STEP_SIZE="${INITIAL_STEP_SIZE:-0.25/255}"

# Fast plumbing check. Smoke outputs are not benchmark results.
SMOKE_TEST="${SMOKE_TEST:-false}"
SMOKE_STEPS="${SMOKE_STEPS:-2}"

PER_DATASET_BATCH_SIZE="${PER_DATASET_BATCH_SIZE:-2}"

PER_CATEGORY_EFFECTIVE_BATCH_SIZE="${PER_CATEGORY_EFFECTIVE_BATCH_SIZE:-8}"
PER_CATEGORY_MICRO_BATCH_SIZE="${PER_CATEGORY_MICRO_BATCH_SIZE:-2}"

PER_IMAGE_BATCH_SIZE="${PER_IMAGE_BATCH_SIZE:-2}"

# Segmentation-aware local objective. The target-class focal and soft-Dice
# terms are evaluated on patch tokens. Real defect masks focus abnormal->normal
# attacks; normal->abnormal attacks use a fixed synthetic region by default.
LOCAL_FOCAL_WEIGHT="${LOCAL_FOCAL_WEIGHT:-0.5}"
LOCAL_DICE_WEIGHT="${LOCAL_DICE_WEIGHT:-0.5}"
LOCAL_FOCAL_GAMMA="${LOCAL_FOCAL_GAMMA:-2.0}"
LOCAL_DICE_SMOOTH="${LOCAL_DICE_SMOOTH:-1.0}"
LOCAL_BACKGROUND_WEIGHT="${LOCAL_BACKGROUND_WEIGHT:-0.1}"
NORMAL_LOCAL_TARGET="${NORMAL_LOCAL_TARGET:-fixed_region}"
NORMAL_TARGET_REGION_FRACTION="${NORMAL_TARGET_REGION_FRACTION:-0.25}"
NORMAL_TARGET_CENTER_X="${NORMAL_TARGET_CENTER_X:-0.5}"
NORMAL_TARGET_CENTER_Y="${NORMAL_TARGET_CENTER_Y:-0.5}"

# Relaxed loss: s(x)=z_abnormal(x)-z_normal(x) at image level and TopK(H(x))
# at pixel level. The setup ID selects the loss; these configure K only.
MARGIN_TOPK_FRACTION_NORMAL_TO_ABNORMAL="${MARGIN_TOPK_FRACTION_NORMAL_TO_ABNORMAL:-0.20}"
MARGIN_TOPK_FRACTION_ABNORMAL_TO_NORMAL="${MARGIN_TOPK_FRACTION_ABNORMAL_TO_NORMAL:-0.40}"

# Upload the prompt-training artifacts to Kaggle and replace these sample paths.
# Only *_learnable_prompt setups read them; frozen setups ignore them.
LEARNABLE_PROMPT_MVTEC_CHECKPOINT="${LEARNABLE_PROMPT_MVTEC_CHECKPOINT:-/ABSOLUTE/PATH/TO/artifacts/prompts/mvtec/prompts_epoch15.pt}"
LEARNABLE_PROMPT_VISA_CHECKPOINT="${LEARNABLE_PROMPT_VISA_CHECKPOINT:-/ABSOLUTE/PATH/TO/artifacts/prompts/visa/prompts_epoch15.pt}"

# Cosine decay prevents a sign-PGD iterate from bouncing indefinitely on the
# Linf boundary. Full-training checkpoint losses are recorded at this interval.
STEP_SIZE_SCHEDULE="${STEP_SIZE_SCHEDULE:-cosine}"
STEP_SIZE_MIN_RATIO="${STEP_SIZE_MIN_RATIO:-0.1}"
# The full selected attack-training set is used for checkpoint selection.
# Evaluate it periodically because a full pass after every PGD update is costly.
DIAGNOSTIC_INTERVAL="${DIAGNOSTIC_INTERVAL:-50}"
