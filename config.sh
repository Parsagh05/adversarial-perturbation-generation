#!/usr/bin/env bash
# Edit the three paths, then run: bash train.sh

MVTEC_ROOT="${MVTEC_ROOT:-/ABSOLUTE/PATH/TO/mvtec_anomaly_detection}"
VISA_ROOT="${VISA_ROOT:-/ABSOLUTE/PATH/TO/VisA_20220922}"
OUTPUT_BASE="${OUTPUT_BASE:-/ABSOLUTE/PATH/TO/canonical_clip_outputs}"

# all, or a comma-separated subset of:
# steps500_eps2, steps500_eps4, steps800_eps2, steps800_eps4
RUN_SETUPS="${RUN_SETUPS:-all}"

# Pin the external feature-loader implementation used by every run.
ANOMALYCLIP_COMMIT="${ANOMALYCLIP_COMMIT:-3911738c0867544f545a076ad78f3f11d9ecbfdf}"

# Within each dataset/category, downsample to equal label counts, then split
# each label 50% attack_train and 50% evaluation/test.
SPLIT_SEED="${SPLIT_SEED:-111}"
EVALUATION_FRACTION="${EVALUATION_FRACTION:-0.50}"

# Fraction used from the attack_train half. Use one value per run.
# Later change it to 0.05, 0.10, 0.25, 0.50, or 1.00 for data-efficiency.
ATTACK_TRAIN_FRACTION="${ATTACK_TRAIN_FRACTION:-1.00}"

# Comma-separated subset to generate. Use mvtec or visa to split long Kaggle
# runs across sessions; the default preserves the complete two-dataset run.
GENERATION_DATASETS="${GENERATION_DATASETS:-mvtec,visa}"

RUN_PER_DATASET="${RUN_PER_DATASET:-true}"
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

# Cosine decay prevents a sign-PGD iterate from bouncing indefinitely on the
# Linf boundary. Full-training checkpoint losses are recorded at this interval.
STEP_SIZE_SCHEDULE="${STEP_SIZE_SCHEDULE:-cosine}"
STEP_SIZE_MIN_RATIO="${STEP_SIZE_MIN_RATIO:-0.1}"
# The full selected attack-training set is used for checkpoint selection.
# Evaluate it periodically because a full pass after every PGD update is costly.
DIAGNOSTIC_INTERVAL="${DIAGNOSTIC_INTERVAL:-50}"
