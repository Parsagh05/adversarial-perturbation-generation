#!/usr/bin/env bash
# Edit the three paths, then run: bash train.sh

MVTEC_ROOT="${MVTEC_ROOT:-/ABSOLUTE/PATH/TO/mvtec_anomaly_detection}"
VISA_ROOT="${VISA_ROOT:-/ABSOLUTE/PATH/TO/VisA_20220922}"
OUTPUT_BASE="${OUTPUT_BASE:-/ABSOLUTE/PATH/TO/canonical_clip_outputs}"

# The setup matrix is the Cartesian product of these two lists with the two
# loss formulations and the two prompt families. Setup IDs are derived from
# these values, so new entries name themselves.
#
# Each SETUP_EPOCHS entry is one dataset:cross:category:image budget, because
# the scopes solve different problems: a per-dataset delta must satisfy
# hundreds of images at once, a per-category delta about a dozen, a per-image
# delta exactly one. The cross value is used only by fullcross, which
# optimizes a separate delta on the complete source; halfcross delivers the
# per-dataset delta and has nothing to budget. A bare number means every scope
# uses it, and the older dataset:category:image form still works with cross
# inheriting the dataset budget, which is what it used before it had one.
# An epoch is one pass over whatever that delta trains on; the runner derives
# the PGD step count as ceil(epochs * ceil(n_images / batch)). Budgets therefore
# stay constant when the training set changes size, as it does between
# SPLIT_PROTOCOL=balanced and full. A per-image delta trains on one image, so
# there an epoch is one PGD step.
#   SETUP_EPOCHS="7.14:100:100"       reproduces the historical 800/200/100
#   SETUP_EPOCHS="7.14:5:100:100"     the same, but cross_dataset at 5 epochs
#   SETUP_EPOCHS="7.14:100:100,5:60:50"  sweep two of them
#   SETUP_EPOCHS="100"                same budget for every scope
SETUP_EPOCHS="${SETUP_EPOCHS:-7.14:100:100}"
SETUP_EPSILONS="${SETUP_EPSILONS:-2/255,4/255}"

# all, or a comma-separated subset of the generated setup IDs. Append
# _learnable_prompt to any frozen ID to select its learned-prompt counterpart.
RUN_SETUPS="${RUN_SETUPS:-all}"

# frozen: run only WinCLIP prompt setups
# learnable: run only object-agnostic learned-prompt setups
# both: run both prompt families selected by RUN_SETUPS
PROMPT_SETUP="${PROMPT_SETUP:-both}"

# Pin the external feature-loader implementation used by every run.
ANOMALYCLIP_COMMIT="${ANOMALYCLIP_COMMIT:-3911738c0867544f545a076ad78f3f11d9ecbfdf}"

# balanced: per category keep min(normal, abnormal) of each label, discarding
#   the surplus, then split. Equal label counts; the historical protocol.
# full: keep every image and split each label by EVALUATION_FRACTION, so the
#   category's natural class ratio survives and nothing is discarded.
#   FULL_DATA_CROSS independently selects the cross-dataset cohorts, while
#   per-image covers every test image rather than only the held-out half.
SPLIT_PROTOCOL="${SPLIT_PROTOCOL:-balanced}"
# Cross-dataset cohort policy under either split protocol. true trains on both
# retained source halves and attacks both retained target halves. false reuses
# the ordinary per-dataset delta (source attack_train -> target evaluation).
# It does not alter the per-dataset scope.
FULL_DATA_CROSS="${FULL_DATA_CROSS:-true}"

# Fraction of each category/label stratum held out for evaluation. Applies to
# both protocols: 0.50 is the historical half, 0.30 keeps more for training.
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

# PGD step size, shared by every scope. Flat unless STEP_SIZE_SCHEDULE decays
# it. 0.25/255 is eps/16 at eps=4/255; UAT uses eps/8, which would be 0.5/255.
# Step counts are derived per scope; see SETUP_EPOCHS above.
INITIAL_STEP_SIZE="${INITIAL_STEP_SIZE:-0.25/255}"

# Fast plumbing check. Smoke outputs are not benchmark results.
SMOKE_TEST="${SMOKE_TEST:-false}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-0.02}"

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
#
# MARGIN_HINGE_DISPLACEMENT saturates an image's margin contribution once it
# has moved this far toward the attacked class, measured from its own clean
# margin, so an already-fooled image stops pulling the shared delta. Empty
# disables it and keeps the unbounded margin. It applies to margin_topk only
# and names the setup (0.25 -> _hinge0p25), so an A/B cannot collide. Pick a
# value from the per-image margins of a completed run; see the README.
MARGIN_HINGE_DISPLACEMENT="${MARGIN_HINGE_DISPLACEMENT:-}"

# Write bundle.zip beside each bundle. The directory already holds every
# file the archive does; the archive exists for shipping a bundle on its
# own, so a pipeline that evaluates in place pays for it and uses none of
# it. false skips the write and the re-open verification with it.
WRITE_BUNDLE_ARCHIVES="${WRITE_BUNDLE_ARCHIVES:-true}"

# Which images the per-image scope attacks. evaluation, the default,
# uses the evaluation partition - the cohort every other scope is scored
# on - so per-image stays comparable with them. all attacks every retained
# image instead, which is sound on its own terms because a per-image delta
# fits the image it attacks and holds nothing out, but it is a different
# cohort from the other scopes and a consumer scoring the evaluation
# partition will disagree with the manifest about the bundle's size.
# It names the setup (all -> _alltargets), so an A/B cannot collide.
PER_IMAGE_ATTACK_COHORT="${PER_IMAGE_ATTACK_COHORT:-evaluation}"

# Gradient accumulation on the shared update: m = decay * m + g, stepping
# along sign(m). Empty or 0 is plain sign-PGD. 0.9 is the recommended value
# when enabling it; 1.0 never forgets, which is MI-FGSM's convention and
# cancels MARGIN_HINGE_DISPLACEMENT. Applies to the scopes that share one
# delta, not to per-image, and names the setup (0.9 -> _mom0p9).
MOMENTUM_DECAY="${MOMENTUM_DECAY:-}"

# Which iterate the optimization returns. final, the default, returns the last
# step, which is what every universal-attack paper does. best instead scores
# the delta over the whole attack-train cohort every DIAGNOSTIC_INTERVAL steps
# and keeps the lowest, with the clean delta as the baseline, so a run that
# never beats clean returns zeros. The trajectory is the same either way; only
# the kept point differs, and best names the setup (_best).
CHECKPOINT_SELECTION="${CHECKPOINT_SELECTION:-final}"

# Shorter budgets to capture inside one run, written exactly like
# SETUP_EPOCHS so each snapshot names a complete setup of its own and every
# scope stops at its own boundary. Empty disables it. Each entry must be a
# prefix of the run's budget: no scope may exceed it, and the triple may not
# equal it. Requires STEP_SIZE_SCHEDULE=constant, since a decaying step
# depends on the total budget and the snapshot would not equal a standalone
# run of that budget.
#   SETUP_EPOCHS="20:400:400" SNAPSHOT_EPOCHS="5:100:100,10:200:200"
#   -> ep5_cat100_img100_..., ep10_cat200_img200_..., plus the run's own
SNAPSHOT_EPOCHS="${SNAPSHOT_EPOCHS:-}"
MARGIN_TOPK_FRACTION_NORMAL_TO_ABNORMAL="${MARGIN_TOPK_FRACTION_NORMAL_TO_ABNORMAL:-0.20}"
MARGIN_TOPK_FRACTION_ABNORMAL_TO_NORMAL="${MARGIN_TOPK_FRACTION_ABNORMAL_TO_NORMAL:-0.40}"

# Only *_learnable_prompt setups read any of this; frozen setups ignore it.
#
# Prompts fitted under one SPLIT_PROTOCOL or ATTACK_TRAIN_FRACTION saw
# different images than a run using another, and pairing them raises no error
# on either side. Leave both paths empty and the launcher resolves the
# checkpoint these settings require under PROMPT_TRAINING_OUTPUT_ROOT,
# training one if it is not there. Set a path to prefer a specific checkpoint;
# it is still rejected and retrained when it describes a different split.
LEARNABLE_PROMPT_MVTEC_CHECKPOINT="${LEARNABLE_PROMPT_MVTEC_CHECKPOINT:-}"
LEARNABLE_PROMPT_VISA_CHECKPOINT="${LEARNABLE_PROMPT_VISA_CHECKPOINT:-}"

# Checkpoints are filed by cohort, so balanced and full never overwrite each
# other: <root>/<protocol>[_trainNN]/<dataset>/prompts_epoch<N>.pt
#
# Published prompts are searched first, in order, and are read-only. The
# default is the Kaggle dataset holding the balanced and full cohorts:
#   kaggle.com/datasets/parsaorbot/learned-prompts
# Anything that does not describe the current run is retrained into
# PROMPT_TRAINING_OUTPUT_ROOT instead, which is the only place written.
PROMPT_TRAINING_SEARCH_ROOTS="${PROMPT_TRAINING_SEARCH_ROOTS:-/kaggle/input/learned-prompts/prompts}"
PROMPT_TRAINING_OUTPUT_ROOT="${PROMPT_TRAINING_OUTPUT_ROOT:-$OUTPUT_BASE/prompts}"
# Point PROMPT_TRAINING_ROOT at a local clone to skip the fetch entirely.
PROMPT_TRAINING_ROOT="${PROMPT_TRAINING_ROOT:-}"
PROMPT_TRAINING_GIT_URL="${PROMPT_TRAINING_GIT_URL:-https://github.com/Parsagh05/object-agnostic-prompt-training.git}"
# The branch tip is taken at run time rather than a pinned commit, so the
# prompts always come from the current training code. The commit it resolved
# to is logged and recorded in the checkpoint's manifest.json.
PROMPT_TRAINING_BRANCH="${PROMPT_TRAINING_BRANCH:-main}"
# The published checkpoints were fitted with these; changing either retrains.
PROMPT_TRAINING_EPOCHS="${PROMPT_TRAINING_EPOCHS:-15}"
PROMPT_TRAINING_BATCH_SIZE="${PROMPT_TRAINING_BATCH_SIZE:-2}"

# constant, linear or cosine. constant is the default and matches the
# universal-attack literature: UAP, UAT and CD-UAP use no budget-normalised
# decay. It also keeps a run sliceable -- a decaying step depends on the total
# step count, so the first N steps of a long run differ from an N-step run,
# whereas with a flat step they are identical. The decaying schedules end at
# STEP_SIZE_MIN_RATIO and name themselves in the setup ID.
STEP_SIZE_SCHEDULE="${STEP_SIZE_SCHEDULE:-constant}"
STEP_SIZE_MIN_RATIO="${STEP_SIZE_MIN_RATIO:-0.1}"
# The full selected attack-training set is used for checkpoint selection.
# Evaluate it periodically because a full pass after every PGD update is costly.
DIAGNOSTIC_INTERVAL="${DIAGNOSTIC_INTERVAL:-50}"
