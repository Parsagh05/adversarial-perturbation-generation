#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "$ROOT/config.sh"
set +a

export CUDA_VISIBLE_DEVICES="$GPU"
export PROJECT_ROOT="$ROOT"
PIPELINE_OUTPUT="$OUTPUT_BASE"
export WORK_DIR="${WORK_DIR:-$PIPELINE_OUTPUT/runtime}"
export ATTACK_TRAIN_FRACTION
export FULL_DATA_CROSS="${FULL_DATA_CROSS:-true}"
export MARGIN_HINGE_DISPLACEMENT="${MARGIN_HINGE_DISPLACEMENT:-}"
export MOMENTUM_DECAY="${MOMENTUM_DECAY:-}"
export CHECKPOINT_SELECTION="${CHECKPOINT_SELECTION:-final}"
export SNAPSHOT_EPOCHS="${SNAPSHOT_EPOCHS:-}"
export PER_IMAGE_EFFECTIVE_BATCH_SIZE="$PER_IMAGE_BATCH_SIZE"
export PER_IMAGE_MICRO_BATCH_SIZE="$PER_IMAGE_BATCH_SIZE"
export DIRECTIONS="${DIRECTIONS:-normal_to_abnormal,abnormal_to_normal}"
export LOSS_MODES="${LOSS_MODES:-global,local,combined}"
export USE_AMP="${USE_AMP:-true}"
export CACHE_INPUTS_IN_RAM="${CACHE_INPUTS_IN_RAM:-true}"
export OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-false}"
export WRITE_BUNDLE_ARCHIVES="${WRITE_BUNDLE_ARCHIVES:-true}"
export PER_IMAGE_EVALUATION_FRACTION="${PER_IMAGE_EVALUATION_FRACTION:-1.0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1

case "${FULL_DATA_CROSS,,}" in
  true|false) ;;
  *)
    echo "FULL_DATA_CROSS must be true or false" >&2
    exit 2
    ;;
esac

for name in OUTPUT_BASE; do
  value="${!name}"
  [[ "$value" != /ABSOLUTE/PATH/TO/* ]] || { echo "Edit $name in config.sh" >&2; exit 2; }
done
SELECTED_PROTOCOL_DATASETS="$SOURCE_DATASETS,$EVALUATION_DATASETS"
case ",$SELECTED_PROTOCOL_DATASETS," in
  *,mvtec,*) [[ -d "$MVTEC_ROOT" ]] || { echo "Missing MVTec directory: $MVTEC_ROOT" >&2; exit 2; } ;;
esac
case ",$SELECTED_PROTOCOL_DATASETS," in
  *,visa,*) [[ -d "$VISA_ROOT" ]] || { echo "Missing VisA directory: $VISA_ROOT" >&2; exit 2; } ;;
esac
mkdir -p "$WORK_DIR" "$PIPELINE_OUTPUT/setups"

PYTHON="${PYTHON_BIN:-python3}"
USE_VENV="${USE_VENV:-false}"
if [[ "${USE_VENV,,}" == "true" ]]; then
  VENV_DIR="$ROOT/.venv"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    if ! "$PYTHON" -m venv --system-site-packages "$VENV_DIR"; then
      echo "WARNING: virtualenv creation failed; using $PYTHON directly." >&2
    fi
  fi
  if [[ -x "$VENV_DIR/bin/python" ]] && "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
    PYTHON="$VENV_DIR/bin/python"
  else
    echo "WARNING: virtualenv has no working pip; using $PYTHON directly." >&2
  fi
fi
"$PYTHON" -m pip --version >/dev/null
echo "Python runtime: $PYTHON"

clone_pinned() {
  local url="$1" dest="$2" commit="$3"
  if [[ ! -d "$dest/.git" ]]; then
    git clone --filter=blob:none --no-checkout "$url" "$dest"
  fi
  git -C "$dest" fetch --depth 1 origin "$commit"
  git -C "$dest" checkout --detach --force FETCH_HEAD
  [[ "$(git -C "$dest" rev-parse HEAD)" == "$commit" ]] || {
    echo "Pinned checkout mismatch for $dest" >&2
    exit 2
  }
}
clone_pinned \
  "https://github.com/zqhang/AnomalyCLIP.git" \
  "$WORK_DIR/AnomalyCLIP" \
  "$ANOMALYCLIP_COMMIT"

SETUP_STAMP="$WORK_DIR/.generation_dependencies_ready"
if [[ ! -f "$SETUP_STAMP" ]]; then
  "$PYTHON" -m pip install -q -r "$ROOT/requirements.txt"
  touch "$SETUP_STAMP"
fi

# The setup matrix is defined once in setup_catalog.py; deriving it here keeps
# the launcher and audit_generation.py from drifting apart.
SETUP_TABLE="$(PYTHONPATH="$ROOT" "$PYTHON" - <<'PYEOF'
import os
import sys
from setup_catalog import (
    SETUPS,
    effective_setup_id,
    full_data_cross_setting,
    checkpoint_selection_setting,
    SCOPE_DIRECTORIES,
    assert_snapshots_fit,
    compose_setup_id,
    scope_output_path,
    settings_tag,
    margin_hinge_setting,
    momentum_decay_setting,
    snapshot_epochs_setting,
    split_protocol_setting,
    step_size_schedule_setting,
)

# The effective epoch budget and train fraction are known before any setup runs,
# so the output name is resolved here rather than after an override.
smoke = os.environ.get("SMOKE_TEST", "false").strip().lower() in {"1", "true", "yes", "on"}
override = float(os.environ["SMOKE_EPOCHS"]) if smoke else None
fraction = float(os.environ.get("ATTACK_TRAIN_FRACTION", "1.0"))
protocol = split_protocol_setting()
full_data_cross = full_data_cross_setting()
schedule = step_size_schedule_setting()
hinge = margin_hinge_setting()
momentum = momentum_decay_setting()
selection = checkpoint_selection_setting()
# A snapshot budget is an ordinary setup that happens to be produced by a
# longer run. Emitting it as another row means the launcher, the audit and the
# packaging treat it exactly like a standalone run, which is the point.
snapshots = () if override is not None else snapshot_epochs_setting()
produced = {}


def name_for(setup, budget):
    return compose_setup_id(
        *budget, setup.epsilon_label, setup.loss_formulation, setup.prompt_mode,
        fraction, protocol, full_data_cross, schedule, hinge, momentum,
        selection,
    )

for setup_id, setup in SETUPS.items():
    # A smoke override collapses every scope onto one count.
    epochs = setup.epochs if override is None else override
    cross_epochs = setup.cross_epochs if override is None else override
    category_epochs = setup.category_epochs if override is None else override
    image_epochs = setup.image_epochs if override is None else override
    effective = effective_setup_id(
        setup, override, fraction, protocol, full_data_cross, schedule, hinge,
        momentum, selection,
    )
    if effective in produced:
        # A single step override collapses every step count onto one name, so
        # distinct catalog rows would otherwise overwrite each other's output.
        print(
            f"note: {setup_id} collapses onto {effective}, already covered by "
            f"{produced[effective]}; skipping the duplicate",
            file=sys.stderr,
        )
        continue
    produced[effective] = setup_id
    budget = (epochs, cross_epochs, category_epochs, image_epochs)
    assert_snapshots_fit(snapshots, budget)
    rows = [(budget, effective)]
    for snapshot in snapshots:
        snapshot_name = name_for(setup, snapshot)
        if snapshot_name in produced:
            print(
                f"note: snapshot {snapshot} of {setup_id} collapses onto "
                f"{snapshot_name}, already covered by {produced[snapshot_name]}; "
                "skipping the duplicate",
                file=sys.stderr,
            )
            continue
        produced[snapshot_name] = setup_id
        rows.append((snapshot, snapshot_name))
    # The longest budget is emitted first so its run writes the shorter
    # budgets' deltas before their own rows are reached.
    spec = ";".join(
        ":".join(str(value) for value in snapshot) + "=" + snapshot_name
        for snapshot, snapshot_name in rows[1:]
    )
    settings = settings_tag(
        setup.epsilon_label, setup.loss_formulation, fraction, protocol,
        full_data_cross, schedule, hinge, momentum, selection,
    )
    for index, (row_budget, row_name) in enumerate(rows):
        print("\x1f".join((
            setup_id, *(str(value) for value in row_budget),
            setup.epsilon_label, setup.loss_formulation, setup.prompt_mode,
            row_name, spec if index == 0 else "", settings,
            *(
                scope_output_path(scope, row_budget, setup.prompt_mode, settings)
                for scope in SCOPE_DIRECTORIES
            ),
        )))
PYEOF
)"
[[ -n "$SETUP_TABLE" ]] || { echo "Could not read the setup catalog" >&2; exit 2; }
mapfile -t SETUP_IDS < <(cut -d$'\x1f' -f1 <<< "$SETUP_TABLE")

case "${RUN_PER_DATASET,,},${RUN_CROSS_DATASET,,},${RUN_PER_CATEGORY,,},${RUN_PER_IMAGE,,}" in
  false,false,false,false)
    echo "At least one of the four attack scopes must be enabled" >&2
    exit 2
    ;;
esac

PROMPT_SETUP="${PROMPT_SETUP,,}"
case "$PROMPT_SETUP" in
  frozen|learnable|both) ;;
  *)
    echo "PROMPT_SETUP must be frozen, learnable, or both; got: $PROMPT_SETUP" >&2
    exit 2
    ;;
esac

if [[ "$RUN_SETUPS" != "all" ]]; then
  IFS=',' read -r -a requested_setups <<< "$RUN_SETUPS"
  for requested in "${requested_setups[@]}"; do
    known=false
    for candidate in "${SETUP_IDS[@]}"; do
      [[ "$requested" == "$candidate" ]] && known=true && break
    done
    [[ "$known" == "true" ]] || {
      echo "Unknown RUN_SETUPS value: $requested" >&2
      exit 2
    }
  done
fi

selected() {
  local id="$1" prompt_mode="$2"
  local base_id="${id%_learnable_prompt}"
  [[ "$RUN_SETUPS" == "all" || ",$RUN_SETUPS," == *",$id,"* || ",$RUN_SETUPS," == *",$base_id,"* ]] || return 1
  case "$PROMPT_SETUP" in
    frozen) [[ "$prompt_mode" == "frozen_winclip" ]] ;;
    learnable) [[ "$prompt_mode" == "learnable_object_agnostic" ]] ;;
    both) return 0 ;;
  esac
}

selected_count=0
while IFS=$'\x1f' read -r id _ _ _ _ _ _ prompt_mode _ _ _ _ _ _ _; do
  if selected "$id" "$prompt_mode"; then
    selected_count=$((selected_count + 1))
  fi
done <<< "$SETUP_TABLE"
[[ "$selected_count" -gt 0 ]] || {
  echo "RUN_SETUPS and PROMPT_SETUP did not select a compatible setup" >&2
  exit 2
}

while IFS=$'\x1f' read -r id epochs cross_epochs category_epochs image_epochs epsilon loss_formulation prompt_mode effective_id snapshot_spec settings_tag bundle_per_dataset bundle_cross_dataset bundle_per_category bundle_per_image; do
  selected "$id" "$prompt_mode" || continue
  if [[ "$prompt_mode" == "frozen_winclip" ]]; then
    prompt_folder="frozen_prompt"
  else
    prompt_folder="learnable_prompt"
  fi
  # The split depends only on the settings, so it is computed once and shared
  # by every scope, budget and prompt family underneath it. Each bundle keeps
  # its own copy so it stays independently evaluable.
  settings_root="$PIPELINE_OUTPUT/setups/$settings_tag"
  setup_root="$PIPELINE_OUTPUT/setups/$bundle_per_dataset"
  # Absolute bundle roots for the shorter budgets this run also produces. Empty
  # for a snapshot row, which produces none of its own. Only the scopes that
  # share one optimisation pass need them, so each entry carries all four.
  snapshot_roots=""
  if [[ -n "$snapshot_spec" ]]; then
    IFS=';' read -r -a snapshot_entries <<< "$snapshot_spec"
    for entry in "${snapshot_entries[@]}"; do
      snapshot_roots+="${entry%%=*}=$PIPELINE_OUTPUT/setups;"
    done
  fi
  echo "===== SETUP $effective_id (requested $id): prompt=$prompt_mode loss=$loss_formulation epochs=dataset:$epochs/cross:$cross_epochs/category:$category_epochs/image:$image_epochs epsilon=$epsilon fraction=$ATTACK_TRAIN_FRACTION ====="
  (
    export OUTPUT_BASE="$settings_root"
    export SETUP_ID="$effective_id"
    export SETTINGS_TAG="$settings_tag"
    export PROTOCOL_DIR="$settings_root/protocol"
    export BUNDLE_PER_DATASET="$PIPELINE_OUTPUT/setups/$bundle_per_dataset"
    export BUNDLE_CROSS_DATASET="$PIPELINE_OUTPUT/setups/$bundle_cross_dataset"
    export BUNDLE_PER_CATEGORY="$PIPELINE_OUTPUT/setups/$bundle_per_category"
    export BUNDLE_PER_IMAGE="$PIPELINE_OUTPUT/setups/$bundle_per_image"
    export ATTACK_TRAIN_CSV="$PROTOCOL_DIR/attack_train_indices.csv"
    export EVALUATION_CSV="$PROTOCOL_DIR/evaluation_test_indices.csv"
    export EPSILON="$epsilon"
    export LOSS_FORMULATION="$loss_formulation"
    export PROMPT_MODE="$prompt_mode"
    # Each scope fits a different number of images per delta, so each carries
    # its own epoch budget; the runner derives its PGD step count from that and
    # its own training-set size. cross_dataset has no separate budget: halfcross
    # reuses the per-dataset delta and fullcross uses the per-dataset budget.
    export PER_DATASET_EPOCHS="$epochs"
    export PER_CROSS_EPOCHS="$cross_epochs"
    export SNAPSHOT_SETUP_ROOTS="$snapshot_roots"
    export PER_CATEGORY_EPOCHS="$category_epochs"
    export PER_IMAGE_EPOCHS="$image_epochs"
    export PER_DATASET_STEP_SIZE="$INITIAL_STEP_SIZE"
    export PER_CATEGORY_STEP_SIZE="$INITIAL_STEP_SIZE"
    export PER_IMAGE_STEP_SIZE="$INITIAL_STEP_SIZE"
    export PER_DATASET_ATTACK_TRAIN_FRACTIONS="$ATTACK_TRAIN_FRACTION"
    export PER_CATEGORY_ATTACK_TRAIN_FRACTIONS="$ATTACK_TRAIN_FRACTION"
    mkdir -p "$PROTOCOL_DIR" "$BUNDLE_PER_DATASET/logs"

    "$PYTHON" "$ROOT/common.py" split | tee "$BUNDLE_PER_DATASET/logs/00_split.log"

    if [[ "$prompt_mode" == "learnable_object_agnostic" ]]; then
      # Resolved after the split so the prompts can be fitted on this run's own
      # attack-training cohort. Progress goes to stderr and the log; stdout is
      # the resolved checkpoint paths, which replace whatever config.sh held.
      prompt_env="$BUNDLE_PER_DATASET/logs/01_prompts.env"
      "$PYTHON" "$ROOT/ensure_prompt_checkpoint.py" 2>&1 >"$prompt_env" \
        | tee "$BUNDLE_PER_DATASET/logs/01_prompts.log" >&2
      source "$prompt_env"
    fi

    run_mode() {
      local enabled="$1" name="$2" script="$3"
      if [[ "${enabled,,}" == "true" ]]; then
        echo "===== $id/$name ====="
        "$PYTHON" "$ROOT/$script" 2>&1 | tee "$BUNDLE_PER_DATASET/logs/$name.log"
      fi
    }

    # Same-dataset and cross-dataset share one optimization pass and are
    # emitted as two independent bundles by run_per_dataset.py.
    dataset_settings=""
    [[ "${RUN_PER_DATASET,,}" == "true" ]] && dataset_settings="same_dataset"
    [[ "${RUN_CROSS_DATASET,,}" == "true" ]] && \
      dataset_settings="${dataset_settings:+$dataset_settings,}cross_dataset"
    if [[ -n "$dataset_settings" ]]; then
      export DATASET_TRANSFER_SETTINGS="$dataset_settings"
      echo "===== $id/dataset_scopes ($dataset_settings) ====="
      "$PYTHON" "$ROOT/run_per_dataset.py" 2>&1 | tee "$BUNDLE_PER_DATASET/logs/per_dataset.log"
    fi

    run_mode "$RUN_PER_CATEGORY" per_category run_per_category.py
    run_mode "$RUN_PER_IMAGE"    per_image    run_per_image.py
  )
done <<< "$SETUP_TABLE"

"$PYTHON" "$ROOT/audit_generation.py"
export PIPELINE_OUTPUT
"$PYTHON" "$ROOT/package_full_outputs.py"
echo "Done. Results are in: $PIPELINE_OUTPUT/setups"
echo "Combined archive: $PIPELINE_OUTPUT/full_outputs.zip"
