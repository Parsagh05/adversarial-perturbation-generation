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
export PER_IMAGE_EFFECTIVE_BATCH_SIZE="$PER_IMAGE_BATCH_SIZE"
export PER_IMAGE_MICRO_BATCH_SIZE="$PER_IMAGE_BATCH_SIZE"
export DIRECTIONS="${DIRECTIONS:-normal_to_abnormal,abnormal_to_normal}"
export LOSS_MODES="${LOSS_MODES:-global,local,combined}"
export USE_AMP="${USE_AMP:-true}"
export CACHE_INPUTS_IN_RAM="${CACHE_INPUTS_IN_RAM:-true}"
export OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-false}"
export PER_IMAGE_EVALUATION_FRACTION="${PER_IMAGE_EVALUATION_FRACTION:-1.0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1

for name in OUTPUT_BASE; do
  value="${!name}"
  [[ "$value" != /ABSOLUTE/PATH/TO/* ]] || { echo "Edit $name in config.sh" >&2; exit 2; }
done
case ",$GENERATION_DATASETS," in
  *,mvtec,*) [[ -d "$MVTEC_ROOT" ]] || { echo "Missing MVTec directory: $MVTEC_ROOT" >&2; exit 2; } ;;
esac
case ",$GENERATION_DATASETS," in
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

SETUP_IDS=(steps500_eps2 steps500_eps4 steps800_eps2 steps800_eps4)
SETUP_STEPS=(500 500 800 800)
SETUP_EPS=(2/255 4/255 2/255 4/255)

case "${RUN_PER_DATASET,,},${RUN_PER_CATEGORY,,},${RUN_PER_IMAGE,,}" in
  false,false,false) echo "At least one attack scope must be enabled" >&2; exit 2 ;;
esac

if [[ "$RUN_SETUPS" != "all" ]]; then
  IFS=',' read -r -a requested_setups <<< "$RUN_SETUPS"
  for requested in "${requested_setups[@]}"; do
    case "$requested" in
      steps500_eps2|steps500_eps4|steps800_eps2|steps800_eps4) ;;
      *) echo "Unknown RUN_SETUPS value: $requested" >&2; exit 2 ;;
    esac
  done
fi

selected() {
  local id="$1"
  [[ "$RUN_SETUPS" == "all" || ",$RUN_SETUPS," == *",$id,"* ]]
}

selected_count=0
for id in "${SETUP_IDS[@]}"; do
  if selected "$id"; then
    selected_count=$((selected_count + 1))
  fi
done
[[ "$selected_count" -gt 0 ]] || {
  echo "RUN_SETUPS did not select a known setup" >&2
  exit 2
}

for index in "${!SETUP_IDS[@]}"; do
  id="${SETUP_IDS[$index]}"
  selected "$id" || continue
  steps="${SETUP_STEPS[$index]}"
  epsilon="${SETUP_EPS[$index]}"
  if [[ "${SMOKE_TEST,,}" == "true" ]]; then
    steps="$SMOKE_STEPS"
  fi
  setup_root="$PIPELINE_OUTPUT/setups/$id"
  echo "===== SETUP $id: steps=$steps epsilon=$epsilon ====="
  (
    export OUTPUT_BASE="$setup_root"
    export SETUP_ID="$id"
    export PROTOCOL_DIR="$setup_root/protocol"
    export ATTACK_TRAIN_CSV="$PROTOCOL_DIR/attack_train_indices.csv"
    export EVALUATION_CSV="$PROTOCOL_DIR/evaluation_test_indices.csv"
    export EPSILON="$epsilon"
    export PER_DATASET_STEPS="$steps"
    export PER_CATEGORY_STEPS="$steps"
    export PER_IMAGE_STEPS="$steps"
    export PER_DATASET_STEP_SIZE="$INITIAL_STEP_SIZE"
    export PER_CATEGORY_STEP_SIZE="$INITIAL_STEP_SIZE"
    export PER_IMAGE_STEP_SIZE="$INITIAL_STEP_SIZE"
    export PER_DATASET_ATTACK_TRAIN_FRACTIONS="$ATTACK_TRAIN_FRACTION"
    export PER_CATEGORY_ATTACK_TRAIN_FRACTIONS="$ATTACK_TRAIN_FRACTION"
    mkdir -p "$PROTOCOL_DIR" "$setup_root/logs"

    "$PYTHON" "$ROOT/common.py" split | tee "$setup_root/logs/00_split.log"

    run_mode() {
      local enabled="$1" name="$2" script="$3"
      if [[ "${enabled,,}" == "true" ]]; then
        echo "===== $id/$name ====="
        "$PYTHON" "$ROOT/$script" 2>&1 | tee "$setup_root/logs/$name.log"
      fi
    }

    run_mode "$RUN_PER_DATASET"  per_dataset  run_per_dataset.py
    run_mode "$RUN_PER_CATEGORY" per_category run_per_category.py
    run_mode "$RUN_PER_IMAGE"    per_image    run_per_image.py
  )
done

"$PYTHON" "$ROOT/audit_generation.py"
echo "Done. Results are in: $PIPELINE_OUTPUT/setups"
