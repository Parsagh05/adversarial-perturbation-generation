# Run instructions

Run this project in a CUDA-enabled Linux, Kaggle, or WSL environment. You need
local MVTec AD and VisA dataset directories.

```bash
git clone https://github.com/Parsagh05/adversarial-perturbation-generation.git
cd adversarial-perturbation-generation
git checkout ff874e6ac7cb0b9e16e73048503e9099271f1121
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

export GENERATION_DATASETS=mvtec
export RUN_SETUPS=steps500_eps2
export RUN_PER_DATASET=true
export RUN_PER_CATEGORY=false
export RUN_PER_IMAGE=false
export SMOKE_TEST=true
export SMOKE_STEPS=2

bash train.sh
```

The smoke test is only for checking the setup. It is not a final result.

## Complete run

Use a new output directory for the final run:

```bash
export MVTEC_ROOT=/absolute/path/to/mvtec_anomaly_detection
export VISA_ROOT=/absolute/path/to/VisA_20220922
export OUTPUT_BASE=/absolute/path/to/final_perturbation_outputs
export PYTHON_BIN="$(command -v python3)"

export GENERATION_DATASETS=mvtec,visa
export RUN_SETUPS=all
export RUN_PER_DATASET=true
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
```

Outputs for each setup are under:

```text
<OUTPUT_BASE>/setups/<setup_id>/
```

The setup IDs are `steps500_eps2`, `steps500_eps4`, `steps800_eps2`, and
`steps800_eps4`.
