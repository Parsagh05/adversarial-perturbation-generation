# How to run

```bash
git clone <repository-url>
cd perturbation-generation

export MVTEC_ROOT=/absolute/path/to/mvtec_anomaly_detection
export VISA_ROOT=/absolute/path/to/VisA_20220922
export OUTPUT_BASE=/absolute/path/to/perturbation_outputs
export PYTHON_BIN=/absolute/path/to/python3

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
