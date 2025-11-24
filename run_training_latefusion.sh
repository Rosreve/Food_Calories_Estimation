#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="./"
TRAIN_CSV="./Nutrition5K/nutrition5k_train.csv"

EPOCHS=50
BATCH_SIZE=32
LR=5e-5
MULTI_LAMBDA=0.3

USE_TRANSFORM=false
ENABLE_ENSEMBLING=false

MODEL="RGBDResNetLateFusion"
FPN_OUT_CH=256
SPATIAL_DROPOUT=0.1
HEAD_DROPOUT=0.3

LOG_FREQ=10
VAL_FREQ=50
NUM_WORKERS=4
SEED=42

PROJECT_NAME="Nutrition5k-Prediction"
PRED_OUT_DIR="test_predictions"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN="${SCRIPT_DIR}/main.py"

# build command
CMD=(
  "python" "$MAIN"
  --data_root "$DATA_ROOT"
  --train_csv "$TRAIN_CSV"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --multi_lambda "$MULTI_LAMBDA"
  --model "$MODEL"
  --fpn_out_ch "$FPN_OUT_CH"
  --spatial_dropout_ratio "$SPATIAL_DROPOUT"
  --head_dropout_ratio "$HEAD_DROPOUT"
  --log_freq "$LOG_FREQ"
  --test_freq "$VAL_FREQ"
  --num_workers "$NUM_WORKERS"
  --project_name "$PROJECT_NAME"
  --pred_out_dir "$PRED_OUT_DIR"
  --seed "$SEED"
)

if [[ "$USE_TRANSFORM" == "true" ]]; then
  CMD+=( --use_transform )
fi

if [[ "$ENABLE_ENSEMBLING" == "true" ]]; then
  CMD+=( --enable_ensembling )
fi

echo "[run_training] launching:"
printf '  %q' "${CMD[@]}"; echo
"${CMD[@]}"