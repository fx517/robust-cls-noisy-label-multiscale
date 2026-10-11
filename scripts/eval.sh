#!/bin/bash
# Evaluate a trained model on the test set.
# Usage: bash scripts/eval.sh <checkpoint_path> <dataset> [data_root]
set -e
cd "$(dirname "$0")/.."

CHECKPOINT=${1:-"./outputs/best_model.pth"}
DATASET=${2:-"webfg400"}
DATA_ROOT=${3:-"./data"}

echo "Evaluating checkpoint: ${CHECKPOINT}"
echo "Dataset: ${DATASET}"
echo "Data root: ${DATA_ROOT}"

python main.py \
    --eval_only \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --resume ${CHECKPOINT} \
    --overrides \
        eval.use_tta=True \
        eval.tta_crops=10

echo "Evaluation complete!"
