#!/bin/bash
# Evaluate a trained model on the test set.
# Usage: bash scripts/eval.sh <checkpoint_path> <dataset> [data_root] [extra overrides ...]
# Example (final 384 model):
#   bash scripts/eval.sh ./outputs/fixres384_hd512_dsp_deft/swa_model.pth webfg400 ./data \
#       model.image_size=384 data.image_size=384
set -e
cd "$(dirname "$0")/.."

CHECKPOINT=${1:-"./outputs/best_model.pth"}
DATASET=${2:-"webfg400"}
DATA_ROOT=${3:-"./data"}
shift 3 2>/dev/null || true
EXTRA_OVERRIDES="$@"

echo "Evaluating checkpoint: ${CHECKPOINT}"
echo "Dataset: ${DATASET}"
echo "Data root: ${DATA_ROOT}"
if [ -n "${EXTRA_OVERRIDES}" ]; then
    echo "Extra overrides: ${EXTRA_OVERRIDES}"
fi

python main.py \
    --eval_only \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --resume ${CHECKPOINT} \
    --overrides \
        eval.use_tta=True \
        eval.tta_crops=10 \
        ${EXTRA_OVERRIDES}

echo "Evaluation complete!"
