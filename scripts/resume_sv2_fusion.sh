#!/bin/bash
# Resume of experiment #2 (sv2_fusion_neg3) from the epoch-4 best_model checkpoint.
# Already trained 4 epochs (best 50.71%); continues from epoch 5 up to epoch 40.
set -e
cd "$(dirname "$0")/.."

EXP_NAME="sv2_fusion_neg3"
mkdir -p outputs/${EXP_NAME}

# Back up the previous train.log (keep the history of the first 4 epochs)
if [ -f outputs/${EXP_NAME}/train.log ]; then
    cp outputs/${EXP_NAME}/train.log outputs/${EXP_NAME}/train.log.part1
fi

python -u main.py \
    --dataset webfg400 \
    --data_root ./data \
    --output_dir ./outputs \
    --exp_name ${EXP_NAME} \
    --seed 42 \
    --resume ./outputs/${EXP_NAME}/best_model.pth \
    --overrides model.prompt_fusion_weight=-3.0 \
    2>&1 | tee -a outputs/${EXP_NAME}/train.log
