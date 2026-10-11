#!/bin/bash
# Single-variable experiment #2 (control for #1, where fusion=-1.0 reached 75.81%).
# The only change: model.prompt_fusion_weight -1.0 -> -3.0 (sigmoid(-3.0) ~= 0.047,
# i.e. the text branch is almost switched off).
# Everything else identical to sv1.
set -e
cd "$(dirname "$0")/.."

EXP_NAME="sv2_fusion_neg3"
mkdir -p outputs/${EXP_NAME}

python -u main.py \
    --dataset webfg400 \
    --data_root ./data \
    --output_dir ./outputs \
    --exp_name ${EXP_NAME} \
    --seed 42 \
    --overrides model.prompt_fusion_weight=-3.0 \
    2>&1 | tee outputs/${EXP_NAME}/train.log
