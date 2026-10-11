#!/bin/bash
# Single-variable experiment #1 (baseline: 74.83%).
# The only change: model.prompt_fusion_weight 0.2 -> -1.0 (sigmoid(-1.0) ~= 0.27).
# Everything else identical to reproduce_default_baseline.
set -e
cd "$(dirname "$0")/.."

EXP_NAME="sv1_fusion_neg1"
mkdir -p outputs/${EXP_NAME}

python -u main.py \
    --dataset webfg400 \
    --data_root ./data \
    --output_dir ./outputs \
    --exp_name ${EXP_NAME} \
    --seed 42 \
    --overrides model.prompt_fusion_weight=-1.0 \
    2>&1 | tee outputs/${EXP_NAME}/train.log
