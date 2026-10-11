#!/bin/bash
# Reproduction run: fully default configuration (no overrides) on WebFG-400,
# to pin down the true baseline of the current code (default config + dataset overrides
# from get_config("webfg400"): cosine head, prompt learner, RandAug, MixUp/CutMix,
# OneCycle, lr 5e-4, 40 epochs).
set -e
cd "$(dirname "$0")/.."

EXP_NAME="reproduce_default_baseline"
mkdir -p outputs/${EXP_NAME}

python -u main.py \
    --dataset webfg400 \
    --data_root ./data \
    --output_dir ./outputs \
    --exp_name ${EXP_NAME} \
    --seed 42 \
    2>&1 | tee outputs/${EXP_NAME}/train.log
