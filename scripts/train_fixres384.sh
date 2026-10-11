#!/bin/bash
# Single-variable experiment: FixRes, image_size 224 -> 384 (the only change).
# Reference run: sv1_fusion_neg1 (224, fusion=-1.0, single-crop val 75.81%).
# Everything else identical to sv1 (batch=64, accum=2, fusion=-1.0, seed=42).
set -e
cd "$(dirname "$0")/.."

EXP_NAME="fixres384"
mkdir -p outputs/${EXP_NAME}

python -u main.py \
    --dataset webfg400 \
    --data_root ./data \
    --output_dir ./outputs \
    --exp_name ${EXP_NAME} \
    --seed 42 \
    --overrides model.image_size=384 data.image_size=384 model.prompt_fusion_weight=-1.0 \
    2>&1 | tee outputs/${EXP_NAME}/train.log
