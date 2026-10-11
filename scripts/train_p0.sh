#!/bin/bash
# P0 correction experiment (aligned with community findings on the same track):
#   1) Linear head (fc, hidden=0) instead of the cosine head
#   2) No heavy augmentation; CLIP standard preprocessing only
#   3) Higher LR + cosine schedule (head lr=5e-3, LoRA lr=1e-3)
#   4) No prompt learner
#   5) No label refinement
# Kept from the project framework: ELR+GCE + EMA teacher + GMM + SupCon.
set -e
cd "$(dirname "$0")/.."

EXP_NAME="p0_linhead_noaug_lr1e-3_cosine"
mkdir -p outputs/${EXP_NAME}

python -u main.py \
    --dataset webfg400 \
    --data_root ./data \
    --output_dir ./outputs \
    --exp_name ${EXP_NAME} \
    --seed 42 \
    --overrides \
        model.classifier_type='"fc"' \
        model.classifier_hidden_dim=0 \
        model.use_prompt_learner=False \
        model.zero_shot_init=False \
        data.use_randaug=False \
        data.use_mixup=False \
        data.use_cutmix=False \
        train.lr=1e-3 \
        train.lr_head_multiplier=5 \
        train.lr_scheduler='"cosine"' \
        train.epochs=50 \
        train.warmup_epochs=5 \
        train.early_stopping_patience=15 \
        loss.use_label_refinement=False
