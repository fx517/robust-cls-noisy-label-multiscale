#!/bin/bash
# Train CLIP + LoRA + ELR on WebiNat-5000.
# WebiNat-5000: 5000 classes, ~580K web-crawled images (plants, insects, birds, ...),
# extremely fine-grained with severe label noise and a long-tail distribution.
#
# Note: WebiNat-5000 and WebFG-400 are two separate datasets.
# Unzip the WebiNat-5000 train.zip / test.zip into this directory before training.
set -e
cd "$(dirname "$0")/.."

DATA_ROOT="./data"
OUTPUT_DIR="./outputs"

# Smaller batch due to the larger number of classes
EXP_NAME="webinat5000_r16_lambda5_beta09_bs32_lr1e-4"

python main.py \
    --dataset webinat5000 \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --exp_name ${EXP_NAME} \
    --seed 42 \
    --detect_noise \
    --overrides \
        model.lora_r=16 \
        model.lora_alpha=32 \
        model.lora_dropout=0.1 \
        model.classifier_hidden_dim=1024 \
        model.classifier_dropout=0.3 \
        loss.loss_type='"elr"' \
        loss.elr_lambda=5.0 \
        loss.elr_beta=0.9 \
        data.use_randaug=True \
        data.use_mixup=False \
        data.label_smoothing=0.1 \
        train.lr=1e-4 \
        train.lr_backbone_multiplier=0.05 \
        train.lr_head_multiplier=1.0 \
        train.batch_size=32 \
        train.gradient_accumulation_steps=2 \
        train.epochs=80 \
        train.warmup_epochs=8 \
        train.lr_scheduler='"cosine"' \
        train.use_self_distillation=True \
        train.distill_weight=0.3

echo "Training complete! Results in ${OUTPUT_DIR}/${EXP_NAME}"
