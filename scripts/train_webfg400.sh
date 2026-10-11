#!/bin/bash
# Train CLIP + LoRA + ELR on WebFG-400.
# WebFG-400: 400 classes, ~43K web-crawled images (aircraft, cars, birds) with label noise.
# Recipe: cosine head + CoOp prompt learner + GMM selection + SupCon + self-distillation.
set -e
cd "$(dirname "$0")/.."

DATA_ROOT="./data"
OUTPUT_DIR="./outputs"

# Experiment name encodes the key hyperparameters
EXP_NAME="webfg400_r16_lambda3_cosine_gmm_supcon"

python main.py \
    --dataset webfg400 \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --exp_name ${EXP_NAME} \
    --seed 42 \
    --detect_noise \
    --overrides \
        model.lora_r=16 \
        model.lora_alpha=32 \
        model.lora_dropout=0.1 \
        model.classifier_hidden_dim=512 \
        model.classifier_dropout=0.3 \
        model.classifier_type='"cosine"' \
        model.use_prompt_learner=True \
        model.prompt_n_ctx=8 \
        model.prompt_fusion_weight=0.3 \
        loss.loss_type='"elr+gce"' \
        loss.elr_lambda=3.0 \
        loss.elr_beta=0.7 \
        loss.gce_q=0.7 \
        loss.elr_weight=1.0 \
        loss.gce_weight=0.5 \
        loss.use_gmm_selection=True \
        loss.gmm_warmup_epochs=10 \
        loss.use_label_refinement=True \
        loss.refine_alpha=0.5 \
        loss.use_supcon=True \
        loss.supcon_weight=0.1 \
        loss.supcon_temperature=0.07 \
        data.use_randaug=True \
        data.use_mixup=False \
        data.label_smoothing=0.1 \
        train.lr=1e-4 \
        train.batch_size=64 \
        train.epochs=100 \
        train.warmup_epochs=5 \
        train.lr_scheduler='"cosine"' \
        train.use_self_distillation=True \
        train.distill_weight=0.5 \
        train.ema_teacher_alpha=0.999

echo "Training complete! Results in ${OUTPUT_DIR}/${EXP_NAME}"
