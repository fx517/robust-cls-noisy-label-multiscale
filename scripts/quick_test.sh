#!/bin/bash
# Quick test script — creates synthetic data and runs a full training + eval cycle
# Use this to verify the pipeline works end-to-end before training on real data
#
# Usage: bash scripts/quick_test.sh

set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo " Quick Test: CLIP + LoRA + ELR Pipeline"
echo "========================================"

# Step 1: Create synthetic test data (50 classes, 30 samples/class, 20% noise)
echo -e "\n[1/4] Creating test dataset..."
python scripts/create_test_data.py \
    --num_classes 50 \
    --samples_per_class 30 \
    --test_samples 200 \
    --noise_rate 0.2 \
    --data_root ./data

# Step 2: Quick training (5 epochs, small batch, no aug for speed)
echo -e "\n[2/4] Quick training (5 epochs)..."
python main.py \
    --dataset webfg400 \
    --data_root ./data \
    --output_dir ./outputs \
    --exp_name quick_test \
    --seed 42 \
    --detect_noise \
    --overrides \
        data.train_dir='"test_train"' \
        data.test_dir='"test_test"' \
        model.num_classes=50 \
        model.lora_r=8 \
        model.lora_alpha=16 \
        model.classifier_hidden_dim=256 \
        model.classifier_dropout=0.2 \
        model.classifier_type='"cosine"' \
        model.use_prompt_learner=True \
        model.prompt_n_ctx=4 \
        model.prompt_fusion_weight=0.3 \
        loss.loss_type='"elr+gce"' \
        loss.elr_lambda=3.0 \
        loss.elr_beta=0.7 \
        loss.gce_q=0.7 \
        loss.elr_weight=1.0 \
        loss.gce_weight=0.5 \
        loss.use_gmm_selection=True \
        loss.gmm_warmup_epochs=2 \
        loss.use_label_refinement=False \
        loss.use_supcon=False \
        data.use_randaug=False \
        data.use_mixup=False \
        data.label_smoothing=0.1 \
        train.lr=1e-4 \
        train.batch_size=16 \
        train.epochs=5 \
        train.warmup_epochs=1 \
        train.lr_scheduler='"cosine"' \
        train.use_self_distillation=True \
        train.distill_weight=0.3 \
        train.ema_teacher_alpha=0.999 \
        train.ema_teacher_warmup=30 \
        train.early_stopping_patience=3

echo -e "\n[3/4] Evaluating best model..."
python main.py \
    --eval_only \
    --resume ./outputs/quick_test/best_model.pth \
    --dataset webfg400 \
    --data_root ./data \
    --overrides \
        data.train_dir='"test_train"' \
        data.test_dir='"test_test"' \
        model.num_classes=50

echo -e "\n[4/4] Running TTA evaluation..."
python main.py \
    --eval_only \
    --resume ./outputs/quick_test/best_model.pth \
    --dataset webfg400 \
    --data_root ./data \
    --overrides \
        data.train_dir='"test_train"' \
        data.test_dir='"test_test"' \
        model.num_classes=50 \
        eval.use_tta=True \
        eval.tta_crops=10

echo -e "\n========================================"
echo " Quick test complete!"
echo " Results in: ./outputs/quick_test/"
echo "========================================"
