#!/usr/bin/env python3
"""
Evaluate TTA (multi-scale + flip) on the labeled validation split.

The baseline 74.83% is single-crop val accuracy. This script measures the
multi-scale+flip TTA accuracy on the same split to verify the TTA fix.

Usage:
    python scripts/eval_tta_val.py \
        --resume outputs/reproduce_default_baseline/best_model.pth \
        --data_root ./data
"""

import argparse
import sys
from pathlib import Path

# Allow running from anywhere by adding the project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from data import get_dataloaders
from data.transforms import get_train_transforms, get_test_transforms
from models import CLIPLoRAModel
from trainers.evaluator import Evaluator


def build_model(checkpoint, pretrained_path, num_classes, device, image_size=224):
    config = checkpoint.get("config", None)
    if config is not None:
        cm = config.model
        return CLIPLoRAModel(
            num_classes=num_classes,
            image_size=image_size,
            pretrained=pretrained_path,
            lora_r=cm.lora_r,
            lora_alpha=cm.lora_alpha,
            lora_dropout=cm.lora_dropout,
            classifier_type=cm.classifier_type,
            classifier_hidden_dim=cm.classifier_hidden_dim,
            classifier_dropout=cm.classifier_dropout,
            zero_shot_init=cm.zero_shot_init,
            use_prompt_learner=cm.use_prompt_learner,
            prompt_n_ctx=cm.prompt_n_ctx,
            prompt_fusion_weight=cm.prompt_fusion_weight,
        ).to(device)
    return CLIPLoRAModel(
        num_classes=num_classes, image_size=image_size, pretrained=pretrained_path
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True, help="Best model checkpoint")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224,
                        help="Fallback image size used only when the checkpoint stores no config")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA teacher weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    num_classes = 750
    pretrained_path = f"{args.data_root}/ViT-B-32_openai.bin"
    if not Path(pretrained_path).exists():
        pretrained_path = "openai"

    print("Loading checkpoint...")
    checkpoint = torch.load(args.resume, map_location=device)
    # Adopt the resolution from the checkpoint config when present (FixRes checkpoints are
    # resolution-specific); fall back to --image_size otherwise.
    cfg = checkpoint.get("config", None)
    image_size = int(getattr(cfg.model, "image_size", args.image_size)) if cfg is not None else args.image_size
    print(f"image_size={image_size}")
    model = build_model(checkpoint, pretrained_path, num_classes, device, image_size=image_size)
    model.load_state_dict(
        checkpoint["ema_teacher"]["shadow_state_dict"]
        if args.use_ema and "ema_teacher" in checkpoint
        else checkpoint["model_state_dict"]
    )
    print("Using EMA teacher weights" if args.use_ema else "Using student weights")
    model.eval()
    print(f"Loaded {args.resume}")

    train_root = f"{args.data_root}/train"
    test_root = f"{args.data_root}/test"

    # TTA val loader (multi-scale + flip transform) with the exact val split
    train_transform = get_train_transforms(image_size=image_size)
    tta_transform = get_test_transforms(image_size=image_size, crop_size=image_size, is_tta=True)
    _, _, _, val_loader = get_dataloaders(
        train_root=train_root,
        test_root=test_root,
        train_transform=train_transform,
        test_transform=tta_transform,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        val_split=0.1,
    )

    evaluator = Evaluator(model, device, use_amp=True)
    results = evaluator.evaluate_tta(val_loader, num_classes=num_classes)
    print(f"\nTTA (multi-scale+flip) val Top-1: {results['top1_acc'] * 100:.2f}%")
    print("Baseline single-crop val Top-1: 74.83%")


if __name__ == "__main__":
    main()