#!/usr/bin/env python3
"""
Generate test set predictions and pred_results.csv for competition submission.

Usage:
    python scripts/predict_test.py \
        --resume outputs/fast30/best_model.pth \
        --data_root ./data \
        --output pred_results.csv
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import CLIPLoRAModel
from data.dataset import IndexedImageDataset
from data.transforms import get_test_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True, help="Path to best model checkpoint")
    parser.add_argument("--data_root", default="./data", help="Data root directory")
    parser.add_argument("--output", default="pred_results.csv", help="Output CSV file")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--use_tta", action="store_true", help="Use TTA for better accuracy")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA teacher weights instead of student")
    parser.add_argument("--logit_adjust", action="store_true",
                        help="Logit adjustment: correct long-tail train prior to uniform test prior")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint = torch.load(args.resume, map_location=device)
    config = checkpoint.get("config", None)

    num_classes = 750
    # Models from the FixRes resolution-boost experiment (224->384) must be rebuilt
    # according to the image_size in the checkpoint, otherwise the position-embedding
    # size mismatch makes load_state_dict fail.
    image_size = int(getattr(config.model, "image_size", 224)) if config is not None else 224
    print(f"image_size: {image_size}")

    # Rebuild the training-set class names (consistent with main.py), so that the cosine
    # head registers the frozen anchor buffer (classifier.zero_shot_anchor) at zero-shot
    # init; otherwise load_state_dict reports "Unexpected key(s)" because that key exists
    # in the checkpoint.
    class_names = None
    train_root = Path(args.data_root) / "train"
    class_counts = None
    if train_root.is_dir():
        class_names = sorted(
            [d.name for d in train_root.iterdir()
             if d.is_dir() and not d.name.startswith(".")]
        )
        # Number of training samples per class (class-prior correction for logit adjustment)
        if args.logit_adjust:
            class_counts = [
                sum(1 for _ in (train_root / cn).iterdir() if _.is_file())
                for cn in class_names
            ]
            print(f"logit_adjust: counted {len(class_counts)} classes, "
                  f"min={min(class_counts)} max={max(class_counts)}")

    pretrained_path = f"{args.data_root}/ViT-B-32_openai.bin"
    if not Path(pretrained_path).exists():
        pretrained_path = "openai"

    # Build model (use config from checkpoint if available)
    print("Loading model...")
    if config is not None:
        cfg_model = config.model
        model = CLIPLoRAModel(
            num_classes=num_classes,
            image_size=image_size,
            pretrained=pretrained_path,
            lora_r=cfg_model.lora_r,
            lora_alpha=cfg_model.lora_alpha,
            lora_dropout=cfg_model.lora_dropout,
            classifier_type=cfg_model.classifier_type,
            classifier_hidden_dim=cfg_model.classifier_hidden_dim,
            classifier_dropout=cfg_model.classifier_dropout,
            zero_shot_init=cfg_model.zero_shot_init,
            freeze_prototypes=getattr(cfg_model, "freeze_prototypes", False),
            use_prompt_learner=cfg_model.use_prompt_learner,
            prompt_n_ctx=cfg_model.prompt_n_ctx,
            prompt_fusion_weight=cfg_model.prompt_fusion_weight,
            class_names=class_names,
        )
    else:
        model = CLIPLoRAModel(
            num_classes=num_classes,
            image_size=image_size,
            pretrained=pretrained_path,
            lora_r=64, lora_alpha=64,
            classifier_type="cosine",
            classifier_hidden_dim=1024,
            zero_shot_init=True,
            use_prompt_learner=True,
            prompt_n_ctx=16,
            prompt_fusion_weight=0.2,
        )
    model.load_state_dict(
        checkpoint["ema_teacher"]["shadow_state_dict"]
        if args.use_ema and "ema_teacher" in checkpoint
        else checkpoint["model_state_dict"]
    )
    print("Using EMA teacher weights" if args.use_ema else "Using student weights")
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {args.resume}")

    # Test dataset
    test_transform = get_test_transforms(
        image_size=image_size,
        crop_size=image_size,
        is_tta=args.use_tta,
    )
    test_dataset = IndexedImageDataset(
        root=f"{args.data_root}/test",
        transform=test_transform,
        is_train=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    # Generate predictions
    print(f"Predicting {len(test_dataset)} test images...")
    all_filenames = []
    all_logits = []

    use_amp = device.type == "cuda"
    tta_chunk = 512  # max images per forward pass, avoiding the memory spike from the 20-view expansion

    with torch.no_grad():
        for images, _, _ in tqdm(test_loader, desc="Predicting"):
            if args.use_tta:
                bs, nc, c, h, w = images.shape
                images = images.view(-1, c, h, w).to(device)
                # Chunked forward (fp16); sum first, then average over the views
                logits_chunks = []
                for i in range(0, images.size(0), tta_chunk):
                    chunk = images[i:i + tta_chunk]
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            logits_chunks.append(model(chunk))
                    else:
                        logits_chunks.append(model(chunk))
                logits = torch.cat(logits_chunks, dim=0).view(bs, nc, -1).mean(dim=1)
            else:
                images = images.to(device)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(images)
                else:
                    logits = model(images)

            all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits)
    if args.logit_adjust and class_counts is not None:
        # Bayesian correction: subtract the training-set class-frequency prior log(n_y), correcting toward a uniform test distribution
        adj = torch.log(torch.tensor(class_counts, dtype=torch.float32))
        all_logits = all_logits - adj.unsqueeze(0)
        print("Applied logit adjustment (uniform test prior)")
    all_preds = all_logits.argmax(dim=1)

    # Write pred_results.csv
    print(f"Writing {len(test_dataset)} predictions to {args.output}...")
    with open(args.output, "w") as f:
        for i, (path, _) in enumerate(test_dataset.samples):
            filename = Path(path).name
            f.write(f"{filename},{all_preds[i].item()}\n")

    print(f"Done! Result saved to {args.output}")
    print(f"First 5 predictions:")
    with open(args.output) as f:
        for _ in range(5):
            print(f"  {f.readline().strip()}")


if __name__ == "__main__":
    main()
