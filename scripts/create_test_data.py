#!/usr/bin/env python3
"""
Create synthetic test dataset for pipeline validation.

Generates:
    data/test_train/  — synthetic train images with class folders (some noise injected)
    data/test_test/   — synthetic test images (flat, no labels attached)

Each "image" is a 224x224 RGB tensor with class-specific noise patterns,
simulating real data structure for quick pipeline verification.
"""

import random
import numpy as np
from pathlib import Path
from PIL import Image


def create_test_dataset(
    num_classes: int = 50,
    samples_per_class: int = 30,
    test_samples: int = 200,
    noise_rate: float = 0.2,
    image_size: int = 224,
    data_root: str = "./data",
):
    """Create synthetic fine-grained dataset with controllable noise."""

    train_dir = Path(data_root) / "test_train"
    test_dir = Path(data_root) / "test_test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating test dataset:")
    print(f"  Classes: {num_classes}")
    print(f"  Train samples per class: {samples_per_class}")
    print(f"  Test samples: {test_samples}")
    print(f"  Noise rate: {noise_rate * 100:.0f}%")
    print(f"  Image size: {image_size}x{image_size}")

    # --- Class-specific pattern seeds ---
    np.random.seed(42)
    class_patterns = {}
    for c in range(num_classes):
        # Each class has a unique noise pattern
        class_patterns[c] = np.random.randn(3, image_size // 16, image_size // 16)

    # --- Generate training images ---
    train_samples = []
    noisy_count = 0
    sample_idx = 0

    for c in range(num_classes):
        class_dir = train_dir / f"class_{c:03d}"
        class_dir.mkdir(exist_ok=True)

        for s in range(samples_per_class):
            # Determine if this sample is noisy
            is_noisy = random.random() < noise_rate
            if is_noisy:
                noisy_count += 1
                # Assign to random wrong class
                true_class = c
                assigned_class = random.choice(
                    [i for i in range(num_classes) if i != c]
                )
            else:
                true_class = c
                assigned_class = c

            # Generate image with class-specific pattern + random noise
            pattern = class_patterns[true_class]
            pattern_upsampled = np.kron(pattern, np.ones((1, 16, 16)))
            image = pattern_upsampled + 0.1 * np.random.randn(3, image_size, image_size)
            image = np.clip((image * 0.3 + 0.5) * 255, 0, 255).astype(np.uint8)

            # Save with unique name
            filename = f"img_{sample_idx:06d}.jpg"
            img_path = class_dir / filename
            # PIL expects (H, W, C)
            pil_img = Image.fromarray(image.transpose(1, 2, 0))
            pil_img.save(img_path)

            train_samples.append((str(img_path), assigned_class))
            sample_idx += 1

    print(f"  Created {sample_idx} train images ({noisy_count} noisy, "
          f"{noisy_count/sample_idx*100:.1f}%)")

    # --- Generate test images ---
    test_samples_list = []
    for t in range(test_samples):
        true_class = t % num_classes  # Even distribution
        pattern = class_patterns[true_class]
        pattern_upsampled = np.kron(pattern, np.ones((1, 16, 16)))
        image = pattern_upsampled + 0.05 * np.random.randn(3, image_size, image_size)
        image = np.clip((image * 0.3 + 0.5) * 255, 0, 255).astype(np.uint8)

        filename = f"test_{t:06d}.jpg"
        img_path = test_dir / filename
        pil_img = Image.fromarray(image.transpose(1, 2, 0))
        pil_img.save(img_path)

        test_samples_list.append((str(img_path), true_class))

    print(f"  Created {len(test_samples_list)} test images")

    # Save ground truth labels for test set (for evaluation purposes)
    gt_path = Path(data_root) / "test_gt.txt"
    with open(gt_path, "w") as f:
        for path, label in test_samples_list:
            f.write(f"{Path(path).name}\t{label}\n")
    print(f"  Test ground truth saved to {gt_path}")

    # Summary
    print(f"\nDataset ready at {train_dir} and {test_dir}")
    print(f"Quick test command:")
    print(f"  python main.py --dataset webfg400 "
          f"--data_root {data_root} "
          f"--overrides "
          f"data.train_dir='\"test_train\"' "
          f"data.test_dir='\"test_test\"' "
          f"model.num_classes={num_classes} "
          f"train.epochs=5 "
          f"train.warmup_epochs=1 "
          f"data.use_randaug=False "
          f"train.use_self_distillation=False "
          f"train.batch_size=16")
    print()

    return train_samples, test_samples_list, train_dir, test_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=50)
    parser.add_argument("--samples_per_class", type=int, default=30)
    parser.add_argument("--test_samples", type=int, default=200)
    parser.add_argument("--noise_rate", type=float, default=0.2)
    parser.add_argument("--data_root", type=str, default="./data")
    args = parser.parse_args()

    create_test_dataset(
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        test_samples=args.test_samples,
        noise_rate=args.noise_rate,
        data_root=args.data_root,
    )
