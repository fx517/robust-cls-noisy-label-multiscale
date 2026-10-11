"""
Data augmentation and transforms for fine-grained image recognition.

Uses CLIP normalization stats. Training augmentations include RandAugment,
while test transforms use center crop or multi-crop for TTA.
"""

from typing import Tuple

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

# CLIP ViT-B/32 normalization
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_train_transforms(
    image_size: int = 224,
    use_randaug: bool = True,
    randaug_n: int = 2,
    randaug_m: int = 10,
) -> transforms.Compose:
    """
    Training transforms with RandAugment for regularization.

    RandAugment helps prevent overfitting to noisy labels by applying
    aggressive data augmentation.
    """
    if not use_randaug:
        # Community alignment conclusion (empirically verified in the Luqhhh repo
        # for the same competition track): for fine-grained + noisy-label tasks,
        # strong augmentations (RandomResizedCrop/ColorJitter/RandomErasing) bring
        # no positive gain and even a negative one. Only the standard CLIP
        # preprocessing is kept (Resize 256 -> CenterCrop 224).
        return transforms.Compose([
            transforms.Resize(
                int(image_size * 256 / 224),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    aug_list = [
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.5, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
    ]

    if use_randaug:
        try:
            from torchvision.transforms import RandAugment
            aug_list.append(RandAugment(num_ops=randaug_n, magnitude=randaug_m))
        except ImportError:
            # Fallback if torchvision < 0.12
            pass

    aug_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ])

    return transforms.Compose(aug_list)


class MultiScaleTTA:
    """
    Multi-scale + flip test-time augmentation.

    At each resize scale, produces 10 distinct views via TenCrop
    (5 positions x original/flipped) and stacks them into (N, C, H, W).

    Fixes the previous TTA defects:
    - 224x224 TenCrop severe overlap: crops are now taken from images resized
      to scales LARGER than the crop size, so crops are actually distinct.
    - Lack of horizontal flip: TenCrop includes the horizontally-flipped views.
    - Lack of multi-scale: multiple resize scales are used.
    """

    def __init__(
        self,
        image_size: int = 224,
        crop_size: int = 224,
        scales: Tuple[int, ...] = None,
    ):
        self.crop_size = crop_size
        self.scales = (
            tuple(scales)
            if scales is not None
            else (image_size + 32, image_size + 64)
        )
        self.normalize = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

    def __call__(self, img) -> torch.Tensor:
        views = []
        for s in self.scales:
            resized = TF.resize(
                img, s, interpolation=InterpolationMode.BICUBIC
            )
            for crop in TF.ten_crop(resized, self.crop_size):
                views.append(self.normalize(TF.to_tensor(crop)))
        return torch.stack(views)  # (num_scales * 10, C, H, W)


def get_test_transforms(
    image_size: int = 224,
    crop_size: int = 224,
    is_tta: bool = False,
    tta_crops: int = 10,
) -> transforms.Compose:
    """
    Test transforms. Single center crop by default; multi-crop for TTA mode.

    Args:
        image_size: Resize short edge to this size
        crop_size: Center crop size
        is_tta: If True, use multi-scale + flip TTA (overrides the previous
            overlapping 224x224 TenCrop)
        tta_crops: Unused; kept for backward compatibility.
    """
    if is_tta:
        return transforms.Compose([
            MultiScaleTTA(
                image_size=image_size,
                crop_size=crop_size,
            ),
        ])

    return transforms.Compose([
        transforms.Resize(
            image_size, interpolation=InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


class MixUpCutMixCollator:
    """
    Batch collator that applies MixUp and/or CutMix to training batches.

    Supports two modes:
    - keep_hard_labels=False (default): Standard MixUp/CutMix with soft labels.
      Best for GCE/CE loss but NOT compatible with ELR's per-sample EMA targets.
    - keep_hard_labels=True: Apply MixUp/CutMix to images only, keep original
      hard labels. Compatible with ELR. Still provides strong image-space
      regularization by creating interpolated/augmented images.
    """

    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 1.0,
        label_smoothing: float = 0.1,
        use_mixup: bool = True,
        use_cutmix: bool = True,
        keep_hard_labels: bool = True,
        mix_prob: float = 0.5,
    ):
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.label_smoothing = label_smoothing
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.keep_hard_labels = keep_hard_labels
        self.mix_prob = mix_prob

    def __call__(self, batch):
        images, labels, indices = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)
        indices = torch.tensor(indices)

        if not self.use_mixup and not self.use_cutmix:
            return images, labels, indices

        # Only apply mixing with probability mix_prob
        if torch.rand(1).item() > self.mix_prob:
            return images, labels, indices

        # Decide between mixup and cutmix
        if self.use_mixup and self.use_cutmix:
            use_cutmix = torch.rand(1).item() < 0.5
        else:
            use_cutmix = self.use_cutmix

        if use_cutmix and images.size(0) > 1:
            images, labels = self._cutmix(images, labels)
        elif self.use_mixup and images.size(0) > 1:
            images, labels = self._mixup(images, labels)

        return images, labels, indices

    def _mixup(self, images, labels):
        """Apply MixUp: x' = λ*x_i + (1-λ)*x_j."""
        lam = float(
            torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample()
        )
        lam = max(lam, 1 - lam)  # Enforce lam >= 0.5 for stability

        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)

        mixed_images = lam * images + (1 - lam) * images[index]

        if self.keep_hard_labels:
            # Keep original hard labels for ELR compatibility
            # Image mixing alone provides strong regularization
            return mixed_images, labels

        # Soft labels: standard MixUp
        labels_onehot = torch.zeros(
            batch_size, self.num_classes, device=images.device
        )
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1.0)

        labels_onehot_b = torch.zeros(
            batch_size, self.num_classes, device=images.device
        )
        labels_onehot_b.scatter_(1, labels[index].unsqueeze(1), 1.0)

        mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot_b

        if self.label_smoothing > 0:
            mixed_labels = (
                mixed_labels * (1 - self.label_smoothing)
                + self.label_smoothing / self.num_classes
            )

        return mixed_images, mixed_labels

    def _cutmix(self, images, labels):
        """Apply CutMix: replace a patch of x_i with x_j."""
        lam = float(
            torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample()
        )
        lam = max(lam, 1 - lam)

        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)

        # Generate bounding box
        W, H = images.size(3), images.size(2)
        cut_rat = (1 - lam) ** 0.5
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = torch.randint(0, W, (1,)).item()
        cy = torch.randint(0, H, (1,)).item()
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, cx + cut_w // 2)
        y2 = min(H, cy + cut_h // 2)

        # Apply cutmix
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        if self.keep_hard_labels:
            # Keep original hard labels for ELR compatibility
            return mixed_images, labels

        # Adjust lambda to actual area ratio
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

        labels_onehot = torch.zeros(
            batch_size, self.num_classes, device=images.device
        )
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1.0)

        labels_onehot_b = torch.zeros(
            batch_size, self.num_classes, device=images.device
        )
        labels_onehot_b.scatter_(1, labels[index].unsqueeze(1), 1.0)

        mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot_b

        if self.label_smoothing > 0:
            mixed_labels = (
                mixed_labels * (1 - self.label_smoothing)
                + self.label_smoothing / self.num_classes
            )

        return mixed_images, mixed_labels
