"""
Indexed image dataset for noisy-label training with ELR.

The ELR loss requires sample indices to maintain per-sample EMA targets.
This dataset wraps standard image folders and returns (image, label, index).
"""

from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, List, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

from .sampler import WeightedRandomSampler as CBWeightedSampler


class IndexedImageDataset(Dataset):
    """
    Image classification dataset that returns per-sample indices.

    Required by ELR loss which maintains a memory bank of (N, C) target estimates.
    Supports class-balanced and single-folder layouts.

    Expected directory structure:
        train/
            class_001/
                img1.jpg
                img2.jpg
                ...
            class_002/
                ...
        test/
            img1.jpg
            img2.jpg
            ...
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_train: bool = True,
        class_to_idx: Optional[dict] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train

        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: dict = class_to_idx if class_to_idx else {}

        if is_train:
            self._load_train_folder(extensions)
        else:
            self._load_test_folder(extensions)

    def _load_train_folder(self, extensions: Tuple[str, ...]) -> None:
        """Load class-subfolder structure for training data."""
        class_dirs = sorted(
            [
                d
                for d in self.root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
        )

        if not class_dirs:
            raise RuntimeError(f"No class directories found in {self.root}")

        self.classes = [d.name for d in class_dirs]
        if not self.class_to_idx:
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            for ext in extensions:
                for img_path in class_dir.glob(f"*{ext}"):
                    self.samples.append((str(img_path), class_idx))
                # Also check uppercase
                for img_path in class_dir.glob(f"*{ext.upper()}"):
                    self.samples.append((str(img_path), class_idx))

        # Deduplicate (in case both .jpg and .JPG are matched)
        self.samples = sorted(set(self.samples))

    def _load_test_folder(self, extensions: Tuple[str, ...]) -> None:
        """Load flat-folder structure for test data."""
        for ext in extensions:
            for img_path in self.root.glob(f"*{ext}"):
                self.samples.append((str(img_path), -1))  # -1 = unknown label
            for img_path in self.root.glob(f"*{ext.upper()}"):
                self.samples.append((str(img_path), -1))
        self.samples = sorted(set(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """
        Returns:
            image: (C, H, W) tensor
            label: integer class label (-1 for test set)
            index: sample index (required by ELR loss)
        """
        path, target = self.samples[index]

        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            # Fallback: return a blank image on read error
            print(f"Warning: failed to load {path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target, index

    def get_class_distribution(self) -> dict:
        """Return class-to-count mapping (for long-tail analysis)."""
        dist = {cls_name: 0 for cls_name in self.classes}
        idx_to_name = {v: k for k, v in self.class_to_idx.items()}
        for _, label in self.samples:
            name = idx_to_name.get(label, "unknown")
            dist[name] = dist.get(name, 0) + 1
        return dist

    def detect_noisy_labels(
        self, target_matrix: torch.Tensor
    ) -> Tuple[List[int], List[int]]:
        """
        Use ELR target estimates to detect potentially noisy samples.

        A sample is likely mislabeled if argmax(target_ema) != given_label.

        Args:
            target_matrix: (N, C) ELR target estimates from training

        Returns:
            noisy_indices: list of indices suspected to be noisy
            clean_indices: list of indices suspected to be clean
        """
        noisy_indices = []
        clean_indices = []

        for idx, (_, label) in enumerate(self.samples):
            if idx >= target_matrix.shape[0]:
                break
            predicted_label = target_matrix[idx].argmax().item()
            if predicted_label != label:
                noisy_indices.append(idx)
            else:
                clean_indices.append(idx)

        return noisy_indices, clean_indices


def stratified_split(samples, val_split, seed=42):
    """Stratified split by class, returns (train_idx, val_idx) lists.

    Unlike the original random_split: each class keeps a fixed val_split ratio
    (at least 1 sample) as held-out validation, ensuring long-tail tail classes
    also have validation samples that did not participate in training, so a
    class-balanced honest evaluation can be done (same distribution as the
    official balanced test set).

    Determinism: samples are already sorted and deduplicated when produced by
    IndexedImageDataset, and this function additionally sorts within each class
    before rng.shuffle, so given the same samples/val_split/seed the result is
    consistent across processes, and the evaluation script can reproduce the
    same split.
    """
    import random as _random

    by_class = defaultdict(list)
    for i, (_, lbl) in enumerate(samples):
        by_class[lbl].append(i)

    rng = _random.Random(seed)
    train_idx, val_idx = [], []
    for lbl in sorted(by_class.keys()):
        idxs = sorted(by_class[lbl])  # stable order, reproducible across processes
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_split)))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx


def get_dataloaders(
    train_root: str,
    test_root: str,
    train_transform: Callable,
    test_transform: Callable,
    batch_size: int = 64,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    val_split: float = 0.1,
    use_balanced_sampler: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader,
           IndexedImageDataset, torch.utils.data.DataLoader]:
    """
    Create train, test, and validation dataloaders.

    When test set has no labels (competition format), creates a validation
    set by splitting the training data.

    Returns:
        train_loader, test_loader, train_dataset, val_loader
    """
    train_dataset = IndexedImageDataset(
        root=train_root,
        transform=train_transform,
        is_train=True,
    )

    test_dataset = IndexedImageDataset(
        root=test_root,
        transform=test_transform,
        is_train=False,
    )

    # Create train/val split from labeled training data (stratified held-out)
    num_train = len(train_dataset)
    train_idx, val_idx = stratified_split(
        train_dataset.samples, val_split=val_split, seed=42
    )
    num_val = len(val_idx)
    num_train_final = len(train_idx)

    train_subset = Subset(train_dataset, train_idx)

    # CRITICAL: Create a separate val dataset with test transforms (no augmentation)
    # The val subset shares indices with train_dataset, but needs clean transforms
    val_dataset = IndexedImageDataset(
        root=train_root,
        transform=test_transform,
        is_train=True,
    )
    val_subset = Subset(val_dataset, val_idx)

    if use_balanced_sampler:
        # Balanced sampling: oversample tail classes by effective number of samples
        # (Class-Balanced, Cui et al. CVPR 2019), mitigating the distribution shift
        # between the long-tail training set and the official uniform test set.
        # train_subset.indices keep the original dataset indices, and the label
        # semantics align with ELR's per-sample index.
        labels = [train_dataset.samples[i][1] for i in train_subset.indices]
        weights = CBWeightedSampler.compute_weights(
            labels, num_classes=len(train_dataset.classes)
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_subset),
            replacement=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            drop_last=True,
        )
        print(f"  Balanced sampling enabled: WeightedRandomSampler "
              f"(Class-Balanced beta=0.9999, {len(train_subset)} samples/epoch)")
    else:
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            drop_last=True,
        )

    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    print(f"  Train/Val split: {num_train_final} train + {num_val} val "
          f"(from {num_train} total labeled)")

    return train_loader, test_loader, train_dataset, val_loader
