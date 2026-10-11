"""
Balanced sampling strategies for long-tail noisy-label datasets.

Web-crawled fine-grained datasets often have severely imbalanced class
distributions. This module provides balanced batch sampling to mitigate
the long-tail effect during training.
"""

from typing import Iterator, List

import numpy as np
import torch
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    """
    Samples batches where each class appears roughly equally often.

    This helps prevent the model from being biased toward head classes
    in long-tail distributions typical of web-crawled data.

    Algorithm:
        1. Group sample indices by class
        2. Each batch: randomly select N classes, then randomly select
           batch_size/N samples from each selected class
    """

    def __init__(
        self,
        labels: List[int],
        num_classes: int,
        samples_per_class: int = 4,
        batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        """
        Args:
            labels: List of labels for all training samples
            num_classes: Total number of classes
            samples_per_class: Number of samples per class per batch
            batch_size: Total batch size (should be divisible by samples_per_class)
            shuffle: Whether to shuffle indices
            drop_last: Whether to drop the last incomplete batch
        """
        self.labels = np.array(labels)
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Group indices by class
        self.class_indices: List[np.ndarray] = []
        for c in range(num_classes):
            indices = np.where(self.labels == c)[0]
            if len(indices) > 0:
                self.class_indices.append(indices)
            else:
                self.class_indices.append(np.array([], dtype=np.int64))

        # Effective number of classes (exclude empty classes)
        self.valid_classes = [
            c for c in range(num_classes) if len(self.class_indices[c]) > 0
        ]

        # Number of classes per batch
        self.classes_per_batch = batch_size // samples_per_class
        if batch_size % samples_per_class != 0:
            self.classes_per_batch = max(1, self.classes_per_batch)
            self.samples_per_class = batch_size // self.classes_per_batch

        # Total number of batches
        min_class_size = min(
            len(self.class_indices[c]) for c in self.valid_classes
        )
        self.num_batches = min_class_size // self.samples_per_class
        self.num_batches = min(
            self.num_batches,
            len(self.valid_classes) // self.classes_per_batch,
        )

        if self.num_batches == 0:
            # Fallback: random sampling
            self.num_batches = len(self.labels) // batch_size

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle within each class
        for c in range(self.num_classes):
            if len(self.class_indices[c]) > 0:
                np.random.shuffle(self.class_indices[c])

        # Generate batches
        for batch_idx in range(self.num_batches):
            batch = []

            # Select classes for this batch
            selected_classes = np.random.choice(
                self.valid_classes,
                size=self.classes_per_batch,
                replace=len(self.valid_classes) < self.classes_per_batch,
            )

            for cls in selected_classes:
                # Get samples_per_class indices from this class
                start = (batch_idx * self.samples_per_class) % len(
                    self.class_indices[cls]
                )
                indices = self.class_indices[cls][
                    start : start + self.samples_per_class
                ]
                if len(indices) < self.samples_per_class:
                    # Wrap around
                    indices = np.concatenate([
                        indices,
                        self.class_indices[cls][
                            : self.samples_per_class - len(indices)
                        ],
                    ])
                batch.extend(indices.tolist())

            if self.shuffle:
                np.random.shuffle(batch)

            yield batch[: self.batch_size]

    def __len__(self) -> int:
        return self.num_batches


class WeightedRandomSampler:
    """
    Weighted sampling to oversample tail classes.

    Uses effective number of samples per class to compute weights,
    following the Class-Balanced Loss paper (Cui et al., CVPR 2019).
    """

    @staticmethod
    def compute_weights(
        labels: List[int], num_classes: int, beta: float = 0.9999
    ) -> torch.Tensor:
        """
        Compute per-sample weights based on class frequency.

        weight_i = (1 - beta) / (1 - beta^{n_i})
        where n_i is the number of samples of class i.

        Args:
            labels: List of labels for all training samples
            num_classes: Total number of classes
            beta: Hyperparameter controlling re-weighting strength
                  (0 = no re-weighting, 1 = inverse frequency)

        Returns:
            Per-sample weights tensor
        """
        labels = np.array(labels)
        class_counts = np.bincount(labels, minlength=num_classes)
        effective_num = 1.0 - np.power(beta, class_counts)
        effective_num = np.where(effective_num == 0, 1.0, effective_num)

        # Smooth with small epsilon
        per_cls_weights = (1.0 - beta) / (effective_num + 1e-8)
        per_cls_weights = per_cls_weights / per_cls_weights.sum() * num_classes

        sample_weights = per_cls_weights[labels]
        return torch.from_numpy(sample_weights.astype(np.float32))
