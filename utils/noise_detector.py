"""
Noise label detection and analysis utilities.

Provides methods to:
1. Identify potentially mislabeled samples using ELR targets
2. Analyze noise patterns (per-class noise rates, confusion patterns)
3. Generate clean/noisy subsets for curriculum learning
"""

from typing import Tuple, List, Dict

import torch
import numpy as np


class NoiseDetector:
    """
    Noise label detector using ELR target estimates.

    After training with ELR, the per-sample EMA targets capture the model's
    "early" beliefs about each sample. Mismatches between EMA targets and
    given labels indicate likely label noise.
    """

    def __init__(self, target_matrix: torch.Tensor, given_labels: torch.Tensor):
        """
        Args:
            target_matrix: (N, C) ELR EMA targets from training
            given_labels: (N,) original training labels
        """
        self.target_matrix = target_matrix
        self.given_labels = given_labels
        self.num_samples = len(given_labels)
        self.num_classes = target_matrix.shape[1]

    def detect_by_argmax(self) -> torch.Tensor:
        """
        Detect noisy labels: argmax(EMA_target) != given_label.

        Returns:
            (N,) boolean tensor, True = estimated noisy
        """
        predicted_labels = self.target_matrix.argmax(dim=1)
        return predicted_labels != self.given_labels.to(predicted_labels.device)

    def detect_by_confidence(
        self, threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Detect noisy labels: EMA confidence on given label < threshold.

        A sample is considered noisy if the EMA target probability
        for the given label is below the threshold.

        Args:
            threshold: Confidence threshold (0-1). Lower = stricter.

        Returns:
            (N,) boolean tensor, True = estimated noisy
        """
        given_label_probs = self.target_matrix[
            torch.arange(self.num_samples),
            self.given_labels.to(self.target_matrix.device),
        ]
        return given_label_probs < threshold

    def detect_by_entropy(
        self, percentile: float = 90.0
    ) -> torch.Tensor:
        """
        Detect noisy labels: top percentile by prediction entropy.

        Noisy samples tend to have high-entropy predictions (the model
        is unsure because the label contradicts the visual evidence).

        Args:
            percentile: Entropy percentile threshold

        Returns:
            (N,) boolean tensor, True = estimated noisy
        """
        probs = torch.clamp(self.target_matrix, 1e-7, 1.0)
        entropy = -(probs * torch.log(probs)).sum(dim=1)
        threshold = torch.quantile(entropy, percentile / 100.0)
        return entropy > threshold

    def get_clean_noisy_split(
        self, method: str = "argmax", **kwargs
    ) -> Tuple[List[int], List[int]]:
        """
        Split dataset indices into clean and noisy subsets.

        Args:
            method: Detection method ("argmax", "confidence", "entropy")
            **kwargs: Method-specific parameters

        Returns:
            (clean_indices, noisy_indices) as lists
        """
        if method == "argmax":
            is_noisy = self.detect_by_argmax()
        elif method == "confidence":
            is_noisy = self.detect_by_confidence(**kwargs)
        elif method == "entropy":
            is_noisy = self.detect_by_entropy(**kwargs)
        else:
            raise ValueError(f"Unknown detection method: {method}")

        is_noisy_np = is_noisy.cpu().numpy()
        all_indices = np.arange(self.num_samples)

        clean_indices = all_indices[~is_noisy_np].tolist()
        noisy_indices = all_indices[is_noisy_np].tolist()

        return clean_indices, noisy_indices

    def analyze_noise(self) -> Dict:
        """
        Analyze noise patterns in the dataset.

        Returns:
            Dict with:
            - estimated_noise_rate: Overall estimated noise rate
            - per_class_noise_rate: Noise rate per class
            - confusion_pairs: Most common (given_label -> predicted_label) confusions
            - high_noise_classes: Classes with highest estimated noise rates
        """
        is_noisy = self.detect_by_argmax()
        noise_rate = is_noisy.float().mean().item()

        labels = self.given_labels.to(self.target_matrix.device)
        predicted = self.target_matrix.argmax(dim=1)

        # Per-class noise rate
        per_class_noise = {}
        for c in range(self.num_classes):
            class_mask = labels == c
            if class_mask.any():
                per_class_noise[c] = is_noisy[class_mask].float().mean().item()

        # Most confused pairs (given label, predicted label) for noisy samples
        confusion_pairs = {}
        noisy_indices = is_noisy.nonzero(as_tuple=True)[0]
        for idx in noisy_indices[:1000]:  # Limit for performance
            pair = (labels[idx].item(), predicted[idx].item())
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

        top_confusions = sorted(
            confusion_pairs.items(), key=lambda x: x[1], reverse=True
        )[:20]

        # High-noise classes
        high_noise = sorted(
            per_class_noise.items(), key=lambda x: x[1], reverse=True
        )[:20]

        return {
            "estimated_noise_rate": noise_rate,
            "num_noisy_samples": is_noisy.sum().item(),
            "num_total_samples": self.num_samples,
            "per_class_noise_rate": per_class_noise,
            "top_confusion_pairs": [
                {"given": g, "predicted": p, "count": c}
                for (g, p), c in top_confusions
            ],
            "high_noise_classes": [
                {"class_id": c, "noise_rate": r}
                for c, r in high_noise
            ],
        }
