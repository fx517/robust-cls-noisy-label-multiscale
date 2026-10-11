"""
Evaluation module for fine-grained image classification.

Supports standard evaluation and test-time augmentation (TTA) with
multi-crop voting for improved accuracy.
"""

from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class Evaluator:
    """
    Evaluation wrapper for classification models.

    Features:
    - Top-1 and Top-5 accuracy computation
    - Test-time augmentation with multi-crop voting
    - Class-wise accuracy breakdown
    - Confusion matrix support
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        use_amp: bool = True,
    ):
        self.model = model
        self.device = device
        self.use_amp = use_amp

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        num_classes: int = 400,
        return_confusion: bool = False,
        desc: str = "Evaluating",
    ) -> dict:
        """
        Evaluate model on a dataloader.

        Args:
            dataloader: Test dataloader
            num_classes: Number of classes
            return_confusion: Whether to compute confusion matrix
            desc: Progress bar description

        Returns:
            Dict with keys: top1_acc, top5_acc, per_class_acc (optional),
                           confusion_matrix (optional), num_samples
        """
        self.model.eval()

        correct_top1 = 0
        correct_top5 = 0
        total = 0

        # Per-class tracking
        class_correct = torch.zeros(num_classes, device=self.device)
        class_total = torch.zeros(num_classes, device=self.device)

        # Confusion matrix
        if return_confusion:
            confusion = torch.zeros(
                num_classes, num_classes, device=self.device, dtype=torch.long
            )
        else:
            confusion = None

        pbar = tqdm(dataloader, desc=desc, leave=False)
        for batch in pbar:
            images, labels, _ = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)

            # Top-1 accuracy
            _, preds = logits.topk(1, dim=1)
            correct_top1 += (preds.squeeze(1) == labels).sum().item()

            # Top-5 accuracy
            _, preds_top5 = logits.topk(min(5, num_classes), dim=1)
            correct_top5 += sum(
                labels[i].item() in preds_top5[i].tolist()
                for i in range(len(labels))
            )

            total += labels.size(0)

            # Per-class stats
            for c in range(num_classes):
                mask = labels == c
                class_total[c] += mask.sum()
                if mask.any():
                    class_correct[c] += (
                        (preds.squeeze(1)[mask] == c).sum().item()
                    )

            if confusion is not None:
                for i in range(len(labels)):
                    confusion[labels[i].item(), preds[i].item()] += 1

            # Update progress bar
            pbar.set_postfix({
                "acc": f"{100.0 * correct_top1 / total:.2f}%"
            })

        results = {
            "top1_acc": correct_top1 / total if total > 0 else 0.0,
            "top5_acc": correct_top5 / total if total > 0 else 0.0,
            "num_samples": total,
        }

        # Per-class accuracy
        non_empty = class_total > 0
        if non_empty.any():
            per_class_acc = class_correct[non_empty] / class_total[non_empty]
            results["per_class_acc"] = per_class_acc.cpu().tolist()
            results["mean_per_class_acc"] = per_class_acc.mean().item()
            results["min_per_class_acc"] = per_class_acc.min().item()
            results["max_per_class_acc"] = per_class_acc.max().item()

        if confusion is not None:
            results["confusion_matrix"] = confusion.cpu()

        return results

    @torch.no_grad()
    def evaluate_tta(
        self,
        dataloader: DataLoader,
        num_classes: int = 400,
        tta_crops: int = 10,
    ) -> dict:
        """
        Evaluate with test-time augmentation (multi-crop voting).

        For each image, generates tta_crops (via TenCrop) and averages
        the predictions for a more robust final prediction.

        Args:
            dataloader: Test dataloader (should use TTA transforms)
            num_classes: Number of classes
            tta_crops: Number of crops per image (5 or 10)

        Returns:
            Dict with top1_acc, top5_acc
        """
        self.model.eval()

        all_preds = []
        all_labels = []

        pbar = tqdm(dataloader, desc="Evaluating TTA", leave=False)
        for images, labels, _ in pbar:
            # TTA: images shape (B, tta_crops, C, H, W)
            bs, nc, c, h, w = images.shape
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Merge batch and crop dimensions
            images = images.view(-1, c, h, w)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
            logits = logits.view(bs, nc, -1)

            # Average predictions across crops
            avg_logits = logits.mean(dim=1)
            preds = avg_logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        top1 = (all_preds == all_labels).float().mean().item()

        return {
            "top1_acc": top1,
            "num_samples": len(all_labels),
        }

    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
        return_probs: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate predictions for a test set (no labels needed).

        Args:
            dataloader: Test dataloader
            return_probs: If True, also return softmax probabilities

        Returns:
            (predictions, probabilities or None)
        """
        self.model.eval()

        all_preds = []
        all_probs = [] if return_probs else None

        for images, _, _ in tqdm(dataloader, desc="Predicting"):
            images = images.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)

            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())

            if return_probs:
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu())

        all_preds = torch.cat(all_preds)

        if return_probs:
            all_probs = torch.cat(all_probs)
            return all_preds, all_probs

        return all_preds, None
