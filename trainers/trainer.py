"""
Main training loop for CLIP + LoRA + ELR fine-tuning.

Handles:
- Mixed precision training (AMP)
- Gradient accumulation for large effective batch sizes
- EMA Teacher for self-distillation (Mean Teacher style)
- GMM-based clean/noisy sample selection (DivideMix-style)
- Supervised contrastive loss (SupCon) for better features
- Label co-refinement for noisy samples
- Logging (TensorBoard, W&B)
- Checkpointing with best-model tracking
- Early stopping
"""

from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.cuda.amp import GradScaler

from tqdm import tqdm

from .evaluator import Evaluator
from .lr_scheduler import build_scheduler
from losses.ema_teacher import EMATeacher
from losses.noise_filter import GMMSampleSelector, CoRefinementModule
from losses.supcon import NoiseRobustSupConLoss
from losses.co_teaching import CoTeachingSelector, CoTeachingLoss


class Trainer:
    """
    Trainer for CLIP + LoRA + ELR on noisy-label fine-grained datasets.

    Features:
    - AMP training for memory efficiency
    - EMA teacher model for stable self-distillation
    - GMM-based dynamic clean/noisy sample separation
    - Supervised contrastive loss on clean samples
    - Label co-refinement for potential noisy samples
    - Gradient accumulation for large effective batch sizes
    - Early stopping on validation metric
    - Automatic checkpointing of best model
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config,
        device: Optional[torch.device] = None,
        test_loader: DataLoader = None,
        num_train_full: int = None,
        full_train_dataset = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.config = config
        self.full_train_dataset = full_train_dataset

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        # Move the criterion (including ELR's target buffer) onto the GPU, eliminating the
        # per-batch GPU<->CPU sync (the ELR target used to be on CPU, so every forward/get_targets triggered one).
        self.criterion = self.criterion.to(self.device)

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Scheduler
        steps_per_epoch = len(train_loader) // config.train.gradient_accumulation_steps
        self.scheduler = build_scheduler(
            self.optimizer,
            scheduler_type=config.train.lr_scheduler,
            warmup_epochs=config.train.warmup_epochs,
            total_epochs=config.train.epochs,
            warmup_lr=config.train.warmup_lr,
            min_lr=config.train.min_lr,
            steps_per_epoch=steps_per_epoch,
        )
        self.scheduler_is_per_batch = (config.train.lr_scheduler == "onecycle")

        # AMP
        self.use_amp = config.train.use_amp
        self.scaler = GradScaler(enabled=self.use_amp)

        # Evaluator
        self.evaluator = Evaluator(self.model, self.device, use_amp=self.use_amp)

        # --- EMA Teacher (for self-distillation) ---
        self.use_ema_teacher = config.train.use_self_distillation
        if self.use_ema_teacher:
            self.ema_teacher = EMATeacher(
                self.model,
                alpha=config.train.ema_teacher_alpha,
                warmup_steps=config.train.ema_teacher_warmup,
                use_dynamic_alpha=True,
            )
            self.ema_teacher.to(self.device)
            print(f"[Trainer] EMA Teacher enabled (alpha={config.train.ema_teacher_alpha})")
        else:
            self.ema_teacher = None

        # --- GMM Sample Selection ---
        num_gmm = num_train_full if num_train_full is not None else len(train_loader.dataset)
        self.use_gmm = config.loss.use_gmm_selection
        if self.use_gmm:
            self.sample_selector = GMMSampleSelector(
                num_samples=num_gmm,
                momentum=config.loss.gmm_momentum,
                warmup_epochs=config.loss.gmm_warmup_epochs,
                p_threshold=config.loss.gmm_p_threshold,
                use_confidence=True,
            )
            print(f"[Trainer] GMM Sample Selection enabled "
                  f"(warmup={config.loss.gmm_warmup_epochs} epochs)")
        else:
            self.sample_selector = None

        # --- Label Co-Refinement ---
        self.use_label_refine = config.loss.use_label_refinement
        if self.use_label_refine:
            self.refiner = CoRefinementModule(
                num_classes=config.model.num_classes,
                refine_alpha=config.loss.refine_alpha,
                warmup_epochs=config.loss.gmm_warmup_epochs,
                max_alpha=config.loss.refine_max_alpha,
            )
            print(f"[Trainer] Label Co-Refinement enabled")
        else:
            self.refiner = None

        # --- SupCon Loss ---
        self.use_supcon = config.loss.use_supcon
        if self.use_supcon:
            self.supcon_loss = NoiseRobustSupConLoss(
                temperature=config.loss.supcon_temperature,
                num_classes=config.model.num_classes,
            )
            print(f"[Trainer] SupCon loss enabled "
                  f"(temp={config.loss.supcon_temperature}, "
                  f"weight={config.loss.supcon_weight})")
        else:
            self.supcon_loss = None

        # --- Co-Teaching (dual network) ---
        self.use_co_teaching = config.loss.use_co_teaching
        if self.use_co_teaching:
            import copy
            self.model_b = copy.deepcopy(self.model)
            # Different initialization for B
            for param_b in self.model_b.parameters():
                if param_b.requires_grad:
                    nn.init.normal_(param_b.data, std=0.001)

            self.optimizer_b = self._build_optimizer_for_model(self.model_b)
            self.scheduler_b = build_scheduler(
                self.optimizer_b,
                scheduler_type=config.train.lr_scheduler,
                warmup_epochs=config.train.warmup_epochs,
                total_epochs=config.train.epochs,
                warmup_lr=config.train.warmup_lr,
                min_lr=config.train.min_lr,
                steps_per_epoch=len(train_loader) // config.train.gradient_accumulation_steps,
            )

            # Reuse criterion but create separate instance for model B
            # (if ELR-based, need separate EMA targets)
            self.criterion_b = copy.deepcopy(criterion)

            self.co_teach_selector = CoTeachingSelector(
                forget_rate=config.loss.co_teach_forget_rate,
                num_gradual=config.loss.co_teach_num_gradual,
                num_classes=config.model.num_classes,
            )
            self.co_teach_loss = CoTeachingLoss(
                self.criterion, self.criterion_b, self.co_teach_selector
            )
            self.model_b = self.model_b.to(self.device)
            print(f"[Trainer] Co-Teaching enabled: forget_rate={config.loss.co_teach_forget_rate}")
        else:
            self.model_b = None
            self.optimizer_b = None
            self.criterion_b = None
            self.co_teach_loss = None

        # Output paths
        self.output_dir = Path(config.output_dir) / config.exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.log_dir) / config.exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.global_step = 0

        # SWA (Stochastic Weight Averaging) — averages model weights over last ~25% epochs
        self.swa_model = None
        self.swa_start_epoch = int(config.train.epochs * 0.75)  # Start SWA at 75% mark
        self.swa_count = 0

        # Metrics history
        self.train_losses = []
        self.val_accs = []

        # --- GRIP class soft labels ---
        # Class-level soft distribution matrix (C, C), initialized to the identity matrix
        # (i.e., initially equivalent to one-hot hard labels).
        # During training it is updated each epoch with the class-wise mean of the ELR
        # targets, gradually replacing hard labels with class soft labels.
        self.class_soft = None
        if getattr(config.loss, 'grip_enabled', False):
            self.class_soft = torch.eye(
                config.model.num_classes, device=self.device
            )

        # Logger
        self.writer = None
        if config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    name=config.exp_name,
                    config=vars(config) if hasattr(config, "__dict__") else {},
                )
                self.use_wandb = True
            except ImportError:
                self.use_wandb = False
        else:
            self.use_wandb = False

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(self.log_dir)
        except ImportError:
            self.tb_writer = None

        # --- Honest validation ---
        # Maintain a denoised EMA soft target for each validation sample; its argmax serves
        # as the "honest reference label", removing the contamination of noisy labels on
        # early stopping / model selection.
        self.honest_validation = getattr(
            config.train, 'honest_validation', False
        )
        self.honest_ema_beta = getattr(config.train, 'honest_ema_beta', 0.9)
        if self.honest_validation:
            # val_loader.dataset is a random_split Subset: .indices are global indices,
            # .dataset is an IndexedImageDataset (its samples order matches train_dataset,
            # and set-dedup hashes agree within the same process).
            val_ds = val_loader.dataset
            val_indices = val_ds.indices
            val_labels = [val_ds.dataset.samples[i][1] for i in val_indices]
            self.val_ema_target = torch.zeros(
                len(val_indices), config.model.num_classes
            )
            self.val_ema_target[
                torch.arange(len(val_indices)),
                torch.tensor(val_labels, dtype=torch.long),
            ] = 1.0
            print(f"[Trainer] Honest validation enabled "
                  f"(val EMA target size={len(val_indices)}, "
                  f"beta={self.honest_ema_beta})")
        else:
            self.val_ema_target = None

    def _build_optimizer(self):
        """Build optimizer with per-parameter learning rates."""
        return self._build_optimizer_for_model(self.model)

    def _build_optimizer_for_model(self, model):
        cfg = self.config.train

        backbone_params = []
        lora_params = []
        head_params = []
        prompt_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" in name or "lora_B" in name:
                lora_params.append(param)
            elif "classifier" in name:
                head_params.append(param)
            elif "prompt_learner" in name:
                prompt_params.append(param)
            elif "visual" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        param_groups = []

        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": cfg.lr * cfg.lr_backbone_multiplier,
                "name": "backbone",
            })
        if lora_params:
            param_groups.append({
                "params": lora_params,
                "lr": cfg.lr,
                "name": "lora",
            })
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": cfg.lr * cfg.lr_head_multiplier,
                "name": "head",
            })
        if prompt_params:
            param_groups.append({
                "params": prompt_params,
                "lr": cfg.lr * 0.5,
                "name": "prompt",
            })
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": cfg.lr,
                "name": "other",
            })

        if cfg.optimizer == "adamw":
            return AdamW(
                param_groups,
                lr=cfg.lr,
                betas=cfg.betas,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "sgd":
            return SGD(
                param_groups,
                lr=cfg.lr,
                momentum=0.9,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """Main training loop."""
        if resume_from:
            self._load_checkpoint(resume_from)

        cfg = self.config.train
        start_epoch = self.current_epoch

        print(f"\n{'='*60}")
        print(f"Training {self.config.exp_name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {start_epoch + 1} -> {cfg.epochs}")
        print(f"Batch size: {cfg.batch_size} "
              f"(×{cfg.gradient_accumulation_steps} accumulation)")
        print(f"Optimizer: {cfg.optimizer}, LR: {cfg.lr}")
        print(f"AMP: {self.use_amp}")
        print(f"EMA Teacher: {self.use_ema_teacher}")
        print(f"GMM Selection: {self.use_gmm}")
        print(f"SupCon: {self.use_supcon}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")

        for epoch in range(start_epoch, cfg.epochs):
            self.current_epoch = epoch

            # Train one epoch
            if self.use_co_teaching:
                train_loss = self._train_epoch_coteach(epoch)
            else:
                train_loss = self._train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Full GMM update at epoch level (after worst samples stabilized)
            if self.use_gmm and epoch >= self.config.loss.gmm_warmup_epochs:
                if hasattr(self.criterion, 'elr') and self.criterion.elr is not None:
                    all_targets = self.criterion.elr.get_all_targets().cpu()  # sklearn needs CPU
                    # Use full_train_dataset (not subset) to get all labels
                    ds = self.full_train_dataset if self.full_train_dataset is not None else self.train_loader.dataset
                    all_labels = torch.tensor(
                        [ds.samples[i][1] for i in range(len(ds))]
                    )
                    self.sample_selector.full_update(all_targets, all_labels, epoch)
                    stats = self.sample_selector.get_probs()
                    print(f"  GMM: clean_ratio={stats['clean_ratio']:.2%}, "
                          f"means=[{stats['gmm_mean_clean']:.4f}, "
                          f"{stats['gmm_mean_noisy']:.4f}]")

            # GRIP class soft label update (at the end of each epoch, using the class mean of ELR targets)
            if (self.config.loss.grip_enabled
                    and epoch >= self.config.loss.grip_warmup
                    and hasattr(self.criterion, 'elr')
                    and self.criterion.elr is not None):
                all_targets = self.criterion.elr.get_all_targets().cpu()  # (N, C) CPU
                ds = (self.full_train_dataset
                      if self.full_train_dataset is not None
                      else self.train_loader.dataset)
                all_labels = torch.tensor(
                    [ds.samples[i][1] for i in range(len(ds))]
                )
                self._update_class_soft(all_targets, all_labels)

            # Evaluate
            if (epoch + 1) % self.config.eval_interval == 0:
                if self.honest_validation:
                    hv = self._honest_validate()
                    val_acc = hv["honest_acc"]  # use honest acc for model selection
                    self.val_accs.append(val_acc)
                    log_str = (
                        f"Epoch {epoch + 1:3d}/{cfg.epochs} | "
                        f"Loss: {train_loss:.4f} | "
                        f"Val Acc (honest): {hv['honest_acc'] * 100:.2f}% | "
                        f"raw: {hv['raw_acc'] * 100:.2f}%"
                    )
                    print(log_str)
                else:
                    val_results = self.evaluator.evaluate(
                        self.val_loader,
                        num_classes=self.config.model.num_classes,
                        desc=f"Val Epoch {epoch + 1}",
                    )
                    val_acc = val_results["top1_acc"]
                    self.val_accs.append(val_acc)
                    log_str = (
                        f"Epoch {epoch + 1:3d}/{cfg.epochs} | "
                        f"Loss: {train_loss:.4f} | "
                        f"Val Acc: {val_acc * 100:.2f}%"
                    )
                    if "top5_acc" in val_results:
                        log_str += f" | Top-5: {val_results['top5_acc'] * 100:.2f}%"
                    if "mean_per_class_acc" in val_results:
                        log_str += (
                            f" | Mean-PC: {val_results['mean_per_class_acc'] * 100:.2f}%"
                        )
                    print(log_str)

                # SWA update (average model weights over late training)
                if epoch >= self.swa_start_epoch:
                    if self.swa_model is None:
                        import copy
                        self.swa_model = copy.deepcopy(self.model)
                        print(f"  SWA started at epoch {epoch + 1}")
                    else:
                        # Running average: θ_swa = (n*θ_swa + θ_new) / (n+1)
                        self.swa_count += 1
                        for swa_param, model_param in zip(
                            self.swa_model.parameters(), self.model.parameters()
                        ):
                            swa_param.data = (
                                self.swa_count * swa_param.data + model_param.data
                            ) / (self.swa_count + 1)

                # TensorBoard
                if self.tb_writer:
                    self.tb_writer.add_scalar("train/loss", train_loss, epoch)
                    self.tb_writer.add_scalar("val/top1_acc", val_acc, epoch)
                    self.tb_writer.add_scalar(
                        "lr", self.optimizer.param_groups[0]["lr"], epoch
                    )

                # W&B
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": train_loss,
                        "val/top1_acc": val_acc,
                        "epoch": epoch,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    })
                    if self.use_gmm and self.sample_selector:
                        stats = self.sample_selector.get_probs()
                        wandb.log({
                            "gmm/clean_ratio": stats["clean_ratio"],
                            "epoch": epoch,
                        })

                # Plateau scheduler (OneCycleLR already stepped per batch)
                if cfg.lr_scheduler == "plateau":
                    self.scheduler.step(val_acc)
                elif cfg.lr_scheduler != "onecycle":
                    self.scheduler.step()

                # Best model tracking
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_checkpoint("best_model.pth", epoch, val_acc)
                    print(f"  >> New best model! Acc: {val_acc * 100:.2f}%")
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= cfg.early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1} "
                          f"(best: {self.best_acc * 100:.2f}% at epoch "
                          f"{self.best_epoch + 1})")
                    break

            # Regular checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(
                    f"checkpoint_epoch_{epoch + 1}.pth", epoch, 0.0
                )

        # Training complete
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best Val Acc: {self.best_acc * 100:.2f}% "
              f"(epoch {self.best_epoch + 1})")

        # Evaluate SWA model
        if self.swa_model is not None:
            original_model = self.model
            self.model = self.swa_model.to(self.device)
            self.model.eval()
            swa_results = self.evaluator.evaluate(
                self.val_loader,
                num_classes=self.config.model.num_classes,
                desc="SWA Val",
            )
            swa_acc = swa_results["top1_acc"]
            print(f"SWA Val Acc: {swa_acc * 100:.2f}%")

            # Save SWA model if it's better
            swa_path = self.output_dir / "swa_model.pth"
            torch.save({
                "model_state_dict": self.swa_model.state_dict(),
                "val_acc": swa_acc,
                "swa_count": self.swa_count,
                "config": self.config,
            }, swa_path)
            print(f"SWA model saved to {swa_path}")

            # Restore original (or keep SWA if better)
            if swa_acc > self.best_acc:
                print(f"  SWA is better! Using SWA model.")
                self.model = self.swa_model
            else:
                self.model = original_model

        print(f"{'='*60}")

        if self.use_wandb:
            import wandb
            wandb.finish()

        if self.tb_writer:
            self.tb_writer.close()

        return {
            "best_acc": self.best_acc,
            "best_epoch": self.best_epoch,
            "train_losses": self.train_losses,
            "val_accs": self.val_accs,
        }

    @torch.no_grad()
    def _honest_validate(self) -> Dict[str, float]:
        """
        Honest validation: evaluate with the denoised EMA target as the reference label,
        rather than the original noisy labels.

        Each validation sample keeps a soft EMA target (initialized to the one-hot of the
        original label); each validation round smoothly updates it with the model's softmax
        output. Its argmax reflects the model's "early + stable" belief (more likely correct
        for noisy samples, matching ELR's early-learning assumption).

        honest_acc = final prediction == argmax(EMA target), which measures the model's
        resistance to noisy-label memorization and is a closer proxy to clean test
        performance than the noisy validation acc.

        Returns:
            {"honest_acc": ..., "raw_acc": ...}
        """
        self.model.eval()
        beta = self.honest_ema_beta
        correct_raw = 0
        correct_honest = 0
        total = 0
        offset = 0
        for images, labels, _ in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            b = images.size(0)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
            preds = logits.argmax(dim=1)
            probs = F.softmax(logits, dim=1)

            # EMA update of the denoised reference label (aligned to the local val order)
            tgt = self.val_ema_target[offset:offset + b].to(self.device)
            tgt = beta * tgt + (1 - beta) * probs
            self.val_ema_target[offset:offset + b] = tgt.cpu()

            correct_raw += (preds == labels).sum().item()
            correct_honest += (preds == tgt.argmax(dim=1)).sum().item()
            total += b
            offset += b

        return {
            "honest_acc": correct_honest / total if total else 0.0,
            "raw_acc": correct_raw / total if total else 0.0,
        }

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        if self.ema_teacher:
            self.ema_teacher.train()
        cfg = self.config.train

        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch + 1}/{cfg.epochs}",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            images, labels, indices = batch
            images = images.to(self.device)
            indices = indices.to(self.device)

            # Handle label format (hard or soft)
            is_soft = (
                labels.dtype in (torch.float16, torch.float32, torch.float64)
                and labels.dim() > 1
            )
            if is_soft:
                labels = labels.to(self.device)
            else:
                labels = labels.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # Forward pass
                features, logits = self.model(images, return_features=True)

                # --- Per-sample loss for GMM ---
                if is_soft:
                    per_sample_loss = -(labels * F.log_softmax(logits, dim=1)
                                       ).sum(dim=1)
                else:
                    per_sample_loss = F.cross_entropy(
                        logits, labels, reduction='none'
                    )

                # --- GMM sample selection (per-batch update) ---
                clean_mask = None
                if self.use_gmm and self.sample_selector:
                    elr_batch = None
                    if hasattr(self.criterion, 'elr') and self.criterion.elr is not None:
                        elr_batch = self.criterion.elr.get_targets(indices)
                    self.sample_selector.update(
                        indices, per_sample_loss, elr_batch, labels, epoch
                    )
                    clean_mask = self.sample_selector.clean_mask[indices.cpu()].to(self.device)

                # --- Main loss (ELR / GCE / Combined) ---
                loss = self._compute_main_loss(indices, logits, labels, features)

                # --- Semi-supervised: noisy samples self-correct with ELR target argmax pseudo-labels ---
                # The noisy subset flagged by GMM no longer "toughs out" the noisy labels; an
                # extra pseudo-label from the argmax of ELR's EMA target pulls it toward the
                # model's early stable belief (an incremental regularizer, not changing the main loss).
                if (self.config.loss.semi_supervised and clean_mask is not None
                        and hasattr(self.criterion, 'elr')
                        and self.criterion.elr is not None):
                    noisy_mask = ~clean_mask
                    if noisy_mask.any():
                        elr_tgt = self.criterion.elr.get_targets(indices)
                        pseudo_labels = elr_tgt.argmax(dim=1)
                        semi_loss = F.cross_entropy(
                            logits[noisy_mask], pseudo_labels[noisy_mask]
                        )
                        loss = loss + self.config.loss.semi_weight * semi_loss

                # --- DeFT soft-label fusion: correct noisy labels with the CLIP zero-shot clean reference ---
                # "Clean logits" computed from visual features and the frozen zero-shot text
                # prototype serve as soft-label supervision, removing the contamination of the
                # noisy hard labels on the model (the core idea of DeFT).
                if self.config.loss.deft_weight > 0 and not is_soft:
                    zs_anchor = getattr(self.model.classifier, "zero_shot_anchor", None)
                    if zs_anchor is not None and zs_anchor.numel() > 0:
                        zs_logits = (
                            features @ zs_anchor.T
                            * self.model.classifier.logit_scale.detach()
                        )
                        p_zs = F.softmax(
                            zs_logits / self.config.loss.deft_temperature, dim=1
                        ).detach()
                        log_probs = F.log_softmax(logits, dim=1)
                        if getattr(self.config.loss, 'deft_detect', False):
                            # Noise detection: the support of the hard label relative to the zero-shot top-1 as the trust level
                            p_zs_at_label = p_zs.gather(1, labels.unsqueeze(1)).squeeze(1)
                            p_zs_max = p_zs.max(dim=1).values
                            trust = p_zs_at_label / (p_zs_max + 1e-7)  # (B,) 0~1
                            # Suspicious samples (low trust) get the zero-shot soft label strongly; clean samples (high trust) get almost none
                            deft_loss = -((1 - trust) * (p_zs * log_probs).sum(dim=1)).mean()
                        else:
                            deft_loss = -(p_zs * log_probs).sum(dim=1).mean()
                        loss = loss + self.config.loss.deft_weight * deft_loss

                # --- GRIP class soft labels: replace one-hot with the class-level soft distribution to reduce overfitting to noisy hard labels ---
                if (self.config.loss.grip_enabled and not is_soft
                        and epoch >= self.config.loss.grip_warmup):
                    soft_cls = self.class_soft[labels]  # (B, C)
                    grip_log_probs = F.log_softmax(logits, dim=1)
                    grip_loss = -(soft_cls * grip_log_probs).sum(dim=1).mean()
                    loss = loss + self.config.loss.grip_weight * grip_loss

                # --- Label refinement ---
                if self.use_label_refine and self.refiner and not is_soft:
                    if clean_mask is not None:
                        noisy_mask = ~clean_mask
                        if noisy_mask.any():
                            with torch.no_grad():
                                pred_probs = F.softmax(logits.detach(), dim=1)
                            refined_labels = self.refiner.refine_labels(
                                labels, pred_probs, noisy_mask, epoch
                            )
                            # Compute refinement loss with soft labels
                            log_probs = F.log_softmax(logits, dim=1)
                            refine_loss = -(
                                refined_labels * log_probs
                            ).sum(dim=1).mean()
                            loss = loss + 0.3 * refine_loss

                # --- SupCon loss (on clean samples only) ---
                if self.use_supcon and self.supcon_loss and clean_mask is not None:
                    if clean_mask.any() and self.ema_teacher:
                        teacher_feats = self.ema_teacher.generate_features(
                            images[clean_mask]
                        )
                        supcon_l = self.supcon_loss(
                            student_features=features[clean_mask],
                            teacher_features=teacher_feats,
                            labels=labels[clean_mask]
                            if not is_soft
                            else labels[clean_mask].argmax(dim=1),
                            clean_mask=torch.ones(
                                clean_mask.sum(), device=self.device
                            ).bool(),
                        )
                        loss = loss + self.config.loss.supcon_weight * supcon_l

                # --- Self-distillation from EMA teacher ---
                if self.use_ema_teacher and self.ema_teacher is not None:
                    if self.global_step >= cfg.ema_teacher_warmup:
                        with torch.no_grad():
                            teacher_logits = self.ema_teacher.generate_logits(images)

                        distill_loss = self._distillation_loss(
                            logits, teacher_logits, cfg.distill_temperature
                        )
                        # Progressive distillation: ramp up weight from 0 to distill_weight
                        # over first 60% of training epochs
                        total_epochs = cfg.epochs
                        progress = min(1.0, epoch / max(1, int(total_epochs * 0.6)))
                        progressive_weight = cfg.distill_weight * progress
                        loss = loss + progressive_weight * distill_loss

                # Scale for gradient accumulation
                loss = loss / cfg.gradient_accumulation_steps

            # Backward with AMP
            self.scaler.scale(loss).backward()

            # Gradient step
            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # OneCycleLR steps per optimizer step
                if self.scheduler_is_per_batch:
                    self.scheduler.step()

                # Update EMA teacher after each optimizer step
                if self.use_ema_teacher and self.ema_teacher:
                    self.ema_teacher.step(self.global_step)

            total_loss += loss.item() * cfg.gradient_accumulation_steps
            num_batches += 1
            self.global_step += 1

            # Progress bar
            avg_loss = total_loss / num_batches
            status = {"loss": f"{avg_loss:.4f}"}
            if self.use_gmm and clean_mask is not None:
                status["clean%"] = f"{clean_mask.float().mean():.2f}"
            pbar.set_postfix(status)

            # Interval logging
            if self.global_step % self.config.log_interval == 0:
                if self.tb_writer:
                    self.tb_writer.add_scalar(
                        "train/step_loss",
                        loss.item() * cfg.gradient_accumulation_steps,
                        self.global_step,
                    )

        return total_loss / max(num_batches, 1)

    def _update_class_soft(self, all_targets: torch.Tensor, all_labels: torch.Tensor):
        """
        GRIP class soft label update: obtain the class-level soft distribution from the
        class-wise mean of the ELR targets, then EMA-blend it with the historical class
        soft labels. Class soft labels smooth single-sample noisy labels and reduce
        overfitting of noisy hard labels to class prototypes.
        """
        num_classes = self.class_soft.shape[0]
        onehot = F.one_hot(all_labels, num_classes).float()  # (N, C) CPU
        cls_sum = onehot.T @ all_targets.float()  # (C, C) CPU sum of within-class ELR targets
        cls_cnt = onehot.sum(dim=0)  # (C,) number of samples per class
        class_soft_new = cls_sum / (cls_cnt.unsqueeze(1) + 1e-6)  # (C, C)
        alpha = self.config.loss.grip_alpha
        self.class_soft = (
            alpha * self.class_soft
            + (1 - alpha) * class_soft_new.to(self.class_soft.device)
        )

    def _train_epoch_coteach(self, epoch: int) -> float:
        """Co-Teaching epoch: two models select clean samples for each other."""
        self.model.train()
        self.model_b.train()
        cfg = self.config.train

        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()
        self.optimizer_b.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"CoTeach Epoch {epoch + 1}/{cfg.epochs}", leave=False)

        for batch_idx, batch in enumerate(pbar):
            images, labels, indices = batch
            images = images.to(self.device)
            indices = indices.to(self.device)
            is_soft = (labels.dtype in (torch.float16, torch.float32, torch.float64) and labels.dim() > 1)
            labels = labels.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # Forward both models
                logits_a = self.model(images)
                logits_b = self.model_b(images)

                # Co-Teaching: each model selects clean samples for the other
                ct_result = self.co_teach_loss.compute_losses(
                    indices, logits_a, logits_b, labels, epoch
                )

                loss = (ct_result["loss_a"] + ct_result["loss_b"]) / 2
                loss = loss / cfg.gradient_accumulation_steps

            # Backward model A
            self.scaler.scale(ct_result["loss_a"] / cfg.gradient_accumulation_steps).backward(retain_graph=True)
            # Backward model B
            self.scaler.scale(ct_result["loss_b"] / cfg.gradient_accumulation_steps).backward()

            # Gradient step for both
            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                for opt in [self.optimizer, self.optimizer_b]:
                    self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.model_b.parameters(), cfg.max_grad_norm)
                    self.scaler.step(opt)

                self.scaler.update()
                self.optimizer.zero_grad()
                self.optimizer_b.zero_grad()

                if self.scheduler_is_per_batch:
                    self.scheduler.step()
                    self.scheduler_b.step()

            total_loss += loss.item() * cfg.gradient_accumulation_steps
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item() * cfg.gradient_accumulation_steps:.4f}",
                "R": f"{ct_result['remember_rate']:.2f}",
                "cla": f"{ct_result['clean_ratio_a']:.2f}",
                "clb": f"{ct_result['clean_ratio_b']:.2f}",
            })

        # Epoch-level scheduler step
        if not self.scheduler_is_per_batch:
            for sched in [self.scheduler, self.scheduler_b]:
                if cfg.lr_scheduler == "plateau":
                    pass  # Will be called by caller
                elif cfg.lr_scheduler != "onecycle":
                    sched.step()

        return total_loss / max(num_batches, 1)

    def _compute_main_loss(
        self,
        indices: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the main supervised loss (ELR/GCE/CE)."""
        # Try ELR-style (with indices)
        try:
            loss = self.criterion(indices, logits, labels)
        except TypeError:
            try:
                loss = self.criterion(logits, labels)
            except TypeError:
                loss = F.cross_entropy(logits, labels)
        return loss

    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 4.0,
    ) -> torch.Tensor:
        """KL divergence distillation loss."""
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        return F.kl_div(
            student_soft, teacher_soft, reduction="batchmean"
        ) * (temperature ** 2)

    def _save_checkpoint(self, filename: str, epoch: int, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_acc": self.best_acc,
            "best_epoch": self.best_epoch,
            "val_acc": val_acc,
            "config": self.config,
            "train_losses": self.train_losses,
            "val_accs": self.val_accs,
        }

        # Save Co-Teaching model B
        if self.model_b is not None:
            checkpoint["model_b_state_dict"] = self.model_b.state_dict()
            checkpoint["optimizer_b_state_dict"] = self.optimizer_b.state_dict()
            if self.scheduler_b:
                checkpoint["scheduler_b_state_dict"] = self.scheduler_b.state_dict()

        # Save EMA teacher
        if self.ema_teacher:
            checkpoint["ema_teacher"] = self.ema_teacher.state_dict()

        # Save ELR targets
        if hasattr(self.criterion, "elr") and self.criterion.elr is not None:
            checkpoint["elr_targets"] = self.criterion.elr.get_all_targets().cpu()

        # Save GMM state
        if self.sample_selector:
            checkpoint["loss_history"] = self.sample_selector.loss_history
            checkpoint["loss_count"] = self.sample_selector.loss_count

        # Save GRIP class soft labels
        if self.class_soft is not None:
            checkpoint["class_soft"] = self.class_soft.cpu()

        path = self.output_dir / filename
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")

    def _load_checkpoint(self, path: str):
        """Load model checkpoint and resume training."""
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"] + 1
        self.best_acc = checkpoint.get("best_acc", 0.0)
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_accs = checkpoint.get("val_accs", [])

        # Restore EMA teacher
        if "ema_teacher" in checkpoint and self.ema_teacher:
            self.ema_teacher.load_state_dict(checkpoint["ema_teacher"])

        # Restore ELR targets
        if "elr_targets" in checkpoint and hasattr(self.criterion, "elr"):
            if self.criterion.elr is not None:
                self.criterion.elr.target.copy_(
                    checkpoint["elr_targets"].to(
                        device=self.criterion.elr.target.device,
                        dtype=self.criterion.elr.target.dtype,
                    )
                )

        # Restore GMM state (keep on CPU for sklearn compatibility)
        if "loss_history" in checkpoint and self.sample_selector:
            self.sample_selector.loss_history = checkpoint["loss_history"].cpu()
            self.sample_selector.loss_count = checkpoint["loss_count"].cpu()

        # Restore GRIP class soft labels
        if "class_soft" in checkpoint and self.class_soft is not None:
            self.class_soft = checkpoint["class_soft"].to(self.class_soft.device)

        print(f"  Resumed from epoch {self.current_epoch}")

    def detect_noisy_labels(self, given_labels: torch.Tensor):
        """Run noise detection after training (ELR + GMM)."""
        results = {}

        if hasattr(self.criterion, "detect_noisy_labels"):
            noisy_mask = self.criterion.detect_noisy_labels(given_labels)
            num_noisy = noisy_mask.sum().item()
            print(f"\n=== Noise Detection (ELR argmax) ===")
            print(f"  Noisy: {num_noisy}/{len(given_labels)} "
                  f"({100 * num_noisy / len(given_labels):.2f}%)")
            results["elr_noisy_mask"] = noisy_mask

        if self.sample_selector:
            gmm_noisy = self.sample_selector.noisy_mask
            num_gmm_noisy = gmm_noisy.sum().item()
            print(f"\n=== Noise Detection (GMM) ===")
            print(f"  Noisy: {num_gmm_noisy}/{len(given_labels)} "
                  f"({100 * num_gmm_noisy / len(given_labels):.2f}%)")
            print(f"  GMM means: {self.sample_selector.gmm_means}")
            results["gmm_noisy_mask"] = gmm_noisy

        return results.get("elr_noisy_mask", None)
