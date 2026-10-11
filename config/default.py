"""
Default configuration for CLIP + LoRA + ELR training on noisy-label fine-grained datasets.

All hyperparameters are documented with their effects.
Modify via command-line overrides or by creating dataset-specific configs.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    """CLIP ViT-B/32 backbone with LoRA adapters."""

    # --- CLIP backbone ---
    clip_model: str = "ViT-B-32"
    pretrained: str = "openai"  # openai | laion2b_s34b_b79k | or local path
    pretrained_local: str = "./data/ViT-B-32_openai.bin"  # Local weights (fallback if HF blocked)
    freeze_backbone: bool = True  # Freeze CLIP backbone weights
    image_size: int = 224

    # --- LoRA ---
    lora_r: int = 64  # LoRA rank (increased for fine-grained 750-class task)
    lora_alpha: int = 64  # LoRA scaling factor
    lora_dropout: float = 0.1
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "v_proj", "k_proj", "out_proj", "c_fc", "c_proj",
    )  # QKV + MLP layers in ViT for more capacity

    # --- Classification head ---
    num_classes: int = 400  # Override per dataset
    classifier_dropout: float = 0.2
    classifier_hidden_dim: int = 512  # Matches the CLIP feature dimension so the cosine head needs no projection and the zero-shot prior applies directly
    classifier_type: str = "cosine"  # cosine is more noise-robust
    zero_shot_init: bool = True  # Init classifier with CLIP text encoder (class names)
    freeze_prototypes: bool = False  # Freeze the cosine prototypes (only logit_scale is learned) so noise cannot corrupt the CLIP prior

    # --- Prompt Learning (CoOp-style) ---
    use_prompt_learner: bool = True  # Learnable text prompts for multimodal classification
    prompt_n_ctx: int = 16  # More context tokens for richer text representation
    prompt_init: str = "a photo of a"  # Initial context words
    prompt_fusion_weight: float = 0.2  # Weight for text branch logits (0-1), start conservative


@dataclass
class DataConfig:
    """Data loading and augmentation."""

    # --- Paths ---
    data_root: str = "./data"
    train_dir: str = "train"
    test_dir: str = "test"

    # --- Input ---
    image_size: int = 224
    mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073)  # CLIP stats
    std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711)

    # --- Augmentation ---
    use_randaug: bool = True
    randaug_n: int = 3  # More augmentation operations
    randaug_m: int = 15  # Higher magnitude for stronger regularization
    use_mixup: bool = True
    mixup_alpha: float = 0.5  # Slightly stronger mixing
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    label_smoothing: float = 0.1

    # --- Loading ---
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    val_split: float = 0.1  # Fraction of training data for validation
    use_balanced_sampler: bool = True  # Balanced sampling mitigates the long tail (the official test set is uniformly distributed)


@dataclass
class LossConfig:
    """Loss function configuration."""

    # --- ELR (Early-Learning Regularization) ---
    elr_lambda: float = 5.0  # Stronger regularization for noisy data
    elr_beta: float = 0.9  # Slower EMA for more stable target estimation

    # --- GCE (Generalized Cross-Entropy) ---
    gce_q: float = 0.7  # q < 1.0 for noise robustness; q=1.0 is standard CE

    # --- Combined loss ---
    loss_type: str = "elr+gce"  # Combine ELR and GCE for best noise robustness
    elr_weight: float = 1.0  # Weight for ELR term
    gce_weight: float = 0.3  # Weight for GCE term (lower to not dominate)

    # --- Double-Softmax (DSPT) and DeFT ---
    ds_weight: float = 1.0  # Weight of the double-softmax term (active when loss_type contains "ds")
    deft_weight: float = 0.0  # DeFT soft-label fusion weight (>0 enables it; corrects noisy labels with the clean CLIP zero-shot reference)
    deft_temperature: float = 1.0  # Temperature of the zero-shot soft labels
    deft_detect: bool = False  # DeFT noise detection: give sample-level trust from the zero-shot support of the hard label, strongly correcting only suspicious samples

    # --- GRIP class soft labels ---
    grip_enabled: bool = False  # Class soft-label supervision (GRIP idea: replace one-hot with a class-level soft distribution to reduce overfitting to noisy hard labels)
    grip_alpha: float = 0.9  # EMA update coefficient for the class soft labels (larger = slower)
    grip_weight: float = 0.3  # Loss weight for the class soft labels
    grip_warmup: int = 5  # Starting epoch for the class soft labels

    # --- SupCon (Supervised Contrastive Learning) ---
    use_supcon: bool = True  # Enable supervised contrastive loss for better features
    supcon_weight: float = 0.1  # Weight for contrastive loss
    supcon_temperature: float = 0.07  # Contrastive temperature
    supcon_clean_only: bool = True  # Only apply SupCon to GMM-clean samples

    # --- GMM Sample Selection (DivideMix-style) ---
    use_gmm_selection: bool = True  # Enable dynamic clean/noisy separation
    gmm_warmup_epochs: int = 10  # Epochs before starting GMM selection
    gmm_momentum: float = 0.9  # EMA momentum for loss history per sample
    gmm_p_threshold: float = 0.5  # Posterior threshold for clean classification

    # --- Label Co-Refinement ---
    use_label_refinement: bool = True  # Correct noisy labels with model predictions
    refine_alpha: float = 0.3  # Initial trust in model predictions (lower = safer)
    refine_max_alpha: float = 0.7  # Maximum trust in model predictions

    # --- Co-Teaching (dual-network mutual sample selection) ---
    use_co_teaching: bool = False  # Disabled: empirically hurts with ELR+cosine setup
    co_teach_forget_rate: float = 0.3
    co_teach_num_gradual: int = 10

    # --- Semi-Supervised Loop (DivideMix-style, pseudo-label regularization on noisy samples) ---
    semi_supervised: bool = True  # On the GMM noisy subset, use ELR target argmax pseudo-labels for self-correcting CE
    semi_weight: float = 0.3  # Pseudo-label regularization weight (additive; leaves the hard-label main loss unchanged)


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # --- Optimization ---
    optimizer: str = "adamw"  # adamw | sgd
    lr: float = 5e-4  # Higher LR for LoRA + classifier head
    lr_backbone_multiplier: float = 0.05  # Very low LR for frozen backbone
    lr_head_multiplier: float = 1.5  # Higher LR for classifier head
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.999)

    # --- Scheduler ---
    lr_scheduler: str = "onecycle"  # OneCycleLR for faster convergence
    warmup_epochs: int = 5
    warmup_lr: float = 1e-6
    min_lr: float = 1e-7

    # --- Training loop ---
    epochs: int = 40  # OneCycle converges fast
    batch_size: int = 64
    gradient_accumulation_steps: int = 2  # Effective batch size = 128
    max_grad_norm: float = 1.0

    # --- Mixed precision ---
    use_amp: bool = True  # Automatic Mixed Precision (float16)
    amp_dtype: str = "float16"

    # --- Knowledge distillation (teacher-free / self-distillation) ---
    use_self_distillation: bool = True
    distill_temperature: float = 3.0  # Slightly lower temperature
    distill_weight: float = 0.3  # Lower weight: let supervised loss dominate first
    ema_teacher_alpha: float = 0.999  # EMA momentum for teacher model
    ema_teacher_warmup: int = 200  # Longer warmup before distillation kicks in

    # --- Honest Validation ---
    honest_validation: bool = True  # Evaluate against denoised reference labels from the val EMA target, stripping out noisy-label contamination
    honest_ema_beta: float = 0.9  # Smoothing coefficient of the val EMA target (larger = slower, trusting earlier beliefs more)

    # --- Early stopping ---
    early_stopping_patience: int = 12  # ~30% of training, reasonable for noisy data
    early_stopping_metric: str = "val_acc"


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    batch_size: int = 128
    num_workers: int = 8
    use_tta: bool = True  # Test-time augmentation (multi-crop voting)
    tta_crops: int = 10  # Number of crops for TTA


@dataclass
class DefaultConfig:
    """Main configuration aggregating all sub-configs."""

    # --- Experiment ---
    exp_name: str = "clip_lora_elr"
    dataset: str = "webfg400"  # webfg400 | webinat5000
    seed: int = 42
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    resume_from: Optional[str] = None  # Path to checkpoint to resume

    # --- Sub-configs ---
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # --- Logging ---
    log_interval: int = 50  # Log every N steps
    eval_interval: int = 1  # Evaluate every N epochs
    save_interval: int = 10  # Save checkpoint every N epochs
    use_wandb: bool = False
    wandb_project: str = "clip-lora-elr-noise"


def get_config(dataset: str = "webfg400", **overrides) -> DefaultConfig:
    """
    Get configuration for a specific dataset with optional overrides.

    Args:
        dataset: "webfg400" or "webinat5000"
        **overrides: Any config field to override (key=value)

    Returns:
        DefaultConfig with dataset-specific settings applied
    """
    cfg = DefaultConfig()

    if dataset == "webfg400":
        cfg.dataset = "webfg400"
        cfg.model.num_classes = 750  # Actual data has 750 classes (semifinal round, leaderboard B)
        # 750 classes, ~149k train images, ~200 images/class
        cfg.train.epochs = 40  # Consistent with the 75.73% baseline; 60 epochs + onecycle previously broke the schedule and regressed by about 11 points
        cfg.train.batch_size = 64
        cfg.train.gradient_accumulation_steps = 2  # Effective batch = 128
        cfg.loss.elr_lambda = 5.0
        cfg.loss.elr_beta = 0.9
        cfg.train.lr = 5e-4
        cfg.train.lr_scheduler = "onecycle"
    elif dataset == "webinat5000":
        cfg.dataset = "webinat5000"
        cfg.model.num_classes = 5000
        # WebiNat-5000: 5000 classes, ~580k train images
        cfg.train.epochs = 80
        cfg.train.batch_size = 32  # More classes -> smaller batch
        cfg.loss.elr_lambda = 7.0  # Stronger regularization for more noise
        cfg.loss.elr_beta = 0.95  # Even slower EMA for 5000 classes
        cfg.train.gradient_accumulation_steps = 4  # Effective batch = 128
        cfg.train.lr = 3e-4
        cfg.train.lr_scheduler = "onecycle"
        cfg.model.classifier_hidden_dim = 1024
        cfg.model.lora_r = 64
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Apply overrides
    for key, value in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    return cfg
