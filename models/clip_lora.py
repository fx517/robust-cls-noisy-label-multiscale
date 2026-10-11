"""
CLIP + LoRA model for fine-grained image recognition.

Combines:
1. Frozen CLIP ViT-B/32 visual encoder with LoRA adapters
2. Trainable classification head (FC or cosine-similarity based)
3. (Optional) Learnable text prompts (CoOp-style) for text-guided classification
4. Integration with EMA teacher for self-distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("Warning: open_clip not installed. Install via: pip install open-clip-torch")

from .lora import apply_lora_to_clip, count_lora_params
from .classifier import ClassificationHead
from .prompt_learner import UnifiedPromptLearner


class CLIPLoRAModel(nn.Module):
    """
    CLIP ViT-B/32 with LoRA adapters for noisy-label fine-grained recognition.

    Supports:
    - LoRA-adapted visual encoder (frozen backbone, only LoRA matrices trainable)
    - FC or cosine-similarity classifier (cosine is more noise-robust)
    - CoOp-style learnable text prompt learner (optional, fused with visual branch)
    - EMA teacher support via get_visual_features()

    Args:
        num_classes: Number of target classes
        clip_model: CLIP model variant (default: ViT-B-32)
        pretrained: Pretrained weights source
        lora_r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: LoRA dropout rate
        classifier_hidden_dim: Hidden dimension in classification head
        classifier_dropout: Dropout in classification head
        classifier_type: "fc" (standard) or "cosine" (noise-robust cosine prototypes)
        freeze_backbone: Freeze CLIP backbone weights
        use_prompt_learner: Enable CoOp-style learnable text prompts
        prompt_n_ctx: Number of learnable context tokens for prompts
        prompt_fusion_weight: Initial fusion weight for text branch (0=visual only, 1=text only)
    """

    def __init__(
        self,
        num_classes: int = 400,
        image_size: int = 224,
        clip_model: str = "ViT-B-32",
        pretrained: str = "openai",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        classifier_hidden_dim: int = 512,
        classifier_dropout: float = 0.3,
        classifier_type: str = "fc",
        freeze_backbone: bool = True,
        use_prompt_learner: bool = False,
        prompt_n_ctx: int = 8,
        prompt_fusion_weight: float = 0.3,
        zero_shot_init: bool = False,
        class_names: list = None,
        freeze_prototypes: bool = False,
    ):
        super().__init__()

        if not OPEN_CLIP_AVAILABLE:
            raise ImportError(
                "open_clip is required. Install: pip install open-clip-torch"
            )

        self.num_classes = num_classes
        self.use_prompt_learner = use_prompt_learner
        self.prompt_fusion_weight = prompt_fusion_weight
        self.zero_shot_init = zero_shot_init

        # Load CLIP model — try local weights first if available
        import os as _os
        local_path = getattr(self, '_local_pretrained_path', None)
        if local_path is None and pretrained == "openai":
            # Try common local paths
            candidates = [
                "./data/ViT-B-32_openai.bin",
                "data/ViT-B-32_openai.bin",
            ]
            for cand in candidates:
                if _os.path.exists(cand):
                    local_path = cand
                    break

        if local_path and _os.path.exists(local_path):
            print(f"[CLIPLoRA] Loading pretrained weights from local: {local_path}")
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_model, pretrained=local_path
            )
        else:
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_model, pretrained=pretrained
            )
        self.visual = self.clip_model.visual

        # FixRes: interpolate positional_embedding when target resolution != 224
        self._interpolate_pos_encoding(image_size)

        # Feature dimension from CLIP ViT
        self.feature_dim = self.visual.output_dim  # 512 for ViT-B/32

        # Apply LoRA to the visual encoder
        if lora_r > 0:
            apply_lora_to_clip(
                self.visual,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        # Freeze backbone if requested
        if freeze_backbone:
            for name, param in self.visual.named_parameters():
                if "lora_A" not in name and "lora_B" not in name:
                    param.requires_grad_(False)

            # Also freeze the CLIP text encoder (used only for zero-shot init)
            for param in self.clip_model.parameters():
                param.requires_grad_(False)
            # Unfreeze visual LoRA params (they were frozen by the loop above)
            for name, param in self.visual.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    param.requires_grad_(True)

        # --- Zero-shot init: encode class names with CLIP text encoder ---
        text_prototypes = None
        if zero_shot_init and class_names is not None and classifier_type == "cosine":
            text_prototypes = self._encode_class_names(class_names)
            print(f"[CLIPLoRA] Zero-shot init: encoded {len(class_names)} class names")

        # Classification head
        self.classifier = ClassificationHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=classifier_dropout,
            classifier_type=classifier_type,
            zero_shot_weight=text_prototypes,
            freeze_prototypes=freeze_prototypes,
        )

        # Prompt learner (CoOp-style) for text-guided classification
        if use_prompt_learner:
            self.prompt_learner = UnifiedPromptLearner(
                num_classes=num_classes,
                feature_dim=self.feature_dim,
                n_ctx=prompt_n_ctx,
                use_prompts=True,
            )
            self.fusion_weight = nn.Parameter(
                torch.tensor(prompt_fusion_weight)
            )
        else:
            self.prompt_learner = None
            self.fusion_weight = None

        # Print model stats
        trainable, total = count_lora_params(self)
        total_pct = 100 * trainable / total if total > 0 else 0
        print(
            f"[CLIPLoRA] Total params: {total:,} | Trainable: {trainable:,} "
            f"({total_pct:.2f}%)"
        )
        if zero_shot_init:
            print(f"[CLIPLoRA] Zero-shot classifier init enabled")
        if use_prompt_learner:
            print(f"[CLIPLoRA] Prompt learner: n_ctx={prompt_n_ctx}")
        if classifier_type == "cosine":
            print("[CLIPLoRA] Cosine classifier enabled")

    def _interpolate_pos_encoding(self, image_size: int):
        """
        FixRes-style interpolation of the CLIP positional embedding.

        The pretrained ViT-B/32 positional_embedding has shape (50, 768),
        matching 224x224 input (7x7=49 patches + 1 CLS). When training at a
        higher resolution (e.g. 384), we bicubically interpolate the 7x7 grid
        of patch positions to the new grid (12x12=144), keeping the CLS token
        position unchanged.
        """
        visual = self.visual
        if not hasattr(visual, "positional_embedding"):
            return
        pos_embed = visual.positional_embedding.data
        if pos_embed.dim() != 2 or pos_embed.shape[0] <= 1:
            return

        num_patches = pos_embed.shape[0] - 1
        patch_size = visual.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        old_grid = int(round(num_patches ** 0.5))
        new_grid = image_size // patch_size
        if old_grid == new_grid:
            return

        dim = pos_embed.shape[-1]
        cls_pos = pos_embed[0:1]  # (1, dim) CLS token position
        patch_pos = pos_embed[1:]  # (old_grid^2, dim)

        # Reshape to (1, dim, old_grid, old_grid) and bicubically interpolate
        patch_pos = (
            patch_pos.reshape(old_grid, old_grid, dim)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        patch_pos = F.interpolate(
            patch_pos,
            size=(new_grid, new_grid),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = (
            patch_pos.squeeze(0)
            .permute(1, 2, 0)
            .reshape(new_grid * new_grid, dim)
        )

        new_pos_embed = torch.cat([cls_pos, patch_pos], dim=0)  # (new_grid^2+1, dim)
        visual.positional_embedding = nn.Parameter(new_pos_embed)
        print(
            f"[CLIPLoRA] Interpolated positional_embedding: "
            f"{num_patches} -> {new_grid * new_grid} patches "
            f"(grid {old_grid}x{old_grid} -> {new_grid}x{new_grid})"
        )

    def _encode_class_names(self, class_names: list) -> torch.Tensor:
        """Encode class folder names to text features for zero-shot init."""
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        device = next(self.visual.parameters()).device

        # Build prompts: "a photo of a {class_name}"
        prompts = [f"a photo of a {name}." for name in class_names]

        text_features_list = []
        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                tokens = tokenizer(batch_prompts).to(device)
                feats = self.clip_model.encode_text(tokens)
                feats = nn.functional.normalize(feats, dim=-1)
                text_features_list.append(feats.cpu())

        text_features = torch.cat(text_features_list, dim=0)  # (C, 512)
        return text_features

    def forward(
        self, images: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through visual encoder and classifier.

        Args:
            images: (B, 3, H, W) input images
            return_features: If True, also return visual features

        Returns:
            logits: (B, num_classes) or (features, logits) if return_features
        """
        # Extract normalized visual features
        features = self.get_visual_features(images)

        # Visual classification
        logits_visual = self.classifier(features)

        # Optionally fuse with text branch (prompt learner)
        if self.use_prompt_learner and self.prompt_learner is not None:
            text_prototypes = self.prompt_learner.forward()  # (C, D)
            # Cosine similarity with text prototypes
            logits_text = features @ text_prototypes.T * 10.0

            # Learnable fusion weight (sigmoid-bounded)
            w = torch.sigmoid(self.fusion_weight)
            logits = (1 - w) * logits_visual + w * logits_text
        else:
            logits = logits_visual

        if return_features:
            return features, logits
        return logits

    def get_visual_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract normalized visual features from CLIP visual encoder.

        Used by EMA teacher and contrastive loss for feature extraction.
        """
        features = self.visual(images)
        if isinstance(features, tuple):
            features = features[0]
        return F.normalize(features, dim=-1)

    def get_text_prototypes(self) -> torch.Tensor:
        """
        Get text-based class prototypes from prompt learner.

        Returns:
            (num_classes, feature_dim) normalized prototypes or None
        """
        if self.prompt_learner is not None:
            return self.prompt_learner.forward()
        return None

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Inference mode: return class predictions."""
        self.eval()
        logits = self.forward(images)
        return logits.argmax(dim=-1)
