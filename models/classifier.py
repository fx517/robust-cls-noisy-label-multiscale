"""
Classification head for CLIP fine-tuning on fine-grained datasets.

Supports both single-head (standard) and multi-head (per-dataset) classifiers.
Includes optional self-distillation target generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    Classification head that maps CLIP visual features to class logits.

    Supports two modes:
    - "fc": Linear -> LayerNorm -> GELU -> Dropout -> Linear (standard)
    - "cosine": Cosine similarity with learnable class prototypes (noise-robust)

    This lightweight head sits on top of the frozen (or LoRA-adapted)
    CLIP visual encoder and maps 512-dim features to class predictions.
    """

    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 400,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        use_layer_norm: bool = True,
        classifier_type: str = "fc",
        zero_shot_weight: torch.Tensor = None,
        freeze_prototypes: bool = False,
    ):
        super().__init__()

        self.classifier_type = classifier_type
        self.num_classes = num_classes

        if classifier_type == "cosine":
            # Cosine classifier: learnable class prototypes
            # When hidden_dim == in_features, do not project and perform cosine
            # classification directly in the CLIP feature space, so the zero-shot
            # CLIP text prototypes can be injected directly; otherwise a random
            # projection would break the semantic structure and invalidate the
            # class prior.
            use_projection = (hidden_dim > 0) and (hidden_dim != in_features)
            self.projection = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ) if use_projection else nn.Identity()

            self.projection_dim = hidden_dim if use_projection else in_features

            # Learnable class weight vectors (prototypes)
            self.weight = nn.Parameter(
                torch.empty(num_classes, self.projection_dim)
            )
            # Use zero-shot text features if provided, else orthogonal init
            if zero_shot_weight is not None:
                if zero_shot_weight.shape[-1] != self.projection_dim:
                    # Project to match hidden_dim if needed
                    self.weight.data = F.normalize(
                        nn.Linear(
                            zero_shot_weight.shape[-1], self.projection_dim,
                            bias=False
                        ).to(zero_shot_weight.device)(zero_shot_weight),
                        dim=-1
                    )
                else:
                    self.weight.data = F.normalize(zero_shot_weight, dim=-1)
                print(f"[Classifier] Initialized {num_classes} prototypes "
                      f"from CLIP text encoder (zero-shot)")
            else:
                nn.init.orthogonal_(self.weight)

            # Save zero-shot prototypes as a frozen anchor (for drift regularization)
            if zero_shot_weight is not None:
                self.register_buffer("zero_shot_anchor", self.weight.data.clone())

            # Learnable scale (like CLIP's logit scale)
            self.logit_scale = nn.Parameter(
                torch.ones([]) * 10.0
            )

        else:
            # Standard FC classifier
            layers = []
            if hidden_dim > 0:
                layers.extend([
                    nn.Linear(in_features, hidden_dim),
                    nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes),
                ])
            else:
                layers.append(nn.Linear(in_features, num_classes))

            self.fc = nn.Sequential(*layers)

        # Initialize final layer with small weights for stable training
        self._init_weights()

        # Optionally freeze prototypes so the CLIP text prior stays intact
        if freeze_prototypes and classifier_type == "cosine":
            self.weight.requires_grad_(False)
            print("[Classifier] Prototypes frozen (only logit_scale learnable)")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor (B, in_features), assumed already normalized

        Returns:
            Logits tensor (B, num_classes)
        """
        if self.classifier_type == "cosine":
            # Project then normalize
            x = self.projection(x)
            x = F.normalize(x, dim=-1)
            w = F.normalize(self.weight, dim=-1)

            # Cosine similarity * learnable scale
            logits = x @ w.T * self.logit_scale
            return logits
        else:
            return self.fc(x)


class MultiHeadClassifier(nn.Module):
    """
    Multi-head classifier for multi-dataset training (optional).

    Each dataset gets its own classification head, sharing the same
    backbone features. This is useful for training on multiple
    fine-grained datasets simultaneously.
    """

    def __init__(
        self,
        in_features: int = 512,
        num_classes_list: list = None,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        if num_classes_list is None:
            num_classes_list = [400, 5000]

        self.heads = nn.ModuleDict()
        for i, num_classes in enumerate(num_classes_list):
            self.heads[f"dataset_{i}"] = ClassificationHead(
                in_features=in_features,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor, head_idx: int = 0) -> torch.Tensor:
        """Forward through a specific classification head."""
        return self.heads[f"dataset_{head_idx}"](x)
