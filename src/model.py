from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import ViTModel


SKIN_TYPES = ["dry", "normal", "oily"]
SKIN_ISSUES = ["acne", "dark_spots", "wrinkles", "redness", "large_pores"]


class MultiTaskViT(nn.Module):
    """
    Multi-task Vision Transformer:
      Head 1 – skin type   (softmax over 3 classes)
      Head 2 – skin issues (sigmoid over 5 binary labels)
    Both heads share a single ViT backbone.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        skin_types: List[str] = None,
        skin_issues: List[str] = None,
    ):
        super().__init__()
        self.skin_types = skin_types or SKIN_TYPES
        self.skin_issues = skin_issues or SKIN_ISSUES

        self.backbone = ViTModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size  # 768 for base

        self.type_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, len(self.skin_types)),
        )

        self.issue_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, len(self.skin_issues)),
        )

    def forward(self, pixel_values: torch.Tensor):
        cls_token = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
        type_logits = self.type_head(cls_token)
        issue_logits = self.issue_head(cls_token)
        return type_logits, issue_logits


def create_multitask_vit(
    model_name: str = "google/vit-base-patch16-224",
    skin_types: List[str] = None,
    skin_issues: List[str] = None,
) -> MultiTaskViT:
    return MultiTaskViT(
        model_name=model_name,
        skin_types=skin_types,
        skin_issues=skin_issues,
    )


def model_summary(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel Summary")
    print("-" * 80)
    print(model)
    print("-" * 80)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable
