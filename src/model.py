from typing import Dict, Tuple

import torch
from transformers import ViTForImageClassification


def create_vit_model(
    num_classes: int,
    model_name: str = "google/vit-base-patch16-224",
    class_names=None,
) -> ViTForImageClassification:
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    id2label: Dict[int, str] = {idx: name for idx, name in enumerate(class_names)}
    label2id: Dict[str, int] = {name: idx for idx, name in enumerate(class_names)}

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model


def model_summary(model: torch.nn.Module) -> Tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel Summary")
    print("-" * 80)
    print(model)
    print("-" * 80)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params
