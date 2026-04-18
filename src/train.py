import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import MultiTaskViT, create_multitask_vit, model_summary, SKIN_TYPES, SKIN_ISSUES
from utils import (
    SkinMultiTaskDataset,
    create_data_splits,
    detect_dataset_root,
    download_dataset,
    evaluate_skin_issues,
    evaluate_skin_type,
    generate_issue_annotations,
    load_annotations,
    plot_training_curves,
    save_annotations,
    save_label_map,
    set_seed,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Transforms
# ═══════════════════════════════════════════════════════════════════════════
def get_transforms(image_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return train_tf, eval_tf


# ═══════════════════════════════════════════════════════════════════════════
#  DataLoader builder
# ═══════════════════════════════════════════════════════════════════════════
def build_dataloaders(
    split_root: Path,
    annotations: Dict[str, Dict],
    skin_types: List[str],
    skin_issues: List[str],
    batch_size: int,
    num_workers: int,
    image_size: int = 224,
) -> Dict[str, DataLoader]:
    train_tf, eval_tf = get_transforms(image_size)
    loaders = {}
    for split, tf in [("train", train_tf), ("val", eval_tf), ("test", eval_tf)]:
        ds = SkinMultiTaskDataset(
            split_dir=split_root / split,
            annotations=annotations,
            split_root=split_root,
            skin_types=skin_types,
            skin_issues=skin_issues,
            transform=tf,
        )
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
    return loaders


# ═══════════════════════════════════════════════════════════════════════════
#  Training / evaluation loops
# ═══════════════════════════════════════════════════════════════════════════
def run_epoch(
    model: MultiTaskViT,
    loader: DataLoader,
    type_criterion: nn.Module,
    issue_criterion: nn.Module,
    optimizer,
    device: str,
    train_mode: bool = True,
    issue_weight: float = 1.0,
) -> Dict[str, float]:
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    type_true, type_pred = [], []
    issue_true, issue_scores = [], []

    with torch.set_grad_enabled(train_mode):
        for images, type_labels, issue_labels in tqdm(loader, leave=False):
            images = images.to(device)
            type_labels = type_labels.to(device)
            issue_labels = issue_labels.to(device)

            type_logits, issue_logits = model(images)

            loss_type = type_criterion(type_logits, type_labels)
            loss_issue = issue_criterion(issue_logits, issue_labels)
            loss = loss_type + issue_weight * loss_issue

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            type_true.extend(type_labels.cpu().tolist())
            type_pred.extend(type_logits.argmax(dim=1).cpu().tolist())
            issue_true.append(issue_labels.detach().cpu().numpy())
            issue_scores.append(torch.sigmoid(issue_logits).detach().cpu().numpy())

    n = len(loader.dataset)
    issue_true_arr = np.vstack(issue_true)
    issue_scores_arr = np.vstack(issue_scores)
    issue_pred_bin = (issue_scores_arr >= 0.5).astype(int)

    return {
        "loss": total_loss / n,
        "type_acc": accuracy_score(type_true, type_pred),
        "issue_f1": f1_score(issue_true_arr, issue_pred_bin, average="macro", zero_division=0),
        "type_true": type_true,
        "type_pred": type_pred,
        "issue_true": issue_true_arr,
        "issue_scores": issue_scores_arr,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  ONNX export
# ═══════════════════════════════════════════════════════════════════════════
def export_onnx(model: MultiTaskViT, path: Path, image_size: int = 224, device: str = "cpu"):
    model.eval()

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            t, i = self.m(x)
            return t, i

    wrapped = Wrapper(model).to(device)
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    torch.onnx.export(
        wrapped, dummy, str(path),
        input_names=["pixel_values"],
        output_names=["type_logits", "issue_logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "type_logits": {0: "batch"},
            "issue_logits": {0: "batch"},
        },
        opset_version=18,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Train multi-task ViT for skin analysis")
    p.add_argument("--dataset", default="dima806/skin-types-image-detection-vit")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--issue_weight", type=float, default=1.0,
                   help="Weight for the skin-issue BCE loss relative to skin-type CE loss")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--model_name", default="google/vit-base-patch16-224")
    p.add_argument("--output_dir", default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_data"
    split_dir = output_dir / "processed_data"
    artifacts = output_dir / "artifacts"
    eval_dir = output_dir / "evaluation"
    artifacts.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    skin_types = SKIN_TYPES
    skin_issues = SKIN_ISSUES

    # ── Step 1: Download ─────────────────────────────────────────────
    print("Step 1/7: Downloading dataset...")
    download_dataset(args.dataset, raw_dir)
    source_root = detect_dataset_root(raw_dir)

    # ── Step 2: Split ────────────────────────────────────────────────
    print("Step 2/7: Creating train/val/test split...")
    split_root, class_names = create_data_splits(
        source_root, split_dir, seed=args.seed,
    )
    save_label_map(skin_types, skin_issues, artifacts / "label_map.json")

    # ── Step 3: Annotations ──────────────────────────────────────────
    print("Step 3/7: Generating multi-label annotations...")
    ann_path = artifacts / "annotations.json"
    annotations = generate_issue_annotations(split_root, skin_types, skin_issues, seed=args.seed)
    save_annotations(annotations, ann_path)
    print(f"  {len(annotations)} samples annotated ({len(skin_issues)} issue labels each)")

    # ── Step 4: DataLoaders ──────────────────────────────────────────
    print("Step 4/7: Building dataloaders...")
    loaders = build_dataloaders(
        split_root, annotations, skin_types, skin_issues,
        batch_size=args.batch_size, num_workers=args.num_workers, image_size=args.image_size,
    )

    # ── Step 5: Model ────────────────────────────────────────────────
    print("Step 5/7: Loading pretrained multi-task ViT...")
    model = create_multitask_vit(args.model_name, skin_types, skin_issues)
    model_summary(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    type_criterion = nn.CrossEntropyLoss()
    issue_criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {k: [] for k in [
        "train_loss", "val_loss",
        "train_type_acc", "val_type_acc",
        "train_issue_f1", "val_issue_f1",
    ]}
    best_score = 0.0
    best_path = artifacts / "best_model.pth"

    # ── Step 6: Training ─────────────────────────────────────────────
    print("Step 6/7: Training...")
    for epoch in range(args.epochs):
        train = run_epoch(model, loaders["train"], type_criterion, issue_criterion,
                          optimizer, device, train_mode=True, issue_weight=args.issue_weight)
        val = run_epoch(model, loaders["val"], type_criterion, issue_criterion,
                        optimizer, device, train_mode=False, issue_weight=args.issue_weight)
        scheduler.step()

        history["train_loss"].append(train["loss"])
        history["val_loss"].append(val["loss"])
        history["train_type_acc"].append(train["type_acc"])
        history["val_type_acc"].append(val["type_acc"])
        history["train_issue_f1"].append(train["issue_f1"])
        history["val_issue_f1"].append(val["issue_f1"])

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Loss: {train['loss']:.4f}/{val['loss']:.4f} | "
            f"Type Acc: {train['type_acc']:.4f}/{val['type_acc']:.4f} | "
            f"Issue F1: {train['issue_f1']:.4f}/{val['issue_f1']:.4f}"
        )

        combined = val["type_acc"] + val["issue_f1"]
        if combined > best_score:
            best_score = combined
            torch.save({
                "model_state_dict": model.state_dict(),
                "skin_types": skin_types,
                "skin_issues": skin_issues,
                "model_name": args.model_name,
                "image_size": args.image_size,
                "vit_config": model.backbone.config.to_dict(),
            }, best_path)

    plot_training_curves(history, eval_dir / "training_curves.png")

    # ── Step 7: Evaluation ───────────────────────────────────────────
    print("Step 7/7: Evaluating on test set...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test = run_epoch(model, loaders["test"], type_criterion, issue_criterion,
                     optimizer, device, train_mode=False)

    print(f"Test Type Acc: {test['type_acc']:.4f}  |  Test Issue F1: {test['issue_f1']:.4f}")
    evaluate_skin_type(test["type_true"], test["type_pred"], skin_types, eval_dir)
    evaluate_skin_issues(test["issue_true"], test["issue_scores"], skin_issues, eval_dir)

    # ── Save final model ─────────────────────────────────────────────
    final_path = artifacts / "skin_multitask_vit_final.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "skin_types": skin_types,
        "skin_issues": skin_issues,
        "model_name": args.model_name,
        "image_size": args.image_size,
        "vit_config": model.backbone.config.to_dict(),
    }, final_path)
    print(f"Model saved: {final_path}")

    # ── ONNX export ──────────────────────────────────────────────────
    onnx_path = artifacts / "skin_multitask_vit.onnx"
    try:
        export_onnx(model, onnx_path, image_size=args.image_size, device=device)
        print(f"ONNX exported: {onnx_path}")
    except Exception as e:
        print(f"ONNX export skipped: {e}")

    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
