import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import AutoImageProcessor

from model import create_vit_model, model_summary
from utils import (
    create_data_splits,
    detect_dataset_root,
    download_dataset,
    evaluate_and_report,
    plot_training_curves,
    save_label_map,
    set_seed,
)


def get_transforms(image_size: int = 224):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(
    data_root: Path, batch_size: int, num_workers: int, image_size: int = 224
) -> Tuple[Dict[str, DataLoader], List[str]]:
    train_tf, eval_tf = get_transforms(image_size=image_size)
    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_root / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_tf)

    class_names = train_ds.classes
    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }
    return loaders, class_names


def run_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    train_mode: bool = True,
) -> Tuple[float, float]:
    if train_mode:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    y_true, y_pred = [], []

    pbar = tqdm(dataloader, leave=False)
    with torch.set_grad_enabled(train_mode):
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

    avg_loss = epoch_loss / len(dataloader.dataset)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc


def evaluate_test_set(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating test", leave=False):
            images = images.to(device)
            outputs = model(pixel_values=images).logits
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())
    return y_true, y_pred


def export_onnx(
    model,
    output_path: Path,
    image_size: int = 224,
    device: str = "cpu",
) -> None:
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    class ViTWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(pixel_values=x).logits

    wrapped = ViTWrapper(model).to(device)
    torch.onnx.export(
        wrapped,
        dummy,
        str(output_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )


def quantize_onnx(onnx_input: Path, onnx_output: Path) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        print("onnxruntime quantization not available. Skipping ONNX quantization.")
        return

    quantize_dynamic(
        model_input=str(onnx_input),
        model_output=str(onnx_output),
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized ONNX model saved to: {onnx_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT for skin type detection")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dima806/skin-types-image-detection-vit",
        help="Kaggle dataset slug",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_data"
    split_dir = output_dir / "processed_data"
    artifacts_dir = output_dir / "artifacts"
    eval_dir = output_dir / "evaluation"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1/7: Downloading dataset...")
    download_dataset(args.dataset, raw_dir)
    source_root = detect_dataset_root(raw_dir)

    print("Step 2/7: Creating train/val/test split...")
    split_root, class_names = create_data_splits(
        source_root=source_root,
        output_root=split_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
    )
    save_label_map(class_names, artifacts_dir / "label_map.json")

    print("Step 3/7: Building dataloaders...")
    loaders, class_names = build_dataloaders(
        split_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    print("Step 4/7: Loading pretrained ViT...")
    _ = AutoImageProcessor.from_pretrained(args.model_name)
    model = create_vit_model(
        num_classes=len(class_names),
        model_name=args.model_name,
        class_names=class_names,
    )
    model_summary(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = artifacts_dir / "best_model.pth"

    print("Step 5/7: Training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(
            model, loaders["train"], criterion, optimizer, device, train_mode=True
        )
        val_loss, val_acc = run_epoch(
            model, loaders["val"], criterion, optimizer, device, train_mode=False
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "model_name": args.model_name,
                    "image_size": args.image_size,
                },
                best_model_path,
            )

    plot_training_curves(history, eval_dir / "training_curves.png")

    print("Step 6/7: Evaluating on test set...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred = evaluate_test_set(model, loaders["test"], device)
    test_acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    evaluate_and_report(y_true, y_pred, class_names, eval_dir)

    print("Step 7/7: Saving models and exporting ONNX...")
    final_model_path = artifacts_dir / "skin_vit_classifier_final.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "model_name": args.model_name,
            "image_size": args.image_size,
        },
        final_model_path,
    )
    print(f"PyTorch model saved to: {final_model_path}")

    onnx_path = artifacts_dir / "skin_vit_classifier.onnx"
    quant_onnx_path = artifacts_dir / "skin_vit_classifier_int8.onnx"
    export_onnx(model, onnx_path, image_size=args.image_size, device=device)
    print(f"ONNX model exported to: {onnx_path}")
    quantize_onnx(onnx_path, quant_onnx_path)

    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
