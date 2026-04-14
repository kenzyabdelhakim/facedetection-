import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SKIN_CLASSES = ["dry", "normal", "oily"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_dataset(dataset_slug: str, download_dir: Path) -> Path:
    """
    Try downloading the Kaggle dataset via opendatasets (interactive credentials)
    or the kaggle CLI. Falls back to generating a synthetic demo dataset.
    """
    ensure_dir(download_dir)

    # --- Attempt 1: opendatasets (prompts for user/key if no kaggle.json) ---
    try:
        import opendatasets as od

        kaggle_url = f"https://www.kaggle.com/datasets/{dataset_slug}"
        od.download(kaggle_url, data_dir=str(download_dir))
        print("Dataset downloaded via opendatasets.")
        return download_dir
    except BaseException as e:
        print(f"opendatasets download skipped ({e}). Trying kaggle API...")

    # --- Attempt 2: kaggle API (needs kaggle.json) ---
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_slug, path=str(download_dir), unzip=True, quiet=False)
        print("Dataset downloaded via Kaggle API.")
        return download_dir
    except BaseException as e:
        print(f"Kaggle API download skipped ({e}).")

    # --- Attempt 3: generate synthetic demo data so the pipeline can run ---
    print("Generating synthetic demo dataset (150 images per class)...")
    _generate_synthetic_skin_dataset(download_dir, classes=SKIN_CLASSES, images_per_class=150)
    return download_dir


def _generate_synthetic_skin_dataset(
    root: Path, classes: List[str], images_per_class: int = 150, size: int = 224
) -> None:
    """
    Create realistic-looking synthetic face-skin images so the full pipeline
    can be demonstrated end-to-end without any external dataset.
    Each class gets a slightly different colour palette and texture.
    """
    rng = random.Random(42)

    palette = {
        "dry":    {"base": (210, 180, 160), "var": 30, "noise": 25},
        "normal": {"base": (220, 195, 170), "var": 15, "noise": 10},
        "oily":   {"base": (230, 200, 175), "var": 10, "noise": 8},
    }

    for cls in classes:
        cls_dir = root / cls
        ensure_dir(cls_dir)
        p = palette.get(cls, palette["normal"])

        for i in range(images_per_class):
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            for c in range(3):
                base = p["base"][c] + rng.randint(-p["var"], p["var"])
                channel = np.full((size, size), base, dtype=np.float32)
                noise = np.random.normal(0, p["noise"], (size, size))
                channel = np.clip(channel + noise, 0, 255)
                arr[:, :, c] = channel.astype(np.uint8)

            img = Image.fromarray(arr, "RGB")

            if cls == "oily":
                draw = ImageDraw.Draw(img)
                for _ in range(rng.randint(3, 8)):
                    cx, cy = rng.randint(20, size - 20), rng.randint(20, size - 20)
                    r = rng.randint(10, 30)
                    bright = tuple(min(255, v + 40) for v in p["base"])
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=bright)
                img = img.filter(ImageFilter.GaussianBlur(radius=2))
            elif cls == "dry":
                draw = ImageDraw.Draw(img)
                for _ in range(rng.randint(8, 20)):
                    x0, y0 = rng.randint(0, size), rng.randint(0, size)
                    x1 = x0 + rng.randint(10, 50)
                    y1 = y0 + rng.randint(-2, 2)
                    dark = tuple(max(0, v - 35) for v in p["base"])
                    draw.line([(x0, y0), (x1, y1)], fill=dark, width=1)
            else:
                img = img.filter(ImageFilter.GaussianBlur(radius=1))

            img.save(cls_dir / f"{cls}_{i:04d}.jpg", "JPEG", quality=90)


def _collect_images_by_class(root_dir: Path) -> Dict[str, List[Path]]:
    class_to_files: Dict[str, List[Path]] = {}

    for class_dir in sorted([d for d in root_dir.iterdir() if d.is_dir()]):
        files = [
            p
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if files:
            class_to_files[class_dir.name] = files

    return class_to_files


def detect_dataset_root(download_dir: Path) -> Path:
    """
    Detect an ImageFolder-like root (each class in a subfolder).
    """
    direct = _collect_images_by_class(download_dir)
    if direct:
        return download_dir

    for sub in sorted([d for d in download_dir.iterdir() if d.is_dir()]):
        if _collect_images_by_class(sub):
            return sub

    raise FileNotFoundError(
        f"Could not find class folders with images inside: {download_dir}"
    )


def create_data_splits(
    source_root: Path,
    output_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Path, List[str]]:
    """
    Copy images into ImageFolder split structure:
    output_root/train/<class>, output_root/val/<class>, output_root/test/<class>.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(seed)
    class_to_files = _collect_images_by_class(source_root)
    if not class_to_files:
        raise ValueError(f"No class folders with images found in {source_root}")

    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ["train", "val", "test"]:
        ensure_dir(output_root / split)

    class_names = sorted(class_to_files.keys())

    for class_name in class_names:
        files = class_to_files[class_name]
        rng.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        split_map = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val : n_train + n_val + n_test],
        }

        for split_name, split_files in split_map.items():
            split_class_dir = output_root / split_name / class_name
            ensure_dir(split_class_dir)
            for file_path in split_files:
                shutil.copy2(file_path, split_class_dir / file_path.name)

    return output_root, class_names


def plot_training_curves(history: Dict[str, List[float]], save_path: Path) -> None:
    ensure_dir(save_path.parent)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def evaluate_and_report(
    true_labels: List[int],
    pred_labels: List[int],
    class_names: List[str],
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)
    cm = confusion_matrix(true_labels, pred_labels)
    report = classification_report(
        true_labels, pred_labels, target_names=class_names, digits=4
    )

    print("\nClassification Report:")
    print(report)

    with (output_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report)

    plt.figure(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()


def save_label_map(class_names: List[str], path: Path) -> None:
    data = {idx: name for idx, name in enumerate(class_names)}
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")
