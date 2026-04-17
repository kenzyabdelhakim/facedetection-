import csv
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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix,
)
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SKIN_TYPES = ["dry", "normal", "oily"]
SKIN_ISSUES = ["acne", "dark_spots", "wrinkles", "redness", "large_pores"]

# Probabilistic rules: for each skin type, the base probability of each issue.
# These are used when generating synthetic multi-label annotations.
ISSUE_PROBS_BY_TYPE = {
    "dry":    {"acne": 0.15, "dark_spots": 0.30, "wrinkles": 0.55, "redness": 0.45, "large_pores": 0.10},
    "normal": {"acne": 0.10, "dark_spots": 0.15, "wrinkles": 0.15, "redness": 0.10, "large_pores": 0.10},
    "oily":   {"acne": 0.60, "dark_spots": 0.25, "wrinkles": 0.10, "redness": 0.30, "large_pores": 0.55},
}


# ═══════════════════════════════════════════════════════════════════════════
#  Seed & filesystem helpers
# ═══════════════════════════════════════════════════════════════════════════
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset download (Kaggle → synthetic fallback)
# ═══════════════════════════════════════════════════════════════════════════
def download_dataset(dataset_slug: str, download_dir: Path) -> Path:
    ensure_dir(download_dir)

    try:
        import opendatasets as od
        od.download(f"https://www.kaggle.com/datasets/{dataset_slug}", data_dir=str(download_dir))
        print("Dataset downloaded via opendatasets.")
        return download_dir
    except BaseException as e:
        print(f"opendatasets download skipped ({e}). Trying kaggle API...")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_slug, path=str(download_dir), unzip=True, quiet=False)
        print("Dataset downloaded via Kaggle API.")
        return download_dir
    except BaseException as e:
        print(f"Kaggle API download skipped ({e}).")

    print("Generating synthetic demo dataset (150 images per class)...")
    _generate_synthetic_skin_dataset(download_dir, classes=SKIN_TYPES, images_per_class=150)
    return download_dir


# ═══════════════════════════════════════════════════════════════════════════
#  Synthetic image generation
# ═══════════════════════════════════════════════════════════════════════════
def _generate_synthetic_skin_dataset(
    root: Path, classes: List[str], images_per_class: int = 150, size: int = 224
) -> None:
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


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset discovery & splitting
# ═══════════════════════════════════════════════════════════════════════════
def _collect_images_by_class(root_dir: Path) -> Dict[str, List[Path]]:
    class_to_files: Dict[str, List[Path]] = {}
    for class_dir in sorted([d for d in root_dir.iterdir() if d.is_dir()]):
        files = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if files:
            class_to_files[class_dir.name] = files
    return class_to_files


def detect_dataset_root(download_dir: Path) -> Path:
    direct = _collect_images_by_class(download_dir)
    if direct:
        return download_dir
    for sub in sorted([d for d in download_dir.iterdir() if d.is_dir()]):
        if _collect_images_by_class(sub):
            return sub
    raise FileNotFoundError(f"Could not find class folders with images inside: {download_dir}")


def create_data_splits(
    source_root: Path,
    output_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Path, List[str]]:
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

        split_map = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split_name, split_files in split_map.items():
            split_class_dir = output_root / split_name / class_name
            ensure_dir(split_class_dir)
            for fp in split_files:
                shutil.copy2(fp, split_class_dir / fp.name)

    return output_root, class_names


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-label annotation generation
# ═══════════════════════════════════════════════════════════════════════════
def generate_issue_annotations(
    split_root: Path,
    skin_types: List[str],
    skin_issues: List[str],
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    For each image in the split, produce a dict:
        { "relative/path.jpg": {"skin_type": "oily", "issues": [0,1,0,0,1]} }

    If a real annotations CSV exists (annotations.csv next to split_root),
    it is loaded. Otherwise labels are *simulated* based on skin-type priors.

    CSV format (for real datasets):
        filename, skin_type, acne, dark_spots, wrinkles, redness, large_pores
        img001.jpg, oily, 1, 0, 0, 1, 1
    """
    csv_path = split_root.parent / "annotations.csv"
    if csv_path.exists():
        return _load_annotations_csv(csv_path, skin_issues)

    rng = random.Random(seed)
    annotations: Dict[str, Dict] = {}

    for split in ["train", "val", "test"]:
        split_dir = split_root / split
        if not split_dir.exists():
            continue
        for type_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
            stype = type_dir.name
            probs = ISSUE_PROBS_BY_TYPE.get(stype, ISSUE_PROBS_BY_TYPE["normal"])
            for img_path in sorted(type_dir.rglob("*")):
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                rel = img_path.relative_to(split_root).as_posix()
                issues = [1 if rng.random() < probs[iss] else 0 for iss in skin_issues]
                annotations[rel] = {"skin_type": stype, "issues": issues}

    return annotations


def _load_annotations_csv(csv_path: Path, skin_issues: List[str]) -> Dict[str, Dict]:
    annotations = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            stype = row["skin_type"]
            issues = [int(row.get(iss, 0)) for iss in skin_issues]
            annotations[fname] = {"skin_type": stype, "issues": issues}
    return annotations


def save_annotations(annotations: Dict[str, Dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)


def load_annotations(path: Path) -> Dict[str, Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
#  PyTorch Dataset for multi-task learning
# ═══════════════════════════════════════════════════════════════════════════
class SkinMultiTaskDataset(Dataset):
    """
    Returns (image_tensor, type_label_int, issue_label_vector) per sample.
    """

    def __init__(
        self,
        split_dir: Path,
        annotations: Dict[str, Dict],
        split_root: Path,
        skin_types: List[str],
        skin_issues: List[str],
        transform=None,
    ):
        self.skin_types = skin_types
        self.skin_issues = skin_issues
        self.transform = transform
        self.type2idx = {t: i for i, t in enumerate(skin_types)}

        self.samples: List[Tuple[Path, int, List[int]]] = []
        for img_path in sorted(split_dir.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel = img_path.relative_to(split_root).as_posix()
            ann = annotations.get(rel)
            if ann is None:
                continue
            type_idx = self.type2idx[ann["skin_type"]]
            self.samples.append((img_path, type_idx, ann["issues"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, type_idx, issues = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, type_idx, torch.tensor(issues, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  Plotting & evaluation
# ═══════════════════════════════════════════════════════════════════════════
def plot_training_curves(history: Dict[str, List[float]], save_path: Path) -> None:
    ensure_dir(save_path.parent)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_type_acc"], label="Train")
    axes[1].plot(epochs, history["val_type_acc"], label="Val")
    axes[1].set_title("Skin Type Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, history["train_issue_f1"], label="Train")
    axes[2].plot(epochs, history["val_issue_f1"], label="Val")
    axes[2].set_title("Skin Issues F1 (macro)")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def evaluate_skin_type(
    true_labels: List[int],
    pred_labels: List[int],
    class_names: List[str],
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)
    cm = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, target_names=class_names, digits=4)
    print("\n[Skin Type] Classification Report:")
    print(report)
    with (output_dir / "skin_type_report.txt").open("w", encoding="utf-8") as f:
        f.write(report)

    plt.figure(figsize=(7, 7))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(cmap="Blues", xticks_rotation=45)
    plt.title("Skin Type Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "skin_type_confusion.png", dpi=200)
    plt.close()


def evaluate_skin_issues(
    true_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    issue_names: List[str],
    output_dir: Path,
    threshold: float = 0.5,
) -> None:
    ensure_dir(output_dir)
    pred_bin = (pred_matrix >= threshold).astype(int)
    report = classification_report(true_matrix, pred_bin, target_names=issue_names, digits=4, zero_division=0)
    print("\n[Skin Issues] Multi-label Classification Report:")
    print(report)
    with (output_dir / "skin_issues_report.txt").open("w", encoding="utf-8") as f:
        f.write(report)


# ═══════════════════════════════════════════════════════════════════════════
#  Label map save / load
# ═══════════════════════════════════════════════════════════════════════════
def save_label_map(skin_types: List[str], skin_issues: List[str], path: Path) -> None:
    data = {"skin_types": skin_types, "skin_issues": skin_issues}
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_label_map(path: Path) -> Tuple[List[str], List[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["skin_types"], data["skin_issues"]


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")
