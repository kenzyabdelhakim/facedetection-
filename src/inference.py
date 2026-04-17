"""
Multi-task inference: returns skin type + detected skin issues.

Output format:
    {
      "skin_type": "Oily",
      "skin_type_confidence": 0.93,
      "issues": ["Acne", "Large Pores"],
      "issue_scores": {"acne": 0.87, "dark_spots": 0.12, ...}
    }
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import MultiTaskViT, create_multitask_vit


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


# ═══════════════════════════════════════════════════════════════════════════
#  Checkpoint loading
# ═══════════════════════════════════════════════════════════════════════════
def load_checkpoint(checkpoint_path: Path, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    skin_types = ckpt["skin_types"]
    skin_issues = ckpt["skin_issues"]
    image_size = ckpt.get("image_size", 224)

    model = create_multitask_vit(
        model_name=ckpt["model_name"],
        skin_types=skin_types,
        skin_issues=skin_issues,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, skin_types, skin_issues, image_size


# ═══════════════════════════════════════════════════════════════════════════
#  Core prediction
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict_pil(
    model: MultiTaskViT,
    pil_image: Image.Image,
    skin_types: List[str],
    skin_issues: List[str],
    device: str = "cpu",
    issue_threshold: float = 0.5,
) -> Dict:
    x = TRANSFORM(pil_image).unsqueeze(0).to(device)
    type_logits, issue_logits = model(x)

    type_probs = torch.softmax(type_logits, dim=1)[0]
    type_idx = type_probs.argmax().item()

    issue_probs = torch.sigmoid(issue_logits)[0]
    detected = [
        skin_issues[i] for i, p in enumerate(issue_probs) if p.item() >= issue_threshold
    ]

    return {
        "skin_type": skin_types[type_idx],
        "skin_type_confidence": type_probs[type_idx].item(),
        "type_probs": {n: type_probs[i].item() for i, n in enumerate(skin_types)},
        "issues": detected,
        "issue_scores": {n: issue_probs[i].item() for i, n in enumerate(skin_issues)},
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Image / Camera helpers
# ═══════════════════════════════════════════════════════════════════════════
def predict_image(image_path: Path, checkpoint_path: Path, device: str = "cpu") -> Dict:
    model, stypes, sissues, _ = load_checkpoint(checkpoint_path, device)
    pil = Image.open(image_path).convert("RGB")
    return predict_pil(model, pil, stypes, sissues, device)


def predict_from_camera(
    checkpoint_path: Path, device: str = "cpu", camera_id: int = 0
) -> Dict:
    model, stypes, sissues, _ = load_checkpoint(checkpoint_path, device)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Press SPACE to capture, ESC to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imshow("Skin Analysis - Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Cancelled.")
        if key == 32:
            break

    cap.release()
    cv2.destroyAllWindows()
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return predict_pil(model, pil, stypes, sissues, device)


# ═══════════════════════════════════════════════════════════════════════════
#  ONNX inference
# ═══════════════════════════════════════════════════════════════════════════
def predict_onnx(
    image_path: Path,
    onnx_path: Path,
    label_map_path: Path,
    image_size: int = 224,
) -> Dict:
    import onnxruntime as ort

    with label_map_path.open("r") as f:
        lm = json.load(f)
    skin_types = lm["skin_types"]
    skin_issues = lm["skin_issues"]

    pil = Image.open(image_path).convert("RGB")
    x = TRANSFORM(pil).unsqueeze(0).numpy().astype(np.float32)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    type_logits, issue_logits = sess.run(None, {"pixel_values": x})

    # softmax for types
    e = np.exp(type_logits - type_logits.max(axis=1, keepdims=True))
    type_probs = (e / e.sum(axis=1, keepdims=True))[0]
    type_idx = int(type_probs.argmax())

    # sigmoid for issues
    issue_probs = 1.0 / (1.0 + np.exp(-issue_logits[0]))
    detected = [skin_issues[i] for i, p in enumerate(issue_probs) if p >= 0.5]

    return {
        "skin_type": skin_types[type_idx],
        "skin_type_confidence": float(type_probs[type_idx]),
        "type_probs": {n: float(type_probs[i]) for i, n in enumerate(skin_types)},
        "issues": detected,
        "issue_scores": {n: float(issue_probs[i]) for i, n in enumerate(skin_issues)},
    }


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Multi-task skin analysis inference")
    p.add_argument("--checkpoint", default="outputs/artifacts/skin_multitask_vit_final.pth")
    p.add_argument("--image", default=None)
    p.add_argument("--use_camera", action="store_true")
    p.add_argument("--camera_id", type=int, default=0)
    p.add_argument("--onnx", default=None)
    p.add_argument("--label_map", default="outputs/artifacts/label_map.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.onnx:
        if not args.image:
            raise ValueError("--onnx requires --image")
        result = predict_onnx(Path(args.image), Path(args.onnx), Path(args.label_map))
    elif args.use_camera:
        result = predict_from_camera(Path(args.checkpoint), device, args.camera_id)
    else:
        if not args.image:
            raise ValueError("Provide --image or --use_camera")
        result = predict_image(Path(args.image), Path(args.checkpoint), device)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
