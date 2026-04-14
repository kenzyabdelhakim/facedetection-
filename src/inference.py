import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import create_vit_model


def preprocess(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return tf(image).unsqueeze(0)


def load_checkpoint(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    model = create_vit_model(
        num_classes=len(class_names),
        model_name=checkpoint["model_name"],
        class_names=class_names,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, checkpoint.get("image_size", 224)


def predict_image(
    image_path: Path, checkpoint_path: Path, device: str = "cpu"
) -> Tuple[str, float]:
    model, class_names, image_size = load_checkpoint(checkpoint_path, device)
    image = Image.open(image_path).convert("RGB")
    x = preprocess(image, image_size=image_size).to(device)

    with torch.no_grad():
        logits = model(pixel_values=x).logits
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    return class_names[pred_idx.item()], conf.item()


def predict_from_camera(
    checkpoint_path: Path, device: str = "cpu", camera_id: int = 0
) -> Tuple[str, float]:
    model, class_names, image_size = load_checkpoint(checkpoint_path, device)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Press SPACE to capture frame, ESC to quit.")
    frame = None
    while True:
        ok, current = cap.read()
        if not ok:
            continue
        cv2.imshow("Skin Type Detection - Camera", current)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Camera inference cancelled by user.")
        if key == 32:  # SPACE
            frame = current
            break

    cap.release()
    cv2.destroyAllWindows()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    x = preprocess(image, image_size=image_size).to(device)

    with torch.no_grad():
        logits = model(pixel_values=x).logits
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    return class_names[pred_idx.item()], conf.item()


def load_label_map(label_map_path: Path) -> List[str]:
    with label_map_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [data[str(i)] for i in range(len(data))]


def predict_onnx(
    image_path: Path,
    onnx_path: Path,
    label_map_path: Path,
    image_size: int = 224,
) -> Tuple[str, float]:
    import onnxruntime as ort

    class_names = load_label_map(label_map_path)
    image = Image.open(image_path).convert("RGB")
    x = preprocess(image, image_size=image_size).numpy().astype(np.float32)

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    logits = session.run(None, {"pixel_values": x})[0]
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    pred_idx = int(np.argmax(probs, axis=1)[0])
    confidence = float(probs[0, pred_idx])
    return class_names[pred_idx], confidence


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for skin type classifier")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/artifacts/skin_vit_classifier_final.pth",
    )
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--use_camera", action="store_true")
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument("--label_map", type=str, default="outputs/artifacts/label_map.json")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.onnx:
        if not args.image:
            raise ValueError("When using --onnx, please provide --image.")
        pred, conf = predict_onnx(
            image_path=Path(args.image),
            onnx_path=Path(args.onnx),
            label_map_path=Path(args.label_map),
        )
    elif args.use_camera:
        pred, conf = predict_from_camera(
            checkpoint_path=Path(args.checkpoint),
            device=device,
            camera_id=args.camera_id,
        )
    else:
        if not args.image:
            raise ValueError("Please provide --image or use --use_camera.")
        pred, conf = predict_image(
            image_path=Path(args.image),
            checkpoint_path=Path(args.checkpoint),
            device=device,
        )

    print(f"Predicted skin type: {pred} (confidence={conf:.4f})")


if __name__ == "__main__":
    main()
