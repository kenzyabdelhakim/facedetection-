"""
Lightweight ONNX inference script for edge devices (Raspberry Pi, x86 mini PCs, etc.).
This script uses only ONNX Runtime + OpenCV + NumPy.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def load_labels(label_map_path: Path):
    with label_map_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [data[str(i)] for i in range(len(data))]


def preprocess_bgr(frame_bgr: np.ndarray, image_size: int = 224) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (image_size, image_size))
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    return tensor


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def run_camera_inference(
    model_path: Path,
    label_map_path: Path,
    camera_id: int = 0,
    image_size: int = 224,
):
    labels = load_labels(label_map_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Press ESC to exit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        x = preprocess_bgr(frame, image_size=image_size)
        logits = session.run(None, {input_name: x})[0]
        probs = softmax(logits)

        idx = int(np.argmax(probs[0]))
        conf = float(probs[0, idx])
        text = f"{labels[idx]} ({conf:.2f})"

        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Embedded Skin Type Inference", frame)

        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight ONNX camera inference")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/artifacts/skin_vit_classifier_int8.onnx",
    )
    parser.add_argument(
        "--label_map",
        type=str,
        default="outputs/artifacts/label_map.json",
    )
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_camera_inference(
        model_path=Path(args.model),
        label_map_path=Path(args.label_map),
        camera_id=args.camera_id,
        image_size=args.image_size,
    )
