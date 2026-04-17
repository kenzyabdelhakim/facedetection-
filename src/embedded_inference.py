"""
Lightweight multi-task ONNX inference for edge devices.
Outputs both skin type and detected skin issues.
Dependencies: onnxruntime, opencv-python, numpy only.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def load_labels(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["skin_types"], data["skin_issues"]


def preprocess(frame_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size)).astype(np.float32) / 255.0
    tensor = (resized - 0.5) / 0.5
    return np.expand_dims(np.transpose(tensor, (2, 0, 1)), 0)


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def run_camera(model_path: Path, label_path: Path, camera_id: int = 0, size: int = 224):
    skin_types, skin_issues = load_labels(label_path)
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Press ESC to exit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        x = preprocess(frame, size)
        type_logits, issue_logits = sess.run(None, {inp: x})

        type_probs = softmax(type_logits)[0]
        type_idx = int(np.argmax(type_probs))
        type_conf = float(type_probs[type_idx])

        issue_probs = sigmoid(issue_logits)[0]
        detected = [skin_issues[i] for i, p in enumerate(issue_probs) if p >= 0.5]

        # Draw skin type
        type_text = f"Type: {skin_types[type_idx]} ({type_conf:.0%})"
        cv2.putText(frame, type_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw issues
        if detected:
            issues_text = "Issues: " + ", ".join(
                i.replace("_", " ").title() for i in detected
            )
        else:
            issues_text = "Issues: None"
        cv2.putText(frame, issues_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        # Draw individual issue bars
        y0 = 90
        for i, iss in enumerate(skin_issues):
            p = float(issue_probs[i])
            nice = iss.replace("_", " ").title()
            bar_w = int(p * 150)
            colour = (0, 255, 0) if p < 0.5 else (0, 0, 255)
            cv2.rectangle(frame, (10, y0), (10 + bar_w, y0 + 12), colour, -1)
            cv2.putText(frame, f"{nice}: {p:.0%}", (170, y0 + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            y0 += 20

        cv2.imshow("Skin Analysis (Edge)", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Edge multi-task skin analysis")
    p.add_argument("--model", default="outputs/artifacts/skin_multitask_vit.onnx")
    p.add_argument("--label_map", default="outputs/artifacts/label_map.json")
    p.add_argument("--camera_id", type=int, default=0)
    p.add_argument("--image_size", type=int, default=224)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_camera(Path(args.model), Path(args.label_map), args.camera_id, args.image_size)
