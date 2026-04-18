"""
Serial Bridge: Python ↔ Arduino Mega
─────────────────────────────────────
Runs the ViT skin analysis model on the PC/Raspberry Pi.
Listens for requests from Arduino over Serial, captures an image,
runs inference, and sends the result back.

Protocol:
  Arduino → Python:  "REQ:SCAN\n"       (user pressed Scan on TFT)
  Python  → Arduino: "RESULT:OILY,ACNE,DARK_SPOTS\n"

Usage:
  python serial_bridge.py --port COM5 --checkpoint outputs/artifacts/skin_multitask_vit_final.pth
  python serial_bridge.py --port /dev/ttyUSB0   # Linux / Raspberry Pi
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import serial
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import load_multitask_vit

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


class SkinBridge:
    def __init__(self, checkpoint: str, port: str, baud: int = 9600,
                 camera_id: int = 0, issue_threshold: float = 0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.issue_threshold = issue_threshold
        self.camera_id = camera_id

        print(f"[bridge] Loading model from {checkpoint} ...")
        self.model, self.skin_types, self.skin_issues, self.image_size = \
            load_multitask_vit(checkpoint, self.device)
        print(f"[bridge] Model ready on {self.device.upper()}")

        print(f"[bridge] Opening serial {port} @ {baud} baud ...")
        self.ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2)  # wait for Arduino reset after serial connection
        print("[bridge] Serial connected")

    @torch.no_grad()
    def predict(self, pil_image: Image.Image) -> str:
        """Run inference and return the protocol string, e.g. 'OILY,ACNE,DARK_SPOTS'."""
        x = TRANSFORM(pil_image).unsqueeze(0).to(self.device)
        type_logits, issue_logits = self.model(x)

        type_probs = torch.softmax(type_logits, dim=1)[0]
        type_idx = type_probs.argmax().item()
        skin_type = self.skin_types[type_idx].upper()

        issue_probs = torch.sigmoid(issue_logits)[0]
        detected = []
        for i, p in enumerate(issue_probs):
            if p.item() >= self.issue_threshold:
                detected.append(self.skin_issues[i].upper())

        parts = [skin_type] + detected
        return ",".join(parts)

    def capture_and_predict(self) -> str:
        """Open camera, grab a frame, close, predict."""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            return "ERROR"

        # let the camera warm up
        for _ in range(10):
            cap.read()

        ok, frame = cap.read()
        cap.release()

        if not ok:
            return "ERROR"

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return self.predict(pil)

    def send(self, msg: str):
        line = msg.strip() + "\n"
        self.ser.write(line.encode("ascii"))
        self.ser.flush()
        print(f"[bridge] TX → {msg}")

    def run(self):
        """Main loop: listen for Arduino commands, respond."""
        print("[bridge] Listening for Arduino commands ... (Ctrl+C to quit)")
        buf = ""
        while True:
            try:
                raw = self.ser.read(256)
                if raw:
                    text = raw.decode("ascii", errors="replace")
                    buf += text
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        print(f"[bridge] RX ← {line}")

                        if line == "REQ:SCAN":
                            print("[bridge] Capturing image ...")
                            result = self.capture_and_predict()
                            self.send(f"RESULT:{result}")

                        elif line == "READY":
                            print("[bridge] Arduino is ready.")

                        elif line.startswith("DISPENSING:") or line.startswith("DONE:"):
                            print(f"[bridge] Arduino: {line}")

                        else:
                            print(f"[bridge] Unknown: {line}")
            except KeyboardInterrupt:
                print("\n[bridge] Shutting down.")
                self.ser.close()
                break
            except Exception as e:
                print(f"[bridge] Error: {e}")
                time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="Python ↔ Arduino skin analysis bridge")
    parser.add_argument("--port", required=True, help="Serial port (COM5, /dev/ttyUSB0, etc.)")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--checkpoint",
                        default="outputs/artifacts/skin_multitask_vit_final.pth",
                        help="Path to trained .pth checkpoint")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Issue detection threshold")
    args = parser.parse_args()

    bridge = SkinBridge(
        checkpoint=args.checkpoint,
        port=args.port,
        baud=args.baud,
        camera_id=args.camera,
        issue_threshold=args.threshold,
    )
    bridge.run()


if __name__ == "__main__":
    main()
