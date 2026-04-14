"""
Skin Type Detection GUI
-----------------------
Live camera feed with real-time and capture-based classification.
Uses the trained ViT model (.pth) or ONNX model for inference.
"""

import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageTk
from torchvision import transforms

from model import create_vit_model

# ── Paths (defaults – override via CLI or Settings) ─────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "outputs" / "artifacts" / "skin_vit_classifier_final.pth"
DEFAULT_ONNX = ROOT / "outputs" / "artifacts" / "skin_vit_classifier.onnx"
DEFAULT_LABEL_MAP = ROOT / "outputs" / "artifacts" / "label_map.json"

# ── Colour palette ──────────────────────────────────────────────────────────
BG = "#1a1a2e"
PANEL_BG = "#16213e"
ACCENT = "#0f3460"
HIGHLIGHT = "#e94560"
TEXT_FG = "#e0e0e0"
TEXT_MUTED = "#8d99ae"
SUCCESS = "#2ecc71"
WARNING = "#f39c12"

SKIN_COLOURS = {
    "dry": "#e67e22",
    "normal": "#2ecc71",
    "oily": "#3498db",
}

# ── Preprocessing (must match training) ─────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ═══════════════════════════════════════════════════════════════════════════
#  Model back-end (loads once, used from any thread)
# ═══════════════════════════════════════════════════════════════════════════
class SkinClassifier:
    def __init__(self, checkpoint_path: Path, device: str = "cpu"):
        self.device = device
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.class_names = ckpt["class_names"]
        self.image_size = ckpt.get("image_size", 224)
        self.model = create_vit_model(
            num_classes=len(self.class_names),
            model_name=ckpt["model_name"],
            class_names=self.class_names,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, pil_image: Image.Image):
        x = TRANSFORM(pil_image).unsqueeze(0).to(self.device)
        logits = self.model(pixel_values=x).logits
        probs = torch.softmax(logits, dim=1)[0]
        idx = probs.argmax().item()
        return self.class_names[idx], probs[idx].item(), {
            n: probs[i].item() for i, n in enumerate(self.class_names)
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Main GUI Application
# ═══════════════════════════════════════════════════════════════════════════
class SkinDetectionApp:
    CAMERA_W, CAMERA_H = 640, 480

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Skin Type Detection  -  ViT Classifier")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.minsize(900, 620)

        self.cap = None
        self.running = False
        self.live_classify = False
        self.last_pred_time = 0.0
        self.classifier = None
        self.model_loaded = False
        self._photo_ref = None  # prevent GC of PhotoImage

        self._build_ui()
        self._load_model_async()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI Construction ─────────────────────────────────────────────────
    def _build_ui(self):
        # Top bar
        top = tk.Frame(self.root, bg=ACCENT, height=52)
        top.pack(fill="x")
        tk.Label(
            top, text="  Skin Type Detection", font=("Segoe UI", 16, "bold"),
            fg="white", bg=ACCENT, anchor="w",
        ).pack(side="left", padx=12, pady=8)
        self.status_lbl = tk.Label(
            top, text="Loading model...", font=("Segoe UI", 10),
            fg=WARNING, bg=ACCENT,
        )
        self.status_lbl.pack(side="right", padx=16)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=10)

        # Left: camera
        left = tk.Frame(body, bg=PANEL_BG, bd=0, highlightthickness=1,
                        highlightbackground=ACCENT)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.canvas = tk.Canvas(left, bg="#000", highlightthickness=0,
                                width=self.CAMERA_W, height=self.CAMERA_H)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)

        # Right: controls + results
        right = tk.Frame(body, bg=PANEL_BG, width=280, bd=0,
                         highlightthickness=1, highlightbackground=ACCENT)
        right.pack(side="right", fill="y", padx=(5, 0))
        right.pack_propagate(False)

        self._section(right, "Camera Controls")
        btn_frame = tk.Frame(right, bg=PANEL_BG)
        btn_frame.pack(fill="x", padx=12, pady=(0, 6))

        self.btn_start = self._button(btn_frame, "Start Camera", self._toggle_camera)
        self.btn_start.pack(fill="x", pady=2)

        self.btn_capture = self._button(btn_frame, "Capture & Classify", self._capture)
        self.btn_capture.pack(fill="x", pady=2)
        self.btn_capture.config(state="disabled")

        self.live_var = tk.BooleanVar(value=False)
        self.chk_live = tk.Checkbutton(
            btn_frame, text="  Live classification", variable=self.live_var,
            command=self._toggle_live, font=("Segoe UI", 10),
            fg=TEXT_FG, bg=PANEL_BG, selectcolor=ACCENT,
            activebackground=PANEL_BG, activeforeground=TEXT_FG,
        )
        self.chk_live.pack(anchor="w", pady=4)

        self._section(right, "Or Upload Image")
        upload_frame = tk.Frame(right, bg=PANEL_BG)
        upload_frame.pack(fill="x", padx=12, pady=(0, 10))
        self._button(upload_frame, "Browse Image...", self._upload_image).pack(fill="x")

        self._section(right, "Result")
        self.result_frame = tk.Frame(right, bg=PANEL_BG)
        self.result_frame.pack(fill="x", padx=12)

        self.lbl_pred = tk.Label(
            self.result_frame, text="---", font=("Segoe UI", 22, "bold"),
            fg=HIGHLIGHT, bg=PANEL_BG,
        )
        self.lbl_pred.pack(pady=(4, 0))

        self.lbl_conf = tk.Label(
            self.result_frame, text="", font=("Segoe UI", 11),
            fg=TEXT_MUTED, bg=PANEL_BG,
        )
        self.lbl_conf.pack()

        self._section(right, "Class Probabilities")
        self.bar_frame = tk.Frame(right, bg=PANEL_BG)
        self.bar_frame.pack(fill="x", padx=12, pady=(0, 8))

        self.bars = {}
        for cls in ["dry", "normal", "oily"]:
            row = tk.Frame(self.bar_frame, bg=PANEL_BG)
            row.pack(fill="x", pady=3)

            tk.Label(
                row, text=cls.capitalize(), font=("Segoe UI", 10, "bold"),
                fg=TEXT_FG, bg=PANEL_BG, width=8, anchor="w",
            ).pack(side="left")

            bar_bg = tk.Frame(row, bg="#2d3436", height=18)
            bar_bg.pack(side="left", fill="x", expand=True, padx=(4, 4))
            bar_bg.pack_propagate(False)

            bar_fill = tk.Frame(bar_bg, bg=SKIN_COLOURS.get(cls, HIGHLIGHT), height=18)
            bar_fill.place(relx=0, rely=0, relheight=1.0, relwidth=0.0)

            pct_lbl = tk.Label(
                row, text="0%", font=("Segoe UI", 9),
                fg=TEXT_MUTED, bg=PANEL_BG, width=5, anchor="e",
            )
            pct_lbl.pack(side="right")

            self.bars[cls] = (bar_fill, pct_lbl)

        # Bottom hint
        tk.Label(
            right, text="Press Space to capture\nEsc to stop camera",
            font=("Segoe UI", 9), fg=TEXT_MUTED, bg=PANEL_BG, justify="center",
        ).pack(side="bottom", pady=10)

        # Keyboard shortcuts
        self.root.bind("<space>", lambda e: self._capture())
        self.root.bind("<Escape>", lambda e: self._stop_camera())

    def _section(self, parent, title):
        tk.Label(
            parent, text=title, font=("Segoe UI", 11, "bold"),
            fg=TEXT_MUTED, bg=PANEL_BG, anchor="w",
        ).pack(fill="x", padx=12, pady=(12, 2))
        sep = tk.Frame(parent, bg=ACCENT, height=1)
        sep.pack(fill="x", padx=12, pady=(0, 6))

    def _button(self, parent, text, command):
        btn = tk.Button(
            parent, text=text, command=command,
            font=("Segoe UI", 10, "bold"), fg="white", bg=HIGHLIGHT,
            activebackground="#c0392b", activeforeground="white",
            relief="flat", cursor="hand2", padx=10, pady=6,
        )
        return btn

    # ── Model Loading ───────────────────────────────────────────────────
    def _load_model_async(self):
        def _load():
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.classifier = SkinClassifier(DEFAULT_CHECKPOINT, device)
                self.model_loaded = True
                self.root.after(0, lambda: self.status_lbl.config(
                    text=f"Model ready  ({device.upper()})", fg=SUCCESS))
            except Exception as e:
                self.root.after(0, lambda: self.status_lbl.config(
                    text=f"Model error: {e}", fg=HIGHLIGHT))
        threading.Thread(target=_load, daemon=True).start()

    # ── Camera ──────────────────────────────────────────────────────────
    def _toggle_camera(self):
        if self.running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open laptop camera.")
            return
        self.running = True
        self.btn_start.config(text="Stop Camera", bg="#c0392b")
        self.btn_capture.config(state="normal")
        self._update_camera()

    def _stop_camera(self):
        self.running = False
        self.live_classify = False
        self.live_var.set(False)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_start.config(text="Start Camera", bg=HIGHLIGHT)
        self.btn_capture.config(state="disabled")
        self.canvas.delete("all")

    def _update_camera(self):
        if not self.running or not self.cap:
            return
        ok, frame = self.cap.read()
        if ok:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)

            # Live classification (throttled to once per second)
            if self.live_classify and self.model_loaded:
                now = time.time()
                if now - self.last_pred_time > 1.0:
                    self.last_pred_time = now
                    threading.Thread(
                        target=self._classify_pil, args=(pil.copy(),), daemon=True
                    ).start()

            self._show_on_canvas(pil)

        self.root.after(30, self._update_camera)

    def _show_on_canvas(self, pil_img: Image.Image):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            cw, ch = self.CAMERA_W, self.CAMERA_H

        pil_img = pil_img.resize((cw, ch), Image.LANCZOS)
        self._photo_ref = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo_ref)

    # ── Capture & Classify ──────────────────────────────────────────────
    def _capture(self):
        if not self.running or not self.cap:
            return
        if not self.model_loaded:
            messagebox.showinfo("Wait", "Model is still loading...")
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._show_on_canvas(pil.copy())
        threading.Thread(target=self._classify_pil, args=(pil,), daemon=True).start()

    def _upload_image(self):
        if not self.model_loaded:
            messagebox.showinfo("Wait", "Model is still loading...")
            return
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if not path:
            return
        pil = Image.open(path).convert("RGB")
        self._show_on_canvas(pil.copy())
        threading.Thread(target=self._classify_pil, args=(pil,), daemon=True).start()

    def _toggle_live(self):
        self.live_classify = self.live_var.get()

    # ── Classification (runs off-thread) ────────────────────────────────
    def _classify_pil(self, pil_img: Image.Image):
        try:
            label, conf, probs = self.classifier.predict(pil_img)
            self.root.after(0, lambda: self._update_result(label, conf, probs))
        except Exception as e:
            self.root.after(0, lambda: self.lbl_pred.config(text=f"Error: {e}"))

    def _update_result(self, label: str, conf: float, probs: dict):
        colour = SKIN_COLOURS.get(label, HIGHLIGHT)
        self.lbl_pred.config(text=label.upper(), fg=colour)
        self.lbl_conf.config(text=f"Confidence: {conf:.1%}")

        for cls, (bar_fill, pct_lbl) in self.bars.items():
            p = probs.get(cls, 0)
            bar_fill.place(relwidth=max(p, 0.01))
            pct_lbl.config(text=f"{p:.0%}")

    # ── Cleanup ─────────────────────────────────────────────────────────
    def _on_close(self):
        self._stop_camera()
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    SkinDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
