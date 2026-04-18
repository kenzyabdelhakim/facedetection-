"""
Skin Analysis GUI  –  Multi-task ViT
────────────────────────────────────
Live camera feed with real-time skin type + skin issue detection.
"""

import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import torch
from PIL import Image, ImageTk
from torchvision import transforms

from model import load_multitask_vit, SKIN_TYPES, SKIN_ISSUES

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "outputs" / "artifacts" / "skin_multitask_vit_final.pth"

# ── Theme ────────────────────────────────────────────────────────────────
BG = "#1a1a2e"
PANEL_BG = "#16213e"
ACCENT = "#0f3460"
HIGHLIGHT = "#e94560"
TEXT_FG = "#e0e0e0"
TEXT_MUTED = "#8d99ae"
SUCCESS = "#2ecc71"
WARNING = "#f39c12"

SKIN_COLOURS = {"dry": "#e67e22", "normal": "#2ecc71", "oily": "#3498db"}

ISSUE_COLOURS = {
    "acne": "#e74c3c",
    "dark_spots": "#8e44ad",
    "wrinkles": "#f39c12",
    "redness": "#e94560",
    "large_pores": "#00b894",
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


# ═══════════════════════════════════════════════════════════════════════════
#  Classifier backend
# ═══════════════════════════════════════════════════════════════════════════
class SkinClassifier:
    def __init__(self, checkpoint_path: Path, device: str = "cpu"):
        self.device = device
        self.model, self.skin_types, self.skin_issues, self.image_size = (
            load_multitask_vit(str(checkpoint_path), device)
        )

    @torch.no_grad()
    def predict(self, pil_image: Image.Image):
        x = TRANSFORM(pil_image).unsqueeze(0).to(self.device)
        type_logits, issue_logits = self.model(x)

        type_probs = torch.softmax(type_logits, dim=1)[0]
        type_idx = type_probs.argmax().item()

        issue_probs = torch.sigmoid(issue_logits)[0]
        detected = [
            self.skin_issues[i]
            for i, p in enumerate(issue_probs) if p.item() >= 0.5
        ]

        return {
            "skin_type": self.skin_types[type_idx],
            "skin_type_confidence": type_probs[type_idx].item(),
            "type_probs": {n: type_probs[i].item() for i, n in enumerate(self.skin_types)},
            "issues": detected,
            "issue_scores": {n: issue_probs[i].item() for i, n in enumerate(self.skin_issues)},
        }


# ═══════════════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════════════
class SkinDetectionApp:
    CAMERA_W, CAMERA_H = 640, 480

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Skin Analysis  –  Multi-task ViT")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.minsize(960, 680)

        self.cap = None
        self.running = False
        self.live_classify = False
        self.last_pred_time = 0.0
        self.classifier = None
        self.model_loaded = False
        self._photo_ref = None

        self._build_ui()
        self._load_model_async()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Build UI ─────────────────────────────────────────────────────
    def _build_ui(self):
        # Top bar
        top = tk.Frame(self.root, bg=ACCENT, height=52)
        top.pack(fill="x")
        tk.Label(top, text="  Skin Analysis", font=("Segoe UI", 16, "bold"),
                 fg="white", bg=ACCENT).pack(side="left", padx=12, pady=8)
        self.status_lbl = tk.Label(top, text="Loading model...",
                                   font=("Segoe UI", 10), fg=WARNING, bg=ACCENT)
        self.status_lbl.pack(side="right", padx=16)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=10)

        # Left: camera
        left = tk.Frame(body, bg=PANEL_BG, highlightthickness=1, highlightbackground=ACCENT)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.canvas = tk.Canvas(left, bg="#000", highlightthickness=0,
                                width=self.CAMERA_W, height=self.CAMERA_H)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)

        # Right panel (scrollable area)
        right = tk.Frame(body, bg=PANEL_BG, width=310, highlightthickness=1,
                         highlightbackground=ACCENT)
        right.pack(side="right", fill="y", padx=(5, 0))
        right.pack_propagate(False)

        # --- Controls ---
        self._section(right, "Controls")
        btn_f = tk.Frame(right, bg=PANEL_BG)
        btn_f.pack(fill="x", padx=12, pady=(0, 4))

        self.btn_start = self._btn(btn_f, "Start Camera", self._toggle_camera)
        self.btn_start.pack(fill="x", pady=2)
        self.btn_capture = self._btn(btn_f, "Capture & Classify", self._capture)
        self.btn_capture.pack(fill="x", pady=2)
        self.btn_capture.config(state="disabled")

        self.live_var = tk.BooleanVar()
        tk.Checkbutton(btn_f, text="  Live classification", variable=self.live_var,
                       command=self._toggle_live, font=("Segoe UI", 10),
                       fg=TEXT_FG, bg=PANEL_BG, selectcolor=ACCENT,
                       activebackground=PANEL_BG, activeforeground=TEXT_FG
                       ).pack(anchor="w", pady=4)

        self._btn(btn_f, "Browse Image...", self._upload_image).pack(fill="x", pady=2)

        # --- Skin Type Result ---
        self._section(right, "Skin Type")
        rf = tk.Frame(right, bg=PANEL_BG)
        rf.pack(fill="x", padx=12)
        self.lbl_type = tk.Label(rf, text="---", font=("Segoe UI", 22, "bold"),
                                 fg=HIGHLIGHT, bg=PANEL_BG)
        self.lbl_type.pack(pady=(2, 0))
        self.lbl_conf = tk.Label(rf, text="", font=("Segoe UI", 10),
                                 fg=TEXT_MUTED, bg=PANEL_BG)
        self.lbl_conf.pack()

        # Type bars
        self.type_bars = {}
        for cls in SKIN_TYPES:
            row = tk.Frame(rf, bg=PANEL_BG)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=cls.capitalize(), font=("Segoe UI", 9, "bold"),
                     fg=TEXT_FG, bg=PANEL_BG, width=7, anchor="w").pack(side="left")
            bar_bg = tk.Frame(row, bg="#2d3436", height=14)
            bar_bg.pack(side="left", fill="x", expand=True, padx=4)
            bar_bg.pack_propagate(False)
            bar_fill = tk.Frame(bar_bg, bg=SKIN_COLOURS.get(cls, HIGHLIGHT), height=14)
            bar_fill.place(relx=0, rely=0, relheight=1, relwidth=0)
            pct = tk.Label(row, text="0%", font=("Segoe UI", 8),
                           fg=TEXT_MUTED, bg=PANEL_BG, width=5, anchor="e")
            pct.pack(side="right")
            self.type_bars[cls] = (bar_fill, pct)

        # --- Skin Issues ---
        self._section(right, "Detected Issues")
        self.issues_frame = tk.Frame(right, bg=PANEL_BG)
        self.issues_frame.pack(fill="x", padx=12)
        self.lbl_no_issues = tk.Label(self.issues_frame, text="None detected",
                                      font=("Segoe UI", 10), fg=TEXT_MUTED, bg=PANEL_BG)
        self.lbl_no_issues.pack(anchor="w")

        # Issue score bars
        self._section(right, "Issue Scores")
        isf = tk.Frame(right, bg=PANEL_BG)
        isf.pack(fill="x", padx=12, pady=(0, 6))
        self.issue_bars = {}
        for iss in SKIN_ISSUES:
            row = tk.Frame(isf, bg=PANEL_BG)
            row.pack(fill="x", pady=2)
            nice = iss.replace("_", " ").title()
            tk.Label(row, text=nice, font=("Segoe UI", 9),
                     fg=TEXT_FG, bg=PANEL_BG, width=11, anchor="w").pack(side="left")
            bar_bg = tk.Frame(row, bg="#2d3436", height=14)
            bar_bg.pack(side="left", fill="x", expand=True, padx=4)
            bar_bg.pack_propagate(False)
            colour = ISSUE_COLOURS.get(iss, HIGHLIGHT)
            bar_fill = tk.Frame(bar_bg, bg=colour, height=14)
            bar_fill.place(relx=0, rely=0, relheight=1, relwidth=0)
            pct = tk.Label(row, text="0%", font=("Segoe UI", 8),
                           fg=TEXT_MUTED, bg=PANEL_BG, width=5, anchor="e")
            pct.pack(side="right")
            self.issue_bars[iss] = (bar_fill, pct)

        # Keyboard hints
        tk.Label(right, text="Space = capture  |  Esc = stop camera",
                 font=("Segoe UI", 8), fg=TEXT_MUTED, bg=PANEL_BG
                 ).pack(side="bottom", pady=8)

        self.root.bind("<space>", lambda e: self._capture())
        self.root.bind("<Escape>", lambda e: self._stop_camera())

    # ── Helpers ──────────────────────────────────────────────────────
    def _section(self, parent, title):
        tk.Label(parent, text=title, font=("Segoe UI", 11, "bold"),
                 fg=TEXT_MUTED, bg=PANEL_BG, anchor="w").pack(fill="x", padx=12, pady=(10, 1))
        tk.Frame(parent, bg=ACCENT, height=1).pack(fill="x", padx=12, pady=(0, 4))

    def _btn(self, parent, text, cmd):
        return tk.Button(parent, text=text, command=cmd,
                         font=("Segoe UI", 10, "bold"), fg="white", bg=HIGHLIGHT,
                         activebackground="#c0392b", activeforeground="white",
                         relief="flat", cursor="hand2", padx=10, pady=5)

    # ── Model ────────────────────────────────────────────────────────
    def _load_model_async(self):
        def _load():
            try:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                self.classifier = SkinClassifier(DEFAULT_CHECKPOINT, dev)
                self.model_loaded = True
                self.root.after(0, lambda: self.status_lbl.config(
                    text=f"Model ready ({dev.upper()})", fg=SUCCESS))
            except Exception as e:
                self.root.after(0, lambda: self.status_lbl.config(
                    text=f"Error: {e}", fg=HIGHLIGHT))
        threading.Thread(target=_load, daemon=True).start()

    # ── Camera ───────────────────────────────────────────────────────
    def _toggle_camera(self):
        self._stop_camera() if self.running else self._start_camera()

    def _start_camera(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera.")
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
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if self.live_classify and self.model_loaded:
                now = time.time()
                if now - self.last_pred_time > 1.0:
                    self.last_pred_time = now
                    threading.Thread(target=self._classify, args=(pil.copy(),), daemon=True).start()
            self._show(pil)
        self.root.after(30, self._update_camera)

    def _show(self, pil_img):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            cw, ch = self.CAMERA_W, self.CAMERA_H
        self._photo_ref = ImageTk.PhotoImage(pil_img.resize((cw, ch), Image.LANCZOS))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo_ref)

    # ── Capture / Upload ─────────────────────────────────────────────
    def _capture(self):
        if not self.running or not self.cap or not self.model_loaded:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._show(pil.copy())
        threading.Thread(target=self._classify, args=(pil,), daemon=True).start()

    def _upload_image(self):
        if not self.model_loaded:
            messagebox.showinfo("Wait", "Model is still loading...")
            return
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if not path:
            return
        pil = Image.open(path).convert("RGB")
        self._show(pil.copy())
        threading.Thread(target=self._classify, args=(pil,), daemon=True).start()

    def _toggle_live(self):
        self.live_classify = self.live_var.get()

    # ── Classification ───────────────────────────────────────────────
    def _classify(self, pil_img):
        try:
            result = self.classifier.predict(pil_img)
            self.root.after(0, lambda: self._update_result(result))
        except Exception as e:
            self.root.after(0, lambda: self.lbl_type.config(text=f"Error"))

    def _update_result(self, r: dict):
        # Skin type
        stype = r["skin_type"]
        colour = SKIN_COLOURS.get(stype, HIGHLIGHT)
        self.lbl_type.config(text=stype.upper(), fg=colour)
        self.lbl_conf.config(text=f"Confidence: {r['skin_type_confidence']:.1%}")

        for cls, (bar, pct) in self.type_bars.items():
            v = r["type_probs"].get(cls, 0)
            bar.place(relwidth=max(v, 0.01))
            pct.config(text=f"{v:.0%}")

        # Issues
        for w in self.issues_frame.winfo_children():
            w.destroy()

        issues = r["issues"]
        if not issues:
            tk.Label(self.issues_frame, text="No issues detected",
                     font=("Segoe UI", 10), fg=SUCCESS, bg=PANEL_BG).pack(anchor="w")
        else:
            for iss in issues:
                nice = iss.replace("_", " ").title()
                score = r["issue_scores"].get(iss, 0)
                colour = ISSUE_COLOURS.get(iss, HIGHLIGHT)
                tk.Label(
                    self.issues_frame,
                    text=f"  {nice}  ({score:.0%})",
                    font=("Segoe UI", 10, "bold"), fg=colour, bg=PANEL_BG,
                ).pack(anchor="w", pady=1)

        # Issue bars
        for iss, (bar, pct) in self.issue_bars.items():
            v = r["issue_scores"].get(iss, 0)
            bar.place(relwidth=max(v, 0.01))
            pct.config(text=f"{v:.0%}")

    # ── Cleanup ──────────────────────────────────────────────────────
    def _on_close(self):
        self._stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    SkinDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
