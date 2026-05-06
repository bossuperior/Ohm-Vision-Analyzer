import cv2
import json
import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from src.vision.camera_loader import CameraLoader
from src.inference.model_engine import ModelEngine
from src.vision.breadboard_warper import BreadboardWarper
from src.topology.grid_mapper import GridMapper

CONFIG_PATH = "config/ui_state.json"

# ── Colors ────────────────────────────────────────────────────────────────────
BOX_COLORS = {
    "resistor": (0, 200, 255),
    "wire":     (255, 180, 0),
}
DEFAULT_BOX_COLOR = (180, 180, 180)

RESISTOR_KP_COLORS = [
    (0, 0, 255),   # Leg_0  — red
    (0, 0, 255),   # Leg_1  — red
    (0, 255, 0),   # Body_2 — green
    (0, 255, 0),   # Body_3 — green
]

# ── Drawing helpers ────────────────────────────────────────────────────────────
def transform_points(pts, matrix):
    if len(pts) == 0:
        return pts
    src = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(src, matrix).reshape(-1, 2)


def transform_box(box, matrix):
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    t = transform_points(corners, matrix)
    return int(t[:, 0].min()), int(t[:, 1].min()), \
           int(t[:, 0].max()), int(t[:, 1].max())


def draw_results(frame, results, class_names, matrix=None):
    for i, (box, cls_id, score) in enumerate(
            zip(results.boxes, results.class_ids, results.scores)):
        name  = class_names.get(int(cls_id), f"cls{int(cls_id)}")
        color = BOX_COLORS.get(name, DEFAULT_BOX_COLOR)

        if matrix is not None:
            x1, y1, x2, y2 = transform_box(box, matrix)
        else:
            x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {score:.2f}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if len(results.keypoints) > i:
            kp_data     = results.keypoints[i]
            is_resistor = (name == "resistor")
            raw_kps     = [(j, kp[:2]) for j, kp in enumerate(kp_data) if kp[2] > 0.5]
            xy_only     = [kp for _, kp in raw_kps]

            if matrix is not None and xy_only:
                xy_only = transform_points(xy_only, matrix)

            kp_map = {j: tuple(map(int, xy_only[k])) for k, (j, _) in enumerate(raw_kps)}

            for j, pt in kp_map.items():
                c = RESISTOR_KP_COLORS[j] if is_resistor and j < len(RESISTOR_KP_COLORS) \
                    else (0, 255, 255)
                cv2.circle(frame, pt, 5, c, -1)

            if is_resistor and 2 in kp_map and 3 in kp_map:
                cv2.line(frame, kp_map[2], kp_map[3], (0, 255, 0), 2)


# ── Dark-theme palette ─────────────────────────────────────────────────────────
BG       = "#1a1a2e"
PANEL_BG = "#16213e"
ACCENT   = "#0f3460"
TEXT     = "#e0e0e0"
DIM      = "#888888"
GREEN    = "#4ade80"
RED      = "#f87171"

# Default slider values
DEFAULTS = {
    "margin":  80,
    "shift_x": 100,   # raw var (shift_x + 100)
    "shift_y": 100,
    "off_x":   0,
    "off_y":   0,
    "pitch_x": 254,   # ×10
    "pitch_y": 274,
}


class OhmVisionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Ohm-Vision Analyzer")
        self.window.configure(bg=BG)
        self.window.resizable(True, True)
        self.window.minsize(900, 600)

        self._fullscreen = False
        self._cfg = self._load_config()

        self.camera      = CameraLoader(camera_id=1, width=1280, height=720)
        self.engine      = ModelEngine(model_path="models/Yolo_v8n_pose_weights.onnx",
                                       model_type="yolov8")
        self.transformer = BreadboardWarper(output_width=810, output_height=540)
        self.grid_mapper = GridMapper(target_w=810, target_h=540)
        self.class_names = self.engine.engine.names
        self.show_grid   = False

        self._build_ui()
        self._apply_warp()
        self._apply_grid()

        self.window.bind("<q>",   lambda e: self.on_closing())
        self.window.bind("<Q>",   lambda e: self.on_closing())
        self.window.bind("<g>",   lambda e: self._toggle_grid())
        self.window.bind("<G>",   lambda e: self._toggle_grid())
        self.window.bind("<F11>", lambda e: self._toggle_fullscreen())
        self.window.bind("<Escape>", lambda e: self._exit_fullscreen())
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.camera.start()
        self.window.after(50, self._update_frame)

    # ── Config persistence ─────────────────────────────────────────────────────
    def _load_config(self):
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    data = json.load(f)
                    return {k: data.get(k, DEFAULTS[k]) for k in DEFAULTS}
            except Exception:
                pass
        return dict(DEFAULTS)

    def _save_config(self):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        state = {
            "margin":  self.var_margin.get(),
            "shift_x": self.var_shift_x.get(),
            "shift_y": self.var_shift_y.get(),
            "off_x":   self.var_off_x.get(),
            "off_y":   self.var_off_y.get(),
            "pitch_x": self.var_pitch_x.get(),
            "pitch_y": self.var_pitch_y.get(),
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(state, f, indent=2)

    # ── Fullscreen ─────────────────────────────────────────────────────────────
    def _toggle_fullscreen(self):
        self._fullscreen = not self._fullscreen
        self.window.attributes("-fullscreen", self._fullscreen)
        hint = "[Q] Quit   [G] Grid   [F11] Fullscreen   [Esc] Exit Fullscreen"
        self.hint_label.config(text=hint)

    def _exit_fullscreen(self):
        if self._fullscreen:
            self._fullscreen = False
            self.window.attributes("-fullscreen", False)

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.window, bg=ACCENT, height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Ohm-Vision Analyzer",
                 font=("Arial", 15, "bold"), bg=ACCENT, fg="white").pack(side="left", padx=16, pady=12)
        self.status_badge = tk.Label(hdr, text="  SEARCHING  ",
                                     font=("Arial", 10, "bold"),
                                     bg=RED, fg="white", padx=6)
        self.status_badge.pack(side="right", padx=16, pady=13)

        # Body
        body = tk.Frame(self.window, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # Left: canvas (expands with window)
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(left, bg="#000",
                                highlightthickness=2, highlightbackground=ACCENT)
        self.canvas.pack(fill="both", expand=True)

        self.hint_label = tk.Label(
            left,
            text="[Q] Quit   [G] Grid   [F11] Fullscreen   [Esc] Exit Fullscreen",
            font=("Consolas", 9), bg=BG, fg=DIM)
        self.hint_label.pack(pady=4)

        # Right: fixed-width control panel
        panel = tk.Frame(body, bg=PANEL_BG, width=220, relief="flat")
        panel.pack(side="right", fill="y", padx=(10, 0))
        panel.pack_propagate(False)

        tk.Label(panel, text="Controls", font=("Arial", 12, "bold"),
                 bg=PANEL_BG, fg=TEXT).pack(pady=(14, 6))

        self._divider(panel)
        self._section_label(panel, "Warp")
        self.var_margin  = tk.IntVar(value=self._cfg["margin"])
        self.var_shift_x = tk.IntVar(value=self._cfg["shift_x"])
        self.var_shift_y = tk.IntVar(value=self._cfg["shift_y"])
        self._slider(panel, "Margin",  self.var_margin,  0, 300, self._apply_warp)
        self._slider(panel, "Shift X", self.var_shift_x, 0, 200, self._apply_warp)
        self._slider(panel, "Shift Y", self.var_shift_y, 0, 200, self._apply_warp)

        self._divider(panel)
        self._section_label(panel, "Grid")
        self.var_off_x   = tk.IntVar(value=self._cfg["off_x"])
        self.var_off_y   = tk.IntVar(value=self._cfg["off_y"])
        self.var_pitch_x = tk.IntVar(value=self._cfg["pitch_x"])
        self.var_pitch_y = tk.IntVar(value=self._cfg["pitch_y"])
        self._slider(panel, "Off X",   self.var_off_x,   0, 200, self._apply_grid)
        self._slider(panel, "Off Y",   self.var_off_y,   0, 200, self._apply_grid)
        self._slider(panel, "Pitch X", self.var_pitch_x, 0, 500, self._apply_grid)
        self._slider(panel, "Pitch Y", self.var_pitch_y, 0, 500, self._apply_grid)

        self._divider(panel)

        btn_area = tk.Frame(panel, bg=PANEL_BG)
        btn_area.pack(fill="x", padx=12, pady=10)

        self.btn_grid = tk.Button(btn_area, text="Grid  OFF",
                                  command=self._toggle_grid,
                                  bg=ACCENT, fg="white",
                                  font=("Arial", 10), relief="flat",
                                  activebackground="#1a4a8a",
                                  cursor="hand2", height=2)
        self.btn_grid.pack(fill="x", pady=(0, 6))

        tk.Button(btn_area, text="Quit",
                  command=self.on_closing,
                  bg="#7f1d1d", fg="white",
                  font=("Arial", 10), relief="flat",
                  activebackground="#991b1b",
                  cursor="hand2", height=2).pack(fill="x")

        self._divider(panel)
        self.readout = tk.Label(panel, text="",
                                font=("Consolas", 8), bg=PANEL_BG,
                                fg=DIM, justify="left")
        self.readout.pack(padx=12, pady=6, anchor="w")

    # ── Widget helpers ─────────────────────────────────────────────────────────
    def _divider(self, parent):
        tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill="x", padx=8, pady=4)

    def _section_label(self, parent, text):
        tk.Label(parent, text=text, font=("Arial", 9, "bold"),
                 bg=PANEL_BG, fg=DIM).pack(anchor="w", padx=14, pady=(2, 0))

    def _slider(self, parent, label, var, lo, hi, cmd):
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill="x", padx=10, pady=1)
        tk.Label(row, text=label, width=7, anchor="w",
                 font=("Arial", 8), bg=PANEL_BG, fg=TEXT).pack(side="left")
        tk.Scale(row, variable=var, from_=lo, to=hi,
                 orient="horizontal", length=118, showvalue=False,
                 bg=PANEL_BG, fg=TEXT, troughcolor=ACCENT,
                 highlightthickness=0, sliderlength=14, bd=0,
                 command=lambda _: cmd()).pack(side="left")
        tk.Label(row, textvariable=var, width=4,
                 font=("Consolas", 8), bg=PANEL_BG, fg=GREEN).pack(side="left")

    # ── Callbacks ──────────────────────────────────────────────────────────────
    def _apply_warp(self):
        self.transformer.margin  = self.var_margin.get()
        self.transformer.shift_x = self.var_shift_x.get() - 100
        self.transformer.shift_y = self.var_shift_y.get() - 100

    def _apply_grid(self):
        self.grid_mapper.set_params(
            self.var_off_x.get(), self.var_off_y.get(),
            pitch_x=self.var_pitch_x.get() / 10.0,
            pitch_y=self.var_pitch_y.get() / 10.0,
        )

    def _toggle_grid(self):
        self.show_grid = not self.show_grid
        if self.show_grid:
            self.btn_grid.config(text="Grid  ON", bg="#166534")
        else:
            self.btn_grid.config(text="Grid  OFF", bg=ACCENT)

    # ── Main update loop ───────────────────────────────────────────────────────
    def _update_frame(self):
        frame = self.camera.get_frame()
        if frame is not None:
            success, warped, _ = self.transformer.process(frame)

            if success:
                # Predict on warped frame — same domain as training data
                results = self.engine.predict(warped)
                display = warped.copy()
                if self.show_grid:
                    display = self.grid_mapper.draw_grid_overlay(display)
                draw_results(display, results, self.class_names, matrix=None)
                cv2.putText(display, "BOARD OK",
                            (12, 32), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 220, 80), 2)
                self.status_badge.config(text="  BOARD OK  ", bg=GREEN, fg="#052e16")
            else:
                # Fallback: predict on raw frame when board not detected
                results = self.engine.predict(frame)
                display = cv2.resize(frame, (810, 540))
                draw_results(display, results, self.class_names, matrix=None)
                cv2.putText(display, "SEARCHING FOR ARUCO TAGS...",
                            (12, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (80, 80, 255), 2)
                self.status_badge.config(text="  SEARCHING  ", bg=RED, fg="white")

            # Fit inside canvas while preserving 810×540 aspect ratio (letterbox)
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw > 10 and ch > 10:
                src_h, src_w = display.shape[:2]
                scale  = min(cw / src_w, ch / src_h)
                new_w  = int(src_w * scale)
                new_h  = int(src_h * scale)
                resized = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                padded  = np.zeros((ch, cw, 3), dtype=np.uint8)
                x0 = (cw - new_w) // 2
                y0 = (ch - new_h) // 2
                padded[y0:y0 + new_h, x0:x0 + new_w] = resized
                display = padded

            t  = self.transformer
            gm = self.grid_mapper
            self.readout.config(
                text=(f"Margin  {t.margin}\n"
                      f"Shift X {t.shift_x:+d}\n"
                      f"Shift Y {t.shift_y:+d}\n"
                      f"Off X   {gm.offset_x}\n"
                      f"Off Y   {gm.offset_y}\n"
                      f"Pitch X {gm.pitch_x:.1f}\n"
                      f"Pitch Y {gm.pitch_y:.1f}"))

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        self.window.after(15, self._update_frame)

    def on_closing(self):
        self._save_config()
        self.camera.stop()
        self.window.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = OhmVisionApp(root)
    root.mainloop()
