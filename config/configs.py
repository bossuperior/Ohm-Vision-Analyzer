import os
import json
from src.vision import color_mapping

CONFIG_PATH = "config/ui_state.json"

COLOR_NAMES = [
    'BLACK', 'BROWN', 'RED', 'ORANGE', 'YELLOW', 'GOLD',
    'GREEN', 'BLUE', 'VIOLET', 'GRAY', 'WHITE', 'SILVER',
]

DEFAULTS = {
    "margin":  80,
    "shift_x": 100,
    "shift_y": 100,
    "off_x":   0,
    "off_y":   0,
    "pitch_x": 254,
    "pitch_y": 274,
}

# ── Pose model ────────────────────────────────────────────────────────────────
POSE_BACKEND = 'yolo'
POSE_MODEL   = 'models/Pose Model/Yolo_v8s/Yolo_v8s_pose_weights.onnx'
POSE_CONF    = 0.5
POSE_IOU     = 0.45

# ── Classification model (YOLO cls — จำแนกค่าความต้านทานจากภาพ crop ตัวต้านทาน)
CLS_BACKEND  = 'yolo_cls'
CLS_MODEL    = 'models/Cls Models/Yolo_v8n_cls.pt'
CLS_DEVICE   = 'cpu'

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_ID     = 0
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720

# ── Pose model class IDs (2-class: resistor=0, wire=1) ───────────────────────
CLS_RESISTOR = 0   # resistor → crop ระหว่าง 2 keypoints → classify ค่าความต้านทาน + topology
CLS_WIRE     = 1   # wire → topology เท่านั้น (node merging)

# ── Debug ─────────────────────────────────────────────────────────────────────
DEBUG_CROPS     = True               # บันทึก crop + annotated debug image ทุกครั้งที่ classify
DEBUG_CROP_DIR  = "debug/crops"      # folder ที่บันทึก (สร้างอัตโนมัติ)

# Theme colors (can be overridden by config)
BG       = "#1a1a2e"
PANEL_BG = "#16213e"
ACCENT   = "#0f3460"
TEXT     = "#e0e0e0"
DIM      = "#888888"
GREEN    = "#4ade80"
RED      = "#f87171"

def load_ui_state() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                data = json.load(f)
            return {k: data.get(k, DEFAULTS[k]) for k in DEFAULTS}
        except Exception:
            pass
    return dict(DEFAULTS)


def load_color_refs():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                data = json.load(f)
            for name, hsv in data.get("color_refs", {}).items():
                if name in color_mapping.REF_COLORS:
                    color_mapping.REF_COLORS[name] = tuple(int(v) for v in hsv)
        except Exception:
            pass


def save_ui_state(vals: dict):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    state = {**vals, "color_refs": {k: list(v) for k, v in color_mapping.REF_COLORS.items()}}
    with open(CONFIG_PATH, "w") as f:
        json.dump(state, f, indent=2)
