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

# Class IDs — must match model training order
CLS_RESISTOR_LEAD   = 0   # "resistor"     — ขา (leads) → circuit topology
CLS_RESISTOR_4B     = 1   # "Resistor_4B"  — body 4-band → color reading
CLS_RESISTOR_5B     = 2   # "Resistor_5B"  — body 5-band → color reading
CLS_WIRE            = 3   # "wire"         → node merging
CLS_RESISTOR_BODIES = {CLS_RESISTOR_4B, CLS_RESISTOR_5B}

#Theme colors (can be overridden by config)
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
