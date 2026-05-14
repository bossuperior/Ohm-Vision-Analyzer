# ==========================================
# CONFIG: Reference Colors (HSV in OpenCV scale: H=0-179, S=0-255, V=0-255)
# ==========================================
import numpy as np

# ── ตรวจสอบค่า HSV ──────────────────────────────────────────────
# OpenCV H range = 0-179 (ครึ่งหนึ่งของ 360°)
# RED:    H ≈ 0-10  (และ 170-179 แบบ wrap-around)
# ORANGE: H ≈ 10-20
# YELLOW: H ≈ 20-30
# GREEN:  H ≈ 40-80
# BLUE:   H ≈ 100-130
# VIOLET: H ≈ 130-145   ← ค่าเดิม 68 ผิด! (68 คือ cyan-green)
# GRAY/WHITE/BLACK/SILVER: S ต่ำมาก
# ─────────────────────────────────────────────────────────────────

REF_COLORS = {
    'BLACK':  (0,    0,   30),
    'BROWN':  (17,  110, 100),   # H≈17 (warm brown), S moderate, V moderate
    'RED':    (5,   210, 180),
    'ORANGE': (15,  240, 220),
    'YELLOW': (28,  230, 220),
    'GOLD':   (22,  130, 150),
    'GREEN':  (60,  200, 180),
    'BLUE':   (115, 200, 190),
    'VIOLET': (135, 160, 160),
    'GRAY':   (0,    0,  120),
    'WHITE':  (0,    0,  230),
    'SILVER': (0,    30, 200),
}

COLOR_VALS = {
    'BLACK': 0, 'BROWN': 1, 'RED': 2, 'ORANGE': 3, 'YELLOW': 4,
    'GREEN': 5, 'BLUE': 6, 'VIOLET': 7, 'GRAY': 8, 'WHITE': 9,
    'GOLD': -1, 'SILVER': -2
}

BODY_LOWER = np.array([5, 50, 80])
BODY_UPPER = np.array([35, 180, 255])
