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
    'BLACK':  (0,   0,   40),    # V ต่ำมาก, S ต่ำ
    'BROWN':  (12, 140, 130),    # orange-brown, H≈10-15, V ไม่สูงมาก
    'RED':    (2,  200, 150),    # H≈0-5, S สูง
    'ORANGE': (14, 210, 190),    # H≈12-18
    'YELLOW': (28, 200, 210),    # H≈25-30
    'GOLD':   (25, 120, 190),    # metallic gold: H≈25, S ต่ำกว่า YELLOW
    'GREEN':  (60, 200, 140),    # H≈55-65
    'BLUE':   (110, 200, 140),   # H≈105-120
    'VIOLET': (138, 130, 160),   # H≈130-145  ← แก้จาก 68 (ผิด) เป็น 138
    'GRAY':   (0,   18, 140),    # S ต่ำมาก, V กลาง
    'WHITE':  (0,   12, 220),    # S ต่ำมาก, V สูง
    'SILVER': (0,   20, 180),    # คล้าย GRAY แต่ V สูงกว่า
}

COLOR_VALS = {
    'BLACK': 0, 'BROWN': 1, 'RED': 2, 'ORANGE': 3, 'YELLOW': 4,
    'GREEN': 5, 'BLUE': 6, 'VIOLET': 7, 'GRAY': 8, 'WHITE': 9,
    'GOLD': -1, 'SILVER': -2
}

BODY_LOWER = np.array([5, 50, 80])
BODY_UPPER = np.array([35, 180, 255])
