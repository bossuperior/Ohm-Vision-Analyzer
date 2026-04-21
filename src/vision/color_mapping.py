# ==========================================
# CONFIG: Reference Colors (Tuned Version)
# ==========================================
import numpy as np

REF_COLORS = {
    'BLACK': (0, 0, 45),
    'BROWN': (30, 116, 179),
    'RED': (2, 185, 155),
    'ORANGE': (13, 200, 190),
    'YELLOW': (32, 176, 205),
    'GOLD': (27, 117, 200),
    'GREEN': (60, 190, 150),
    'BLUE': (112, 190, 150),
    'VIOLET': (68, 97, 186),
    'GRAY': (0, 15, 140),
    'WHITE': (0, 10, 220),
    'SILVER': (0, 18, 185),
}

COLOR_VALS = {
    'BLACK': 0, 'BROWN': 1, 'RED': 2, 'ORANGE': 3, 'YELLOW': 4,
    'GREEN': 5, 'BLUE': 6, 'VIOLET': 7, 'GRAY': 8, 'WHITE': 9,
    'GOLD': -1, 'SILVER': -2
}

BODY_LOWER = np.array([5, 50, 80])
BODY_UPPER = np.array([35, 180, 255])
