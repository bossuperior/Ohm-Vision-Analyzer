# ==========================================
# CONFIG: Reference Colors (Tuned Version)
# ==========================================
import numpy as np

REF_COLORS = {
    'BLACK': (0, 0, 30),
    'BROWN': (13, 110, 60),
    'RED': (35, 150, 170),
    'ORANGE': (20, 255, 255),
    'YELLOW': (85, 200, 200),
    'GOLD': (20, 140, 140),
    'GREEN': (110, 200, 200),
    'BLUE': (130, 200, 200),
    'VIOLET': (140, 180, 180),
    'GRAY': (0, 0, 100),
    'WHITE': (0, 0, 240),
    'SILVER': (0, 0, 200)
}

COLOR_VALS = {
    'BLACK': 0, 'BROWN': 1, 'RED': 2, 'ORANGE': 3, 'YELLOW': 4,
    'GREEN': 5, 'BLUE': 6, 'VIOLET': 7, 'GRAY': 8, 'WHITE': 9,
    'GOLD': -1, 'SILVER': -2
}

BODY_LOWER = np.array([5, 50, 80])
BODY_UPPER = np.array([35, 180, 255])