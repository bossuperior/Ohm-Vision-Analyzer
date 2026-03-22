REF_COLORS = {
    'BLACK': (0, 0, 30),
    'BROWN': (13, 110, 60),

    # RED: ตั้งค่ากลางๆ ไว้ที่ 35 เพื่อดักจับทั้งแดงสดและแดงเพี้ยนส้ม
    'RED': (35, 150, 170),

    'ORANGE': (20, 255, 255),

    # YELLOW: ดันหนีไปไกลๆ (85) เพื่อไม่ให้แย่งซีนสีแดงที่เพี้ยน
    'YELLOW': (85, 200, 200),

    # GOLD: ตั้งค่าให้ซีดกว่าแดง (S=140) และมืดกว่าแดง
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

# ช่วงสีลำตัว (Body Color) สำหรับการตรวจจับเบื้องต้น
BODY_LOWER = np.array([5, 50, 80])
BODY_UPPER = np.array([35, 180, 255])


# ==========================================
# 2. Helper Functions
# ==========================================

def closest_color(hsv_val):
    """หาคู่สีที่ใกล้ที่สุด โดยมีการถ่วงน้ำหนัก (Weighted Distance)"""
    h, s, v = hsv_val
    min_dist = float('inf')
    best_color = 'UNKNOWN'

    for name, ref_hsv in REF_COLORS.items():
        diff_h = abs(h - ref_hsv[0])
        if diff_h > 90: diff_h = 180 - diff_h

        diff_s = abs(s - ref_hsv[1])
        diff_v = abs(v - ref_hsv[2])

        # --- Weighting Logic (ปรับจูนแล้ว) ---
        if name in ['BLACK', 'WHITE', 'GRAY', 'SILVER', 'BROWN']:
            # กลุ่มนี้เน้นความสว่าง (V)
            weight_h, weight_s, weight_v = 1.0, 1.0, 5.0

        elif name in ['YELLOW']:
            # Yellow ต้อง Hue ตรงจริงๆ ห้ามมั่ว
            weight_h, weight_s, weight_v = 10.0, 1.0, 1.0

        elif name in ['RED', 'GOLD', 'ORANGE']:
            # Red vs Gold: Hue ใกล้กันมาก ให้ดู Saturation/Value ช่วย
            weight_h, weight_s, weight_v = 3.0, 2.0, 2.0

        else:
            weight_h, weight_s, weight_v = 4.0, 1.0, 1.0

        dist = np.sqrt((diff_h * weight_h) ** 2 + (diff_s * weight_s) ** 2 + (diff_v * weight_v) ** 2)

        if dist < min_dist:
            min_dist = dist
            best_color = name

    return best_color


def fix_false_colors(bands):
    """Logic Fix: แก้ไขสีตามกฎความเป็นจริง (Heuristic Rules)"""
    if not bands: return bands

    for i, band in enumerate(bands):
        h, s, v = band['mean_hsv']
        color = band['color']

        # RULE 1: แก้ Saturated GOLD -> RED (สีทองต้องซีด ถ้าสดคือแดง)
        if color == 'GOLD':
            if s > 130:
                band['color'] = 'RED';
                band['val'] = 2
            elif v < 100:  # ถ้ามืดเกินไป คือ Brown
                band['color'] = 'BROWN';
                band['val'] = 1

        # RULE 2: แก้ Saturated BROWN -> RED (น้ำตาลต้องตุ่น ถ้าสดคือแดง)
        if color == 'BROWN' and s > 150:
            band['color'] = 'RED';
            band['val'] = 2

        # RULE 3: แก้สีเพี้ยน (Yellow/Green/Blue -> RED) ใน Band 1-2
        if i < 2 and color in ['YELLOW', 'GREEN', 'BLUE']:
            if s > 80 and v > 80:
                # print(f"  [Logic Fix] Band {i+1} detected as {color} (Hue Shift). Changing to RED.")
                band['color'] = 'RED';
                band['val'] = 2

    # RULE 4: Positional Check (Band 1 & 2 ห้ามเป็น Gold/Silver)
    if bands[0]['color'] in ['GOLD', 'SILVER']:
        bands[0]['color'] = 'RED';
        bands[0]['val'] = 2
    if len(bands) >= 2 and bands[1]['color'] in ['GOLD', 'SILVER']:
        bands[1]['color'] = 'RED';
        bands[1]['val'] = 2

    # RULE 5: Last Band Logic (Tolerance vs Multiplier)
    last_band = bands[-1]

    # ถ้าเจอ 3 แถบ และแถบสุดท้ายดูเหมือนแดงเข้ม/สด อย่าเปลี่ยนเป็น Gold (มันคือ Multiplier x100)
    is_vivid_red = (last_band['mean_hsv'][1] > 100)
    if len(bands) == 3 and last_band['color'] == 'RED' and is_vivid_red:
        pass  # ปล่อยไว้เป็น RED

    # กรณีอื่นๆ ถ้าแถบสุดท้ายเป็น RED/ORANGE/BROWN ให้เปลี่ยนเป็น Tolerance (Gold)
    elif last_band['color'] in ['RED', 'ORANGE', 'BROWN']:
        last_band['color'] = 'GOLD';
        last_band['val'] = -1

    return bands