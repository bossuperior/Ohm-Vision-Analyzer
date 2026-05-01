import cv2
import numpy as np
from src.vision.color_mapping import REF_COLORS, COLOR_VALS


class BandDetector:
    def __init__(self):
        pass

    # ──────────────────────────────────────────
    # 1. Color matching (weighted HSV distance)
    # ──────────────────────────────────────────
    def closest_color(self, hsv_val):
        h, s, v = hsv_val
        min_dist = float('inf')
        best_color = 'UNKNOWN'

        for name, ref_hsv in REF_COLORS.items():
            diff_h = abs(h - ref_hsv[0])
            if diff_h > 90:
                diff_h = 180 - diff_h   # Hue wrap-around (RED อยู่ทั้ง H≈0 และ H≈179)

            diff_s = abs(s - ref_hsv[1])
            diff_v = abs(v - ref_hsv[2])

            # ── Weight ตามประเภทสี ──────────────────────────────────
            if name in ['BLACK', 'WHITE', 'GRAY', 'SILVER']:
                # Achromatic: V สำคัญที่สุด, H แทบไม่มีความหมาย
                weight_h, weight_s, weight_v = 0.5, 2.0, 5.0
            elif name in ['BROWN']:
                # Brown: H + V สำคัญ (dark orange-ish)
                weight_h, weight_s, weight_v = 3.0, 2.0, 4.0
            elif name in ['GOLD', 'SILVER']:
                # Metallic: S ต่ำ, V กลาง
                weight_h, weight_s, weight_v = 2.0, 3.0, 3.0
            elif name in ['RED', 'ORANGE']:
                # Red/Orange: H สำคัญมาก (แยกกัน), S สูง
                weight_h, weight_s, weight_v = 4.0, 2.0, 1.5
            elif name in ['YELLOW']:
                weight_h, weight_s, weight_v = 8.0, 1.5, 1.0
            elif name in ['VIOLET']:
                # Violet: H สำคัญมาก (H≈138 ต้องแยกจาก BLUE H≈110)
                weight_h, weight_s, weight_v = 5.0, 1.5, 1.0
            else:
                weight_h, weight_s, weight_v = 4.0, 1.5, 1.0

            dist = np.sqrt(
                (diff_h * weight_h) ** 2 +
                (diff_s * weight_s) ** 2 +
                (diff_v * weight_v) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                best_color = name

        return best_color

    # ──────────────────────────────────────────
    # 2. Post-processing: แก้สีที่ match ผิดบ่อย
    # ──────────────────────────────────────────
    def fix_false_colors(self, bands):
        if not bands:
            return bands

        for band in bands:
            h, s, v = band['mean_hsv']

            # GOLD ที่มี Saturation สูงมากจริงๆ คือ ORANGE
            if band['color'] == 'GOLD' and s > 185:
                band['color'] = 'ORANGE'
                band['val'] = COLOR_VALS['ORANGE']

            # YELLOW ที่ถูก desaturate กล้องจนเหมือน GOLD
            if band['color'] == 'GOLD' and s < 130 and v > 195:
                band['color'] = 'YELLOW'
                band['val'] = COLOR_VALS['YELLOW']

            # BROWN ที่ V สูงมากผิดปกติ อาจเป็น ORANGE
            if band['color'] == 'BROWN' and v > 200 and s > 170:
                band['color'] = 'ORANGE'
                band['val'] = COLOR_VALS['ORANGE']

            # GRAY ที่ V สูงมาก คือ WHITE
            if band['color'] == 'GRAY' and v > 200 and s < 20:
                band['color'] = 'WHITE'
                band['val'] = COLOR_VALS['WHITE']

            # VIOLET ที่มี H ต่ำเกินไป (< 90) ไม่น่าใช่ violet จริง
            if band['color'] == 'VIOLET' and h < 90:
                # อาจเป็น BLUE แทน
                if h >= 80:
                    band['color'] = 'BLUE'
                    band['val'] = COLOR_VALS['BLUE']

        # Tolerance bands (GOLD/SILVER) ต้องอยู่ท้ายเสมอ
        if bands and bands[0]['color'] in ['GOLD', 'SILVER']:
            bands.reverse()

        # ถ้า band แรกยัง GOLD (หลัง reverse) คือ Yellow ที่ถูก map ผิด
        if bands and bands[0]['color'] == 'GOLD':
            bands[0]['color'] = 'YELLOW'
            bands[0]['val'] = COLOR_VALS['YELLOW']

        return bands

    # ──────────────────────────────────────────
    # 3. Band location detection (projection)
    # ──────────────────────────────────────────
    def detect_bands_projection(self, roi):
        h, w = roi.shape[:2]

        # ── Estimate body color จากขอบซ้าย-ขวา (ไม่ใช่ทั้งภาพ) ──
        # เหตุผล: ถ้าใช้ median ของทั้งภาพ แถบสีจะ bias body color
        edge_frac = 0.12
        left_w  = max(1, int(w * edge_frac))
        right_w = max(1, int(w * edge_frac))
        left_sample  = roi[:, :left_w].reshape(-1, 3)
        right_sample = roi[:, w - right_w:].reshape(-1, 3)
        edge_samples = np.vstack([left_sample, right_sample])
        body_color = np.median(edge_samples, axis=0)

        # ── Column difference projection ──────────────────────────
        col_medians = np.median(roi, axis=0)
        diffs = np.linalg.norm(col_medians - body_color, axis=1)
        diffs_norm = cv2.normalize(diffs, None, 0, 255,
                                   cv2.NORM_MINMAX).astype(np.uint8)

        # ตัด margin ขอบซ้าย-ขวาออก (กัน false positive จาก lighting gradient)
        margin = max(int(w * 0.08), left_w + 2)
        diffs_norm[:margin] = 0
        diffs_norm[-margin:] = 0

        diffs_2d = np.expand_dims(diffs_norm, axis=0)

        # ── Thresholding ──────────────────────────────────────────
        _, thresh = cv2.threshold(
            diffs_2d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        temp_cnts, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ถ้า Otsu ให้ผลน้อยกว่า 3 bands ลอง fixed threshold แทน
        if len(temp_cnts) < 3:
            _, thresh = cv2.threshold(diffs_2d, 18, 255, cv2.THRESH_BINARY)

        # Morphological close: เชื่อม fragment เล็กๆ ของแถบเดียวกัน
        kernel = np.ones((1, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        thresh_img = thresh.reshape(1, -1)
        contours, _ = cv2.findContours(
            thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ── Filter by size ────────────────────────────────────────
        min_band_w = max(2, int(w * 0.02))   # แถบต้องกว้างอย่างน้อย 2% ของภาพ
        max_band_w = int(w * 0.35)            # แถบไม่ควรกว้างเกิน 35% (น่าจะไม่ใช่แถบ)

        band_centers = []
        for cnt in contours:
            x, _, w_cnt, _ = cv2.boundingRect(cnt)
            if w_cnt < min_band_w or w_cnt > max_band_w:
                continue
            center_x = x + w_cnt // 2
            band_centers.append({'x': center_x, 'w': w_cnt})

        band_centers.sort(key=lambda k: k['x'])
        return band_centers
