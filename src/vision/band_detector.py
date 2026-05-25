import cv2
import numpy as np
from src.vision.color_mapping import REF_COLORS, COLOR_VALS


class BandDetector:
    def __init__(self):
        pass

    # ──────────────────────────────────────────
    # 1. Color matching (weighted HSV distance)
    # ──────────────────────────────────────────
    def closest_color(self, hsv_val, exclude=None):
        h, s, v = hsv_val
        min_dist = float('inf')
        best_color = 'UNKNOWN'
        _exclude = exclude or set()

        for name, ref_hsv in REF_COLORS.items():
            if name in _exclude:
                continue
            diff_h = abs(h - ref_hsv[0])
            if diff_h > 90:
                diff_h = 180 - diff_h

            diff_s = abs(s - ref_hsv[1])
            diff_v = abs(v - ref_hsv[2])

            if name in ['BLACK', 'WHITE', 'GRAY', 'SILVER']:
                # achromatic: V เป็นตัวแยกหลัก, S ช่วยกัน pull ไปยัง achromatic ผิด
                weight_h, weight_s, weight_v = 1.0, 3.0, 5.0
            elif name == 'BROWN':
                weight_h, weight_s, weight_v = 3.0, 2.0, 1.0
            elif name in ['YELLOW']:
                weight_h, weight_s, weight_v = 10.0, 1.0, 1.0
            elif name in ['RED', 'GOLD', 'ORANGE']:
                weight_h, weight_s, weight_v = 3.0, 2.0, 2.0
            else:
                weight_h, weight_s, weight_v = 4.0, 1.0, 1.0

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

        # RULE 0: Direction — reverse only when first is tolerance AND last is NOT tolerance
        # ถ้าทั้งปลายทั้งสองเป็น gold-like อาจเป็น YELLOW ถูก misclassify → ไม่ reverse
        _TOL = ('GOLD', 'SILVER', 'BROWN')
        if (len(bands) >= 2
                and bands[0]['color'] in ('GOLD', 'SILVER')
                and bands[-1]['color'] not in _TOL):
            bands.reverse()

        for band in bands:
            h, s, _ = band['mean_hsv']
            color = band['color']

            # RULE 1: GOLD ที่สดผิดปกติมากคือ RED (threshold 200 เพราะ actual gold มี S≈150-160)
            if color == 'GOLD' and s > 200:
                band['color'] = 'RED'; band['val'] = COLOR_VALS['RED']

            # RULE 2: BROWN ที่ saturated สูงมากผิดปกติคือ RED (threshold 220 เพราะ actual brown มี S≈170-180)
            if band['color'] == 'BROWN' and s > 220:
                band['color'] = 'RED'; band['val'] = COLOR_VALS['RED']

            # RULE 3: VIOLET ต้องมี H > 80 (purple/violet ใน OpenCV H≈125-145)
            # ถ้า calibration ผิด (เช่น VIOLET ref H≈25) จะดึง orange-band เข้า VIOLET
            if band['color'] == 'VIOLET' and h < 80:
                nc = self.closest_color(band['mean_hsv'], exclude={'VIOLET', 'WHITE'})
                band['color'] = nc; band['val'] = COLOR_VALS.get(nc, -99)

            # RULE 5: WHITE ต้องมี S ต่ำ (achromatic) — ถ้า S > 60 แสดงว่า calibration ผิด
            if band['color'] == 'WHITE' and s > 60:
                nc = self.closest_color(band['mean_hsv'], exclude={'WHITE'})
                band['color'] = nc; band['val'] = COLOR_VALS.get(nc, -99)

        # RULE 4: Positional — Band 0-1 ต้องไม่เป็น GOLD/SILVER
        # GOLD ที่ตำแหน่ง digit มักเป็น YELLOW ที่ค่า S ต่ำ → แปลงเป็น YELLOW ไม่ใช่ RED
        for pos in (0, 1):
            if len(bands) > pos:
                if bands[pos]['color'] == 'GOLD':
                    bands[pos]['color'] = 'YELLOW'; bands[pos]['val'] = COLOR_VALS['YELLOW']
                elif bands[pos]['color'] == 'SILVER':
                    bands[pos]['color'] = 'RED';    bands[pos]['val'] = COLOR_VALS['RED']

        # RULE 6: Last band — RED/ORANGE/BROWN สุดท้ายให้ถือเป็น tolerance GOLD
        #         ยกเว้น 3-band ที่แถบสุดท้ายเป็น RED สด (multiplier x100)
        # RULE 6: ORANGE ท้ายสุดน่าจะเป็น GOLD misread
        # ไม่แตะ BROWN (ใช้เป็น 1% tolerance ใน 5-band) หรือ RED (อาจเป็น multiplier)
        if bands and bands[-1]['color'] == 'ORANGE':
            bands[-1]['color'] = 'GOLD'; bands[-1]['val'] = COLOR_VALS['GOLD']

        return bands

    # ──────────────────────────────────────────
    # 3a. Adaptive-threshold band detection
    # ──────────────────────────────────────────
    def detect_bands_adaptive(self, roi):
        """
        Adaptive threshold + connected components บน horizontal crop.
        รองรับทั้ง body สว่าง (beige) และ body มืด (น้ำเงิน):
          - Primary   : BINARY_INV — แถบมืดบน body สว่าง
          - Secondary : BINARY     — แถบสว่างบน body น้ำเงิน/มืด
        """
        h, w = roi.shape[:2]

        # ใช้เฉพาะส่วนกลาง (15-85%) หลีกเลี่ยง shadow ขอบโค้งของ body
        r0, r1 = int(h * 0.15), int(h * 0.85)
        center  = roi[r0:r1] if r1 > r0 + 4 else roi
        ch      = center.shape[0]

        gray    = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 2)

        block   = max(11, int(w * 0.15)) | 1
        k_open  = np.ones((max(3, int(ch * 0.60)), 1), np.uint8)
        k_close = np.ones((1, max(3, int(w * 0.04))), np.uint8)
        edge_guard = max(4, int(w * 0.09))
        min_gap    = max(4, int(w * 0.06))

        def _extract(thresh_img):
            m = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, k_open)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close, iterations=2)
            n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            out = []
            for i in range(1, n):
                bx, _, bw, bh, _ = st[i]
                cx = bx + bw // 2
                if bh >= ch * 0.50 and bw <= w * 0.35 and edge_guard <= cx <= w - edge_guard:
                    out.append({'x': cx, 'w': bw})
            return sorted(out, key=lambda k: k['x'])

        def _merge(cs):
            merged = []
            for c in cs:
                if merged and c['x'] - merged[-1]['x'] < min_gap:
                    if c['w'] > merged[-1]['w']:
                        merged[-1] = c
                else:
                    merged.append(c)
            return merged

        # ── Primary: dark bands on bright body ───────────────────────────────
        thresh_inv = cv2.adaptiveThreshold(blurred, 255,
                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 4)
        centers = _merge(_extract(thresh_inv))

        # ── Secondary: bright bands on dark/blue body ─────────────────────────
        # trigger เมื่อหาได้น้อยกว่า 3 band หรือ body มืด (median gray < 140)
        body_brightness = float(np.median(
            blurred[int(ch * 0.1):int(ch * 0.9), int(w * 0.2):int(w * 0.8)]))
        if len(centers) < 3 or body_brightness < 140:
            thresh_fwd = cv2.adaptiveThreshold(blurred, 255,
                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 4)
            extra_raw = _extract(thresh_fwd)
            extra = [c for c in extra_raw
                     if not any(abs(c['x'] - e['x']) < min_gap for e in centers)]
            if extra:
                centers = _merge(sorted(centers + extra, key=lambda k: k['x']))

        # Cap 5 bands — เก็บเฉพาะอันที่กว้างที่สุด (contrast มากสุด)
        if len(centers) > 5:
            centers = sorted(centers, key=lambda k: -k['w'])[:5]
            centers.sort(key=lambda k: k['x'])

        return centers

    # ──────────────────────────────────────────
    # 3b. Projection-based band detection (fallback)
    # ──────────────────────────────────────────
    MIN_CONTRAST = 28  # diffs_norm max ต่ำกว่านี้ = profile แบนเกิน ไม่ detect

    def _threshold_and_extract(self, diffs_norm, w):
        """Run Otsu + fallback thresholds + morph-close → band_centers list."""
        # ── Minimum contrast guard ────────────────────────────────────────────
        # ถ้า profile แบนมาก (เช่น body สีเดียวกับแถบ หรือภาพเบลอมาก) → คืนว่าง
        if int(np.max(diffs_norm)) < self.MIN_CONTRAST:
            return []

        diffs_2d = np.expand_dims(diffs_norm, axis=0)
        _, thresh = cv2.threshold(
            diffs_2d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        temp_cnts, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(temp_cnts) < 3:
            for thr in (20, 15, 10, 6, 4):
                _, thresh = cv2.threshold(diffs_2d, thr, 255, cv2.THRESH_BINARY)
                t_cnts, _ = cv2.findContours(
                    thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(t_cnts) >= 3:
                    break
        kernel = np.ones((5, 1), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh_img = thresh.reshape(1, -1)
        contours, _ = cv2.findContours(
            thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_band_w = max(2, int(w * 0.015))
        max_band_w = int(w * 0.40)
        # ── Edge suppression: ไม่นับแถบที่อยู่ใกล้ขอบ crop เกินไป ────────────
        edge_guard = max(4, int(w * 0.09))
        centers = []
        for cnt in contours:
            x, _, w_cnt, _ = cv2.boundingRect(cnt)
            cx = x + w_cnt // 2
            if min_band_w <= w_cnt <= max_band_w and edge_guard <= cx <= w - edge_guard:
                centers.append({'x': cx, 'w': w_cnt})
        centers.sort(key=lambda k: k['x'])
        return centers

    def _diffs_from_body(self, center, body_color, w, edge_w):
        """Compute normalized column-diff profile and zero margin."""
        col_medians = np.median(center, axis=0)
        diffs = np.linalg.norm(col_medians - body_color, axis=1)
        diffs_norm = cv2.normalize(diffs, None, 0, 255,
                                   cv2.NORM_MINMAX).astype(np.uint8)
        margin = max(int(w * 0.08), edge_w + 2)
        diffs_norm[:margin] = 0
        diffs_norm[-margin:] = 0
        return diffs_norm

    def _inpaint_body_ref(self, center, band_centers, w):
        """Inpaint ทับตำแหน่งแถบที่หาได้ → ได้ภาพ body สะอาด ใช้เป็น reference."""
        mask = np.zeros(center.shape[:2], dtype=np.uint8)
        for bc in band_centers:
            half = max(bc['w'] // 2 + 2, 3)
            x1 = max(0, bc['x'] - half)
            x2 = min(w, bc['x'] + half)
            mask[:, x1:x2] = 255
        if not mask.any():
            return center
        return cv2.inpaint(center, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    def detect_bands_projection(self, roi):
        h, w = roi.shape[:2]
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        roi_balanced = cv2.merge((l, a, b))
        roi_balanced = cv2.cvtColor(roi_balanced, cv2.COLOR_LAB2BGR)

        r0 = max(0, int(h * 0.25))
        r1 = min(h, int(h * 0.75))
        center = roi_balanced[r0:r1] if r1 > r0 else roi

        # ── body_color pass 1: inner-edge (ข้ามส่วนขอบที่อาจเป็น background) ──
        # crop_from_keypoints มี margin ~30px → ขอบนอกอาจเป็น breadboard ไม่ใช่ body
        # ใช้แถบ 15-25% จากขอบแต่ละด้าน ซึ่งน่าจะอยู่บน body end-cap (สีเบจ)
        edge_w     = max(2, int(w * 0.12))
        inner_s    = max(edge_w, int(w * 0.15))
        inner_e    = max(inner_s + 2, int(w * 0.25))
        left_px    = center[:, inner_s:inner_e].reshape(-1, 3).astype(np.float32)
        right_px   = center[:, w - inner_e:w - inner_s].reshape(-1, 3).astype(np.float32)
        left_hsv   = cv2.cvtColor(center[:, inner_s:inner_e], cv2.COLOR_BGR2HSV)
        right_hsv  = cv2.cvtColor(center[:, w - inner_e:w - inner_s], cv2.COLOR_BGR2HSV)
        lsat = np.mean(left_hsv[:, :, 1])
        rsat = np.mean(right_hsv[:, :, 1])
        body_color = np.median(left_px if lsat <= rsat else right_px, axis=0)

        diffs_norm    = self._diffs_from_body(center, body_color, w, edge_w)
        band_centers  = self._threshold_and_extract(diffs_norm, w)

        # ── body_color pass 2: min-sat จากส่วนกลาง (fallback) ──────
        # ค้นหา column ที่ saturation ต่ำที่สุดในบริเวณกลาง (body area)
        # เพื่อเลี่ยง background ที่ขอบ crop
        if len(band_centers) < 3:
            hsv_center  = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
            cs          = int(w * 0.20)
            ce          = int(w * 0.80)
            if ce > cs:
                col_sat = np.mean(hsv_center[:, cs:ce, 1], axis=0)
                sat_thr = np.percentile(col_sat, 20)
                low_rel = np.where(col_sat <= sat_thr)[0]
                low_cols = cs + low_rel
                if len(low_cols) > 0:
                    bc2 = np.median(
                        center[:, low_cols].reshape(-1, 3).astype(np.float32), axis=0)
                    d2  = self._diffs_from_body(center, bc2, w, edge_w)
                    bc2_centers = self._threshold_and_extract(d2, w)
                    if len(bc2_centers) > len(band_centers):
                        band_centers = bc2_centers

        # ── body_color pass 3: inpainting refinement ──────────────
        # inpaint ทับแถบที่หาได้บางส่วน → body สะอาด → re-detect
        if 0 < len(band_centers) < 3:
            body_img = self._inpaint_body_ref(center, band_centers, w)
            # ใช้ inner-edge ของภาพ inpainted เป็น body reference (ไม่ใช้ global median)
            li  = body_img[:, inner_s:inner_e].reshape(-1, 3).astype(np.float32)
            ri  = body_img[:, w - inner_e:w - inner_s].reshape(-1, 3).astype(np.float32)
            lhi = cv2.cvtColor(body_img[:, inner_s:inner_e], cv2.COLOR_BGR2HSV)
            rhi = cv2.cvtColor(body_img[:, w - inner_e:w - inner_s], cv2.COLOR_BGR2HSV)
            ls3 = np.mean(lhi[:, :, 1])
            rs3 = np.mean(rhi[:, :, 1])
            bc3 = np.median(li if ls3 <= rs3 else ri, axis=0)
            d3  = self._diffs_from_body(center, bc3, w, edge_w)
            bc3_centers = self._threshold_and_extract(d3, w)
            if len(band_centers) < len(bc3_centers) <= 5:
                band_centers = bc3_centers

        return band_centers
