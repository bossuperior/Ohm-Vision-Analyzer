import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from src.vision.band_detector import BandDetector
from src.vision.color_mapping import COLOR_VALS, REF_COLORS

_BAND_BGR = {
    'BLACK':  (30,  30,  30),   'BROWN':  (30,  80, 160),
    'RED':    (0,   0,  220),   'ORANGE': (0,  140, 255),
    'YELLOW': (0,  220, 220),   'GOLD':   (0,  190, 200),
    'GREEN':  (0,  160,  50),   'BLUE':   (220,  80,   0),
    'VIOLET': (180,  0,  160),  'GRAY':   (140, 140, 140),
    'WHITE':  (240, 240, 240),  'SILVER': (180, 180, 200),
    'UNKNOWN':(80,  80,  80),
}


class BandReader:
    def __init__(self):
        self.detector = BandDetector()

    # ── 0a. Affine-crop จาก body class (Resistor_4B/5B) ─────────────────────────
    # ใช้ first → last visible keypoint (band positions) เป็น axis
    # ครอบตัวถังทั้งหมดตั้งแต่แถบแรกถึงแถบสุดท้าย
    def crop_from_body_keypoints(self, warped: np.ndarray,
                                 kp_data: np.ndarray) -> np.ndarray:
        if kp_data is None or len(kp_data) < 2:
            return None
        visible = [kp for kp in kp_data if kp[2] >= 0.5]
        if len(visible) < 2:
            return None
        return self._affine_crop(warped, visible[0][:2], visible[-1][:2])

    # ── 0b. Affine-crop: ดึง rotated rectangle ตาม Body_2→Body_3 ──────────────
    def crop_from_keypoints(self, warped: np.ndarray,
                            kp_data: np.ndarray) -> np.ndarray:
        if kp_data is None or len(kp_data) < 4:
            return None
        if kp_data[2][2] < 0.5 or kp_data[3][2] < 0.5:
            return None
        return self._affine_crop(warped, kp_data[2][:2], kp_data[3][:2])

    # ── 0c. Shared affine crop implementation ────────────────────────────────
    def _affine_crop(self, warped: np.ndarray,
                     p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        p2 = p2.astype(float)
        p3 = p3.astype(float)
        dx, dy = p3[0] - p2[0], p3[1] - p2[1]
        length = float(np.hypot(dx, dy))
        if length < 10:
            return None

        cx, cy   = (p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2
        cos_a, sin_a = dx / length, dy / length
        perp_pad = int(np.clip(length * 0.20, 12, 30))

        out_w = int(length) + 60
        out_h = perp_pad * 2

        tx = cx - (out_w / 2) * cos_a + (out_h / 2) * sin_a
        ty = cy - (out_w / 2) * sin_a - (out_h / 2) * cos_a
        M  = np.float32([[cos_a, -sin_a, tx],
                         [sin_a,  cos_a, ty]])

        crop = cv2.warpAffine(warped, M, (out_w, out_h),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        return crop if crop.size > 0 else None

    # ── 1. ตัดขอบซ้าย-ขวาเล็กน้อย (กัน lead ที่อาจเลยเข้ามา) ───────────────
    def _shrink_roi(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        x1, x2 = int(w * 0.10), int(w * 0.90)
        y1, y2 = int(h * 0.05), int(h * 0.95)
        return img[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else img

    # ── 1b. Unsharp mask sharpening ──────────────────────────────────────────
    def _sharpen(self, img: np.ndarray, amount: float = 1.6, sigma: float = 2.0) -> np.ndarray:
        """
        Unsharp mask: result = img + amount*(img − blur)
        ทำให้ edge ระหว่างแถบสีและ body คมขึ้น ช่วยกล้อง focus ไม่ชัด
        """
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
        return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)

    # ── 1c. Body isolation: หา x-range ของ resistor body ภายใน crop ───────────
    def _isolate_body(self, roi: np.ndarray):
        """
        Brightness projection บน column แยก body (สว่าง) จาก breadboard (มืด).
        คืน (body_roi, x_offset_ใน_roi_เดิม)
        """
        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 3)

        # ใช้เฉพาะแถวกลาง 20-80% หลีกเลี่ยง background ด้านบน/ล่าง
        r0, r1 = int(h * 0.20), int(h * 0.80)
        col_bright = np.mean(blur[r0:r1], axis=0).astype(float)

        # Otsu บน 1D profile
        profile_u8 = np.clip(col_bright, 0, 255).astype(np.uint8).reshape(1, -1)
        thr, _ = cv2.threshold(profile_u8, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        body_cols = np.where(col_bright > float(thr))[0]

        # fallback: ถ้าหา body ไม่ได้ หรือแคบเกิน ใช้ full width
        if len(body_cols) < int(w * 0.20):
            return roi, 0

        x0 = max(0, int(body_cols[0]) - 4)
        x1 = min(w, int(body_cols[-1]) + 5)
        if x1 - x0 < 20:
            return roi, 0

        return roi[:, x0:x1], x0

    # ── 2a. Gray World AWB ────────────────────────────────────────────────────
    def _awb(self, img: np.ndarray) -> np.ndarray:
        result  = img.astype(np.float32)
        all_avg = np.mean(result)
        for ch in range(3):
            avg = np.mean(result[:, :, ch])
            if avg > 0:
                result[:, :, ch] = np.clip(result[:, :, ch] * (all_avg / avg), 0, 255)
        return result.astype(np.uint8)

    # ── 2b. CLAHE contrast enhancement ───────────────────────────────────────
    _clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))

    def _clahe_enhance(self, img: np.ndarray) -> np.ndarray:
        """เพิ่ม contrast บน L channel (LAB) — ช่วย band ที่ overexposed / low contrast"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ── 2c. Board-referenced WB ───────────────────────────────────────────────
    BOARD_BEIGE_BGR = np.array([185, 195, 200], dtype=np.float32)

    def _board_ref_wb(self, img: np.ndarray) -> np.ndarray:
        _, w    = img.shape[:2]
        edge_w  = max(1, int(w * 0.10))
        left    = img[:, :edge_w].reshape(-1, 3).astype(np.float32)
        right   = img[:, w - edge_w:].reshape(-1, 3).astype(np.float32)
        measured = np.median(np.vstack([left, right]), axis=0)
        if np.any(measured < 30) or np.any(measured > 240):
            return img
        scale = np.clip(self.BOARD_BEIGE_BGR / (measured + 1e-6), 0.5, 2.0)
        return np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    # ── 3. Public entry ───────────────────────────────────────────────────────
    def calculate(self, cropped: np.ndarray,
                  band_count_hint: int = None) -> Tuple[str, List[Dict], float]:
        if cropped is None or cropped.size == 0:
            return "Unknown", [], 0.0
        sharp    = self._sharpen(cropped)
        body_roi, body_x0 = self._isolate_body(sharp)
        roi = self._awb(body_roi)
        roi = self._clahe_enhance(roi)
        roi = self._shrink_roi(roi)
        bands = self._scan_bands(roi)
        if not bands:
            return "Unknown", [], 0.0
        bands = self.detector.fix_false_colors(bands)
        bands = self._apply_count_hint(bands, band_count_hint)
        bands, known_ohms, known_tol = self._resolve_by_rules(bands, band_count_hint)
        x_off = body_x0 + int(body_roi.shape[1] * 0.10)
        for b in bands:
            b['x_crop'] = b['x'] + x_off
        if known_ohms is not None:
            return self._label_from_ohms(known_ohms, known_tol), bands, known_ohms
        return self._calculate_ohms(bands)

    # ── 3a. Rule-based constraint satisfaction ───────────────────────────────
    # All valid color combinations — derived from conditional rules (bidirectional)
    # 5B: first→second, second→multiplier, third=BLACK fixed, tolerance=BROWN fixed
    # 4B: first→second, second→multiplier, tolerance=GOLD fixed
    _5B_COMBOS = (
        (('BROWN',  'BLACK',  'BLACK', 'BROWN',  'BROWN'),  {'ohms':   1_000, 'tol': '1%'}),
        (('BROWN',  'BLACK',  'BLACK', 'ORANGE', 'BROWN'),  {'ohms': 100_000, 'tol': '1%'}),
        (('BLUE',   'GRAY',   'BLACK', 'BROWN',  'BROWN'),  {'ohms':   6_800, 'tol': '1%'}),
        (('WHITE',  'BROWN',  'BLACK', 'BLACK',  'BROWN'),  {'ohms':     910, 'tol': '1%'}),
        (('GRAY',   'RED',    'BLACK', 'BLACK',  'BROWN'),  {'ohms':     820, 'tol': '1%'}),
        (('YELLOW', 'BROWN',  'BLACK', 'SILVER', 'BROWN'),  {'ohms':     4.1, 'tol': '1%'}),
    )
    _4B_COMBOS = (
        (('ORANGE', 'WHITE',  'BROWN',  'GOLD'),  {'ohms':    390, 'tol': '5%'}),
        (('YELLOW', 'VIOLET', 'BROWN',  'GOLD'),  {'ohms':    470, 'tol': '5%'}),
        (('YELLOW', 'VIOLET', 'RED',    'GOLD'),  {'ohms':  4_700, 'tol': '5%'}),
        (('BROWN',  'BLACK',  'BROWN',  'GOLD'),  {'ohms':    100, 'tol': '5%'}),
        (('BROWN',  'BLACK',  'RED',    'GOLD'),  {'ohms':  1_000, 'tol': '5%'}),
        (('BROWN',  'BLACK',  'ORANGE', 'GOLD'),  {'ohms': 10_000, 'tol': '5%'}),
    )

    _TOL_4B   = {'GOLD'}
    _TOL_5B   = {'BROWN'}
    _FIRST_4B = {'BROWN', 'ORANGE', 'YELLOW'}
    _FIRST_5B = {'BROWN', 'YELLOW', 'BLUE', 'WHITE', 'GRAY'}

    def _apply_count_hint(self, bands: List[Dict],
                          hint: int) -> List[Dict]:
        if not bands or len(bands) < 2:
            return bands

        first_c = bands[0]['color']
        last_c  = bands[-1]['color']

        # ── Step 1: ตรวจ reading direction ──────────────────────────────────
        tol_on_left    = first_c in self._TOL_4B or first_c in self._TOL_5B
        start_on_right = last_c  in self._FIRST_4B or last_c in self._FIRST_5B
        if tol_on_left and start_on_right:
            bands   = list(reversed(bands))
            first_c, last_c = last_c, first_c

        # ── Step 2: กำหนด expected band count จากสีแถบสุดท้าย ───────────────
        if last_c in self._TOL_4B:
            expected = 4
        elif last_c in self._TOL_5B:
            expected = 5
        else:
            expected = hint

        # ── Step 3: ตัดแถบส่วนเกิน ──────────────────────────────────────────
        if expected is not None and len(bands) > expected:
            trimmed = sorted(bands, key=lambda b: -b.get('w', 1))[:expected]
            bands   = sorted(trimmed, key=lambda b: b['x'])

        return bands

    def _resolve_by_rules(self, bands: List[Dict],
                          n_bands: int) -> Tuple[List[Dict], Optional[float], str]:
        """
        Constraint satisfaction ด้วย HSV distance scoring:
        - เปรียบเทียบ detected bands กับทุก valid combination พร้อมกัน
        - ทุก position cross-check กันอัตโนมัติ (bidirectional)
        - คืน (corrected_bands, known_ohms) — ohms จาก dict เฉลย, None ถ้าไม่มี combo
        """
        combos = self._5B_COMBOS if n_bands == 5 else self._4B_COMBOS
        if not bands or not combos:
            return bands, None, '?%'

        # score เฉพาะตำแหน่งที่มี band จริง — ไม่ reject ถ้าจำนวนไม่ตรง
        n_score      = min(len(bands), n_bands) if n_bands else len(bands)
        detected_hsv = [b['mean_hsv'] for b in bands[:n_score]]

        def hsv_dist(hsv, color: str) -> float:
            ref = REF_COLORS.get(color)
            if ref is None:
                return 999.0
            dh = min(abs(hsv[0] - ref[0]), 180 - abs(hsv[0] - ref[0]))
            ds = abs(hsv[1] - ref[1])
            dv = abs(hsv[2] - ref[2])
            # Per-color weights mirror those in BandDetector.closest_color
            if color in ('BLACK', 'WHITE', 'GRAY', 'SILVER'):
                return dh * 1.0 + ds * 3.0 + dv * 5.0
            if color == 'BROWN':
                return dh * 3.0 + ds * 2.0 + dv * 1.0
            if color == 'YELLOW':
                return dh * 10.0 + ds * 1.0 + dv * 1.0
            if color in ('RED', 'GOLD', 'ORANGE'):
                return dh * 3.0 + ds * 2.0 + dv * 2.0
            return dh * 4.0 + ds * 1.0 + dv * 1.0

        best_combo, best_meta, best_score = None, {}, float('inf')
        for combo_colors, meta in combos:
            score = sum(hsv_dist(detected_hsv[i], combo_colors[i]) for i in range(n_score))
            score -= sum(8.0 for i in range(n_score) if bands[i]['color'] == combo_colors[i])
            if score < best_score:
                best_score, best_combo, best_meta = score, combo_colors, meta

        if best_combo is None:
            return bands, None, '?%'

        # แก้สีแถบตาม combo ที่ match ได้ (เฉพาะตำแหน่งที่มีจริง)
        corrected = []
        for i, band in enumerate(bands):
            b = dict(band)
            if i < len(best_combo):
                b['color'] = best_combo[i]
                b['val']   = COLOR_VALS.get(best_combo[i], -99)
            corrected.append(b)
        return corrected, best_meta.get('ohms'), best_meta.get('tol', '?%')

    # ── 3b. Debug annotation ──────────────────────────────────────────────────
    def annotate_crop(self, crop: np.ndarray, bands: List[Dict],
                      result: str, scale: int = 4) -> np.ndarray:
        """
        วาด band overlay บน crop ดั้งเดิม:
          - scale up เพื่อให้อ่านง่าย
          - แถบแต่ละอัน: เส้นตั้ง (สีของแถบ) + label ชื่อสี + HSV
          - แถบสีสรุปด้านบน (color swatch)
          - ผลลัพธ์ (ohm label) ด้านล่าง
        """
        if crop is None or crop.size == 0:
            return np.zeros((100, 300, 3), dtype=np.uint8)

        h, w = crop.shape[:2]
        vis = cv2.resize(crop, (w * scale, h * scale),
                         interpolation=cv2.INTER_NEAREST)
        vh, vw = vis.shape[:2]

        # ── swatch strip ด้านบน ────────────────────────────────────────
        swatch_h = 28
        swatch   = np.zeros((swatch_h, vw, 3), dtype=np.uint8)
        n = len(bands)
        if n > 0:
            sw = vw // n
            for k, b in enumerate(bands):
                bgr = _BAND_BGR.get(b['color'], _BAND_BGR['UNKNOWN'])
                x1s, x2s = k * sw, (k + 1) * sw
                swatch[:, x1s:x2s] = bgr
                txt_col = (0, 0, 0) if sum(bgr) > 350 else (255, 255, 255)
                cv2.putText(swatch, b['color'][:3],
                            (x1s + 2, 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, txt_col, 1, cv2.LINE_AA)

        # ── วาดเส้น + label บน vis ────────────────────────────────────
        for b in bands:
            cx  = b.get('x_crop', b['x'])
            scx = int(cx * scale)
            bgr = _BAND_BGR.get(b['color'], _BAND_BGR['UNKNOWN'])
            # เส้นตั้งบาง + เส้นตั้งหนาโปร่งแสง
            overlay = vis.copy()
            cv2.line(overlay, (scx, 0), (scx, vh), bgr, max(2, scale))
            vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
            cv2.line(vis, (scx, 0), (scx, vh), bgr, 1)
            # label: ชื่อสี + H,S,V
            h_v, s_v, v_v = (int(x) for x in b['mean_hsv'])
            label = f"{b['color']} H{h_v}S{s_v}V{v_v}"
            lx = max(1, scx - 2)
            cv2.putText(vis, label, (lx, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, label, (lx, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        bgr, 1, cv2.LINE_AA)

        # ── band strips แต่ละแถบ crop จริงจากรูปต้นฉบับ ──────────────
        STRIP_H = 48  # ความสูง strip แต่ละเส้น (px)
        STRIP_W = 32  # ความกว้าง strip แต่ละเส้น
        GAP     = 4   # ช่องว่างระหว่าง strip
        if bands:
            strips = []
            for b in bands:
                cx   = b.get('x_crop', b['x'])
                hw   = max(b.get('w', 6) // 2, 4)
                x1c  = max(0, cx - hw)
                x2c  = min(w, cx + hw)
                patch = crop[:, x1c:x2c] if x2c > x1c else np.zeros((h, 4, 3), dtype=np.uint8)
                patch_r = cv2.resize(patch, (STRIP_W, STRIP_H), interpolation=cv2.INTER_LINEAR)
                bgr = _BAND_BGR.get(b['color'], _BAND_BGR['UNKNOWN'])
                # border ด้วยสีของ band
                cv2.rectangle(patch_r, (0, 0), (STRIP_W - 1, STRIP_H - 1), bgr, 2)
                strips.append(patch_r)
                if GAP > 0:
                    strips.append(np.zeros((STRIP_H, GAP, 3), dtype=np.uint8))

            strips_row = np.hstack(strips)
            # pad ให้ตรงกับ vw
            if strips_row.shape[1] < vw:
                pad = np.zeros((STRIP_H, vw - strips_row.shape[1], 3), dtype=np.uint8)
                strips_row = np.hstack([strips_row, pad])
            else:
                strips_row = strips_row[:, :vw]
            # label "band crops" ซ้ายบน
            cv2.putText(strips_row, "band crops", (2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            strips_row = np.zeros((STRIP_H, vw, 3), dtype=np.uint8)
            cv2.putText(strips_row, "no bands", (4, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 220), 1, cv2.LINE_AA)

        # ── result label ด้านล่าง ─────────────────────────────────────
        res_strip = np.zeros((22, vw, 3), dtype=np.uint8)
        good = result not in ("Unknown", "Read Error", "Calc Error", "?", "")
        rc   = (0, 220, 80) if good else (60, 60, 220)
        cv2.putText(res_strip, result, (4, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, rc, 1, cv2.LINE_AA)

        return np.vstack([swatch, strips_row, vis, res_strip])

    # ── 4. Band scanning ──────────────────────────────────────────────────────
    def _scan_bands(self, roi: np.ndarray) -> List[Dict]:
        h, w = roi.shape[:2]

        # CLAHE บน LAB เพื่อเพิ่ม local contrast ก่อน detect (หลักการเดิม)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        roi_enh = cv2.cvtColor(cv2.merge((l, a, b_ch)), cv2.COLOR_LAB2BGR)

        band_locs = self.detector.detect_bands_adaptive(roi_enh)
        if len(band_locs) < 3:
            proj_locs = self.detector.detect_bands_projection(roi_enh)
            if len(proj_locs) > len(band_locs):
                band_locs = proj_locs
        if not band_locs:
            return []

        # Sample color จากภาพ enhanced เดียวกัน, แค่ 30-70% กลาง (หลีก specular)
        hsv_roi = cv2.cvtColor(roi_enh, cv2.COLOR_BGR2HSV)
        ys, ye  = max(0, int(h * 0.30)), min(h, int(h * 0.70))
        if ye <= ys:
            ys, ye = 0, h

        bands = []
        for loc in band_locs:
            cx, w_band = loc['x'], loc['w']
            x1 = max(0, cx - max(1, w_band // 4))
            x2 = min(w, cx + max(1, w_band // 4))
            patch = hsv_roi[ys:ye, x1:x2]
            if patch.size == 0:
                continue
            # Histogram peak: find the dominant hue bin, then median S/V near that peak
            flat   = patch.reshape(-1, 3).astype(np.float32)
            h_int  = flat[:, 0].clip(0, 179).astype(np.int32)
            hist   = np.bincount(h_int, minlength=180).astype(np.float32)
            hist   = np.convolve(hist, np.ones(7) / 7.0, mode='same')   # smooth 7-bin window
            peak_h = float(np.argmax(hist))
            # Use pixels within ±12° of peak for S/V estimation
            near   = np.abs(flat[:, 0] - peak_h) < 12
            grp    = flat[near] if near.sum() > 3 else flat
            mean_hsv = (
                peak_h,
                float(np.median(grp[:, 1])),
                float(np.median(grp[:, 2])),
            )
            color    = self.detector.closest_color(mean_hsv)
            bands.append({
                'color':    color,
                'val':      COLOR_VALS.get(color, -99),
                'x':        cx,
                'w':        w_band,
                'mean_hsv': mean_hsv,
            })
        return bands

    # ── 5. Ohm calculation (4-band and 5-band) ───────────────────────────────
    _TOL_4 = {'GOLD': '5%', 'SILVER': '10%', 'BROWN': '1%'}
    _TOL_5 = {'GOLD': '5%', 'SILVER': '10%', 'BROWN': '1%',
               'RED': '2%', 'GREEN': '0.5%', 'BLUE': '0.25%', 'VIOLET': '0.1%'}

    def _label_from_ohms(self, ohms: float, tol: str) -> str:
        if   ohms >= 1e6: return f"{ohms/1e6:.2f}M Ohms +-{tol}"
        elif ohms >= 1e3: return f"{ohms/1e3:.2f}k Ohms +-{tol}"
        else:             return f"{ohms:.1f} Ohms +-{tol}"

    def _calculate_ohms(self, bands: List[Dict]) -> Tuple[str, List[Dict], float]:
        if len(bands) < 3:
            return "Unknown", bands, 0.0

        last_color = bands[-1]['color']
        n = len(bands)

        # ── ตรวจรูปแบบ 4-band vs 5-band ──────────────────────────────────────
        if n >= 5 and last_color in self._TOL_5:
            tolerance, calc_bands, five_band = self._TOL_5[last_color], bands[:-1], True
        elif n >= 4 and last_color in self._TOL_4:
            calc = bands[:-1]
            five_band  = len(calc) >= 4
            tolerance, calc_bands = self._TOL_4[last_color], calc
        elif n == 5:
            tolerance, calc_bands, five_band = "20%", bands, True
        else:
            tolerance, calc_bands, five_band = "20%", bands, False

        try:
            if five_band and len(calc_bands) >= 4:
                d1, d2, d3, mul = (calc_bands[i]['val'] for i in range(4))
                if -99 in (d1, d2, d3, mul):
                    return "Read Error", bands, 0.0
                base = d1 * 100 + d2 * 10 + d3
            else:
                d1, d2 = calc_bands[0]['val'], calc_bands[1]['val']
                mul = calc_bands[2]['val'] if len(calc_bands) >= 3 else 0
                if -99 in (d1, d2, mul):
                    return "Read Error", bands, 0.0
                base = d1 * 10 + d2

            multiplier = 10 ** mul if mul >= 0 else (0.1 if mul == -1 else 0.01)
            ohms = base * multiplier
            if ohms <= 0 or ohms > 10e6:
                return "Read Error", bands, 0.0

            if   ohms >= 1e6: label = f"{ohms/1e6:.2f}M Ohms +-{tolerance}"
            elif ohms >= 1e3: label = f"{ohms/1e3:.2f}k Ohms +-{tolerance}"
            else:             label = f"{ohms:.1f} Ohms +-{tolerance}"
            return label, bands, ohms

        except Exception as e:
            print(f"Calc error: {e}")
            return "Calc Error", bands, 0.0
