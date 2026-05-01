import cv2
import numpy as np
from typing import List, Dict, Tuple
from src.vision.band_detector import BandDetector
from src.vision.color_mapping import COLOR_VALS


class BandReader:
    def __init__(self, y_start_pct: float = 0.25, y_end_pct: float = 0.75):
        self.y_start_pct = y_start_pct
        self.y_end_pct = y_end_pct
        self.detector = BandDetector()

    # ──────────────────────────────────────────────────────────
    # 1. Crop ขอบออก (เอาเฉพาะตัวถัง ตัดขาออก)
    # ──────────────────────────────────────────────────────────
    def _shrink_roi(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        x1 = int(w * 0.12)
        x2 = int(w * 0.88)
        y1 = int(h * 0.05)
        y2 = int(h * 0.95)
        if x2 <= x1 or y2 <= y1:
            return img
        return img[y1:y2, x1:x2]

    # ──────────────────────────────────────────────────────────
    # 2. Auto White Balance — Gray World assumption
    # ──────────────────────────────────────────────────────────
    def _awb(self, img: np.ndarray) -> np.ndarray:
        result = img.astype(np.float32)
        for ch in range(3):
            avg = np.mean(result[:, :, ch])
            if avg > 0:
                all_avg = np.mean(result)
                result[:, :, ch] = np.clip(result[:, :, ch] * (all_avg / avg), 0, 255)
        return result.astype(np.uint8)

    # ──────────────────────────────────────────────────────────
    # 2b. Board-referenced White Balance
    #     ใช้สีขอบบอร์ด (เบจ) เป็น reference ปรับ color cast
    #     แม้แสง LED เหลือง/น้ำเงินทำให้บอร์ดดูต่างออกไป
    #     ก็สามารถ normalize กลับมาให้ใกล้เคียง neutral ได้
    # ──────────────────────────────────────────────────────────
    # BGR ของ breadboard สีเบจมาตรฐาน (วัดจาก neutral white light)
    BOARD_BEIGE_BGR = np.array([185, 195, 200], dtype=np.float32)  # B, G, R

    def _board_ref_wb(self, img: np.ndarray) -> np.ndarray:
        """
        ปรับ white balance โดยใช้ขอบซ้าย-ขวาของ resistor crop เป็น reference
        (ขอบคือพื้นบอร์ด ไม่ใช่แถบสี)
        """
        h, w = img.shape[:2]
        edge_w = max(1, int(w * 0.10))

        # sample ขอบซ้าย + ขวา (พื้นบอร์ด)
        left  = img[:, :edge_w].reshape(-1, 3).astype(np.float32)
        right = img[:, w - edge_w:].reshape(-1, 3).astype(np.float32)
        measured = np.median(np.vstack([left, right]), axis=0)  # [B, G, R]

        # ถ้า measured มืดหรือสว่างเกินไป ให้ skip (ไม่น่าเชื่อถือ)
        if np.any(measured < 30) or np.any(measured > 240):
            return img

        scale = self.BOARD_BEIGE_BGR / (measured + 1e-6)
        # จำกัด scale ไม่ให้ extreme เกินไป (กัน overexpose)
        scale = np.clip(scale, 0.5, 2.0)

        result = img.astype(np.float32) * scale[np.newaxis, np.newaxis, :]
        return np.clip(result, 0, 255).astype(np.uint8)

    # ──────────────────────────────────────────────────────────
    # 3. Public entry point
    # ──────────────────────────────────────────────────────────
    def calculate(self, cropped_resistor: np.ndarray) -> Tuple[str, List[Dict], float]:
        if cropped_resistor is None or cropped_resistor.size == 0:
            return "Unknown", [], 0.0

        cleaned_roi = self._shrink_roi(cropped_resistor)
        cleaned_roi = self._board_ref_wb(cleaned_roi)  # board-referenced WB ก่อน
        cleaned_roi = self._awb(cleaned_roi)            # Gray World ตาม
        bands = self._scan_bands(cleaned_roi)

        if not bands:
            return "Unknown", [], 0.0

        bands = self.detector.fix_false_colors(bands)
        resistance_str, bands, total_ohms = self._calculate_ohms(bands)
        return resistance_str, bands, total_ohms

    # ──────────────────────────────────────────────────────────
    # 4. Band scanning (Multi-strip sampling + voting)
    # ──────────────────────────────────────────────────────────
    def _scan_bands(self, roi: np.ndarray) -> List[Dict]:
        # ── Pre-processing ────────────────────────────────────
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        l = clahe.apply(l)
        roi_bright = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        # ── Detect band locations ─────────────────────────────
        band_locs = self.detector.detect_bands_projection(roi_bright)
        if not band_locs:
            return []

        h, w = roi.shape[:2]

        # ── Multi-strip sampling: sample หลาย row แล้ว vote ──
        # แทนที่จะ sample แค่ช่วง y เดียว ให้ sample หลายแถว
        # แล้วเอาสีที่โหวตชนะในแต่ละ band
        strip_centers = [0.30, 0.40, 0.50, 0.60, 0.70]
        strip_half_h  = int(h * 0.06)   # ความสูงของแต่ละ strip

        hsv_roi = cv2.cvtColor(roi_bright, cv2.COLOR_BGR2HSV)

        detected_bands = []
        for loc in band_locs:
            cx     = loc['x']
            w_band = loc['w']

            # กว้างของ sample region = ครึ่งหนึ่งของ band width
            # (ป้องกันติดสีข้างเคียง)
            safe_half = max(1, w_band // 3)
            x1 = max(0, cx - safe_half)
            x2 = min(w, cx + safe_half)

            # เก็บ HSV mean จากแต่ละ strip
            strip_hsv_list = []
            for yc_pct in strip_centers:
                yc = int(h * yc_pct)
                ys = max(0, yc - strip_half_h)
                ye = min(h, yc + strip_half_h)
                patch = hsv_roi[ys:ye, x1:x2]
                if patch.size == 0:
                    continue
                # เอา median แทน mean เพื่อลด outlier จาก glare
                strip_hsv_list.append(np.median(
                    patch.reshape(-1, 3), axis=0))

            if not strip_hsv_list:
                continue

            # Vote: แต่ละ strip โหวตสี → เอาสีที่ชนะมากที่สุด
            color_votes: Dict[str, int] = {}
            for hsv_val in strip_hsv_list:
                color = self.detector.closest_color(hsv_val)
                color_votes[color] = color_votes.get(color, 0) + 1

            # สีที่ชนะ
            winner_color = max(color_votes, key=color_votes.get)

            # mean HSV สำหรับ fix_false_colors
            mean_hsv = tuple(np.mean(strip_hsv_list, axis=0))
            val = COLOR_VALS.get(winner_color, -99)

            detected_bands.append({
                'color':    winner_color,
                'val':      val,
                'x':        cx,
                'w':        w_band,
                'mean_hsv': mean_hsv,
                'votes':    color_votes
            })

        return detected_bands

    # ──────────────────────────────────────────────────────────
    # 5. Calculate Ohms from bands
    # ──────────────────────────────────────────────────────────
    def _calculate_ohms(self, bands: List[Dict]) -> Tuple[str, List[Dict], float]:
        if len(bands) < 3:
            return "Unknown", bands, 0.0

        # ── แก้ทิศทางการอ่าน (Gold/Silver ไม่ใช่แถบแรก) ──────
        if bands[0]['color'] in ['GOLD', 'SILVER']:
            bands.reverse()

        tolerance  = "20%"
        calc_bands = bands.copy()

        # ── แยก tolerance band ออกจากท้าย ─────────────────────
        last = bands[-1]
        if last['color'] == 'GOLD':
            tolerance  = "5%"
            calc_bands = bands[:-1]
        elif last['color'] == 'SILVER':
            tolerance  = "10%"
            calc_bands = bands[:-1]
        elif last['color'] == 'BROWN':
            # 5-band precision resistor: last band = 1% tolerance
            tolerance  = "1%"
            calc_bands = bands[:-1]

        if len(calc_bands) < 2:
            return "Error", bands, 0.0

        try:
            digit1 = calc_bands[0]['val']
            digit2 = calc_bands[1]['val']

            multiplier_idx = calc_bands[2]['val'] if len(calc_bands) >= 3 else 0

            if -99 in [digit1, digit2, multiplier_idx]:
                return "Read Error", bands, 0.0

            if multiplier_idx >= 0:
                multiplier = 10 ** multiplier_idx
            elif multiplier_idx == -1:
                multiplier = 0.1
            elif multiplier_idx == -2:
                multiplier = 0.01
            else:
                multiplier = 1

            total_ohms = ((digit1 * 10) + digit2) * multiplier

            if total_ohms <= 0 or total_ohms > 10e6:
                return "Read Error", bands, 0.0

            if total_ohms >= 1e6:
                formatted = f"{total_ohms / 1e6:.2f}M Ohms +/-{tolerance}"
            elif total_ohms >= 1e3:
                formatted = f"{total_ohms / 1e3:.2f}k Ohms +/-{tolerance}"
            else:
                formatted = f"{total_ohms:.1f} Ohms +/-{tolerance}"

            return formatted, bands, total_ohms

        except Exception as e:
            print(f"Calculation error: {e}")
            return "Calc Error", bands, 0.0
