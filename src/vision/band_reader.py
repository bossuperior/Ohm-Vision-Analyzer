import cv2
import numpy as np
from typing import List, Dict, Tuple
from src.vision.band_detector import BandDetector
from src.vision.color_mapping import COLOR_VALS

class BandReader:
    def __init__(self, y_start_pct: float = 0.30, y_end_pct: float = 0.70):
        # Configurable sample region to avoid cylindrical glare.
        self.y_start_pct = y_start_pct
        self.y_end_pct = y_end_pct
        self.detector = BandDetector()

    def _shrink_roi(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        x1 = int(w * 0.15)
        x2 = int(w * 0.85)
        y1 = int(h * 0.05)
        y2 = int(h * 0.95)

        if x2 <= x1 or y2 <= y1:
            return img
        return img[y1:y2, x1:x2]

    def _awb(self, img: np.ndarray) -> np.ndarray:
        result = img.astype(np.float32)
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        avg_all = (avg_b + avg_g + avg_r) / 3.0

        if avg_b > 0 and avg_g > 0 and avg_r > 0:
            result[:, :, 0] = np.clip(result[:, :, 0] * (avg_all / avg_b), 0, 255)
            result[:, :, 1] = np.clip(result[:, :, 1] * (avg_all / avg_g), 0, 255)
            result[:, :, 2] = np.clip(result[:, :, 2] * (avg_all / avg_r), 0, 255)
        return result.astype(np.uint8)

    def calculate(self, cropped_resistor: np.ndarray) -> Tuple[str, List[Dict], float]:
        """
        Public entry point. Scans bands and returns the string value and raw data.
        Returns: ("1.50k Ohms 5%", [{'color': 'BROWN'...}, ...])
        """
        cleaned_roi = self._shrink_roi(cropped_resistor)
        cleaned_roi = self._awb(cleaned_roi)
        bands = self._scan_bands(cleaned_roi)

        if not bands:
            return "Unknown", [],0.0
        bands = self.detector.fix_false_colors(bands)
        resistance_str, bands, total_ohms = self._calculate_ohms(bands)
        return resistance_str, bands, total_ohms

    def _scan_bands(self, roi: np.ndarray) -> List[Dict]:
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        roi_balanced = cv2.merge((l, a, b))
        roi_bright = cv2.cvtColor(roi_balanced, cv2.COLOR_LAB2BGR)

        #Extract band colors from the cropped resistor image (Region of Interest) and return a list of detected bands with their colors and positions
        band_locs = self.detector.detect_bands_projection(roi_bright)
        detected_bands = []
        if not band_locs:
            return detected_bands

        hsv_roi = cv2.cvtColor(roi_bright, cv2.COLOR_BGR2HSV)
        h, w = roi.shape[:2]
        y1, y2 = int(h * self.y_start_pct), int(h * self.y_end_pct)

        for loc in band_locs:
            cx = loc['x']
            w_band = loc['w']

            # Dynamic safe width to prevent sampling the breadboard background
            safe_w = max(1, w_band // 2)
            x1 = max(0, cx - safe_w // 2)
            x2 = min(w, cx + safe_w // 2)
            sample_region = hsv_roi[y1:y2, x1:x2]

            if sample_region.size == 0:
                continue

            mean_color = cv2.mean(sample_region)[:3]
            color_name = self.detector.closest_color(mean_color)
            val = COLOR_VALS.get(color_name, -99)

            detected_bands.append({
                'color': color_name,
                'val': val,
                'x': cx,
                'w': w_band,
                'mean_hsv': mean_color
            })

        return detected_bands

    def _calculate_ohms(self, bands: List[Dict]) -> Tuple[str, List[Dict], float]:
        # Calculate the resistance value based on the detected band colors, applying standard resistor color code rules and handling special cases for tolerance bands
        if len(bands) < 3:
            return "Unknown", bands, 0.0

        # 1. Handle reading direction (Gold/Silver are never the first band)
        if bands[0]['color'] in ['GOLD', 'SILVER']:
            bands.reverse()

        tolerance = "20%"  # Default if missing
        calc_bands = bands.copy()  # Avoid mutating the original list

        # 2. Extract Tolerance from the last band
        last_band = bands[-1]
        if last_band['color'] in ['GOLD', 'SILVER']:
            tolerance = "5%" if last_band['color'] == 'GOLD' else "10%"
            calc_bands = bands[:-1]
        elif last_band['color'] == 'BROWN':
            # Very common in precision 5-band resistors!
            tolerance = "1%"
            calc_bands = bands[:-1]

        if len(calc_bands) < 2:
            return "Error", bands, 0.0

        try:
            # 3. Calculate Base Value
            digit1 = calc_bands[0]['val']
            digit2 = calc_bands[1]['val']

            # 4. Determine Multiplier
            if len(calc_bands) >= 3:
                multiplier_idx = calc_bands[2]['val']
            else:
                multiplier_idx = 0  # Fallback

            # Prevent math errors if a color matched incorrectly
            if multiplier_idx == -99 or digit1 == -99 or digit2 == -99:
                return "Read Error", bands, 0.0

            # Note: Ensure your COLOR_VALS dict has GOLD = -1 and SILVER = -2
            if multiplier_idx >= 0:
                multiplier = 10 ** multiplier_idx
            elif multiplier_idx == -1:
                multiplier = 0.1
            elif multiplier_idx == -2:
                multiplier = 0.01
            else:
                multiplier = 1

            total_ohms = ((digit1 * 10) + digit2) * multiplier

            # Reject physically impossible values for common breadboard resistors
            if total_ohms <= 0 or total_ohms > 10e6:
                return "Read Error", bands, 0.0

            # 5. Format Output
            if total_ohms >= 1e6:
                formatted_str = f"{total_ohms / 1e6:.2f}M Ohms +/-{tolerance}"
            elif total_ohms >= 1e3:
                formatted_str = f"{total_ohms / 1e3:.2f}k Ohms +/-{tolerance}"
            else:
                formatted_str = f"{total_ohms:.1f} Ohms +/-{tolerance}"
            
            return formatted_str, bands, total_ohms

        except Exception as e:
            print(f"Calculation error: {e}")
            return "Calc Error", bands, 0.0