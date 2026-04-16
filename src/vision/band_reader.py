import cv2
import numpy as np
from typing import List, Dict, Tuple

# Assuming these are available in your vision module
from src.vision.band_detector import detect_bands_projection, closest_color, COLOR_VALS


class BandReader:
    def __init__(self, y_start_pct: float = 0.15, y_end_pct: float = 0.35):
        # Configurable sample region to avoid cylindrical glare.
        # Moving these to __init__ allows you to easily tweak them
        # depending on your webcam's lighting without touching the core logic.
        self.y_start_pct = y_start_pct
        self.y_end_pct = y_end_pct

    def calculate(self, cropped_resistor: np.ndarray) -> Tuple[str, List[Dict]]:
        """
        Public entry point. Scans bands and returns the string value and raw data.
        Returns: ("1.50k Ohms 5%", [{'color': 'BROWN'...}, ...])
        """
        bands = self._scan_bands(cropped_resistor)

        if not bands:
            return "Unknown", []

        resistance_str = self._calculate_ohms(bands)
        return resistance_str, bands

    def _scan_bands(self, roi: np.ndarray) -> List[Dict]:
        """Private helper: Extracts colors from the image ROI."""
        band_locs = detect_bands_projection(roi)
        detected_bands = []

        if not band_locs:
            return detected_bands

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
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
            color_name = closest_color(mean_color)
            val = COLOR_VALS.get(color_name, -99)

            detected_bands.append({
                'color': color_name,
                'val': val,
                'x': cx,
                'w': w_band,
                'mean_hsv': mean_color
            })

        return detected_bands

    def _calculate_ohms(self, bands: List[Dict]) -> str:
        """Private helper: Applies Ohm math based on standard resistor codes."""
        if len(bands) < 3:
            return "Unknown"

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
            return "Error"

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
                return "Read Error"

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

            # 5. Format Output
            if total_ohms >= 1e6:
                return f"{total_ohms / 1e6:.2f}M Ohms ±{tolerance}"
            elif total_ohms >= 1e3:
                return f"{total_ohms / 1e3:.2f}k Ohms ±{tolerance}"
            else:
                return f"{total_ohms:.1f} Ohms ±{tolerance}"

        except Exception as e:
            print(f"Calculation error: {e}")
            return "Calc Error"