import cv2
import numpy as np
from vision.band_detector import detect_bands_projection,closest_color, COLOR_VALS

def scan_bands_using_projection(roi):
    band_locs = detect_bands_projection(roi)
    detected_bands = []
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    for loc in band_locs:
        cx = loc['x']
        w_band = loc['w']
        h, w = roi.shape[:2]
        safe_w = max(1, w_band // 2)
        x1 = max(0, cx - safe_w // 2)
        x2 = min(w, cx + safe_w // 2)

        y1, y2 = int(h * 0.15), int(h * 0.35)
        sample_region = hsv_roi[y1:y2, x1:x2]

        if sample_region.size == 0: continue
        mean_color = cv2.mean(sample_region)[:3]

        color_name = closest_color(mean_color)
        val = COLOR_VALS.get(color_name, -99)
        
        detected_bands.append({
            'color': color_name, 'val': val, 'x': cx, 'w': w_band, 'mean_hsv': mean_color
        })
    return detected_bands

def calculate_resistance(bands):
    if len(bands) < 3: return "Unknown"
    if bands[0]['color'] in ['GOLD', 'SILVER']:
        bands.reverse()
    tolerance = "20%"
    last_band = bands[-1]
    calc_bands = bands
    if last_band['color'] in ['GOLD', 'SILVER']:
        if last_band['color'] == 'GOLD': tolerance = "5%"
        if last_band['color'] == 'SILVER': tolerance = "10%"
        calc_bands = bands[:-1]
    if len(calc_bands) < 2: return "Error"
    try:
        digit1 = calc_bands[0]['val']
        digit2 = calc_bands[1]['val']
        if len(calc_bands) == 2: multiplier_idx = 1
        else: multiplier_idx = calc_bands[2]['val']
        multiplier = 10 ** multiplier_idx if multiplier_idx >= 0 else (0.1 if multiplier_idx == -1 else 0.01)
        total_ohms = ((digit1 * 10) + digit2) * multiplier
        if total_ohms >= 1e6: return f"{total_ohms / 1e6:.2f}M Ohms {tolerance}"
        elif total_ohms >= 1e3: return f"{total_ohms / 1e3:.2f}k Ohms {tolerance}"
        else: return f"{total_ohms:.1f} Ohms {tolerance}"
    except:
        return "Calc Error"