import cv2
import numpy as np
from src.vision.color_mapping import REF_COLORS, COLOR_VALS

def closest_color(hsv_val):
    h, s, v = hsv_val
    min_dist = float('inf') #Infinite distance to ensure any real color will be closer
    best_color = 'UNKNOWN'

    for name, ref_hsv in REF_COLORS.items():
        diff_h = abs(h - ref_hsv[0]) 
        if diff_h > 90: diff_h = 180 - diff_h # Hue 180 degree wrap-around

        diff_s = abs(s - ref_hsv[1])
        diff_v = abs(v - ref_hsv[2])

        #Weights to prioritize hue for vivid colors and value for metallics
        if name in ['BLACK', 'WHITE', 'GRAY', 'SILVER', 'BROWN']:
            weight_h, weight_s, weight_v = 1.0, 1.0, 5.0
        elif name in ['YELLOW']:
            weight_h, weight_s, weight_v = 10.0, 1.0, 1.0
        elif name in ['RED', 'GOLD', 'ORANGE']:
            weight_h, weight_s, weight_v = 3.0, 2.0, 2.0
        else:
            weight_h, weight_s, weight_v = 4.0, 1.0, 1.0
        #Euclidean distance in weighted HSV space
        dist = np.sqrt((diff_h * weight_h) ** 2 + (diff_s * weight_s) ** 2 + (diff_v * weight_v) ** 2)
        if dist < min_dist:
            min_dist = dist
            best_color = name
    return best_color

def fix_false_colors(bands):
    if not bands: return bands
    for i, band in enumerate(bands):
        h, s, v = band['mean_hsv']
        color = band['color']
        #Fix Lighting & Saturation Issues
        if color == 'GOLD':
            if s > 130:
                band['color'] = 'RED'; band['val'] = 2
            elif v < 100:
                band['color'] = 'BROWN'; band['val'] = 1
        if color == 'BROWN' and s > 150:
            band['color'] = 'RED'; band['val'] = 2
        #Hue Shift Override
        if i < 2 and color in ['YELLOW', 'GREEN', 'BLUE']:
            if s > 80 and v > 80:
                band['color'] = 'RED'; band['val'] = 2
    #Handle cases where first bands are misread as GOLD/SILVER due to lighting
    if bands[0]['color'] in ['GOLD', 'SILVER']:
        bands[0]['color'] = 'RED'; bands[0]['val'] = 2
    if len(bands) >= 2 and bands[1]['color'] in ['GOLD', 'SILVER']:
        bands[1]['color'] = 'RED'; bands[1]['val'] = 2
    last_band = bands[-1]
    #Tolerance Band Correction for Red/Gold/Orange
    is_vivid_red = (last_band['mean_hsv'][1] > 100)
    if len(bands) == 3 and last_band['color'] == 'RED' and is_vivid_red:
        pass
    elif last_band['color'] in ['RED', 'ORANGE', 'BROWN']:
        last_band['color'] = 'GOLD'; last_band['val'] = -1
    return bands

def detect_bands_projection(roi):
    h, w = roi.shape[:2]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    #l = Lightness channel for better contrast, a and b for color info
    l, a, b = cv2.split(lab)
    #Apply CLAHE to enhance contrast, especially for low-light or faded bands
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    roi_balanced = cv2.merge((l, a, b))
    roi_balanced = cv2.cvtColor(roi_balanced, cv2.COLOR_LAB2BGR)
    #Calculate color distance from body color for each column to find band locations
    body_color = np.median(roi_balanced.reshape(-1, 3), axis=0)
    #Use median to reduce influence of outliers and noise, then find columns that differ significantly from body color
    col_medians = np.median(roi_balanced, axis=0)
    diffs = np.linalg.norm(col_medians - body_color, axis=1)
    diffs_norm = cv2.normalize(diffs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #Ignore edges to reduce false positives from lighting gradients or reflections
    margin = int(w * 0.08)
    diffs_norm[:margin] = 0
    diffs_norm[-margin:] = 0
    #Use Otsu's thresholding to find significant peaks in the color difference, which correspond to band locations
    _, thresh = cv2.threshold(diffs_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    temp_cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(temp_cnts) < 3:
        _, thresh = cv2.threshold(diffs_norm, 20, 255, cv2.THRESH_BINARY)
    #Use morphological closing to connect fragmented parts of bands, which can occur due to lighting or surface imperfections
    kernel = np.ones((5, 1), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #Find contours again after closing to get more complete band shapes, then filter by size to remove noise
    thresh_img = thresh.reshape(1, -1)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Calculate the center x-coordinate and width of each detected band contour, then sort by x-coordinate to maintain left-to-right order
    band_centers = []
    for cnt in contours:
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        if w_cnt < w * 0.02: continue
        center_x = x + w_cnt // 2
        band_centers.append({'x': center_x, 'w': w_cnt})

    band_centers.sort(key=lambda k: k['x'])
    return band_centers
