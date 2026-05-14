import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# ==========================================
# 1. CONFIG: Reference Colors (Tuned Version)
# ==========================================
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


def get_resistor_img(img, box):
    """ตัดภาพและหมุนให้เป็นแนวนอน"""
    rect = cv2.boxPoints(box)
    rect = np.intp(rect)
    s = rect.sum(axis=1)
    tl = rect[np.argmin(s)]
    br = rect[np.argmax(s)]
    diff = np.diff(rect, axis=1)
    tr = rect[np.argmin(diff)]
    bl = rect[np.argmax(diff)]
    wA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    wB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(wA), int(wB))
    hA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    hB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(hA), int(hB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    pts = np.array([tl, tr, br, bl], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


# ==========================================
# 3. Projection & Scanning Logic
# ==========================================

def detect_bands_projection(roi):
    h, w = roi.shape[:2]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    roi_balanced = cv2.merge((l, a, b))
    roi_balanced = cv2.cvtColor(roi_balanced, cv2.COLOR_LAB2BGR)

    body_color = np.median(roi_balanced.reshape(-1, 3), axis=0)
    col_medians = np.median(roi_balanced, axis=0)
    diffs = np.linalg.norm(col_medians - body_color, axis=1)
    diffs_norm = cv2.normalize(diffs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # [TWEAK] Margin 8% เพื่อให้เห็นแถบที่ชิดขอบ
    margin = int(w * 0.08)
    diffs_norm[:margin] = 0
    diffs_norm[-margin:] = 0

    # [TWEAK] Otsu Thresholding (ดีที่สุดสำหรับภาพชัด)
    _, thresh = cv2.threshold(diffs_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # [TWEAK] Fallback: ถ้า Otsu จับได้น้อยกว่า 3 แถบ ให้ลองลด Threshold ลง
    temp_cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(temp_cnts) < 3:
        _, thresh = cv2.threshold(diffs_norm, 20, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 1), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    thresh_img = thresh.reshape(1, -1)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    band_centers = []
    for cnt in contours:
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        if w_cnt < w * 0.02: continue
        center_x = x + w_cnt // 2
        band_centers.append({'x': center_x, 'w': w_cnt})

    band_centers.sort(key=lambda k: k['x'])
    return band_centers


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

        # [TWEAK] Sampling ส่วนบน (15%-35%) เพื่อหลบแสงสะท้อน
        y1, y2 = int(h * 0.15), int(h * 0.35)
        sample_region = hsv_roi[y1:y2, x1:x2]

        if sample_region.size == 0: continue
        mean_color = cv2.mean(sample_region)[:3]

        color_name = closest_color(mean_color)
        val = COLOR_VALS.get(color_name, -99)

        print(f"  Band at x={cx}: Read HSV={mean_color} -> Detected: {color_name}")
        detected_bands.append({
            'color': color_name,
            'val': val,
            'x': cx,
            'w': w_band,
            'mean_hsv': mean_color
        })
    return detected_bands


def calculate_resistance(bands):
    if len(bands) < 3: return "Unknown"

    # Direction Check
    if bands[0]['color'] in ['GOLD', 'SILVER']:
        print("  -> Reading backwards? Reversing bands...")
        bands.reverse()

    tolerance = "20%"  # Default if no tolerance band
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

        # Logic การหา Multiplier
        if len(calc_bands) == 2:
            # ถ้าเจอแค่ 2 แถบ (+Tolerance)
            # เคส Red-Red-(Brownหาย)-Gold = 220 Ohm
            multiplier_idx = 1
        else:
            multiplier_idx = calc_bands[2]['val']

        multiplier = 10 ** multiplier_idx if multiplier_idx >= 0 else (0.1 if multiplier_idx == -1 else 0.01)
        total_ohms = ((digit1 * 10) + digit2) * multiplier

        if total_ohms >= 1e6:
            return f"{total_ohms / 1e6:.2f}M Ohms {tolerance}"
        elif total_ohms >= 1e3:
            return f"{total_ohms / 1e3:.2f}k Ohms {tolerance}"
        else:
            return f"{total_ohms:.1f} Ohms {tolerance}"
    except:
        return "Calc Error"
def detect_bands(roi_bgr, debug=False):
    # 1) เตรียมภาพ
    h_img, w_img = roi_bgr.shape[:2]

    # mask ตัวต้านทาน (พื้นหลังดำ => Gray ต่ำมาก)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 5)
    im_show = cv2.resize(blur, [800, 600])
    cv2.imshow("blur", im_show)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    im_show = cv2.resize(thresh, [800, 600])
    cv2.imshow("thresh", im_show)

    kernel = np.ones((3, 9), np.uint8)
    body_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    im_show = cv2.resize(body_mask, [800, 600])
    cv2.imshow("body_mask_OPEN", im_show)
    kernel = np.ones((9, 3), np.uint8)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    im_show = cv2.resize(body_mask, [800, 600])
    cv2.imshow("body_mask_close", im_show)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        body_mask, connectivity=8
    )
    band = np.zeros_like(body_mask, np.uint8)
    count = 0
    for i in range(1, num_labels):  # label 0 = background
        x, y, w, h, area = stats[i]
        # 2) เอาเฉพาะรูเล็ก ๆ (เช่น area < 40 pixel)
        if area > 500:
            count += 1
            band[labels == i] = 255
            roi_band = roi_bgr[y:y + h, x:x + w]
            cv2.imshow(f"roi_band_{i}", roi_band)
    # im_show = cv2.resize(band, [800, 600])
    # cv2.imshow("band", im_show)
    # print(count)
# from CV3_histogram_equalization.global_histogram_equalization import equalize_hist
# ==========================================
# 4. Main Execution
# ==========================================
def main():
    # *** เปลี่ยนชื่อไฟล์ตรงนี้ ***
    path = 'images/mix.PNG'

    img = cv2.imread(path)
    if img is None:
        print(f"Error: Cannot load image at {path}")
        return

    # Resize ให้ประมวลผลเร็วและดูง่าย
    if img.shape[1] > 1000:
        scale = 1000 / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    original = img.copy()

    # 1. Body Detection (HSV)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # im_show = cv2.resize(gray, [800, 600])
    # cv2.imshow("img show", im_show)
    # b,g,r = cv2.split(img)
    # brightness = (r+g+b)/3
    # diff1 = abs(r-g)
    # diff2 = abs(r-b)
    # diff3 = abs(g-b)
    # # gray_mask = (brightness>100) & (diff1<20) & (diff2<20) & (diff3<20)
    # gray_mask = (brightness > 40)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    gray_mask = (s<70)&(v>50)
    img[gray_mask]=[0,0,0]
    im_show = cv2.resize(img, [800, 600])
    cv2.imshow("img show", im_show)

    mask = np.zeros(img.shape[:2], np.uint8)
    mask[gray_mask] = 255
    im_show = cv2.resize(mask, [800, 600])
    cv2.imshow("mask", im_show)
    # หา connected components ของสีขาว
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    # เตรียม mask สำหรับเก็บเฉพาะรู
    hole_mask = np.zeros_like(mask, np.uint8)

    for i in range(1, num_labels):  # label 0 = background
        x, y, w, h, area = stats[i]
        # 2) เอาเฉพาะรูเล็ก ๆ (เช่น area < 40 pixel)
        if area < 1000:
            hole_mask[labels == i] = 255

    im_show = cv2.resize(hole_mask, [800, 600])
    cv2.imshow("hole_mask", im_show)

    inpaint = cv2.inpaint(img, hole_mask, 3, cv2.INPAINT_TELEA)
    im_show = cv2.resize(inpaint, [800, 600])
    cv2.imshow("inpaint", im_show)

    gray = cv2.cvtColor(inpaint, cv2.COLOR_BGR2GRAY)
    im_show = cv2.resize(gray, [800, 600])
    cv2.imshow("img show", im_show)

    blur = cv2.GaussianBlur(gray, (5, 5), 5)
    im_show = cv2.resize(blur, [800, 600])
    cv2.imshow("blur", im_show)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    im_show = cv2.resize(thresh, [800, 600])
    cv2.imshow("thresh", im_show)
    kernel = np.ones((5, 5), np.uint8)
    body_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    im_show = cv2.resize(body_mask, [800, 600])
    cv2.imshow("body_mask_close", im_show)
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    resistor_rois = []
    count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 1000 & w>100:  # กรอง noise
            count = count + 1
            roi = inpaint[y:y + h, x:x + w]
            band = detect_bands(roi)
            print(band)
            # resistor_rois.append(roi)
            # cv2.imshow(f"ROI {count}", roi)

    # kernel = np.ones((3, 3), np.uint8)
    # body_mask = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations=1)
    # im_show = cv2.resize(body_mask, [800, 600])
    # cv2.imshow("body_mask_erod", im_show)
    # # 3. หา Contour
    # contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # count = 0
    # for cnt in contours:
    #     if cv2.contourArea(cnt) < 1000: continue
    #     rect = cv2.minAreaRect(cnt)
    #     (cx, cy), (w, h), angle = rect
    #     ratio = max(w, h) / (min(w, h) + 1e-5)
    #     if ratio < 1.8 or ratio > 6.0: continue
    #
    #     count += 1
    #     box = cv2.boxPoints(rect)
    #     box = np.intp(box)
    #
    #     # 2. Crop ROI
    #     roi = get_resistor_img(img, rect)
    #
        # 3. Process Bands
        #     bands = scan_bands_using_projection(roi)  # Scan
        #     bands = fix_false_colors(bands)  # Fix Logic
        #     ohm_str = calculate_resistance(bands)  # Calculate
        #
        #     # 4. Visualization
        #     band_colors = "-".join([b['color'][:3] for b in bands])
        #     print(f"Resistor {count}: {band_colors} -> {ohm_str}")
        #
        #     cv2.drawContours(original, [box], 0, (0, 255, 0), 2)
        #     # วาง Text เหนือกล่องนิดหน่อย
        #     cv2.putText(original, ohm_str, (box[1][0], box[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        #
        #     # Debug ROI Window
        #     debug_roi = roi.copy()
        #     for b in bands:
        #         cv2.line(debug_roi, (b['x'], 0), (b['x'], roi.shape[0]), (0, 255, 0), 2)
        #     cv2.imshow(f"ROI {count}", debug_roi)

    # img_final = cv2.resize(original, (800, 800))
    # cv2.imshow("Final Result", img_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()