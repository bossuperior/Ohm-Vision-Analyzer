import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

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
