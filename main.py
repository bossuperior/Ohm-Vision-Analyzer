import cv2
import numpy as np  

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