import cv2
import numpy as np

def resist_body_detector(frame):
    process_frame = frame.copy()

    #Filterout breadboard background by HSV thresholding (เบรดบอร์ดจะมีสีอิ่มตัวต่ำและสว่างพอสมควร)
    hsv = cv2.cvtColor(process_frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gray_mask = (s < 70) & (v > 50)
    process_frame[gray_mask] = [0, 0, 0] 

    # Convert to grayscale and apply Gaussian Blur
    gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 5)
    
    # Use Adaptive Thresholding to get binary image
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Use Morphological Operations to clean up the moise
    kernel_open = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    kernel_close = np.ones((9, 9), np.uint8)
    body_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Find contours in the body mask and filter them based on size and aspect ratio to identify potential resistor bodies
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    boxes = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
        if (w * h > 1000) and (aspect_ratio < 5.0) and (w < 200) and (h < 200):
            roi = frame[y:y + h, x:x + w]
            rois.append(roi)
            boxes.append((x, y, w, h))
    return rois, boxes, body_mask