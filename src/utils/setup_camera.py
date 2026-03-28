import tkinter as tk

import cv2
import numpy as np

from src.topology.grid_mapper import draw_grid
from src.vision.cv_body_detector import cv_body_detector

# 5.4 cm = 54 mm -> 540 px
# 8.1 cm = 81 mm -> 810 px
BOARD_WIDTH = 540
BOARD_HEIGHT = 810

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cv2.namedWindow('Camera')
root = tk.Tk()


def close_window():
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    root.destroy()


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # --- Auto-Detect Horizontal / Vertical ---
    width_px = np.linalg.norm(rect[0] - rect[1])
    height_px = np.linalg.norm(rect[1] - rect[2])
    if width_px > height_px:
        target_w = 810
        target_h = 540
    else:
        target_w = 540
        target_h = 810
    return rect, target_w, target_h


def detect_board(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    src_points = None
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        biggest_contour = contours[0]

        peri = cv2.arcLength(biggest_contour, True)
        approx = cv2.approxPolyDP(biggest_contour, 0.02 * peri, True)
        if len(approx) == 4:
            src_points = approx.reshape((4, 2))
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            for (x, y) in src_points:
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

    return src_points


def camera():
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            clean_frame = frame.copy()
            display_frame = frame.copy()

            src_corners = detect_board(display_frame)
            cv2.imshow('Camera', display_frame)
            if src_corners is not None:
                src_pts, target_w, target_h = order_points(src_corners)
                dst_pts = np.array([
                    [0, 0],
                    [target_w, 0],
                    [target_w, target_h],
                    [0, target_h]
                ], dtype="float32")
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                crop_board = cv2.warpPerspective(clean_frame, matrix, (target_w, target_h))
                cv2.imshow('Cropped Breadboard', crop_board)
                # rois, boxes, debug_mask = cv_body_detector(crop_board)
                # grid_board = draw_grid(crop_board, target_w, target_h)
                # for (x, y, w, h) in boxes:
                #     cv2.rectangle(grid_board, (x, y), (x + w, y + h), (255, 0, 255), 2)  # สีม่วงแดง
                #     cv2.putText(grid_board, "Resistor", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                # cv2.imshow('Breadboard', grid_board)
            cv2.waitKey(1)
        root.after(10, camera)

def setup_camera():
    root.title("Control Panel")
    root.geometry("200x100")
    close_button = tk.Button(root, text="Close Camera", command=close_window, bg="red", fg="white", font=("Arial", 12))
    close_button.pack(expand=True, fill='both', padx=20, pady=20)
    camera()
    root.mainloop()
