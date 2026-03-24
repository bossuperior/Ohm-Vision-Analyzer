import cv2
import numpy as np
import tkinter as tk
from src.topology.grid_mapper import draw_grid, topology_click
from src.vision.resist_body_detector import resist_body_detector

click_points = []
# 5.4 cm = 54 mm -> 540 px
# 8.1 cm = 81 mm -> 810 px
BOARD_WIDTH = 540
BOARD_HEIGHT = 810

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(click_points) < 4:
        click_points.append((x, y))
        print(f"Point {len(click_points)} selected: ({x}, {y})")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', mouse_callback)
root = tk.Tk()

def close_window():
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    root.destroy()

def draw_points(frame):
    for idx, point in enumerate(click_points):
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
        cv2.putText(frame, f"P{idx+1}", (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    sum = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(sum)] 
    rect[2] = pts[np.argmax(sum)] 
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

def camera():
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            clean_frame = frame.copy()
            if len(click_points) < 4:
                draw_points(frame)
                cv2.imshow('Camera', frame)
            else:
                try:
                    if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) >= 1:
                        cv2.destroyWindow('Camera')
                except:
                    pass
                ordered_pts, target_w, target_h = order_points(click_points)
                dst_pts = np.array([
                    [0, 0],
                    [target_w, 0],
                    [target_w, target_h],
                    [0, target_h]
                ], dtype="float32")              
                matrix = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
                crop_board = cv2.warpPerspective(clean_frame, matrix, (target_w, target_h))
                rois, boxes, debug_mask = resist_body_detector(crop_board)
                grid_board = draw_grid(crop_board, target_w, target_h)
                for (x, y, w, h) in boxes:
                    cv2.rectangle(grid_board, (x, y), (x + w, y + h), (255, 0, 255), 2) # สีม่วงแดง
                    cv2.putText(grid_board, "Resistor", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                cv2.imshow('Breadboard', grid_board)
                cv2.setMouseCallback('Breadboard', topology_click, param=(target_w, target_h))
            cv2.waitKey(1)
        root.after(10, camera)
        
def setup_camera():
    root.title("Control Panel")
    root.geometry("200x100")
    close_button = tk.Button(root, text="Close Camera", command=close_window, bg="red", fg="white", font=("Arial", 12))
    close_button.pack(expand=True, fill='both', padx=20, pady=20)
    camera()
    root.mainloop()
