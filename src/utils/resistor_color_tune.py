import cv2
import numpy as np
import os
import sys

# Make the script runnable from any CWD by putting the project root on sys.path.
# File layout: <PROJECT_ROOT>/src/utils/resistor_color_tune.py  ->  parents[2] == project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.model_engine import ModelEngine
from src.vision.perspective_transform import PerspectiveTransformer, PointSmoother
from src.vision.camera_loader import CameraLoader

# Keyboard mapping for user inputs
COLOR_NAMES = {
    ord('0'): 'BLACK', ord('1'): 'BROWN', ord('2'): 'RED',
    ord('3'): 'ORANGE', ord('4'): 'YELLOW', ord('5'): 'GREEN',
    ord('6'): 'BLUE', ord('7'): 'VIOLET', ord('8'): 'GRAY',
    ord('9'): 'WHITE', ord('g'): 'GOLD', ord('s'): 'SILVER'
}
clicked_pt = (405, 270)

def mouse_callback(event, x, y, flags, param):
    global clicked_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pt = (x, y)

def save_calibration(calibrated_hsv):
    """Auto-generates color_mapping.py with the exact structure and calibrated values."""
    # Anchor to PROJECT_ROOT so saving works regardless of the current working directory.
    filepath = os.path.join(PROJECT_ROOT, "src", "vision", "color_mapping.py")
    
    base_refs = {
        'BLACK':  (0,   0,   45),
        'BROWN':  (10, 115, 100),
        'RED':    (2,  185, 155),
        'ORANGE': (13, 200, 190),
        'YELLOW': (26, 135, 175),
        'GOLD':   (22,  85, 148),
        'GREEN':  (60, 190, 150),
        'BLUE':   (112, 190, 150),
        'VIOLET': (132, 165, 135),
        'GRAY':   (0,   15, 140),
        'WHITE':  (0,   10, 220),
        'SILVER': (0,   18, 185),
    }
    
    for c, hsv in calibrated_hsv.items():
        base_refs[c] = tuple(hsv)
        
    content = "# ==========================================\n"
    content += "# CONFIG: Reference Colors (Tuned Version)\n"
    content += "# ==========================================\n"
    content += "import numpy as np\n\n"
    content += "REF_COLORS = {\n"
    
    for c, hsv in base_refs.items():
        content += f"    '{c}': {hsv},\n"
        
    content += "}\n\n"
    content += "COLOR_VALS = {\n"
    content += "    'BLACK': 0, 'BROWN': 1, 'RED': 2, 'ORANGE': 3, 'YELLOW': 4,\n"
    content += "    'GREEN': 5, 'BLUE': 6, 'VIOLET': 7, 'GRAY': 8, 'WHITE': 9,\n"
    content += "    'GOLD': -1, 'SILVER': -2\n"
    content += "}\n\n"
    
    content += "BODY_LOWER = np.array([5, 50, 80])\n"
    content += "BODY_UPPER = np.array([35, 180, 255])\n"
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"\n SUCCESS! Auto-saved perfectly formatted config to: {filepath}")

def main():
    global clicked_pt
    # ==========================
    # System Initialization
    # ==========================
    camera_id = 1
    camera = CameraLoader(camera_id=camera_id)
    camera.start()

    print(" Loading YOLO Model for Breadboard Detection...")
    model_path = os.path.join(PROJECT_ROOT, "models", "Yolo_v8n_pose_weights.onnx")
    engine = ModelEngine(model_path=model_path, model_type="yolov8")
    transformer = PerspectiveTransformer()
    point_smoother = PointSmoother()
    calibrated_hsv = {}

    cv2.namedWindow("Ohm-Vision Color Tuner")
    cv2.setMouseCallback("Ohm-Vision Color Tuner", mouse_callback)

    print("=========================================")
    print("  Smart Resistor Color Tuner v2.0 ")
    print("=========================================")
    print("""INSTRUCTIONS:
    1.  Click at color band on the resistor
    2.  Press 0-9 or g/s to save color code at the pixel
       Press 0 = BLACK
       Press 1 = BROWN
       Press 2 = RED
       Press 3 = ORANGE
       Press 4 = YELLOW
       Press 5 = GREEN
       Press 6 = BLUE
       Press 7 = VIOLET
       Press 8 = GRAY
       Press 9 = WHITE
       Press g = GOLD
       Press s = SILVER

    4. Press 'Enter' to SAVE color_mapping.py file
    5. Press 'q' to quit without saving""")
    print("=========================================")

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        detection_results = engine.predict(frame)
        if detection_results.has_board():
            board_corners = detection_results.get_board_corners()
            stable_corners = point_smoother.update(board_corners)
            display_frame, _ = transformer.warp(frame, stable_corners)
        else:
            display_frame = frame.copy()
            cv2.putText(display_frame, "NO BOARD DETECTED - PLEASE SHOW BREADBOARD", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        h, w = display_frame.shape[:2]
        box_size = 12
        
        cx, cy = clicked_pt
        cx = max(box_size, min(w - box_size, cx))
        cy = max(box_size, min(h - box_size, cy))

        cv2.rectangle(display_frame, (cx - box_size, cy - box_size), (cx + box_size, cy + box_size), (0, 255, 0), 2)
        cv2.circle(display_frame, (cx, cy), 2, (0, 0, 255), -1)

        y_offset = 30
        cv2.putText(display_frame, "Calibrated Colors:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(display_frame, "Calibrated Colors:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, (c_name, hsv) in enumerate(calibrated_hsv.items()):
            text = f"{c_name}: {hsv}"
            y_pos = y_offset + 25 * (i + 1)
            cv2.putText(display_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(display_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Ohm-Vision Color Tuner", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Exiting without saving...")
            break
        elif key == 13: # Enter key
            save_calibration(calibrated_hsv)
            break
        elif key in COLOR_NAMES:
            color_name = COLOR_NAMES[key]
            y1, y2 = cy - box_size, cy + box_size
            x1, x2 = cx - box_size, cx + box_size
            roi = display_frame[y1:y2, x1:x2]
            
            if roi.size > 0:
                # Apply CLAHE logic
                lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                roi_balanced = cv2.merge((l, a, b))
                roi_bright = cv2.cvtColor(roi_balanced, cv2.COLOR_LAB2BGR)
                
                # Convert to HSV and mean
                hsv_roi = cv2.cvtColor(roi_bright, cv2.COLOR_BGR2HSV)
                mean_hsv = cv2.mean(hsv_roi)[:3]
                
                calibrated_hsv[color_name] = [int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])]
                print(f" Sampled [{color_name}] at (x:{cx}, y:{cy}) -> HSV: {calibrated_hsv[color_name]}")

    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()