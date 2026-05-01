"""
resistor_color_tune.py
-----------------------
Interactive tool for calibrating band color HSV values.
Click on a color band → press the matching key → press Enter to save.

Keys: 0=BLACK 1=BROWN 2=RED 3=ORANGE 4=YELLOW 5=GREEN
      6=BLUE  7=VIOLET 8=GRAY  9=WHITE  g=GOLD  s=SILVER
      Enter=Save   q=Quit without saving
"""
import cv2
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.model_engine import ModelEngine
from src.vision.perspective_transform import PerspectiveTransformer, PointSmoother
from src.vision.camera_loader import CameraLoader

COLOR_NAMES = {
    ord('0'): 'BLACK', ord('1'): 'BROWN', ord('2'): 'RED',
    ord('3'): 'ORANGE', ord('4'): 'YELLOW', ord('5'): 'GREEN',
    ord('6'): 'BLUE', ord('7'): 'VIOLET', ord('8'): 'GRAY',
    ord('9'): 'WHITE', ord('g'): 'GOLD', ord('s'): 'SILVER',
}

clicked_pt   = (405, 270)
display_size = (810, 540)   # ค่า default ก่อนรู้ขนาด frame จริง


def mouse_callback(event, x, y, flags, param):
    global clicked_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pt = (x, y)


def save_calibration(calibrated_hsv: dict):
    """เขียน color_mapping.py ใหม่ด้วยค่าที่ calibrate แล้ว"""
    filepath = os.path.join(PROJECT_ROOT, "src", "vision", "color_mapping.py")

    # Base reference — ใช้ค่าที่แก้ไขแล้ว (VIOLET = 138, ไม่ใช่ 68)
    base_refs = {
        'BLACK':  (0,    0,   45),
        'BROWN':  (12,  140,  130),
        'RED':    (2,   185,  155),
        'ORANGE': (13,  200,  190),
        'YELLOW': (26,  135,  175),
        'GOLD':   (25,  120,  190),
        'GREEN':  (60,  190,  150),
        'BLUE':   (112, 190,  150),
        'VIOLET': (138, 130,  160),   # ✅ ค่าที่ถูกต้อง (ไม่ใช่ 132 หรือ 68)
        'GRAY':   (0,    15,  140),
        'WHITE':  (0,    10,  220),
        'SILVER': (0,    18,  185),
    }

    # override ด้วยค่าที่ผู้ใช้ calibrate
    for c, hsv in calibrated_hsv.items():
        base_refs[c] = tuple(hsv)

    lines = [
        "# ==========================================",
        "# CONFIG: Reference Colors (Tuned Version)",
        "# ==========================================",
        "import numpy as np",
        "",
        "REF_COLORS = {",
    ]
    for c, hsv in base_refs.items():
        lines.append(f"    '{c}': {hsv},")
    lines += [
        "}",
        "",
        "COLOR_VALS = {",
        "    'BLACK': 0, 'BROWN': 1, 'RED': 2, 'ORANGE': 3, 'YELLOW': 4,",
        "    'GREEN': 5, 'BLUE': 6, 'VIOLET': 7, 'GRAY': 8, 'WHITE': 9,",
        "    'GOLD': -1, 'SILVER': -2",
        "}",
        "",
        "BODY_LOWER = np.array([5, 50, 80])",
        "BODY_UPPER = np.array([35, 180, 255])",
        "",
    ]

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n✅ Saved color_mapping.py → {filepath}")


def main():
    global clicked_pt, display_size

    camera = CameraLoader(camera_id=1)
    camera.start()

    model_path   = os.path.join(PROJECT_ROOT, "models", "Yolo_v8n_pose_weights.onnx")
    engine       = ModelEngine(model_path=model_path, model_type="yolov8")
    transformer  = PerspectiveTransformer()
    smoother     = PointSmoother()
    calibrated_hsv: dict = {}

    WIN = "Ohm-Vision Color Tuner"
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, mouse_callback)

    print("=" * 45)
    print("  Ohm-Vision Color Tuner")
    print("=" * 45)
    print("1. แสดงบอร์ดให้กล้อง → ภาพจะถูก warp อัตโนมัติ")
    print("2. Click บนแถบสี → กด 0-9 / g / s เพื่อบันทึก")
    print("3. Enter = บันทึก color_mapping.py")
    print("4. q = ออกโดยไม่บันทึก")
    print("=" * 45)

    last_valid_corners = None
    board_miss_count   = 0
    MISS_TOLERANCE     = 8

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        results = engine.predict(frame)

        # ── Board corner tracking ─────────────────────────────────────
        if results.has_board():
            raw_corners = results.get_board_corners()   # shape (4, 2)

            # ✅ Fix #1: ตรวจ corners ว่า valid ก่อน warp
            if transformer.validate_corners(raw_corners):
                last_valid_corners = raw_corners
                board_miss_count   = 0
            else:
                board_miss_count += 1
        else:
            board_miss_count += 1

        # ── Warp / fallback ──────────────────────────────────────────
        display_frame = None

        if board_miss_count <= MISS_TOLERANCE and last_valid_corners is not None:
            stable = smoother.update(last_valid_corners)

            # ✅ Fix #2: warp อาจ return (None, _) → ต้อง check ก่อนใช้
            warped, _ = transformer.warp(frame, stable)
            if warped is not None and warped.std() > 15:
                display_frame = warped
            else:
                smoother.reset()

        if display_frame is None:
            # Fallback: แสดง frame ดิบ พร้อม warning
            display_frame = frame.copy()
            cv2.putText(display_frame,
                        "NO BOARD — point camera at breadboard",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        # ── ✅ Fix #3: อัปเดต display_size ตาม frame จริง ────────────
        #    ป้องกัน clicked_pt ชี้นอก frame เมื่อ warp เปลี่ยนขนาด
        display_size = (display_frame.shape[1], display_frame.shape[0])
        h, w = display_frame.shape[:2]

        # Clamp cursor ให้อยู่ใน frame
        box_size = 12
        cx = max(box_size, min(w - box_size - 1, clicked_pt[0]))
        cy = max(box_size, min(h - box_size - 1, clicked_pt[1]))

        # วาด crosshair
        cv2.rectangle(display_frame,
                      (cx - box_size, cy - box_size),
                      (cx + box_size, cy + box_size),
                      (0, 255, 0), 2)
        cv2.circle(display_frame, (cx, cy), 2, (0, 0, 255), -1)

        # แสดงสีที่ calibrate แล้ว
        y_off = 30
        cv2.putText(display_frame, "Calibrated:", (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(display_frame, "Calibrated:", (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for i, (c_name, hsv) in enumerate(calibrated_hsv.items()):
            y_pos = y_off + 25 * (i + 1)
            text  = f"{c_name}: H={hsv[0]} S={hsv[1]} V={hsv[2]}"
            cv2.putText(display_frame, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(display_frame, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

        cv2.imshow(WIN, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quit without saving.")
            break

        elif key == 13:  # Enter
            save_calibration(calibrated_hsv)
            break

        elif key in COLOR_NAMES:
            color_name = COLOR_NAMES[key]
            y1 = max(0, cy - box_size);  y2 = min(h, cy + box_size)
            x1 = max(0, cx - box_size);  x2 = min(w, cx + box_size)
            roi = display_frame[y1:y2, x1:x2]

            if roi.size == 0:
                print(f"  [skip] ROI ว่างเปล่า")
                continue

            # CLAHE บน LAB แล้วแปลงเป็น HSV
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            roi_balanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

            hsv_roi  = cv2.cvtColor(roi_balanced, cv2.COLOR_BGR2HSV)
            mean_hsv = cv2.mean(hsv_roi)[:3]

            calibrated_hsv[color_name] = [int(mean_hsv[0]),
                                           int(mean_hsv[1]),
                                           int(mean_hsv[2])]
            print(f"  [{color_name}] @ ({cx},{cy}) → HSV {calibrated_hsv[color_name]}")

    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
