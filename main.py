import cv2
import numpy as np
import time
from src.vision.camera_loader import CameraLoader
from src.inference.model_engine import ModelEngine
from src.vision.breadboard_warper import BreadboardWarper
from src.topology.grid_mapper import GridMapper

BOX_COLORS = {
    "resistor": (0, 200, 255),
    "wire":     (255, 180, 0),
}
DEFAULT_BOX_COLOR = (180, 180, 180)

# resistor keypoint colors: Leg_0, Leg_1, Body_2, Body_3
RESISTOR_KP_COLORS = [
    (0, 0, 255),    # Leg_0  — แดง
    (0, 0, 255),    # Leg_1  — แดง
    (0, 255, 0),    # Body_2 — เขียว
    (0, 255, 0),    # Body_3 — เขียว
]


def transform_points(pts, matrix):
    """Apply perspective matrix to an array of (x, y) points."""
    if len(pts) == 0:
        return pts
    src = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(src, matrix)
    return dst.reshape(-1, 2)


def transform_box(box, matrix):
    """Transform xyxy box through perspective matrix, return new xyxy."""
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    transformed = transform_points(corners, matrix)
    nx1 = int(transformed[:, 0].min())
    ny1 = int(transformed[:, 1].min())
    nx2 = int(transformed[:, 0].max())
    ny2 = int(transformed[:, 1].max())
    return nx1, ny1, nx2, ny2


def draw_results(frame, results, class_names, matrix=None):
    for i, (box, cls_id, score) in enumerate(
            zip(results.boxes, results.class_ids, results.scores)):
        name  = class_names.get(int(cls_id), f"cls{int(cls_id)}")
        color = BOX_COLORS.get(name, DEFAULT_BOX_COLOR)

        if matrix is not None:
            x1, y1, x2, y2 = transform_box(box, matrix)
        else:
            x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {score:.2f}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if len(results.keypoints) > i:
            kp_data = results.keypoints[i]  # (K, 3): x, y, conf
            is_resistor = (name == "resistor")

            # collect valid keypoints with their original index
            raw_kps  = [(j, kp[:2]) for j, kp in enumerate(kp_data) if kp[2] > 0.5]
            xy_only  = [kp for _, kp in raw_kps]

            if matrix is not None and xy_only:
                xy_only = transform_points(xy_only, matrix)

            kp_map = {j: tuple(map(int, xy_only[k])) for k, (j, _) in enumerate(raw_kps)}

            for j, pt in kp_map.items():
                color = RESISTOR_KP_COLORS[j] if is_resistor and j < len(RESISTOR_KP_COLORS) else (0, 255, 255)
                cv2.circle(frame, pt, 5, color, -1)

            # วาดเส้น Body_2 → Body_3 สำหรับ resistor
            if is_resistor and 2 in kp_map and 3 in kp_map:
                cv2.line(frame, kp_map[2], kp_map[3], (0, 255, 0), 2)


def main():
    print(" Starting Ohm-Vision\n")
    camera      = CameraLoader(camera_id=1)
    engine      = ModelEngine(model_path="models/Yolo_v8n_pose_weights.onnx",
                               model_type="yolov8")
    transformer = BreadboardWarper(output_width=810, output_height=540)

    class_names = engine.engine.names

    grid_mapper  = GridMapper(target_w=810, target_h=540)
    show_grid    = False

    WIN = "Ohm-Vision"
    cv2.namedWindow(WIN)
    # Crop trackbars — Margin + ShiftX/Y (0-200, center=100)
    cv2.createTrackbar("Margin",      WIN, transformer.margin,      300, lambda _: None)
    cv2.createTrackbar("Shift X",     WIN, 100 + transformer.shift_x, 200, lambda _: None)
    cv2.createTrackbar("Shift Y",     WIN, 100 + transformer.shift_y, 200, lambda _: None)
    # GridMapper calibration trackbars (PitchX/Y = ค่าจริง × 10 เพื่อให้ปรับทีละ 0.1px)
    cv2.createTrackbar("Grid OffX",   WIN, grid_mapper.offset_x,              200, lambda _: None)
    cv2.createTrackbar("Grid OffY",   WIN, grid_mapper.offset_y,              200, lambda _: None)
    cv2.createTrackbar("Grid PitchX", WIN, int(grid_mapper.pitch_x * 10),     500, lambda _: None)
    cv2.createTrackbar("Grid PitchY", WIN, int(grid_mapper.pitch_y * 10),     500, lambda _: None)

    camera.start()
    time.sleep(1)

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue

            # อ่านค่า crop realtime
            transformer.margin  = cv2.getTrackbarPos("Margin",  WIN)
            transformer.shift_x = cv2.getTrackbarPos("Shift X", WIN) - 100
            transformer.shift_y = cv2.getTrackbarPos("Shift Y", WIN) - 100

            # อ่านค่า GridMapper realtime
            off_x   = cv2.getTrackbarPos("Grid OffX",   WIN)
            off_y   = cv2.getTrackbarPos("Grid OffY",   WIN)
            pitch_x = cv2.getTrackbarPos("Grid PitchX", WIN) / 10.0
            pitch_y = cv2.getTrackbarPos("Grid PitchY", WIN) / 10.0
            grid_mapper.set_params(off_x, off_y, pitch_x=pitch_x, pitch_y=pitch_y)

            results = engine.predict(frame)
            success, warped, matrix = transformer.process(frame)

            if success:
                display_frame = warped.copy()
                if show_grid:
                    display_frame = grid_mapper.draw_grid_overlay(display_frame)
                draw_results(display_frame, results, class_names, matrix=matrix)
                cv2.putText(display_frame, "BOARD OK [ArUco]",
                            (20, 40), cv2.FONT_HERSHEY_DUPLEX,
                            1.0, (0, 255, 0), 2)
                t = transformer
                cv2.putText(display_frame,
                            f"Margin:{t.margin} ShiftX:{t.shift_x} ShiftY:{t.shift_y}  |  OffX:{off_x} OffY:{off_y} PX:{pitch_x:.1f} PY:{pitch_y:.1f}",
                            (10, display_frame.shape[0] - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
            else:
                display_frame = frame.copy()
                draw_results(display_frame, results, class_names, matrix=None)
                cv2.putText(display_frame, "SEARCHING FOR ARUCO TAGS...",
                            (20, 60), cv2.FONT_HERSHEY_DUPLEX,
                            1.0, (0, 0, 255), 2)

            hint = "[Q] Exit  [G] Toggle Grid"
            cv2.putText(display_frame, hint,
                        (10, display_frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

            cv2.imshow(WIN, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                show_grid = not show_grid

    except Exception as e:
        import traceback
        print(f" Error: {e}")
        traceback.print_exc()
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print(" Done.")


if __name__ == "__main__":
    main()
