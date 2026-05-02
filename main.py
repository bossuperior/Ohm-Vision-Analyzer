import cv2
import numpy as np
import time
from src.vision.camera_loader import CameraLoader
from src.inference.model_engine import ModelEngine
from src.vision.breadboard_warper import BreadboardWarper

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

    camera.start()
    time.sleep(1)

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue

            # Detect on raw frame (matches training data perspective)
            results = engine.predict(frame)

            success, warped, matrix = transformer.process(frame)

            if success:
                display_frame = warped.copy()
                # Transform detected boxes/keypoints into warped coordinate space
                draw_results(display_frame, results, class_names, matrix=matrix)
                cv2.putText(display_frame, "BOARD OK [ArUco]",
                            (20, 40), cv2.FONT_HERSHEY_DUPLEX,
                            1.0, (0, 255, 0), 2)
            else:
                display_frame = frame.copy()
                # Draw raw detections on original frame while searching
                draw_results(display_frame, results, class_names, matrix=None)
                cv2.putText(display_frame, "SEARCHING FOR ARUCO TAGS...",
                            (20, 60), cv2.FONT_HERSHEY_DUPLEX,
                            1.0, (0, 0, 255), 2)

            cv2.putText(display_frame, "Press [Q] to Exit",
                        (10, display_frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

            cv2.imshow("Ohm-Vision", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
