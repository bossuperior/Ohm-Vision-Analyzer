import cv2
import numpy as np
import time
from src.vision.camera_loader import CameraLoader
from src.inference.model_engine import ModelEngine
from src.vision.perspective_transform import PerspectiveTransformer, PointSmoother


def find_board_corners_cv(frame: np.ndarray,
                           bx1: int, by1: int, bx2: int, by2: int):
    margin = 30
    x1 = max(0, bx1 - margin);  y1 = max(0, by1 - margin)
    x2 = min(frame.shape[1], bx2 + margin)
    y2 = min(frame.shape[0],  by2 + margin)
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive threshold เพื่อจับขอบแม้แสงไม่สม่ำเสมอ
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (crop.shape[0] * crop.shape[1]) * 0.10:
            continue   # เล็กเกินไป — ไม่ใช่บอร์ด
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_area = area
            best_quad = approx

    if best_quad is None:
        return None

    # แปลงกลับ full-frame coordinates
    corners = best_quad.reshape(4, 2).astype(float)
    corners[:, 0] += x1
    corners[:, 1] += y1
    return corners


def main():
    print(" Starting Ohm-Vision — Board Detection Test")

    camera       = CameraLoader(camera_id=1)
    engine       = ModelEngine(model_path="models/Yolo_v8n_pose_weights.onnx",
                                model_type="yolov8")
    transformer  = PerspectiveTransformer()
    point_smoother = PointSmoother()

    last_valid_box  = None
    board_miss_count = 0
    MISS_TOLERANCE   = 8

    camera.start()
    time.sleep(1)

    # Mode toggle ด้วย keyboard:  'p'=perspective  'b'=bbox  'c'=cv-corners
    current_mode = "cv"   # เริ่มด้วย classical CV corners

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue

            results = engine.predict(frame)
            display_frame = frame.copy()

            # ── อัปเดต bbox ───────────────────────────────────────
            conf = 0.0
            if results.has_board():
                board_idx = np.where(results.class_ids == 0)[0][0]
                box  = results.boxes[board_idx]
                conf = float(results.scores[board_idx])
                last_valid_box = tuple(map(int, box[:4]))
                board_miss_count = 0
            else:
                board_miss_count += 1

            board_box = (last_valid_box
                         if board_miss_count <= MISS_TOLERANCE else None)

            # ── Board visible ─────────────────────────────────────
            if board_box is not None:
                bx1, by1, bx2, by2 = board_box

                # วาด bbox บน original frame
                cv2.rectangle(display_frame, (bx1, by1), (bx2, by2),
                               (0, 255, 0), 2)

                warped = None
                used_method = current_mode

                # ── Mode: Classical CV corners ────────────────────
                if current_mode == "cv":
                    cv_corners = find_board_corners_cv(
                        frame, bx1, by1, bx2, by2)

                    if cv_corners is not None:
                        # วาด CV corners บน original frame
                        for pt in cv_corners:
                            cv2.circle(display_frame,
                                       (int(pt[0]), int(pt[1])),
                                       10, (0, 165, 255), -1)  # สีส้ม

                        if transformer.validate_corners(cv_corners):
                            stable = point_smoother.update(cv_corners)
                            candidate, _ = transformer.warp(frame, stable)
                            if candidate is not None and candidate.std() > 15:
                                warped = candidate
                                used_method = "cv-corners"
                            else:
                                point_smoother.reset()

                # ── Mode: Model keypoints ─────────────────────────
                elif current_mode == "perspective":
                    if results.has_board():
                        board_idx = np.where(results.class_ids == 0)[0][0]
                        model_kpts = results.keypoints[board_idx][:4]

                        # วาด model keypoints (สีเหลือง)
                        for kp in model_kpts:
                            cv2.circle(display_frame,
                                       (int(kp[0]), int(kp[1])),
                                       8, (0, 255, 255), -1)  # สีเหลือง

                        if transformer.validate_corners(model_kpts):
                            stable = point_smoother.update(model_kpts)
                            candidate, _ = transformer.warp(frame, stable)
                            if candidate is not None and candidate.std() > 15:
                                warped = candidate
                                used_method = "model-kpts"
                            else:
                                point_smoother.reset()

                # ── Fallback / Mode: BBox crop ────────────────────
                if warped is None:
                    pad = 6
                    cx1 = max(0, bx1 - pad);  cy1 = max(0, by1 - pad)
                    cx2 = min(frame.shape[1], bx2 + pad)
                    cy2 = min(frame.shape[0], by2 + pad)
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        warped = cv2.resize(crop, (810, 540))
                    used_method = "bbox-fallback"
                    point_smoother.reset()

                if warped is not None:
                    display_frame = warped.copy()

                # ── Labels ────────────────────────────────────────
                cv2.putText(display_frame, "BOARD OK",
                            (20, 40), cv2.FONT_HERSHEY_DUPLEX,
                            1.0, (0, 255, 0), 2)
                cv2.putText(display_frame, f"conf: {conf:.2f}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 220, 0), 2)
                cv2.putText(display_frame, f"[{used_method}]",
                            (display_frame.shape[1] - 220, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (100, 255, 100), 2)
                cv2.putText(display_frame,
                            "Keys: [c]=CV corners  [p]=Model kpts  [b]=BBox",
                            (10, display_frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (200, 200, 0), 1)

            else:
                cv2.putText(display_frame, "NO BOARD DETECTED",
                            (20, 60), cv2.FONT_HERSHEY_DUPLEX,
                            1.0, (0, 0, 255), 2)

            cv2.imshow("Ohm-Vision — Board Test", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                current_mode = "cv"
                point_smoother.reset()
                print(" Mode: Classical CV corners")
            elif key == ord('p'):
                current_mode = "perspective"
                point_smoother.reset()
                print(" Mode: Model keypoints")
            elif key == ord('b'):
                current_mode = "bbox"
                point_smoother.reset()
                print(" Mode: BBox crop")

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
