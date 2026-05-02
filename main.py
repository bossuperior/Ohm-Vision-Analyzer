import cv2
import numpy as np
import time
from src.vision.camera_loader import CameraLoader
from src.inference.model_engine import ModelEngine
from src.vision.breadboard_warper import BreadboardWarper

def main():
    print(" Starting Ohm-Vision\n")
    camera       = CameraLoader(camera_id=1)
    engine       = ModelEngine(model_path="models/Yolo_v8n_pose_weights.onnx",
                                model_type="yolov8")
    transformer  = BreadboardWarper(output_width=810, output_height=540)

    last_valid_box  = None
    board_miss_count = 0
    MISS_TOLERANCE   = 8

    camera.start()
    time.sleep(1)
    current_mode = "aruco"

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue

            results = engine.predict(frame)
            display_frame = frame.copy()
            #B-Box
            conf = 0.0
            if results.has_board():
                board_idx = np.where(results.class_ids == 0)[0][0]
                conf = float(results.scores[board_idx])
            success, warped = transformer.process(frame)

            if success:
                display_frame = warped.copy()

                cv2.putText(display_frame, "BOARD OK [ArUco]",
                            (20, 40), cv2.FONT_HERSHEY_DUPLEX,
                            1.0, (0, 255, 0), 2)
                cv2.putText(display_frame, f"YOLO conf: {conf:.2f}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 220, 0), 2)
            else:
                cv2.putText(display_frame, "SEARCHING FOR ARUCO TAGS...",
                            (20, 60), cv2.FONT_HERSHEY_DUPLEX,
                            1.0, (0, 0, 255), 2)
                
                cv2.putText(display_frame,
                        "Press [Q] to Exit",
                        (10, display_frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 0), 1)

                cv2.imshow("Ohm-Vision — Board Test", display_frame)
            
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
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
