import cv2
import numpy as np
import time
from collections import Counter, deque
from src.vision.camera_loader import CameraLoader
from src.inference.model_engine import ModelEngine
from src.vision.perspective_transform import PerspectiveTransformer, PointSmoother
from src.vision.resist_body_detector import BodyDetector, TemporalFilter
from src.vision.band_reader import BandReader
from src.vision.band_detector import BandDetector
from src.topology.circuit_detector import CircuitDetector
from src.topology.grid_mapper import GridMapper

def main():
    print(" Starting Ohm-Vision Analyzer Pipeline...")

    # =======================
    # PHASE 1: Initialization
    # =======================
    camera = CameraLoader(camera_id=1)
    
    # Model Engine Loader
    engine = ModelEngine(model_path="models/Yolo_v8n_pose_weights.onnx", model_type="yolov8")
    
    # Vision & Analysis Modules
    transformer = PerspectiveTransformer()
    body_detector = BodyDetector()
    band_detector = BandDetector()
    band_reader = BandReader()
    circuit_detector = CircuitDetector()
    point_smoother = PointSmoother()
    grid_mapper = GridMapper()

    temporal_filter = TemporalFilter(history_size=7)
    circuit_history = deque(maxlen=10)
    resistance_history = deque(maxlen=10)

    camera.start()
    time.sleep(1)

    try:
        # ===========================
        # PHASE 2: The Real-Time Loop
        # ===========================
        while True:
            # Get a frame from the camera
            frame = camera.get_frame()
            if frame is None:
                continue

            # Send the frame to the Model Engine for inference
            detection_results = engine.predict(frame)
            display_frame = frame.copy()

            # Check if we detected the board
            if detection_results.has_board():
                # Get 4 corner points of the board for perspective transform
                board_corners = detection_results.get_board_corners()
                stable_corners = point_smoother.update(board_corners)
                warped_board, matrix = transformer.warp(frame, stable_corners)
                display_frame = warped_board.copy()

                # display_frame = grid_mapper.draw_grid_overlay(display_frame)

                board_results = engine.predict(warped_board)

                # Crop out the resistors based on detected keypoints and bounding boxes
                resistors = body_detector.extract_resistors(warped_board, board_results)
                
                detected_ohms = []
                resistor_indices = np.where(board_results.class_ids == 1)[0]
                for draw_idx, res in enumerate(resistors):
                    # Read the color bands from the cropped resistor image and calculate the Ohm value
                    resistance_str, bands, total_ohms = band_reader.calculate(res.image_crop)
                    global_id = resistor_indices[res.id] if res.id < len(resistor_indices) else res.id
                    resistance_str, total_ohms = temporal_filter.update_and_get(global_id, resistance_str, total_ohms)
                    detected_ohms.append({
                        "id": global_id,
                        "string_val": resistance_str,
                        "numeric_val": total_ohms,
                        "keypoints": res.keypoints
                    })

                    # Offset each label by draw_idx rows to prevent overlap when boxes cluster
                    tx, ty = res.text_position
                    label_pos = (tx, max(10, ty - draw_idx * 40))
                    cv2.putText(display_frame, resistance_str, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Debug: show raw detected band colors so you can tune REF_COLORS
                    band_colors = " | ".join(b['color'] for b in bands) if bands else "no bands"
                    debug_pos = (tx, min(display_frame.shape[0] - 10, label_pos[1] + 20))
                    cv2.putText(display_frame, band_colors, debug_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

                component_data = []
                for i in range(len(board_results.boxes)):
                    cls_id = board_results.class_ids[i]
                    if cls_id in [1, 2]: # 1: Resistor, 2: Wire
                        component_data.append({
                            'id': i,
                            'keypoints': board_results.keypoints[i]
                        })
                #Mapping detected components to the breadboard grid and determining their electrical nodes
                mapped_components = grid_mapper.map_to_holes(component_data)
                circuit_type, G, total_r = circuit_detector.analyze_topology(mapped_components, detected_ohms)

                # ========================
                # Temporal Smoothing Logic
                # ========================
                if circuit_type is not None:
                    circuit_history.append(circuit_type)
                    resistance_history.append(total_r)

                # --- Circuit Voting ---
                if len(circuit_history) >= 3:
                    stable_circuit_type = Counter(circuit_history).most_common(1)[0][0]
                else:
                    stable_circuit_type = circuit_type if circuit_type is not None else "UNKNOWN"

                # --- Resistance Voting ---
                inf_count = sum(1 for r in resistance_history if r == float('inf'))

                if inf_count > len(resistance_history) / 2:
                    stable_r = float('inf')
                else:
                    clean_r = [round(r, 1) for r in resistance_history if r != float('inf')]
                    stable_r = Counter(clean_r).most_common(1)[0][0] if clean_r else float('inf')

                cv2.putText(display_frame,
                            f"Topology: {stable_circuit_type}",
                            (20, 40),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.8,
                            (255, 0, 255),
                            2)

                if stable_r > 0 and stable_r != float('inf'):
                    cv2.putText(display_frame,
                                f"R Total: {stable_r} Ohms",
                                (20, 70),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.8,
                                (0, 255, 255),
                                2)

                elif stable_r == float('inf'):
                    cv2.putText(display_frame,
                                f"R Total: OPEN CIRCUIT",
                                (20, 70),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.8,
                                (0, 0, 255),
                                2)
            else:
                circuit_history.append("NO BOARD")
                resistance_history.append(float('inf'))
                cv2.putText(display_frame,
                            "NO BOARD DETECTED",
                            (20, 110),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.8,
                            (0, 0, 255),
                            2)
            # ===============================
            # PHASE 3: Display & Exit Control
            # ===============================
            cv2.imshow("Ohm-Vision Live Analyzer", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(" Exiting program...")
                break

    except Exception as e:
        print(f" Error occurred: {e}")

    finally:
        # ================
        # PHASE 4: Cleanup
        # ================
        camera.stop()
        cv2.destroyAllWindows()
        print("Ohm-Vision Analyzer Pipeline safely terminated.")

if __name__ == "__main__":
    main()