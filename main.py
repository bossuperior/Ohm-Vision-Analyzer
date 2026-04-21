import cv2
import numpy as np
import time
from src.vision.camera_loader import CameraLoader
from src.inference.model_engine import ModelEngine
from src.vision.perspective_transform import PerspectiveTransformer, PointSmoother
from src.vision.resist_body_detector import BodyDetector
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

                display_frame = grid_mapper.draw_grid_overlay(display_frame)

                board_results = engine.predict(warped_board)

                # Crop out the resistors based on detected keypoints and bounding boxes
                resistors = body_detector.extract_resistors(warped_board, board_results)
                
                detected_ohms = []
                resistor_indices = np.where(board_results.class_ids == 1)[0]
                for res in resistors:
                    # Read the color bands from the cropped resistor image and calculate the Ohm value
                    resistance_str, bands, total_ohms = band_reader.calculate(res.image_crop)
                    global_id = resistor_indices[res.id] if res.id < len(resistor_indices) else res.id
                    detected_ohms.append({
                        "id": global_id,
                        "string_val": resistance_str, 
                        "numeric_val": total_ohms,   
                        "keypoints": res.keypoints
                    })
                    
                    # Draw the detected resistor and its value on the display frame
                    cv2.putText(display_frame, resistance_str, res.text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
                
                cv2.putText(display_frame, f"Topology: {circuit_type}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 255), 2)
                if total_r > 0 and total_r != float('inf'):
                    cv2.putText(display_frame, f"R Total: {total_r} Ohms", (20, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
                elif total_r == float('inf'):
                    cv2.putText(display_frame, f"R Total: OPEN CIRCUIT", (20, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

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