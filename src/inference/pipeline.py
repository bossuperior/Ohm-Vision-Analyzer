import cv2
import numpy as np

# Import all your modular tools
from src.utils.camera_loader import CameraLoader
from src.utils.point_smoother import PointSmoother
from src.inference.model_engine import PoseModel  # Your selected engine (YOLO/RTM/HRNet)
from src.vision.perspective_transform import img_transform
from src.vision.band_reader import BandReader
from src.vision.perspective_transform import order_points
from src.topology.grid_mapper import GridMapper
from src.topology.circuit_detector import CircuitDetector


class IntelligentBreadboardPipeline:
    def __init__(self, camera: CameraLoader, pose_model: PoseModel):
        self.camera = camera
        self.pose_model = pose_model
        
        # Initialize the new modules you mentioned
        self.band_reader = BandReader()
        self.grid_mapper = GridMapper()
        self.circuit_detector = CircuitDetector()
        self.is_running = False
        self.smoother = PointSmoother(alpha=0.15)

    def process_single_frame(self):
        ret, frame = self.camera.get_frame()
        if not ret:
            return

        clean_frame = frame.copy()
        display_frame = frame.copy()

        # Use AI instead of Canny Edge Detection!
        pose_data = self.pose_model.predict(clean_frame)
        board_corners = self._extract_corners(pose_data)

        if board_corners is not None and len(board_corners) == 4:

            # Sort the points first so we know which is which
            ordered_corners = order_points(board_corners)
            stable_corners = self.smoother.update(ordered_corners)

            # Draw the numbered points on the main camera feed
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # BGR
            labels = ["0-TL", "1-TR", "2-BR", "3-BL"]

            for i in range(4):
                x, y = int(stable_corners[i][0]), int(stable_corners[i][1])
                # Draw the dot
                cv2.circle(display_frame, (x, y), 8, colors[i], -1)
                # Draw the label next to the dot
                cv2.putText(display_frame, labels[i], (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

            # Crop and flatten the breadboard
            crop_board, transform_matrix = img_transform(clean_frame, stable_corners)
            cv2.imshow('Cropped Breadboard', crop_board)
        else:
            # If the AI loses sight of the board, reset the smoother's memory
            self.smoother.reset()

        cv2.imshow('Camera', display_frame)
        cv2.waitKey(1)

    def start(self):
        self.camera.start()
        self.is_running = True
        
        while self.is_running:
            ret, frame = self.camera.get_frame()
            if not ret:
                break
                
            # --- STEP 1: Detect Breadboard & Crop ---
            # Run your pose model to find the 4 corners of the breadboard
            pose_data = self.pose_model.predict(frame)
            board_corners = self._extract_corners(pose_data)
            
            # If we see a breadboard, zoom and crop into it
            if board_corners is not None:
                # Use OpenCV getPerspectiveTransform & warpPerspective (imported utility)
                working_frame = img_transform(frame, board_corners)
                
                # --- STEP 2: Detect Resistors/Wires & Read Bands ---
                # Now we look ONLY at the cropped breadboard for components
                # (You might use a separate object detection model here, or the same pose model)
                component_data = self.pose_model.predict(working_frame) 
                
                # Extract resistor values
                resistor_values, total_resistance = self.band_reader.calculate(
                    working_frame, component_data
                )
                
                # --- STEP 3: Map to Grid & Build NetworkX Graph ---
                # Translate (x,y) pixel coordinates to breadboard grid (A1, B5, etc.)
                grid_nodes = self.grid_mapper.map_to_holes(component_data)
                
                # Build the graph and determine Series/Parallel
                circuit_type, graph = self.circuit_detector.analyze_topology(
                    grid_nodes, resistor_values
                )
                
                # --- VISUALIZATION ---
                annotated_frame = self._draw_diagnostics(
                    working_frame, resistor_values, total_resistance, circuit_type
                )
                
                cv2.imshow("Intelligent Breadboard", annotated_frame)
                
            else:
                # If no breadboard detected, just show the normal camera feed
                cv2.imshow("Intelligent Breadboard", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

    def _extract_corners(self, pose_data):
        # Check if any keypoints were detected at all
        if len(pose_data['keypoints']) == 0:
            return None

        # Assuming the model outputs shape (num_objects, num_keypoints, 2)
        # We grab the keypoints for the first detected object (index 0)
        detected_points = pose_data['keypoints'][0]

        # Check if we have at least 4 points for the breadboard
        if len(detected_points) >= 4:
            # Slice the first 4 keypoints (the corners of the breadboard)
            corners = detected_points[:4]
            return np.array(corners, dtype="float32")

        return None
        
    def _draw_diagnostics(self, frame, values, total, c_type):
        """Helper to overlay the text and bounding boxes on the UI."""
        # Draw text like "R_Total: 1500 Ohms", "Type: Parallel"
        return frame

    def stop(self):
        self.is_running = False
        self.camera.stop()
        cv2.destroyAllWindows()