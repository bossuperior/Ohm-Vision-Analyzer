import cv2
import numpy as np
from ultralytics import YOLO

class DetectionResult: #Result Data Structure
    def __init__(self, boxes, keypoints, class_ids):
        self.boxes = boxes
        self.keypoints = keypoints
        self.class_ids = class_ids

    def has_board(self):
        # Check if any detected object is the board (class_id == 0)
        return 0 in self.class_ids

    def get_board_corners(self):
        # Get the 4 corner points of the board
        board_idx = np.where(self.class_ids == 0)[0][0]
        return self.keypoints[board_idx][:4]

    def get_all_keypoints(self):
        # Return all keypoints for further analysis (e.g., circuit topology)
        return self.keypoints

# =====================
# THE ONNX MODEL ENGINE
# =====================
class ModelEngine:
    def __init__(self, model_path, model_type="yolov8", **kwargs):
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        print(f" Initializing ONNX Model Engine: {self.model_type.upper()}")
        
        if self.model_type == "yolov8":
            self.engine = YOLO(self.model_path, task="pose")
            print(f" Yolov8n Pose model (ONNX) loaded successfully")
        elif self.model_type == "rtm-pose":
            self.engine = self._init_rtmpose(**kwargs)
        elif self.model_type == "higherhrnet":
            self.engine = self._init_higherhrnet(**kwargs)
        else:
            raise ValueError(f" Not supported: {self.model_type}")

    # --------------------------
    # 1. YOLOv8 POSE ONNX ENGINE
    # --------------------------
    def _predict_yolo(self, frame):
        results = self.engine(frame, verbose=False)[0]
        
        # Extract boxes, keypoints, and class IDs from the results
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else np.array([])
        class_ids = results.boxes.cls.cpu().numpy() if results.boxes else np.array([])
        keypoints = results.keypoints.xy.cpu().numpy() if results.keypoints else np.array([])
        
        return DetectionResult(boxes, keypoints, class_ids)

    # -----------------------
    # 2. RTM-POSE ONNX ENGINE
    # -----------------------
    def _init_rtmpose(self, **kwargs):
        print(" Preparing RTM-Pose environment...")
        return None

    def _predict_rtmpose(self, frame):
        return DetectionResult(np.array([]), np.array([]), np.array([]))

    # --------------------------
    # 3. HigherHRNet ONNX ENGINE
    # --------------------------
    def _init_higherhrnet(self, **kwargs):
        print(" Preparing HigherHRNet environment...")
        return None

    def _predict_higherhrnet(self, frame):
        return DetectionResult(np.array([]), np.array([]), np.array([]))

    # ==========================
    # PUBLIC METHOD
    # ==========================
    def predict(self, frame):
        if self.model_type == "yolov8":
            return self._predict_yolo(frame)
        elif self.model_type == "rtm-pose":
            return self._predict_rtmpose(frame)
        elif self.model_type == "higherhrnet":
            return self._predict_higherhrnet(frame)

    def release_resources(self):
        if hasattr(self, 'engine') and self.engine is not None:
            del self.engine
        print(f" Resources for {self.model_type.upper()} released. Memory cleared.")