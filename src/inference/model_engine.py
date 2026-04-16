import numpy as np
import torch
import gc
from abc import ABC, abstractmethod

class PoseModel(ABC):
    """Abstract base class ensuring all models have the same interface."""
    def __init__(self, weights_path: str, **kwargs):
        self.weights_path = weights_path
        self.model = None
        # Capture any extra arguments (like config files for RTMPose)
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def predict(self, frame: np.ndarray) -> dict:
        pass
        
    @abstractmethod
    def release_resources(self):
        pass


class HigherHRNetEngine(PoseModel):
    def _load_model(self):
        print("Initializing HigherHRNet Engine...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            # Inline the model loader logic here
            self.model = torch.load(
                self.weights_path,
                map_location=self.device,
                weights_only=False  # Change to True if loading a state_dict in the future
            )

            # Set to evaluation mode for inference
            if hasattr(self.model, 'eval'):
                self.model.eval()

            print(f"✅ HigherHRNet loaded successfully to {self.device} from {self.weights_path}")

        except Exception as e:
            print(f"Error loading HigherHRNet model: {e}")
            self.model = None

    def predict(self, frame: np.ndarray) -> dict:
        if self.model is None:
             return {'boxes': [], 'keypoints': []}
             
        # Preprocess frame -> Tensor -> self.model(tensor) -> Postprocess -> Return standardized format
        return {'boxes': [], 'keypoints': []}

    def release_resources(self):
        """Inline the model unloading logic to free VRAM"""
        if self.model is not None:
            del self.model
            self.model = None
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            print("♻️ HigherHRNet resources released and memory freed.")


class YOLOPoseEngine(PoseModel):
    def _load_model(self):
        from ultralytics import YOLO
        print("Initializing YOLOv8 Engine...")
        self.model = YOLO(self.weights_path,task = 'pose')

    def predict(self, frame: np.ndarray) -> dict:
        if self.model is None:
             return {'boxes': [], 'keypoints': []}
             
        results = self.model(frame, verbose=False, half=True)[0] # half=True saves energy
        return {
            'boxes': results.boxes.xyxy.cpu().numpy() if results.boxes else [],
            'keypoints': results.keypoints.xy.cpu().numpy() if results.keypoints else []
        }

    def release_resources(self):
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("♻️ YOLOv8 resources released.")


class RTMPoseEngine(PoseModel):
    def __init__(self, pose_config: str, weights_path: str, det_model: str = 'yolox_tiny'):
        # RTMPose (Top-Down) usually requires a bounding box detector first.
        self.pose_config = pose_config
        self.det_model = det_model
        super().__init__(weights_path)

    def _load_model(self):
        print("Initializing RTMPose Engine...")
        try:
            from mmpose.apis import MMPoseInferencer
            
            # MMPoseInferencer handles both the detector and the pose estimator
            self.model = MMPoseInferencer(
                pose2d=self.pose_config,
                pose2d_weights=self.weights_path,
                det_model=self.det_model, 
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("✅ RTMPose loaded successfully.")
        except ImportError:
            print("Error: mmpose is not installed. Please run 'pip install mmpose'.")
            self.model = None

    def predict(self, frame: np.ndarray) -> dict:
        if self.model is None:
            return {'boxes': [], 'keypoints': []}

        # MMPose inferencer returns a generator. We use next() to get the frame's result.
        result_generator = self.model(frame, return_datasamples=False, show=False)
        results = next(result_generator)

        extracted_boxes = []
        extracted_keypoints = []

        # Parse the MMPose output format
        if 'predictions' in results and len(results['predictions']) > 0:
            for pred in results['predictions'][0]: 
                if 'bbox' in pred:
                    extracted_boxes.append(pred['bbox'][0]) 
                if 'keypoints' in pred:
                    extracted_keypoints.append(pred['keypoints'])

        return {
            'boxes': np.array(extracted_boxes),
            'keypoints': np.array(extracted_keypoints)
        }

    def release_resources(self):
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("RTMPose resources released and memory freed.")