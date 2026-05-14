import numpy as np
from ultralytics import YOLO


class DetectionResult:
    def __init__(self, boxes, keypoints, class_ids, scores=None):
        self.boxes      = boxes
        self.keypoints  = keypoints
        self.class_ids  = class_ids
        self.scores     = scores if scores is not None else np.ones(len(class_ids))


class ModelEngine:
    def __init__(self, model_path: str, conf: float = 0.75, iou: float = 0.45):
        self.engine = YOLO(model_path, task="pose")
        self._conf  = conf
        self._iou   = iou

    def predict(self, frame) -> DetectionResult:
        r = self.engine(frame, verbose=False, conf=self._conf, iou=self._iou)[0]
        boxes     = r.boxes.xyxy.cpu().numpy()      if r.boxes     else np.array([])
        class_ids = r.boxes.cls.cpu().numpy()       if r.boxes     else np.array([])
        scores    = r.boxes.conf.cpu().numpy()      if r.boxes     else np.array([])
        keypoints = r.keypoints.data.cpu().numpy()  if r.keypoints else np.array([])
        return DetectionResult(boxes, keypoints, class_ids, scores)
