from __future__ import annotations
import cv2
import numpy as np
from typing import Literal

BackendPose = Literal['yolo', 'rtmpose']
BackendCls  = Literal['yolo_cls', 'onnx', 'efficientnet', 'shufflenet', 'mobilenet']


class DetectionResult:
    def __init__(self, boxes, keypoints, class_ids, scores=None):
        self.boxes     = boxes
        self.keypoints = keypoints
        self.class_ids = class_ids
        self.scores    = scores if scores is not None else np.ones(len(class_ids))


# ─────────────────────────────────────────────────────────────────
# Keypoint Detection
# backend='yolo'    → YOLOv8n-pose หรือ YOLOv8s-pose (ต่างกันแค่ model_path)
# backend='rtmpose' → RTMPose via rtmlib
# ─────────────────────────────────────────────────────────────────

class ModelEngine:
    def __init__(self, backend: BackendPose, model_path: str,
                 conf: float = 0.5, iou: float = 0.45, **kwargs):
        self.backend = backend
        self._conf   = conf
        self._iou    = iou

        if backend == 'yolo':
            from ultralytics import YOLO
            self._model = YOLO(model_path, task='pose')

        elif backend == 'rtmpose':
            from rtmlib import RTMPose
            self._model = RTMPose(
                pose=model_path,
                det=kwargs.get('det_model'),
                device=kwargs.get('device', 'cuda'),
                backend=kwargs.get('onnx_backend', 'onnxruntime'),
                score_threshold=conf,
            )
        else:
            raise ValueError(f"Unknown pose backend: {backend!r}")

    def predict(self, frame: np.ndarray) -> DetectionResult:
        if self.backend == 'yolo':
            return self._predict_yolo(frame)
        return self._predict_rtmpose(frame)

    def _predict_yolo(self, frame: np.ndarray) -> DetectionResult:
        r = self._model(frame, verbose=False, conf=self._conf, iou=self._iou)[0]
        boxes     = r.boxes.xyxy.cpu().numpy()     if r.boxes     else np.array([])
        class_ids = r.boxes.cls.cpu().numpy()      if r.boxes     else np.array([])
        scores    = r.boxes.conf.cpu().numpy()     if r.boxes     else np.array([])
        keypoints = r.keypoints.data.cpu().numpy() if r.keypoints else np.array([])
        return DetectionResult(boxes, keypoints, class_ids, scores)

    def _predict_rtmpose(self, frame: np.ndarray) -> DetectionResult:
        # rtmlib คืน (keypoints, scores): shape (N, K, 2), (N, K)
        keypoints, scores = self._model(frame)
        if keypoints is None or len(keypoints) == 0:
            return DetectionResult(np.array([]), np.array([]), np.array([]))

        vis          = (scores > self._conf).astype(float)[..., np.newaxis]
        kpts_with_vis = np.concatenate([keypoints, vis], axis=-1)  # (N, K, 3)
        n             = len(keypoints)
        return DetectionResult(
            boxes     = np.zeros((n, 4)),
            keypoints = kpts_with_vis,
            class_ids = np.zeros(n),
            scores    = scores.mean(axis=-1),
        )


# ─────────────────────────────────────────────────────────────────
# Resistor Value Classification
# backend='yolo_cls'   → YOLO classify  (ultralytics)
# backend='shufflenet' → ShuffleNetV2   (torchvision)
# backend='mobilenet'  → MobileNetV3    (torchvision)
# ─────────────────────────────────────────────────────────────────

class ClassificationEngine:
    _MEAN  = [0.485, 0.456, 0.406]
    _STD   = [0.229, 0.224, 0.225]
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def __init__(self, backend: BackendCls, model_path: str,
                 num_classes: int = 10, device: str = 'cuda',
                 class_names: list[str] | None = None):
        self.backend     = backend
        self._device     = device
        self.class_names = class_names or [str(i) for i in range(num_classes)]

        if backend == 'yolo_cls':
            from ultralytics import YOLO
            self._model = YOLO(model_path)

        elif backend == 'onnx':
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            providers = (['CUDAExecutionProvider'] if device == 'cuda' else []) + \
                        ['CPUExecutionProvider']
            self._session    = ort.InferenceSession(model_path, opts, providers=providers)
            self._input_name = self._session.get_inputs()[0].name

        elif backend == 'efficientnet':
            import torch
            import torchvision.models as M
            net = M.efficientnet_b0(weights=None)
            net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, num_classes)
            net.load_state_dict(torch.load(model_path, map_location=device,
                                           weights_only=True))
            self._model     = net.to(device).eval()
            self._transform = self._make_transform()

        elif backend == 'shufflenet':
            import torch
            import torchvision.models as M
            net    = M.shufflenet_v2_x1_0(weights=None)
            net.fc = torch.nn.Linear(1024, num_classes)
            net.load_state_dict(torch.load(model_path, map_location=device,
                                           weights_only=True))
            self._model     = net.to(device).eval()
            self._transform = self._make_transform()

        elif backend == 'mobilenet':
            import torch
            import torchvision.models as M
            net               = M.mobilenet_v3_small(weights=None)
            net.classifier[3] = torch.nn.Linear(1024, num_classes)
            net.load_state_dict(torch.load(model_path, map_location=device,
                                           weights_only=True))
            self._model     = net.to(device).eval()
            self._transform = self._make_transform()

        else:
            raise ValueError(f"Unknown classification backend: {backend!r}")

    def _make_transform(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self._MEAN, self._STD),
        ])

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """CLAHE on L-channel + BGR→RGB — must match training pipeline."""
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def predict(self, crop: np.ndarray) -> tuple[str, float]:
        """คืน (class_name, confidence). crop คือ BGR numpy array จาก OpenCV."""
        if self.backend == 'yolo_cls':
            return self._predict_yolo_cls(crop)
        if self.backend == 'onnx':
            return self._predict_onnx(crop)
        return self._predict_torch(crop)

    def _predict_onnx(self, crop: np.ndarray) -> tuple[str, float]:
        rgb  = self._preprocess(crop)
        img  = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
        mean = np.array(self._MEAN, dtype=np.float32).reshape(1, 1, 3)
        std  = np.array(self._STD,  dtype=np.float32).reshape(1, 1, 3)
        img  = ((img - mean) / std).transpose(2, 0, 1)[np.newaxis]  # (1,3,H,W)
        out  = self._session.run(None, {self._input_name: img})[0][0]  # (num_classes,)
        probs = np.exp(out - out.max())
        probs /= probs.sum()
        cid  = int(probs.argmax())
        conf = float(probs[cid])
        name = self.class_names[cid] if cid < len(self.class_names) else str(cid)
        return name, conf

    def _predict_yolo_cls(self, crop: np.ndarray) -> tuple[str, float]:
        rgb  = self._preprocess(crop)
        r    = self._model(rgb, verbose=False)[0]
        cid  = int(r.probs.top1)
        conf = float(r.probs.top1conf)
        name = self.class_names[cid] if cid < len(self.class_names) else str(cid)
        return name, conf

    def _predict_torch(self, crop: np.ndarray) -> tuple[str, float]:
        import torch
        rgb = self._preprocess(crop)
        x   = self._transform(rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            probs = torch.softmax(self._model(x), dim=1)[0]
        cid  = int(probs.argmax())
        conf = float(probs[cid])
        name = self.class_names[cid] if cid < len(self.class_names) else str(cid)
        return name, conf
