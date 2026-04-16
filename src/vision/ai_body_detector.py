import cv2
import numpy as np
from ultralytics import YOLO

class AIBodyDetector:
    def __init__(self, model_path="models/Yolo_v8n_pose_weights.pt"):
        try:
            self.model = YOLO(model_path)
            print(f"✅ AI Ready! โหลดโมเดลสำเร็จจาก: {model_path}")
        except Exception as e:
            print(f"❌ โหลดโมเดลไม่สำเร็จ ตรวจสอบ path ของไฟล์ .pt: {e}")
            self.model = None

    def detect(self, frame, conf_threshold=0.5):
        if self.model is None:
            return []
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        detected_objects = []
        
        # Get results for each detected object
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            
            if boxes is None or len(boxes) == 0:
                continue
                
            for i in range(len(boxes)):
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
                w, h = x2 - x1, y2 - y1
                box = (x1, y1, w, h)
                
                # Get class name and confidence
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.model.names[class_id]
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # Get keypoints 
                pts = []
                if keypoints is not None:
                    kp = keypoints.xy[i].cpu().numpy()
                    for point in kp:
                        # Check if keypoint is valid (not [0, 0])
                        if point[0] != 0 and point[1] != 0:
                            pts.append((int(point[0]), int(point[1])))
                            
                detected_objects.append({
                    "class_name": class_name,   
                    "confidence": confidence,   
                    "box": box,                 
                    "keypoints": pts             
                })
                
        return detected_objects

    def draw_results(self, frame, detected_objects):
        display_obj = frame.copy()
        
        for obj in detected_objects:
            x, y, w, h = obj["box"]
            cls_name = obj["class_name"]
            color = (0, 255, 0) if "resistor" in cls_name.lower() else (0, 165, 255)
            
            # Draw bounding box and label
            cv2.rectangle(display_obj, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_obj, f"{cls_name} {obj['confidence']:.2f}", 
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw red keypoints
            for idx, kp in enumerate(obj["keypoints"]):
                cv2.circle(display_obj, kp, 5, (0, 0, 255), -1) 
                cv2.putText(display_obj, str(idx+1), (kp[0]+5, kp[1]-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return display_obj