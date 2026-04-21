import cv2
import numpy as np
import math
from typing import List, Tuple, Any


class ResistorCrop:
    """Data structure to hold cropped resistor image and related info"""

    def __init__(self, res_id, image_crop, text_position, keypoints):
        self.id = res_id
        self.image_crop = image_crop  # ภาพตัวต้านทานที่โดนตัดและหมุนให้ขนานกับพื้น
        self.text_position = text_position  # พิกัด x, y สำหรับวาดตัวหนังสือ
        self.keypoints = keypoints  # พิกัดขาทั้ง 2 ข้าง


class BodyDetector:
    def __init__(self, resistor_class_id=1):
        self.resistor_class_id = resistor_class_id

    def extract_resistors(self, image: np.ndarray, detection_results: Any) -> List[ResistorCrop]:
        """
        ตัดภาพตัวต้านทาน "ทุกตัว" ที่ AI หาเจอ และหมุนภาพให้ตั้งตรง
        """
        extracted_list = []

        if detection_results is None or len(detection_results.class_ids) == 0:
            return extracted_list

        # ดึงเฉพาะ Index ที่ AI บอกว่าเป็นตัวต้านทาน (ID = 1)
        resistor_indices = np.where(detection_results.class_ids == self.resistor_class_id)[0]

        for res_idx in resistor_indices:
            box = detection_results.boxes[res_idx]
            kpts = detection_results.keypoints[res_idx]

            conf = detection_results.scores[res_idx]

            if conf < 0.55 or len(kpts) < 2:
                continue

            kp1_x, kp1_y = kpts[0][:2]
            kp2_x, kp2_y = kpts[1][:2]

            # Leg span < 60 px on the warped 810x540 image is almost certainly a breadboard-hole false positive
            if math.hypot(kp2_x - kp1_x, kp2_y - kp1_y) < 60:
                continue

            x1, y1, x2, y2 = map(int, box[:4])
            text_pos = (x1, max(10, y1 - 10))

            # 2. คำนวณมุมเอียง (Angle)
            dx = kp2_x - kp1_x
            dy = kp2_y - kp1_y
            angle = math.degrees(math.atan2(dy, dx))

            # 3. สร้าง ROI เผื่อพื้นที่รอบๆ ไว้หมุนภาพ
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            pad = max(x2 - x1, y2 - y1) // 2 + 20
            H, W = image.shape[:2]

            px1, py1 = max(0, cx - pad), max(0, cy - pad)
            px2, py2 = min(W, cx + pad), min(H, cy + pad)

            roi = image[py1:py2, px1:px2]
            if roi.size == 0:
                continue

            roi_cx, roi_cy = cx - px1, cy - py1

            # 4. หมุนภาพให้ตัวต้านทานนอนขนานกับพื้น
            matrix = cv2.getRotationMatrix2D((roi_cx, roi_cy), angle, 1.0)
            rotated_roi = cv2.warpAffine(roi, matrix, (roi.shape[1], roi.shape[0]))

            # 5. ตัดภาพครั้งสุดท้าย เอาเฉพาะส่วน "ลำตัว" (หนาขึ้นนิดนึงเพื่อกันแหว่ง)
            body_length = int(math.hypot(dx, dy)) + 15
            body_thickness = min(x2 - x1, y2 - y1) + 15

            rx1 = max(0, roi_cx - body_length // 2)
            ry1 = max(0, roi_cy - body_thickness // 2)
            rx2 = min(rotated_roi.shape[1], roi_cx + body_length // 2)
            ry2 = min(rotated_roi.shape[0], roi_cy + body_thickness // 2)

            final_crop = rotated_roi[ry1:ry2, rx1:rx2]

            # 6. เก็บเข้าคลังแสง
            if final_crop.size > 0 and final_crop.shape[0] >= 10 and final_crop.shape[1] >= 10:
                # ใช้ res_idx เป็น ID เพื่อเชื่อมโยงกับ Bounding Box หลัก
                extracted_list.append(ResistorCrop(res_idx, final_crop, text_pos, kpts))

        return self._deduplicate(extracted_list)

    def _deduplicate(self, detections: List[ResistorCrop], threshold: int = 90) -> List[ResistorCrop]:
        """Drop detections whose midpoint centroid is within `threshold` px of an already-kept one."""
        keep: List[ResistorCrop] = []
        keep_centers: List[np.ndarray] = []
        for res in detections:
            center = (np.array(res.keypoints[0][:2]) + np.array(res.keypoints[1][:2])) / 2
            if not any(np.linalg.norm(center - c) < threshold for c in keep_centers):
                keep.append(res)
                keep_centers.append(center)
        return keep


# ==========================================
# 🌟 เสาหลักที่ 4 ของจริง: ระบบหน่วงเวลาและโหวตค่า (Temporal Filter)
# ==========================================
from collections import defaultdict, deque
from statistics import mode


class TemporalFilter:
    """
    ระบบกันข้อความกระพริบ (Flicker Prevention)
    ใช้หลักการ Majority Voting: จำค่า 10 เฟรมหลังสุด และโชว์ค่าที่โผล่มาบ่อยที่สุด
    """

    def __init__(self, history_size: int = 7):
        self.history_size = history_size
        self.ohm_history = defaultdict(lambda: deque(maxlen=history_size))
        self.str_history = defaultdict(lambda: deque(maxlen=history_size))

    def update_and_get(self, res_id: int, current_str: str, current_ohm: float) -> Tuple[str, float]:
        if current_ohm <= 0 or "Error" in current_str or "Unknown" in current_str:
            if self.str_history[res_id]:
                return mode(self.str_history[res_id]), mode(self.ohm_history[res_id])
            return current_str, current_ohm

        self.ohm_history[res_id].append(current_ohm)
        self.str_history[res_id].append(current_str)

        try:
            return mode(self.str_history[res_id]), mode(self.ohm_history[res_id])
        except Exception:
            return self.str_history[res_id][-1], self.ohm_history[res_id][-1]

    def clear_stale_ids(self, current_ids: List[int]):
        """
        ลบความจำของตัวต้านทานที่หายไปจากหน้าจอแล้ว (กันเมมโมรี่บวม)
        เรียกใช้ฟังก์ชันนี้ตอนท้ายของลูปใน main.py
        """
        stale_ids = [res_id for res_id in self.ohm_history.keys() if res_id not in current_ids]
        for res_id in stale_ids:
            del self.ohm_history[res_id]
            del self.str_history[res_id]