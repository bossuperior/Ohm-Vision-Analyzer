import numpy as np
import cv2


class PointSmoother:
    def __init__(self, alpha: float = 0.4, reset_threshold: float = 60.0):
        self.previous_points = None
        self.reset_threshold = reset_threshold
        self.alpha = alpha          # alpha สูงขึ้น = ติดตามเร็วขึ้น
        self.frame_count = 0        # นับเฟรมเพื่อ warm-up period

    def update(self, current_points: np.ndarray) -> np.ndarray:
        if current_points is None or len(current_points) != 4:
            return self.previous_points

        # เฟรมแรก: snap ทันที ไม่ smooth
        if self.previous_points is None:
            self.previous_points = current_points.astype(float)
            self.frame_count = 1
            return self.previous_points

        # ป้องกันจุดสลับที่ (Index Flipping)
        current_points = self.reorder_by_previous(current_points, self.previous_points)

        # คำนวณระยะห่างจากจุดเดิม (ก่อน smooth)
        raw_distances = np.linalg.norm(current_points - self.previous_points, axis=1)
        max_jump = np.max(raw_distances)

        # ถ้าบอร์ดขยับเยอะมาก: snap ทันที (reset)
        if max_jump > self.reset_threshold:
            self.previous_points = current_points.astype(float)
            self.frame_count = 1
            return self.previous_points

        # Warm-up: 5 เฟรมแรก ให้ alpha สูงสุด (ติดตามเร็ว)
        self.frame_count += 1
        if self.frame_count <= 5:
            effective_alpha = 0.9
        else:
            # Dynamic alpha: ขยับเยอะ → ติดตามเร็ว / ขยับน้อย → ลดการสั่น
            motion = np.mean(raw_distances)
            if motion < 1.5:
                effective_alpha = 0.1   # นิ่งมาก: กันสั่น
            elif motion < 8:
                effective_alpha = 0.4   # ขยับปกติ: smooth ดี
            else:
                effective_alpha = 0.8   # ขยับเยอะ: ติดตามเร็ว

        smoothed = effective_alpha * current_points + (1 - effective_alpha) * self.previous_points
        self.previous_points = smoothed
        return smoothed

    def reset(self):
        self.previous_points = None
        self.frame_count = 0

    def reorder_by_previous(self, current, previous):
        """จับคู่จุดใหม่ให้ตรงกับจุดเก่า (Nearest Neighbor)"""
        reordered = []
        used = set()
        for p_prev in previous:
            distances = np.linalg.norm(current - p_prev, axis=1)
            for idx in np.argsort(distances):
                if idx not in used:
                    used.add(idx)
                    reordered.append(current[idx])
                    break
        return np.array(reordered)


class PerspectiveTransformer:
    def __init__(self, target_w: int = 810, target_h: int = 540, padding: int = 25):
        self.target_w = target_w
        self.target_h = target_h
        self.padding = padding

    def order_points(self, pts):
        """เรียงจุด TL, TR, BR, BL ด้วยวิธี Sum/Diff (เสถียรกว่า arctan2 เมื่อภาพเบี้ยว)"""
        pts = np.array(pts, dtype="float32")
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # Top-Left:     ผลรวม x+y น้อยสุด
        rect[2] = pts[np.argmax(s)]   # Bottom-Right: ผลรวม x+y มากสุด

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-Right:    y-x น้อยสุด
        rect[3] = pts[np.argmax(diff)]  # Bottom-Left:  y-x มากสุด

        return rect

    def validate_corners(self, pts) -> bool:
        """
        ตรวจสอบว่า corner keypoints สมเหตุสมผลก่อน warp
        - ต้องไม่เป็น [0,0] ทุกจุด
        - ต้องมี span พอสมควร (ป้องกัน degenerate transform)
        """
        pts = np.array(pts, dtype="float32")
        if np.all(pts == 0):
            return False
        xs, ys = pts[:, 0], pts[:, 1]
        span_x = xs.max() - xs.min()
        span_y = ys.max() - ys.min()
        # บอร์ดต้องมีพื้นที่อย่างน้อย 50x50 px ในภาพต้นทาง
        if span_x < 50 or span_y < 50:
            return False
        return True

    def warp(self, image, pts):
        """
        ทำ Perspective Transform: ตัดและยืดภาพบอร์ดให้ตรง
        คืนค่า: (warped_image, transform_matrix)
                 ถ้า corners ไม่สมเหตุสมผล คืนภาพต้นทางและ identity matrix
        """
        # Guard: ถ้า corners เสีย ไม่ต้อง warp
        if not self.validate_corners(pts):
            return image, np.eye(3, dtype="float32")

        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        width_top    = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left  = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)

        max_width  = max(int(width_top),    int(width_bottom))
        max_height = max(int(height_left),  int(height_right))

        # ป้องกัน degenerate transform (ขนาดเล็กเกินไป)
        if max_width < 50 or max_height < 50:
            return image, np.eye(3, dtype="float32")

        p = self.padding

        if max_height > max_width:
            # Portrait mode: หมุน 90° ให้บอร์ดนอนแนวนอน
            # src TL,TR,BR,BL → dst TR,BR,BL,TL (90° CW rotation)
            dst_pts = np.array([
                [self.target_w - p - 1, p],
                [self.target_w - p - 1, self.target_h - p - 1],
                [p,                     self.target_h - p - 1],
                [p,                     p]
            ], dtype="float32")
        else:
            # Landscape mode: map ตรงๆ TL→TL, TR→TR, BR→BR, BL→BL
            dst_pts = np.array([
                [p,                     p],
                [self.target_w - p - 1, p],
                [self.target_w - p - 1, self.target_h - p - 1],
                [p,                     self.target_h - p - 1]
            ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(rect, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (self.target_w, self.target_h))

        return warped, matrix
