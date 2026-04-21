import numpy as np
import cv2

class PointSmoother:
    def __init__(self, alpha: float = 0.15, reset_threshold: float = 50.0):
        self.alpha = alpha
        self.previous_points = None
        # If a point moves more than 50 pixels in 1 frame, reset memory!
        self.reset_threshold = reset_threshold

    def update(self, current_points: np.ndarray) -> np.ndarray:
        if self.previous_points is None:
            self.previous_points = current_points
            return current_points

        # Calculate the distance every point moved
        distances = np.linalg.norm(current_points - self.previous_points, axis=1)
        max_jump = np.max(distances)

        # If it jumped drastically (Index Flip or fast movement), wipe the memory
        if max_jump > self.reset_threshold:
            # print(" Rapid movement or rotation detected! Resetting stabilization.")
            self.previous_points = current_points
            return current_points

        # Otherwise, blend smoothly
        smoothed_points = (self.alpha * current_points) + ((1 - self.alpha) * self.previous_points)
        self.previous_points = smoothed_points
        return smoothed_points

    def reset(self):
        self.previous_points = None

class PerspectiveTransformer:
    def __init__(self):
        pass

    def order_points(self,pts):
        pts = np.array(pts, dtype="float32")
        #Sort the points based on their angle from the center to ensure consistent ordering
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        sorted_pts = pts[np.argsort(angles)]
        s = sorted_pts.sum(axis=1)
        tl_index = np.argmin(s)
        # Returns [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        return np.roll(sorted_pts, -tl_index, axis=0)
    
    def warp(self,image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # Auto-rotate logic: Determine if the board is more vertical or horizontal based on the corner points
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)

        max_width = max(int(width_top), int(width_bottom))
        max_height = max(int(height_left), int(height_right))
        target_w, target_h = 810, 540

        if max_height > max_width:
            # The physical board is VERTICAL.
            # We map the corners to automatically rotate it 90-degrees clockwise into a flat landscape.
            dst_pts = np.array([
                [target_w - 1, 0],  # Top-Left point moves to Top-Right
                [target_w - 1, target_h - 1],  # Top-Right point moves to Bottom-Right
                [0, target_h - 1],  # Bottom-Right point moves to Bottom-Left
                [0, 0]  # Bottom-Left point moves to Top-Left
            ], dtype="float32")
        else:
            # The physical board is already HORIZONTAL. Standard mapping.
            dst_pts = np.array([
                [0, 0],
                [target_w - 1, 0],
                [target_w - 1, target_h - 1],
                [0, target_h - 1]
            ], dtype="float32")

        # Apply the transform. The output is now guaranteed to be 810x540 and perfectly straight.
        matrix = cv2.getPerspectiveTransform(rect, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (target_w, target_h))

        return warped, matrix