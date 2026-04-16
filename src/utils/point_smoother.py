import numpy as np


import numpy as np

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