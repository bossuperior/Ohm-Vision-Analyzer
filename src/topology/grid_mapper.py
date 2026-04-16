import cv2
import numpy as np
from typing import List, Dict, Tuple


class GridMapper:
    """
    Translates pixel coordinates from the cropped breadboard image
    into logical electrical nodes (e.g., 'Row_5_Top', 'Power_Top_Plus').
    """

    def __init__(self, total_rows: int = 30, offset_length: int = 35, offset_width: int = 40):
        # Configuration parameters
        self.total_rows = total_rows
        self.offset_length = offset_length
        self.offset_width = offset_width

        # Ratios defining the zones on a standard 400-tie point breadboard
        self.terminal_top_ratio = 0.18
        self.terminal_bottom_ratio = 0.82
        self.rails_top_ratio = 0.35
        self.rails_bottom_ratio = 0.65

    def _calculate_spacing(self, target_len: int) -> float:
        """Calculates the pixel distance between each hole."""
        return (target_len - (2 * self.offset_length)) / (self.total_rows - 1)

    def map_pixel_to_node(self, x: float, y: float, target_w: int, target_h: int) -> Tuple[str, int, int]:
        """
        Takes a single (x, y) coordinate and snaps it to the nearest physical breadboard hole.
        Returns: (Node Name, Snapped X, Snapped Y)
        """
        is_horizontal = target_w > target_h
        node_name = "Unknown"
        snapped_x, snapped_y = int(x), int(y)

        if is_horizontal:
            spacing = self._calculate_spacing(target_w)

            # Find nearest row index (1-based)
            row_idx = int(round((x - self.offset_length) / spacing)) + 1
            row_idx = max(1, min(row_idx, self.total_rows))

            snapped_x = int(self.offset_length + (row_idx - 1) * spacing)

            power_top_bound = target_h * self.terminal_top_ratio
            power_bottom_bound = target_h * self.terminal_bottom_ratio

            # --- Power Rails Zones ---
            if y < power_top_bound:
                is_plus = y < power_top_bound / 2
                ratio = self.rails_top_ratio if is_plus else self.rails_bottom_ratio
                snapped_y = int(power_top_bound * ratio)
                node_name = "Power_Top_Plus" if is_plus else "Power_Top_Minus"

            elif y > power_bottom_bound:
                is_plus = y < target_h - (target_h - power_bottom_bound) / 2
                ratio = self.rails_bottom_ratio if is_plus else self.rails_top_ratio
                snapped_y = int(target_h - (target_h - power_bottom_bound) * ratio)
                node_name = "Power_Bottom_Plus" if is_plus else "Power_Bottom_Minus"

            # --- Terminal Strips Zones ---
            else:
                side = "Top" if y < target_h / 2 else "Bottom"
                node_name = f"Row_{row_idx}_{side}"
                snapped_y = int(y)  # Keep original Y for terminal strips, or snap to nearest 5 holes later

        return node_name, snapped_x, snapped_y

    def map_to_holes(self, component_data: dict, target_w: int, target_h: int) -> List[Dict]:
        """
        Takes the AI Keypoint outputs and maps every component's legs to breadboard nodes.
        Assumes component_data['keypoints'] contains arrays of [x, y] coordinates.
        """
        mapped_components = []

        # Safely check if keypoints exist
        if 'keypoints' not in component_data or len(component_data['keypoints']) == 0:
            return mapped_components

        # Iterate through every detected component (resistor/wire)
        for i, kpts in enumerate(component_data['keypoints']):
            # A resistor or jumper wire should have exactly 2 keypoints (the legs)
            if len(kpts) >= 2:
                leg1_x, leg1_y = kpts[0]
                leg2_x, leg2_y = kpts[1]

                # Map Leg 1
                node1, sx1, sy1 = self.map_pixel_to_node(leg1_x, leg1_y, target_w, target_h)
                # Map Leg 2
                node2, sx2, sy2 = self.map_pixel_to_node(leg2_x, leg2_y, target_w, target_h)

                mapped_components.append({
                    'id': i,
                    'node1': node1,
                    'node2': node2,
                    'snapped_points': [(sx1, sy1), (sx2, sy2)]
                })

        return mapped_components

    def draw_grid_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Optional debugging tool: Draws the grid logic over the image
        to ensure spacing and offsets are perfectly aligned with reality.
        """
        grid_img = frame.copy()
        target_h, target_w = grid_img.shape[:2]
        is_horizontal = target_w > target_h

        if is_horizontal:
            spacing = self._calculate_spacing(target_w)
            top_bound = int(target_h * self.terminal_top_ratio)
            bot_bound = int(target_h * self.terminal_bottom_ratio)

            cv2.line(grid_img, (0, top_bound), (target_w, top_bound), (255, 0, 0), 1)
            cv2.line(grid_img, (0, bot_bound), (target_w, bot_bound), (255, 0, 0), 1)

            for i in range(self.total_rows):
                x = int(self.offset_length + (i * spacing))
                cv2.line(grid_img, (x, top_bound), (x, bot_bound), (0, 150, 0), 1)
        # Add else block for vertical debugging if needed...

        return grid_img