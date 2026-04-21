import numpy as np
import cv2
from typing import Tuple, List, Dict

class GridMapper:
    def __init__(self, target_w: int = 810, target_h: int = 540, offset_x: int = 45, offset_y: int = 40):
        self.target_w = target_w
        self.target_h = target_h
        
        # Offset from board edges to the first hole
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.cols = 30
        
        # 🌟 THE PITCH MAP (ไร้ขั้ว ไร้ตัวอักษร สนใจแค่ Node ทางกายภาพ)
        self.y_pitch_map = {
            0: ("Rail_A", "Power_Top"),    
            1: ("Rail_B", "Power_Top"),     
            3: ("Hole", "Terminal_Top"),
            4: ("Hole", "Terminal_Top"),
            5: ("Hole", "Terminal_Top"),
            6: ("Hole", "Terminal_Top"),
            7: ("Hole", "Terminal_Top"),
            10: ("Hole", "Terminal_Bottom"),
            11: ("Hole", "Terminal_Bottom"),
            12: ("Hole", "Terminal_Bottom"),
            13: ("Hole", "Terminal_Bottom"),
            14: ("Hole", "Terminal_Bottom"),
            16: ("Rail_C", "Power_Bottom"),
            17: ("Rail_D", "Power_Bottom") 
        }
        
        # Calculate the pitch (distance between holes)
        self.pitch_x = (self.target_w - 2 * self.offset_x) / (self.cols - 1)
        self.pitch_y = (self.target_h - 2 * self.offset_y) / 17.0

    def _get_nearest_y_pitch(self, y_unit: float) -> int:
        valid_pitches = list(self.y_pitch_map.keys())
        return min(valid_pitches, key=lambda p: abs(p - y_unit))

    def map_pixel_to_node(self, x: float, y: float) -> Tuple[str, str, Tuple[int, int]]:
        # Find Column X
        col_idx = int(round((x - self.offset_x) / self.pitch_x))
        col_idx = max(0, min(col_idx, self.cols - 1))
        col_num = col_idx + 1 
        
        # Find Row Y and Zone
        y_unit = (y - self.offset_y) / self.pitch_y
        nearest_pitch = self._get_nearest_y_pitch(y_unit)
        label, zone = self.y_pitch_map[nearest_pitch]
        
        # Assign electrical node names based on the zone
        if zone == "Power_Top":
            electrical_node = f"Power_Top_{label}"   
            hole_id = f"Power_Top_{label}_Col_{col_num}"

        elif zone == "Power_Bottom":
            electrical_node = f"Power_Bottom_{label}"   
            hole_id = f"Power_Bottom_{label}_Col_{col_num}"

        elif zone == "Terminal_Top":
            electrical_node = f"Node_Top_{col_num}"
            hole_id = f"Terminal_Top_Col_{col_num}"

        elif zone == "Terminal_Bottom":
            electrical_node = f"Node_Bottom_{col_num}"
            hole_id = f"Terminal_Bottom_Col_{col_num}"
            
        # Calculate snapped pixel coordinates
        snapped_x = int(self.offset_x + (col_idx * self.pitch_x))
        snapped_y = int(self.offset_y + (nearest_pitch * self.pitch_y))
            
        return hole_id, electrical_node, (snapped_x, snapped_y)

    def map_to_holes(self, component_data: dict) -> List[Dict]:
        mapped_components = []

        if 'keypoints' not in component_data or len(component_data['keypoints']) == 0:
            return mapped_components

        for i, kpts in enumerate(component_data['keypoints']):
            if len(kpts) >= 2:
                leg1_x, leg1_y = kpts[0][:2]
                leg2_x, leg2_y = kpts[1][:2]
                hole1, elec1, snap1 = self.map_pixel_to_node(leg1_x, leg1_y)
                hole2, elec2, snap2 = self.map_pixel_to_node(leg2_x, leg2_y)

                mapped_components.append({
                    'id': i,
                    'node1': elec1,          
                    'node2': elec2,
                    'hole1_name': hole1,    
                    'hole2_name': hole2,
                    'snapped_points': [snap1, snap2]
                })

        return mapped_components

    def draw_grid_overlay(self, frame: np.ndarray) -> np.ndarray:
        grid_img = frame.copy()
        
        for col_idx in range(self.cols):
            x = int(self.offset_x + (col_idx * self.pitch_x))
            for pitch, (label, zone) in self.y_pitch_map.items():
                y = int(self.offset_y + (pitch * self.pitch_y))
                
                color = (0, 255, 0) # Green for holes
                
                if "Power" in zone:
                    if label in ["Rail_A", "Rail_C"]:
                        color = (0, 0, 255) # Red for Rail A/C
                    else:
                        color = (255, 0, 0) # Blue for Rail B/D
                        
                cv2.circle(grid_img, (x, y), 3, color, -1)

        return grid_img