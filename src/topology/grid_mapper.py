import numpy as np
import cv2
from typing import Tuple, List, Dict

class GridMapper:
    def __init__(self, target_w: int = 810, target_h: int = 540, offset_x: int = 45, offset_y: int = 40):
        self.target_w = target_w
        self.target_h = target_h
        self._node_cache: Dict = {}  # (comp_id, leg_idx) → (hole_id, elec_node, snap_px)
        
        # Offset from board edges to the first hole
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.cols = 30

        self.y_pitch_map = {
            0:  ("Rail_A", "Power_Top"),
            1:  ("Rail_B", "Power_Top"),
            # Terminal_Top 6 rows (a-f)
            3:  ("Hole", "Terminal_Top"),
            4:  ("Hole", "Terminal_Top"),
            5:  ("Hole", "Terminal_Top"),
            6:  ("Hole", "Terminal_Top"),
            7:  ("Hole", "Terminal_Top"),
            8:  ("Hole", "Terminal_Top"),   # แถวชิดกลาง
            # middle divider (no holes) — gap 0 step
            # Terminal_Bottom 6 rows (g-l)
            9:  ("Hole", "Terminal_Bottom"),  # แถวชิดกลาง
            10: ("Hole", "Terminal_Bottom"),
            11: ("Hole", "Terminal_Bottom"),
            12: ("Hole", "Terminal_Bottom"),
            13: ("Hole", "Terminal_Bottom"),
            14: ("Hole", "Terminal_Bottom"),
            # bottom gap 2 steps (15)
            16: ("Rail_C", "Power_Bottom"),
            17: ("Rail_D", "Power_Bottom"),
        }
        
        self._recompute_pitch()

    def set_params(self, offset_x: int, offset_y: int,
                   pitch_x: float = None, pitch_y: float = None, cols: int = None):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self._node_cache.clear()  # pitch/offset change invalidates cached snaps
        if cols is not None and cols > 1:
            self.cols = cols
        if pitch_x is not None and pitch_x > 0:
            self.pitch_x = pitch_x
        if pitch_y is not None and pitch_y > 0:
            self.pitch_y = pitch_y
        # ถ้าไม่ได้ set pitch ตรงๆ ให้คำนวณจาก cols
        if pitch_x is None:
            self._recompute_pitch()

    def _recompute_pitch(self):
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

    def snap_to_nearest_hole(self, x: float, y: float) -> Tuple[int, int]:
        """Return pixel (x, y) of the nearest breadboard hole for a given point."""
        _, _, snapped_px = self.map_pixel_to_node(x, y)
        return snapped_px

    def map_to_holes(self, component_data: List[Dict]) -> List[Dict]:
        # Drop cache entries for components no longer visible
        active = {comp['id'] for comp in component_data}
        self._node_cache = {k: v for k, v in self._node_cache.items() if k[0] in active}

        mapped_components = []
        for comp in component_data:
            kpts = comp.get('keypoints', [])
            if len(kpts) < 2:
                continue

            nodes = []
            for leg_idx, (lx, ly) in enumerate((kpts[0][:2], kpts[1][:2])):
                hole, elec, snap = self.map_pixel_to_node(lx, ly)

                # Hysteresis: only accept a different hole if keypoint crossed 60% of pitch
                # from the cached snap. This prevents jitter at hole boundaries from
                # causing topology flip-flop.
                key = (comp['id'], leg_idx)
                if key in self._node_cache:
                    c_hole, c_elec, c_snap = self._node_cache[key]
                    if hole != c_hole:
                        dx = abs(lx - c_snap[0]) / self.pitch_x
                        dy = abs(ly - c_snap[1]) / self.pitch_y
                        if dx < 0.6 and dy < 0.6:
                            hole, elec, snap = c_hole, c_elec, c_snap

                self._node_cache[key] = (hole, elec, snap)
                nodes.append((hole, elec, snap))

            mapped_components.append({
                'id': comp['id'],
                'node1': nodes[0][1],
                'node2': nodes[1][1],
                'hole1_name': nodes[0][0],
                'hole2_name': nodes[1][0],
                'snapped_points': [nodes[0][2], nodes[1][2]]
            })

        return mapped_components

    def draw_grid_overlay(self, frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        overlay = frame.copy()

        # จัดกลุ่ม pitch index ตาม zone
        zone_groups = {
            "Power_Top_A":    [0],
            "Power_Top_B":    [1],
            "Terminal_Top":   [3, 4, 5, 6, 7, 8],
            "Terminal_Bottom":[9, 10, 11, 12, 13, 14],
            "Power_Bottom_C": [16],
            "Power_Bottom_D": [17],
        }
        zone_colors = {
            "Power_Top_A":     (0, 0, 255),    # แดง
            "Power_Top_B":     (255, 100, 0),  # น้ำเงินเข้ม
            "Terminal_Top":    (0, 200, 80),   # เขียว
            "Terminal_Bottom": (0, 180, 255),  # ฟ้า
            "Power_Bottom_C":  (0, 0, 200),   # แดงเข้ม
            "Power_Bottom_D":  (200, 80, 0),  # น้ำเงิน
        }

        x_start = int(self.offset_x)
        x_end   = int(self.offset_x + (self.cols - 1) * self.pitch_x)

        for zone_name, pitches in zone_groups.items():
            color = zone_colors[zone_name]
            ys = [int(self.offset_y + p * self.pitch_y) for p in pitches]
            y_top = min(ys)
            y_bot = max(ys)

            is_power = "Power" in zone_name

            if is_power:
                # Power rail → เส้นแนวนอนยาวตลอด
                for y in ys:
                    cv2.line(overlay, (x_start, y), (x_end, y), color, 2)
            else:
                # Terminal → เส้นแนวตั้งต่อ 5 รูในแต่ละคอลัมน์
                for col_idx in range(self.cols):
                    x = int(self.offset_x + col_idx * self.pitch_x)
                    cv2.line(overlay, (x, y_top), (x, y_bot), color, 2)
                    # จุดเล็กที่หัว-ท้าย
                    cv2.circle(overlay, (x, y_top), 2, color, -1)
                    cv2.circle(overlay, (x, y_bot), 2, color, -1)

        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)