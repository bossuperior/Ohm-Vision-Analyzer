import cv2
import numpy as np

class BreadboardWarper:
    def __init__(self, output_width=810, output_height=540, margin=80):
        self.width = output_width
        self.height = output_height
        self.margin = margin
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def process(self, frame):
        # Aruco Detection
        corners, ids, _ = self.detector.detectMarkers(frame)

        # If no tags found or less than 4 tags detected, return original frame
        if ids is None or len(ids) < 4:
            return False, frame, None

        # Get the center points of the detected tags
        tag_centers = {}
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in [0, 1, 2, 3]:
                # corners[i][0] มี 4 มุมย่อยของ Tag นั้นๆ -> หาค่าเฉลี่ยเพื่อเอาจุดกึ่งกลาง
                c = corners[i][0]
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))
                tag_centers[marker_id] = [cx, cy]

        if len(tag_centers) < 4:
            return False, frame, None

        # Sort tag_centers by ID to ensure consistent order: 0, 1, 2, 3
        src_pts = np.float32([
            tag_centers[0], 
            tag_centers[1], 
            tag_centers[2], 
            tag_centers[3]
        ])

        # Destination points — tag centers map outside the frame by `margin`
        # so the area between tags (the board) fills the full output
        m = self.margin
        dst_pts = np.float32([
            [-m, -m],                                       # ซ้ายบน
            [self.width - 1 + m, -m],                      # ขวาบน
            [self.width - 1 + m, self.height - 1 + m],     # ขวาล่าง
            [-m, self.height - 1 + m]                       # ซ้ายล่าง
        ])

        # Calculate perspective transform matrix and warp the image
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_frame = cv2.warpPerspective(frame, matrix, (self.width, self.height))

        return True, warped_frame, matrix