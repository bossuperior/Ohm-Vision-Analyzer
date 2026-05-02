import cv2
import numpy as np

class BreadboardWarper:
    def __init__(self, output_width=810, output_height=540,
                 margin=80, shift_x=0, shift_y=0):
        self.width = output_width
        self.height = output_height
        # crop per-side: tag center maps this many pixels outside the output frame
        # เพิ่มค่า = ตัดขอบด้านนั้นมากขึ้น (แนบบอร์ดมากขึ้น)
        # ลดค่า = เห็นพื้นที่รอบบอร์ดมากขึ้น
        self.margin  = margin
        self.shift_x = shift_x   # บวก = เลื่อนขวา, ลบ = เลื่อนซ้าย
        self.shift_y = shift_y   # บวก = เลื่อนลง,  ลบ = เลื่อนขึ้น

        self.aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector     = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def process(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)

        if ids is None or len(ids) < 4:
            return False, frame, None

        tag_centers = {}
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in [0, 1, 2, 3]:
                c = corners[i][0]
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))
                tag_centers[marker_id] = [cx, cy]

        if len(tag_centers) < 4:
            return False, frame, None

        src_pts = np.float32([
            tag_centers[0],   # ซ้ายบน
            tag_centers[1],   # ขวาบน
            tag_centers[2],   # ขวาล่าง
            tag_centers[3],   # ซ้ายล่าง
        ])

        W, H = self.width - 1, self.height - 1
        m  = self.margin
        sx = self.shift_x
        sy = self.shift_y
        # margin ทุกด้าน + shift ซ้าย/ขวา/บน/ล่าง
        dst_pts = np.float32([
            [-m + sx,  -m + sy],   # ซ้ายบน
            [ W+m+sx,  -m + sy],   # ขวาบน
            [ W+m+sx,   H+m+sy],   # ขวาล่าง
            [-m + sx,   H+m+sy],   # ซ้ายล่าง
        ])

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped  = cv2.warpPerspective(frame, matrix, (self.width, self.height))

        return True, warped, matrix
