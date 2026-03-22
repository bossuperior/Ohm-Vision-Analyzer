import cv2
import numpy as np

def img_reader(img, box, padding=0):
    """
    ตัดภาพตัวต้านทานจาก Bounding Box และปรับให้เป็นแนวนอนเสมอ
    :param img: ภาพต้นฉบับ
    :param box: พิกัด Bounding Box จากโมเดล
    :param padding: จำนวนพิกเซลที่ต้องการเผื่อขอบไว้ (ป้องกันการตัดโดนแถบสีขาด)
    """
    # 1. แปลง Box เป็นพิกัด 4 จุด
    rect = cv2.boxPoints(box)
    rect = np.intp(rect)

    # 2. จัดเรียงพิกัด (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    ordered_rect = np.zeros((4, 2), dtype="float32")
    
    s = rect.sum(axis=1)
    ordered_rect[0] = rect[np.argmin(s)]  # Top-Left
    ordered_rect[2] = rect[np.argmax(s)]  # Bottom-Right
    
    diff = np.diff(rect, axis=1)
    ordered_rect[1] = rect[np.argmin(diff)]  # Top-Right
    ordered_rect[3] = rect[np.argmax(diff)]  # Bottom-Left

    (tl, tr, br, bl) = ordered_rect

    # 3. คำนวณความกว้างและความสูง
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # ดักจับ Error กรณีกล่องมีขนาดเล็กเกินไป (ป้องกัน OpenCV แครช)
    if max_width == 0 or max_height == 0:
        return None

    # 4. กำหนดจุดเป้าหมาย (Destination Points) พร้อมบวก Padding
    dst = np.array([
        [padding, padding],
        [max_width - 1 + padding, padding],
        [max_width - 1 + padding, max_height - 1 + padding],
        [padding, max_height - 1 + padding]
    ], dtype="float32")

    # 5. ทำ Perspective Transform
    M = cv2.getPerspectiveTransform(ordered_rect, dst)
    # ขยายขนาดภาพ Output ตาม Padding ที่ตั้งไว้
    warped = cv2.warpPerspective(img, M, (max_width + (padding * 2), max_height + (padding * 2)))

    # 6. ตรวจสอบและหมุนให้ภาพเป็นแนวนอน (Landscape) เสมอ
    if warped.shape[0] > warped.shape[1]:  # ถ้าความสูง > ความกว้าง
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped