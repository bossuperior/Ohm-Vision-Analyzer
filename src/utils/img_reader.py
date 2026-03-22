import cv2
import numpy as np

def img_reader(img, box, padding=0):
    """
    ตัดภาพตัวต้านทานจาก Bounding Box และปรับให้เป็นแนวนอนเสมอ
    :param img: ภาพต้นฉบับ
    :param box: พิกัด Bounding Box สามารถรับได้ 3 รูปแบบ:
                1. [x_min, y_min, x_max, y_max] (เช่น จาก YOLO ธรรมดา)
                2. ((cx, cy), (w, h), angle) (จาก cv2.minAreaRect)
                3. พิกัด 4 จุด [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :param padding: จำนวนพิกเซลที่ต้องการเผื่อขอบไว้
    :return: (warped_img, true_length) ภาพที่ดัดแล้ว และความยาวจริงของตัวต้านทาน (ไม่รวม padding)
    """
    # ==========================================
    # 1. จัดการรูปแบบของ Box (แก้ข้อ 1)
    # ==========================================
    if isinstance(box, tuple) and len(box) == 3 and isinstance(box[0], tuple):
        # กรณีมาจาก cv2.minAreaRect()
        rect = cv2.boxPoints(box)
    elif len(box) == 4 and isinstance(box[0], (int, float, np.integer, np.floating)):
        # กรณีเป็น [x_min, y_min, x_max, y_max] 
        x_min, y_min, x_max, y_max = box
        rect = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ])
    else:
        # กรณีเป็นพิกัด 4 มุมอยู่แล้ว
        rect = np.array(box)

    rect = np.intp(rect)

    # ==========================================
    # 2. จัดเรียงพิกัด (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    # ==========================================
    ordered_rect = np.zeros((4, 2), dtype="float32")
    
    s = rect.sum(axis=1)
    ordered_rect[0] = rect[np.argmin(s)]  # Top-Left
    ordered_rect[2] = rect[np.argmax(s)]  # Bottom-Right
    
    diff = np.diff(rect, axis=1)
    ordered_rect[1] = rect[np.argmin(diff)]  # Top-Right
    ordered_rect[3] = rect[np.argmax(diff)]  # Bottom-Left

    (tl, tr, br, bl) = ordered_rect

    # ==========================================
    # 3. คำนวณความกว้างและความสูง
    # ==========================================
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    if max_width == 0 or max_height == 0:
        return None, 0

    # ==========================================
    # 4. ทำ Perspective Transform เผื่อขอบ Padding
    # ==========================================
    dst = np.array([
        [padding, padding],
        [max_width - 1 + padding, padding],
        [max_width - 1 + padding, max_height - 1 + padding],
        [padding, max_height - 1 + padding]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_rect, dst)
    warped = cv2.warpPerspective(img, M, (max_width + (padding * 2), max_height + (padding * 2)))

    # ==========================================
    # 5. หมุนภาพและคำนวณความยาวจริง (แก้ข้อ 2)
    # ==========================================
    true_length = max_width
    if warped.shape[0] > warped.shape[1]:  # ถ้าตั้งฉาก (สูง > กว้าง)
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        true_length = max_height  # หลังหมุน ความยาวคือด้านที่เคยเป็นความสูง

    return warped, true_length