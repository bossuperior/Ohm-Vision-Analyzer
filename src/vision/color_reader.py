import numpy as np
import cv2

# [REF_COLORS และ COLOR_VALS ใช้ของเดิมของคุณได้เลย เพราะคุณจูนเข้ากับกล้องคุณมาแล้ว]
REF_COLORS = {
    'BLACK': (0, 0, 30), 'BROWN': (13, 110, 60), 'RED': (175, 150, 170), # ปรับ RED ให้อยู่ขอบวงล้อสีแดงของ OpenCV
    'ORANGE': (15, 255, 255), 'YELLOW': (30, 200, 200), 'GOLD': (20, 140, 140),
    'GREEN': (60, 200, 200), 'BLUE': (110, 200, 200), 'VIOLET': (140, 180, 180),
    'GRAY': (0, 0, 100), 'WHITE': (0, 0, 240), 'SILVER': (0, 0, 200)
}

COLOR_VALS = {
    'BLACK': 0, 'BROWN': 1, 'RED': 2, 'ORANGE': 3, 'YELLOW': 4,
    'GREEN': 5, 'BLUE': 6, 'VIOLET': 7, 'GRAY': 8, 'WHITE': 9,
    'GOLD': -1, 'SILVER': -2
}

def closest_color(hsv_val):
    """หาคู่สีที่ใกล้ที่สุด (โค้ดเดิมของคุณทำไว้ดีแล้ว ขอคงไว้ตามเดิม)"""
    h, s, v = hsv_val
    min_dist = float('inf')
    best_color = 'UNKNOWN'

    for name, ref_hsv in REF_COLORS.items():
        diff_h = abs(h - ref_hsv[0])
        # OpenCV Hue range is 0-179, so wrap around at 180
        if diff_h > 90: diff_h = 180 - diff_h

        diff_s = abs(s - ref_hsv[1])
        diff_v = abs(v - ref_hsv[2])

        # Weighting Logic
        if name in ['BLACK', 'WHITE', 'GRAY', 'SILVER', 'BROWN']:
            weight_h, weight_s, weight_v = 1.0, 1.0, 5.0
        elif name in ['YELLOW']:
            weight_h, weight_s, weight_v = 10.0, 1.0, 1.0
        elif name in ['RED', 'GOLD', 'ORANGE']:
            weight_h, weight_s, weight_v = 3.0, 2.0, 2.0
        else:
            weight_h, weight_s, weight_v = 4.0, 1.0, 1.0

        dist = np.sqrt((diff_h * weight_h) ** 2 + (diff_s * weight_s) ** 2 + (diff_v * weight_v) ** 2)

        if dist < min_dist:
            min_dist = dist
            best_color = name

    return best_color

def fix_false_colors(bands):
    """
    Logic Fix: แก้ไขสีตาม 'กฎทางไฟฟ้า' เท่านั้น (ไม่อิงตามความเพี้ยนของกล้อง)
    เพื่อให้สามารถอ่านตัวต้านทานได้ทุกค่าบนโลก
    """
    if not bands: return bands

    for i, band in enumerate(bands):
        color = band['color']

        # กฎข้อที่ 1: แถบแรก (Band 1) จะต้องไม่ใช่ สีดำ, ทอง หรือ เงิน (ตามหลักไฟฟ้า)
        # ถ้าอ่านได้สีเหล่านี้ แปลว่าระบบมองเงา/แสงสะท้อนผิด ให้ดันไปหาสีที่ใกล้เคียงแทน
        if i == 0 and color in ['BLACK', 'GOLD', 'SILVER']:
            if color == 'BLACK': 
                band['color'] = 'BROWN'  # ดำมักเพี้ยนมาจากน้ำตาลเข้ม
                band['val'] = 1
            else:
                band['color'] = 'ORANGE' # ทอง/เงินสว่างๆ มักเพี้ยนมาจากส้มหรือเหลือง
                band['val'] = 3

        # กฎข้อที่ 2: แถบตัวเลข (Digit Bands - ยกเว้นแถบสุดท้าย) จะต้องไม่ใช่ ทอง หรือ เงิน
        # สมมติมี 4 แถบ (0,1,2,3) แถบ 0, 1 ต้องเป็นตัวเลข
        if i < len(bands) - 2 and color in ['GOLD', 'SILVER']:
            band['color'] = 'YELLOW' if color == 'GOLD' else 'WHITE'
            band['val'] = COLOR_VALS[band['color']]

    # กฎข้อที่ 3: แถบสุดท้าย (Tolerance) ปกติจะเป็น ทอง, เงิน, น้ำตาล, หรือ แดง
    # ถ้าระบบอ่านแถบสุดท้ายได้สีแปลกๆ (เช่น ส้ม, เหลือง) ให้เหมาว่าเป็น ทอง ไว้ก่อน (ค่า Default ที่เจอบ่อยสุด)
    last_band = bands[-1]
    if last_band['color'] not in ['GOLD', 'SILVER', 'BROWN', 'RED']:
        # เช็คความสว่าง ถ้าสว่างมากให้เป็นเงิน ถ้าตุ่นๆ ให้เป็นทอง
        if last_band['mean_hsv'][2] > 180:
            last_band['color'] = 'SILVER'
            last_band['val'] = -2
        else:
            last_band['color'] = 'GOLD'
            last_band['val'] = -1

    return bands

def white_balance(img):
    """
    ปรับสมดุลแสงสีขาวด้วย Gray World Assumption
    ช่วยแก้ปัญหาสีเพี้ยนจากสภาพแสง (อมเหลือง, อมฟ้า)
    """
    # แยกช่องสี B, G, R
    b, g, r = cv2.split(img.astype(np.float32))
    
    # หาค่าเฉลี่ยของแต่ละช่องสี
    m_b, m_g, m_r = np.mean(b), np.mean(g), np.mean(r)
    
    # หาค่าเฉลี่ยรวมของทั้งภาพ
    avg_total = (m_b + m_g + m_r) / 3.0
    
    # ชดเชยสีแต่ละช่องให้สมดุล
    # เติม 1e-5 เพื่อป้องกัน Error หารด้วยศูนย์ในกรณีภาพมืดสนิท
    b = np.clip(b * (avg_total / (m_b + 1e-5)), 0, 255).astype(np.uint8)
    g = np.clip(g * (avg_total / (m_g + 1e-5)), 0, 255).astype(np.uint8)
    r = np.clip(r * (avg_total / (m_r + 1e-5)), 0, 255).astype(np.uint8)
    
    return cv2.merge([b, g, r])

def extract_color_bands(warped_img, num_bands_expected=4):
    """
    หั่นภาพตัวต้านทานเป็นส่วนๆ เพื่อสกัดสี HSV
    """
    # 1. ทำ White Balance ก่อนเพื่อความแม่นยำของสี
    wb_img = white_balance(warped_img)

    h, w = wb_img.shape[:2]
    
    # 2. Crop Center (ตัดขอบบน-ล่างทิ้ง)
    # ตัวต้านทานมักมีความมันวาวตรงขอบ ทำให้เกิดแสงสะท้อน (Specular Highlight) เราจึงเอาแค่ตรงกลาง
    roi_top = int(h * 0.3)
    roi_bottom = int(h * 0.7)
    center_roi = wb_img[roi_top:roi_bottom, :]

    # 3. แปลงเป็น HSV
    hsv_roi = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)

    # 4. ยุบภาพแนวตั้งให้เหลือ 1D Array (เฉลี่ยค่าสีในแต่ละคอลัมน์)
    # จะได้ Array 1 มิติ ความยาวเท่ากับความกว้างภาพ
    column_avg = np.mean(hsv_roi, axis=0).astype(int)

    # 5. หั่นภาพตามแนวขวาง (Slicing)
    # แบ่งภาพออกเป็นโซนๆ (ตามจำนวนแถบ + เผื่อระยะขอบ)
    zones = num_bands_expected + 2 
    zone_width = w // zones
    
    bands_data = []
    
    for i in range(1, zones - 1): # ข้ามขอบซ้ายสุดและขวาสุด (มักเป็นขาเหล็ก)
        start_x = i * zone_width
        end_x = (i + 1) * zone_width
        
        # ดึงค่าสีเฉลี่ยในโซนนั้น
        zone_hsv = column_avg[start_x:end_x]
        mean_hsv = np.mean(zone_hsv, axis=0)
        
        # ปัดเศษให้เป็น int เพื่อส่งเข้า closest_color
        mean_hsv = [int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])]
        
        # ป้องกันการดึงสีของลำตัวตัวต้านทาน (Body Color)
        # คุณสามารถใช้ BODY_LOWER / BODY_UPPER ที่คุณมีมาเช็คตรงนี้ได้
        # ในตัวอย่างนี้ สมมติว่าสีที่สว่างมากและ Saturation ต่ำคือ Body
        if mean_hsv[1] < 40 and mean_hsv[2] > 180:
            continue # ข้ามสีที่เป็นพื้นหลังของตัวต้านทาน
            
        color_name = closest_color(mean_hsv) # เรียกใช้ฟังก์ชันของคุณ
        
        # ดักจับไม่ให้แอดสี UNKNOWN เข้าไป
        if color_name != 'UNKNOWN':
            bands_data.append({
                'mean_hsv': mean_hsv,
                'color': color_name,
                'val': COLOR_VALS.get(color_name, 0)
            })

    # ส่งเข้าฟังก์ชันตรวจสอบกฎทางไฟฟ้าของคุณ
    return fix_false_colors(bands_data)