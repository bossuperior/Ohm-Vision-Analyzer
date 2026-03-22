def map_pixel_to_node(x, y, board_width, board_height, total_rows=30):
    """
    แปลงพิกัด (x, y) ของขาอุปกรณ์ ให้กลายเป็นชื่อ Node บนเบรดบอร์ด
    เช่น Row_1_Top, Row_15_Bottom
    """
    # 1. ป้องกันค่า x, y ทะลุขอบภาพ
    x = max(0, min(x, board_width - 1))
    y = max(0, min(y, board_height - 1))

    # 2. คำนวณว่าอยู่ "แถว" (Row) ที่เท่าไหร่ (สมมติเบรดบอร์ดมี 30 แถว)
    row_index = int((x / board_width) * total_rows) + 1

    # 3. คำนวณว่าอยู่ร่องบน (A-E) หรือร่องล่าง (F-J)
    # สมมติว่าร่องกลางเบรดบอร์ดแบ่งครึ่งที่ y = board_height / 2
    if y < (board_height / 2):
        side = "Top"    # ร่อง A-E
    else:
        side = "Bottom" # ร่อง F-J

    return f"Row_{row_index}_{side}"

def extract_components_coordinates(bounding_boxes, board_width, board_height):
    """
    รับกล่อง Bounding Box (จาก YOLO) แล้วคืนค่าจุดหัว-ท้ายของอุปกรณ์
    เพื่อส่งไปหา map_pixel_to_node
    """
    components = []
    for box, val in bounding_boxes:
        # สมมติ box คือ [x_min, y_min, x_max, y_max] ของตัวต้านทานแนวนอน
        x_min, y_min, x_max, y_max = box
        
        # ขาซ้าย
        node1 = map_pixel_to_node(x_min, (y_min + y_max)//2, board_width, board_height)
        # ขาขวา
        node2 = map_pixel_to_node(x_max, (y_min + y_max)//2, board_width, board_height)
        
        components.append({'type': 'resistor', 'nodes': (node1, node2), 'value': val})
    return components