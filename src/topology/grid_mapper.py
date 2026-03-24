import cv2

TOTAL_ROWS = 30
offset_length = 35   
offset_width = 40

terminal_top_ratio = 0.18
terminal_bottom_ratio = 0.82
rails_top_ratio = 0.35
rails_bottom_ratio = 0.65

def calculate_spacing(target_len):
    return (target_len - (2 * offset_length)) / (TOTAL_ROWS - 1)

def draw_grid(frame, target_w, target_h):
    grid_img = frame.copy()
    is_horizontal = target_w > target_h

    if is_horizontal: # Horizontal orientation
        spacing = calculate_spacing(target_w)
        terminal_top_bound = int(target_h * terminal_top_ratio)
        terminal_bottom_bound = int(target_h * terminal_bottom_ratio)
        # Draw rails boundaries (top and bottom)
        cv2.line(grid_img, (0, int(terminal_top_bound * rails_top_ratio)), (target_w, int(terminal_top_bound * rails_top_ratio)), (0, 0, 255), 1)
        cv2.line(grid_img, (0, int(terminal_top_bound * rails_bottom_ratio)), (target_w, int(terminal_top_bound * rails_bottom_ratio)), (255, 0, 0), 1)
        bottom_h = target_h - terminal_bottom_bound
        cv2.line(grid_img, (0, int(terminal_bottom_bound + bottom_h * rails_top_ratio)), (target_w, int(terminal_bottom_bound + bottom_h * rails_top_ratio)), (0, 0, 255), 1)
        cv2.line(grid_img, (0, int(terminal_bottom_bound + bottom_h * rails_bottom_ratio)), (target_w, int(terminal_bottom_bound + bottom_h * rails_bottom_ratio)), (255, 0, 0), 1)
        # Draw vertical grid lines for terminal strips (blue)
        for i in range(TOTAL_ROWS):
            x = int(offset_length + (i * spacing))
            cv2.line(grid_img, (x, terminal_top_bound), (x, terminal_bottom_bound), (0, 150, 0), 1)
            
        # Draw center line to separate top and bottom power rails
        cv2.line(grid_img, (0, target_h//2), (target_w, target_h//2), (0, 0, 255), 2)
        
    else: # Vertical orientation
        spacing = calculate_spacing(target_h)
        power_left_bound = int(target_w * terminal_top_ratio)
        power_right_bound = int(target_w * terminal_bottom_ratio)
        
        # Draw rails boundaries (left and right)
        cv2.line(grid_img, (power_left_bound, 0), (power_left_bound, target_h), (255, 255, 0), 1)
        cv2.line(grid_img, (power_right_bound, 0), (power_right_bound, target_h), (255, 255, 0), 1)
        right_w = target_w - power_right_bound
        cv2.line(grid_img, (int(power_right_bound + right_w * rails_top_ratio), 0), (int(power_right_bound + right_w * rails_top_ratio), target_h), (0, 0, 255), 1)
        cv2.line(grid_img, (int(power_right_bound + right_w * rails_bottom_ratio), 0), (int(power_right_bound + right_w * rails_bottom_ratio), target_h), (255, 0, 0), 1)
        
        for i in range(TOTAL_ROWS):
             y = int(offset_length + (i * spacing))
             cv2.line(grid_img, (power_left_bound, y), (power_right_bound, y), (0, 150, 0), 1)
            
        cv2.line(grid_img, (target_w//2, 0), (target_w//2, target_h), (0, 0, 255), 2)
        
    return grid_img

def map_pixel_to_node(x, y, target_w, target_h, total_rows=TOTAL_ROWS):
    is_horizontal = target_w > target_h
    node_name = ""
    snapped_x, snapped_y = x, y

    if is_horizontal: # Horizontal orientation
         spacing = calculate_spacing(target_w)
         # Find nearest row index (1-based)
         row_idx = int(round((x - offset_length) / spacing)) + 1
         row_idx = max(1, min(row_idx, TOTAL_ROWS))
         # Calculate snapped coordinates
         snapped_x = int(offset_length + (row_idx - 1) * spacing)
         
         power_top_bound = target_h * terminal_top_ratio
         power_bottom_bound = target_h * terminal_bottom_ratio
         
         # Power Rails Zones
         if y < power_top_bound:
            is_plus = y < power_top_bound / 2
            ratio = rails_top_ratio if is_plus else rails_bottom_ratio
            snapped_y = int(power_top_bound * ratio)
            node_name = "Power_Top_Plus" if is_plus else "Power_Top_Minus"
         elif y > power_bottom_bound:
            is_plus = y < target_h - (target_h - power_bottom_bound) / 2
            ratio = rails_bottom_ratio if is_plus else rails_top_ratio
            snapped_y = int(target_h - (target_h - power_bottom_bound) * ratio)
            node_name = "Power_Bottom_Plus" if is_plus else "Power_Bottom_Minus"
         # Terminal Strips Zones
         else: 
            side = "Top" if y < target_h / 2 else "Bottom"
            node_name = f"Row_{row_idx}_{side}"
            snapped_y = y  # Keep original y for terminal strips
             
    else: # Vertical orientation
        spacing = calculate_spacing(target_h)
        row_idx = int(round((y - offset_length) / spacing)) + 1
        row_idx = max(1, min(row_idx, TOTAL_ROWS))
        
        snapped_y = int(offset_length + (row_idx - 1) * spacing)
        power_left_bound = target_w * terminal_top_ratio
        power_right_bound = target_w * terminal_bottom_ratio
        
        if x < power_left_bound:
            is_plus = x < power_left_bound / 2
            ratio = rails_top_ratio if is_plus else rails_bottom_ratio
            snapped_x = int(power_left_bound * ratio)
            node_name = "Power_Left_Plus" if is_plus else "Power_Left_Minus"
        elif x > power_right_bound:
            is_plus = x < target_w - (target_w - power_right_bound) / 2
            node_name = "Power_Right_Plus" if is_plus else "Power_Right_Minus"
            ratio = rails_top_ratio if is_plus else rails_bottom_ratio
            snapped_x = int(power_right_bound + (target_w - power_right_bound) * ratio)
        else: 
            side = "Left" if x < target_w / 2 else "Right"
            node_name = f"Row_{row_idx}_{side}"
            snapped_x = x  # Keep original x for terminal strips
    return node_name, snapped_x, snapped_y

def topology_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        target_w, target_h = param
        node_name, snapped_x, snapped_y = map_pixel_to_node(x, y, target_w, target_h)

        print(f"📍 พิกัด (x={snapped_x}, y={snapped_y}) คือจุดเชื่อมต่อ: {node_name}")