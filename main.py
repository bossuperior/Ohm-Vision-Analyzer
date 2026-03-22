import cv2
# Import โมดูลที่เราเขียนไว้ทั้งหมด
from src.utils.img_reader import img_reader
from src.vision.color_reader import scan_bands_statistical
from src.analyzer.cal_resistance import calculate_resistance
from src.topology.grid_mapper import extract_components_coordinates
from src.topology.graph_builder import CircuitGraph
from src.analyzer.solver import solve_total_resistance

def run_ohm_vision(image_path, resistor_boxes, wire_boxes):
    print("--- 1. Vision Module: Reading Image ---")
    raw_img = cv2.imread(image_path)
    h_img, w_img = raw_img.shape[:2]
    
    # ดึงค่า R จากภาพ
    processed_resistors = []
    for box in resistor_boxes:
        warped_img, true_len = img_reader(raw_img, box, padding=10)
        bands_data = scan_bands_statistical(warped_img, true_len, padding=10)
        numeric_ohm, display_str = calculate_resistance(bands_data)
        
        if numeric_ohm is not None:
            processed_resistors.append((box, numeric_ohm))
            print(f"Detected: {display_str}")

    print("\n--- 2. Topology Module: Building Graph ---")
    # แมปพิกัดลงบอร์ด
    components = extract_components_coordinates(processed_resistors, w_img, h_img)
    
    circuit = CircuitGraph()
    for comp in components:
        circuit.add_component(comp['type'], comp['nodes'][0], comp['nodes'][1], comp['value'])
        
    # สมมติมีสายจั๊มเปอร์ (ต้องมี wire_boxes ด้วยในอนาคต)
    # circuit.add_component('wire', 'Row_1_Top', 'Row_5_Top', 0)
    
    circuit.simplify_wires()
    circuit.print_circuit()

    print("\n--- 3. Analyzer Module: Solving Circuit ---")
    # กำหนดจุดวัดไฟ (Probe) สมมติวัดจากแถว 1 ไปแถว 10
    start_probe = "Row_1_Top"
    end_probe = "Row_10_Bottom"
    
    r_total = solve_total_resistance(circuit.get_graph(), start_probe, end_probe)
    print(f"==============================")
    print(f" R Total ({start_probe} to {end_probe}) = {r_total} Ohms")
    print(f"==============================")

# ทดสอบระบบ
if __name__ == "__main__":
    # จำลองกล่อง Bounding Box [x_min, y_min, x_max, y_max]
    mock_r1_box = [10, 50, 100, 60]
    mock_r2_box = [120, 50, 220, 60]
    
    run_ohm_vision("data/raw/test1.jpg", [mock_r1_box, mock_r2_box], [])