import networkx as nx
import matplotlib.pyplot as plt
from topology.grid_mapper import map_pixel_to_node

def process_ai_output(ai_keypoints, image_w, image_h):
    circuit_data = []
    for item in ai_keypoints:
        x1, y1, x2, y2, obj_type, value = item
        
        node_a, _, _ = map_pixel_to_node(x1, y1, image_w, image_h)
        node_b, _, _ = map_pixel_to_node(x2, y2, image_w, image_h)
        
        circuit_data.append({
            "type": obj_type,
            "val": value,
            "node_a": node_a,
            "node_b": node_b
        })
    return circuit_data

def read_circuit():
    G = nx.Graph()

    # Mock Data
    components = [
        {"type": "resistor", "val": 100, "node_a": "Power_Plus", "node_b": "Row_5"},
        {"type": "resistor", "val": 220, "node_a": "Row_5", "node_b": "Row_10"},
        {"type": "resistor", "val": 330, "node_a": "Row_5", "node_b": "Row_12"}, # ต่อขนานกับตัวบน
        {"type": "wire", "val": 0, "node_a": "Row_10", "node_b": "Power_Minus"},
    ]

    # Create graph edges based on the components
    for idx, comp in enumerate(components):
        G.add_edge(comp["node_a"], comp["node_b"], 
                   weight=comp["val"], 
                   type=comp["type"], 
                   name=f"{comp['type']}_{idx+1}")

    return G

def check_circuit_status(G, start_node="Power_Plus", end_node="Power_Minus"):
    print("--- 🔍 เริ่มการวิเคราะห์วงจร ---")
    
    # Open Circuit Check
    if nx.has_path(G, start_node, end_node):
        print("✅ วงจรสมบูรณ์: กระแสไฟฟ้าไหลครบวงจร")
        # พิมพ์เส้นทางที่ไฟไหลผ่าน
        paths = list(nx.all_simple_paths(G, start_node, end_node))
        print(f"พบเส้นทางไหลของกระแสไฟฟ้า {len(paths)} เส้นทาง")
    else:
        print("❌ วงจรขาด (Open Circuit)!: กระแสไฟไหลไม่ถึงขั้วลบ")

    # Floating Node
    for node in G.nodes():
        if G.degree(node) == 1 and node not in [start_node, end_node]:
            print(f"⚠️ คำเตือน: พบอุปกรณ์ขาลอยอยู่ที่ช่อง {node} รูปร่างวงจรอาจจะไม่สมบูรณ์")