import networkx as nx

def reduce_parallel(graph):
    """ยุบตัวต้านทานที่ต่อขนานกัน (Parallel)"""
    reduced = False
    for u in list(graph.nodes()):
        for v in list(graph.neighbors(u)):
            # ถ้ามีเส้นเชื่อมระหว่าง u กับ v มากกว่า 1 เส้น (ขนานกัน)
            if graph.number_of_edges(u, v) > 1:
                edges = graph[u][v]
                # คำนวณสูตรขนาน: 1 / (1/R1 + 1/R2 + ...)
                sum_inv = sum(1.0 / data['weight'] for key, data in edges.items() if data['weight'] > 0)
                if sum_inv > 0:
                    r_eq = 1.0 / sum_inv
                    # ลบเส้นเดิมทิ้งทั้งหมด แล้วสร้างเส้นใหม่เส้นเดียว
                    keys_to_remove = list(edges.keys())
                    for k in keys_to_remove:
                        graph.remove_edge(u, v, key=k)
                    graph.add_edge(u, v, type='resistor', weight=r_eq)
                    reduced = True
    return reduced

def reduce_series(graph):
    """ยุบตัวต้านทานที่ต่ออนุกรมกัน (Series)"""
    reduced = False
    for node in list(graph.nodes()):
        # ถ้า Node นั้นมีแขนเชื่อมแค่ 2 ทาง (degree = 2) แปลว่าต่ออนุกรมชัวร์ๆ
        if graph.degree(node) == 2:
            neighbors = list(graph.neighbors(node))
            if len(neighbors) == 2:
                u, v = neighbors[0], neighbors[1]
                # ดึงค่า R ออกมาบวกกัน
                r1 = graph[u][node][0]['weight']
                r2 = graph[node][v][0]['weight']
                r_eq = r1 + r2
                
                # ลบโหนดตรงกลางทิ้ง แล้วลากเส้นเชื่อม u กับ v โดยตรง
                graph.remove_node(node)
                graph.add_edge(u, v, type='resistor', weight=r_eq)
                reduced = True
                break # ยุบทีละโหนด ป้องกัน Graph เปลี่ยนรูประหว่างวนลูป
    return reduced

def solve_total_resistance(graph, start_node, end_node):
    """ฟังก์ชันหลัก: ลูปการยุบวงจรจนกว่าจะเหลือแค่ R ตัวเดียว"""
    G_copy = graph.copy()
    
    # วนลูปยุบขนานและอนุกรมสลับกันไปเรื่อยๆ จนกว่าจะยุบไม่ได้แล้ว
    while True:
        if reduce_parallel(G_copy): continue
        if reduce_series(G_copy): continue
        break # ถ้าทำทั้งคู่ไม่ได้แล้ว ให้หลุดลูป
        
    # เช็คว่ายุบจนเหลือเส้นเดียวเชื่อมระหว่าง start กับ end หรือยัง
    if G_copy.has_edge(start_node, end_node):
        return G_copy[start_node][end_node][0]['weight']
    else:
        return "Complex Circuit (Star-Delta needed) or Broken Circuit"