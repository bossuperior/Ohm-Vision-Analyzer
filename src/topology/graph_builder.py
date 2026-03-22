import networkx as nx

class CircuitGraph:
    def __init__(self):
        # สร้าง MultiGraph รองรับการต่อขนาน
        self.graph = nx.MultiGraph()
    
    def add_component(self, comp_type, node1, node2, value=None):
        """
        เพิ่มอุปกรณ์ลงในกราฟ
        """
        if comp_type == 'wire':
            # เก็บเป็น edge ไว้ก่อน เพื่อให้ง่ายต่อการ Debug 
            self.graph.add_edge(node1, node2, type='wire', weight=0)
            
        elif comp_type == 'resistor':
            self.graph.add_edge(node1, node2, type='resistor', weight=value)

    def simplify_wires(self):
        """
        ยุบ Node ที่เชื่อมต่อกันด้วยสายจั๊มเปอร์ให้กลายเป็น Node เดียวกัน (Short Circuit)
        เพื่อป้องกัน Error หารด้วยศูนย์ตอนคำนวณ
        """
        # 1. ค้นหาสายไฟ (wire) ทั้งหมดในวงจร
        wires = [(u, v) for u, v, data in self.graph.edges(data=True) if data.get('type') == 'wire']

        # 2. ทำการยุบ Node เข้าด้วยกัน
        for u, v in wires:
            if self.graph.has_node(u) and self.graph.has_node(v) and u != v:
                # ฟังก์ชัน contracted_nodes จะดึง Edge ทั้งหมดของ v ไปผูกกับ u แล้วลบ v ทิ้ง
                self.graph = nx.contracted_nodes(self.graph, u, v, self_loops=False)

        # 3. ลบ Edge ที่เป็นสายไฟทิ้งให้หมด เพราะตอนนี้ Node ชนกันเรียบร้อยแล้ว
        edges_to_remove = [(u, v, k) for u, v, k, data in self.graph.edges(data=True, keys=True) if data.get('type') == 'wire']
        self.graph.remove_edges_from(edges_to_remove)

    def get_graph(self):
        return self.graph

    def print_circuit(self):
        """สำหรับ Debug ดูว่าวงจรต่อกันถูกไหม"""
        print("--- Current Circuit Topology ---")
        for u, v, data in self.graph.edges(data=True):
            print(f"{data['type'].upper()}: Node {u} <---> Node {v} | Value: {data['weight']} Ohms")
        print("--------------------------------")