import networkx as nx

class CircuitGraph:
    def __init__(self):
        # สร้าง MultiGraph เพราะจุด Node 2 จุด อาจจะมีตัวต้านทานต่อขนานกันมากกว่า 1 ตัวก็ได้
        self.graph = nx.MultiGraph()
    
    def add_component(self, comp_type, node1, node2, value=None):
        """
        เพิ่มอุปกรณ์ลงในกราฟ
        :param comp_type: 'resistor' หรือ 'wire'
        :param node1: ชื่อ Node ฝั่งที่ 1 (เช่น 'Row_10_AE')
        :param node2: ชื่อ Node ฝั่งที่ 2 (เช่น 'Row_15_FJ')
        :param value: ค่าความต้านทาน (ถ้าเป็น wire ค่าจะเป็น 0)
        """
        if comp_type == 'wire':
            # สายจั๊มเปอร์ คือตัวต้านทานที่มีค่า 0 โอห์ม (ในทางอุดมคติ)
            # หรืออาจจะยุบ (Merge) 2 Node นี้เข้าด้วยกันเลยก็ได้
            self.graph.add_edge(node1, node2, type='wire', weight=0)
            
        elif comp_type == 'resistor':
            self.graph.add_edge(node1, node2, type='resistor', weight=value)

    def get_graph(self):
        return self.graph

    def print_circuit(self):
        """สำหรับ Debug ดูว่าวงจรต่อกันถูกไหม"""
        for u, v, data in self.graph.edges(data=True):
            print(f"{data['type'].upper()}: Node {u} <---> Node {v} | Value: {data['weight']} Ohms")