import networkx as nx
from typing import List, Dict, Tuple
from src.topology.rtotal_calculator import RTotalCalculator

class CircuitDetector:
    def __init__(self):
        # Define a template graph for a Wheatstone Bridge
        # 4 nodes (A, B, C, D) with 5 resistor edges: A-B, A-C, B-D, C-D, and B-C (crossbar)
        self.wheatstone_template = nx.Graph()
        self.wheatstone_template.add_edges_from([
            (1, 2), (1, 3), (2, 4), (3, 4), (2, 3)
        ])

    def analyze_topology(self, mapped_components: List[Dict], resistor_values: List[Dict]) -> Tuple[str, nx.Graph, float]:
        """
        Public Entry Point.
        Takes grid locations and color band values to analyze the circuit.
        Returns: (Circuit Type String, NetworkX Graph, Total Resistance)
        """
        if not mapped_components:
            return "Open Circuit", nx.MultiGraph(), 0.0

        # 1. Build the raw physical graph
        G = self._build_raw_graph(mapped_components, resistor_values)

        # 2. Collapse jumper wires (merge nodes) to create the true logical schematic
        G = self._collapse_wires(G)

        # 3. Detect Topology (Series, Parallel, Wheatstone, Complex)
        circuit_type = self._classify_topology(G)

        # 4. Calculate total resistance (Auto-Detect Rails)
        calculator = RTotalCalculator()
        
        possible_starts = ["Power_Top_Rail_A", "Power_Top_Rail_B"]
        possible_ends = ["Power_Bottom_Rail_C", "Power_Bottom_Rail_D"]
        
        start_node = next((n for n in possible_starts if G.has_node(n)), None)
        end_node = next((n for n in possible_ends if G.has_node(n)), None)
        
        total_r = 0.0 
        
        # ถ้าเจอทั้งจุดเริ่มต้นและจุดสิ้นสุด ให้รันคณิตศาสตร์หา R รวม
        if start_node and end_node:
            total_r = calculator.calculate(G, start_node, end_node)
        elif G.number_of_nodes() > 0 and circuit_type != "Open Circuit":
            # กรณีผู้ใช้ไม่ได้ต่อลงรางไฟเลย (ต่อลอยๆ กลางบอร์ด) 
            # เราสามารถดึงโหนดที่อยู่ซ้ายสุด-ขวาสุด มาคำนวณแก้ขัดได้ครับ (ถ้าต้องการ)
            pass

        return circuit_type, G, total_r

    def _build_raw_graph(self, mapped_components: List[Dict], resistor_values: List[Dict]) -> nx.MultiGraph:
        """Constructs a graph where Nodes = Breadboard Rows, Edges = Components."""
        # We use MultiGraph because two resistors can be plugged into the exact same rows (Parallel)
        G = nx.MultiGraph()

        # For simplicity, assuming mapped_components and resistor_values align by index/ID
        for comp in mapped_components:
            node_a = comp['node1']
            node_b = comp['node2']
            comp_id = comp['id']

            # Match the value from the BandReader (using a safe fallback)
            # You may need to adjust this lookup based on how you pass data from the pipeline
            val_data = next((item for item in resistor_values if item.get('id') == comp_id), None)

            # Determine if this component is a wire or a resistor
            # If BandReader couldn't read it or it's flagged as a wire, treat it as 0 Ohms
            if val_data and 'Ohms' in val_data.get('string_val', ''):
                comp_type = 'resistor'
                ohms = val_data.get('numeric_val', 0.0)
            else:
                comp_type = 'wire'
                ohms = 0.0

            G.add_edge(node_a, node_b, key=comp_id, type=comp_type, ohms=ohms)

        return G

    def _collapse_wires(self, G: nx.MultiGraph) -> nx.MultiGraph:
        """
        Critical Logical Step:
        Finds all edges that are 'wires' and merges their endpoint nodes.
        This transforms physical breadboard geography into a logical schematic.
        """
        # We iterate over a copy of the edges to safely modify the graph
        wire_edges = [(u, v, k) for u, v, k, data in G.edges(keys=True, data=True) if data.get('type') == 'wire']

        for u, v, k in wire_edges:
            if G.has_node(u) and G.has_node(v) and u != v:
                # Contract node v into node u.
                # All resistors connected to v will automatically re-route to connect to u!
                G = nx.contracted_nodes(G, u, v, self_loops=False)

        return G

    def _classify_topology(self, G: nx.MultiGraph) -> str:
        """Analyzes the node degrees and connections to determine the circuit type."""
        # Convert MultiGraph to simple Graph to analyze structural shape
        # (This ignores duplicate parallel edges temporarily for shape detection)
        simple_G = nx.Graph(G)

        num_edges = simple_G.number_of_edges()
        num_nodes = simple_G.number_of_nodes()

        # Edge cases
        if num_edges == 0:
            return "Open Circuit"
        if not nx.is_connected(simple_G):
            return "Disconnected / Multiple Subcircuits"

        # Check 1: Simple Series
        # In a perfect series circuit, every internal node connects exactly 2 components (degree 2)
        # and the endpoints (where power goes) connect 1 component (degree 1).
        degrees = [deg for node, deg in simple_G.degree()]
        if all(d <= 2 for d in degrees) and num_nodes == num_edges + 1:
            return "Series Circuit"

        # Check 2: Simple Parallel
        # In a perfect parallel circuit, there are exactly 2 main nodes, and ALL components connect them.
        if num_nodes == 2 and G.number_of_edges() > 1:  # Note: Using G here to count multiple edges
            return "Parallel Circuit"

        # Check 3: Wheatstone Bridge (Using Graph Isomorphism!)
        # networkx handles the complex math of checking if our user's graph
        # perfectly matches the structural shape of a Wheatstone Bridge template.
        if nx.is_isomorphic(simple_G, self.wheatstone_template):
            return "Wheatstone Bridge"

        # Fallback
        return "Complex / Mixed Circuit"