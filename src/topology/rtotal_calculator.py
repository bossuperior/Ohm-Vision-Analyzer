import networkx as nx
import numpy as np

class RTotalCalculator:
    def __init__(self):
        pass

    def calculate(self, G: nx.MultiGraph, node_start: str, node_end: str) -> float:
        '''Calculates the total resistance between two nodes in a circuit graph 
        using the Admittance Matrix method. Returns the equivalent resistance in Ohms.
        '''
        if not G.has_node(node_start) or not G.has_node(node_end):
            return 0.0
            
        if node_start == node_end:
            return 0.0
        
        if not nx.has_path(G, node_start, node_end):
            print(f" Open Circuit: No path between {node_start} and {node_end}")
            return float('inf')

        '''
        Simplify the graph by collapsing parallel edges into a single edge with equivalent conductance (G = 1/R).
        Then, construct the Admittance Matrix (Y-Matrix) for the simplified graph, and solve for 
        the equivalent resistance between the two specified nodes.
        '''
        simple_G = nx.Graph()
        for u, v, data in G.edges(data=True):
            ohms = data.get('ohms', 0.0)
            
            #Prevent division by zero for wires or unread resistors 
            if ohms <= 0:
                ohms = 1e-6 
            conductance = 1.0 / ohms
            
            if simple_G.has_edge(u, v):
                # If an edge already exists, we are in a parallel configuration, so we add the conductances
                simple_G[u][v]['conductance'] += conductance
            else:
                simple_G.add_edge(u, v, conductance=conductance)

        # Prepare the Y-Matrix based on the simplified graph
        nodes = list(simple_G.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        Y = np.zeros((n, n))
        
        # Initialize the Y-Matrix with conductance values
        for u, v, data in simple_G.edges(data=True):
            i, j = node_to_idx[u], node_to_idx[v]
            g = data['conductance']
            Y[i, i] += g
            Y[j, j] += g
            Y[i, j] -= g
            Y[j, i] -= g

        # Define the current injection vector (I) for the two nodes of interest
        # If I = 1 ; V = R_total. If I = 0 ; V = 0. By setting I[node_start] = 1 and I[node_end] = -1, we can solve for V and thus find R_total.
        I = np.zeros(n)
        start_idx = node_to_idx[node_start]
        end_idx = node_to_idx[node_end]
        I[start_idx] = 1.0
        I[end_idx] = -1.0

        # Define ground/reference node for calculation (we can choose any node as ground, but we will exclude it from the Y-Matrix to solve for the remaining nodes)
        reduced_Y = np.delete(np.delete(Y, end_idx, axis=0), end_idx, axis=1)
        reduced_I = np.delete(I, end_idx)

        try:
            # Voltage calculation using Ohm's Law in matrix form: V = Y^-1 * I
            V = np.linalg.solve(reduced_Y, reduced_I)
            
            if start_idx > end_idx:
                v_start = V[start_idx - 1]
            else:
                v_start = V[start_idx]
            total_resistance = abs(v_start)
            return round(total_resistance, 2)
            
        except np.linalg.LinAlgError:
            print(" Matrix is singular (Open Circuit detected)")
            return float('inf')