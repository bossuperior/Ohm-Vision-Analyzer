from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import networkx as nx
from src.topology.rtotal_calculator import RTotalCalculator


def _fmt(ohms: float) -> str:
    if ohms >= 1e6: return f"{ohms/1e6:.2f}M"
    if ohms >= 1e3: return f"{ohms/1e3:.2f}k"
    return f"{ohms:.1f}"


class CircuitAnalyzer:

    @staticmethod
    def apply_wires(all_mapped: List[Dict], wire_ids: Set[int]) -> List[Dict]:
        """Union-Find: merge nodes connected by wires, return non-wire components with merged nodes."""
        parent: Dict[str, str] = {}

        def find(n: str) -> str:
            parent.setdefault(n, n)
            root = n
            while parent[root] != root:
                root = parent[root]
            while parent[n] != root:
                parent[n], n = root, parent[n]
            return root

        def union(a: str, b: str) -> None:
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a

        for c in all_mapped:
            if c['id'] in wire_ids:
                union(c['node1'], c['node2'])

        return [
            {**c, 'node1': find(c['node1']), 'node2': find(c['node2'])}
            for c in all_mapped if c['id'] not in wire_ids
        ]

    def analyze(self, components: List[Dict]) -> Dict:
        """
        components: list of {'id': int, 'node1': str, 'node2': str, 'ohms': float}
        Returns:    {'type': str, 'total_ohms': float, 'formula': str, 'extra': dict}
        """
        if not components:
            return {'type': '—', 'total_ohms': 0.0, 'formula': '', 'extra': {}}

        if len(components) == 1:
            ohms = components[0].get('ohms', 0.0)
            return {'type': 'Single', 'total_ohms': ohms,
                    'formula': _fmt(ohms) if ohms > 0 else '?', 'extra': {}}

        topo  = self._topology(components)
        known = [c for c in components if c.get('ohms', 0.0) > 0]
        total, formula, extra = self._calc(topo, known, components)
        return {'type': topo, 'total_ohms': total, 'formula': formula, 'extra': extra}

    # ── Topology detection ─────────────────────────────────────────────────────
    def _topology(self, components: List[Dict]) -> str:
        if not self._is_connected(components):
            # 4 ตัวต้านทานแต่ขาด edge เดียว (≤2 subgraph) → น่าจะเป็น Wheatstone Bridge
            if len(components) == 4 and self._count_subgraphs(components) <= 2:
                return 'Wheatstone Bridge'
            return 'Not Connected'

        degree: Dict[str, int] = defaultdict(int)
        for c in components:
            degree[c['node1']] += 1
            degree[c['node2']] += 1

        # Parallel: ทุก component ใช้ node pair เดียวกัน
        pairs = [frozenset([c['node1'], c['node2']]) for c in components]
        if len(set(pairs)) == 1:
            return 'Parallel'

        n_nodes = len(degree)
        max_deg = max(degree.values())
        n_term  = sum(1 for d in degree.values() if d == 1)

        # Ring: ทุก node มี degree=2 และ #node == #component (สมบูรณ์)
        if all(d == 2 for d in degree.values()) and n_nodes == len(components):
            return 'Wheatstone Bridge' if len(components) == 4 else 'Ring'

        # 5-resistor Wheatstone Bridge: 4 nodes, degree pattern [2,2,3,3]
        if len(components) == 5 and n_nodes == 4 and sorted(degree.values()) == [2, 2, 3, 3]:
            return 'Wheatstone Bridge (5R)'

        # Wheatstone Bridge (ผ่อนเงื่อนไข) — 4 ตัวต้านทาน
        # รองรับ keypoint ผิดตำแหน่งหลายกรณี: split-node, merge-node, extra junction
        if len(components) == 4 and max_deg <= 4 and 3 <= n_nodes <= 6:
            return 'Wheatstone Bridge'

        # Series: ไม่มี branching, มี terminal 2 จุดพอดี
        if max_deg <= 2 and n_term == 2:
            return 'Series'

        return 'Mixed'

    def _count_subgraphs(self, components: List[Dict]) -> int:
        adj: Dict[str, set] = defaultdict(set)
        all_nodes: Set[str] = set()
        for c in components:
            adj[c['node1']].add(c['node2'])
            adj[c['node2']].add(c['node1'])
            all_nodes.update([c['node1'], c['node2']])
        visited: Set[str] = set()
        count = 0
        for node in all_nodes:
            if node not in visited:
                count += 1
                queue = [node]
                while queue:
                    v = queue.pop()
                    if v in visited:
                        continue
                    visited.add(v)
                    queue.extend(adj[v] - visited)
        return count

    def _is_connected(self, components: List[Dict]) -> bool:
        """BFS — ตรวจว่า component ทุกตัวอยู่ใน connected graph เดียวกัน"""
        adj: Dict[str, set] = defaultdict(set)
        for c in components:
            adj[c['node1']].add(c['node2'])
            adj[c['node2']].add(c['node1'])

        nodes   = set(adj.keys())
        visited = {next(iter(nodes))}
        queue   = list(visited)
        while queue:
            n = queue.pop()
            for nb in adj[n]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return visited == nodes

    # ── Resistance calculation ─────────────────────────────────────────────────
    def _calc(self, topo: str, known: List[Dict],
              all_comps: List[Dict]) -> Tuple[float, str, dict]:

        n_total, n_known = len(all_comps), len(known)
        suffix = f' [{n_known}/{n_total}]' if n_known < n_total else ''

        if topo == 'Wheatstone Bridge':
            return self._calc_wheatstone(all_comps, suffix)

        if topo == 'Wheatstone Bridge (5R)':
            return self._calc_wheatstone5(all_comps, suffix)

        if topo == 'Not Connected':
            parts = [_fmt(c['ohms']) for c in known]
            desc  = ', '.join(parts) if parts else f'{n_total} component(s) (reading...)'
            return 0.0, desc, {}

        if not known:
            return 0.0, f'{n_total} component(s) (reading...)', {}

        vals = [c['ohms'] for c in known]

        if topo == 'Series':
            total   = sum(vals)
            formula = ' + '.join(_fmt(v) for v in vals) + f' = {_fmt(total)} Ohm{suffix}'

        elif topo == 'Parallel':
            inv     = sum(1.0 / v for v in vals if v > 0)
            total   = 1.0 / inv if inv > 0 else 0.0
            formula = ' // '.join(_fmt(v) for v in vals) + f' = {_fmt(total)} Ohm{suffix}'

        else:  # Mixed / Ring — ใช้ nodal analysis (admittance matrix)
            total, formula = self._calc_nodal(all_comps, known, suffix)

        return total, formula, {}

    # ── Mixed / Ring (nodal analysis) ─────────────────────────────────────────
    def _calc_nodal(self, all_comps: List[Dict], known: List[Dict],
                    suffix: str) -> Tuple[float, str]:
        vals = [c['ohms'] for c in known]
        fallback = ', '.join(_fmt(v) for v in vals) + suffix

        if len(known) < len(all_comps):
            return 0.0, fallback  # ยังอ่านไม่ครบ

        # สร้าง graph
        G = nx.MultiGraph()
        for c in all_comps:
            ohms = c.get('ohms', 0.0)
            G.add_edge(c['node1'], c['node2'], ohms=ohms if ohms > 0 else 1e-6)

        # หา terminals จาก simple degree (collapse parallel edges)
        # ทำให้ node ที่มีขา 2 เส้นขนานกัน ยังถูกนับ degree=1 ได้ถูกต้อง
        simple_neighbors: Dict[str, Set[str]] = defaultdict(set)
        for c in all_comps:
            simple_neighbors[c['node1']].add(c['node2'])
            simple_neighbors[c['node2']].add(c['node1'])

        # เลือก 2 nodes ที่มี simple-degree ต่ำที่สุด = terminal ของวงจร
        sorted_nodes = sorted(simple_neighbors, key=lambda n: len(simple_neighbors[n]))
        if len(sorted_nodes) < 2:
            return 0.0, fallback

        min_deg = len(simple_neighbors[sorted_nodes[0]])
        terminals = [n for n in sorted_nodes if len(simple_neighbors[n]) == min_deg]
        if len(terminals) >= 2:
            node_start, node_end = terminals[0], terminals[-1]
        else:
            # terminal เดียว (เช่น ขนานล้วน) → ใช้คู่ที่ degree ห่างกันมากที่สุด
            node_start, node_end = sorted_nodes[0], sorted_nodes[-1]

        try:
            calc  = RTotalCalculator()
            total = calc.calculate(G, node_start, node_end)
            if total <= 0 or total == float('inf'):
                return 0.0, fallback
            formula = (', '.join(_fmt(v) for v in vals)
                       + f' → Rtotal = {_fmt(total)} Ohm{suffix}')
            return total, formula
        except Exception:
            return 0.0, fallback

    # ── Wheatstone Bridge ──────────────────────────────────────────────────────
    def _calc_wheatstone(self, components: List[Dict],
                         suffix: str) -> Tuple[float, str, dict]:
        """
        วิเคราะห์ Wheatstone Bridge:
          - เรียง 4 ตัวเป็น cycle A-B-C-D-A
          - Req(A-C) = (R1+R2) // (R4+R3)
          - Req(B-D) = (R2+R3) // (R1+R4)
          - Balanced: R1*R3 ≈ R2*R4
        """
        cycle = self._order_cycle(components)
        if cycle is None:
            # nodes ≠ 4 (keypoint misalignment) → nodal analysis fallback
            known = [c for c in components if c.get('ohms', 0.0) > 0]
            n_known = len(known)
            if n_known < len(components):
                return 0.0, f'{n_known}/4 resistors known{suffix}', {}
            total, formula = self._calc_nodal(components, known, suffix)
            extra = {'balanced': None, 'req_ac': total, 'req_bd': 0.0}
            return total, formula, extra

        def get_ohms(n1, n2):
            for c in components:
                if {c['node1'], c['node2']} == {n1, n2}:
                    return c.get('ohms', 0.0)
            return 0.0

        # R1=A-B, R2=B-C, R3=C-D, R4=D-A
        R1 = get_ohms(cycle[0], cycle[1])
        R2 = get_ohms(cycle[1], cycle[2])
        R3 = get_ohms(cycle[2], cycle[3])
        R4 = get_ohms(cycle[3], cycle[0])

        extra = {'R': [R1, R2, R3, R4], 'nodes': cycle,
                 'balanced': None, 'req_ac': 0.0, 'req_bd': 0.0}

        if 0 in (R1, R2, R3, R4):
            known_cnt = sum(1 for r in (R1,R2,R3,R4) if r > 0)
            return 0.0, f'{known_cnt}/4 resistors known{suffix}', extra

        # Equivalent resistances
        req_ac = (R1+R2) * (R4+R3) / (R1+R2+R4+R3)
        req_bd = (R2+R3) * (R1+R4) / (R2+R3+R1+R4)

        # Balance: |R1*R3 - R2*R4| / max(product) < 5%
        p1, p2  = R1 * R3, R2 * R4
        balanced = abs(p1 - p2) / max(p1, p2, 1e-9) < 0.05

        extra.update({'balanced': balanced, 'req_ac': req_ac, 'req_bd': req_bd})

        formula = (f'R1={_fmt(R1)}  R2={_fmt(R2)}\n'
                   f'R3={_fmt(R3)}  R4={_fmt(R4)}')

        return req_ac, formula, extra

    # ── Wheatstone Bridge (5R — loaded bridge with galvanometer) ──────────────
    def _calc_wheatstone5(self, components: List[Dict],
                          suffix: str) -> Tuple[float, str, dict]:
        """
        5-resistor loaded Wheatstone Bridge:
          A-B=R1, B-C=R2, A-D=R3, D-C=R4, B-D=Rg
          Balance: R1*R4 = R2*R3 (same ratio as 4R bridge)
          Req: exact nodal analysis (no simple formula when Rg≠∞)
        """
        degree: Dict[str, int] = defaultdict(int)
        for c in components:
            degree[c['node1']] += 1
            degree[c['node2']] += 1

        terminals = [n for n, d in degree.items() if d == 2]
        middles   = [n for n, d in degree.items() if d == 3]

        known   = [c for c in components if c.get('ohms', 0.0) > 0]
        n_known = len(known)
        n_total = len(components)

        if len(terminals) != 2 or len(middles) != 2:
            total, formula = self._calc_nodal(components, known, suffix)
            return total, formula, {}

        # แยก Rg (ตัวกลาง B-D) ออกจาก 4 แขน
        rg_comp  = None
        arm_comps: List[Dict] = []
        for c in components:
            if {c['node1'], c['node2']} == set(middles):
                rg_comp = c
            else:
                arm_comps.append(c)

        if n_known < n_total:
            parts = [_fmt(c['ohms']) for c in known]
            return 0.0, f'{n_known}/{n_total} known: ' + ', '.join(parts), {}

        rg_ohms = rg_comp.get('ohms', 0.0) if rg_comp else 0.0

        # ระบุ 4 แขน: A-B=R1, B-C=R2, A-D=R3, D-C=R4
        A, C = terminals[0], terminals[1]
        B, D = middles[0], middles[1]

        def get_arm(n1: str, n2: str) -> float:
            for c in arm_comps:
                if {c['node1'], c['node2']} == {n1, n2}:
                    return c.get('ohms', 0.0)
            return 0.0

        R1 = get_arm(A, B); R2 = get_arm(B, C)
        R3 = get_arm(A, D); R4 = get_arm(D, C)
        if 0 in (R1, R2, R3, R4):          # ลอง swap B↔D
            R1 = get_arm(A, D); R2 = get_arm(D, C)
            R3 = get_arm(A, B); R4 = get_arm(B, C)

        # Nodal analysis → Req แม่นยำ
        G = nx.MultiGraph()
        for c in components:
            ohms = c.get('ohms', 0.0)
            G.add_edge(c['node1'], c['node2'], ohms=ohms if ohms > 0 else 1e-6)
        try:
            total = RTotalCalculator().calculate(G, A, C)
            if total <= 0 or total == float('inf'):
                total = 0.0
        except Exception:
            total = 0.0

        # Balance: R1/R2 = R3/R4  →  R1*R4 = R2*R3
        p1, p2   = R1 * R4, R2 * R3
        balanced = abs(p1 - p2) / max(p1, p2, 1e-9) < 0.05 if (p1 > 0 and p2 > 0) else False
        rg_str = _fmt(rg_ohms) if rg_ohms > 0 else '?'
        formula = (f'R1={_fmt(R1)}  R2={_fmt(R2)}\n'
                   f'R3={_fmt(R3)}  R4={_fmt(R4)}\n'
                   f'Rg={rg_str}')

        extra = {'R': [R1, R2, R3, R4], 'Rg': rg_ohms, 'balanced': balanced,
                 'req_ac': total, 'req_bd': 0.0}
        return total, formula, extra

    def _order_cycle(self, components: List[Dict]) -> Optional[List[str]]:
        """เรียง nodes เป็น cycle A→B→C→D→A"""
        adj: Dict[str, List[str]] = defaultdict(list)
        for c in components:
            adj[c['node1']].append(c['node2'])
            adj[c['node2']].append(c['node1'])

        nodes = list(adj.keys())
        if len(nodes) != 4:
            return None

        # Traverse
        start = nodes[0]
        cycle, prev, curr = [start], None, start
        for _ in range(3):
            nxt = [n for n in adj[curr] if n != prev]
            if not nxt:
                return None
            cycle.append(nxt[0])
            prev, curr = curr, nxt[0]
        return cycle  # [A, B, C, D]
