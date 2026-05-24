from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict


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

        else:  # Mixed / Ring
            total   = 0.0
            formula = ', '.join(_fmt(v) for v in vals) + suffix

        return total, formula, {}

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
            return 0.0, 'Bridge topology error', {}

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

        bal_str = '[OK] Balanced' if balanced else '[!!] Unbalanced'
        formula = (f'R1={_fmt(R1)}  R2={_fmt(R2)}\n'
                   f'R3={_fmt(R3)}  R4={_fmt(R4)}\n'
                   f'Req={_fmt(req_ac)} / {_fmt(req_bd)}')

        return req_ac, formula, extra

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
