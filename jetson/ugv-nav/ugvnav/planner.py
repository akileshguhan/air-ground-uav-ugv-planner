"""A* over accepted segments, cost = length / speed factor, turn-radius pruning."""
from __future__ import annotations
import heapq
import math
import numpy as np
from .features import turning_radius_from_three_points


class HorizonPlanner:
    def __init__(self, min_turn_radius_m: float):
        self.min_turn_radius_m = float(min_turn_radius_m)

    def find_path(self, wps, adj, start, goal):
        def h(n):
            return float(np.linalg.norm(wps[n] - wps[goal]))
        st = (start, None)
        openq = [(h(start), 0.0, st, [start])]; best = {st: 0.0}
        while openq:
            _, g, (cur, prev), path = heapq.heappop(openq)
            if cur == goal:
                return True, path, g
            for nb, cost in adj.get(cur, []):
                if len(path) >= 2 and turning_radius_from_three_points(
                        wps[path[-2]], wps[cur], wps[nb]) < self.min_turn_radius_m:
                    continue
                ng = g + cost; s = (nb, cur)
                if ng < best.get(s, math.inf):
                    best[s] = ng; heapq.heappush(openq, (ng + h(nb), ng, s, path + [nb]))
        return False, [], math.inf
