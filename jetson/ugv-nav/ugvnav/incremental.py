"""
incremental.py -- continuous ingest and planning on the UGV.

The batch pipeline loads every frame after the flight and then plans, so all
of ingest (about 2.2 s of a 3.6 s cycle on the board) sits on the critical
path. Here frames are folded into the terrain map AS THEY ARRIVE, while the
drone is still flying; only planning remains when a path is needed.

    socket thread:   on_frame(points, pose)  ~ one frame's worth of work
    planner thread:  plan(ugv_xy)            graph + features + gating + A*

Python 3.8 / numpy 1.17 compatible.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np

from .d455_ingest import camera_to_world, drop_invalid
from .path_msg import PathMsg

_OFF = np.int64(2 ** 31)


def _cell_keys(xy: np.ndarray, cell_m: float) -> np.ndarray:
    """Global, fixed grid: cell boundaries never move as the map grows, so a
    cell's running sums stay valid forever. Packs (ix, iy) into one int64."""
    ij = np.floor(xy / cell_m).astype(np.int64)
    return (ij[:, 0] + _OFF) * np.int64(2 ** 32) + (ij[:, 1] + _OFF)


class IncrementalHeightmap:
    """Running per-cell sums of x, y, z and a count. Equivalent to
    d455_ingest.heightmap_downsample(reducer="mean") over all frames so far,
    but updated one frame at a time. (A running MEDIAN is not possible without
    keeping every point -- this is why mean is the default reducer.)"""

    def __init__(self, cell_m: float = 0.05, min_points_per_cell: int = 2,
                 min_range_m: float = 0.3, max_range_m: float = 6.0):
        self.cell_m = float(cell_m)
        self.min_points = int(min_points_per_cell)
        self.min_range = float(min_range_m)
        self.max_range = float(max_range_m)
        self._keys = np.zeros(0, dtype=np.int64)
        self._sums = np.zeros((0, 4))           # sx, sy, sz, n
        self._lock = threading.Lock()
        self.frames = 0

    def add_frame(self, cam_points: np.ndarray, position, quaternion) -> int:
        pts = camera_to_world(drop_invalid(cam_points, self.min_range, self.max_range),
                              position, quaternion)
        if pts.shape[0] == 0:
            return 0
        # reduce the frame to per-cell sums first (small), then merge
        k = _cell_keys(pts[:, :2], self.cell_m)
        uk, inv = np.unique(k, return_inverse=True)
        fs = np.stack([np.bincount(inv, weights=pts[:, 0]),
                       np.bincount(inv, weights=pts[:, 1]),
                       np.bincount(inv, weights=pts[:, 2]),
                       np.bincount(inv).astype(float)], axis=1)
        with self._lock:
            allk = np.concatenate([self._keys, uk])
            mk, minv = np.unique(allk, return_inverse=True)
            allsum = np.concatenate([self._sums, fs], axis=0)
            merged = np.stack([np.bincount(minv, weights=allsum[:, c]) for c in range(4)], axis=1)
            self._keys, self._sums = mk, merged
            self.frames += 1
        return int(pts.shape[0])

    def snapshot(self) -> np.ndarray:
        with self._lock:
            s = self._sums.copy()
        s = s[s[:, 3] >= self.min_points]
        return s[:, :3] / s[:, 3:4]

    def num_cells(self) -> int:
        return int(self._keys.size)


class UGVRuntime:
    """Continuous ingest + planning from the UGV's current position.

    Planning window: from `back_m` behind the UGV to `horizon_m` ahead, and
    `lateral_m` either side, measured along the UGV -> drone direction. This
    keeps planning cost bounded as the explored map grows, instead of
    re-planning over everything ever seen.
    """

    def __init__(self, segment_builder, evaluator, planner,
                 heightmap: Optional[IncrementalHeightmap] = None,
                 horizon_m: float = 25.0, back_m: float = 1.0, lateral_m: float = 3.0,
                 axis_window_m: float = 6.0):
        self.sb = segment_builder
        self.ev = evaluator
        self.planner = planner
        self.map = heightmap or IncrementalHeightmap()
        self.horizon_m = float(horizon_m)
        self.back_m = float(back_m)
        self.lateral_m = float(lateral_m)
        self._frontier: Optional[np.ndarray] = None   # latest drone position
        self._drone_track = []                         # drone xy history
        self.axis_window_m = float(axis_window_m)
        self._path_id = 0
        self.last_timing = {}

    # ---- called on the receive thread, once per frame -------------------
    def on_frame(self, cam_points, position, quaternion) -> float:
        t0 = time.perf_counter()
        self.map.add_frame(cam_points, position, quaternion)
        self._frontier = np.asarray(position, dtype=float)[:2]
        self._drone_track.append(self._frontier.copy())
        return time.perf_counter() - t0

    def _flight_axis(self, ugv_xy):
        """Direction of the drone's recent travel. Lanes are laid out relative
        to the drone's flight line, NOT re-centred on the UGV: re-centring put
        the UGV's corridor wherever it happened to be mid-manoeuvre (y = -1.02
        in testing), which clipped a trench edge the fixed lanes kept clear of.
        Using UGV -> drone would also tilt the whole graph whenever the UGV
        sits off the flight line."""
        tr = self._drone_track
        latest = tr[-1]
        for past in reversed(tr[:-1]):
            if np.linalg.norm(latest - past) >= self.axis_window_m:
                return (latest - past) / np.linalg.norm(latest - past)
        if len(tr) >= 2 and np.linalg.norm(latest - tr[0]) > 0.5:
            return (latest - tr[0]) / np.linalg.norm(latest - tr[0])
        d = latest - ugv_xy                      # first frames: fall back
        return d / (np.linalg.norm(d) + 1e-12)

    # ---- called by the planner, whenever a path is needed ---------------
    def plan(self, ugv_xy) -> Tuple[Optional[PathMsg], dict]:
        T = {}
        ugv_xy = np.asarray(ugv_xy, dtype=float)[:2]
        if self._frontier is None:
            return None, {"reason": "no frames yet"}
        u = self._flight_axis(ugv_xy)
        if float((self._frontier - ugv_xy) @ u) < 1.0:
            return None, {"reason": "UGV has caught up with the drone"}
        lat = np.array([-u[1], u[0]])
        # origin: the point on the drone's flight line level with the UGV, so
        # lanes sit at fixed offsets from that line and station 0 is at the UGV
        origin = self._frontier + float((ugv_xy - self._frontier) @ u) * u

        t0 = time.perf_counter()
        cloud = self.map.snapshot()
        rel = cloud[:, :2] - origin
        along, side = rel @ u, rel @ lat
        win = (along >= -self.back_m) & (along <= self.horizon_m) & (np.abs(side) <= self.lateral_m)
        cloud = cloud[win]
        T["snapshot+window"] = time.perf_counter() - t0
        if cloud.shape[0] < 100:
            return None, {"reason": "too little terrain ahead", "timing": T}

        t0 = time.perf_counter()
        wps, edges, ep, _ = self.sb.build_graph(cloud, axis=u, origin=origin)
        T["graph"] = time.perf_counter() - t0
        if not edges:
            return None, {"reason": "no candidate segments", "timing": T}

        t0 = time.perf_counter()
        msgs = self.sb.build_segment_msgs(wps, edges, ep)
        T["features"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        results = self.ev.evaluate_all(msgs)
        T["gating"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        res = dict(zip(edges, results))
        adj = {}
        for (a, b), r in res.items():
            if r.feasible:
                adj.setdefault(a, []).append(
                    (b, float(np.linalg.norm(wps[b] - wps[a])) / max(r.recommended_speed_factor, 0.1)))
        n_st = max(int(k.split("_")[0][1:]) for k in wps) + 1
        # start: the station-0 waypoint nearest the UGV
        s0 = [k for k in wps if k.startswith("S0_")]
        start = min(s0, key=lambda k: float(np.linalg.norm(wps[k][:2] - ugv_xy)))
        # goal: reach the FURTHEST station any feasible path can get to
        best, best_cost = None, math.inf
        for st in range(n_st - 1, 0, -1):
            for g in sorted(k for k in wps if k.startswith("S{}_".format(st))):
                ok, p, c = self.planner.find_path(wps, adj, start, g)
                if ok and c < best_cost:
                    best, best_cost = p, c
            if best:
                break
        T["astar"] = time.perf_counter() - t0
        self.last_timing = T

        info = {"timing": T, "segments": len(edges),
                "feasible": int(sum(r.feasible for r in results)),
                "edges": edges, "results": res, "edge_indices": dict(self.sb.edge_indices),
                "cloud": cloud, "waypoints": wps}
        if not best:
            info["reason"] = "no feasible path from UGV position"
            return None, info
        self._path_id += 1
        msg = PathMsg(path_id=self._path_id, created_s=time.time(),
                      waypoints=[tuple(float(v) for v in wps[w]) for w in best],
                      speed_factors=[float(res[(best[i], best[i + 1])].recommended_speed_factor)
                                     for i in range(len(best) - 1)],
                      cost=float(best_cost))
        info["path_nodes"] = best
        return msg, info


class PlannerThread:
    """Runs plan() repeatedly off the receive thread, so incoming frames are
    never blocked by a planning pass (the old receive loop planned on the
    socket thread). get_ugv_xy comes from UGV localisation -- not built yet."""

    def __init__(self, runtime: UGVRuntime, get_ugv_xy: Callable[[], Tuple[float, float]],
                 on_path: Callable[[PathMsg], None], period_s: float = 1.0):
        self.rt, self.get_ugv_xy, self.on_path = runtime, get_ugv_xy, on_path
        self.period_s = float(period_s)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._last_frames = -1

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=5.0)

    def _loop(self):
        while not self._stop.is_set():
            if self.rt.map.frames != self._last_frames:      # only replan on new data
                self._last_frames = self.rt.map.frames
                msg, _ = self.rt.plan(self.get_ugv_xy())
                if msg is not None:
                    self.on_path(msg)
            self._stop.wait(self.period_s)
