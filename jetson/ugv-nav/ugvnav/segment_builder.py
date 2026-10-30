"""Candidate waypoint graph + per-segment PCA description."""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from .features import PointCloudFeatureExtractor
from .protocol import CandidateSegmentMsg


class SegmentBuilder:
    def __init__(self, extractor: PointCloudFeatureExtractor, station_spacing_m=2.0,
                 lane_spacing_m=1.2, n_lanes=3, corridor_radius_m=0.5, min_points_per_edge=30):
        self.fx = extractor
        self.station_spacing = float(station_spacing_m)
        self.lane_spacing = float(lane_spacing_m)
        self.n_lanes = int(n_lanes)
        self.corridor_r = float(corridor_radius_m)
        self.min_points = int(min_points_per_edge)

    def build_graph(self, cloud, axis=None, origin=None):
        """axis:   travel direction (2-vector). If None, taken from PCA of the
                   cloud -- fine for one static cloud, but the eigenvector's
                   sign is arbitrary, so across cycles the graph could flip
                   direction. The runtime passes UGV -> drone instead, which is
                   stable by construction.
        origin:    (x, y) the lanes and stations are laid out from. If None,
                   the cloud centroid. The runtime passes the UGV position, so
                   station 0 / centre lane is where the vehicle actually is.
        Limitation: stations lie along one straight line, so a single graph
        cannot represent a curved route; curves come from replanning each
        cycle as the axis turns."""
        xy = cloud[:, :2]
        c = xy.mean(axis=0) if origin is None else np.asarray(origin, dtype=float)[:2]
        if axis is None:
            w, v = np.linalg.eigh(np.cov((xy - xy.mean(axis=0)).T))
            u = v[:, int(np.argmax(w))]
            if u[0] < 0:          # single-cloud convenience only; see docstring
                u = -u
        else:
            u = np.asarray(axis, dtype=float)[:2]
            u = u / (np.linalg.norm(u) + 1e-12)
        lat = np.array([-u[1], u[0]])
        along = (xy - c) @ u; latp = (xy - c) @ lat
        stations = np.arange(along.min() + 1.0, along.max() - 1.0, self.station_spacing)
        offsets = (np.arange(self.n_lanes) - self.n_lanes // 2) * self.lane_spacing
        wps: Dict[str, np.ndarray] = {}
        for si, s in enumerate(stations):
            for li, l in enumerate(offsets):
                m = (np.abs(along - s) < self.corridor_r) & (np.abs(latp - l) < self.corridor_r)
                z = float(np.median(cloud[m, 2])) if m.any() else float(np.median(cloud[:, 2]))
                p = c + s * u + l * lat
                wps["S{}_L{}".format(si, li)] = np.array([p[0], p[1], z])
        edges: List[Tuple[str, str]] = []; ep: Dict[Tuple[str, str], np.ndarray] = {}
        # Row indices into `cloud` per edge, so visualisation can colour exactly
        # the points each segment was evaluated on (no re-derived corridors).
        self.edge_indices: Dict[Tuple[str, str], np.ndarray] = {}
        for si in range(len(stations) - 1):
            for li in range(self.n_lanes):
                for lj in range(self.n_lanes):
                    if abs(li - lj) > 1:
                        continue
                    a = "S{}_L{}".format(si, li); b = "S{}_L{}".format(si + 1, lj)
                    s0, s1 = stations[si], stations[si + 1]; l0, l1 = offsets[li], offsets[lj]
                    m = ((along >= s0 - self.corridor_r) & (along <= s1 + self.corridor_r) &
                         (latp >= min(l0, l1) - self.corridor_r) & (latp <= max(l0, l1) + self.corridor_r))
                    if int(m.sum()) < self.min_points:
                        continue
                    edges.append((a, b)); ep[(a, b)] = cloud[m]
                    self.edge_indices[(a, b)] = np.flatnonzero(m)
        return wps, edges, ep, u

    def build_segment_msgs(self, wps, edges, ep):
        out = []
        for (a, b) in edges:
            sid = "{}->{}".format(a, b)
            out.append(CandidateSegmentMsg(sid, tuple(float(x) for x in wps[a]),
                                           tuple(float(x) for x in wps[b]),
                                           self.fx.extract(sid, ep[(a, b)], wps[a], wps[b])))
        return out
