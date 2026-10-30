"""RPLiDAR-era fusion: 2D slices + per-slice pose -> 3D cloud. Kept for the
reference comparison; superseded for height recovery by the D455."""
from __future__ import annotations
import math
from typing import List
import numpy as np
from .protocol import ScanSlice


def quat_to_rotation_matrix(qx, qy, qz, qw) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]])


def slice_to_world_points(sl: ScanSlice, mount_quat) -> np.ndarray:
    if not sl.points:
        return np.zeros((0, 3))
    ang = np.radians(np.array([p.angle_deg for p in sl.points]))
    dist = np.array([p.distance_m for p in sl.points])
    local = np.stack([dist * np.cos(ang), dist * np.sin(ang), np.zeros_like(ang)], axis=1)
    body = local @ quat_to_rotation_matrix(*mount_quat).T
    p = sl.pose
    return body @ quat_to_rotation_matrix(p.qx, p.qy, p.qz, p.qw).T + np.array([p.x, p.y, p.z])


class CloudAccumulator:
    def __init__(self, max_points: int = 4_000_000):
        self._chunks: List[np.ndarray] = []
        self._n = 0
        self._max = int(max_points)

    def add_slice(self, sl, mount_quat):
        pts = slice_to_world_points(sl, mount_quat)
        if pts.shape[0] == 0:
            return
        if self._n + pts.shape[0] > self._max:
            raise MemoryError("CloudAccumulator cap exceeded")
        self._chunks.append(pts); self._n += pts.shape[0]

    def as_array(self):
        return np.concatenate(self._chunks, axis=0) if self._chunks else np.zeros((0, 3))


def write_pcd_binary(path, points):
    n = int(points.shape[0])
    hdr = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\n"
           "SIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {}\nHEIGHT 1\n"
           "VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {}\nDATA binary\n").format(n, n)
    with open(path, "wb") as f:
        f.write(hdr.encode("ascii")); f.write(points.astype(np.float32).tobytes())
