"""RPLiDAR-era input: simulated 2D push-broom slices fused into a 3D cloud."""
import math
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ugvnav.protocol import Pose, ScanPoint, ScanSlice          # noqa: E402
from ugvnav.fusion import CloudAccumulator, quat_to_rotation_matrix  # noqa: E402
from ground_truth_terrain import terrain                         # noqa: E402

MOUNT_QUAT = (0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))


def _slice(i, px, py, pz):
    ang = np.radians(np.linspace(0, 360, 360, endpoint=False))
    d = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], 1) @ \
        quat_to_rotation_matrix(*MOUNT_QUAT).T
    ts = np.arange(0.15, 15.0, 0.03)
    P = np.array([px, py, pz])[None, None, :] + ts[None, :, None] * d[:, None, :]
    below = P[:, :, 2] <= terrain(P[:, :, 0], P[:, :, 1])
    hit = below.any(1); first = np.where(hit, below.argmax(1), 0)
    pts = [ScanPoint(float(np.degrees(ang[k])), float(ts[first[k]]), 15)
           for k in range(360) if hit[k]]
    return ScanSlice(i, Pose(i * 0.1, px, py, pz, 0, 0, 0, 1), pts)


def rplidar_cloud():
    acc = CloudAccumulator(max_points=8_000_000)
    for i, px in enumerate(np.arange(2.0, 30.0, 0.10)):
        acc.add_slice(_slice(i, float(px), 0.0, float(terrain(px, 0.0)) + 5.0), MOUNT_QUAT)
    return acc.as_array()
