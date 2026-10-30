#!/usr/bin/env python3
"""
synth_d455.py -- synthetic Intel RealSense D455 captures over the shared
ground-truth terrain, written as the files a real capture would produce.

Output (per run directory):
  frame_000.ply ...   one binary PLY per frame, float32 x y z, in the camera
                      OPTICAL frame (x right, y down, z forward), organized
                      H*W rows with (0,0,0) for pixels that have no depth --
                      the same shape pyrealsense2's exporter gives with
                      colour disabled
  poses.json          camera-to-world pose per frame + intrinsics

Sensor model (D455 datasheet values):
  depth FOV 87 x 58 deg, stereo baseline 95 mm, 640x360 depth mode,
  depth unit 1 mm, subpixel accuracy 0.08 px.
  Depth noise sigma(z) = z^2 * subpixel / (fx * baseline) -- grows with the
  SQUARE of distance, which is why flying altitude matters.
  1 % random pixel dropout as (0,0,0).

Not modelled: flying pixels at depth edges, IR-pattern texture effects,
sunlight washout, motion blur, pose error. Real frames will be worse than
these in exactly the places the step metric is sensitive to (edges).

Usage:
    python3 scripts/synth_d455.py --altitude 2.5 --out data/d455_h2.5
    python3 scripts/synth_d455.py --altitude 2.5 --no-noise --out data/d455_clean
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ugvnav.d455_ingest import write_ply_xyz          # noqa: E402
from ground_truth_terrain import terrain              # noqa: E402

W, H = 640, 360
HFOV_DEG, VFOV_DEG = 87.0, 58.0
BASELINE_M = 0.095
SUBPIXEL = 0.08
DEPTH_UNIT_M = 0.001
DROPOUT = 0.01
MAX_DEPTH_M = 10.0

FX = (W / 2) / math.tan(math.radians(HFOV_DEG / 2))
FY = (H / 2) / math.tan(math.radians(VFOV_DEG / 2))
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0

# Nadir camera. Optical z (lens) -> world -z (down); optical y (image down)
# -> world -x, so the top of the image faces the flight direction (+x);
# optical x = y cross z -> world -y. Proper rotation, det = +1.
R_WC = np.array([[0.0, -1.0, 0.0],
                 [-1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0]])


def rotmat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        return [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


def pixel_rays():
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    return np.stack([(u.ravel() - CX) / FX, (v.ravel() - CY) / FY,
                     np.ones(W * H)], axis=1)          # optical frame, z = 1


def cast_frame(cam_pos, noise, rng):
    """Returns (H*W, 3) optical-frame points with (0,0,0) for invalid pixels."""
    d_c = pixel_rays()
    d_w = d_c @ R_WC.T
    depth = np.zeros(d_c.shape[0])
    ts = np.arange(0.3, MAX_DEPTH_M, 0.05)
    chunk = 20000
    for s in range(0, d_c.shape[0], chunk):
        dw = d_w[s:s + chunk]
        P = cam_pos[None, None, :] + ts[None, :, None] * dw[:, None, :]
        below = P[:, :, 2] <= terrain(P[:, :, 0], P[:, :, 1])
        hit = below.any(axis=1)
        k = below.argmax(axis=1)
        lo = np.where(k > 0, ts[np.maximum(k - 1, 0)], ts[0])
        hi = ts[k]
        for _ in range(12):                          # bisection refine
            mid = 0.5 * (lo + hi)
            pm = cam_pos[None, :] + mid[:, None] * dw
            b = pm[:, 2] <= terrain(pm[:, 0], pm[:, 1])
            hi = np.where(b, mid, hi)
            lo = np.where(b, lo, mid)
        depth[s:s + chunk] = np.where(hit, hi, 0.0)

    valid = depth > 0
    if noise:
        sigma = depth ** 2 * SUBPIXEL / (FX * BASELINE_M)
        depth = depth + rng.normal(0.0, 1.0, depth.shape) * sigma
    depth = np.round(depth / DEPTH_UNIT_M) * DEPTH_UNIT_M
    valid &= depth > 0
    valid &= rng.uniform(size=depth.shape) >= DROPOUT
    pts = d_c * depth[:, None]
    pts[~valid] = 0.0
    return pts


def generate(out_dir, altitude, noise=True, x_start=2.0, x_end=30.0,
             spacing=2.0, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    q = rotmat_to_quat(R_WC)
    poses = {}
    xs = np.arange(x_start, x_end + 1e-9, spacing)
    for i, x in enumerate(xs):
        cam = np.array([x, 0.0, float(terrain(x, 0.0)) + altitude])
        name = "frame_{:03d}.ply".format(i)
        write_ply_xyz(os.path.join(out_dir, name), cast_frame(cam, noise, rng))
        poses[name] = {"position": cam.tolist(), "quaternion": q}
    meta = {
        "sensor": "D455 (synthetic)", "width": W, "height": H,
        "fx": FX, "fy": FY, "cx": CX, "cy": CY,
        "baseline_m": BASELINE_M, "altitude_m": altitude, "noise": noise,
        "frame_convention": "optical: x right, y down, z forward",
        "pose_convention": "camera-to-world, quaternion [qx, qy, qz, qw]",
        "frames": poses,
    }
    with open(os.path.join(out_dir, "poses.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--altitude", type=float, default=2.5)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--out", default="data/d455")
    a = ap.parse_args()
    generate(a.out, a.altitude, noise=not a.no_noise)
    print("wrote", a.out)
