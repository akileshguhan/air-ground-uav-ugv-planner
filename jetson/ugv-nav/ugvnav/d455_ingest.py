"""
d455_ingest.py -- turns RealSense D455 point cloud frames into the world-frame
cloud the UGV pipeline expects.

How a D455 frame differs from the old RPLiDAR-fused cloud, and what this module
does about each difference:

  1. Frame convention. D455 points are in the camera's OPTICAL frame
     (x right, y down, z forward along the lens). The pipeline assumes a world
     frame with z up. -> camera_to_world() applies the camera pose.
  2. Invalid pixels. Pixels with no depth are exported as (0, 0, 0).
     -> drop_invalid() removes them, plus anything outside the usable range.
  3. Density. One 640x360 frame is ~230k points; a pass over the survey area is
     millions, ~100x the RPLiDAR cloud. -> heightmap_downsample().
  4. Noise. Stereo depth error grows with distance squared (roughly
     z^2 * subpixel / (focal * baseline)). The pipeline's step metric takes the
     largest elevation jump between neighbouring points, so raw noise reads
     directly as fake steps. -> heightmap_downsample() takes the MEDIAN height
     per grid cell, which suppresses it.
  5. Height is measured, not synthesised. Unlike the RPLiDAR, no per-slice
     motion stitching is needed to get height. Merging several frames still
     needs each frame's pose (merge_frames()).

heightmap_downsample() assumes 2.5D terrain: one ground height per (x, y)
cell. Overhangs and anything under a branch collapse into a single height,
so the overhead-clearance check is not meaningful on its output.

Python 3.8 / numpy 1.17 compatible.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np

from .fusion import quat_to_rotation_matrix


# --------------------------------------------------------------------------
# File I/O
# --------------------------------------------------------------------------

_PLY_TYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
    "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4",
}


def read_ply_xyz(path: str) -> np.ndarray:
    """Read x, y, z from a PLY vertex element (binary little-endian or ASCII).

    Extra per-vertex properties (colour, normals, texture coords, which
    pyrealsense2 can include) are parsed and ignored.
    """
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header.append(line)
            if line == "end_header":
                break
        body_offset = f.tell()

    fmt = next(l.split()[1] for l in header if l.startswith("format"))
    n = 0
    props: List[Tuple[str, str]] = []
    in_vertex = False
    for l in header:
        parts = l.split()
        if not parts:
            continue
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                n = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            if parts[1] == "list":
                raise ValueError("list properties in vertex element not supported")
            props.append((parts[2], _PLY_TYPES[parts[1]]))

    names = [p[0] for p in props]
    if fmt == "binary_little_endian":
        dt = np.dtype([(name, "<" + t) for name, t in props])
        with open(path, "rb") as f:
            f.seek(body_offset)
            data = np.fromfile(f, dtype=dt, count=n)
        return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    if fmt == "ascii":
        with open(path, "rb") as f:
            f.seek(body_offset)
            arr = np.loadtxt(f, max_rows=n, ndmin=2)
        return arr[:, [names.index("x"), names.index("y"), names.index("z")]]
    raise ValueError("unsupported PLY format: {}".format(fmt))


def write_ply_xyz(path: str, points: np.ndarray) -> None:
    """Binary PLY, float32 x y z -- the shape pyrealsense2's exporter produces
    when colour is disabled."""
    pts = np.ascontiguousarray(points, dtype="<f4")
    header = ("ply\nformat binary_little_endian 1.0\n"
              "element vertex {}\nproperty float x\nproperty float y\n"
              "property float z\nend_header\n").format(pts.shape[0])
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts.tobytes())


def load_poses(path: str) -> Dict[str, dict]:
    """poses.json: {"frames": {"frame_000.ply": {"position": [x,y,z],
    "quaternion": [qx,qy,qz,qw]}, ...}} -- camera-to-world, i.e. the optical
    frame's pose in the world frame."""
    with open(path) as f:
        return json.load(f)["frames"]


# --------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------

def drop_invalid(points: np.ndarray, min_range_m: float = 0.3,
                 max_range_m: float = 6.0) -> np.ndarray:
    """Remove (0,0,0) no-depth pixels, non-finite values and out-of-range
    returns. Range is measured along the optical axis (camera z)."""
    finite = np.all(np.isfinite(points), axis=1)
    z = points[:, 2]
    keep = finite & (z > min_range_m) & (z < max_range_m)
    return points[keep]


def camera_to_world(points: np.ndarray, position, quaternion) -> np.ndarray:
    R = quat_to_rotation_matrix(*quaternion)
    return points @ R.T + np.asarray(position, dtype=float)


def merge_frames(frames: List[Tuple[np.ndarray, list, list]],
                 min_range_m: float = 0.3, max_range_m: float = 6.0) -> np.ndarray:
    """frames: list of (camera_frame_points, position, quaternion)."""
    out = [camera_to_world(drop_invalid(p, min_range_m, max_range_m), pos, q)
           for p, pos, q in frames]
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3))


def heightmap_downsample(points: np.ndarray, cell_m: float = 0.05,
                         min_points_per_cell: int = 2,
                         reducer: str = "mean") -> np.ndarray:
    """One point per (x, y) grid cell: mean x, mean y, and mean or median z.

    reducer="mean"   single pass with np.bincount, no sorting -- O(N). For
                     Gaussian stereo noise the mean is also slightly MORE precise
                     than the median. Default.
    reducer="median" sorts every point (O(N log N)); several times slower on
                     millions of points, but robust to outliers such as flying
                     pixels at depth edges. Use if real frames show them.

    Cells with fewer than min_points_per_cell returns are dropped as
    unsupported. See module docstring for the 2.5D assumption.
    """
    if points.shape[0] == 0:
        return points
    ij = np.floor(points[:, :2] / cell_m).astype(np.int64)
    ij -= ij.min(axis=0)
    ny = int(ij[:, 1].max()) + 1
    key = ij[:, 0] * ny + ij[:, 1]
    n_cells = int(key.max()) + 1

    if reducer == "mean":
        if n_cells > 50_000_000:        # grid too sparse for a dense bincount
            _, key = np.unique(key, return_inverse=True)
            n_cells = int(key.max()) + 1
        counts = np.bincount(key, minlength=n_cells)
        sel = counts >= min_points_per_cell
        c = counts[sel]
        mx = np.bincount(key, weights=points[:, 0], minlength=n_cells)[sel] / c
        my = np.bincount(key, weights=points[:, 1], minlength=n_cells)[sel] / c
        mz = np.bincount(key, weights=points[:, 2], minlength=n_cells)[sel] / c
        return np.stack([mx, my, mz], axis=1)

    if reducer != "median":
        raise ValueError("reducer must be 'mean' or 'median'")
    order = np.lexsort((points[:, 2], key))      # by cell, then z within cell
    k = key[order]
    p = points[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    counts = np.diff(np.r_[starts, k.size])
    # Sum over EVERY cell boundary first, then select -- reducing over only the
    # kept boundaries would fold dropped cells into their neighbours.
    sx = np.add.reduceat(p[:, 0], starts)
    sy = np.add.reduceat(p[:, 1], starts)
    sel = counts >= min_points_per_cell
    mx = sx[sel] / counts[sel]
    my = sy[sel] / counts[sel]
    mz = p[starts[sel] + (counts[sel] - 1) // 2, 2]
    return np.stack([mx, my, mz], axis=1)
