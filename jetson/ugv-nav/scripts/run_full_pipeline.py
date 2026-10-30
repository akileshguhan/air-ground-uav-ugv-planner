#!/usr/bin/env python3
"""
run_full_pipeline.py -- one complete UGV planning cycle, from D455 frames on
disk to the path message sent back to the drone.

    D455 frames (PLY + poses.json)
      -> read -> drop invalid -> camera-to-world -> merge      (ingest)
      -> heightmap downsample                                   (noise + density)
      -> candidate graph -> PCA features -> capability gating -> A*
      -> PathMsg  (what the drone receives)

Every stage is timed, including ingest. Outputs, under --out:

  path_msg.bin / path_msg.json   the drone-bound result (binary is verified by
                                 decoding it back before writing)
  01_input_cloud.pcd             the cloud the UGV pipeline actually receives
                                 (after ingest, before feature extraction)
  02_output_feasibility.ply      points coloured by gating outcome
  03_output_roughness.ply        per-point PCA curvature (edges light up)
  04_path_waypoints.ply          candidate waypoints, selected route in green
  05_path_line.obj               selected route as a polyline
  06_output_slope.ply            per-point PCA slope (the ramp shows up)
  summary.json                   timings, counts, ground-truth checks

Usage:
    python3 scripts/run_full_pipeline.py --frames data/compare/noisy_2.5
    python3 scripts/run_full_pipeline.py --frames <real capture dir> --no-truth
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from ugvnav.compat import describe_environment                          # noqa: E402
from ugvnav.d455_ingest import (read_ply_xyz, load_poses, drop_invalid,
                                camera_to_world, heightmap_downsample)   # noqa: E402
from ugvnav.fusion import write_pcd_binary                               # noqa: E402
from ugvnav.features import PointCloudFeatureExtractor                   # noqa: E402
from ugvnav.segment_builder import SegmentBuilder                        # noqa: E402
from ugvnav.capability_model import UGVCapabilityModel                   # noqa: E402
from ugvnav.evaluator import UGVSideEvaluator                            # noqa: E402
from ugvnav.planner import HorizonPlanner                                # noqa: E402
from ugvnav.path_msg import PathMsg                                      # noqa: E402

GREEN = (40, 190, 60)
RED = (210, 40, 40)
GREY = (150, 150, 150)


# --------------------------------------------------------------------------
# Export helpers
# --------------------------------------------------------------------------

def write_ply_rgb(path, pts, rgb):
    """Binary PLY with colour -- CloudCompare shows colour by default."""
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.empty(pts.shape[0], dtype=dt)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex {}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n").format(pts.shape[0])
    with open(path, "wb") as f:
        f.write(hdr.encode("ascii"))
        f.write(arr.tobytes())


def heat(v, lo, hi):
    """blue (lo) -> green -> red (hi)."""
    t = np.clip((np.asarray(v, float) - lo) / max(hi - lo, 1e-12), 0, 1)
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return np.stack([r, g, b], axis=1) * 255


def write_obj_polyline(path, pts):
    with open(path, "w") as f:
        f.write("o selected_path\n")
        for p in pts:
            f.write("v {:.5f} {:.5f} {:.5f}\n".format(*p))
        f.write("l " + " ".join(str(i + 1) for i in range(len(pts))) + "\n")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/compare/noisy_2.5",
                    help="directory with frame_*.ply and poses.json")
    ap.add_argument("--out", default="data/full_run")
    ap.add_argument("--cell", type=float, default=0.05, help="heightmap cell (m)")
    ap.add_argument("--reducer", choices=("mean", "median"), default="mean",
                    help="mean: O(N), default. median: robust to flying pixels, slower")
    ap.add_argument("--no-truth", action="store_true",
                    help="skip ground-truth scoring (use for real captures)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(describe_environment())
    print("frames: {}".format(a.frames))
    print()

    T = {}

    # ---- ingest -------------------------------------------------------
    t0 = time.perf_counter()
    poses = load_poses(os.path.join(a.frames, "poses.json"))
    raw = [(read_ply_xyz(os.path.join(a.frames, n)), p) for n, p in sorted(poses.items())]
    T["1 read frames"] = time.perf_counter() - t0
    n_raw = sum(r[0].shape[0] for r in raw)

    t0 = time.perf_counter()
    merged = np.concatenate([camera_to_world(drop_invalid(pts), p["position"], p["quaternion"])
                             for pts, p in raw], axis=0)
    T["2 filter + transform + merge"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cloud = heightmap_downsample(merged, a.cell, reducer=a.reducer)
    T["3 heightmap downsample"] = time.perf_counter() - t0

    # ---- pipeline -----------------------------------------------------
    fx = PointCloudFeatureExtractor(k_neighbors=12, n_sample=60)
    sb = SegmentBuilder(fx, 2.0, 1.2, 3, 0.5, 30)

    t0 = time.perf_counter()
    wps, edges, ep, _ = sb.build_graph(cloud)
    T["4 candidate graph"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    msgs = sb.build_segment_msgs(wps, edges, ep)
    T["5 feature extraction"] = time.perf_counter() - t0

    model = UGVCapabilityModel()
    t0 = time.perf_counter()
    results = UGVSideEvaluator(model).evaluate_all(msgs)
    T["6 capability gating"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    adj = {}
    res_by_edge = dict(zip(edges, results))
    for (x, y), r in res_by_edge.items():
        if r.feasible:
            adj.setdefault(x, []).append(
                (y, float(np.linalg.norm(wps[y] - wps[x])) / max(r.recommended_speed_factor, 0.1)))
    last = max(int(k.split("_")[0][1:]) for k in wps)
    planner = HorizonPlanner(model.min_turn_radius_m)
    best, best_cost = None, math.inf
    for s in sorted(k for k in wps if k.startswith("S0_")):
        for g in sorted(k for k in wps if k.startswith("S{}_".format(last))):
            ok, p, c = planner.find_path(wps, adj, s, g)
            if ok and c < best_cost:
                best, best_cost = p, c
    T["7 A* search"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if best:
        msg = PathMsg(path_id=int(time.time()), created_s=time.time(),
                      waypoints=[tuple(float(v) for v in wps[w]) for w in best],
                      speed_factors=[float(res_by_edge[(best[i], best[i + 1])].recommended_speed_factor)
                                     for i in range(len(best) - 1)],
                      cost=float(best_cost))
        blob = msg.pack()
        back = PathMsg.unpack(blob)
        assert back.waypoints == msg.waypoints, "path message failed to round-trip"
    T["8 pack path message"] = time.perf_counter() - t0

    # ---- report -------------------------------------------------------
    ingest = T["1 read frames"] + T["2 filter + transform + merge"] + T["3 heightmap downsample"]
    plan = sum(v for k, v in T.items() if k[0] in "45678")
    print("=== timing ===")
    for k, v in T.items():
        print("  {:<30} {:>7.3f} s".format(k, v))
    print("  {:<30} {:>7.3f} s".format("ingest subtotal", ingest))
    print("  {:<30} {:>7.3f} s".format("planning subtotal", plan))
    print("  {:<30} {:>7.3f} s".format("FULL CYCLE", ingest + plan))
    print()
    print("=== data ===")
    print("  frames {}   raw rows {}   valid merged {}   into pipeline {}".format(
        len(raw), n_raw, merged.shape[0], cloud.shape[0]))
    print("  segments {}   feasible {}   rejection reasons {}".format(
        len(edges), sum(r.feasible for r in results),
        dict(Counter(f for r in results for f in r.failed_checks))))
    if best:
        print("  path: {} waypoints, cost {:.2f}, speed factors {:.2f}-{:.2f}, message {} bytes".format(
            len(best), best_cost, min(msg.speed_factors), max(msg.speed_factors), len(blob)))
    else:
        print("  NO PATH FOUND")
    print()

    summary = {"timing_s": T, "full_cycle_s": ingest + plan, "frames": len(raw),
               "raw_rows": n_raw, "merged_points": int(merged.shape[0]),
               "pipeline_points": int(cloud.shape[0]), "segments": len(edges),
               "feasible": int(sum(r.feasible for r in results)),
               "path_waypoints": len(best) if best else 0}

    # ---- ground truth (synthetic data only) ---------------------------
    if not a.no_truth:
        from ground_truth_terrain import terrain, GROUND_TRUTH as gt

        def blocked(idx):
            p = cloud[idx]
            return bool(np.any((p[:, 0] > gt["trench_x"]) &
                               (p[:, 0] < gt["trench_x"] + gt["trench_width_m"]) &
                               (p[:, 1] > gt["trench_y_min"])))
        fr = sum(1 for e, r in res_by_edge.items() if not r.feasible and not blocked(sb.edge_indices[e]))
        fa = sum(1 for e, r in res_by_edge.items() if r.feasible and blocked(sb.edge_indices[e]))
        safe = bool(best) and not any(blocked(sb.edge_indices[(best[i], best[i + 1])])
                                      for i in range(len(best) - 1))
        err = float(np.mean(np.abs(cloud[:, 2] - terrain(cloud[:, 0], cloud[:, 1]))))
        print("=== ground truth ===")
        print("  height error {:.4f} m | false rejections {} | false acceptances {} | "
              "path avoids trench: {}".format(err, fr, fa, safe))
        print()
        summary.update({"height_err_m": err, "false_rej": fr, "false_acc": fa,
                        "path_avoids_trench": safe})

    # ---- outputs -------------------------------------------------------
    if best:
        with open(os.path.join(a.out, "path_msg.bin"), "wb") as f:
            f.write(blob)
        with open(os.path.join(a.out, "path_msg.json"), "w") as f:
            f.write(msg.to_json())

    write_pcd_binary(os.path.join(a.out, "01_input_cloud.pcd"), cloud)

    # feasibility: corridors overlap, so a point can lie in several segments.
    # Green = at least one FEASIBLE segment covers it (a traversable way through
    # exists); red = covered only by rejected segments; grey = no corridor.
    # Painting rejected on top instead would hide feasible lanes under rejected
    # diagonals and make the route appear to cross rejected ground.
    col = np.tile(np.array(GREY, float), (cloud.shape[0], 1))
    for e, r in res_by_edge.items():
        if not r.feasible:
            col[sb.edge_indices[e]] = RED
    for e, r in res_by_edge.items():
        if r.feasible:
            col[sb.edge_indices[e]] = GREEN
    write_ply_rgb(os.path.join(a.out, "02_output_feasibility.ply"), cloud, col)

    # per-point PCA over the whole cloud -- the feature-extraction output itself
    normals, curv = fx._normals_and_curvature(cloud, np.arange(cloud.shape[0]))
    slope = np.degrees(np.arccos(np.clip(normals[:, 2], -1, 1)))
    write_ply_rgb(os.path.join(a.out, "03_output_roughness.ply"), cloud, heat(curv, 0.0, 0.05))
    write_ply_rgb(os.path.join(a.out, "06_output_slope.ply"), cloud, heat(slope, 0.0, 30.0))

    ids = list(wps.keys())
    on = set(best or [])
    write_ply_rgb(os.path.join(a.out, "04_path_waypoints.ply"),
                  np.array([wps[i] for i in ids]),
                  np.array([GREEN if i in on else GREY for i in ids], float))
    if best:
        write_obj_polyline(os.path.join(a.out, "05_path_line.obj"), [wps[w] for w in best])

    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print("outputs in {}/".format(a.out))
    print("CloudCompare order: 01 (input) -> 03 roughness, 06 slope (features) -> "
          "02 feasibility (gating) -> 04 + 05 (A* route)")


if __name__ == "__main__":
    main()
