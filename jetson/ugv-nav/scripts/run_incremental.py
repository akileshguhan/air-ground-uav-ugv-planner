#!/usr/bin/env python3
"""
run_incremental.py -- simulates the real operating loop.

  drone: flies at 1 m/s, sends one D455 frame every 2 m
  UGV:   starts LAG_M behind; after each frame it
           1. folds the frame into its terrain map      (on_frame)
           2. replans from its CURRENT position         (plan)
           3. drives 2 m along the path it just got

Checks:
  - incremental map == batch heightmap over the same frames (equivalence)
  - every plan: no feasible segment crosses the trench (0 false acceptances)
  - the UGV's actual driven track never enters the trench
  - timing per frame (ingest) and per plan

Usage:
    python3 scripts/run_incremental.py --frames data/compare/noisy_2.5
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from ugvnav.compat import describe_environment                        # noqa: E402
from ugvnav.d455_ingest import (read_ply_xyz, load_poses, drop_invalid,
                                camera_to_world, heightmap_downsample)  # noqa: E402
from ugvnav.features import PointCloudFeatureExtractor                  # noqa: E402
from ugvnav.segment_builder import SegmentBuilder                       # noqa: E402
from ugvnav.capability_model import UGVCapabilityModel                  # noqa: E402
from ugvnav.evaluator import UGVSideEvaluator                           # noqa: E402
from ugvnav.planner import HorizonPlanner                               # noqa: E402
from ugvnav.incremental import IncrementalHeightmap, UGVRuntime         # noqa: E402
from ugvnav.path_msg import PathMsg                                     # noqa: E402
from ugvnav.fusion import write_pcd_binary                              # noqa: E402

LAG_M = 6.0
STEP_M = 2.0      # UGV distance driven between frames (drone frame spacing)


def in_trench(p, gt, margin=0.0):
    return ((p[..., 0] > gt["trench_x"] - margin) &
            (p[..., 0] < gt["trench_x"] + gt["trench_width_m"] + margin) &
            (p[..., 1] > gt["trench_y_min"] - margin))


def advance(pos, ahead, dist):
    """Drive `dist` metres from `pos` through the waypoints still `ahead`.
    Returns (new_position, waypoints still ahead, sampled track). Stops at the
    last waypoint: never drives past the end of a planned path."""
    cur, ahead, track, remaining = np.asarray(pos, float), list(ahead), [], dist
    while ahead and remaining > 1e-9:
        nxt = np.asarray(ahead[0][:2], float)
        seg = nxt - cur
        L = float(np.linalg.norm(seg))
        if L < 1e-9:
            ahead.pop(0)
            continue
        if remaining < L:
            end = cur + seg * (remaining / L)
            track += list(np.linspace(cur, end, 20))
            return end, ahead, track
        track += list(np.linspace(cur, nxt, 20))
        remaining -= L
        cur = nxt
        ahead.pop(0)
    return cur, ahead, track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/compare/noisy_2.5")
    ap.add_argument("--out", default="data/incremental_run")
    ap.add_argument("--no-truth", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(describe_environment())

    poses = load_poses(os.path.join(a.frames, "poses.json"))
    names = sorted(poses)
    frames = [(read_ply_xyz(os.path.join(a.frames, n)), poses[n]) for n in names]

    gt = None
    if not a.no_truth:
        from ground_truth_terrain import GROUND_TRUTH as gt

    # ---- 1. equivalence with the batch heightmap ------------------------
    hm = IncrementalHeightmap(cell_m=0.05)
    for pts, p in frames:
        hm.add_frame(pts, p["position"], p["quaternion"])
    inc = hm.snapshot()
    merged = np.concatenate([camera_to_world(drop_invalid(pts), p["position"], p["quaternion"])
                             for pts, p in frames])
    bat = heightmap_downsample(merged, 0.05, reducer="mean")
    o1 = np.lexsort((inc[:, 1], inc[:, 0])); o2 = np.lexsort((bat[:, 1], bat[:, 0]))
    same = inc.shape == bat.shape and np.allclose(inc[o1], bat[o2], atol=1e-9)
    print("\n[equivalence] incremental {} cells vs batch {} cells -> identical: {}".format(
        inc.shape[0], bat.shape[0], same))

    # ---- 2. live loop ---------------------------------------------------
    fx = PointCloudFeatureExtractor(k_neighbors=12, n_sample=60)
    model = UGVCapabilityModel()
    rt = UGVRuntime(SegmentBuilder(fx, 2.0, 1.2, 3, 0.5, 30), UGVSideEvaluator(model),
                    HorizonPlanner(model.min_turn_radius_m),
                    heightmap=IncrementalHeightmap(cell_m=0.05))

    first = np.asarray(frames[0][1]["position"][:2], float)
    ugv = first.copy()
    track = [ugv.copy()]
    ingest_t, plan_t, rows, violations, false_acc_total = [], [], [], 0, 0
    ahead = []            # committed waypoints still to drive

    print("\n frame  drone_x  ugv(x, y)        ingest   plan   segs  feas  path  note")
    for i, (pts, p) in enumerate(frames):
        ingest_t.append(rt.on_frame(pts, p["position"], p["quaternion"]))
        drone_x = p["position"][0]
        if drone_x - first[0] < LAG_M:                # UGV waits until the drone is ahead
            print(" {:>5}  {:>7.1f}  waiting ({:.1f} m ahead)".format(i, drone_x, drone_x - first[0]))
            continue
        t0 = time.perf_counter()
        msg, info = rt.plan(ugv)
        plan_t.append(time.perf_counter() - t0)

        note = ""
        if gt is not None and "results" in info:
            cloud, idx = info["cloud"], info["edge_indices"]
            fa = sum(1 for e, r in info["results"].items()
                     if r.feasible and in_trench(cloud[idx[e]], gt).any())
            false_acc_total += fa
            if fa:
                note += " FALSE-ACCEPT x{}".format(fa)
        if msg is None:
            note += " " + info.get("reason", "")
        else:
            back = PathMsg.unpack(msg.pack())
            assert back.waypoints == msg.waypoints
            ahead = list(msg.waypoints[1:])      # new plan replaces the remainder
        print(" {:>5}  {:>7.1f}  ({:>5.2f}, {:>5.2f})  {:>6.3f}  {:>5.3f}  {:>4}  {:>4}  {:>4}  {}".format(
            i, drone_x, ugv[0], ugv[1], ingest_t[-1], plan_t[-1], info.get("segments", 0),
            info.get("feasible", 0), len(msg.waypoints) if msg else 0, note.strip()))

        ugv, ahead, seg_track = advance(ugv, ahead, STEP_M)
        track += seg_track

    # the drone has stopped; the UGV keeps replanning until it runs out of map
    while True:
        msg, info = rt.plan(ugv)
        if msg is not None:
            ahead = list(msg.waypoints[1:])
        new, ahead, seg_track = advance(ugv, ahead, STEP_M)
        track += seg_track
        if np.linalg.norm(new - ugv) < 0.05:
            print("  end: stopped at ({:.2f}, {:.2f}) -- {}".format(
                new[0], new[1], info.get("reason", "end of planned path")))
            break
        ugv = new

    track = np.array(track)
    if gt is not None:
        violations = int(in_trench(track, gt).sum())

    print("\n=== summary ===")
    print("  ingest per frame   mean {:.3f} s  max {:.3f} s   (runs while the drone flies)".format(
        np.mean(ingest_t), np.max(ingest_t)))
    print("  plan per cycle     mean {:.3f} s  max {:.3f} s   (the critical path)".format(
        np.mean(plan_t), np.max(plan_t)))
    print("  UGV drove {:.1f} m, from x={:.1f} to x={:.1f}".format(
        float(np.sum(np.linalg.norm(np.diff(track, axis=0), axis=1))), track[0, 0], track[-1, 0]))
    if gt is not None:
        print("  false acceptances over all plans: {}".format(false_acc_total))
        print("  track points inside the trench:   {}".format(violations))

    write_pcd_binary(os.path.join(a.out, "map_final.pcd"), rt.map.snapshot())
    with open(os.path.join(a.out, "ugv_track.obj"), "w") as f:
        f.write("o ugv_track\n")
        cz = rt.map.snapshot()
        from scipy.spatial import cKDTree
        tree = cKDTree(cz[:, :2])
        _, nn = tree.query(track)
        for (x, y), z in zip(track, cz[nn, 2]):
            f.write("v {:.4f} {:.4f} {:.4f}\n".format(x, y, z + 0.05))
        f.write("l " + " ".join(str(i + 1) for i in range(len(track))) + "\n")
    print("\n  outputs: {}/map_final.pcd, ugv_track.obj  (open together in CloudCompare)".format(a.out))
    return 0 if (same and false_acc_total == 0 and violations == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
