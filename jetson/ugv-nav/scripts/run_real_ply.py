#!/usr/bin/env python3
"""
run_real_ply.py -- run the UGV pipeline on a single, pre-merged world-frame
point cloud (e.g. data/real/terrain_map.ply) and compare against the
independent reference produced for it (expected_values.json).

Unlike D455 input there are no frames or poses: the cloud is already in the
world frame, so ingest is replaced by
    load -> drop overhead points -> crop -> heightmap
and the rest of the pipeline is unchanged.

Two tests:
  A. landmarks   each reference region is cropped exactly as the reference
                 did (axis-aligned square), run through the PCA feature
                 extractor, and compared with the reference's plane-fit and
                 secant slopes. Different methods, so agreement is evidence.
  B. corridor    the reference's 30 m x 6 m flat corridor is run through the
                 full pipeline (graph -> features -> gating -> A*) and must
                 produce a path end to end.

Usage:
    python3 scripts/run_real_ply.py --ply data/real/terrain_map.ply \\
        --reference data/real/expected_values.json
    python3 scripts/run_real_ply.py ... --cells 0.25,0.5     # compare cell sizes
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

from ugvnav.compat import describe_environment                     # noqa: E402
from ugvnav.d455_ingest import read_ply_xyz, heightmap_downsample   # noqa: E402
from ugvnav.fusion import write_pcd_binary                          # noqa: E402
from ugvnav.features import PointCloudFeatureExtractor              # noqa: E402
from ugvnav.segment_builder import SegmentBuilder                   # noqa: E402
from ugvnav.capability_model import UGVCapabilityModel              # noqa: E402
from ugvnav.evaluator import UGVSideEvaluator                       # noqa: E402
from ugvnav.planner import HorizonPlanner                           # noqa: E402
from ugvnav.protocol import CandidateSegmentMsg                     # noqa: E402
from run_full_pipeline import write_ply_rgb, write_obj_polyline, GREEN, RED, GREY  # noqa: E402


# --------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------

def drop_overhead(points, column_m=0.5, max_above_ground_m=1.5):
    """Remove points far above the lowest point in their column.

    The heightmap assumes ONE ground height per cell. Anything overhead -- the
    reference found 25 plates at z ~ 20 m in this file -- would be averaged
    into the ground and invent a ~10 m bump.

    Column size matters. The steepest real face here is 61.6 deg (tan = 1.85).
    A 1 m column would span 1.85 m of genuine ground rise and cut the top off
    steep faces. A 0.5 m column spans at most its diagonal, 0.71 m x 1.85 =
    1.31 m, under the 1.5 m threshold. The plates are 0.73 m across and the
    reference found ground under all of them, so every column beneath a plate
    still has ground points to measure from. Returns (kept_points, n_removed).
    """
    ij = np.floor(points[:, :2] / column_m).astype(np.int64)
    ij -= ij.min(axis=0)
    key = ij[:, 0] * (int(ij[:, 1].max()) + 1) + ij[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    col_min = np.minimum.reduceat(points[order, 2], starts)
    group = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, ks.size]))
    per_point_min = np.empty(points.shape[0])
    per_point_min[order] = col_min[group]
    keep = points[:, 2] <= per_point_min + max_above_ground_m
    return points[keep], int((~keep).sum())


MIN_PTS = 4   # per heightmap cell; the reference recommends >= 4


def crop_square(points, cx, cy, sx, sy):
    m = (np.abs(points[:, 0] - cx) <= sx / 2.0) & (np.abs(points[:, 1] - cy) <= sy / 2.0)
    return points[m]


def crop_corridor(points, start, u, length, width, margin):
    lat = np.array([-u[1], u[0]])
    rel = points[:, :2] - start
    a, s = rel @ u, rel @ lat
    m = (a >= -margin) & (a <= length + margin) & (np.abs(s) <= width / 2.0 + margin)
    return points[m]


def scatter_diagnostic(raw, hm):
    """Is the scatter random (averaging removes it) or real bumps (it won't)?

    Fits a plane to the raw points and to the heightmap cells. If scatter is
    random, the cell RMS should be close to raw RMS / sqrt(points per cell).
    """
    def plane_rms(p):
        A = np.c_[p[:, 0] - p[:, 0].mean(), p[:, 1] - p[:, 1].mean(), np.ones(len(p))]
        coef, *_ = np.linalg.lstsq(A, p[:, 2], rcond=None)
        return float(np.sqrt(np.mean((p[:, 2] - A @ coef) ** 2)))
    raw_rms, cell_rms = plane_rms(raw), plane_rms(hm)
    per_cell = raw.shape[0] / max(hm.shape[0], 1)
    expected = raw_rms / math.sqrt(per_cell)
    return raw_rms, cell_rms, expected


# --------------------------------------------------------------------------
# Pipeline configuration scaled to the heightmap cell
# --------------------------------------------------------------------------

def make_extractor(cell):
    """The step patch must hold enough cells to fit a plane; its default
    radius (0.2 m) was set for 5 cm cells. 2.5 cells keeps ~20 cells in a
    patch at any cell size."""
    # step_trim_percent=5 and gap_mode="holes": needed for sparse, noisy
    # clouds like this one (see features.py); the D455 path keeps its
    # verified defaults (1 % trim, spacing gaps).
    return PointCloudFeatureExtractor(
        k_neighbors=12, n_sample=60,
        step_window_m=max(0.2, 2.5 * cell), step_min_points=8,
        step_trim_percent=5.0, gap_mode="holes")


def make_builder(fx, cell):
    # A 3 m x 1 m lane corridor holds 3 / cell^2 cells: 48 at 0.25 m, 12 at 0.5 m.
    # PCA needs k+1 = 13, so require at least that many.
    return SegmentBuilder(fx, station_spacing_m=2.0, lane_spacing_m=1.2, n_lanes=3,
                          corridor_radius_m=0.5, min_points_per_edge=13)


# --------------------------------------------------------------------------

def test_landmarks(points, ref, cell, model):
    fx = make_extractor(cell)
    ev = UGVSideEvaluator(model)
    print("\n  {:<16} {:>6} {:>7} {:>7} | {:>7} {:>7} {:>6} | {:>6} {:>7} {:>6} | {}".format(
        "region", "cells", "ref A", "ref B", "PCA mn", "PCA mx", "d(mn)",
        "step", "cellRMS", "rawRMS", "verdict"))
    rows = []
    for L in ref["landmarks"]:
        cx, cy = L["centre_xy_m"]
        sx, sy = L["size_xy_m"]
        raw = crop_square(points, cx, cy, sx, sy)
        hm = heightmap_downsample(raw, cell, min_points_per_cell=MIN_PTS, reducer="mean")
        # measure along the reference's uphill direction so incline/decline
        # split the same way; the magnitude does not depend on it
        d = math.radians(L["slope_method_b_secant"]["direction_deg"])
        u = np.array([math.cos(d), math.sin(d)])
        half = 0.5 * min(sx, sy)
        z0 = float(np.median(hm[:, 2]))
        start = np.array([cx - u[0] * half, cy - u[1] * half, z0])
        end = np.array([cx + u[0] * half, cy + u[1] * half, z0])
        f = fx.extract(L["name"], hm, start, end)
        ok, score, speed, failed = model.evaluate(f)
        pca_mean = max(f.mean_incline_deg, f.mean_decline_deg)
        pca_max = max(f.max_incline_deg, f.max_decline_deg)
        ref_a = L["slope_method_a_plane_fit"]["slope_deg"]
        ref_b = L["slope_method_b_secant"]["slope_deg"]
        raw_rms, cell_rms, expected = scatter_diagnostic(raw, hm)
        verdict = "feasible ({:.2f})".format(score) if ok else "REJECT: " + ",".join(failed)
        print("  {:<16} {:>6} {:>7.2f} {:>7.2f} | {:>7.2f} {:>7.2f} {:>+6.2f} | {:>6.3f} {:>7.3f} {:>6.3f} | {}".format(
            L["name"], hm.shape[0], ref_a, ref_b, pca_mean, pca_max, pca_mean - ref_a,
            f.max_step_height_m, cell_rms, raw_rms, verdict))
        rows.append({"name": L["name"], "kind": L["kind"], "cells": int(hm.shape[0]),
                     "ref_plane_fit_deg": ref_a, "ref_secant_deg": ref_b,
                     "pca_mean_slope_deg": pca_mean, "pca_max_slope_deg": pca_max,
                     "step_m": f.max_step_height_m, "roughness": f.roughness,
                     "raw_rms_m": raw_rms, "cell_rms_m": cell_rms,
                     "cell_rms_if_random_m": expected,
                     "feasible": ok, "failed": failed})
    return rows


def test_corridor(points, ref, cell, model, out_dir):
    C = ref["corridor"]
    start = np.asarray(C["start_xy_m"], float)
    u = np.asarray(C["direction_unit_xy"], float)
    L, W = C["length_m"], C["width_m"]
    T = {}

    t0 = time.perf_counter()
    raw = crop_corridor(points, start, u, L, W, margin=1.0)
    cloud = heightmap_downsample(raw, cell, min_points_per_cell=MIN_PTS, reducer="mean")
    T["crop+heightmap"] = time.perf_counter() - t0

    fx = make_extractor(cell)
    sb = make_builder(fx, cell)
    t0 = time.perf_counter()
    wps, edges, ep, _ = sb.build_graph(cloud, axis=u, origin=start)
    T["graph"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    msgs = sb.build_segment_msgs(wps, edges, ep)
    T["features"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = UGVSideEvaluator(model).evaluate_all(msgs)
    T["gating"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    by = dict(zip(edges, res))
    adj = {}
    for (a, b), r in by.items():
        if r.feasible:
            adj.setdefault(a, []).append((b, float(np.linalg.norm(wps[b] - wps[a])) /
                                          max(r.recommended_speed_factor, 0.1)))
    n_st = max(int(k.split("_")[0][1:]) for k in wps) + 1
    s0 = min((k for k in wps if k.startswith("S0_")),
             key=lambda k: float(np.linalg.norm(wps[k][:2] - start)))
    planner = HorizonPlanner(model.min_turn_radius_m)
    best, cost = None, math.inf
    for st in range(n_st - 1, 0, -1):
        for g in sorted(k for k in wps if k.startswith("S{}_".format(st))):
            ok, p, c = planner.find_path(wps, adj, s0, g)
            if ok and c < cost:
                best, cost = p, c
        if best:
            reached = st
            break
    T["astar"] = time.perf_counter() - t0

    steps = [m.features.max_step_height_m for m in msgs]
    slopes = [max(m.features.max_incline_deg, m.features.max_decline_deg) for m in msgs]
    print("\n  raw points {}  ->  heightmap cells {}  ({:.1f} pts/cell)".format(
        raw.shape[0], cloud.shape[0], raw.shape[0] / max(cloud.shape[0], 1)))
    print("  segments {}   feasible {}   rejection reasons {}".format(
        len(edges), sum(r.feasible for r in res), dict(Counter(f for r in res for f in r.failed_checks))))
    print("  per-segment max slope: median {:.2f} deg, max {:.2f} deg   (reference: worst 2 m slice {:.2f} deg)".format(
        float(np.median(slopes)), float(np.max(slopes)), C["max_slice_slope_plane_fit_deg"]))
    print("  per-segment step:      median {:.3f} m,   max {:.3f} m     (limits: nominal 0.10, hard 0.15)".format(
        float(np.median(steps)), float(np.max(steps))))
    if best:
        dist = float(np.linalg.norm(wps[best[-1]][:2] - start))
        print("  PATH: {} waypoints, reaches {:.1f} m of {:.0f} m, cost {:.2f}".format(
            len(best), dist, L, cost))
    else:
        print("  NO PATH through the corridor")
    print("  timing: " + "  ".join("{} {:.3f}s".format(k, v) for k, v in T.items()))

    # CloudCompare exports
    os.makedirs(out_dir, exist_ok=True)
    write_pcd_binary(os.path.join(out_dir, "01_input_cloud.pcd"), cloud)
    col = np.tile(np.array(GREY, float), (cloud.shape[0], 1))
    for e, r in by.items():
        if not r.feasible:
            col[sb.edge_indices[e]] = RED
    for e, r in by.items():
        if r.feasible:
            col[sb.edge_indices[e]] = GREEN
    write_ply_rgb(os.path.join(out_dir, "02_output_feasibility.ply"), cloud, col)
    if best:
        write_obj_polyline(os.path.join(out_dir, "05_path_line.obj"), [wps[w] for w in best])
    return {"cells": int(cloud.shape[0]), "segments": len(edges),
            "feasible": int(sum(r.feasible for r in res)),
            "reasons": dict(Counter(f for r in res for f in r.failed_checks)),
            "step_median_m": float(np.median(steps)), "step_max_m": float(np.max(steps)),
            "path_waypoints": len(best) if best else 0, "timing_s": T}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default="data/real/terrain_map.ply")
    ap.add_argument("--reference", default="data/real/expected_values.json")
    ap.add_argument("--cells", default="0.3,0.5", help="heightmap cell size(s), comma separated")
    ap.add_argument("--out", default="data/real_run")
    a = ap.parse_args()
    print(describe_environment())

    with open(a.reference) as f:
        ref = json.load(f)

    t0 = time.perf_counter()
    points = read_ply_xyz(a.ply)
    t_load = time.perf_counter() - t0
    inv = ref["inventory"]
    print("\nloaded {} points in {:.1f} s  (reference: {})".format(
        points.shape[0], t_load, inv["n_points"]))
    same = (points.shape[0] == inv["n_points"] and
            np.allclose(points.min(axis=0), inv["bbox_min_xyz"], atol=1e-3) and
            np.allclose(points.max(axis=0), inv["bbox_max_xyz"], atol=1e-3))
    print("  count and bounding box match the reference: {}".format(same))

    t0 = time.perf_counter()
    points, removed = drop_overhead(points)
    ov = ref["overhead_features_not_terrain"]
    print("  overhead filter removed {} points in {:.1f} s  (reference found {} in {} plates)".format(
        removed, time.perf_counter() - t0, ov["total_points"], ov["count"]))

    model = UGVCapabilityModel()   # PLACEHOLDER limits -- see config
    report = {"load_s": t_load, "matches_reference": bool(same), "overhead_removed": removed,
              "runs": {}}
    for cell in [float(c) for c in a.cells.split(",")]:
        print("\n" + "=" * 100)
        print("HEIGHTMAP CELL {:.2f} m   (step patch radius {:.2f} m)".format(cell, max(0.2, 2.5 * cell)))
        print("=" * 100)
        print("A. landmarks  (slopes in deg; d(mn) = PCA mean minus reference plane fit)")
        lm = test_landmarks(points, ref, cell, model)
        print("\n  cellRMS vs rawRMS: if scatter were random, cellRMS would be about "
              "rawRMS / sqrt(points per cell).")
        print("B. corridor  (reference: flat, whole-corridor slope {:.2f} deg)".format(
            ref["corridor"]["whole_corridor_slope_plane_fit_deg"]))
        cor = test_corridor(points, ref, cell, model,
                            os.path.join(a.out, "cell_{:.2f}".format(cell)))
        report["runs"]["{:.2f}".format(cell)] = {"landmarks": lm, "corridor": cor}

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "real_ply_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("\nreport: {}/real_ply_report.json   CloudCompare files: {}/cell_*/".format(a.out, a.out))


if __name__ == "__main__":
    main()
