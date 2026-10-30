#!/usr/bin/env python3
"""
plan_from_ply.py -- one planning cycle on the Jetson for ANY world-frame .ply:

    load -> drop overhead points -> crop a corridor -> heightmap
         -> graph -> PCA features -> capability gating -> A*  -> path JSON

The path JSON is what the drone flies (px4_waypoint_mission.py reads
"waypoints"). Waypoints are written in the CORRIDOR frame:
    x = metres along --heading from --start,  y = metres to the left,
so with the mission's default "body-start" frame the drone flies the route
with its nose pointing down the corridor at takeoff.
World-frame coordinates are kept in "waypoints_world".

"source_sha256" is the hash of the input .ply, so the drone can tell that a
path answers the file IT sent (and not an older path left on disk).

The corridor:
  --start X,Y --heading DEG   explicit (recommended for a known map), or
  omitted                     auto: through the cloud's centre, along its
                              longest horizontal direction.
For terrain_map.ply the known-flat test corridor is
  --start 55,20 --heading 45 --length 30

Exit code 0 = path written, 3 = no path, 2 = bad input.
Python 3.8 compatible.

USAGE
  python3 scripts/plan_from_ply.py --ply cloud.ply --out planned_path.json
"""
import argparse
import hashlib
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

from ugvnav.d455_ingest import read_ply_xyz, heightmap_downsample      # noqa: E402
from ugvnav.capability_model import UGVCapabilityModel                 # noqa: E402
from ugvnav.evaluator import UGVSideEvaluator                          # noqa: E402
from ugvnav.planner import HorizonPlanner                              # noqa: E402
from run_real_ply import (drop_overhead, crop_corridor, make_extractor,  # noqa: E402
                          make_builder, MIN_PTS)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def auto_corridor(points, length):
    """Centre of the cloud, longest horizontal direction (PCA on x, y)."""
    xy = points[:: max(1, points.shape[0] // 200000), :2]
    centre = xy.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((xy - centre).T))
    u = v[:, int(np.argmax(w))]
    if u[0] < 0 or (abs(u[0]) < 1e-9 and u[1] < 0):
        u = -u                                  # deterministic sign
    along = (xy - centre) @ u
    span = float(np.percentile(along, 98) - np.percentile(along, 2))
    length = min(length, max(span - 2.0, 4.0))
    start = centre - u * length / 2.0
    return start, u, length


def plan(points, start, u, length, width, cell, log):
    T = {}
    t0 = time.perf_counter()
    raw = crop_corridor(points, start, u, length, width, margin=1.0)
    if raw.shape[0] < 1000:
        raise ValueError("only %d points inside the corridor -- check --start/--heading" % raw.shape[0])
    cloud = heightmap_downsample(raw, cell, min_points_per_cell=MIN_PTS, reducer="mean")
    T["heightmap"] = time.perf_counter() - t0
    log("corridor: %d raw points -> %d heightmap cells (%.2f m)" % (raw.shape[0], cloud.shape[0], cell))

    fx = make_extractor(cell)
    sb = make_builder(fx, cell)
    model = UGVCapabilityModel()            # PLACEHOLDER limits -- see config
    t0 = time.perf_counter()
    wps, edges, ep, _ = sb.build_graph(cloud, axis=u, origin=start)
    msgs = sb.build_segment_msgs(wps, edges, ep)
    res = UGVSideEvaluator(model).evaluate_all(msgs)
    T["graph+features+gating"] = time.perf_counter() - t0
    log("segments %d, feasible %d, rejection reasons %s" % (
        len(edges), sum(r.feasible for r in res),
        dict(Counter(f for r in res for f in r.failed_checks))))

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
    for st in range(n_st - 1, 0, -1):          # farthest reachable station first
        for g in sorted(k for k in wps if k.startswith("S{}_".format(st))):
            ok, p, c = planner.find_path(wps, adj, s0, g)
            if ok and c < cost:
                best, cost = p, c
        if best:
            break
    T["astar"] = time.perf_counter() - t0
    if not best:
        return None, T
    speeds = [float(by[(a, b)].recommended_speed_factor) for a, b in zip(best[:-1], best[1:])]
    return {"names": best, "world": [wps[w] for w in best], "speeds": speeds, "cost": cost}, T


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plan a UGV path through a .ply and write path JSON.")
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", default="planned_path.json")
    ap.add_argument("--start", help="corridor start X,Y in the cloud's frame (default: auto)")
    ap.add_argument("--heading", type=float, help="corridor direction, deg from +x toward +y (default: auto)")
    ap.add_argument("--length", type=float, default=30.0, help="corridor length, m")
    ap.add_argument("--width", type=float, default=6.0, help="corridor width, m")
    ap.add_argument("--cell", type=float, default=0.5, help="heightmap cell, m (0.5 suits terrain_map.ply)")
    ap.add_argument("--no-overhead-filter", action="store_true")
    a = ap.parse_args(argv)

    t_all = time.perf_counter()

    def log(msg):
        print("[plan %6.1fs] %s" % (time.perf_counter() - t_all, msg), flush=True)

    if not os.path.exists(a.ply):
        print("no such file: %s" % a.ply)
        return 2
    sha = file_sha256(a.ply)
    try:
        points = read_ply_xyz(a.ply)
    except Exception as e:
        print("cannot read %s: %s" % (a.ply, e))
        return 2
    points = points[np.isfinite(points).all(axis=1)]
    log("loaded %d points from %s" % (points.shape[0], os.path.basename(a.ply)))
    if not a.no_overhead_filter:
        points, removed = drop_overhead(points)
        log("overhead filter removed %d points" % removed)

    if a.start and a.heading is not None:
        start = np.array([float(v) for v in a.start.split(",")], float)
        h = math.radians(a.heading)
        u = np.array([math.cos(h), math.sin(h)])
        length = a.length
        mode = "explicit"
    else:
        start, u, length = auto_corridor(points, a.length)
        mode = "auto"
    log("corridor (%s): start (%.2f, %.2f), heading %.1f deg, %.1f m x %.1f m" % (
        mode, start[0], start[1], math.degrees(math.atan2(u[1], u[0])), length, a.width))

    try:
        result, T = plan(points, start, u, length, a.width, a.cell, log)
    except ValueError as e:
        print("planning failed: %s" % e)
        return 3
    log("timing: " + "  ".join("%s %.2fs" % kv for kv in T.items()))
    if result is None:
        print("NO PATH through the corridor -- every route was rejected")
        return 3

    lat = np.array([-u[1], u[0]])
    z0 = float(result["world"][0][2])
    local = []
    for p in result["world"]:
        rel = np.asarray(p[:2]) - start
        local.append([round(float(rel @ u), 3), round(float(rel @ lat), 3), round(float(p[2]) - z0, 3)])
    reach = local[-1][0]

    prev_id = 0
    if os.path.exists(a.out):
        try:
            with open(a.out) as f:
                prev_id = int(json.load(f).get("path_id", 0))
        except (ValueError, TypeError):
            prev_id = 0
    out = {
        "path_id": prev_id + 1,
        "created_s": time.time(),
        "source": os.path.basename(a.ply),
        "source_sha256": sha,
        "frame": "corridor: x = m along heading from start, y = m to the left, z = ground height rel. to first waypoint",
        "corridor": {"start_xy": [float(start[0]), float(start[1])],
                     "heading_deg": math.degrees(math.atan2(u[1], u[0])),
                     "length_m": length, "width_m": a.width, "cell_m": a.cell, "mode": mode},
        "waypoints": local,
        "waypoints_world": [[float(v) for v in p] for p in result["world"]],
        "speed_factors": result["speeds"],
        "cost": float(result["cost"]),
    }
    tmp = a.out + ".part"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, a.out)       # the link server never sees a half-written file
    log("PATH %d: %d waypoints, reaches %.1f m of %.1f m -> %s" % (
        out["path_id"], len(local), reach, length, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
