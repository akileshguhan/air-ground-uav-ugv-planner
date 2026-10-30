"""Runs the real UGV pipeline on any world-frame cloud and scores it against
the known terrain. Shared by the RPLiDAR and D455 comparisons."""
import math
import os
import sys
import time
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ugvnav.features import PointCloudFeatureExtractor   # noqa: E402
from ugvnav.segment_builder import SegmentBuilder        # noqa: E402
from ugvnav.capability_model import UGVCapabilityModel   # noqa: E402
from ugvnav.evaluator import UGVSideEvaluator            # noqa: E402
from ugvnav.planner import HorizonPlanner                # noqa: E402
from ground_truth_terrain import terrain, GROUND_TRUTH  # noqa: E402


def run(cloud):
    g = GROUND_TRUTH
    t = {}
    fx = PointCloudFeatureExtractor(k_neighbors=12, n_sample=60)
    sb = SegmentBuilder(fx, 2.0, 1.2, 3, 0.5, 30)
    t0 = time.perf_counter(); wps, edges, ep, _ = sb.build_graph(cloud); t["graph"] = time.perf_counter() - t0
    t0 = time.perf_counter(); msgs = sb.build_segment_msgs(wps, edges, ep); t["features"] = time.perf_counter() - t0
    model = UGVCapabilityModel(); res = UGVSideEvaluator(model).evaluate_all(msgs)

    # ground truth per edge: the trench is the ONLY genuinely impassable feature
    # under the placeholder limits (12 deg < 15 nominal, 0.08 m < 0.10 nominal)
    def truly_blocked(p):
        return bool(np.any((p[:, 0] > g["trench_x"]) &
                           (p[:, 0] < g["trench_x"] + g["trench_width_m"]) &
                           (p[:, 1] > g["trench_y_min"])))
    false_rej = sum(1 for e, r in zip(edges, res) if not r.feasible and not truly_blocked(ep[e]))
    false_acc = sum(1 for e, r in zip(edges, res) if r.feasible and truly_blocked(ep[e]))

    adj = {}
    for (a, b), r in zip(edges, res):
        if r.feasible:
            adj.setdefault(a, []).append((b, float(np.linalg.norm(wps[b] - wps[a])) /
                                          max(r.recommended_speed_factor, 0.1)))
    last = max(int(k.split("_")[0][1:]) for k in wps)
    pl = HorizonPlanner(model.min_turn_radius_m)
    best, bc = None, math.inf
    t0 = time.perf_counter()
    for s in sorted(k for k in wps if k.startswith("S0_")):
        for gl in sorted(k for k in wps if k.startswith("S{}_".format(last))):
            ok, p, c = pl.find_path(wps, adj, s, gl)
            if ok and c < bc:
                best, bc = p, c
    t["astar"] = time.perf_counter() - t0

    F = [m.features for m in msgs]
    ramp = [f for f in F if g["ramp_start_x"] < (f.start[0] + f.end[0]) / 2 < g["ramp_end_x"]]
    step = [f for f in F if f.start[0] <= g["step_x"] <= f.end[0]]
    trench = [f for f in F if f.start[0] <= g["trench_x"] <= f.end[0]]
    return {
        "points": int(cloud.shape[0]),
        "recon_err": float(np.mean(np.abs(cloud[:, 2] - terrain(cloud[:, 0], cloud[:, 1])))),
        "edges": len(edges), "feasible": sum(r.feasible for r in res),
        "false_rej": false_rej, "false_acc": false_acc,
        "ramp_deg": float(np.mean([f.max_incline_deg for f in ramp])) if ramp else float("nan"),
        "ramp_mean_deg": float(np.mean([f.mean_incline_deg for f in ramp])) if ramp else float("nan"),
        "step_m": max(f.max_step_height_m for f in step) if step else float("nan"),
        "trench_m": max(f.max_step_height_m for f in trench) if trench else float("nan"),
        "path_len": len(best) if best else 0,
        "path_safe": (bool(best) and not any(
            truly_blocked(ep[(best[i], best[i + 1])]) for i in range(len(best) - 1))),
        "reasons": Counter(x for r in res for x in r.failed_checks),
        "pts_per_seg": int(np.median([m.features.n_points for m in msgs])),
        "t": t,
    }
