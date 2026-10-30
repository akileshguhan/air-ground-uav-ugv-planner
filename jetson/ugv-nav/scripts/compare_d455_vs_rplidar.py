#!/usr/bin/env python3
"""
compare_d455_vs_rplidar.py -- does a D455 input reproduce the RPLiDAR-era
pipeline results on the same ground-truth terrain?

Each configuration changes ONE thing from the previous, so any difference in
results can be attributed:

  A  RPLiDAR reference            the established baseline
  B  D455 clean, raw              file format + camera geometry only (no noise)
  C  D455 clean, heightmap        does downsampling itself bias anything?
  D  D455 noisy, raw              realistic noise, fed straight in
  E  D455 noisy, heightmap        realistic noise + the ingest fix
  F  D455 noisy, heightmap, 4 m   same, flown higher (noise grows with z^2)

D455 configs read their frames back from PLY files through ugvnav.d455_ingest,
so the file-format path is exercised end to end, not bypassed.

Usage:
    python3 scripts/compare_d455_vs_rplidar.py
    python3 scripts/compare_d455_vs_rplidar.py --skip-raw-noisy   # faster
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from ugvnav.compat import describe_environment                # noqa: E402
from ugvnav.d455_ingest import (read_ply_xyz, load_poses, merge_frames,
                                heightmap_downsample)          # noqa: E402
from ugvnav.fusion import write_pcd_binary                     # noqa: E402
from synth_d455 import generate                                # noqa: E402
from rplidar_reference import rplidar_cloud                    # noqa: E402
from pipeline_run import run                                   # noqa: E402
from ground_truth_terrain import GROUND_TRUTH                  # noqa: E402


def load_d455_dir(d):
    poses = load_poses(os.path.join(d, "poses.json"))
    frames = [(read_ply_xyz(os.path.join(d, n)), p["position"], p["quaternion"])
              for n, p in sorted(poses.items())]
    return merge_frames(frames, min_range_m=0.3, max_range_m=6.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/compare")
    ap.add_argument("--cell", type=float, default=0.05, help="heightmap cell size (m)")
    ap.add_argument("--skip-raw-noisy", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.data, exist_ok=True)
    print(describe_environment())
    print()

    dirs = {"clean_2.5": (2.5, False), "noisy_2.5": (2.5, True), "noisy_4.0": (4.0, True)}
    for name, (alt, noise) in dirs.items():
        d = os.path.join(a.data, name)
        if not os.path.exists(os.path.join(d, "poses.json")):
            t0 = time.time()
            generate(d, alt, noise=noise)
            print("generated {:<10} ({:.0f}s)".format(name, time.time() - t0))
    print()

    clouds = {}
    clouds["A  RPLiDAR reference"] = rplidar_cloud()
    clean = load_d455_dir(os.path.join(a.data, "clean_2.5"))
    noisy = load_d455_dir(os.path.join(a.data, "noisy_2.5"))
    noisy4 = load_d455_dir(os.path.join(a.data, "noisy_4.0"))
    clouds["B  D455 clean, raw"] = clean
    clouds["C  D455 clean, heightmap"] = heightmap_downsample(clean, a.cell)
    if not a.skip_raw_noisy:
        clouds["D  D455 noisy, raw"] = noisy
    clouds["E  D455 noisy, heightmap"] = heightmap_downsample(noisy, a.cell)
    clouds["F  D455 noisy, heightmap, 4m"] = heightmap_downsample(noisy4, a.cell)

    results = {}
    for name, cloud in clouds.items():
        t0 = time.time()
        results[name] = run(cloud)
        tag = name.split()[0]
        write_pcd_binary(os.path.join(a.data, "input_{}.pcd".format(tag)), cloud)
        print("ran {:<32} {:>9} pts  {:.1f}s".format(name, cloud.shape[0], time.time() - t0))
    print()

    g = GROUND_TRUTH
    rows = [
        ("points into pipeline", "points", "{:d}"),
        ("median points / segment", "pts_per_seg", "{:d}"),
        ("height error vs truth (m)", "recon_err", "{:.4f}"),
        ("segments", "edges", "{:d}"),
        ("feasible", "feasible", "{:d}"),
        ("FALSE rejections", "false_rej", "{:d}"),
        ("FALSE acceptances", "false_acc", "{:d}"),
        ("ramp max incline (true {:.1f})".format(g["ramp_angle_deg"]), "ramp_deg", "{:.2f}"),
        ("ramp mean incline (true {:.1f})".format(g["ramp_angle_deg"]), "ramp_mean_deg", "{:.2f}"),
        ("step (true {:.3f} m)".format(g["step_height_m"]), "step_m", "{:.3f}"),
        ("trench (true {:.2f} m)".format(g["trench_depth_m"]), "trench_m", "{:.3f}"),
        ("path waypoints", "path_len", "{:d}"),
        ("path avoids trench", "path_safe", "{}"),
    ]
    names = list(results)
    short = [n.split()[0] for n in names]
    print("{:<30}".format("") + "".join("{:>11}".format(s) for s in short))
    print("-" * (30 + 11 * len(names)))
    for label, key, fmt in rows:
        print("{:<30}".format(label) + "".join(
            "{:>11}".format(fmt.format(results[n][key])) for n in names))
    print()
    print("feature-extraction time (s):" + "".join(
        "{:>11.3f}".format(results[n]["t"]["features"]) for n in names))
    print()
    for n in names:
        print("{:<32} rejection reasons: {}".format(n, dict(results[n]["reasons"])))
    print()
    print("legend:")
    for n in names:
        print("  " + n)

    out = {n: {k: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
               for k, v in r.items() if k not in ("reasons", "t")}
           for n, r in results.items()}
    with open(os.path.join(a.data, "comparison.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("\ninput clouds for CloudCompare: {}/input_*.pcd".format(a.data))


if __name__ == "__main__":
    main()
