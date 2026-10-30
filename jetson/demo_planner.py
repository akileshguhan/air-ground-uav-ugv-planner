#!/usr/bin/env python3
"""demo_planner.py -- stand-in for the planner: when the drone sends a cloud,
hand back the fixed route in demo_path.json instead of planning one.
Waypoints in demo_path.json are metres, x = forward, y = LEFT, first point = start."""
import argparse, hashlib, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--ply", required=True)          # the received cloud: only used as the "go" signal
ap.add_argument("--demo", required=True)         # your editable route
ap.add_argument("--out", default="planned_path.json")
a = ap.parse_args()
h = hashlib.sha256()
with open(a.ply, "rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
        h.update(block)
try:
    demo = json.load(open(a.demo))
    wps = [[float(p[0]), float(p[1]), 0.0] for p in demo["waypoints"]]
except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
    print("demo path unreadable (%s): check the JSON in %s" % (e, a.demo)); sys.exit(2)
if len(wps) < 2:
    print("demo path needs at least 2 waypoints"); sys.exit(2)
prev = 0
if os.path.exists(a.out):
    try:
        prev = int(json.load(open(a.out)).get("path_id", 0))
    except (ValueError, TypeError):
        prev = 0
out = {"path_id": prev + 1, "created_s": time.time(), "source": "demo: " + os.path.basename(a.demo),
       "source_sha256": h.hexdigest(), "frame": "x forward, y left, metres from start",
       "waypoints": wps, "speed_factors": [1.0] * (len(wps) - 1), "cost": 0.0}
tmp = a.out + ".part"
json.dump(out, open(tmp, "w"), indent=1)
os.replace(tmp, a.out)
print("DEMO PATH %d: %d waypoints from %s -> %s" % (out["path_id"], len(wps), a.demo, a.out))
for p in wps:
    print("   x=%6.2f  y=%6.2f" % (p[0], p[1]))
