#!/usr/bin/env bash
# run_demo.sh -- like run_jetson.sh, but drone and UGV follow data/demo_path.json
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
cd "$ROOT/link"
echo "== demo route: $ROOT/data/demo_path.json   Jetson IP: $(hostname -I 2>/dev/null | awk '{print $1}')"
exec "$PY" jetson_link_server.py --path-file planned_path.json \
  --on-file-cmd "{python} $ROOT/demo_planner.py --ply {file} --demo $ROOT/data/demo_path.json --out planned_path.json" \
  --ugv-script "$ROOT/ugv_drive/ugv_waypoint_drive.py" --ugv-python "$PY" \
  --ugv-keep-on-link-loss --timeout 10 "$@"
