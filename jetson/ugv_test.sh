#!/usr/bin/env bash
# ugv_test.sh -- test the UGV on its own: no Pi, no network, no drone.
# Run the steps in this order the first time.
#
#   ~/airground/ugv_test.sh sim                  1. simulated robot, motors OFF
#   ~/airground/ugv_test.sh check                2. short left spin: motors and IMU agree?
#   ~/airground/ugv_test.sh calibrate            3. drive 3 s, you measure, saves speed
#                                                   (only if base_duty / battery / floor changed)
#   ~/airground/ugv_test.sh square [SCALE] [--dry-run]
#                                                4. drive a square (default 0.5 m)
#   ~/airground/ugv_test.sh lane [SCALE] [--dry-run]
#                                                5. ready-made lane-change route (default 0.2 -> 5.6 m)
#   ~/airground/ugv_test.sh plan [PLY]           6. plan from a .ply (default data/terrain_map.ply)
#   ~/airground/ugv_test.sh path [SCALE] [--dry-run]
#                                                7. drive the path from step 6 (default 0.1 -> ~2.8 m)
# Wheels on the ground for 2-7, battery switch within reach. Ctrl-C stops the motors.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
DRIVE="$ROOT/ugv_drive/ugv_waypoint_drive.py"
PLAN="$ROOT/data/planned_path_local.json"     # separate from the live loop's link/planned_path.json

cd "$ROOT/ugv_drive"                          # the driver reads ugv_drive_config.json from here
step="${1:-help}"; shift || true
case "$step" in
  sim)       "$PY" "$DRIVE" --dry-run --demo square ;;
  check)     "$PY" "$DRIVE" --check-turn ;;
  calibrate) "$PY" "$DRIVE" --calibrate-speed 3 ;;
  square)    SCALE="${1:-0.5}"; shift || true
             "$PY" "$DRIVE" --demo square --scale "$SCALE" "$@" ;;
  lane)      SCALE="${1:-0.2}"; shift || true
             "$PY" "$DRIVE" --waypoints "$ROOT/data/path_lane_change.json" --scale "$SCALE" "$@" ;;
  plan)
    PLY="${1:-$ROOT/data/terrain_map.ply}"
    case "$(basename "$PLY")" in
      terrain_map*) CORRIDOR="--start 55,20 --heading 45 --length 30" ;;
      *)            CORRIDOR="" ;;                                   # auto corridor
    esac
    "$PY" "$ROOT/ugv-nav/scripts/plan_from_ply.py" --ply "$PLY" --out "$PLAN" $CORRIDOR ;;
  path)
    [ -f "$PLAN" ] || { echo "no planned path yet -- run: $0 plan"; exit 1; }
    SCALE="${1:-0.1}"; shift || true
    "$PY" "$DRIVE" --waypoints "$PLAN" --scale "$SCALE" --align none "$@" ;;
  *) sed -n '2,16p' "$ROOT/$(basename "$0")" ;;
esac
