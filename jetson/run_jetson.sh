#!/usr/bin/env bash
# run_jetson.sh -- the UGV side of the full loop, in one command.
#
# Starts the link server, which:
#   1. receives a .ply from the drone's Pi
#   2. plans a path through it (ugv-nav/scripts/plan_from_ply.py)
#   3. sends the path back to the drone
#   4. when the drone takes off, drives the UGV along the same path at the
#      same scale (ugv_drive/ugv_waypoint_drive.py); stops it if the drone
#      aborts, the link drops, or you press Ctrl-C
#
# USAGE (from anywhere):
#   ~/airground/run_jetson.sh                      real UGV, terrain_map corridor
#   ~/airground/run_jetson.sh --dry-run            simulated UGV, motors stay off
#   ~/airground/run_jetson.sh --no-ugv             plan + send path only, UGV never moves
#   ~/airground/run_jetson.sh --ugv-on-plan        UGV drives as soon as a path is planned
#                                                  (no drone flight needed)
#   ~/airground/run_jetson.sh --auto-corridor      for a .ply other than terrain_map
#   ~/airground/run_jetson.sh --corridor X,Y DEG M custom corridor start, heading, length
#   ~/airground/run_jetson.sh --delay 10           UGV waits 10 s more after drone takeoff
#   ~/airground/run_jetson.sh --keep-ugv           if Wi-Fi drops mid-flight, UGV still finishes
#                                                  its path (default: it stops)
#   ~/airground/run_jetson.sh --link-timeout 10    s of Wi-Fi silence before the link counts as lost
#                                                  (default 3)
# Options combine, e.g.  --dry-run --auto-corridor
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"                       # system Python: numpy, scipy, Jetson.GPIO

CORRIDOR="--start 55,20 --heading 45 --length 30"   # known-flat strip of terrain_map.ply
UGV=(--ugv-script "$ROOT/ugv_drive/ugv_waypoint_drive.py" --ugv-python "$PY")
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)        EXTRA+=(--ugv-dry-run) ;;
    --no-ugv)         UGV=() ;;
    --ugv-on-plan)    EXTRA+=(--ugv-on-plan --ugv-scale "${UGV_SCALE:-0.1}") ;;
    --auto-corridor)  CORRIDOR="" ;;
    --corridor)       CORRIDOR="--start $2 --heading $3 --length $4"; shift 3 ;;
    --delay)          EXTRA+=(--ugv-delay "$2"); shift ;;
    --keep-ugv)       EXTRA+=(--ugv-keep-on-link-loss) ;;
    --link-timeout)   EXTRA+=(--timeout "$2"); shift ;;
    -h|--help)        sed -n '2,25p' "$ROOT/$(basename "$0")"; exit 0 ;;
    *) echo "unknown option: $1  (try --help)"; exit 2 ;;
  esac
  shift
done

if [ ${#UGV[@]} -gt 0 ] && [ ! -f "$ROOT/ugv_drive/ugv_drive_config.json" ]; then
  echo "WARNING: no ugv_drive/ugv_drive_config.json (speed calibration)."
  echo "         Copy yours in, or run: ~/airground/ugv_test.sh calibrate"
fi

cd "$ROOT/link"
echo "== airground Jetson: $(hostname -I 2>/dev/null | awk '{print $1}')  port 5600"
exec "$PY" jetson_link_server.py \
  --path-file planned_path.json \
  --on-file-cmd "{python} $ROOT/ugv-nav/scripts/plan_from_ply.py --ply {file} --out planned_path.json $CORRIDOR" \
  "${UGV[@]}" "${EXTRA[@]}"
