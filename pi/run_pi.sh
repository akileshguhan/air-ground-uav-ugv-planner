#!/usr/bin/env bash
# run_pi.sh -- the drone (Raspberry Pi) side, one short command per job.
#
#   Terminal 1 (leave running):
#     ~/airground/run_pi.sh mavros              start MAVROS on the Pixhawk serial link
#
#   Terminal 2, in this order the first time:
#     ~/airground/run_pi.sh status              1. Pi <-> Pixhawk link: mode / armed state
#     ~/airground/run_pi.sh armtest             2. PROPS OFF: arm, hold 5 s, disarm (MANUAL)
#     ~/airground/run_pi.sh link JETSON_IP      3. Pi <-> Jetson Wi-Fi link + speed test (20 s)
#     ~/airground/run_pi.sh cycle JETSON_IP --no-fly
#                                               4. send cloud, get path, show plan, stop
#     ~/airground/run_pi.sh cycle JETSON_IP     5. same, then type "fly": drone flies,
#                                                  UGV drives the same path
#   Flight without the Jetson:
#     ~/airground/run_pi.sh demo --plan-only    show a 1 m out-and-back plan
#     ~/airground/run_pi.sh demo                fly it (1.5 m altitude)
#
# Settings (override per command, e.g.  SCALE=0.2 ~/airground/run_pi.sh cycle ...):
#   PLY=data/terrain_map.ply  SCALE=0.3  ALT=2  SPEED=0.5  PORT=/dev/ttyAMA0  BAUD=921600
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
PLY="${PLY:-$ROOT/data/terrain_map.ply}"
SCALE="${SCALE:-0.3}"; ALT="${ALT:-2}"; SPEED="${SPEED:-0.5}"
PORT="${PORT:-/dev/ttyAMA0}"; BAUD="${BAUD:-921600}"

if [ -f /opt/ros/jazzy/setup.bash ]; then
  source /opt/ros/jazzy/setup.bash          # ROS 2 for MAVROS and the flight scripts
fi
cd "$ROOT"                                  # received paths and logs land in ~/airground

need_ip() { [ -n "$1" ] || { echo "give the Jetson's IP, e.g.: $0 $2 <JETSON_IP>"; exit 2; }; }

job="${1:-help}"; shift || true
case "$job" in
  mavros)
    exec ros2 launch mavros px4.launch fcu_url:="$PORT:$BAUD" ;;
  status)
    exec "$PY" arm_disarm_test.py --status ;;
  armtest)
    exec "$PY" arm_disarm_test.py --mode MANUAL "$@" ;;
  link)
    need_ip "$1" link
    echo "testing the link to $1 for 20 s ..."
    timeout 20 "$PY" rpi_link_client.py --server "$1" --bench 10 --once
    echo "(link test finished)" ;;
  cycle)
    need_ip "$1" cycle; IP="$1"; shift
    [ -f "$PLY" ] || { echo "no point cloud at $PLY -- copy one to ~/airground/data/"; exit 1; }
    exec "$PY" drone_cycle.py --server "$IP" --ply "$PLY" \
      --mission-args "--scale $SCALE --altitude $ALT --speed $SPEED" "$@" ;;
  demo)
    exec "$PY" px4_waypoint_mission.py --demo line --scale 1 --altitude 1.5 --speed "$SPEED" "$@" ;;
  *)
    sed -n '2,23p' "$ROOT/$(basename "$0")" ;;
esac
