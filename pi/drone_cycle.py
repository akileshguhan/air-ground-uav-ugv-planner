#!/usr/bin/env python3
"""
drone_cycle.py -- one full cycle, run on the drone's Raspberry Pi:

    1. send a point cloud (.ply) to the Jetson
    2. wait while the Jetson plans (status messages are printed)
    3. receive the path that answers THIS file (matched by SHA-256, so an old
       path left on the Jetson is never flown by mistake)
    4. show the flight plan (px4_waypoint_mission.py --plan-only)
    5. ask you to type "fly", tell the Jetson the mission has started (so the
       UGV drives the same path at the same --scale, if the Jetson has
       --ugv-script), run px4_waypoint_mission.py, and report how it ended
       (an error stops the UGV)

MAVROS must already be running in another terminal, and this terminal must
have ROS 2 sourced (source /opt/ros/jazzy/setup.bash).

Needs, in the same folder: link_common.py, rpi_link_client.py,
px4_waypoint_mission.py.

USAGE
  python3 drone_cycle.py --server <JETSON_IP> --ply terrain_map.ply \\
      --mission-args "--scale 0.2 --altitude 2 --speed 0.5"
  ... --no-fly        stop after receiving the path (bench test, no drone needed)
"""
import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time

from link_common import DEFAULT_PORT, Link, setup_logging
from rpi_link_client import Client

HERE = os.path.dirname(os.path.abspath(__file__))
log = None


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Cycle(Client):
    """Client that remembers which path answers which file."""

    def __init__(self, args, want_sha):
        super(Cycle, self).__init__(args)
        self.want_sha = want_sha
        self.got_path = threading.Event()
        self.failed = None
        self.planning_seen = False   # only paths planned AFTER our upload count

    def on_message(self, link, h, payload):
        t = h.get("type")
        if t == "status":
            if h.get("source_sha256") not in (None, self.want_sha):
                return                                    # about some other file
            stage = h.get("stage")
            if stage == "planning":
                self.planning_seen = True
                log.info("jetson is planning %s ...", h.get("name"))
            elif stage == "planned":
                log.info("jetson finished planning in %.1f s", h.get("seconds", 0))
            elif stage == "failed":
                self.failed = h.get("reason", "unknown")
                log.error("jetson planning FAILED: %s", self.failed)
                self.got_path.set()
            elif stage == "ugv_started":
                log.info("UGV started driving path %s (scale %s)", h.get("path_id"), h.get("scale"))
            elif stage == "ugv_done":
                log.info("UGV finished (exit code %s)", h.get("rc"))
            elif stage == "ugv_refused":
                log.error("UGV did NOT start: %s", h.get("reason"))
            return
        if t == "path":
            # peek at the source hash before accepting it as the answer
            try:
                src = json.loads(payload.decode("utf-8")).get("source_sha256")
            except ValueError:
                src = None
            if src != self.want_sha or not self.planning_seen:
                log.info("ignoring path %s: it is from before this upload -- waiting for the new plan",
                         h.get("path_id"))
                return                                        # never saved, never flown
            super(Cycle, self).on_message(link, h, payload)   # checks + saves to --path-out
            if os.path.exists(self.args.path_out):
                self.got_path.set()
            return
        super(Cycle, self).on_message(link, h, payload)


def main(argv=None):
    global log
    p = argparse.ArgumentParser(description="Send a cloud, get the path, fly it.")
    p.add_argument("--server", required=True, help="Jetson IP")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--ply", required=True, help="point cloud to send")
    p.add_argument("--wait", type=float, default=600.0, help="s to wait for the path")
    p.add_argument("--path-out", default="received_path.json")
    p.add_argument("--path-history", default="received_paths")
    p.add_argument("--mission", default=os.path.join(HERE, "px4_waypoint_mission.py"))
    p.add_argument("--mission-args", default="--scale 0.2 --altitude 2 --speed 0.5",
                   help="extra arguments for px4_waypoint_mission.py (quoted)")
    p.add_argument("--no-fly", action="store_true", help="stop after the plan is shown")
    p.add_argument("--yes", action="store_true", help="skip the 'fly' confirmation (SITL only!)")
    a = p.parse_args(argv)
    # fields the base Client expects
    a.send_file, a.bench, a.heartbeat, a.timeout, a.status_every, a.once = [], 0, 1.0, 10.0, 5.0, True

    setup_logging("DRONE ")
    import logging
    log = logging.getLogger("link")

    if not os.path.exists(a.ply):
        log.error("no such file: %s", a.ply)
        return 2
    sha = file_sha256(a.ply)
    size_mb = os.path.getsize(a.ply) / 1e6
    log.info("cloud %s: %.1f MB (expect ~%.0f s to send at 2.5 MB/s)", a.ply, size_mb, size_mb / 2.5)

    # 1-3: send and wait for the matching path
    cyc = Cycle(a, sha)
    sock = socket.create_connection((a.server, a.port), timeout=5.0)
    sock.settimeout(None)
    link = Link(sock, "drone", cyc.on_message, cyc.on_close, a.heartbeat, a.timeout).start()
    log.info("connected to jetson %s:%d", a.server, a.port)
    t0 = time.monotonic()
    link.send_file(a.ply, kind="pointcloud")
    log.info("sent in %.1f s; waiting for the path (up to %.0f s) ...", time.monotonic() - t0, a.wait)
    deadline = time.monotonic() + a.wait
    while not cyc.got_path.wait(0.5):
        if not link.alive:
            log.error("link dropped before the path arrived (%s)", link.close_reason)
            return 1
        if time.monotonic() > deadline:
            log.error("no path after %.0f s", a.wait)
            link.close("timeout")
            return 1
    if cyc.failed:
        link.close("planning failed")
        return 1
    log.info("path saved to %s  (round trip %.1f s)", a.path_out, time.monotonic() - t0)

    # 4: show the plan
    mission = [sys.executable, a.mission, "--waypoints", a.path_out] + shlex.split(a.mission_args)
    print("\n" + "=" * 70)
    rc = subprocess.call(mission + ["--plan-only"])
    print("=" * 70)
    if rc != 0:
        link.close("plan refused")
        log.error("mission refused this plan (see REFUSED above) -- adjust --mission-args, e.g. --scale")
        return 1
    if a.no_fly:
        link.close("no-fly")
        log.info("--no-fly: stopping here (UGV not started). To fly this path later:\n  %s",
                 " ".join(shlex.quote(c) for c in mission))
        return 0

    # 5: fly
    if not a.yes:
        print("\nProps on, area clear, RC in hand, MAVROS running? (the UGV moves too, if enabled)")
        if input('Type "fly" to start the mission: ').strip().lower() != "fly":
            link.close("not flying")
            log.info("not flying. Path is kept in %s", a.path_out)
            return 0
    with open(a.path_out) as f:
        path_id = json.load(f).get("path_id")
    m_args, _ = mission_parser().parse_known_args(shlex.split(a.mission_args))
    if not link.alive:
        log.warning("link to the Jetson is gone -- flying anyway, but the UGV will NOT start")
    link.send("mission", stage="started", path_id=path_id, scale=m_args.scale, source_sha256=sha)
    rc = subprocess.call(mission)
    log.info("mission exited with code %d", rc)
    if link.alive:
        link.send("mission", stage="finished", path_id=path_id, rc=rc)
        time.sleep(0.5)                       # let it go out before closing
    link.close("cycle done")
    return rc


def mission_parser():
    """Just enough of px4_waypoint_mission's options to know the scale."""
    mp = argparse.ArgumentParser(add_help=False)
    mp.add_argument("--scale", type=float, default=1.0)
    return mp


if __name__ == "__main__":
    sys.exit(main())
