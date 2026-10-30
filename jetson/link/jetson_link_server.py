#!/usr/bin/env python3
"""
jetson_link_server.py -- runs on the Jetson Xavier NX (UGV). Waits for the
drone's Raspberry Pi to connect over TCP.

  * receives files from the drone (captures / point clouds) into --recv-dir,
    checking each one's SHA-256
  * sends a path JSON to the drone: on connect, and again every time the file
    changes on disk (so the planner can just overwrite path_msg.json)
  * heartbeat + round-trip time; reports link loss within --timeout seconds
  * answers the drone's throughput benchmark
  * with --on-file-cmd: runs the planner on every received .ply/.pcd and tells
    the drone how it went ("status" messages); the planner writes --path-file,
    which is then sent to the drone automatically

Needs link_common.py in the same folder.

FULL LOOP (run inside the ugv-nav venv so the planner has numpy/scipy)
  python3 jetson_link_server.py --path-file planned_path.json \
      --on-file-cmd "{python} ~/ugv-nav/scripts/plan_from_ply.py --ply {file} --out planned_path.json"
  {file} = the received file, {python} = the Python running this server.

UGV (optional, --ugv-script): when the drone reports that its mission has
started, the UGV drives the SAME planned_path.json at the SAME --scale the
drone uses, by running ugv_waypoint_drive.py (from its own folder, so it finds
ugv_drive_config.json). The UGV is stopped if the
drone's mission ends with an error, if the link to the drone is lost, or when
this server is stopped (SIGTERM -> the driver's Ctrl-C path -> motors off).

USAGE
  python3 jetson_link_server.py
  python3 jetson_link_server.py --path-file path_lane_change.json
"""
import argparse
import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time

from link_common import DEFAULT_PORT, Link, setup_logging, sha256

log = logging.getLogger("link")


class Server(object):
    def __init__(self, args):
        self.args = args
        self.link = None
        self.lock = threading.Lock()
        self.path_lock = threading.Lock()   # one path send at a time (connect vs file-change)
        self.path_mtime = None
        self.bench = {}
        os.makedirs(args.recv_dir, exist_ok=True)
        self.plan_cv = threading.Condition()
        self.plan_job = None                # newest file waiting to be planned
        self.ugv_proc = None
        self.ugv_lock = threading.Lock()
        self.ugv_cancel = threading.Event()  # set by stop_ugv, aborts a pending start
        self.mission_active = False          # drone said "started", not yet "finished"
        if args.on_file_cmd:
            threading.Thread(target=self._plan_worker, name="planner", daemon=True).start()

    def current_link(self):
        with self.lock:
            link = self.link
        return link if link is not None and link.alive else None

    def status(self, stage, **fields):
        """Tell the drone how planning is going (if it is connected)."""
        link = self.current_link()
        if link is not None:
            link.send("status", stage=stage, **fields)

    # -- UGV driving --
    def start_ugv(self, path_id, scale, why):
        a = self.args
        if not a.ugv_script:
            return
        try:
            with open(a.path_file) as f:
                on_disk = json.load(f)
        except (OSError, ValueError) as e:
            log.error("UGV not started: cannot read %s (%s)", a.path_file, e)
            return
        if path_id is not None and on_disk.get("path_id") != path_id:
            log.error("UGV not started: drone is flying path %s but %s holds path %s",
                      path_id, a.path_file, on_disk.get("path_id"))
            self.status("ugv_refused", reason="path mismatch")
            return
        script = os.path.abspath(os.path.expanduser(a.ugv_script))
        cmd = [a.ugv_python, "-u", script, "--waypoints", os.path.abspath(a.path_file),   # -u: live output
               "--scale", str(scale), "--align", "none"]
        if a.ugv_dry_run:
            cmd.append("--dry-run")
        self.ugv_cancel.clear()
        if a.ugv_delay > 0:
            log.info("UGV starts in %.0f s (trailing the drone)", a.ugv_delay)
            if self.ugv_cancel.wait(a.ugv_delay):
                log.warning("UGV start cancelled")
                return
        with self.ugv_lock:
            if self.ugv_cancel.is_set():
                return
            if self.ugv_proc is not None and self.ugv_proc.poll() is None:
                log.warning("UGV already driving -- not starting another run")
                return
            log.info("UGV START (%s): path %s, scale %.3f%s", why, on_disk.get("path_id"), scale,
                     " [DRY RUN]" if a.ugv_dry_run else "")
            self.ugv_proc = subprocess.Popen(cmd, cwd=os.path.dirname(script),
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             universal_newlines=True, bufsize=1)
            proc = self.ugv_proc
        self.status("ugv_started", path_id=on_disk.get("path_id"), scale=scale)
        threading.Thread(target=self._ugv_output, args=(proc,), daemon=True).start()

    def _ugv_output(self, proc):
        for line in proc.stdout:
            log.info("ugv| %s", line.rstrip())
        rc = proc.wait()
        log.info("UGV finished (exit code %s)", rc)
        self.status("ugv_done", rc=rc)

    def stop_ugv(self, reason):
        self.ugv_cancel.set()
        with self.ugv_lock:
            proc = self.ugv_proc
        if proc is None or proc.poll() is not None:
            return
        log.warning("STOPPING UGV: %s", reason)
        proc.send_signal(signal.SIGTERM)       # driver turns this into its Ctrl-C path: motors off
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            log.error("UGV did not stop in 5 s -- killing it. CHECK THE MOTORS / battery switch")
            proc.kill()

    # -- planning on received files --
    def request_plan(self, path, h):
        with self.plan_cv:
            if self.plan_job is not None:
                log.info("planner busy; newer file %s replaces queued %s",
                         os.path.basename(path), os.path.basename(self.plan_job[0]))
            self.plan_job = (path, h.get("sha256"), h.get("name"))
            self.plan_cv.notify()

    def _plan_worker(self):
        a = self.args
        while True:
            with self.plan_cv:
                while self.plan_job is None:
                    self.plan_cv.wait()
                path, sha, name = self.plan_job
                self.plan_job = None
            cmd = [tok.format(file=path, python=sys.executable)
                   for tok in shlex.split(os.path.expanduser(a.on_file_cmd))]
            cmd = [os.path.expanduser(c) for c in cmd]
            log.info("planning %s ...", name)
            self.status("planning", name=name, source_sha256=sha)
            t0 = time.monotonic()
            tail = []
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        universal_newlines=True, bufsize=1)
                killer = threading.Timer(a.plan_timeout, proc.kill)
                killer.start()
                for line in proc.stdout:
                    line = line.rstrip()
                    log.info("planner| %s", line)
                    tail = (tail + [line])[-5:]
                rc = proc.wait()
                killer.cancel()
            except OSError as e:
                rc, tail = -1, [str(e)]
            dt = time.monotonic() - t0
            if rc == 0:
                log.info("planning done in %.1f s", dt)
                self.status("planned", name=name, source_sha256=sha, seconds=dt)
                self.send_path()             # path file changed -> goes out now
                if a.ugv_on_plan:
                    self.start_ugv(None, a.ugv_scale or 1.0, "--ugv-on-plan")
            else:
                reason = "exit code %d: %s" % (rc, " | ".join(tail[-2:]))
                log.error("planning FAILED (%s)", reason)
                self.status("failed", name=name, source_sha256=sha, seconds=dt, reason=reason)

    # -- messages from the drone --
    def on_message(self, link, h, payload):
        t = h.get("type")
        if t == "hello":
            log.info("drone says hello: role=%s host=%s", h.get("role"), h.get("host"))
            self.send_path(force=True)
        elif t == "file":
            self.save_file(link, h, payload)
        elif t == "ack":
            log.info("drone ack: %s ok=%s %s", h.get("what"), h.get("ok"), h.get("reason", ""))
        elif t == "bench":
            self.on_bench(link, h, payload)
        elif t == "mission":
            stage = h.get("stage")
            log.info("drone mission %s (path %s, scale %s, rc %s)", stage, h.get("path_id"),
                     h.get("scale"), h.get("rc"))
            if stage == "started":
                self.mission_active = True
                scale = self.args.ugv_scale if self.args.ugv_scale else float(h.get("scale", 1.0))
                threading.Thread(target=self.start_ugv, args=(h.get("path_id"), scale, "drone took off"),
                                 daemon=True).start()
            elif stage == "finished":
                self.mission_active = False
                if h.get("rc") not in (0, None):
                    self.stop_ugv("drone mission ended with an error (rc %s)" % h.get("rc"))
        else:
            log.info("message %s: %s", t, {k: v for k, v in h.items() if k != "t"})

    def save_file(self, link, h, payload):
        name = os.path.basename(h.get("name", "unnamed.bin"))  # never trust a path from the wire
        ok = sha256(payload) == h.get("sha256")
        if not ok:
            log.error("file %s FAILED checksum -- discarded", name)
            link.send("ack", what="file", name=name, ok=False, reason="checksum mismatch")
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(self.args.recv_dir, "%s_%s" % (stamp, name))
        tmp = out + ".part"
        with open(tmp, "wb") as f:
            f.write(payload)
        os.replace(tmp, out)   # appears complete or not at all
        log.info("received %s (%s, %.2f MB) -> %s", name, h.get("kind"), len(payload) / 1e6, out)
        link.send("ack", what="file", name=name, ok=True, bytes=len(payload), saved_as=out)
        ext = os.path.splitext(name)[1].lower()
        if self.args.on_file_cmd and ext in [e.strip().lower() for e in self.args.plan_ext.split(",")]:
            self.request_plan(out, h)

    def on_bench(self, link, h, payload):
        i, n = h["i"], h["n"]
        if i == 0:
            self.bench = {"t0": time.monotonic(), "bytes": 0}
        self.bench["bytes"] += len(payload)
        if i == n - 1:
            dt = max(time.monotonic() - self.bench["t0"], 1e-6)
            mbps = self.bench["bytes"] / dt / 1e6
            log.info("benchmark: %.1f MB in %.2f s = %.2f MB/s (%.1f Mbit/s)",
                     self.bench["bytes"] / 1e6, dt, mbps, mbps * 8)
            link.send("bench_result", mb=self.bench["bytes"] / 1e6, seconds=dt, mb_per_s=mbps)

    # -- path to the drone --
    def send_path(self, force=False):
        with self.path_lock:
            self._send_path(force)

    def _send_path(self, force):
        pf = self.args.path_file
        if not pf or not os.path.exists(pf):
            return
        mtime = os.path.getmtime(pf)
        if not force and mtime == self.path_mtime:
            return
        with self.lock:
            link = self.link
        if link is None or not link.alive or (not force and link.peer_hello is None):
            return   # a new drone gets the path when its hello arrives
        try:
            with open(pf) as f:
                path_obj = json.load(f)
        except ValueError:
            log.warning("%s is not valid JSON yet (being written?) -- will retry", pf)
            return
        if link.send_path(path_obj):
            self.path_mtime = mtime
            log.info("sent path %s (%d waypoints) from %s", path_obj.get("path_id"),
                     len(path_obj.get("waypoints", [])), pf)

    # -- connections --
    def on_close(self, link, reason):
        log.warning("drone %s disconnected: %s", link.peer, reason)
        if self.mission_active:            # link lost mid-flight (a clean finish clears this)
            self.mission_active = False
            if self.args.ugv_keep_on_link_loss:
                log.warning("link lost during the drone's mission -- UGV KEEPS DRIVING its path "
                            "(--ugv-keep-on-link-loss). Ctrl-C here to stop it.")
            else:
                self.stop_ugv("lost contact with the drone during its mission")

    def serve(self):
        a = self.args
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((a.bind, a.port))
        srv.listen(1)
        srv.settimeout(0.5)
        log.info("listening on %s:%d  (Jetson IPs: %s)", a.bind, a.port, local_ips())
        if a.path_file:
            log.info("path file: %s (sent on connect and whenever it changes)", a.path_file)
        last_status = time.monotonic()
        try:
            while True:
                try:
                    sock, addr = srv.accept()
                    with self.lock:
                        old = self.link
                    if old is not None and old.alive:
                        old.close("replaced by new connection from %s" % addr[0])
                    log.info("drone connected from %s:%d", addr[0], addr[1])
                    link = Link(sock, "jetson", self.on_message, self.on_close,
                                a.heartbeat, a.timeout)
                    with self.lock:
                        self.link = link
                    link.start()
                except socket.timeout:
                    pass
                self.send_path()
                if time.monotonic() - last_status > a.status_every:
                    last_status = time.monotonic()
                    with self.lock:
                        link = self.link
                    if link is not None and link.alive:
                        log.info("link OK  %s  rx %.2f MB  tx %.2f MB", link.rtt_summary(),
                                 link.bytes_rx / 1e6, link.bytes_tx / 1e6)
                    else:
                        log.info("waiting for drone ...")
        except KeyboardInterrupt:
            log.info("stopping")
        finally:
            with self.lock:
                if self.link is not None:
                    self.link.close("server stopping")
            self.stop_ugv("server stopping")
            srv.close()


def local_ips():
    try:
        out = os.popen("hostname -I 2>/dev/null").read().split()
        return ", ".join(out) or "unknown"
    except Exception:
        return "unknown"


def main(argv=None):
    p = argparse.ArgumentParser(description="UGV (Jetson) side of the drone link.")
    p.add_argument("--bind", default="0.0.0.0", help="interface to listen on (0.0.0.0 = all)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--path-file", help="path JSON to send to the drone (re-sent when it changes)")
    p.add_argument("--recv-dir", default="received_from_drone")
    p.add_argument("--heartbeat", type=float, default=1.0, help="s between pings")
    p.add_argument("--timeout", type=float, default=3.0, help="s of silence = link lost")
    p.add_argument("--status-every", type=float, default=5.0, help="s between status lines")
    p.add_argument("--on-file-cmd", help="command run on each received point cloud; "
                   "{file} = received file, {python} = this Python (see top of file)")
    p.add_argument("--plan-ext", default=".ply,.pcd", help="file types that trigger --on-file-cmd")
    p.add_argument("--plan-timeout", type=float, default=600.0, help="s before the planner is killed")
    g = p.add_argument_group("UGV driving")
    g.add_argument("--ugv-script", help="path to ugv_waypoint_drive.py (enables UGV driving)")
    g.add_argument("--ugv-python", default="/usr/bin/python3",
                   help="Python with Jetson.GPIO + smbus for the motor script (system Python, not the venv)")
    g.add_argument("--ugv-scale", type=float, default=0.0,
                   help="override the scale (default: the scale the drone reports)")
    g.add_argument("--ugv-delay", type=float, default=0.0, help="extra s to wait after drone takeoff")
    g.add_argument("--ugv-dry-run", action="store_true", help="simulate the UGV, motors stay off")
    g.add_argument("--ugv-keep-on-link-loss", action="store_true",
                   help="if the Wi-Fi link drops mid-flight, let the UGV finish its path "
                        "(the drone also carries on with its own copy)")
    g.add_argument("--ugv-on-plan", action="store_true",
                   help="drive as soon as a path is planned, without the drone (UGV-only test)")
    args = p.parse_args(argv)
    setup_logging("JETSON")
    Server(args).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
