#!/usr/bin/env python3
"""
rpi_link_client.py -- runs on the drone's Raspberry Pi. Connects to the
Jetson (UGV) over TCP and keeps reconnecting if the link drops.

  * receives paths from the Jetson; each one is checked (SHA-256) and saved
    atomically to --path-out (default received_path.json), which
    px4_waypoint_mission.py can fly:  --waypoints received_path.json
    A copy of every path is also kept in received_paths/.
  * sends files to the Jetson (--send-file, repeatable), retried after a
    reconnect until the Jetson acknowledges them
  * optional throughput benchmark (--bench MB)
  * heartbeat + round-trip time; link loss detected within --timeout seconds

Needs link_common.py in the same folder.

USAGE
  python3 rpi_link_client.py --server 192.168.1.20
  python3 rpi_link_client.py --server 192.168.1.20 --send-file test.ply --bench 20
"""
import argparse
import json
import logging
import os
import socket
import sys
import threading
import time

from link_common import DEFAULT_PORT, Link, setup_logging, sha256

log = logging.getLogger("link")


class Client(object):
    def __init__(self, args):
        self.args = args
        self.pending = list(args.send_file or [])   # files not yet acknowledged
        self.pending_lock = threading.Lock()
        self.bench_done = threading.Event()
        os.makedirs(args.path_history, exist_ok=True)

    def on_message(self, link, h, payload):
        t = h.get("type")
        if t == "hello":
            log.info("jetson says hello: host=%s", h.get("host"))
        elif t == "path":
            self.save_path(link, h, payload)
        elif t == "ack":
            if h.get("what") == "file" and h.get("ok"):
                log.info("jetson stored %s (%d bytes)", h.get("name"), h.get("bytes", 0))
                with self.pending_lock:
                    self.pending = [p for p in self.pending if os.path.basename(p) != h.get("name")]
            else:
                log.error("jetson rejected %s: %s", h.get("name"), h.get("reason"))
        elif t == "bench_result":
            log.info("BENCHMARK: %.1f MB in %.2f s = %.2f MB/s (%.1f Mbit/s)",
                     h["mb"], h["seconds"], h["mb_per_s"], h["mb_per_s"] * 8)
            self.bench_done.set()
        else:
            log.info("message %s: %s", t, {k: v for k, v in h.items() if k != "t"})

    def save_path(self, link, h, payload):
        if sha256(payload) != h.get("sha256"):
            log.error("path FAILED checksum -- ignored")
            link.send("ack", what="path", ok=False, reason="checksum mismatch")
            return
        try:
            path_obj = json.loads(payload.decode("utf-8"))
            n = len(path_obj["waypoints"])
        except (ValueError, KeyError, TypeError) as e:
            log.error("path is not valid path JSON (%s) -- ignored", e)
            link.send("ack", what="path", ok=False, reason="bad JSON: %s" % e)
            return
        out = self.args.path_out
        tmp = out + ".part"
        with open(tmp, "wb") as f:
            f.write(payload)
        os.replace(tmp, out)          # the mission script never sees half a file
        hist = os.path.join(self.args.path_history,
                            "path_%s_%s.json" % (h.get("path_id"), time.strftime("%Y%m%d_%H%M%S")))
        with open(hist, "wb") as f:
            f.write(payload)
        log.info("received path %s with %d waypoints -> %s", h.get("path_id"), n, out)
        link.send("ack", what="path", ok=True, path_id=h.get("path_id"), n_waypoints=n)

    def on_close(self, link, reason):
        log.warning("link to jetson closed: %s", reason)

    def run_bench(self, link, mb):
        n = max(1, int(mb))
        chunk = os.urandom(1 << 20)   # 1 MB
        log.info("benchmark: sending %d MB ...", n)
        self.bench_done.clear()
        for i in range(n):
            if not link.send("bench", chunk, i=i, n=n):
                return
        self.bench_done.wait(30)

    def session(self, sock):
        a = self.args
        link = Link(sock, "drone", self.on_message, self.on_close, a.heartbeat, a.timeout).start()
        with self.pending_lock:
            todo = list(self.pending)
        for p in todo:
            if not os.path.exists(p):
                log.error("file not found: %s (dropped)", p)
                with self.pending_lock:
                    self.pending.remove(p)
                continue
            link.send_file(p)
        if a.bench and link.alive:
            self.run_bench(link, a.bench)
            a.bench = 0   # once per run
        last_status = time.monotonic()
        while link.alive:
            time.sleep(0.2)
            if link.alive and time.monotonic() - last_status > a.status_every:
                last_status = time.monotonic()
                log.info("link OK  %s", link.rtt_summary())
        return link.close_reason

    def run(self):
        a = self.args
        delay = 1.0
        log.info("drone link client -> %s:%d", a.server, a.port)
        while True:
            try:
                sock = socket.create_connection((a.server, a.port), timeout=3.0)
                sock.settimeout(None)
            except OSError as e:
                log.warning("cannot reach jetson (%s); retry in %.0f s", e, delay)
                time.sleep(delay)
                delay = min(delay * 2, 5.0)
                continue
            log.info("connected to jetson %s:%d", a.server, a.port)
            delay = 1.0
            self.session(sock)
            if a.once:
                return
            time.sleep(1.0)


def main(argv=None):
    p = argparse.ArgumentParser(description="Drone (Raspberry Pi) side of the UGV link.")
    p.add_argument("--server", required=True, help="Jetson IP or hostname (e.g. 192.168.1.20)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--send-file", action="append", help="file to send (repeatable)")
    p.add_argument("--bench", type=float, default=0, help="throughput test: send this many MB")
    p.add_argument("--path-out", default="received_path.json")
    p.add_argument("--path-history", default="received_paths")
    p.add_argument("--heartbeat", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=3.0)
    p.add_argument("--status-every", type=float, default=5.0)
    p.add_argument("--once", action="store_true", help="exit when the link drops (no reconnect)")
    args = p.parse_args(argv)
    setup_logging("DRONE ")
    try:
        Client(args).run()
    except KeyboardInterrupt:
        log.info("stopping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
