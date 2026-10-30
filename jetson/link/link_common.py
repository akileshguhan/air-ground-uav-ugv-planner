"""
link_common.py -- shared TCP protocol for the drone (Raspberry Pi) <-> UGV
(Jetson Xavier NX) link. Copy this SAME file to both boards, next to the
board's own script (jetson_link_server.py / rpi_link_client.py).

Standard library only. Python 3.8+ (Jetson JetPack 5) and 3.12 (Pi, Ubuntu 24.04).

FRAME FORMAT (every message)
    16-byte prefix : magic b"AGL1" | header length (uint32) | payload length (uint64)
    header         : UTF-8 JSON, always has "type" and "t" (sender time)
    payload        : raw bytes (a file, a path JSON, benchmark data, ...) -- may be empty

MESSAGE TYPES
    hello       both ways, right after connecting (role, hostname)
    ping/pong   heartbeat every 1 s; pong echoes the ping time -> round-trip time
    path        Jetson -> Pi   payload = path JSON (same format as path_msg.json)
    file        Pi -> Jetson   payload = file bytes; header has name, kind, sha256
    ack         reply to path / file (ok, bytes, reason)
    bench       Pi -> Jetson   throughput test chunks
    bench_result Jetson -> Pi  measured MB/s

LINK LOSS
    If nothing at all arrives for --timeout seconds (default 3 s), the link is
    declared lost and closed. The Pi client then reconnects automatically.
    Bytes of a large transfer in progress count as "something arriving", and
    time spent handling a message (e.g. writing a big file to disk) is not
    counted, so a big file does not trip the watchdog.
"""
import hashlib
import json
import logging
import os
import socket
import struct
import threading
import time

MAGIC = b"AGL1"
PREFIX = struct.Struct("!4sIQ")
MAX_HEADER = 1 << 20            # 1 MB of JSON header is already absurd
MAX_PAYLOAD = 2 << 30           # 2 GB
DEFAULT_PORT = 5600
CHUNK = 1 << 16

log = logging.getLogger("link")


def setup_logging(name):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s.%(msecs)03d " + name + " %(message)s",
                        datefmt="%H:%M:%S")


def tune_socket(sock):
    """Low latency for small control messages + OS-level dead-peer detection."""
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for opt, val in (("TCP_KEEPIDLE", 5), ("TCP_KEEPINTVL", 2), ("TCP_KEEPCNT", 3)):
        if hasattr(socket, opt):
            sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), val)


class LinkClosed(Exception):
    pass


def _recv_exact(sock, n, on_bytes):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], min(n - got, CHUNK))
        if k == 0:
            raise LinkClosed("peer closed the connection")
        got += k
        on_bytes()
    return bytes(buf)


def encode_frame(header, payload=b""):
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return PREFIX.pack(MAGIC, len(h), len(payload)) + h, payload


def recv_frame(sock, on_bytes=lambda: None):
    magic, hlen, plen = PREFIX.unpack(_recv_exact(sock, PREFIX.size, on_bytes))
    if magic != MAGIC:
        raise LinkClosed("bad magic %r -- not an AGL1 peer, or stream out of sync" % magic)
    if hlen > MAX_HEADER or plen > MAX_PAYLOAD:
        raise LinkClosed("frame too large (header %d, payload %d)" % (hlen, plen))
    header = json.loads(_recv_exact(sock, hlen, on_bytes).decode("utf-8"))
    payload = _recv_exact(sock, plen, on_bytes) if plen else b""
    return header, payload


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class Link(object):
    """One live connection. Runs a receive thread and a heartbeat thread.

    on_message(link, header, payload) is called from the receive thread for
    every message except ping/pong. on_close(link, reason) is called once.
    """

    def __init__(self, sock, role, on_message, on_close=None,
                 heartbeat_s=1.0, timeout_s=3.0):
        tune_socket(sock)
        self.sock = sock
        self.role = role
        self.peer = "%s:%d" % sock.getpeername()[:2]
        self.on_message = on_message
        self.on_close = on_close
        self.heartbeat_s = heartbeat_s
        self.timeout_s = timeout_s
        self.send_lock = threading.Lock()
        self.closed = threading.Event()
        self.close_reason = None
        self.last_rx = time.monotonic()
        self.rtt_ms = []                   # recent round-trip times
        self.peer_hello = None
        self.bytes_rx = 0
        self.bytes_tx = 0
        self.in_handler = False            # busy saving/processing a message

    # -- lifecycle --
    def start(self):
        threading.Thread(target=self._rx_loop, name="rx", daemon=True).start()
        threading.Thread(target=self._hb_loop, name="hb", daemon=True).start()
        self.send("hello", role=self.role, host=socket.gethostname())
        return self

    @property
    def alive(self):
        return not self.closed.is_set()

    def close(self, reason="closed locally"):
        if self.closed.is_set():
            return
        self.close_reason = reason
        self.closed.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)   # unblocks recv/send in other threads
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
        if self.on_close:
            try:
                self.on_close(self, reason)
            except Exception:
                log.exception("on_close handler failed")

    # -- sending --
    def send(self, msg_type, payload=b"", **fields):
        header = dict(fields)
        header["type"] = msg_type
        header["t"] = time.time()
        prefix_and_header, payload = encode_frame(header, payload)
        try:
            with self.send_lock:
                self.sock.sendall(prefix_and_header)
                if payload:
                    self.sock.sendall(payload)
            self.bytes_tx += len(prefix_and_header) + len(payload)
            return True
        except OSError as e:
            self.close("send failed: %s" % e)
            return False

    def send_file(self, path, kind="capture"):
        with open(path, "rb") as f:
            data = f.read()
        log.info("sending %s (%s, %.2f MB)", os.path.basename(path), kind, len(data) / 1e6)
        return self.send("file", data, name=os.path.basename(path), kind=kind,
                         sha256=sha256(data), bytes=len(data))

    def send_path(self, path_obj):
        data = json.dumps(path_obj).encode("utf-8")
        return self.send("path", data, path_id=path_obj.get("path_id"),
                         n_waypoints=len(path_obj.get("waypoints", [])), sha256=sha256(data))

    # -- stats --
    def rtt_summary(self):
        if not self.rtt_ms:
            return "rtt n/a"
        r = sorted(self.rtt_ms)
        return "rtt median %.1f ms, max %.1f ms (last %d)" % (r[len(r) // 2], r[-1], len(r))

    # -- threads --
    def _touch(self):
        self.last_rx = time.monotonic()

    def _rx_loop(self):
        try:
            while not self.closed.is_set():
                header, payload = recv_frame(self.sock, self._touch)
                self.bytes_rx += PREFIX.size + len(payload)
                t = header.get("type")
                if t == "ping":
                    self.send("pong", echo=header["t"])
                elif t == "pong":
                    self.rtt_ms.append((time.time() - header["echo"]) * 1000.0)
                    self.rtt_ms = self.rtt_ms[-50:]
                else:
                    if t == "hello":
                        self.peer_hello = header
                    self.in_handler = True
                    try:
                        self.on_message(self, header, payload)
                    finally:
                        self.in_handler = False
                        self._touch()
        except (LinkClosed, OSError, ValueError) as e:
            self.close(str(e) if not self.closed.is_set() else self.close_reason)
        except Exception as e:
            log.exception("receive handler crashed")
            self.close("handler error: %s" % e)

    def _hb_loop(self):
        while not self.closed.wait(self.heartbeat_s):
            silent = time.monotonic() - self.last_rx
            # time spent saving a big file is us being busy, not the peer being gone
            if silent > self.timeout_s and not self.in_handler:
                self.close("link lost: nothing received for %.1f s" % silent)
                return
            self.send("ping")
